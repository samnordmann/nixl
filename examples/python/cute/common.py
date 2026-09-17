# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, synchronous teaching harness. Gloo carries metadata, never GPU payloads."""

import os
import time
import traceback
from datetime import timedelta

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.runtime import from_dlpack
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

import nixl_cute as ops


@cute.kernel
def send_kernel(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    size: cutlass.Uint64,
    src_offset: cutlass.Uint64,
    dst_offset: cutlass.Uint64,
    counter_offset: cutlass.Uint64,
    status: cute.Tensor,
    with_signal: cutlass.Constexpr,
):
    result = ops.put(local, remote, size, src_offset, dst_offset)
    status[0] = result
    if cutlass.const_expr(with_signal):
        if result == 0:
            status[1] = ops.signal(remote, counter_offset)


@cute.jit
def send(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    size: cutlass.Uint64,
    src_offset: cutlass.Uint64,
    dst_offset: cutlass.Uint64,
    counter_offset: cutlass.Uint64,
    status: cute.Tensor,
    with_signal: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    send_kernel(
        local, remote, size, src_offset, dst_offset, counter_offset, status, with_signal
    ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@cute.kernel
def wait_kernel(address: cutlass.Uint64, expected: cutlass.Uint64, status: cute.Tensor):
    status[2] = ops.wait(address, expected)


@cute.jit
def wait(
    address: cutlass.Uint64,
    expected: cutlass.Uint64,
    status: cute.Tensor,
    stream: cuda.CUstream,
):
    wait_kernel(address, expected, status).launch(
        grid=(1, 1, 1), block=(1, 1, 1), stream=stream
    )


def gather(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


class World:
    def __init__(self, rows, hidden):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        self.stream = torch.cuda.current_stream()
        self.cu_stream = cuda.CUstream(self.stream.cuda_stream)
        self.send = torch.zeros(
            (self.size, rows, hidden), device="cuda", dtype=torch.bfloat16
        )
        self.recv = torch.zeros_like(self.send)
        self.ready = torch.zeros(self.size, device="cuda", dtype=torch.int64)
        self.status = torch.zeros((self.size, 3), device="cuda", dtype=torch.int32)
        self.stream.synchronize()
        self.agent = nixl_agent(
            f"cute-{self.rank}",
            nixl_agent_config(
                backends=["UCX"],
                enable_prog_thread=True,
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
            ),
        )
        self.registration = self.agent.register_memory(
            [self.send, self.recv, self.ready]
        )
        coords = [
            (
                t.data_ptr(),
                t.numel() * t.element_size(),
                t.get_device(),
                self.agent.name,
            )
            for t in (self.recv, self.ready)
        ]
        self.metadata = gather((self.agent.get_agent_metadata(), coords))
        self.local = self.agent.prep_mem_view(self.agent.get_xfer_descs(self.send))
        self.peers, self.active = {}, ()
        self.expected = [0] * self.size
        self.slot_bytes = rows * hidden * self.send.element_size()

    def membership(self, active):
        """Stage new views, drain/commit together, then release removed views."""
        needed = set(active) - {self.rank} if self.rank in active else set()
        joining = needed - self.peers.keys()
        for peer in sorted(joining):
            self.agent.add_remote_agent(self.metadata[peer][0])
        dist.barrier()
        for peer in sorted(joining):
            self.agent.make_connection(f"cute-{peer}")
            self.agent.send_notif(f"cute-{peer}", b"connected", backend="UCX")
        pending, deadline = {f"cute-{peer}" for peer in joining}, time.monotonic() + 60
        while pending:
            for name, messages in self.agent.get_new_notifs().items():
                if b"connected" in messages:
                    pending.discard(name)
            if time.monotonic() > deadline:
                raise TimeoutError(f"connection handshake: {pending}")
            time.sleep(0.001)
        dist.barrier()
        staged = dict(self.peers)
        for peer in sorted(joining):
            descs = self.agent.get_remote_descs(self.metadata[peer][1], mem_type="VRAM")
            staged[peer] = self.agent.prep_mem_view(descs)
        self.stream.synchronize()
        dist.barrier()  # No old-generation kernel can survive this commit.
        for peer in sorted(self.peers.keys() - needed):
            self.agent.release_mem_view(staged.pop(peer))
            self.agent.remove_remote_agent(f"cute-{peer}")
        self.peers, self.active = staged, tuple(active)
        dist.barrier()

    def exchange(self, with_signal=True):
        """Send one padded slab per active peer, then wait before reading/reusing it."""
        self.stream.synchronize()
        dist.barrier()  # Everyone has consumed the previous contents of recv.
        self.status.zero_()
        if self.rank in self.active:
            self.recv[self.rank].copy_(self.send[self.rank])
            for peer, view in sorted(self.peers.items()):
                send(
                    self.local,
                    view,
                    self.slot_bytes,
                    peer * self.slot_bytes,
                    self.rank * self.slot_bytes,
                    self.rank * 8,
                    from_dlpack(self.status[peer]),
                    with_signal,
                    self.cu_stream,
                )
            if with_signal:
                for peer in sorted(self.peers):
                    self.expected[peer] += 1
                    wait(
                        self.ready.data_ptr() + peer * 8,
                        self.expected[peer],
                        from_dlpack(self.status[peer]),
                        self.cu_stream,
                    )
        self.stream.synchronize()
        if self.status.count_nonzero().item():
            raise RuntimeError(
                f"NIXL device operation failed: {self.status.cpu().tolist()}"
            )
        dist.barrier()  # Teaching harness: also makes source reuse explicit.

    def close(self):
        self.stream.synchronize()
        for view in self.peers.values():
            self.agent.release_mem_view(view)
        self.agent.release_mem_view(self.local)
        dist.barrier()  # All remote views are gone before any owner deregisters.
        for peer in self.peers:
            self.agent.remove_remote_agent(f"cute-{peer}")
        self.agent.deregister_memory(self.registration)
        dist.destroy_process_group()


def run(example, rows, hidden):
    world = None
    try:
        world = World(rows, hidden)
        example(world)
        world.close()
    except BaseException:
        # Do not free remotely borrowed storage after an uncertain device failure.
        # torchrun terminates the job; this prototype has no recovery protocol.
        traceback.print_exc()
        os._exit(1)
