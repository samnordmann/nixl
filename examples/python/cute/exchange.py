# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream-ordered cooperative exchange with GPU-owned sequence/return credits.

Control-plane membership changes drain explicitly. Steady-state launches do
not synchronize the CPU. Use one stream and consume recv before the next call.
"""

from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import nixl_cute as ops
import torch
import torch.distributed as dist
from cutlass.cute.runtime import from_dlpack

COUNTER_SLOTS = 2


@cute.kernel
def small_kernel(
    local: cutlass.Uint64,
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    status: cute.Tensor,
    size: cutlass.Uint64,
    rank: cutlass.Int32,
    world_size: cutlass.Int32,
    threads: cutlass.Constexpr,
    level: cutlass.Constexpr,
):
    """One-peer latency path; a single CTA cannot wait on an unscheduled CTA."""
    tid, _, _ = cute.arch.thread_idx()
    peer, view = plan[0, 0], plan[0, 1]
    if tid == 0:
        previous = step[0]
        if previous > 0:
            ops.atomic_add(1, view, (world_size + rank) * 8)
        ops.wait(ready.iterator.toint() + (world_size + peer) * 8, previous)
        step[0] = previous + 1
    cute.arch.sync_threads()
    result = ops.put(local, view, size, peer * size, rank * size, level=level)
    ops.fence()
    cute.arch.sync_threads()
    if tid == 0:
        status[0, 0] = result
        if result == 0:
            status[0, 0] = ops.atomic_add(1, view, rank * 8)
        ops.wait(ready.iterator.toint() + peer * 8, step[0])


@cute.jit
def launch_small(
    local: cutlass.Uint64,
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    status: cute.Tensor,
    size: cutlass.Uint64,
    rank: cutlass.Int32,
    world_size: cutlass.Int32,
    threads: cutlass.Constexpr,
    level: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    small_kernel(
        local, plan, ready, step, status, size, rank, world_size, threads, level
    ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)


@cute.kernel
def begin_kernel(
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    rank: cutlass.Int32,
    world_size: cutlass.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    previous = step[0]
    cute.arch.sync_threads()
    if tid == 0:
        step[0] = previous + 1
    # One credit covers the whole receive slab. This CTA returns all credits
    # before waiting, so no unscheduled send CTA can hold a required credit.
    if previous > 0:
        for peer in range(tid, plan.shape[0], 128):
            # Previous consumer kernels have completed on this stream.
            ops.atomic_add(1, plan[peer, 1], (world_size + rank) * 8)
    cute.arch.sync_threads()
    for peer in range(tid, plan.shape[0], 128):
        ops.wait(ready.iterator.toint() + (world_size + plan[peer, 0]) * 8, previous)


@cute.kernel
def send_kernel(
    local: cutlass.Uint64,
    plan: cute.Tensor,
    status: cute.Tensor,
    size: cutlass.Uint64,
    rank: cutlass.Int32,
    tile_bytes: cutlass.Constexpr,
    level: cutlass.Constexpr,
):
    tile, peer_row, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    peer, view = plan[peer_row, 0], plan[peer_row, 1]
    offset = cutlass.Uint64(tile * tile_bytes)
    count = cutlass.min(cutlass.Uint64(tile_bytes), size - offset)
    result = ops.put(
        local, view, count, peer * size + offset, rank * size + offset, level=level
    )
    # Complete all writers' system stores before the next kernel signals once
    # per peer. Kernel ordering supplies the grid-wide join without spin barriers.
    ops.fence()
    if tid == 0:
        status[peer_row, tile] = result


@cute.kernel
def wait_kernel(
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    status: cute.Tensor,
    rank: cutlass.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    for peer in range(tid, plan.shape[0], 128):
        failed = cutlass.Int32(0)
        for tile in range(status.shape[1]):
            failed |= status[peer, tile]
        if failed == 0:
            # send_kernel already system-fenced every writer and has completed.
            status[peer, 0] = ops.atomic_add(1, plan[peer, 1], rank * 8)
    cute.arch.sync_threads()
    for peer in range(tid, plan.shape[0], 128):
        ops.wait(ready.iterator.toint() + plan[peer, 0] * 8, step[0])


@cute.jit
def launch_begin(
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    rank: cutlass.Int32,
    world_size: cutlass.Int32,
    stream: cuda.CUstream,
):
    begin_kernel(plan, ready, step, rank, world_size).launch(
        grid=(1, 1, 1), block=(128, 1, 1), stream=stream
    )


@cute.jit
def launch_send(
    local: cutlass.Uint64,
    plan: cute.Tensor,
    status: cute.Tensor,
    size: cutlass.Uint64,
    rank: cutlass.Int32,
    tile_bytes: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    level: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    send_kernel(local, plan, status, size, rank, tile_bytes, level).launch(
        grid=(cute.ceil_div(size, tile_bytes), plan.shape[0], 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cute.jit
def launch_wait(
    plan: cute.Tensor,
    ready: cute.Tensor,
    step: cute.Tensor,
    status: cute.Tensor,
    rank: cutlass.Int32,
    stream: cuda.CUstream,
):
    wait_kernel(plan, ready, step, status, rank).launch(
        grid=(1, 1, 1), block=(128, 1, 1), stream=stream
    )


class Exchange:
    """Prepared epoch. Recreate after membership changes; never outlive its views."""

    def __init__(self, world, tile_bytes=None, threads=128, level=ops.BLOCK):
        if tile_bytes is None:
            # Keep large transfers near 64 CTAs per peer; preserve 256B alignment.
            tile_bytes = (
                16384
                if world.slot_bytes <= 65536
                else max(262144, ((world.slot_bytes + 16383) // 16384) * 256)
            )
        if level not in (ops.THREAD, ops.WARP, ops.BLOCK):
            raise ValueError("supported levels: THREAD, WARP, BLOCK")
        if (level == ops.THREAD and threads != 1) or (
            level == ops.WARP and threads != 32
        ):
            raise ValueError("use exactly one thread/warp for those cooperation levels")
        if (
            threads < 1
            or threads > 1024
            or (level == ops.BLOCK and threads % 32)
            or tile_bytes <= 0
        ):
            raise ValueError("invalid launch geometry")
        if world.ready.numel() != world.size * COUNTER_SLOTS:
            raise ValueError("reserve COUNTER_SLOTS counters per rank")
        self.world, self.active = world, world.active
        self.tile_bytes = tile_bytes
        # Cold-path only: finish the old epoch everywhere before clearing credits.
        world.stream.synchronize()
        dist.barrier()
        world.generation += 1  # Also invalidate an earlier plan of the same peers.
        self.generation = world.generation
        self.local_send, self.local_recv = (
            world.send[world.rank],
            world.recv[world.rank],
        )
        self.tiles = (world.slot_bytes + tile_bytes - 1) // tile_bytes
        self.step = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.plan = torch.tensor(
            sorted(world.peers.items()), dtype=torch.int64, device="cuda"
        )
        self.status = torch.zeros(
            (max(1, len(world.peers)), self.tiles), dtype=torch.int32, device="cuda"
        )
        world.ready.zero_()
        self.launches = []
        if world.peers:
            plan, step, ready, status = map(
                from_dlpack, (self.plan, self.step, world.ready, self.status)
            )
            stream = world.cu_stream
            kernels = (
                (launch_begin, (plan, ready, step, world.rank, world.size), ()),
                (
                    launch_send,
                    (
                        world.local,
                        plan,
                        status,
                        world.slot_bytes,
                        world.rank,
                    ),
                    (tile_bytes, threads, level),
                ),
                (launch_wait, (plan, ready, step, status, world.rank), ()),
            )
            # Measured latency path; never relies on multi-CTA residency.
            if world.slot_bytes <= 4096 and len(world.peers) == 1 and self.tiles == 1:
                kernels = (
                    (
                        launch_small,
                        (
                            world.local,
                            plan,
                            ready,
                            step,
                            status,
                            world.slot_bytes,
                            world.rank,
                            world.size,
                        ),
                        (threads, level),
                    ),
                )
            for fn, args, constants in kernels:
                compiled = cute.compile(fn, *args, *constants, stream)
                self.launches.append(partial(compiled, *args, stream))
        world.stream.synchronize()
        dist.barrier()  # All counters reset before any new epoch publishes.

    def __call__(self):
        if self.generation != self.world.generation:
            raise RuntimeError("rebuild Exchange after membership changes")
        if torch.cuda.current_stream() != self.world.stream:
            raise RuntimeError("produce, exchange and consume on the prepared stream")
        if self.world.rank in self.active:
            self.local_recv.copy_(self.local_send)
            for launch in self.launches:
                launch()
        return self.world.recv

    def check(self):
        self.world.stream.synchronize()
        if self.status.count_nonzero().item():
            raise RuntimeError(f"NIXL device error: {self.status.cpu().tolist()}")
