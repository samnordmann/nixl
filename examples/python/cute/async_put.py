# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit, do independent work, then complete on the GPU. Two ranks required.

One private request per participating thread, retained through completion.
CUDA-IPC completes during submission; overlap requires a capable backend.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import nixl_cute as ops
import torch
from common import run
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def kernel(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    rank: cutlass.Int32,
    requests: cute.Tensor,
    work: cute.Tensor,
    status: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    request = requests.iterator.toint() + tid * ops.REQUEST_BYTES
    result = ops.put_async(
        local,
        remote,
        4096,
        request,
        (1 - rank) * 4096,
        rank * 4096,
        local_index=1,
        remote_index=1,
        channel=2,
        level=ops.BLOCK,
    )
    work[tid] = tid * tid  # Independent of both transfer buffers.
    result = ops.complete(request, result, level=ops.BLOCK)
    status[tid, 0] = result
    ops.fence()
    cute.arch.sync_threads()
    if tid == 0:
        if result == ops.SUCCESS:
            result = ops.atomic_add(
                7,
                remote,
                rank * 8,
                index=0,
                channel=2,
                flags=ops.DEFER,
                request=request,
            )
            status[0, 1] = ops.complete(request, result)
        # Terminal statuses must not touch a request (zero is intentional).
        status[0, 2] = ops.progress(0, ops.SUCCESS)
        status[0, 3] = ops.progress(0, -1) + 1
        status[0, 4] = cutlass.Int32(ops.get_ptr(remote, index=1) != 0)


@cute.jit
def launch(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    rank: cutlass.Int32,
    requests: cute.Tensor,
    work: cute.Tensor,
    status: cute.Tensor,
    stream: cuda.CUstream,
):
    kernel(local, remote, rank, requests, work, status).launch(
        grid=(1, 1, 1), block=(128, 1, 1), stream=stream
    )


def example(world):
    if world.size != 2:
        raise ValueError("run with two ranks")
    world.membership((0, 1))
    peer = 1 - world.rank
    # Nonzero descriptor indices are intentional: duplicate the local entry,
    # and reverse the remote [payload, counter] entries for this example.
    local = world.agent.prep_mem_view(
        world.agent.get_xfer_descs([world.send, world.send])
    )
    descs = world.agent.get_remote_descs(
        list(reversed(world.metadata[peer][1])), mem_type="VRAM"
    )
    remote = world.agent.prep_mem_view(descs)
    requests = torch.empty((128, ops.REQUEST_BYTES), device="cuda", dtype=torch.uint8)
    work = torch.empty(128, device="cuda", dtype=torch.int32)
    status = torch.zeros((128, 5), device="cuda", dtype=torch.int32)
    args = (
        local,
        remote,
        world.rank,
        *map(from_dlpack, (requests, work, status)),
        world.cu_stream,
    )
    compiled = cute.compile(launch, *args)
    world.send.fill_(world.rank + 1)
    compiled(*args)
    # This is a validation boundary, not part of submission/completion.
    world.stream.synchronize()
    import torch.distributed as dist

    dist.barrier()
    assert status[:, :4].count_nonzero().item() == 0, status.cpu().tolist()
    torch.testing.assert_close(
        world.recv[peer], torch.full_like(world.recv[peer], peer + 1), rtol=0, atol=0
    )
    assert world.ready[peer].item() == 7
    torch.testing.assert_close(
        work, torch.arange(128, device="cuda", dtype=torch.int32).square()
    )
    world.agent.release_mem_view(remote)
    world.agent.release_mem_view(local)
    print(
        f"rank {world.rank}: request lifecycle, descriptors, channel, atomic and mapped pointer PASS",
        flush=True,
    )


if __name__ == "__main__":
    run(example, rows=1, hidden=2048)
