# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-node, graph-replay all-to-all: complete exchange including reuse credits.

Optional same-protocol NVCC reference and NCCL all_to_all_single comparator.
Report slowest-rank times, not outgoing-only submission time. No routing/GEMM.
"""

import argparse
import ctypes
import json
import os
import statistics
from functools import partial
from pathlib import Path

import nixl_cute as ops
import torch
import torch.distributed as dist
from common import gather, run
from exchange import COUNTER_SLOTS, Exchange


def native_launch(function, *args):
    error = function(*args)
    if error:
        raise RuntimeError(f"native reference CUDA launch error {error}")


def benchmark(world, args):
    if world.size < 2:
        raise ValueError("benchmark requires at least two ranks")
    world.membership(tuple(range(world.size)))
    group = dist.new_group(backend="nccl") if args.nccl else None
    implementations = ["cute"]
    if args.reference_so:
        native = ctypes.CDLL(str(args.reference_so.resolve())).exchange
        native.argtypes = [
            ctypes.c_uint64,
            *([ctypes.c_void_p] * 4),
            ctypes.c_uint64,
            *([ctypes.c_int] * 6),
            ctypes.c_void_p,
        ]
        native.restype = ctypes.c_int
        implementations.append("native")
    if args.nccl:
        implementations.append("nccl")
    # Nonuniform BF16 payload; reused buffers are intentional steady-state data.
    pattern = torch.arange(world.send[0].numel(), device="cuda", dtype=torch.int32)
    pattern = pattern.remainder(251).to(torch.bfloat16).view_as(world.send[0])
    plans = []  # Keep compiled modules/storage alive across all captures.
    for implementation in implementations:
        level = {"thread": ops.THREAD, "warp": ops.WARP, "block": ops.BLOCK}[args.level]
        exchange = Exchange(world, args.tile_bytes, args.threads, level)
        plans.append(exchange)
        call = exchange
        if implementation == "native":
            exchange.launches = [
                partial(
                    native_launch,
                    native,
                    world.local,
                    exchange.plan.data_ptr(),
                    world.ready.data_ptr(),
                    exchange.step.data_ptr(),
                    exchange.status.data_ptr(),
                    world.slot_bytes,
                    world.rank,
                    world.size,
                    exchange.tiles,
                    exchange.tile_bytes,
                    args.threads,
                    level,
                    world.stream.cuda_stream,
                )
            ]
        elif implementation == "nccl":
            call = partial(dist.all_to_all_single, world.recv, world.send, group=group)
        for peer in range(world.size):
            world.send[peer].copy_(pattern + world.rank * 7 + peer)
        call()
        exchange.check()
        for peer in range(world.size):
            torch.testing.assert_close(
                world.recv[peer], pattern + peer * 7 + world.rank, rtol=0, atol=0
            )
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph, stream=world.stream):
            call()
            call()  # GPU rendezvous absorbs host launch skew before timing.
            start.record(world.stream)
            for _ in range(args.repeat):
                call()
            end.record(world.stream)
        for _ in range(3):
            graph.replay()
        world.stream.synchronize()
        allocated = torch.cuda.memory_allocated()
        samples = []
        torch.cuda.nvtx.range_push(f"measure:{implementation}")
        for _ in range(args.trials):
            dist.barrier()  # Outside the measured graph interval.
            graph.replay()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / args.repeat)
        torch.cuda.nvtx.range_pop()
        assert torch.cuda.memory_allocated() == allocated
        exchange.check()
        for peer in range(world.size):
            torch.testing.assert_close(
                world.recv[peer], pattern + peer * 7 + world.rank, rtol=0, atol=0
            )
        rank_samples = gather(samples)
        if world.rank == 0:
            maxima = [max(row[i] for row in rank_samples) for i in range(args.trials)]
            median = statistics.median(maxima)
            print(
                "RESULT "
                + json.dumps(
                    dict(
                        implementation=implementation,
                        ranks=world.size,
                        bytes_per_peer=world.slot_bytes,
                        tile_bytes=exchange.tile_bytes,
                        threads=args.threads,
                        level=args.level,
                        graph=True,
                        repeat=args.repeat,
                        median_us=median,
                        rank_samples_us=rank_samples,
                        outbound_GBs=(world.size - 1)
                        * world.slot_bytes
                        / (median * 1000),
                    )
                ),
                flush=True,
            )
        del graph
        dist.barrier()
    if group is not None:
        dist.destroy_process_group(group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=1048576)
    parser.add_argument("--tile-bytes", type=int)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--level", choices=("thread", "warp", "block"), default="block")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--reference-so", type=Path)
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 2 or args.repeat <= 0 or args.trials <= 0:
        parser.error(
            "positive even byte count and positive repeat/trial counts required"
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    with torch.cuda.stream(torch.cuda.Stream()):
        run(
            lambda world: benchmark(world, args),
            rows=1,
            hidden=args.bytes // 2,
            counter_slots=COUNTER_SLOTS,
        )
