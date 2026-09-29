# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare serial peer launches with one bounded batch; not end-to-end MoE timing."""

import argparse
import json
import statistics
from functools import partial

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.runtime import from_dlpack

from common import gather, run, send, send_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=4096, dest="size")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 2 or min(args.repeats, args.rounds) <= 0:
        parser.error("bytes must be positive/even; repeats and rounds must be positive")

    def example(world):
        if not 2 <= world.size <= 9:
            raise ValueError("use two to nine GPU processes on one node")
        world.membership(tuple(range(world.size)))

        def check():
            expected = torch.stack(
                [
                    torch.full_like(world.recv[p], p * 10 + world.rank + epoch * 32)
                    for p in range(world.size)
                ]
            )
            torch.testing.assert_close(world.recv, expected, rtol=0, atol=0)

        # Changing payloads and switching modes catches stale signal/cached-plan bugs.
        for epoch in range(4):
            for peer in range(world.size):
                world.send[peer].fill_(world.rank * 10 + peer + epoch * 32)
            world.exchange(batched=bool(epoch % 2))
            check()

        # Freeze the same launchers once: decorated calls regenerate IR even on
        # a cache hit. Neither compilation nor DLPack conversion belongs in timing.
        launchers = {False: [], True: []}
        for peer, view in sorted(world.peers.items()):
            params = (
                world.local,
                view,
                world.slot_bytes,
                peer * world.slot_bytes,
                world.rank * world.slot_bytes,
                world.rank * 8,
                from_dlpack(world.status[peer]),
            )
            compiled = cute.compile(send, *params, True, world.cu_stream)
            launchers[False].append(partial(compiled, *params, world.cu_stream))
        params = (
            world.local,
            from_dlpack(world.batch_plan),
            world.slot_bytes,
            world.rank * 8,
            from_dlpack(world.status),
        )
        compiled = cute.compile(send_batch, *params, True, world.cu_stream)
        launchers[True].append(partial(compiled, *params, world.cu_stream))

        def post(batched):
            for launch in launchers[batched]:
                launch()

        posted = 0
        for batched in (False, True):
            for _ in range(5):
                post(batched)
                posted += 1
        world.stream.synchronize()
        dist.barrier()

        samples = {"serial": [], "batch": []}
        for _ in range(args.rounds):
            for mode in ("serial", "batch", "batch", "serial"):
                dist.barrier()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                with torch.cuda.nvtx.range(f"cute-peer-{mode}"):
                    # Prequeue behind a device delay; exclude that delay from timing.
                    # Check a trace before treating this as a GPU-only latency result.
                    torch.cuda._sleep(100_000_000)
                    start.record(world.stream)
                    for _ in range(args.repeats):
                        post(mode == "batch")
                    end.record(world.stream)
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 1000 / args.repeats)
                posted += args.repeats

        # The microbenchmark reuses unchanged source data, with no receiver reads.
        # Account for every publication before returning to the normal protocol.
        for peer in world.peers:
            world.expected[peer] += posted
        world.exchange(batched=True)
        check()
        all_samples = gather(samples)
        if world.rank == 0:
            worst = {
                mode: [max(values) for values in zip(*(s[mode] for s in all_samples))]
                for mode in samples
            }
            median = {mode: statistics.median(values) for mode, values in worst.items()}
            print(
                json.dumps(
                    {
                        "ranks": world.size,
                        "bytes_per_peer": args.size,
                        "repeats": args.repeats,
                        "rounds": args.rounds,
                        "eager": True,
                        "precompiled": True,
                        "scope": "outgoing PUTs + signals, no receive waits or host barriers",
                        "worst_rank_samples_us": worst,
                        "median_us": median,
                        "batch_vs_serial_pct": 100
                        * (median["batch"] / median["serial"] - 1),
                        "correctness": "PASS",
                    }
                ),
                flush=True,
            )

    run(example, rows=1, hidden=args.size // 2)


if __name__ == "__main__":
    main()
