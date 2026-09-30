# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torchrun GPU check. Set PYTHONPATH=examples/python/cute (plus installed NIXL)."""

import argparse
import os

import torch
import torch.distributed as dist
from common import run
from exchange import COUNTER_SLOTS, Exchange


def example(world, args):
    old = None
    phases = [
        tuple(range(world.size)),
        (0,),
        tuple(range(0, world.size, 2)),
        tuple(range(world.size)),
    ]
    for active in phases:
        world.membership(active)
        if old is not None:
            try:
                old()
                raise AssertionError("stale plan accepted")
            except RuntimeError as error:
                assert "membership" in str(error)
        exchange = Exchange(world, args.tile_bytes, args.threads, args.level)
        with torch.cuda.stream(torch.cuda.Stream()):
            try:
                exchange()
                raise AssertionError("wrong stream accepted")
            except RuntimeError as error:
                assert "stream" in str(error)
        for epoch in range(4):
            for peer in range(world.size):
                world.send[peer].fill_(world.rank * 16 + peer + epoch * 32)
            exchange()
            if world.rank in active:
                for peer in active:
                    expected = peer * 16 + world.rank + epoch * 32
                    torch.testing.assert_close(
                        world.recv[peer],
                        torch.full_like(world.recv[peer], expected),
                        rtol=0,
                        atol=0,
                    )
        exchange.check()
        if world.peers:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=world.stream):
                world.send.add_(1)
                exchange()
            for _ in range(10):
                graph.replay()
            exchange.check()
            for peer in active:
                expected = peer * 16 + world.rank + 96 + 10
                torch.testing.assert_close(
                    world.recv[peer],
                    torch.full_like(world.recv[peer], expected),
                    rtol=0,
                    atol=0,
                )
            del graph
        old = exchange
        dist.barrier()
        print(
            f"rank {world.rank}: active={active}, exchange/graph/guards PASS",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=1048576)
    parser.add_argument("--tile-bytes", type=int)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 2:
        parser.error("positive even byte count required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    with torch.cuda.stream(torch.cuda.Stream()):
        run(
            lambda world: example(world, args),
            rows=1,
            hidden=args.bytes // 2,
            counter_slots=COUNTER_SLOTS,
        )
