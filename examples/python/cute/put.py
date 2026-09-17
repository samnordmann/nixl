# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank PUT, optionally followed by a GPU-observed signal."""

import argparse

import torch

from common import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", action="store_true")
    args = parser.parse_args()

    def example(world):
        if world.size != 2:
            raise ValueError("launch with torchrun --nproc-per-node=2")
        world.membership((0, 1))
        for iteration in range(3):
            # Changing every iteration also checks signal epochs and buffer reuse.
            for peer in range(2):
                world.send[peer].fill_(world.rank * 10 + peer + iteration * 100)
            world.exchange(with_signal=args.signal)
            expected = torch.stack(
                [
                    torch.full_like(
                        world.recv[peer], peer * 10 + world.rank + iteration * 100
                    )
                    for peer in range(2)
                ]
            )
            torch.testing.assert_close(world.recv, expected, rtol=0, atol=0)
        print(
            f"rank {world.rank}: PUT{' + signal' if args.signal else ''} PASS",
            flush=True,
        )

    run(example, rows=1, hidden=256)


if __name__ == "__main__":
    main()
