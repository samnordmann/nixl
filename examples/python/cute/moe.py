# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable MoE communication, not an optimized inference backend.

CPU routing metadata goes through Gloo; BF16 activations/results go through NIXL
from CuTe kernels. All processes stay alive during graceful membership changes.
"""

import argparse

import torch

TOKENS, HIDDEN, TOP_K, EXPERTS_PER_RANK = 8, 128, 2, 2


def membership_plan(text, world_size):
    phases = tuple(
        tuple(int(rank) for rank in phase.split(",")) for phase in text.split(";")
    )
    for active in phases:
        if (
            not active
            or len(set(active)) != len(active)
            or any(not 0 <= r < world_size for r in active)
        ):
            raise ValueError(
                "each phase needs distinct ranks within the fixed world size"
            )
    return phases


def route(rank, active, world_size, generation):
    """A tiny CPU router; records preserve the token/slot/expert correspondence."""
    x = (
        torch.arange(TOKENS * HIDDEN).reshape(TOKENS, HIDDEN).float() / 128
        + rank
        + generation
    ).to(torch.bfloat16)
    scores = torch.randn(
        TOKENS,
        world_size * EXPERTS_PER_RANK,
        generator=torch.Generator().manual_seed(100 + rank + generation),
    )
    for owner in set(range(world_size)) - set(active):
        scores[:, owner * EXPERTS_PER_RANK : (owner + 1) * EXPERTS_PER_RANK] = (
            -torch.inf
        )
    values, experts = scores.topk(TOP_K, dim=1)
    weights = values.softmax(dim=1)
    records = [[] for _ in range(world_size)]
    if rank in active:
        for token in range(TOKENS):
            for slot in range(TOP_K):
                expert = int(experts[token, slot])
                records[expert // EXPERTS_PER_RANK].append((token, slot, expert))
    return x, weights, records


def golden(x, weights, records):
    output = torch.zeros_like(x, dtype=torch.float32)
    for bucket in records:
        for token, slot, expert in bucket:
            expert_output = (x[token].float() * (expert + 1)).to(torch.bfloat16)
            output[token] += expert_output.float() * weights[token, slot]
    return output.to(torch.bfloat16)


def example(world, membership):
    from common import gather

    for generation, active in enumerate(membership_plan(membership, world.size)):
        world.membership(active)
        x, weights, records = route(world.rank, active, world.size, generation)
        # Only route records/counts cross the CPU control plane, not activations.
        all_records = gather(records)
        packed = torch.zeros_like(world.send, device="cpu")
        for owner, bucket in enumerate(records):
            for row, (token, _, _) in enumerate(bucket):
                packed[owner, row] = x[token]
        world.send.copy_(packed)
        world.exchange()  # Dispatch: recv[source, row, hidden].

        world.send.zero_()
        if world.rank in active:
            for source in active:
                bucket = all_records[source][world.rank]
                for row, (_, _, expert) in enumerate(bucket):
                    # Stand-in expert; replace this part with grouped GEMM.
                    world.send[source, row] = (
                        world.recv[source, row].float() * (expert + 1)
                    ).to(torch.bfloat16)
        world.exchange()  # Return expert outputs to their original token owners.

        output = torch.zeros((TOKENS, HIDDEN), dtype=torch.float32, device="cuda")
        for owner, bucket in enumerate(records):
            for row, (token, slot, _) in enumerate(bucket):
                output[token] += world.recv[owner, row].float() * float(
                    weights[token, slot]
                )
        torch.testing.assert_close(
            output.to(torch.bfloat16).cpu(), golden(x, weights, records), rtol=0, atol=0
        )
        print(
            f"rank {world.rank}: generation {generation}, active={active}, MoE PASS",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", default="0,1;0;0,1")
    args = parser.parse_args()
    from common import run

    run(
        lambda world: example(world, args.membership),
        rows=TOKENS * TOP_K,
        hidden=HIDDEN,
    )


if __name__ == "__main__":
    main()
