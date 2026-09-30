# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MoE: GPU pack -> dispatch -> stand-in experts -> return -> combine.

Routing is prepared once per membership epoch. No CPU sync or allocation in
the forward path. This is padded communication, not an optimized MoE backend:
the router is CPU-based, experts are multiplication, and padding costs bandwidth.
"""

import argparse
import os

import torch
from common import gather, run
from exchange import COUNTER_SLOTS, Exchange
from moe import HIDDEN, TOKENS, TOP_K, golden, membership_plan, route


class PreparedMoE:
    def __init__(self, world, generation):
        self.world = world
        x, weights, self.records = route(
            world.rank, world.active, world.size, generation
        )
        self.cpu_x, self.cpu_weights = x, weights
        all_records = gather(self.records)
        capacity = world.send.shape[1]
        packing = torch.full((world.size, capacity), TOKENS, dtype=torch.int64)
        combining = torch.zeros((TOKENS, TOP_K), dtype=torch.int64)
        factors = torch.zeros((world.size, capacity, 1), dtype=torch.bfloat16)
        for owner, bucket in enumerate(self.records):
            for row, (token, slot, _) in enumerate(bucket):
                packing[owner, row] = token
                combining[token, slot] = owner * capacity + row
        for source in world.active:
            for row, (_, _, expert) in enumerate(all_records[source][world.rank]):
                factors[source, row] = expert + 1
        if world.rank not in world.active:
            weights.zero_()
        # A final zero row supplies padding without a separate fill kernel.
        self.x = torch.cat((x, torch.zeros_like(x[:1]))).cuda()
        self.packing = packing.flatten().cuda()
        self.combining = combining.flatten().cuda()
        self.factors = factors.cuda()
        self.weights = weights.reshape(-1, 1).cuda()
        self.returned = torch.empty(
            (TOKENS * TOP_K, HIDDEN), device="cuda", dtype=torch.bfloat16
        )
        self.weighted = torch.empty_like(self.returned, dtype=torch.float32)
        self.summed = torch.zeros((TOKENS, HIDDEN), device="cuda", dtype=torch.float32)
        self.output = torch.zeros_like(self.summed, dtype=torch.bfloat16)
        self.send_rows, self.recv_rows = world.send.flatten(0, 1), world.recv.flatten(
            0, 1
        )
        self.weighted_routes = self.weighted.view(TOKENS, TOP_K, HIDDEN)
        self.exchange = Exchange(world)

    def __call__(self):
        if self.world.rank in self.world.active:
            torch.index_select(self.x, 0, self.packing, out=self.send_rows)
            self.exchange()  # Dispatch, recv[source, packed row].
            torch.mul(self.world.recv, self.factors, out=self.world.send)
            self.exchange()  # Return expert outputs; safely reuses the same slab.
            torch.index_select(self.recv_rows, 0, self.combining, out=self.returned)
            torch.mul(self.returned, self.weights, out=self.weighted)
            torch.sum(self.weighted_routes, dim=1, out=self.summed)
            self.output.copy_(self.summed)
        return self.output


def example(world, membership, graph_replays):
    for generation, active in enumerate(membership_plan(membership, world.size)):
        world.membership(active)
        model = PreparedMoE(world, generation)
        model()  # Warm up every kernel before capture.
        model.exchange.check()
        torch.testing.assert_close(
            model.output.cpu(),
            golden(model.cpu_x, model.cpu_weights, model.records),
            rtol=0,
            atol=0,
        )
        if graph_replays:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=world.stream):
                model.x.mul_(2)  # Changing payloads catch stale-read/credit bugs.
                model()
            for _ in range(graph_replays):
                graph.replay()
            model.exchange.check()
            torch.testing.assert_close(
                model.output.cpu(),
                golden(
                    model.cpu_x * (2**graph_replays), model.cpu_weights, model.records
                ),
                rtol=0,
                atol=0,
            )
            del graph  # Never replay a graph after releasing its epoch's views.
        print(
            f"rank {world.rank}: epoch {generation}, active={active}, prepared MoE PASS",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", default="0,1;0;0,1")
    parser.add_argument("--graph-replays", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.graph_replays <= 8:
        parser.error("graph-replays must be 0..8 for the doubling-payload check")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    with torch.cuda.stream(torch.cuda.Stream()):
        run(
            lambda world: example(world, args.membership, args.graph_replays),
            rows=TOKENS * TOP_K,
            hidden=HIDDEN,
            counter_slots=COUNTER_SLOTS,
        )
