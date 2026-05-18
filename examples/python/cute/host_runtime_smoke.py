#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the host-side `nixl.cute.Agent` scaffold."""

import argparse

import torch

from nixl.cute import Agent, device_api_status


def run(device: str, n: int, backends: list[str]) -> None:
    tensor = torch.arange(n, device=device, dtype=torch.float32)
    agent = Agent("nixl-cute-host-smoke", backends=backends)

    with agent.register_tensor(tensor) as registered:
        metadata = agent.export_metadata([registered])
        view = agent.prepare_view(local=[registered])
        assert registered.address == tensor.data_ptr()
        assert registered.nbytes == tensor.numel() * tensor.element_size()
        assert len(metadata) > 0
        assert len(view.local) == 1

    status = device_api_status()
    print("PASS: NIXL CuTe host runtime")
    print(f"metadata bytes: {len(metadata)}")
    print(f"NIXL device API available: {status.available} ({status.reason})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--backends", default="UCX")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.device, args.n, [b for b in args.backends.split(",") if b])

