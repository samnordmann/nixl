# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype CuTe calls. Handles come from nixl_agent.prep_mem_view()."""

from pathlib import Path

import cutlass
import cutlass.cute as cute

_bitcode = cute.BitCode(str(Path(__file__).with_name("device.bc")))


@cute.extern(name="cute_nixl_put", source=_bitcode)
def _put(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    size: cutlass.Uint64,
    local_offset: cutlass.Uint64,
    remote_offset: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_signal", source=_bitcode)
def _signal(remote: cutlass.Uint64, offset: cutlass.Uint64) -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_wait", source=_bitcode)
def _wait(address: cutlass.Uint64, expected: cutlass.Uint64) -> cutlass.Int32: ...


def put(local, remote, size, local_offset=0, remote_offset=0):
    """Copy descriptor 0 and wait on the GPU. Returns a NIXL status, not a CPU wait."""
    result = _put(
        cutlass.Uint64(local),
        cutlass.Uint64(remote),
        cutlass.Uint64(size),
        cutlass.Uint64(local_offset),
        cutlass.Uint64(remote_offset),
    )
    # Retag cached extern results: CuTe 4.5.1 may otherwise retain signless i32.
    return cutlass.Uint32(result).bitcast(cutlass.Int32)


def signal(remote, offset=0):
    """Increment a 64-bit counter in remote descriptor 1 after preceding PUTs."""
    result = _signal(cutlass.Uint64(remote), cutlass.Uint64(offset))
    return cutlass.Uint32(result).bitcast(cutlass.Int32)


def wait(address, expected):
    """System-acquire a local 64-bit counter before consuming remote payloads."""
    return cutlass.Uint32(
        _wait(cutlass.Uint64(address), cutlass.Uint64(expected))
    ).bitcast(cutlass.Int32)
