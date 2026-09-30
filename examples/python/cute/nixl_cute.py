# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin device bindings; registration, peer discovery and lifetime remain NIXL's.

No stable ABI. Cooperative calls require every lane and uniform arguments,
except for private request storage. Rebuild bitcode with the host NIXL/UCX.
"""

from pathlib import Path

import cutlass
import cutlass.cute as cute

THREAD, WARP, BLOCK = 0, 1, 2
SUCCESS, IN_PROGRESS, DEFER = 0, 1, 1
REQUEST_BYTES = 64  # Native nixlGpuXferStatusH, not a serialized ABI.
_bitcode = cute.BitCode(str(Path(__file__).with_name("device.bc")))


@cute.extern(name="cute_nixl_put", source=_bitcode)
def _put(
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    size: cutlass.Uint64,
    local_index: cutlass.Uint64,
    remote_index: cutlass.Uint64,
    local_offset: cutlass.Uint64,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
    request: cutlass.Uint64,
    level: cutlass.Int32,
) -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_progress", source=_bitcode)
def _progress(
    request: cutlass.Uint64, status: cutlass.Int32, level: cutlass.Int32
) -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_atomic_add", source=_bitcode)
def _atomic_add(
    value: cutlass.Uint64,
    remote: cutlass.Uint64,
    index: cutlass.Uint64,
    offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
    request: cutlass.Uint64,
    level: cutlass.Int32,
) -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_get_ptr", source=_bitcode)
def _get_ptr(remote: cutlass.Uint64, index: cutlass.Uint64) -> cutlass.Uint64: ...


@cute.extern(name="cute_nixl_fence", source=_bitcode)
def _fence() -> cutlass.Int32: ...


@cute.extern(name="cute_nixl_wait", source=_bitcode)
def _wait(address: cutlass.Uint64, expected: cutlass.Uint64) -> cutlass.Int32: ...


def _status(value):
    # CuTe 4.5.1 cached extern results can otherwise retain signless i32.
    return cutlass.Uint32(value).bitcast(cutlass.Int32)


def put(
    local,
    remote,
    size,
    local_offset=0,
    remote_offset=0,
    *,
    local_index=0,
    remote_index=0,
    channel=0,
    flags=0,
    level=THREAD,
    request=0,
):
    """PUT with native indices, offsets, channel, flags and cooperation level.

    request=0 completes on the GPU before returning. A nonzero address submits
    without a wrapper wait: retain one 64-byte, 64-byte-aligned slot per caller
    until completion. Do not reuse source, request or views while in progress.
    Backend submission can itself block or complete immediately.
    """
    return _status(
        _put(
            cutlass.Uint64(local),
            cutlass.Uint64(remote),
            cutlass.Uint64(size),
            cutlass.Uint64(local_index),
            cutlass.Uint64(remote_index),
            cutlass.Uint64(local_offset),
            cutlass.Uint64(remote_offset),
            cutlass.Uint32(channel),
            cutlass.Uint64(flags),
            cutlass.Uint64(request),
            cutlass.Int32(level),
        )
    )


def put_async(
    local,
    remote,
    size,
    request,
    local_offset=0,
    remote_offset=0,
    *,
    local_index=0,
    remote_index=0,
    channel=0,
    flags=DEFER,
    level=THREAD,
):
    """Request-backed submission; caller owns explicit progress/completion."""
    return put(
        local,
        remote,
        size,
        local_offset,
        remote_offset,
        local_index=local_index,
        remote_index=remote_index,
        channel=channel,
        flags=flags,
        level=level,
        request=request,
    )


def progress(request, status, *, level=THREAD):
    """One native progress step; a prior terminal status remains unchanged."""
    return _status(
        _progress(cutlass.Uint64(request), cutlass.Int32(status), cutlass.Int32(level))
    )


@cute.jit
def complete(
    request: cutlass.Uint64, status: cutlass.Int32, *, level: cutlass.Constexpr = THREAD
):
    """GPU wait; independent compute or submissions can precede this call."""
    while status == IN_PROGRESS:
        status = progress(request, status, level=level)
    return status


def atomic_add(
    value, remote, offset=0, *, index=1, channel=0, flags=0, request=0, level=THREAD
):
    """Native cooperative add; nonzero request separates submission/completion."""
    return _status(
        _atomic_add(
            cutlass.Uint64(value),
            cutlass.Uint64(remote),
            cutlass.Uint64(index),
            cutlass.Uint64(offset),
            cutlass.Uint32(channel),
            cutlass.Uint64(flags),
            cutlass.Uint64(request),
            cutlass.Int32(level),
        )
    )


def fence():
    """System release fence; every cooperative writer must participate."""
    return _status(_fence())


def signal(remote, offset=0, *, index=1, channel=0):
    """Fence this caller, then increment a remote counter after completed PUTs.

    Cooperative writers must first all fence and synchronize their group.
    This is not an inter-node flush primitive for incomplete deferred PUTs.
    """
    fence()
    return atomic_add(1, remote, offset, index=index, channel=channel)


def wait(address, expected):
    """System-acquire a local 64-bit counter before consuming remote payloads."""
    return _status(_wait(cutlass.Uint64(address), cutlass.Uint64(expected)))


def get_ptr(remote, index=0):
    """Remote mapped address, or zero if unavailable. Not valid for local views."""
    return cutlass.Uint64(_get_ptr(cutlass.Uint64(remote), cutlass.Uint64(index)))
