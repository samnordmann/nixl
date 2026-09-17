# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL device operations for NVIDIA CuTe DSL kernels.

This optional package lazily selects verified, architecture-matched NIXL device
bitcode for compiled kernels. Host-side registration, metadata exchange,
connection setup, and memory-view preparation remain explicit NIXL agent
operations.
"""

import os
import warnings
from importlib import metadata

_REQUIRED_CUTLASS_DSL_VERSION = "4.5.1"
_UNSAFE_CUTLASS_DSL_OVERRIDE = "NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL"


def _dependency_error(detail: str) -> str:
    return (
        f"nixl.device.cute requires CUTLASS DSL "
        f"{_REQUIRED_CUTLASS_DSL_VERSION}; {detail}.\n"
        "CUDA 12: pip install 'nixl[cute-cu12]' or 'nixl-cu12[cute]'.\n"
        "CUDA 13: no extra is declared because CUTLASS DSL 4.5.1 issue "
        "#3259 makes its cu13 wheel set nondeterministic; use CUTLASS "
        "setup.sh --cu13 in an isolated, independently verified environment.\n"
        f"Source-only development may set {_UNSAFE_CUTLASS_DSL_OVERRIDE}=1 "
        "to bypass version verification explicitly."
    )


def _verify_cutlass_dsl_version() -> None:
    unsafe_override = os.environ.get(_UNSAFE_CUTLASS_DSL_OVERRIDE) == "1"
    try:
        installed_version = metadata.version("nvidia-cutlass-dsl")
    except Exception as exc:
        if not unsafe_override:
            raise ImportError(
                _dependency_error("distribution metadata is missing or unreadable")
            ) from exc
        warnings.warn(
            _dependency_error("distribution metadata could not be verified"),
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if installed_version == _REQUIRED_CUTLASS_DSL_VERSION:
        return
    if not unsafe_override:
        raise ImportError(_dependency_error(f"found version {installed_version!r}"))
    warnings.warn(
        _dependency_error(f"found unqualified version {installed_version!r}"),
        RuntimeWarning,
        stacklevel=2,
    )


_verify_cutlass_dsl_version()

try:
    import cutlass.cute  # noqa: F401
except ImportError as exc:
    raise ImportError(
        _dependency_error("the Python package cannot be imported")
    ) from exc

from ._bitcode import (  # noqa: E402
    NIXL_CUTE_ABI_VERSION,
    BitcodeNotFoundError,
    BitcodeVerificationError,
    UnsupportedArchitectureError,
)
from .memory import MemoryView, compile, make_fake_memory_view  # noqa: E402
from .ops import (  # noqa: E402
    atomic_add,
    atomic_add_post,
    atomic_add_release_gpu_u64,
    atomic_max_release_gpu_u64,
    atomic_max_release_system_u64,
    compare_exchange_status_gpu_i32,
    fence_release_system,
    get_ptr,
    globaltimer_ns,
    load_acquire_gpu_u64,
    load_acquire_system_u64,
    mapped_copy_warp,
    mapped_copy_warp_ptr,
    mapped_copy_warp_ptr_readonly,
    mapped_copy_warp_readonly,
    put,
    put_post,
    store_release_gpu_u64,
    store_release_system_u64,
    sync_grid,
    wait_acquire_gpu_u64,
    wait_acquire_gpu_u64_for,
    wait_acquire_gpu_u64_or_abort,
    wait_acquire_system_u64,
    wait_acquire_system_u64_for,
    wait_acquire_system_u64_for_or_abort,
    wait_acquire_system_u64_or_abort,
    wait_acquire_system_u64_or_aborts,
    wait_acquire_system_u64_until,
)
from .topology import (  # noqa: E402
    query_peer_native_atomics,
    require_peer_native_atomics,
)
from .types import (  # noqa: E402
    NIXL_ERR_BACKEND,
    NIXL_ERR_CANCELED,
    NIXL_ERR_INVALID_PARAM,
    NIXL_ERR_MISMATCH,
    NIXL_ERR_NO_TELEMETRY,
    NIXL_ERR_NOT_ALLOWED,
    NIXL_ERR_NOT_FOUND,
    NIXL_ERR_NOT_POSTED,
    NIXL_ERR_NOT_SUPPORTED,
    NIXL_ERR_REMOTE_DISCONNECT,
    NIXL_ERR_REPOST_ACTIVE,
    NIXL_ERR_UNKNOWN,
    NIXL_IN_PROG,
    NIXL_SUCCESS,
    Flags,
    Scope,
    Status,
)

__all__ = [
    "BitcodeNotFoundError",
    "BitcodeVerificationError",
    "Flags",
    "MemoryView",
    "NIXL_CUTE_ABI_VERSION",
    "NIXL_ERR_BACKEND",
    "NIXL_ERR_CANCELED",
    "NIXL_ERR_INVALID_PARAM",
    "NIXL_ERR_MISMATCH",
    "NIXL_ERR_NOT_ALLOWED",
    "NIXL_ERR_NOT_FOUND",
    "NIXL_ERR_NOT_POSTED",
    "NIXL_ERR_NOT_SUPPORTED",
    "NIXL_ERR_NO_TELEMETRY",
    "NIXL_ERR_REMOTE_DISCONNECT",
    "NIXL_ERR_REPOST_ACTIVE",
    "NIXL_ERR_UNKNOWN",
    "NIXL_IN_PROG",
    "NIXL_SUCCESS",
    "Scope",
    "Status",
    "UnsupportedArchitectureError",
    "atomic_add",
    "atomic_add_post",
    "atomic_add_release_gpu_u64",
    "atomic_max_release_gpu_u64",
    "atomic_max_release_system_u64",
    "compare_exchange_status_gpu_i32",
    "compile",
    "fence_release_system",
    "get_ptr",
    "globaltimer_ns",
    "load_acquire_gpu_u64",
    "load_acquire_system_u64",
    "make_fake_memory_view",
    "mapped_copy_warp",
    "mapped_copy_warp_ptr",
    "mapped_copy_warp_ptr_readonly",
    "mapped_copy_warp_readonly",
    "put",
    "put_post",
    "query_peer_native_atomics",
    "require_peer_native_atomics",
    "store_release_gpu_u64",
    "store_release_system_u64",
    "sync_grid",
    "wait_acquire_gpu_u64",
    "wait_acquire_gpu_u64_for",
    "wait_acquire_gpu_u64_or_abort",
    "wait_acquire_system_u64",
    "wait_acquire_system_u64_for",
    "wait_acquire_system_u64_for_or_abort",
    "wait_acquire_system_u64_or_abort",
    "wait_acquire_system_u64_or_aborts",
    "wait_acquire_system_u64_until",
]
