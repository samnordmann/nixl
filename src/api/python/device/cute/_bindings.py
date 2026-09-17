# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed CuTe DSL declarations for the stable NIXL device C ABI."""

import cutlass
import cutlass.cute as cute
from cutlass.cute import BitCode

from ._bitcode import device_bitcode_path
from ._mlir import LLVMPtr


class _NIXLDeviceBitCode(BitCode):
    """Resolve the rank-local per-SM artifact only when CuTe traces an extern."""

    def __init__(self):
        # BitCode is a frozen dataclass. The dynamic property deliberately has
        # no stored value because package import can precede rank-local set_device.
        pass

    @property
    def path(self) -> str:
        return device_bitcode_path()


_BC = _NIXLDeviceBitCode()


@cute.extern(name="nixl_cute_put_thread_wait", source=_BC)
def put_thread_wait(  # noqa: E704 - CuTe extern declaration
    local_view: LLVMPtr,
    local_index: cutlass.Uint32,
    local_offset: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_put_warp_wait", source=_BC)
def put_warp_wait(  # noqa: E704 - CuTe extern declaration
    local_view: LLVMPtr,
    local_index: cutlass.Uint32,
    local_offset: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_put_thread_post", source=_BC)
def put_thread_post(  # noqa: E704 - CuTe extern declaration
    local_view: LLVMPtr,
    local_index: cutlass.Uint32,
    local_offset: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_put_warp_post", source=_BC)
def put_warp_post(  # noqa: E704 - CuTe extern declaration
    local_view: LLVMPtr,
    local_index: cutlass.Uint32,
    local_offset: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_add_thread_wait", source=_BC)
def atomic_add_thread_wait(  # noqa: E704 - CuTe extern declaration
    value: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_add_warp_wait", source=_BC)
def atomic_add_warp_wait(  # noqa: E704 - CuTe extern declaration
    value: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_add_thread_post", source=_BC)
def atomic_add_thread_post(  # noqa: E704 - CuTe extern declaration
    value: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_add_warp_post", source=_BC)
def atomic_add_warp_post(  # noqa: E704 - CuTe extern declaration
    value: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    channel: cutlass.Uint32,
    flags: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_get_ptr", source=_BC)
def get_ptr(  # type: ignore[empty-body]  # noqa: E704 - CuTe extern declaration
    view: LLVMPtr, index: cutlass.Uint32
) -> LLVMPtr: ...


@cute.extern(name="nixl_cute_mapped_copy_warp_ptr", source=_BC)
def mapped_copy_warp_ptr(  # noqa: E704 - CuTe extern declaration
    source_address: cutlass.Uint64,
    destination_address: cutlass.Uint64,
    size: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_mapped_copy_warp_ptr_readonly", source=_BC)
def mapped_copy_warp_ptr_readonly(  # noqa: E704 - CuTe extern declaration
    source_address: cutlass.Uint64,
    destination_address: cutlass.Uint64,
    size: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_mapped_copy_warp", source=_BC)
def mapped_copy_warp(  # noqa: E704 - CuTe extern declaration
    source_address: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_mapped_copy_warp_readonly", source=_BC)
def mapped_copy_warp_readonly(  # noqa: E704 - CuTe extern declaration
    source_address: cutlass.Uint64,
    remote_view: LLVMPtr,
    remote_index: cutlass.Uint32,
    remote_offset: cutlass.Uint64,
    size: cutlass.Uint64,
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_globaltimer_ns", source=_BC)
def globaltimer_ns() -> cutlass.Uint64: ...  # noqa: E704 - CuTe extern declaration


@cute.extern(name="nixl_cute_load_acquire_system_u64", source=_BC)
def load_acquire_system_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_load_acquire_gpu_u64", source=_BC)
def load_acquire_gpu_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_thread", source=_BC)
def wait_acquire_system_u64_thread(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, least: cutlass.Uint64
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_warp", source=_BC)
def wait_acquire_system_u64_warp(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, least: cutlass.Uint64
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_thread", source=_BC)
def wait_acquire_gpu_u64_thread(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, least: cutlass.Uint64
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_warp", source=_BC)
def wait_acquire_gpu_u64_warp(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, least: cutlass.Uint64
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_or_abort_thread", source=_BC)
def wait_acquire_gpu_u64_or_abort_thread(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_or_abort_thread", source=_BC)
def wait_acquire_system_u64_or_abort_thread(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_or_abort_warp", source=_BC)
def wait_acquire_system_u64_or_abort_warp(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_or_aborts_thread", source=_BC)
def wait_acquire_system_u64_or_aborts_thread(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    local_abort_address: cutlass.Uint64,
    local_abort_least: cutlass.Uint64,
    peer_abort_address: cutlass.Uint64,
    peer_abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_or_aborts_warp", source=_BC)
def wait_acquire_system_u64_or_aborts_warp(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    local_abort_address: cutlass.Uint64,
    local_abort_least: cutlass.Uint64,
    peer_abort_address: cutlass.Uint64,
    peer_abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_or_abort_warp", source=_BC)
def wait_acquire_gpu_u64_or_abort_warp(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_thread_until", source=_BC)
def wait_acquire_system_u64_thread_until(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    deadline_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_warp_until", source=_BC)
def wait_acquire_system_u64_warp_until(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    deadline_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_thread_for", source=_BC)
def wait_acquire_system_u64_thread_for(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_warp_for", source=_BC)
def wait_acquire_system_u64_warp_for(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_thread_for_or_abort", source=_BC)
def wait_acquire_system_u64_thread_for_or_abort(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_system_u64_warp_for_or_abort", source=_BC)
def wait_acquire_system_u64_warp_for_or_abort(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    abort_address: cutlass.Uint64,
    abort_least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_thread_for", source=_BC)
def wait_acquire_gpu_u64_thread_for(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_wait_acquire_gpu_u64_warp_for", source=_BC)
def wait_acquire_gpu_u64_warp_for(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64,
    least: cutlass.Uint64,
    timeout_ns: cutlass.Uint64,
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_store_release_system_u64", source=_BC)
def store_release_system_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Uint64
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_store_release_gpu_u64", source=_BC)
def store_release_gpu_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Uint64
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_add_release_gpu_u64", source=_BC)
def atomic_add_release_gpu_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Uint64
) -> cutlass.Uint64: ...


@cute.extern(name="nixl_cute_atomic_max_release_gpu_u64", source=_BC)
def atomic_max_release_gpu_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Uint64
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_atomic_max_release_system_u64", source=_BC)
def atomic_max_release_system_u64(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Uint64
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_compare_exchange_status_gpu_i32", source=_BC)
def compare_exchange_status_gpu_i32(  # noqa: E704 - CuTe extern declaration
    address: cutlass.Uint64, value: cutlass.Int32
) -> cutlass.Int32: ...


@cute.extern(name="nixl_cute_sync_grid", source=_BC)
def sync_grid() -> cutlass.Int32: ...  # noqa: E704 - CuTe extern declaration


@cute.extern(name="nixl_cute_fence_release_system", source=_BC)
def fence_release_system() -> cutlass.Int32: ...  # noqa: E704 - CuTe extern declaration


@cute.extern(name="nixl_cute_abi_version", source=_BC)
def abi_version() -> cutlass.Uint32: ...  # noqa: E704 - CuTe extern declaration
