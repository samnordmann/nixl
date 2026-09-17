#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU compile/link smoke test for the complete NIXL CuTe device bitcode.

The compile-only kernel references every device export with fake views, then a
harmless ABI-version kernel is launched. A real PUT additionally requires two
live NIXL agents and UCX device-view setup. Set ``NIXL_SOURCE`` when running
directly from a source tree that has no built Python extension.
"""

from __future__ import annotations

import os
import sys
import types

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

if source := os.environ.get("NIXL_SOURCE"):
    # Import the source-only device package without requiring a host extension.
    # MemoryView registration needs this name during package initialization,
    # but the ABI-only kernel below never constructs one.
    class _DeviceViewHandle:
        pass

    root = types.ModuleType("nixl")
    root.__path__ = [os.path.join(source, "src/api/python")]
    setattr(root, "nixl_device_view_handle", _DeviceViewHandle)
    sys.modules["nixl"] = root

import nixl.device.cute as nixl_cute
from nixl.device.cute import _bindings


@cute.kernel
def _abi_kernel(result: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        result[0] = _bindings.abi_version()


@cute.jit
def _launch_abi(result: cute.Tensor, stream: cuda.CUstream):
    _abi_kernel(result).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)


@cute.kernel
def _grid_sync_kernel(result: cute.Tensor):
    """Force block one to consume a value published later by block zero."""

    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    address = result.iterator.toint()
    # CuTe DSL 4.5.1 cannot materialize a Uint64 compile-time literal whose
    # sign bit is set. Bit 62 still gives this runtime proof a distinctive
    # nontrivial payload without tripping that frontend limitation.
    cookie = cutlass.Uint64(0x4001D00D5EED1234)
    if block == 1:
        if lane == 0:
            nixl_cute.store_release_gpu_u64(address, 1)
    else:
        if lane == 0:
            nixl_cute.wait_acquire_gpu_u64(address, 1)
            started = nixl_cute.globaltimer_ns()
            now = started
            while now - started < 250_000:
                now = nixl_cute.globaltimer_ns()
            nixl_cute.store_release_gpu_u64(address + 8, cookie)

    nixl_cute.sync_grid()
    if block == 1:
        if lane == 0:
            observed = nixl_cute.load_acquire_gpu_u64(address + 8)
            nixl_cute.store_release_gpu_u64(address + 16, observed)


@cute.jit
def _launch_grid_sync(result: cute.Tensor, stream: cuda.CUstream):
    _grid_sync_kernel(result).launch(
        grid=[2, 1, 1],
        block=[32, 1, 1],
        stream=stream,
        cooperative=True,
    )


@cute.kernel
def _local_atomic_add_kernel(result: cute.Tensor):
    """Exercise the returned-prior contract used by route allocation."""

    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        address = result.iterator.toint()
        prior = nixl_cute.atomic_add_release_gpu_u64(address, 5)
        nixl_cute.store_release_gpu_u64(address + 8, prior)


@cute.jit
def _launch_local_atomic_add(result: cute.Tensor, stream: cuda.CUstream):
    _local_atomic_add_kernel(result).launch(
        grid=[1, 1, 1], block=[1, 1, 1], stream=stream
    )


@cute.kernel
def _all_exports_kernel(
    local: nixl_cute.MemoryView,
    remote: nixl_cute.MemoryView,
    status_results: cute.Tensor,
    value_results: cute.Tensor,
):
    """Reference every device operation so CuTe must lower its full body."""
    status_results[0] = nixl_cute.put(local, remote, 16, scope=nixl_cute.Scope.THREAD)
    status_results[1] = nixl_cute.put(local, remote, 16, scope=nixl_cute.Scope.WARP)
    status_results[2] = nixl_cute.atomic_add(
        remote, 1, offset=16, scope=nixl_cute.Scope.THREAD
    )
    status_results[3] = nixl_cute.atomic_add(
        remote, 1, offset=16, scope=nixl_cute.Scope.WARP
    )
    value_results[4] = cute.make_ptr(cutlass.Int8, nixl_cute.get_ptr(remote)).toint()
    status_results[5] = nixl_cute.put_post(
        local,
        remote,
        16,
        flags=nixl_cute.Flags.DEFER,
        scope=nixl_cute.Scope.THREAD,
    )
    status_results[6] = nixl_cute.put_post(
        local,
        remote,
        16,
        flags=nixl_cute.Flags.DEFER,
        scope=nixl_cute.Scope.WARP,
    )
    status_results[7] = nixl_cute.atomic_add_post(
        remote, 1, offset=16, scope=nixl_cute.Scope.THREAD
    )
    status_results[8] = nixl_cute.atomic_add_post(
        remote, 1, offset=16, scope=nixl_cute.Scope.WARP
    )
    value_results[9] = nixl_cute.globaltimer_ns()
    # Compile-only fake addresses: the all-exports kernel is never launched.
    value_results[10] = nixl_cute.load_acquire_system_u64(8)
    status_results[11] = nixl_cute.store_release_system_u64(8, 1)
    value_results[12] = nixl_cute.wait_acquire_system_u64(
        8, 0, scope=nixl_cute.Scope.THREAD
    )
    value_results[13] = nixl_cute.wait_acquire_system_u64(
        8, 0, scope=nixl_cute.Scope.WARP
    )
    # Compile-only fake source/view: this kernel is never launched.
    status_results[14] = nixl_cute.mapped_copy_warp(16, remote, 16)
    value_results[15] = nixl_cute.wait_acquire_system_u64_until(
        8, 0, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[16] = nixl_cute.wait_acquire_system_u64_until(
        8, 0, 1, scope=nixl_cute.Scope.WARP
    )
    status_results[17] = nixl_cute.mapped_copy_warp_readonly(16, remote, 16)
    status_results[18] = nixl_cute.mapped_copy_warp_ptr(16, 32, 16)
    status_results[19] = nixl_cute.mapped_copy_warp_ptr_readonly(16, 32, 16)
    status_results[20] = nixl_cute.fence_release_system()
    value_results[21] = nixl_cute.wait_acquire_system_u64_for(
        8, 0, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[22] = nixl_cute.wait_acquire_system_u64_for(
        8, 0, 1, scope=nixl_cute.Scope.WARP
    )
    value_results[23] = nixl_cute.wait_acquire_gpu_u64_for(
        8, 0, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[24] = nixl_cute.wait_acquire_gpu_u64_for(
        8, 0, 1, scope=nixl_cute.Scope.WARP
    )
    status_results[25] = nixl_cute.store_release_gpu_u64(8, 1)
    value_results[26] = nixl_cute.wait_acquire_gpu_u64(
        8, 0, scope=nixl_cute.Scope.THREAD
    )
    value_results[27] = nixl_cute.wait_acquire_gpu_u64(8, 0, scope=nixl_cute.Scope.WARP)
    value_results[28] = nixl_cute.wait_acquire_gpu_u64_or_abort(
        8, 0, 16, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[29] = nixl_cute.wait_acquire_gpu_u64_or_abort(
        8, 0, 16, 1, scope=nixl_cute.Scope.WARP
    )
    value_results[30] = nixl_cute.load_acquire_gpu_u64(8)
    status_results[31] = nixl_cute.sync_grid()
    status_results[32] = nixl_cute.atomic_max_release_gpu_u64(8, 1)
    status_results[33] = nixl_cute.compare_exchange_status_gpu_i32(8, -1)
    value_results[34] = nixl_cute.wait_acquire_system_u64_for_or_abort(
        8, 0, 16, 1, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[35] = nixl_cute.wait_acquire_system_u64_for_or_abort(
        8, 0, 16, 1, 1, scope=nixl_cute.Scope.WARP
    )
    value_results[36] = nixl_cute.wait_acquire_system_u64_or_abort(
        8, 0, 16, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[37] = nixl_cute.wait_acquire_system_u64_or_abort(
        8, 0, 16, 1, scope=nixl_cute.Scope.WARP
    )
    value_results[38] = nixl_cute.wait_acquire_system_u64_or_aborts(
        8, 0, 16, 1, 24, 1, scope=nixl_cute.Scope.THREAD
    )
    value_results[39] = nixl_cute.wait_acquire_system_u64_or_aborts(
        8, 0, 16, 1, 24, 1, scope=nixl_cute.Scope.WARP
    )
    status_results[40] = nixl_cute.atomic_max_release_system_u64(8, 1)
    value_results[41] = nixl_cute.atomic_add_release_gpu_u64(8, 1)


@cute.jit
def _compile_all_exports(
    local: nixl_cute.MemoryView,
    remote: nixl_cute.MemoryView,
    status_results: cute.Tensor,
    value_results: cute.Tensor,
):
    _all_exports_kernel(local, remote, status_results, value_results).launch(
        grid=[1, 1, 1], block=[32, 1, 1]
    )


def main() -> None:
    status_output = torch.zeros(42, dtype=torch.int32, device="cuda")
    value_output = torch.zeros(42, dtype=torch.uint64, device="cuda")
    status_tensor = from_dlpack(status_output).mark_layout_dynamic()
    value_tensor = from_dlpack(value_output).mark_layout_dynamic()

    # This callable is deliberately not invoked: the fake views carry no live
    # NIXL resources. Storing all statuses keeps each external call reachable
    # through LLVM/NVVM/PTX lowering and catches body-only linker regressions.
    for compile_index in range(2):
        # The second lowering is intentional. CUTLASS DSL caches extern FFI
        # declarations process-wide, so this catches cached-result signedness
        # regressions that a single compilation cannot expose.
        nixl_cute.compile(
            _compile_all_exports,
            nixl_cute.make_fake_memory_view("local", (64,)),
            nixl_cute.make_fake_memory_view("remote", (64,)),
            status_tensor,
            value_tensor,
        )
    print("cute_all_exports_compile=PASS sequential_compiles=2")

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    output_tensor = from_dlpack(output).mark_layout_dynamic()
    _launch_abi(output_tensor, stream)
    torch_stream.synchronize()
    actual = int(output[0].item())
    if actual != 3:
        raise RuntimeError(f"wrong NIXL CuTe device ABI version: {actual}")
    print(f"cute_abi_smoke=PASS abi_version={actual}")

    grid_result = torch.zeros(3, dtype=torch.uint64, device="cuda")
    grid_tensor = from_dlpack(grid_result).mark_layout_dynamic()
    _launch_grid_sync(grid_tensor, stream)
    torch_stream.synchronize()
    cookie = 0x4001D00D5EED1234
    observed = int(grid_result[2].item())
    if observed != cookie:
        raise RuntimeError(
            f"cooperative grid sync failed: expected {cookie:#x}, got {observed:#x}"
        )
    print("cute_cooperative_grid_sync=PASS blocks=2")

    atomic_result = torch.tensor([7, 0], dtype=torch.uint64, device="cuda")
    atomic_tensor = from_dlpack(atomic_result).mark_layout_dynamic()
    _launch_local_atomic_add(atomic_tensor, stream)
    torch_stream.synchronize()
    counter, prior = (int(value) for value in atomic_result.cpu().tolist())
    if (counter, prior) != (12, 7):
        raise RuntimeError(
            "same-GPU atomic-add returned the wrong allocation slot: "
            f"counter={counter}, prior={prior}"
        )
    print("cute_local_atomic_add=PASS counter=12 prior=7")


if __name__ == "__main__":
    main()
