# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "contrib" / "build_device_bitcode.py"
_DEVICE_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "api"
    / "gpu"
    / "ucx"
    / "cute"
    / "nixl_device_cute.cu"
)
_SPEC = importlib.util.spec_from_file_location("build_device_bitcode", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_copy_warp_keeps_eight_vector_load_ahead_in_body_and_tail():
    source = _DEVICE_SOURCE.read_text(encoding="utf-8")
    copy_warp = source.split("copyWarp(uint64_t source_address", 1)[1].split(
        "mappedCopyWarp(uint64_t source_address", 1
    )[0]
    steady_state, tail = copy_warp.split(
        "const uint64_t tail = full_vectors + lane;", 1
    )

    assert "constexpr uint64_t unroll = 8;" in steady_state
    assert "NixlCuteInt4 values[unroll];" in steady_state
    assert "tail_batch" not in copy_warp
    assert "NixlCuteInt4 tail_values[unroll];" in tail
    assert tail.count("for (uint32_t j = 0; j < unroll; ++j)") == 2
    assert tail.index("tail_values[j] = loadMapped") < tail.index("storeNoAllocate")


def test_copy_warp_tail_covers_every_remainder_vector_once():
    warp_size = 32
    unroll = 8

    for remainder in range(warp_size * unroll):
        visited = [
            lane + j * warp_size
            for j in range(unroll)
            for lane in range(warp_size)
            if lane + j * warp_size < remainder
        ]
        assert sorted(visited) == list(range(remainder))


def test_clean_llvm_ir_removes_nvvm_ftz_and_function_alignment():
    source = """\
define dso_local i32 @f() align 16 {
  ret i32 0
}
!llvm.module.flags = !{!1, !2, !10}
!1 = !{i32 4, !\"nvvm-reflect-ftz\", i32 1}
!2 = !{i32 1, !\"wchar_size\", i32 4}
!10 = !{i32 1, !\"PIC Level\", i32 2}
"""

    result = _MODULE.clean_llvm_ir(source)

    assert "align 16" not in result
    assert "nvvm-reflect-ftz" not in result
    assert "!llvm.module.flags = !{!2, !10}" in result
    assert "!2 =" in result
    assert "!10 =" in result


def test_device_mode_defines_disable_asserts_only_in_production():
    assert _MODULE._device_mode_defines(False) == ("-DNDEBUG",)
    assert _MODULE._device_mode_defines(True) == (
        "-DNIXL_CUTE_ENABLE_DEVICE_VALIDATION",
    )


@pytest.mark.parametrize(
    "flag",
    (
        "-DNDEBUG",
        "-UNDEBUG",
        "-DNIXL_CUTE_ENABLE_DEVICE_VALIDATION",
        "-DNIXL_CUTE_ENABLE_DEVICE_VALIDATION=0",
    ),
)
def test_builder_rejects_extra_flags_that_override_manifest_mode(flag):
    with pytest.raises(RuntimeError, match="builder-owned device mode macro"):
        _MODULE._validate_extra_flags([flag])


def test_builder_accepts_non_mode_extra_flags():
    _MODULE._validate_extra_flags(
        ["-DNIXL_CUTE_DISABLE_FORCEINLINE", "-DPROJECT_SPECIFIC_FLAG=1"]
    )


_VALID_ABI_IR = """\
define dso_local i32 @nixl_cute_abi_version() {
  ret i32 3
}
define dso_local i32 @nixl_cute_put_thread_wait(ptr %a, i32 %b, i64 %c, ptr %d, i32 %e, i64 %f, i64 %g, i32 %h, i64 %i) {
  ret i32 0
}
define dso_local i32 @nixl_cute_put_warp_wait(ptr %a, i32 %b, i64 %c, ptr %d, i32 %e, i64 %f, i64 %g, i32 %h, i64 %i) {
  ret i32 0
}
define dso_local i32 @nixl_cute_put_thread_post(ptr %a, i32 %b, i64 %c, ptr %d, i32 %e, i64 %f, i64 %g, i32 %h, i64 %i) {
  ret i32 1
}
define dso_local i32 @nixl_cute_put_warp_post(ptr %a, i32 %b, i64 %c, ptr %d, i32 %e, i64 %f, i64 %g, i32 %h, i64 %i) {
  ret i32 1
}
define dso_local i32 @nixl_cute_atomic_add_thread_wait(i64 %a, ptr %b, i32 %c, i64 %d, i32 %e, i64 %f) {
  ; atom.add.release.sys.u64
  ret i32 0
}
define dso_local i32 @nixl_cute_atomic_add_warp_wait(i64 %a, ptr %b, i32 %c, i64 %d, i32 %e, i64 %f) {
  ; atom.add.release.sys.u64
  ret i32 0
}
define dso_local i32 @nixl_cute_atomic_add_thread_post(i64 %a, ptr %b, i32 %c, i64 %d, i32 %e, i64 %f) {
  ; atom.add.release.sys.u64
  ret i32 1
}
define dso_local i32 @nixl_cute_atomic_add_warp_post(i64 %a, ptr %b, i32 %c, i64 %d, i32 %e, i64 %f) {
  ; atom.add.release.sys.u64
  ret i32 1
}
define dso_local i64 @nixl_cute_globaltimer_ns() {
  ret i64 0
}
define dso_local i64 @nixl_cute_load_acquire_system_u64(i64 %a) {
  ; ld.acquire.sys.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_load_acquire_gpu_u64(i64 %a) {
  ; ld.acquire.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_thread(i64 %a, i64 %b) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_warp(i64 %a, i64 %b) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_thread(i64 %a, i64 %b) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_warp(i64 %a, i64 %b) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_or_abort_thread(i64 %a, i64 %b, i64 %c, i64 %d) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_or_abort_warp(i64 %a, i64 %b, i64 %c, i64 %d) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_or_abort_thread(i64 %a, i64 %b, i64 %c, i64 %d) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_or_abort_warp(i64 %a, i64 %b, i64 %c, i64 %d) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_or_aborts_thread(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f) {
  ; ld.acquire.sys.global.u64
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_or_aborts_warp(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f) {
  ; ld.acquire.sys.global.u64
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_thread_until(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_warp_until(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_thread_for(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_warp_for(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.sys.global.u64
  ; ld.relaxed.sys.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_thread_for_or_abort(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e) {
  ; ld.acquire.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_system_u64_warp_for_or_abort(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e) {
  ; ld.acquire.sys.global.u64
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.sys.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_thread_for(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ret i64 0
}
define dso_local i64 @nixl_cute_wait_acquire_gpu_u64_warp_for(i64 %a, i64 %b, i64 %c) {
  ; ld.acquire.gpu.global.u64
  ; ld.relaxed.gpu.global.u64
  ; llvm.nvvm.bar.warp.sync
  ret i64 0
}
define dso_local i32 @nixl_cute_store_release_system_u64(i64 %a, i64 %b) {
  ; st.release.sys.global.u64
  ret i32 0
}
define dso_local i32 @nixl_cute_store_release_gpu_u64(i64 %a, i64 %b) {
  ; st.release.gpu.global.u64
  ret i32 0
}
define dso_local i64 @nixl_cute_atomic_add_release_gpu_u64(i64 %a, i64 %b) {
  ; atom.add.release.gpu.global.u64
  ret i64 0
}
define dso_local i32 @nixl_cute_atomic_max_release_gpu_u64(i64 %a, i64 %b) {
  ; atom.max.release.gpu.global.u64
  ret i32 0
}
define dso_local i32 @nixl_cute_atomic_max_release_system_u64(i64 %a, i64 %b) {
  ; atom.max.release.sys.global.u64
  ret i32 0
}
define dso_local i32 @nixl_cute_compare_exchange_status_gpu_i32(i64 %a, i32 %b) {
  ; atom.cas.release.gpu.global.b32
  ret i32 0
}
define dso_local i32 @nixl_cute_sync_grid() {
  ret i32 0
}
define dso_local i32 @nixl_cute_fence_release_system() {
  ; fence.release.sys
  ret i32 0
}
define dso_local ptr @nixl_cute_get_ptr(ptr %a, i32 %b) {
  ret ptr null
}
define dso_local i32 @nixl_cute_mapped_copy_warp_ptr(i64 %a, i64 %b, i64 %c) {
  ; ld.global.L1::no_allocate.L2::256B.v4.s32
  ; st.global.L1::no_allocate.v4.s32
  ; llvm.nvvm.bar.warp.sync
  ret i32 0
}
define dso_local i32 @nixl_cute_mapped_copy_warp_ptr_readonly(i64 %a, i64 %b, i64 %c) {
  ; ld.global.nc.L1::no_allocate.L2::256B.v4.s32
  ; st.global.L1::no_allocate.v4.s32
  ; llvm.nvvm.bar.warp.sync
  ret i32 0
}
define dso_local i32 @nixl_cute_mapped_copy_warp(i64 %a, ptr %b, i32 %c, i64 %d, i64 %e) {
  ; ld.global.L1::no_allocate.L2::256B.v4.s32
  ; st.global.L1::no_allocate.v4.s32
  ; llvm.nvvm.bar.warp.sync
  ret i32 -9
}
define dso_local i32 @nixl_cute_mapped_copy_warp_readonly(i64 %a, ptr %b, i32 %c, i64 %d, i64 %e) {
  ; ld.global.nc.L1::no_allocate.L2::256B.v4.s32
  ; st.global.L1::no_allocate.v4.s32
  ; llvm.nvvm.bar.warp.sync
  ret i32 -9
}
"""


def test_validate_llvm_ir_accepts_exact_public_abi():
    _MODULE.validate_llvm_ir(_VALID_ABI_IR)


def test_validate_llvm_ir_accepts_exact_allowlisted_intrinsic_call():
    valid = _VALID_ABI_IR.replace(
        "define dso_local i32 @nixl_cute_sync_grid() {\n  ret i32 0",
        "define dso_local i32 @nixl_cute_sync_grid() {\n"
        "  %tid = tail call noundef range(i32 0, 1024) i32 "
        "@llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "  call void @llvm.nvvm.barrier.sync(i32 %tid)\n"
        "  ret i32 0",
    )
    valid += (
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "declare void @llvm.nvvm.barrier.sync(i32)\n"
    )

    _MODULE.validate_llvm_ir(valid)


def test_validate_llvm_ir_rejects_retained_helper_definition():
    invalid = _VALID_ABI_IR + "define internal void @helper() {\n  ret void\n}\n"

    with pytest.raises(RuntimeError, match="retained helper definitions"):
        _MODULE.validate_llvm_ir(invalid)


def test_validate_llvm_ir_rejects_non_allowlisted_external_call():
    invalid = _VALID_ABI_IR.replace(
        "define dso_local i32 @nixl_cute_sync_grid() {\n  ret i32 0",
        "define dso_local i32 @nixl_cute_sync_grid() {\n"
        "  call void @external_helper()\n"
        "  ret i32 0",
    )

    with pytest.raises(RuntimeError, match="non-allowlisted symbol"):
        _MODULE.validate_llvm_ir(invalid)


def test_validate_llvm_ir_rejects_non_allowlisted_declaration():
    invalid = _VALID_ABI_IR + "declare void @external_helper()\n"

    with pytest.raises(RuntimeError, match="non-allowlisted declarations"):
        _MODULE.validate_llvm_ir(invalid)


def test_validate_llvm_ir_rejects_indirect_call():
    invalid = _VALID_ABI_IR.replace(
        "define dso_local i32 @nixl_cute_sync_grid() {\n  ret i32 0",
        "define dso_local i32 @nixl_cute_sync_grid() {\n"
        "  call void %function_pointer()\n"
        "  ret i32 0",
    )

    with pytest.raises(RuntimeError, match="indirect or unparseable"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._PRODUCTION_FORBIDDEN_IR_SYMBOLS)
def test_validate_llvm_ir_rejects_production_diagnostic_symbols(symbol):
    invalid = _VALID_ABI_IR + f"declare void @{symbol}()\n"

    with pytest.raises(RuntimeError, match="forbidden diagnostic symbols"):
        _MODULE.validate_llvm_ir(invalid)

    _MODULE.validate_llvm_ir(invalid, device_validation=True)


@pytest.mark.parametrize(
    "forceinline,attribute",
    [(True, "alwaysinline"), (False, "noinline")],
)
def test_validate_llvm_ir_requires_selected_export_inline_policy(
    forceinline, attribute
):
    attributed = (
        _VALID_ABI_IR.replace(") {", ") #0 {") + f"attributes #0 = {{ {attribute} }}\n"
    )
    _MODULE.validate_llvm_ir(attributed, forceinline=forceinline)

    opposite = "noinline" if forceinline else "alwaysinline"
    with pytest.raises(RuntimeError, match=f"required {attribute}"):
        _MODULE.validate_llvm_ir(
            attributed.replace(attribute, opposite), forceinline=forceinline
        )


@pytest.mark.parametrize(
    "symbol,token",
    [
        (symbol, token)
        for symbol, required in _MODULE._MAPPED_COPY_REQUIRED_IR.items()
        for token in required
    ],
)
def test_validate_llvm_ir_rejects_mapped_copy_instruction_regression(symbol, token):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    assert token in _VALID_ABI_IR[body_start:body_end]
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(token, "removed", 1)
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost required production instructions"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._WARP_ACQUIRE_REQUIRED_IR)
def test_validate_llvm_ir_rejects_warp_acquire_without_memory_barrier(symbol):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(
            "llvm.nvvm.bar.warp.sync", "removed", 1
        )
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost the warp barrier"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol,token", _MODULE._SCOPED_MEMORY_REQUIRED_IR.items())
def test_validate_llvm_ir_rejects_weakened_or_widened_memory_scope(symbol, token):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(token, "wrong.scope", 1)
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost required scoped operation"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol,token", _MODULE._RELAXED_POLL_REQUIRED_IR.items())
def test_validate_llvm_ir_rejects_acquire_only_poll_loop(symbol, token):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(token, "removed", 1)
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost relaxed failed-poll operation"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize(
    "symbol,token",
    [
        (symbol, token)
        for symbol, required in _MODULE._ABORT_OBSERVATION_REQUIRED_IR.items()
        for token in required
    ],
)
def test_validate_llvm_ir_rejects_missing_local_abort_observation(symbol, token):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(token, "removed", 1)
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost same-GPU abort observation"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._TIMER_FREE_WAIT_SYMBOLS)
def test_validate_llvm_ir_rejects_timer_in_timer_free_wait(symbol):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_end]
        + "\n  ; mov.u64 %globaltimer"
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="unexpectedly reads %globaltimer"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._DUAL_ABORT_REQUIRED_IR_COUNTS)
def test_validate_llvm_ir_rejects_missing_peer_abort_observation(symbol):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    body = _VALID_ABI_IR[body_start:body_end]
    invalid_body = body.replace("  ; ld.acquire.sys.global.u64\n", "", 1)
    invalid = _VALID_ABI_IR[:body_start] + invalid_body + _VALID_ABI_IR[body_end:]
    with pytest.raises(RuntimeError, match="lost peer-abort observation"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._ATOMIC_RELEASE_REQUIRED_IR)
def test_validate_llvm_ir_rejects_relaxed_cuda_ipc_atomic(symbol):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace(
            "atom.add.release.sys.u64", "atom.add.sys.u64", 1
        )
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost release ordering"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize("symbol", _MODULE._SYSTEM_RELEASE_FENCE_REQUIRED_IR)
def test_validate_llvm_ir_rejects_missing_system_release_fence(symbol):
    body_start = _VALID_ABI_IR.index(f"@{symbol}(")
    body_end = _VALID_ABI_IR.index("\n}", body_start)
    invalid = (
        _VALID_ABI_IR[:body_start]
        + _VALID_ABI_IR[body_start:body_end].replace("fence.release.sys", "removed", 1)
        + _VALID_ABI_IR[body_end:]
    )
    with pytest.raises(RuntimeError, match="lost its system-scope release fence"):
        _MODULE.validate_llvm_ir(invalid)


@pytest.mark.parametrize(
    "invalid_ir, message",
    [
        (
            _VALID_ABI_IR.replace(
                "@nixl_cute_get_ptr(ptr %a, i32 %b)",
                "@nixl_cute_get_ptr(ptr %a, i64 %b)",
            ),
            "invalid LLVM signature",
        ),
        (
            _VALID_ABI_IR.replace(
                "@nixl_cute_get_ptr(ptr %a, i32 %b)",
                "@nixl_cute_get_ptr(ptr addrspace(1) %a, i32 %b)",
            ),
            "invalid LLVM signature",
        ),
        (
            _VALID_ABI_IR.replace("ret i32 3", "ret i32 4", 1),
            "does not return",
        ),
        (
            _VALID_ABI_IR.replace(
                "define dso_local i32 @nixl_cute_abi_version",
                "define available_externally i32 @nixl_cute_abi_version",
            ),
            "non-public linkage",
        ),
        (
            _VALID_ABI_IR.replace(
                "define dso_local i32 @nixl_cute_abi_version",
                "define dso_local fastcc i32 @nixl_cute_abi_version",
            ),
            "non-C calling convention",
        ),
        (
            _VALID_ABI_IR.replace(
                "define dso_local i32 @nixl_cute_abi_version",
                "define dso_local x86_stdcallcc i32 @nixl_cute_abi_version",
            ),
            "non-C calling convention",
        ),
        (
            _VALID_ABI_IR.replace(
                "define dso_local i32 @nixl_cute_abi_version",
                "define dso_local amdgpu_gfx i32 @nixl_cute_abi_version",
            ),
            "non-C calling convention",
        ),
        (
            _VALID_ABI_IR
            + "define dso_local i32 @nixl_cute_unexpected() {\n  ret i32 0\n}\n",
            "unexpected NIXL CuTe exports",
        ),
    ],
)
def test_validate_llvm_ir_rejects_abi_drift(invalid_ir, message):
    with pytest.raises(RuntimeError, match=message):
        _MODULE.validate_llvm_ir(invalid_ir)
