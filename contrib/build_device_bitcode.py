#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and validate the LLVM bitcode consumed by the NIXL CuTe DSL API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

NIXL_CUTE_ABI_VERSION = 3
SUPPORTED_LLVM_MAJOR = 20
EXPORTED_SIGNATURES = {
    "nixl_cute_abi_version": ("i32", ()),
    "nixl_cute_put_thread_wait": (
        "i32",
        ("ptr", "i32", "i64", "ptr", "i32", "i64", "i64", "i32", "i64"),
    ),
    "nixl_cute_put_warp_wait": (
        "i32",
        ("ptr", "i32", "i64", "ptr", "i32", "i64", "i64", "i32", "i64"),
    ),
    "nixl_cute_put_thread_post": (
        "i32",
        ("ptr", "i32", "i64", "ptr", "i32", "i64", "i64", "i32", "i64"),
    ),
    "nixl_cute_put_warp_post": (
        "i32",
        ("ptr", "i32", "i64", "ptr", "i32", "i64", "i64", "i32", "i64"),
    ),
    "nixl_cute_atomic_add_thread_wait": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i32", "i64"),
    ),
    "nixl_cute_atomic_add_warp_wait": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i32", "i64"),
    ),
    "nixl_cute_atomic_add_thread_post": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i32", "i64"),
    ),
    "nixl_cute_atomic_add_warp_post": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i32", "i64"),
    ),
    "nixl_cute_globaltimer_ns": ("i64", ()),
    "nixl_cute_load_acquire_system_u64": ("i64", ("i64",)),
    "nixl_cute_load_acquire_gpu_u64": ("i64", ("i64",)),
    "nixl_cute_wait_acquire_system_u64_thread": ("i64", ("i64", "i64")),
    "nixl_cute_wait_acquire_system_u64_warp": ("i64", ("i64", "i64")),
    "nixl_cute_wait_acquire_gpu_u64_thread": ("i64", ("i64", "i64")),
    "nixl_cute_wait_acquire_gpu_u64_warp": ("i64", ("i64", "i64")),
    "nixl_cute_wait_acquire_gpu_u64_or_abort_thread": (
        "i64",
        ("i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_gpu_u64_or_abort_warp": (
        "i64",
        ("i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_or_abort_thread": (
        "i64",
        ("i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_or_abort_warp": (
        "i64",
        ("i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_or_aborts_thread": (
        "i64",
        ("i64", "i64", "i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_or_aborts_warp": (
        "i64",
        ("i64", "i64", "i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_thread_until": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_warp_until": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_thread_for": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_warp_for": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_thread_for_or_abort": (
        "i64",
        ("i64", "i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_system_u64_warp_for_or_abort": (
        "i64",
        ("i64", "i64", "i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_gpu_u64_thread_for": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_wait_acquire_gpu_u64_warp_for": (
        "i64",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_store_release_system_u64": ("i32", ("i64", "i64")),
    "nixl_cute_store_release_gpu_u64": ("i32", ("i64", "i64")),
    "nixl_cute_atomic_add_release_gpu_u64": ("i64", ("i64", "i64")),
    "nixl_cute_atomic_max_release_gpu_u64": ("i32", ("i64", "i64")),
    "nixl_cute_atomic_max_release_system_u64": ("i32", ("i64", "i64")),
    "nixl_cute_compare_exchange_status_gpu_i32": ("i32", ("i64", "i32")),
    "nixl_cute_sync_grid": ("i32", ()),
    "nixl_cute_fence_release_system": ("i32", ()),
    "nixl_cute_get_ptr": ("ptr", ("ptr", "i32")),
    "nixl_cute_mapped_copy_warp_ptr": ("i32", ("i64", "i64", "i64")),
    "nixl_cute_mapped_copy_warp_ptr_readonly": (
        "i32",
        ("i64", "i64", "i64"),
    ),
    "nixl_cute_mapped_copy_warp": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i64"),
    ),
    "nixl_cute_mapped_copy_warp_readonly": (
        "i32",
        ("i64", "ptr", "i32", "i64", "i64"),
    ),
}
EXPORTED_SYMBOL_LIST = ",".join(EXPORTED_SIGNATURES)
_LLVM_TYPE_PATTERN = r"ptr(?:\s+addrspace\(\d+\))?|i\d+"
_MAPPED_COPY_REQUIRED_IR = {
    "nixl_cute_mapped_copy_warp_ptr": (
        "ld.global.L1::no_allocate.L2::256B.v4.s32",
        "st.global.L1::no_allocate.v4.s32",
        "llvm.nvvm.bar.warp.sync",
    ),
    "nixl_cute_mapped_copy_warp_ptr_readonly": (
        "ld.global.nc.L1::no_allocate.L2::256B.v4.s32",
        "st.global.L1::no_allocate.v4.s32",
        "llvm.nvvm.bar.warp.sync",
    ),
    "nixl_cute_mapped_copy_warp": (
        "ld.global.L1::no_allocate.L2::256B.v4.s32",
        "st.global.L1::no_allocate.v4.s32",
        "llvm.nvvm.bar.warp.sync",
    ),
    "nixl_cute_mapped_copy_warp_readonly": (
        "ld.global.nc.L1::no_allocate.L2::256B.v4.s32",
        "st.global.L1::no_allocate.v4.s32",
        "llvm.nvvm.bar.warp.sync",
    ),
}
_WARP_ACQUIRE_REQUIRED_IR = (
    "nixl_cute_wait_acquire_system_u64_warp",
    "nixl_cute_wait_acquire_system_u64_warp_until",
    "nixl_cute_wait_acquire_system_u64_warp_for",
    "nixl_cute_wait_acquire_system_u64_warp_for_or_abort",
    "nixl_cute_wait_acquire_gpu_u64_warp",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_warp",
    "nixl_cute_wait_acquire_system_u64_or_abort_warp",
    "nixl_cute_wait_acquire_system_u64_or_aborts_warp",
    "nixl_cute_wait_acquire_gpu_u64_warp_for",
)
_SCOPED_MEMORY_REQUIRED_IR = {
    "nixl_cute_load_acquire_system_u64": "ld.acquire.sys.global.u64",
    "nixl_cute_load_acquire_gpu_u64": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_until": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_until": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_thread": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_warp": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_thread": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_warp": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_system_u64_or_abort_thread": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_or_abort_warp": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_for": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_for": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_for_or_abort": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_for_or_abort": "ld.acquire.sys.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_thread_for": "ld.acquire.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_warp_for": "ld.acquire.gpu.global.u64",
    "nixl_cute_store_release_system_u64": "st.release.sys.global.u64",
    "nixl_cute_store_release_gpu_u64": "st.release.gpu.global.u64",
    "nixl_cute_atomic_add_release_gpu_u64": "atom.add.release.gpu.global.u64",
    "nixl_cute_atomic_max_release_gpu_u64": "atom.max.release.gpu.global.u64",
    "nixl_cute_atomic_max_release_system_u64": "atom.max.release.sys.global.u64",
    "nixl_cute_compare_exchange_status_gpu_i32": "atom.cas.release.gpu.global.b32",
}
_RELAXED_POLL_REQUIRED_IR = {
    "nixl_cute_wait_acquire_system_u64_thread": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_until": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_until": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_thread": "ld.relaxed.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_warp": "ld.relaxed.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_thread": "ld.relaxed.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_warp": "ld.relaxed.gpu.global.u64",
    "nixl_cute_wait_acquire_system_u64_or_abort_thread": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_or_abort_warp": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_for": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_for": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_thread_for_or_abort": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_system_u64_warp_for_or_abort": "ld.relaxed.sys.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_thread_for": "ld.relaxed.gpu.global.u64",
    "nixl_cute_wait_acquire_gpu_u64_warp_for": "ld.relaxed.gpu.global.u64",
}
_ABORT_OBSERVATION_REQUIRED_IR = {
    "nixl_cute_wait_acquire_system_u64_or_abort_thread": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
    "nixl_cute_wait_acquire_system_u64_or_abort_warp": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
    "nixl_cute_wait_acquire_system_u64_or_aborts_thread": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
    "nixl_cute_wait_acquire_system_u64_or_aborts_warp": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
    "nixl_cute_wait_acquire_system_u64_thread_for_or_abort": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
    "nixl_cute_wait_acquire_system_u64_warp_for_or_abort": (
        "ld.acquire.gpu.global.u64",
        "ld.relaxed.gpu.global.u64",
    ),
}
_TIMER_FREE_WAIT_SYMBOLS = (
    "nixl_cute_wait_acquire_system_u64_or_abort_thread",
    "nixl_cute_wait_acquire_system_u64_or_abort_warp",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_thread",
    "nixl_cute_wait_acquire_gpu_u64_or_abort_warp",
    "nixl_cute_wait_acquire_system_u64_or_aborts_thread",
    "nixl_cute_wait_acquire_system_u64_or_aborts_warp",
)
_DUAL_ABORT_REQUIRED_IR_COUNTS = {
    "nixl_cute_wait_acquire_system_u64_or_aborts_thread": (
        ("ld.acquire.sys.global.u64", 2),
        ("ld.relaxed.sys.global.u64", 2),
    ),
    "nixl_cute_wait_acquire_system_u64_or_aborts_warp": (
        ("ld.acquire.sys.global.u64", 2),
        ("ld.relaxed.sys.global.u64", 2),
    ),
}
_ATOMIC_RELEASE_REQUIRED_IR = (
    "nixl_cute_atomic_add_thread_wait",
    "nixl_cute_atomic_add_warp_wait",
    "nixl_cute_atomic_add_thread_post",
    "nixl_cute_atomic_add_warp_post",
)
_SYSTEM_RELEASE_FENCE_REQUIRED_IR = ("nixl_cute_fence_release_system",)
_PRODUCTION_FORBIDDEN_IR_SYMBOLS = ("__assertfail", "printf", "vprintf")
_PRODUCTION_ALLOWED_LLVM_CALLEES = frozenset(
    {
        "llvm.memcpy.p0.p0.i64",
        "llvm.nvvm.bar.warp.sync",
        "llvm.nvvm.barrier.sync",
        "llvm.nvvm.isspacep.local",
        "llvm.nvvm.read.ptx.sreg.ctaid.x",
        "llvm.nvvm.read.ptx.sreg.ctaid.y",
        "llvm.nvvm.read.ptx.sreg.ctaid.z",
        "llvm.nvvm.read.ptx.sreg.nctaid.x",
        "llvm.nvvm.read.ptx.sreg.nctaid.y",
        "llvm.nvvm.read.ptx.sreg.nctaid.z",
        "llvm.nvvm.read.ptx.sreg.tid.x",
        "llvm.nvvm.read.ptx.sreg.tid.y",
        "llvm.nvvm.read.ptx.sreg.tid.z",
        "llvm.nvvm.shfl.sync.idx.i32",
    }
)
_BUILDER_OWNED_MODE_MACROS = frozenset({"NDEBUG", "NIXL_CUTE_ENABLE_DEVICE_VALIDATION"})


def _device_mode_defines(device_validation: bool) -> tuple[str, ...]:
    if device_validation:
        return ("-DNIXL_CUTE_ENABLE_DEVICE_VALIDATION",)
    return ("-DNDEBUG",)


def _validate_extra_flags(extra_flags: list[str]) -> None:
    """Reject flags that can make the manifest disagree with generated code."""
    for flag in extra_flags:
        match = re.fullmatch(r"-[DU]\s*([A-Za-z_]\w*)(?:=.*)?", flag.strip())
        if match and match.group(1) in _BUILDER_OWNED_MODE_MACROS:
            raise RuntimeError(
                f"extra flag {flag!r} overrides builder-owned device mode macro "
                f"{match.group(1)}"
            )


def _validate_production_call_graph(text: str) -> None:
    """Require closed-world, helper-free production LLVM device IR."""
    declarations = set(
        re.findall(r"^declare\s+[^\n@]*@([^\s(]+)\(", text, re.MULTILINE)
    )
    forbidden_declarations = sorted(
        declaration
        for declaration in declarations
        if declaration not in _PRODUCTION_ALLOWED_LLVM_CALLEES
    )
    if forbidden_declarations:
        raise RuntimeError(
            "production device bitcode retains non-allowlisted declarations: "
            f"{forbidden_declarations}"
        )

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split(";", 1)[0]
        opcode = re.search(r"\b(?:callbr|invoke|call)\b", line)
        if opcode is None:
            continue
        instruction = line[opcode.end() :]
        if opcode.group(0) != "call":
            raise RuntimeError(
                "production device bitcode retains unsupported "
                f"{opcode.group(0)} at LLVM IR line {line_number}"
            )
        if re.search(r"^.*?\basm\b", instruction):
            continue
        # Return attributes such as ``range(i32 0, 1024)`` may precede the
        # callee. Select the direct @symbol call operand from the complete
        # instruction instead of treating the first parenthesis as the call.
        direct = re.findall(r"@([A-Za-z$._][A-Za-z0-9$._-]*)\s*\(", instruction)
        if len(direct) != 1:
            raise RuntimeError(
                "production device bitcode retains an indirect or unparseable "
                f"call at LLVM IR line {line_number}"
            )
        callee = direct[0]
        if callee not in _PRODUCTION_ALLOWED_LLVM_CALLEES:
            raise RuntimeError(
                "production device bitcode calls non-allowlisted symbol "
                f"{callee!r} at LLVM IR line {line_number}"
            )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _tool_major(tool: str) -> int:
    result = subprocess.run(
        [tool, "--version"], check=True, text=True, capture_output=True
    )
    match = re.search(r"\bversion\s+(\d+)", result.stdout + result.stderr)
    if not match:
        raise RuntimeError(f"could not determine LLVM version for {tool!r}")
    return int(match.group(1))


def _cuda_home(explicit: str | None) -> Path:
    if explicit:
        result = Path(explicit).resolve()
    elif os.environ.get("CUDA_HOME"):
        result = Path(os.environ["CUDA_HOME"]).resolve()
    else:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("CUDA_HOME is unset and nvcc is not on PATH")
        result = Path(nvcc).resolve().parent.parent
    if not (result / "include" / "cuda.h").is_file():
        raise RuntimeError(f"invalid CUDA toolkit path: {result}")
    return result


def _pkg_config_cflags(package: str) -> list[str]:
    pkg_config = shutil.which("pkg-config")
    if not pkg_config:
        return []
    result = subprocess.run(
        [pkg_config, "--cflags", package], text=True, capture_output=True
    )
    return shlex.split(result.stdout) if result.returncode == 0 else []


def clean_llvm_ir(text: str) -> str:
    """Remove module metadata and function alignment rejected by libNVVM."""
    lines = text.splitlines()
    rejected_ids = {
        match.group(1)
        for line in lines
        if (match := re.match(r"^(!\d+)\s*=.*nvvm-reflect-ftz", line))
    }

    cleaned: list[str] = []
    for line in lines:
        if any(line.startswith(f"{metadata_id} =") for metadata_id in rejected_ids):
            continue
        if line.startswith("!llvm.module.flags ="):
            for metadata_id in rejected_ids:
                line = re.sub(rf"\s*{re.escape(metadata_id)}(?!\d),?", "", line)
            line = re.sub(r"!\{\s*,?\s*", "!{", line)
            line = re.sub(r",\s*}", "}", line)
        if line.startswith("define "):
            line = re.sub(r"\s+align\s+\d+(?=\s|\{)", "", line)
        cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def _llvm_type(fragment: str) -> str:
    match = re.match(rf"\s*({_LLVM_TYPE_PATTERN})(?=\s|$)", fragment)
    if not match:
        raise RuntimeError(f"cannot parse LLVM parameter type from {fragment!r}")
    return re.sub(r"\s+", " ", match.group(1))


def _function_signature(text: str, name: str) -> tuple[str, tuple[str, ...], str]:
    match = re.search(
        rf"^define\s+(?P<prefix>[^\n@]+)@{re.escape(name)}"
        rf"\((?P<params>[^\n]*)\)[^\n]*\{{(?P<body>.*?)^\}}",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"device bitcode does not define required symbol {name}")
    prefix = match.group("prefix")
    if re.search(
        r"\b(?:private|internal|available_externally|linkonce(?:_odr)?|"
        r"weak(?:_odr)?|extern_weak)\b",
        prefix,
    ):
        raise RuntimeError(f"required device symbol {name} has non-public linkage")
    # LLVM adds target-specific conventions over time.  Fail closed on any
    # named ``*cc`` convention (except the explicit default ``ccc``), its
    # numeric spelling, and GPU target-family spellings that do not end in
    # ``cc``.  Enumerating today's names would silently accept future ABI
    # drift.
    calling_convention = re.search(
        r"\b(?:cc\s+\d+|(?!ccc\b)[A-Za-z_][A-Za-z0-9_]*cc|"
        r"(?:ptx|spir|amdgpu)_[A-Za-z0-9_]+)\b",
        prefix,
    )
    if calling_convention:
        raise RuntimeError(
            f"required device symbol {name} has non-C calling convention "
            f"{calling_convention.group(0)!r}"
        )
    returns = re.findall(rf"({_LLVM_TYPE_PATTERN})(?=\s|$)", prefix)
    if not returns:
        raise RuntimeError(f"cannot parse LLVM return type for {name}")
    return_type = re.sub(r"\s+", " ", returns[-1])
    raw_params = match.group("params").strip()
    params = (
        tuple(_llvm_type(fragment) for fragment in raw_params.split(","))
        if raw_params
        else ()
    )
    return return_type, params, match.group("body")


def _has_function_attribute(text: str, name: str, attribute: str) -> bool:
    header = re.search(
        rf"^define\s+[^\n@]*@{re.escape(name)}\([^\n]*\)[^\n]*\{{",
        text,
        flags=re.MULTILINE,
    )
    if not header:
        return False
    if re.search(rf"\b{re.escape(attribute)}\b", header.group(0)):
        return True
    for group in re.findall(r"#(\d+)", header.group(0)):
        definition = re.search(
            rf"^attributes\s+#{re.escape(group)}\s*=\s*\{{(?P<body>[^}}]*)\}}",
            text,
            flags=re.MULTILINE,
        )
        if definition and re.search(
            rf"\b{re.escape(attribute)}\b", definition.group("body")
        ):
            return True
    return False


def validate_llvm_ir(
    text: str,
    *,
    forceinline: bool | None = None,
    device_validation: bool = False,
) -> None:
    """Fail closed unless the optimized module has the exact public ABI."""
    defined = set(re.findall(r"^define\s+[^\n@]*@([^\s(]+)\(", text, re.MULTILINE))
    expected_names = set(EXPORTED_SIGNATURES)
    if defined != expected_names:
        raise RuntimeError(
            "unexpected NIXL CuTe exports or retained helper definitions: "
            f"expected {sorted(expected_names)}, found {sorted(defined)}"
        )

    bodies: dict[str, str] = {}
    for name, expected in EXPORTED_SIGNATURES.items():
        return_type, params, body = _function_signature(text, name)
        actual = (return_type, params)
        if actual != expected:
            raise RuntimeError(
                f"invalid LLVM signature for {name}: expected {expected}, found {actual}"
            )
        bodies[name] = body

    if forceinline is not None:
        required_attribute = "alwaysinline" if forceinline else "noinline"
        missing = [
            name
            for name in EXPORTED_SIGNATURES
            if not _has_function_attribute(text, name, required_attribute)
        ]
        if missing:
            raise RuntimeError(
                f"device exports lost required {required_attribute} attribute: {missing}"
            )

    if not device_validation:
        forbidden = [
            symbol
            for symbol in _PRODUCTION_FORBIDDEN_IR_SYMBOLS
            if re.search(rf"@{re.escape(symbol)}\b", text)
        ]
        if forbidden:
            raise RuntimeError(
                "production device bitcode references forbidden diagnostic "
                f"symbols: {forbidden}"
            )
        _validate_production_call_graph(text)

    if not re.search(
        rf"\bret\s+i32\s+{NIXL_CUTE_ABI_VERSION}\b",
        bodies["nixl_cute_abi_version"],
    ):
        raise RuntimeError(
            "nixl_cute_abi_version does not return the builder ABI version "
            f"{NIXL_CUTE_ABI_VERSION}"
        )

    for name, mapped_copy_instructions in _MAPPED_COPY_REQUIRED_IR.items():
        missing_ir = [
            token for token in mapped_copy_instructions if token not in bodies[name]
        ]
        if missing_ir:
            raise RuntimeError(
                f"{name} lost required production instructions: {missing_ir}"
            )

    for name in _WARP_ACQUIRE_REQUIRED_IR:
        if "llvm.nvvm.bar.warp.sync" not in bodies[name]:
            raise RuntimeError(
                f"{name} lost the warp barrier that propagates acquire ordering"
            )

    for name, required_instruction in _SCOPED_MEMORY_REQUIRED_IR.items():
        if required_instruction not in bodies[name]:
            raise RuntimeError(
                f"{name} lost required scoped operation {required_instruction}"
            )

    for name, required_instruction in _RELAXED_POLL_REQUIRED_IR.items():
        if required_instruction not in bodies[name]:
            raise RuntimeError(
                f"{name} lost relaxed failed-poll operation {required_instruction}"
            )

    for name, abort_observation_instructions in _ABORT_OBSERVATION_REQUIRED_IR.items():
        missing_ir = [
            token
            for token in abort_observation_instructions
            if token not in bodies[name]
        ]
        if missing_ir:
            raise RuntimeError(
                f"{name} lost same-GPU abort observation operations: {missing_ir}"
            )

    for name in _TIMER_FREE_WAIT_SYMBOLS:
        if "%globaltimer" in bodies[name]:
            raise RuntimeError(f"{name} unexpectedly reads %globaltimer")

    for name, requirements in _DUAL_ABORT_REQUIRED_IR_COUNTS.items():
        for token, minimum in requirements:
            if bodies[name].count(token) < minimum:
                raise RuntimeError(
                    f"{name} lost peer-abort observation: expected at least "
                    f"{minimum} instances of {token}"
                )

    for name in _ATOMIC_RELEASE_REQUIRED_IR:
        if "atom.add.release.sys.u64" not in bodies[name]:
            raise RuntimeError(f"{name} lost release ordering for CUDA-IPC publication")

    for name in _SYSTEM_RELEASE_FENCE_REQUIRED_IR:
        if "fence.release.sys" not in bodies[name]:
            raise RuntimeError(f"{name} lost its system-scope release fence")


def build(args: argparse.Namespace) -> None:
    _validate_extra_flags(args.extra_flag)
    if args.llvm_major != SUPPORTED_LLVM_MAJOR:
        raise RuntimeError(
            f"NIXL CuTe ABI {NIXL_CUTE_ABI_VERSION} requires LLVM "
            f"{SUPPORTED_LLVM_MAJOR}, not LLVM {args.llvm_major}"
        )
    tools = [args.clang, args.opt, args.llvm_dis, args.llvm_as]
    majors = {tool: _tool_major(tool) for tool in tools}
    mismatched = {
        tool: major for tool, major in majors.items() if major != args.llvm_major
    }
    if mismatched:
        values = ", ".join(f"{tool}={major}" for tool, major in mismatched.items())
        raise RuntimeError(
            f"CuTe device bitcode requires LLVM {args.llvm_major}; found {values}"
        )

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    depfile = Path(args.depfile).resolve()
    cuda_home = _cuda_home(args.cuda_home)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    depfile.parent.mkdir(parents=True, exist_ok=True)

    include_flags = [f"-I{Path(path).resolve()}" for path in args.include]
    if args.ucx_path:
        include_flags.append(f"-I{Path(args.ucx_path).resolve() / 'include'}")
    else:
        include_flags.extend(_pkg_config_cflags("ucx"))
    include_flags.extend(_pkg_config_cflags("doca-gpunetio"))
    include_flags.extend(
        [f"-I{cuda_home / 'include'}", f"-I{cuda_home / 'include' / 'cccl'}"]
    )

    forceinline = not any(
        re.fullmatch(r"-DNIXL_CUTE_DISABLE_FORCEINLINE(?:=.*)?", flag)
        for flag in args.extra_flag
    )

    with tempfile.TemporaryDirectory(
        prefix="nixl-device-bc-", dir=output.parent
    ) as tmp:
        tmp_dir = Path(tmp)
        unoptimized = tmp_dir / "libnixl_device.unoptimized.bc"
        optimized = tmp_dir / "libnixl_device.optimized.bc"
        llvm_ir = tmp_dir / "libnixl_device.ll"
        cleaned_ir = tmp_dir / "libnixl_device.cleaned.ll"
        final_bc = tmp_dir / output.name
        final_depfile = tmp_dir / depfile.name

        clang_command = [
            args.clang,
            "-std=gnu++17",
            "-x",
            "cuda",
            f"--cuda-path={cuda_home}",
            "--cuda-device-only",
            f"--cuda-gpu-arch={args.cuda_arch}",
            "-Wno-unknown-cuda-version",
            # CUDA 13.2+ leaves this specifier undefined under Clang 20.
            # This matches the upstream NCCL/NVSHMEM LLVM-bitcode workaround.
            "-D_NV_RSQRT_SPECIFIER=",
            "-c",
            "-emit-llvm",
            "-O1",
            "-MMD",
            "-MF",
            str(final_depfile),
            "-MQ",
            str(output),
            *include_flags,
            *args.extra_flag,
            *_device_mode_defines(args.device_validation),
            str(source),
            "-o",
            str(unoptimized),
        ]
        _run(clang_command)
        _run(
            [
                args.opt,
                "--passes=internalize,inline,globaldce",
                f"-internalize-public-api-list={EXPORTED_SYMBOL_LIST}",
                str(unoptimized),
                "-o",
                str(optimized),
            ]
        )
        _run([args.llvm_dis, str(optimized), "-o", str(llvm_ir)])
        optimized_text = llvm_ir.read_text()
        validate_llvm_ir(
            optimized_text,
            forceinline=forceinline,
            device_validation=args.device_validation,
        )
        cleaned_ir.write_text(clean_llvm_ir(optimized_text))
        _run([args.llvm_as, str(cleaned_ir), "-o", str(final_bc)])

        digest = hashlib.sha256(final_bc.read_bytes()).hexdigest()
        manifest_data = {
            "abi_version": NIXL_CUTE_ABI_VERSION,
            "bitcode": output.name,
            "sha256": digest,
            "nixl_version": args.nixl_version,
            "llvm_major": args.llvm_major,
            "cuda_arch": args.cuda_arch,
            "device_validation": args.device_validation,
            "forceinline": forceinline,
        }
        manifest_tmp = tmp_dir / manifest.name
        manifest_tmp.write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n"
        )
        os.replace(final_bc, output)
        os.replace(manifest_tmp, manifest)
        os.replace(final_depfile, depfile)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--depfile", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--llvm-dis", required=True)
    parser.add_argument("--llvm-as", required=True)
    parser.add_argument("--cuda-home")
    parser.add_argument("--ucx-path")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--cuda-arch", default="sm_90")
    parser.add_argument("--llvm-major", type=int, default=SUPPORTED_LLVM_MAJOR)
    parser.add_argument("--nixl-version", required=True)
    parser.add_argument(
        "--device-validation",
        action="store_true",
        help="retain runtime argument/warp-shape checks in device hot paths",
    )
    return parser


if __name__ == "__main__":
    build(_parser().parse_args())
