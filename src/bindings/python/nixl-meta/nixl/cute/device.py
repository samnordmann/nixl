# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing device-side CuTe/NIXL scaffold."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import _bindings
from ._helpers import BitcodeStatus, device_bitcode_status


@dataclass(frozen=True)
class DeviceApiStatus:
    """Result of probing the current NIXL CuTe device binding state."""

    available: bool
    headers_available: bool
    bitcode: BitcodeStatus
    cute_available: bool
    bindings_available: bool
    nixl_device_header: str | None
    ucx_device_header: str | None
    gdaki_header: str | None
    missing: tuple[str, ...]

    @property
    def reason(self) -> str:
        if self.available:
            return "NIXL CuTe device binding is available."
        return "Missing: " + ", ".join(self.missing)


def _existing_file(candidates: list[Path]) -> str | None:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _repo_nixl_device_candidate() -> Path:
    # device.py -> cute -> nixl -> nixl-meta -> python -> bindings -> src
    return (
        Path(__file__).resolve().parents[5]
        / "api"
        / "gpu"
        / "ucx"
        / "nixl_device.cuh"
    )


def _include_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in (
        "NIXL_DEVICE_INCLUDE",
        "NIXL_UCX_INCLUDE",
        "UCX_HOME",
        "NIXL_HOME",
    ):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        roots.append(
            root / "include" if env_name in {"UCX_HOME", "NIXL_HOME"} else root
        )
    roots.extend(
        [
            Path("/opt/hpcx/ucx/include"),
            Path("/usr/local/include"),
            Path("/usr/include"),
        ]
    )
    return roots


def device_api_status() -> DeviceApiStatus:
    """Probe the layers needed by a real CuTe/NIXL device binding.

    The current CUDA-DL base image is expected to have host-side NIXL pieces but
    not the full device-side stack.  In that environment this returns a clear
    unavailable status rather than failing during import.
    """

    nixl_candidates = [
        Path(os.environ["NIXL_DEVICE_HEADER"])
        for _ in [None]
        if os.environ.get("NIXL_DEVICE_HEADER")
    ]
    if os.environ.get("NIXL_HOME"):
        nixl_home = Path(os.environ["NIXL_HOME"])
        nixl_candidates.extend(
            [
                nixl_home / "include" / "nixl_device.cuh",
                nixl_home / "include" / "nixl" / "nixl_device.cuh",
                nixl_home
                / "include"
                / "nixl"
                / "api"
                / "gpu"
                / "ucx"
                / "nixl_device.cuh",
            ]
        )
    nixl_candidates.append(_repo_nixl_device_candidate())

    include_roots = _include_roots()
    ucx_candidates = [
        root / "ucp" / "api" / "device" / "ucp_device_impl.h" for root in include_roots
    ]
    gdaki_candidates = [
        root / "uct" / "ib" / "mlx5" / "gdaki" / "gdaki.cuh" for root in include_roots
    ]

    nixl_header = _existing_file(nixl_candidates)
    ucx_header = _existing_file(ucx_candidates)
    gdaki_header = _existing_file(gdaki_candidates)

    missing = []
    if nixl_header is None:
        missing.append("nixl_device.cuh")
    if ucx_header is None:
        missing.append("ucp/api/device/ucp_device_impl.h")
    if gdaki_header is None:
        missing.append("uct/ib/mlx5/gdaki/gdaki.cuh")

    bitcode = device_bitcode_status()
    if not bitcode.available:
        missing.append("libnixl_device.bc")

    try:
        import cutlass.cute  # noqa: F401

        cute_available = True
    except Exception:
        cute_available = False
        missing.append("nvidia-cutlass-dsl")

    bindings_available = _bindings.has_device_bindings()
    if not bindings_available:
        missing.append("generated cute.ffi bindings")

    headers_available = (
        nixl_header is not None and ucx_header is not None and gdaki_header is not None
    )

    return DeviceApiStatus(
        available=not missing,
        headers_available=headers_available,
        bitcode=bitcode,
        cute_available=cute_available,
        bindings_available=bindings_available,
        nixl_device_header=nixl_header,
        ucx_device_header=ucx_header,
        gdaki_header=gdaki_header,
        missing=tuple(missing),
    )


def is_device_api_available() -> bool:
    """Return whether the bitcode-backed NIXL CuTe device binding exists."""

    return device_api_status().available


def put_block(*args, **kwargs):
    """Future CTA/block-level NIXL device WRITE/PUT."""

    return _bindings.put_block(*args, **kwargs)


def atomic_add(*args, **kwargs):
    """Placeholder for a future NIXL device remote atomic add."""

    return _bindings.atomic_add(*args, **kwargs)


def get_ptr(*args, **kwargs):
    """Placeholder for a future ``nixlGetPtr`` CuTe wrapper."""

    return _bindings.get_ptr(*args, **kwargs)

