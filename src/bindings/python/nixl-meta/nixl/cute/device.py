# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-side CuTe/NIXL capability checks and placeholder operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeviceApiStatus:
    """Result of probing whether NIXL's device API headers are usable."""

    available: bool
    nixl_device_header: str | None
    ucx_device_header: str | None
    gdaki_header: str | None
    missing: tuple[str, ...]

    @property
    def reason(self) -> str:
        if self.available:
            return "NIXL device API headers were found."
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
    for env_name in ("NIXL_DEVICE_INCLUDE", "NIXL_UCX_INCLUDE", "UCX_HOME", "NIXL_HOME"):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        roots.append(root / "include" if env_name in {"UCX_HOME", "NIXL_HOME"} else root)
    roots.extend(
        [
            Path("/opt/hpcx/ucx/include"),
            Path("/usr/local/include"),
            Path("/usr/include"),
        ]
    )
    return roots


def device_api_status() -> DeviceApiStatus:
    """Probe for the headers needed by ``nixl_device.cuh``.

    This does not prove that a CuTe kernel can call NIXL device functions.  It
    only checks the known header prerequisites so examples can fail clearly.
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
                nixl_home / "include" / "nixl" / "api" / "gpu" / "ucx" / "nixl_device.cuh",
            ]
        )
    nixl_candidates.append(_repo_nixl_device_candidate())

    include_roots = _include_roots()
    ucx_candidates = [root / "ucp" / "api" / "device" / "ucp_device_impl.h" for root in include_roots]
    gdaki_candidates = [root / "uct" / "ib" / "mlx5" / "gdaki" / "gdaki.cuh" for root in include_roots]

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

    return DeviceApiStatus(
        available=not missing,
        nixl_device_header=nixl_header,
        ucx_device_header=ucx_header,
        gdaki_header=gdaki_header,
        missing=tuple(missing),
    )


def is_device_api_available() -> bool:
    """Return whether the known NIXL device API header prerequisites exist."""

    return device_api_status().available


def _not_implemented(name: str) -> None:
    status = device_api_status()
    raise NotImplementedError(
        f"nixl.cute.device.{name} is not implemented yet. "
        "The first CuTe prototype only validates host/runtime packaging and "
        f"CuTe tensor interop. Device API status: {status.reason}"
    )


def put_block(*args, **kwargs):
    """Placeholder for a future CTA-level NIXL device WRITE."""

    _not_implemented("put_block")


def atomic_add(*args, **kwargs):
    """Placeholder for a future NIXL device remote atomic add."""

    _not_implemented("atomic_add")


def get_ptr(*args, **kwargs):
    """Placeholder for a future ``nixlGetPtr`` CuTe wrapper."""

    _not_implemented("get_ptr")

