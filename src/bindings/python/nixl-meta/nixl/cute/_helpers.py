# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for a future bitcode-backed CuTe/NIXL device binding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class BitcodeStatus:
    """Result of probing for the future ``libnixl_device.bc`` artifact."""

    available: bool
    path: str | None
    missing: tuple[str, ...]

    @property
    def reason(self) -> str:
        if self.available:
            return f"Found libnixl_device.bc at {self.path}"
        return "Missing: " + ", ".join(self.missing)


def _repo_root() -> Path:
    # _helpers.py -> cute -> nixl -> nixl-meta -> python -> bindings -> src
    return Path(__file__).resolve().parents[5]


def _existing_file(candidates: list[Path]) -> str | None:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _bitcode_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("NIXL_DEVICE_BITCODE"):
        candidates.append(Path(os.environ["NIXL_DEVICE_BITCODE"]))
    if os.environ.get("NIXL_HOME"):
        nixl_home = Path(os.environ["NIXL_HOME"])
        candidates.extend(
            [
                nixl_home / "lib" / "libnixl_device.bc",
                nixl_home / "lib64" / "libnixl_device.bc",
            ]
        )
    candidates.extend(
        [
            _repo_root() / "bindings" / "ir" / "libnixl_device.bc",
            _repo_root() / "api" / "gpu" / "ucx" / "libnixl_device.bc",
            Path("/usr/local/nixl/lib/libnixl_device.bc"),
            Path("/usr/local/lib/libnixl_device.bc"),
        ]
    )
    return candidates


def _pathfinder_bitcode() -> str | None:
    try:
        from cuda.pathfinder import find_bitcode_lib
    except Exception:
        return None
    try:
        return find_bitcode_lib("nixl_device")
    except Exception:
        return None


def device_bitcode_status() -> BitcodeStatus:
    """Return whether the future NIXL device bitcode library is discoverable."""

    path = _existing_file(_bitcode_candidates()) or _pathfinder_bitcode()
    if path is not None:
        return BitcodeStatus(available=True, path=path, missing=())
    return BitcodeStatus(
        available=False,
        path=None,
        missing=(
            "libnixl_device.bc",
            "set NIXL_DEVICE_BITCODE or install a package discoverable by cuda.pathfinder",
        ),
    )


def device_bitcode_path() -> str:
    """Return the discovered bitcode path or raise a clear error."""

    status = device_bitcode_status()
    if status.path is None:
        raise FileNotFoundError(status.reason)
    return status.path


@lru_cache(maxsize=1)
def _bitcode_source():
    from cutlass.cute import BitCode

    return BitCode(device_bitcode_path())


def ffi(**kwargs):
    """Create a ``cute.ffi`` prototype using ``libnixl_device.bc``.

    This mirrors the NCCL CuTe binding shape, but remains unused until NIXL
    provides the C wrapper symbols and packaged device bitcode.
    """

    import cutlass.cute as cute

    return cute.ffi(source=_bitcode_source(), **kwargs)
