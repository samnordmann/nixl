# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate and authenticate the NIXL CuTe DSL device bitcode."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

NIXL_CUTE_ABI_VERSION = 3
NIXL_CUTE_LLVM_MAJOR = 20
_BITCODE_NAME = "libnixl_device.bc"
_MANIFEST_NAME = "nixl_device_abi.json"
_BITCODE_ARCH_PREFIX = "libnixl_device_"
_MANIFEST_ARCH_PREFIX = "nixl_device_abi_"
_OVERRIDE_ENV = "NIXL_DEVICE_BITCODE"
_ALLOW_OVERRIDE_ENV = "NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class BitcodeNotFoundError(FileNotFoundError):
    """Raised when no NIXL device bitcode can be found."""


class BitcodeVerificationError(RuntimeError):
    """Raised when device bitcode has no valid adjacent ABI manifest."""


class UnsupportedArchitectureError(BitcodeVerificationError):
    """Raised when no authenticated bitcode matches the active CUDA device."""


def _allow_override() -> bool:
    value = os.environ.get(_ALLOW_OVERRIDE_ENV, "")
    return value.strip().lower() in _TRUE_VALUES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BitcodeVerificationError(
            f"cannot read NIXL device ABI manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {path} must contain a JSON object"
        )
    return value


def _normalize_arch(value: object) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        major, minor = value
        if (
            isinstance(major, bool)
            or isinstance(minor, bool)
            or not isinstance(major, int)
            or not isinstance(minor, int)
            or major < 0
            or minor < 0
        ):
            raise ValueError(f"invalid CUDA compute capability {value!r}")
        return f"sm_{major}{minor}"

    if not isinstance(value, str):
        raise ValueError(f"invalid CUDA architecture {value!r}")
    arch = value.strip().lower()
    if arch.startswith("compute_"):
        arch = arch[8:]
    elif arch.startswith("sm_"):
        arch = arch[3:]
    elif arch.startswith("sm"):
        arch = arch[2:]
    arch = arch.replace(".", "")
    if not arch.isdigit():
        raise ValueError(f"invalid CUDA architecture {value!r}")
    return f"sm_{int(arch)}"


def _current_device_arch() -> str:
    """Return the rank-local current CUDA device architecture."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        return _normalize_arch(capability)
    except Exception as exc:
        raise UnsupportedArchitectureError(
            "cannot select NIXL CuTe bitcode because the current CUDA device "
            f"architecture is unavailable: {exc}"
        ) from exc


def _manifest_path(path: Path) -> Path:
    name = path.name
    if name.startswith(_BITCODE_ARCH_PREFIX) and name.endswith(".bc"):
        arch = name[len(_BITCODE_ARCH_PREFIX) : -3]
        return path.with_name(f"{_MANIFEST_ARCH_PREFIX}{arch}.json")
    return path.with_name(_MANIFEST_NAME)


def _verify_bitcode(
    path: Path,
    *,
    expected_arch: str,
    require_production: bool,
) -> Path:
    """Verify one bitcode/manifest pair for ``expected_arch``."""
    manifest_path = _manifest_path(path)
    manifest = _load_manifest(manifest_path)
    manifest_abi = manifest.get("abi_version")
    if type(manifest_abi) is not int or manifest_abi != NIXL_CUTE_ABI_VERSION:
        raise BitcodeVerificationError(
            f"NIXL device ABI {manifest_abi!r} in "
            f"{manifest_path} is incompatible with Python ABI "
            f"{NIXL_CUTE_ABI_VERSION}"
        )
    if manifest.get("bitcode") != path.name:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} names "
            f"{manifest.get('bitcode')!r}, expected {path.name!r}"
        )
    manifest_llvm = manifest.get("llvm_major")
    if type(manifest_llvm) is not int or manifest_llvm != NIXL_CUTE_LLVM_MAJOR:
        raise BitcodeVerificationError(
            f"NIXL device bitcode uses LLVM {manifest_llvm!r}; CuTe ABI "
            f"{NIXL_CUTE_ABI_VERSION} requires LLVM {NIXL_CUTE_LLVM_MAJOR}"
        )
    if type(manifest.get("device_validation")) is not bool:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} must record a boolean "
            "device_validation mode"
        )
    if type(manifest.get("forceinline")) is not bool:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} must record a boolean "
            "forceinline mode"
        )
    try:
        manifest_arch = _normalize_arch(manifest.get("cuda_arch"))
    except ValueError as exc:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} has an invalid cuda_arch"
        ) from exc
    if manifest_arch != expected_arch:
        raise UnsupportedArchitectureError(
            f"NIXL device bitcode {path} targets {manifest_arch}, but the current "
            f"CUDA device requires {expected_arch}"
        )
    if require_production and manifest["forceinline"] is not True:
        raise BitcodeVerificationError(
            f"packaged NIXL device bitcode {path} is not a force-inline production build"
        )
    if require_production and manifest["device_validation"] is not False:
        raise BitcodeVerificationError(
            f"packaged NIXL device bitcode {path} retains debug validation branches"
        )
    expected = manifest.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} has an invalid sha256"
        )
    try:
        valid_digest = int(expected, 16) >= 0
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise BitcodeVerificationError(
            f"NIXL device ABI manifest {manifest_path} has an invalid sha256"
        )
    actual = _sha256(path)
    if actual != expected.lower():
        raise BitcodeVerificationError(
            f"NIXL device bitcode hash mismatch for {path}: expected "
            f"{expected.lower()}, got {actual}"
        )
    return path.resolve()


def _existing_candidate(
    path: Path,
    *,
    expected_arch: str,
    require_production: bool,
    explicit: bool = False,
) -> Path | None:
    if path.is_file():
        return _verify_bitcode(
            path,
            expected_arch=expected_arch,
            require_production=require_production,
        )
    if explicit:
        raise BitcodeNotFoundError(
            f"{_OVERRIDE_ENV} points to missing device bitcode: {path}"
        )
    return None


def _packaged_artifact_path(directory: Path, arch: str) -> Path:
    return directory / f"{_BITCODE_ARCH_PREFIX}{arch}.bc"


def _available_packaged_arches(directory: Path) -> tuple[str, ...]:
    arches: set[str] = set()
    for path in directory.glob(f"{_BITCODE_ARCH_PREFIX}sm_*.bc"):
        value = path.name[len(_BITCODE_ARCH_PREFIX) : -3]
        try:
            arches.add(_normalize_arch(value))
        except ValueError:
            continue
    return tuple(sorted(arches, key=lambda value: int(value[3:])))


def device_bitcode_path() -> str:
    """Return authenticated bitcode for the rank-local current CUDA device.

    Production selects an exact per-SM artifact installed adjacent to this
    module. A legacy generic artifact remains valid only when its manifest
    targets the active SM. ``NIXL_DEVICE_BITCODE`` is a development escape hatch
    and additionally requires ``NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE=1`` because a
    manifest cannot prove compatibility with a separately built host UCX
    runtime. That explicit path may use debug validation; packaged artifacts may
    not.
    """
    arch = _current_device_arch()
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        if not _allow_override():
            raise BitcodeVerificationError(
                f"{_OVERRIDE_ENV} is development-only; also set "
                f"{_ALLOW_OVERRIDE_ENV}=1 to acknowledge host/device ABI risk"
            )
        return str(
            _existing_candidate(
                Path(override).expanduser(),
                expected_arch=arch,
                require_production=False,
                explicit=True,
            )
        )

    package_dir = Path(__file__).parent
    packaged = _existing_candidate(
        _packaged_artifact_path(package_dir, arch),
        expected_arch=arch,
        require_production=True,
    )
    if packaged is not None:
        return str(packaged)

    # Safe compatibility for explicit single-SM source installations. Never use
    # this as a fallback after an exact per-SM file failed authentication.
    generic = _existing_candidate(
        package_dir / _BITCODE_NAME,
        expected_arch=arch,
        require_production=True,
    )
    if generic is not None:
        return str(generic)

    available = _available_packaged_arches(package_dir)
    if available:
        raise UnsupportedArchitectureError(
            f"NIXL CuTe bitcode does not support current device {arch}; packaged "
            f"architectures: {', '.join(available)}"
        )
    raise BitcodeNotFoundError(
        f"cannot find packaged NIXL CuTe bitcode for {arch}; install a NIXL "
        "wheel built with CuTe support or use the explicitly unsafe development "
        "override"
    )


# Kept private as construction imports CuTe DSL. Tests and packaging tools can
# exercise device_bitcode_path() without importing that optional dependency.
_device_bitcode_path = device_bitcode_path


__all__ = [
    "BitcodeNotFoundError",
    "BitcodeVerificationError",
    "NIXL_CUTE_ABI_VERSION",
    "NIXL_CUTE_LLVM_MAJOR",
    "UnsupportedArchitectureError",
    "device_bitcode_path",
]
