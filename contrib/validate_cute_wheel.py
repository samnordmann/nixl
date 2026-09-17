#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed unless a wheel has the complete production CuTe artifact matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Iterable

NIXL_CUTE_ABI_VERSION = 3
NIXL_CUTE_LLVM_MAJOR = 20


class WheelValidationError(RuntimeError):
    """Raised when a release wheel contains an unsafe CuTe artifact set."""


def _normalize_arches(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        arch = value.strip().lower()
        if arch.startswith("sm_"):
            arch = arch[3:]
        if not arch.isdigit():
            raise WheelValidationError(f"invalid expected CUDA architecture {value!r}")
        normalized = f"sm_{int(arch)}"
        if normalized in result:
            raise WheelValidationError(
                f"duplicate expected CUDA architecture {normalized}"
            )
        result.append(normalized)
    if not result:
        raise WheelValidationError("at least one CUDA architecture is required")
    return tuple(result)


def validate_cute_wheel(
    wheel: str | Path,
    package_dir: str,
    architectures: Iterable[str],
) -> None:
    """Validate every per-SM bitcode/manifest pair in ``wheel``."""
    wheel = Path(wheel)
    expected_arches = _normalize_arches(architectures)
    expected_paths = {
        f"{package_dir}/{prefix}_{arch}.{extension}"
        for arch in expected_arches
        for prefix, extension in (
            ("libnixl_device", "bc"),
            ("nixl_device_abi", "json"),
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_paths = [
            path for path in names if path.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise WheelValidationError(
                f"{wheel} must contain exactly one dist-info/METADATA file; "
                f"found {metadata_paths}"
            )
        metadata = BytesParser(policy=compat32).parsebytes(
            archive.read(metadata_paths[0])
        )
        wheel_version = metadata.get("Version")
        if not wheel_version:
            raise WheelValidationError(
                f"{wheel} METADATA does not contain a non-empty Version"
            )
        malformed = sorted(path for path in expected_paths if names.count(path) != 1)
        actual_paths = {
            path
            for path in names
            if (
                path.startswith(f"{package_dir}/libnixl_device_")
                and path.endswith(".bc")
            )
            or (
                path.startswith(f"{package_dir}/nixl_device_abi_")
                and path.endswith(".json")
            )
        }
        unexpected = sorted(actual_paths - expected_paths)
        legacy = sorted(
            path
            for path in (
                f"{package_dir}/libnixl_device.bc",
                f"{package_dir}/nixl_device_abi.json",
            )
            if path in names
        )
        if malformed or unexpected or legacy:
            raise WheelValidationError(
                f"{wheel} has an invalid per-SM CuTe artifact set; "
                f"missing or duplicated={malformed}, unexpected={unexpected}, "
                f"unsafe generic={legacy}"
            )

        for arch in expected_arches:
            bitcode_name = f"libnixl_device_{arch}.bc"
            bitcode_path = f"{package_dir}/{bitcode_name}"
            manifest_path = f"{package_dir}/nixl_device_abi_{arch}.json"
            bitcode = archive.read(bitcode_path)
            try:
                manifest = json.loads(archive.read(manifest_path))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WheelValidationError(
                    f"{wheel} has invalid CuTe manifest {manifest_path}: {exc}"
                ) from exc

            if not isinstance(manifest, dict):
                raise WheelValidationError(
                    f"{wheel} CuTe manifest {manifest_path} is not a JSON object"
                )
            if not bitcode:
                raise WheelValidationError(
                    f"{wheel} contains empty CuTe bitcode {bitcode_path}"
                )
            if (
                type(manifest.get("abi_version")) is not int
                or manifest["abi_version"] != NIXL_CUTE_ABI_VERSION
            ):
                raise WheelValidationError(
                    f"{wheel} has CuTe ABI {manifest.get('abi_version')!r} "
                    f"for {arch}, expected {NIXL_CUTE_ABI_VERSION}"
                )
            if (
                type(manifest.get("llvm_major")) is not int
                or manifest["llvm_major"] != NIXL_CUTE_LLVM_MAJOR
            ):
                raise WheelValidationError(
                    f"{wheel} has CuTe bitcode from LLVM "
                    f"{manifest.get('llvm_major')!r} for {arch}, "
                    f"expected {NIXL_CUTE_LLVM_MAJOR}"
                )
            if manifest.get("cuda_arch") != arch:
                raise WheelValidationError(
                    f"{wheel} CuTe manifest {manifest_path} records "
                    f"{manifest.get('cuda_arch')!r}, expected {arch!r}"
                )
            if manifest.get("nixl_version") != wheel_version:
                raise WheelValidationError(
                    f"{wheel} CuTe manifest {manifest_path} records NIXL version "
                    f"{manifest.get('nixl_version')!r}, but wheel METADATA records "
                    f"{wheel_version!r}"
                )
            if manifest.get("bitcode") != bitcode_name:
                raise WheelValidationError(
                    f"{wheel} CuTe manifest {manifest_path} names an invalid artifact"
                )
            if manifest.get("forceinline") is not True:
                raise WheelValidationError(
                    f"{wheel} CuTe bitcode {arch} is not a force-inline production build"
                )
            if manifest.get("device_validation") is not False:
                raise WheelValidationError(
                    f"{wheel} CuTe bitcode {arch} retains debug validation branches"
                )
            actual = hashlib.sha256(bitcode).hexdigest()
            if manifest.get("sha256") != actual:
                raise WheelValidationError(
                    f"{wheel} CuTe bitcode digest does not match {manifest_path}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument(
        "--architectures",
        required=True,
        help="Comma-separated CUDA SM values, for example 90,100 or sm_90,sm_100",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        validate_cute_wheel(
            args.wheel,
            args.package_dir,
            args.architectures.split(","),
        )
    except (OSError, zipfile.BadZipFile, WheelValidationError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
