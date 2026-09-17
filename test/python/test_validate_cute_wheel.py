# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

tomllib: ModuleType | None
try:
    import tomllib as _tomllib

    tomllib = _tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

_ROOT = Path(__file__).parents[2]
_SOURCE = _ROOT / "contrib" / "validate_cute_wheel.py"
_SPEC = importlib.util.spec_from_file_location("validate_cute_wheel", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_PACKAGE_DIR = "nixl_cu12/device/cute"


def _wheel_entries(arches=("sm_90", "sm_100")):
    entries: dict[str, bytes] = {
        "nixl_cu12-1.5.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\n" b"Name: nixl-cu12\n" b"Version: 1.5.0\n\n"
        )
    }
    for arch in arches:
        bitcode_name = f"libnixl_device_{arch}.bc"
        bitcode = f"bitcode-{arch}".encode()
        manifest = {
            "abi_version": 3,
            "llvm_major": 20,
            "cuda_arch": arch,
            "bitcode": bitcode_name,
            "sha256": hashlib.sha256(bitcode).hexdigest(),
            "device_validation": False,
            "forceinline": True,
            "nixl_version": "1.5.0",
        }
        entries[f"{_PACKAGE_DIR}/{bitcode_name}"] = bitcode
        entries[f"{_PACKAGE_DIR}/nixl_device_abi_{arch}.json"] = json.dumps(
            manifest
        ).encode()
    return entries


def _write_wheel(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    wheel = tmp_path / "nixl.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return wheel


def _update_manifest(entries, arch, **updates):
    name = f"{_PACKAGE_DIR}/nixl_device_abi_{arch}.json"
    manifest = json.loads(entries[name])
    manifest.update(updates)
    entries[name] = json.dumps(manifest).encode()


def test_validate_complete_production_cute_wheel(tmp_path):
    wheel = _write_wheel(tmp_path, _wheel_entries())

    _MODULE.validate_cute_wheel(wheel, _PACKAGE_DIR, ("90", "sm_100"))


@pytest.mark.skipif(tomllib is None, reason="tomllib is built in on Python 3.11+")
def test_cute_optional_dependency_matches_qualified_release():
    assert tomllib is not None
    backend = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    meta_source = (
        _ROOT / "src" / "bindings" / "python" / "nixl-meta" / "pyproject.toml.in"
    ).read_text()
    meta = tomllib.loads(
        meta_source.replace("@VERSION@", "1.5.0").replace(
            "@WHEEL_DEPS@", '"nixl-cu12==1.5.0"'
        )
    )

    expected = ["nvidia-cutlass-dsl==4.5.1", "cuda-python>=12.8,<13"]
    assert backend["project"]["optional-dependencies"]["cute"] == expected
    assert "cute" not in meta["project"]["optional-dependencies"]
    assert meta["project"]["optional-dependencies"]["cute-cu12"] == expected


@pytest.mark.skipif(tomllib is None, reason="tomllib is built in on Python 3.11+")
def test_tomlutil_removes_cuda13_cute_extra(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """\
[project]
name = "nixl-cu12"
[project.optional-dependencies]
cute = ["nvidia-cutlass-dsl==4.5.1", "cuda-python>=12.8,<13"]
other = ["example"]
"""
    )

    subprocess.run(
        [
            sys.executable,
            str(_ROOT / "contrib" / "tomlutil.py"),
            "--wheel-name",
            "nixl-cu13",
            "--remove-extra",
            "cute",
            str(project),
        ],
        check=True,
    )
    parsed = tomllib.loads(project.read_text())
    assert parsed["project"]["name"] == "nixl-cu13"
    assert "cute" not in parsed["project"]["optional-dependencies"]
    assert parsed["project"]["optional-dependencies"]["other"] == ["example"]


def test_meson_preserves_cute_device_target_contract():
    meson = (_ROOT / "src" / "api" / "gpu" / "ucx" / "cute" / "meson.build").read_text()

    assert "bitcode_output = 'libnixl_device.bc'" in meson
    assert "manifest_output = 'nixl_device_abi.json'" in meson
    assert "bitcode_target_name = 'nixl_cute_device_bitcode'" in meson
    assert "custom_target(\n      bitcode_target_name," in meson
    assert "alias_target('nixl_cute_device_bitcode', cute_device_artifacts)" in meson


def test_release_wheel_requests_and_validates_multi_sm_artifacts():
    script = (_ROOT / "contrib" / "build-wheel.sh").read_text()

    assert 'NIXL_CUDA_ARCHS="90,100,103,120"' in script
    assert 'CUTE_BITCODE_ARCHS="90,100,120"' in script
    assert 'NIXL_CUDA_ARCHS="80,86,89,90,100,103,120"' in script
    assert 'CUTE_BITCODE_ARCHS="80,86,89,90,100,120"' in script
    assert "-Dnixl_cuda_arch_list=${NIXL_CUDA_ARCHS}" in script
    assert "-Dcute_bitcode_arch=${CUTE_BITCODE_ARCHS}" in script
    assert "TOML_ARGS+=(--remove-extra cute)" in script
    assert "validate_cute_wheel.py" in script
    assert '--architectures "$CUTE_BITCODE_ARCHS"' in script
    assert 'cp -p -- pyproject.toml "$PYPROJECT_BACKUP"' in script
    assert "trap restore_source_and_cleanup EXIT" in script
    assert 'cmp -s -- "$PYPROJECT_BACKUP" pyproject.toml' in script
    assert "EXIT restores pyproject.toml byte-for-byte" in script


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing or duplicated"),
        ("unexpected", "unexpected"),
        ("generic", "unsafe generic"),
        ("wrong_hash", "digest does not match"),
        ("wrong_arch", "expected 'sm_90'"),
        ("wrong_abi", "expected 3"),
        ("wrong_llvm", "expected 20"),
        ("validation", "debug validation branches"),
        ("no_forceinline", "not a force-inline production build"),
        ("wrong_nixl_version", "wheel METADATA records"),
        ("missing_metadata", "exactly one dist-info/METADATA"),
    ],
)
def test_validate_cute_wheel_fails_closed(tmp_path, mutation, message):
    entries = _wheel_entries()
    if mutation == "missing":
        entries.pop(f"{_PACKAGE_DIR}/libnixl_device_sm_100.bc")
    elif mutation == "unexpected":
        entries.update(_wheel_entries(("sm_103",)))
    elif mutation == "generic":
        entries[f"{_PACKAGE_DIR}/libnixl_device.bc"] = b"legacy"
    elif mutation == "wrong_hash":
        _update_manifest(entries, "sm_90", sha256="0" * 64)
    elif mutation == "wrong_arch":
        _update_manifest(entries, "sm_90", cuda_arch="sm_100")
    elif mutation == "wrong_abi":
        _update_manifest(entries, "sm_90", abi_version=True)
    elif mutation == "wrong_llvm":
        _update_manifest(entries, "sm_90", llvm_major=19)
    elif mutation == "validation":
        _update_manifest(entries, "sm_90", device_validation=True)
    elif mutation == "no_forceinline":
        _update_manifest(entries, "sm_90", forceinline=False)
    elif mutation == "wrong_nixl_version":
        _update_manifest(entries, "sm_90", nixl_version="0.0.0")
    elif mutation == "missing_metadata":
        entries.pop("nixl_cu12-1.5.0.dist-info/METADATA")
    wheel = _write_wheel(tmp_path, entries)

    with pytest.raises(_MODULE.WheelValidationError, match=message):
        _MODULE.validate_cute_wheel(wheel, _PACKAGE_DIR, ("90", "100"))
