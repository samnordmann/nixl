# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

_PYTHON_SOURCE = Path(__file__).parents[2] / "src" / "api" / "python"
_BITCODE_SOURCE = _PYTHON_SOURCE / "device" / "cute" / "_bitcode.py"


def _load_bitcode_module():
    name = f"_nixl_test_bitcode_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, _BITCODE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_verified_bitcode(
    directory: Path,
    data: bytes = b"device-bitcode",
    *,
    arch: str = "sm_90",
    per_arch: bool = False,
    device_validation: bool = False,
    forceinline: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"_{arch}" if per_arch else ""
    bitcode = directory / f"libnixl_device{suffix}.bc"
    bitcode.write_bytes(data)
    (directory / f"nixl_device_abi{suffix}.json").write_text(
        json.dumps(
            {
                "abi_version": 3,
                "llvm_major": 20,
                "cuda_arch": arch,
                "bitcode": bitcode.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "device_validation": device_validation,
                "forceinline": forceinline,
            }
        ),
        encoding="utf-8",
    )
    return bitcode


def test_explicit_device_bitcode_is_verified(monkeypatch, tmp_path):
    bitcode = _write_verified_bitcode(tmp_path)
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")

    module = _load_bitcode_module()
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    assert module.device_bitcode_path() == str(bitcode.resolve())


@pytest.mark.parametrize(
    "failure",
    [
        "missing_manifest",
        "wrong_hash",
        "wrong_abi",
        "boolean_abi",
        "wrong_llvm",
        "boolean_llvm",
        "missing_device_validation",
        "non_boolean_device_validation",
        "missing_forceinline",
        "non_boolean_forceinline",
        "missing_cuda_arch",
        "invalid_cuda_arch",
        "wrong_cuda_arch",
    ],
)
def test_device_bitcode_verification_fails_closed(monkeypatch, tmp_path, failure):
    bitcode = tmp_path / "libnixl_device.bc"
    bitcode.write_bytes(b"actual")
    manifest = {
        "abi_version": 3,
        "llvm_major": 20,
        "cuda_arch": "sm_90",
        "bitcode": bitcode.name,
        "sha256": hashlib.sha256(b"actual").hexdigest(),
        "device_validation": False,
        "forceinline": True,
    }
    if failure == "wrong_hash":
        manifest["sha256"] = "0" * 64
    elif failure == "wrong_abi":
        manifest["abi_version"] = 4
    elif failure == "boolean_abi":
        manifest["abi_version"] = True
    elif failure == "wrong_llvm":
        manifest["llvm_major"] = 19
    elif failure == "boolean_llvm":
        manifest["llvm_major"] = True
    elif failure == "missing_device_validation":
        manifest.pop("device_validation")
    elif failure == "non_boolean_device_validation":
        manifest["device_validation"] = 0
    elif failure == "missing_forceinline":
        manifest.pop("forceinline")
    elif failure == "non_boolean_forceinline":
        manifest["forceinline"] = 1
    elif failure == "missing_cuda_arch":
        manifest.pop("cuda_arch")
    elif failure == "invalid_cuda_arch":
        manifest["cuda_arch"] = "native"
    elif failure == "wrong_cuda_arch":
        manifest["cuda_arch"] = "sm_100"
    if failure != "missing_manifest":
        (tmp_path / "nixl_device_abi.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")
    module = _load_bitcode_module()
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    with pytest.raises(module.BitcodeVerificationError):
        module.device_bitcode_path()


def test_override_requires_explicit_risk_acknowledgement(monkeypatch, tmp_path):
    bitcode = _write_verified_bitcode(tmp_path)
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    module = _load_bitcode_module()
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    with pytest.raises(module.BitcodeVerificationError, match="development-only"):
        module.device_bitcode_path()

    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")
    assert module.device_bitcode_path() == str(bitcode.resolve())


def test_packaged_bitcode_is_the_only_production_source(monkeypatch, tmp_path):
    bitcode = _write_verified_bitcode(tmp_path)
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    assert module.device_bitcode_path() == str(bitcode.resolve())


def test_packaged_bitcode_selects_exact_current_sm(monkeypatch, tmp_path):
    sm90 = _write_verified_bitcode(tmp_path, b"sm90", arch="sm_90", per_arch=True)
    sm100 = _write_verified_bitcode(tmp_path, b"sm100", arch="sm_100", per_arch=True)
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")

    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_100")
    assert module.device_bitcode_path() == str(sm100.resolve())
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")
    assert module.device_bitcode_path() == str(sm90.resolve())


def test_packaged_bitcode_rejects_unsupported_current_sm(monkeypatch, tmp_path):
    _write_verified_bitcode(tmp_path, arch="sm_90", per_arch=True)
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_100")

    with pytest.raises(
        module.UnsupportedArchitectureError,
        match=r"current device sm_100; packaged architectures: sm_90",
    ):
        module.device_bitcode_path()


def test_corrupt_exact_sm_artifact_never_falls_back(monkeypatch, tmp_path):
    _write_verified_bitcode(tmp_path, b"generic", arch="sm_90")
    exact = _write_verified_bitcode(tmp_path, b"exact", arch="sm_90", per_arch=True)
    exact.write_bytes(b"tampered")
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    with pytest.raises(module.BitcodeVerificationError, match="hash mismatch"):
        module.device_bitcode_path()


def test_legacy_generic_bitcode_requires_matching_current_sm(monkeypatch, tmp_path):
    bitcode = _write_verified_bitcode(tmp_path, arch="sm_90")
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")

    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")
    assert module.device_bitcode_path() == str(bitcode.resolve())
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_100")
    with pytest.raises(module.UnsupportedArchitectureError, match="targets sm_90"):
        module.device_bitcode_path()


@pytest.mark.parametrize(
    ("device_validation", "forceinline", "message"),
    [
        (True, True, "debug validation branches"),
        (False, False, "not a force-inline production build"),
    ],
)
def test_packaged_bitcode_requires_production_flags(
    monkeypatch, tmp_path, device_validation, forceinline, message
):
    _write_verified_bitcode(
        tmp_path,
        arch="sm_90",
        per_arch=True,
        device_validation=device_validation,
        forceinline=forceinline,
    )
    monkeypatch.delenv("NIXL_DEVICE_BITCODE", raising=False)
    module = _load_bitcode_module()
    module.__file__ = str(tmp_path / "_bitcode.py")
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    with pytest.raises(module.BitcodeVerificationError, match=message):
        module.device_bitcode_path()


def test_explicit_override_permits_authenticated_debug_artifact(monkeypatch, tmp_path):
    bitcode = _write_verified_bitcode(tmp_path, arch="sm_90", device_validation=True)
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")
    module = _load_bitcode_module()
    monkeypatch.setattr(module, "_current_device_arch", lambda: "sm_90")

    assert module.device_bitcode_path() == str(bitcode.resolve())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((8, 0), "sm_80"),
        ((9, 0), "sm_90"),
        ((10, 0), "sm_100"),
        ("compute_103", "sm_103"),
        ("sm_120", "sm_120"),
    ],
)
def test_bitcode_architecture_normalization(value, expected):
    module = _load_bitcode_module()

    assert module._normalize_arch(value) == expected


class _FakeHandle:
    def __init__(
        self,
        handle=0xCAFE,
        valid=True,
        kind="local",
        descriptor_lengths=(256,),
    ):
        self.handle = handle
        self.is_valid = valid
        self.kind = kind
        self.descriptor_lengths = descriptor_lengths


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def bitcast(self, dtype):
        return dtype(self.value)


class _FakeValue:
    def __init__(self, value_type="!llvm.ptr"):
        self.type = value_type


def _install_fake_cutlass(monkeypatch):
    adapters = {}
    cutlass = types.ModuleType("cutlass")
    cutlass.__path__ = []
    cutlass.Int32 = type("Int32", (_FakeScalar,), {})
    cutlass.Int64 = type("Int64", (_FakeScalar,), {})
    cutlass.Uint32 = type("Uint32", (_FakeScalar,), {})
    cutlass.Uint64 = type("Uint64", (_FakeScalar,), {})

    class DSLRuntimeError(RuntimeError):
        pass

    cutlass.DSLRuntimeError = DSLRuntimeError

    def register_jit_arg_adapter(python_type):
        def decorate(adapter):
            adapters[python_type] = adapter
            return adapter

        return decorate

    cutlass.register_jit_arg_adapter = register_jit_arg_adapter

    cute = types.ModuleType("cutlass.cute")
    cute.__path__ = []

    class BitCode:
        def __init__(self, path):
            self.path = path

    cute.BitCode = BitCode

    def extern(*, name, source):
        def decorate(function):
            def invoke(*args):
                return name, args

            invoke.__name__ = function.__name__
            invoke.extern_name = name
            invoke.source = source
            return invoke

        return decorate

    cute.extern = extern
    cutlass.cute = cute

    ir = types.ModuleType("cutlass.cutlass_dsl.ir")

    class Type:
        @staticmethod
        def parse(value):
            return value

    ir.Type = Type
    ir.Value = _FakeValue
    cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")
    cutlass_dsl.__path__ = []
    cutlass_dsl.ir = ir

    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl", cutlass_dsl)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl.ir", ir)
    return adapters


def _purge_test_packages():
    for name in tuple(sys.modules):
        if name == "nixl" or name.startswith("nixl.device"):
            sys.modules.pop(name, None)


@pytest.fixture
def fake_cute_package(monkeypatch, tmp_path):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    root.nixl_device_view_handle = _FakeHandle
    monkeypatch.setitem(sys.modules, "nixl", root)
    adapters = _install_fake_cutlass(monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: "4.5.1",
    )
    bitcode = _write_verified_bitcode(tmp_path)
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")
    package = importlib.import_module("nixl.device.cute")
    yield package, adapters
    _purge_test_packages()


def test_device_package_does_not_import_cutlass(monkeypatch):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    monkeypatch.setitem(sys.modules, "nixl", root)
    for name in tuple(sys.modules):
        if name == "cutlass" or name.startswith("cutlass."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("nixl.device")

    assert "cutlass" not in sys.modules
    _purge_test_packages()


def test_cute_package_reports_supported_optional_dependency(monkeypatch):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    monkeypatch.setitem(sys.modules, "nixl", root)
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: "4.5.1",
    )

    with pytest.raises(ImportError) as error:
        importlib.import_module("nixl.device.cute")

    message = str(error.value)
    assert "CUTLASS DSL 4.5.1" in message
    assert "nixl[cute-cu12]" in message
    assert "nixl-cu12[cute]" in message
    assert "CUDA 13: no extra is declared" in message
    assert "setup.sh --cu13" in message
    _purge_test_packages()


@pytest.mark.parametrize("installed_version", ["4.5.0", "4.6.0"])
def test_cute_package_rejects_unqualified_cutlass_version(
    monkeypatch, installed_version
):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    root.nixl_device_view_handle = _FakeHandle
    monkeypatch.setitem(sys.modules, "nixl", root)
    _install_fake_cutlass(monkeypatch)
    monkeypatch.delenv("NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL", raising=False)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: installed_version,
    )

    with pytest.raises(ImportError) as error:
        importlib.import_module("nixl.device.cute")

    message = str(error.value)
    assert repr(installed_version) in message
    assert "NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL=1" in message
    _purge_test_packages()


@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_cute_package_fails_closed_when_cutlass_version_is_unverifiable(
    monkeypatch, failure
):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    root.nixl_device_view_handle = _FakeHandle
    monkeypatch.setitem(sys.modules, "nixl", root)
    _install_fake_cutlass(monkeypatch)
    monkeypatch.delenv("NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL", raising=False)

    def fail_version_query(distribution):
        if failure == "missing":
            raise importlib.metadata.PackageNotFoundError(distribution)
        raise RuntimeError("corrupt distribution metadata")

    monkeypatch.setattr(importlib.metadata, "version", fail_version_query)

    with pytest.raises(ImportError, match="metadata is missing or unreadable"):
        importlib.import_module("nixl.device.cute")

    _purge_test_packages()


def test_cute_package_unsafe_version_override_is_explicit_and_warns(
    monkeypatch, tmp_path
):
    _purge_test_packages()
    root = types.ModuleType("nixl")
    root.__path__ = [str(_PYTHON_SOURCE)]
    root.nixl_device_view_handle = _FakeHandle
    monkeypatch.setitem(sys.modules, "nixl", root)
    _install_fake_cutlass(monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: "4.6.0",
    )
    bitcode = _write_verified_bitcode(tmp_path)
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(bitcode))
    monkeypatch.setenv("NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE", "1")
    monkeypatch.setenv("NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL", "1")

    with pytest.warns(RuntimeWarning, match="found unqualified version"):
        package = importlib.import_module("nixl.device.cute")

    assert package.NIXL_CUTE_ABI_VERSION == 3
    _purge_test_packages()


def test_low_level_null_agent_gap_contract_is_explicit():
    header = (
        Path(__file__).parents[2] / "src" / "api" / "device" / "gpu" / "nixl_device.cuh"
    ).read_text(encoding="utf-8")
    contract = " ".join(header.split())

    assert "NIXL_NULL_AGENT gap" in contract
    assert "passed as dst" in contract
    assert "passed to this operation" in contract
    assert "nixlGetPtr(), which returns nullptr" in contract


def test_cute_mapped_pointer_contract_uses_the_process_local_base():
    ops = (_PYTHON_SOURCE / "device" / "cute" / "ops.py").read_text(encoding="utf-8")
    header = (
        Path(__file__).parents[2]
        / "src"
        / "api"
        / "gpu"
        / "ucx"
        / "cute"
        / "nixl_device_cute.cuh"
    ).read_text(encoding="utf-8")
    ops_contract = " ".join(ops.split())
    header_contract = " ".join(header.split())

    assert "address in the calling process" in ops_contract
    assert "need not equal the descriptor owner's address" in ops_contract
    assert "returned base plus a descriptor-relative offset" in header_contract
    assert "numeric address equality is not required" in header_contract
    assert "native atomics for every directed" in header_contract
    assert "same-VA" not in ops_contract
    assert "same-VA" not in header_contract


def test_bitcode_binding_resolves_override_lazily(
    fake_cute_package, monkeypatch, tmp_path
):
    package, _ = fake_cute_package
    bindings = importlib.import_module("nixl.device.cute._bindings")
    bitcode_module = importlib.import_module("nixl.device.cute._bitcode")
    replacement = _write_verified_bitcode(tmp_path / "replacement", b"replacement")
    monkeypatch.setattr(bitcode_module, "_current_device_arch", lambda: "sm_90")
    monkeypatch.setenv("NIXL_DEVICE_BITCODE", str(replacement))

    assert "path" not in vars(bindings._BC)
    assert bindings._BC.path == str(replacement.resolve())
    assert (
        package.UnsupportedArchitectureError
        is bitcode_module.UnsupportedArchitectureError
    )


def test_memory_view_protocol_adapter_and_fake(fake_cute_package):
    package, adapters = fake_cute_package
    handle = _FakeHandle()
    view = package.MemoryView(handle)

    view_arg = view.__c_pointers__()[0]
    assert view_arg != 0xCAFE
    assert ctypes.c_void_p.from_address(view_arg).value == 0xCAFE
    adapted = adapters[_FakeHandle](handle)
    assert ctypes.c_void_p.from_address(adapted.__c_pointers__()[0]).value == 0xCAFE
    fake = package.make_fake_memory_view("remote", (64, 128))
    assert isinstance(fake, package.MemoryView)
    assert len(fake.__c_pointers__()) == 1
    assert not isinstance(fake.__c_pointers__()[0], int)
    with pytest.raises(TypeError, match="pointer"):
        ctypes.c_void_p(fake.__c_pointers__()[0])
    assert fake.__get_mlir_types__() == ["!llvm.ptr"]
    traced_fake = fake.__new_from_mlir_values__([_FakeValue()])
    assert traced_fake.kind == "remote"
    assert traced_fake.descriptor_lengths == (64, 128)

    with pytest.raises(ValueError, match="kind"):
        package.make_fake_memory_view("peer")

    handle.is_valid = False
    with pytest.raises(RuntimeError, match="released before launch"):
        view.__c_pointers__()


def test_checked_compile_enforces_fake_view_contracts_before_marshalling(
    fake_cute_package, monkeypatch
):
    package, adapters = fake_cute_package

    def launcher(local, remote, count=1):
        del local, remote, count

    class FakeExecutionArgs:
        def __init__(self):
            self.signature = inspect.signature(launcher)

        def _bind(self, args, kwargs):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return tuple(bound.arguments.values())

        def get_rectified_args_from_original_args(self, args, kwargs):
            return self._bind(args, kwargs)

        def get_rectified_args(self, args, kwargs):
            return self._bind(args, kwargs)

    class FakeCompiled:
        def __init__(self):
            self.execution_args = FakeExecutionArgs()
            self.marshal_attempts = 0
            self.bound_devices = []
            self.marker = "underlying-compiled-callable"

        def __call__(self, *args, **kwargs):
            self.marshal_attempts += 1
            values = self.execution_args.get_rectified_args(args, kwargs)
            for value in values[:2]:
                if isinstance(value, _FakeHandle):
                    value = adapters[_FakeHandle](value)
                value.__c_pointers__()
            return "launched"

        def generate_execution_args(self, *args, **kwargs):
            self.marshal_attempts += 1
            return [args, kwargs], []

        def to(self, device=None):
            self.bound_devices.append(device)
            return self

    underlying = FakeCompiled()
    fake_compile_calls = []

    def fake_compile(function, *args, **kwargs):
        fake_compile_calls.append((function, args, kwargs))
        return underlying

    monkeypatch.setattr(
        sys.modules["cutlass.cute"], "compile", fake_compile, raising=False
    )
    local_prototype = package.make_fake_memory_view("local", (64, 128))
    remote_prototype = package.make_fake_memory_view("remote", (256,))
    compiled = package.compile(launcher, local_prototype, remote_prototype, 7)

    assert fake_compile_calls == [
        (launcher, (local_prototype, remote_prototype, 7), {})
    ]
    assert compiled.marker == "underlying-compiled-callable"
    local = _FakeHandle(kind="local", descriptor_lengths=(64, 256))
    remote = _FakeHandle(kind="remote", descriptor_lengths=(512,))
    assert compiled(local, remote, 9) == "launched"
    assert underlying.marshal_attempts == 1

    def reject_without_marshalling(value, pattern):
        attempts = underlying.marshal_attempts
        with pytest.raises((TypeError, ValueError, RuntimeError), match=pattern):
            compiled(value, remote, 9)
        assert underlying.marshal_attempts == attempts

    reject_without_marshalling(
        _FakeHandle(kind="remote", descriptor_lengths=(64, 128)),
        "compiled prototype requires 'local'",
    )
    reject_without_marshalling(
        _FakeHandle(kind="local", descriptor_lengths=(63, 128)),
        "descriptor 0 has 63 bytes; compiled prototype requires at least 64",
    )
    reject_without_marshalling(
        _FakeHandle(kind="local", descriptor_lengths=(64,)),
        "has 1 descriptors; compiled prototype requires exactly 2",
    )
    reject_without_marshalling(
        _FakeHandle(valid=False, kind="local", descriptor_lengths=(64, 128)),
        "released before launch",
    )

    stale_handle = _FakeHandle(kind="local", descriptor_lengths=(64, 128))
    stale_view = package.MemoryView(stale_handle)
    stale_handle.is_valid = False
    reject_without_marshalling(stale_view, "released before launch")
    reject_without_marshalling(local_prototype, "compile-only fake MemoryView")

    executor = compiled.to(3)
    assert underlying.bound_devices == [3]
    assert executor(local=local, remote=remote, count=11) == "launched"
    attempts = underlying.marshal_attempts
    with pytest.raises(ValueError, match="compiled prototype requires 'remote'"):
        executor.generate_execution_args(
            local,
            _FakeHandle(kind="local", descriptor_lengths=(256,)),
            11,
        )
    assert underlying.marshal_attempts == attempts


def test_compile_without_memory_views_returns_exact_raw_cute_callable(
    fake_cute_package, monkeypatch
):
    package, _ = fake_cute_package

    def launcher(tensor, count):
        del tensor, count

    returned = object()
    calls = []

    def fake_compile(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        return returned

    monkeypatch.setattr(
        sys.modules["cutlass.cute"], "compile", fake_compile, raising=False
    )
    tensor = object()
    option = object()
    compiled = package.compile(launcher, tensor, 7, options=option)

    assert compiled is returned
    assert len(calls) == 1
    function, args, kwargs = calls[0]
    assert function is launcher
    assert args[0] is tensor
    assert args[1] == 7
    assert kwargs == {"options": option}
    assert kwargs["options"] is option


def test_operations_dispatch_only_compile_time_supported_scopes(fake_cute_package):
    package, _ = fake_cute_package
    local_prototype = package.MemoryView(_FakeHandle(kind="local"))
    remote_prototype = package.MemoryView(
        _FakeHandle(kind="remote", descriptor_lengths=(256,) * 4)
    )
    local = local_prototype.__new_from_mlir_values__([_FakeValue()])
    remote = remote_prototype.__new_from_mlir_values__([_FakeValue()])

    name, args = package.put(local, remote, 128, scope=package.Scope.WARP).value
    assert name == "nixl_cute_put_warp_wait"
    assert isinstance(package.put(local, remote, 128), sys.modules["cutlass"].Int32)
    assert args[6].value == 128
    assert args[8].value == int(package.Flags.NONE)

    name, args = package.atomic_add(remote, 7, index=2, scope=package.Scope.WARP).value
    assert name == "nixl_cute_atomic_add_warp_wait"
    assert args[0].value == 7
    assert args[2].value == 2

    name, args = package.put_post(
        local,
        remote,
        128,
        flags=package.Flags.DEFER,
        scope=package.Scope.WARP,
    ).value
    assert name == "nixl_cute_put_warp_post"
    assert args[8].value == int(package.Flags.DEFER)

    name, args = package.atomic_add_post(
        remote, 1, index=2, scope=package.Scope.THREAD
    ).value
    assert name == "nixl_cute_atomic_add_thread_post"
    assert args[2].value == 2

    timer = package.globaltimer_ns()
    assert isinstance(timer, sys.modules["cutlass"].Uint64)
    assert timer.value[0] == "nixl_cute_globaltimer_ns"

    acquire = package.load_acquire_system_u64(0x1000)
    assert acquire.value[0] == "nixl_cute_load_acquire_system_u64"
    assert acquire.value[1][0].value == 0x1000

    gpu_acquire = package.load_acquire_gpu_u64(0x1000)
    assert gpu_acquire.value[0] == "nixl_cute_load_acquire_gpu_u64"
    assert gpu_acquire.value[1][0].value == 0x1000

    wait_acquire = package.wait_acquire_system_u64(0x1000, 9, scope=package.Scope.WARP)
    assert wait_acquire.value[0] == "nixl_cute_wait_acquire_system_u64_warp"
    assert wait_acquire.value[1][0].value == 0x1000
    assert wait_acquire.value[1][1].value == 9

    bounded = package.wait_acquire_system_u64_until(
        0x1000, 11, 123456, scope=package.Scope.WARP
    )
    assert bounded.value[0] == "nixl_cute_wait_acquire_system_u64_warp_until"
    assert bounded.value[1][0].value == 0x1000
    assert bounded.value[1][1].value == 11
    assert bounded.value[1][2].value == 123456

    relative = package.wait_acquire_system_u64_for(
        0x1000, 13, 654321, scope=package.Scope.WARP
    )
    assert relative.value[0] == "nixl_cute_wait_acquire_system_u64_warp_for"
    assert relative.value[1][0].value == 0x1000
    assert relative.value[1][1].value == 13
    assert relative.value[1][2].value == 654321

    abortable_relative = package.wait_acquire_system_u64_for_or_abort(
        0x1000, 15, 0x2000, 16, 777777, scope=package.Scope.WARP
    )
    assert (
        abortable_relative.value[0]
        == "nixl_cute_wait_acquire_system_u64_warp_for_or_abort"
    )
    assert [arg.value for arg in abortable_relative.value[1]] == [
        0x1000,
        15,
        0x2000,
        16,
        777777,
    ]

    gpu_relative = package.wait_acquire_gpu_u64_for(
        0x1000, 17, 222222, scope=package.Scope.WARP
    )
    assert gpu_relative.value[0] == "nixl_cute_wait_acquire_gpu_u64_warp_for"
    assert gpu_relative.value[1][1].value == 17
    assert gpu_relative.value[1][2].value == 222222

    gpu_wait = package.wait_acquire_gpu_u64(0x1000, 19, scope=package.Scope.WARP)
    assert gpu_wait.value[0] == "nixl_cute_wait_acquire_gpu_u64_warp"
    assert gpu_wait.value[1][0].value == 0x1000
    assert gpu_wait.value[1][1].value == 19

    gpu_abort_wait = package.wait_acquire_gpu_u64_or_abort(
        0x1000, 23, 0x2000, 29, scope=package.Scope.WARP
    )
    assert gpu_abort_wait.value[0] == "nixl_cute_wait_acquire_gpu_u64_or_abort_warp"
    assert [arg.value for arg in gpu_abort_wait.value[1]] == [
        0x1000,
        23,
        0x2000,
        29,
    ]

    system_abort_wait = package.wait_acquire_system_u64_or_abort(
        0x1000, 31, 0x2000, 37, scope=package.Scope.WARP
    )
    assert (
        system_abort_wait.value[0] == "nixl_cute_wait_acquire_system_u64_or_abort_warp"
    )
    assert [arg.value for arg in system_abort_wait.value[1]] == [
        0x1000,
        31,
        0x2000,
        37,
    ]

    dual_abort_wait = package.wait_acquire_system_u64_or_aborts(
        0x1000, 41, 0x2000, 43, 0x3000, 47, scope=package.Scope.WARP
    )
    assert (
        dual_abort_wait.value[0] == "nixl_cute_wait_acquire_system_u64_or_aborts_warp"
    )
    assert [arg.value for arg in dual_abort_wait.value[1]] == [
        0x1000,
        41,
        0x2000,
        43,
        0x3000,
        47,
    ]

    release = package.store_release_system_u64(0x1000, 7)
    assert release.value[0] == "nixl_cute_store_release_system_u64"
    assert release.value[1][1].value == 7

    gpu_release = package.store_release_gpu_u64(0x1000, 8)
    assert gpu_release.value[0] == "nixl_cute_store_release_gpu_u64"
    assert gpu_release.value[1][1].value == 8

    gpu_atomic_add = package.atomic_add_release_gpu_u64(0x1000, 3)
    assert gpu_atomic_add.value[0] == "nixl_cute_atomic_add_release_gpu_u64"
    assert gpu_atomic_add.value[1][1].value == 3

    gpu_atomic_max = package.atomic_max_release_gpu_u64(0x1000, 9)
    assert gpu_atomic_max.value[0] == "nixl_cute_atomic_max_release_gpu_u64"
    assert gpu_atomic_max.value[1][1].value == 9

    system_atomic_max = package.atomic_max_release_system_u64(0x1000, 10)
    assert system_atomic_max.value[0] == "nixl_cute_atomic_max_release_system_u64"
    assert system_atomic_max.value[1][1].value == 10

    status_prior = package.compare_exchange_status_gpu_i32(0x1000, -7)
    assert status_prior.value[0] == "nixl_cute_compare_exchange_status_gpu_i32"
    assert status_prior.value[1][1].value == -7

    release_fence = package.fence_release_system()
    assert release_fence.value[0] == "nixl_cute_fence_release_system"

    grid_sync = package.sync_grid()
    assert grid_sync.value[0] == "nixl_cute_sync_grid"

    name, args = package.get_ptr(remote, 3)
    assert name == "nixl_cute_get_ptr"
    assert args[1].value == 3

    status = package.mapped_copy_warp(
        0x1000, remote, 128, remote_index=2, remote_offset=16
    )
    assert status.value[0] == "nixl_cute_mapped_copy_warp"
    assert status.value[1][0].value == 0x1000
    assert status.value[1][2].value == 2
    assert status.value[1][3].value == 16
    assert status.value[1][4].value == 128

    readonly = package.mapped_copy_warp_readonly(
        0x1000, remote, 128, remote_index=2, remote_offset=16
    )
    assert readonly.value[0] == "nixl_cute_mapped_copy_warp_readonly"
    assert readonly.value[1][4].value == 128

    ptr_status = package.mapped_copy_warp_ptr(0x1000, 0x2000, 128)
    assert ptr_status.value[0] == "nixl_cute_mapped_copy_warp_ptr"
    assert ptr_status.value[1][0].value == 0x1000
    assert ptr_status.value[1][1].value == 0x2000
    assert ptr_status.value[1][2].value == 128

    readonly_ptr = package.mapped_copy_warp_ptr_readonly(0x1000, 0x2000, 128)
    assert readonly_ptr.value[0] == "nixl_cute_mapped_copy_warp_ptr_readonly"

    with pytest.raises(NotImplementedError, match="GRID"):
        package.put(local, remote, 1, scope=package.Scope.GRID)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.put(local, remote, 1, scope=package.Scope.BLOCK)
    with pytest.raises(TypeError, match="compile-time"):
        package.put(local, remote, 1, scope=0)
    with pytest.raises(ValueError, match="DEFER"):
        package.put(local, remote, 1, flags=package.Flags.DEFER)
    with pytest.raises(ValueError, match="Flags.NONE"):
        package.put(local, remote, 1, flags=package.Flags(2))
    with pytest.raises(ValueError, match="Flags.NONE or Flags.DEFER"):
        package.put_post(local, remote, 1, flags=package.Flags(2))
    with pytest.raises(ValueError, match="nonzero"):
        package.load_acquire_system_u64(0)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.wait_acquire_system_u64(8, 1, scope=package.Scope.BLOCK)
    with pytest.raises(NotImplementedError, match="GRID"):
        package.wait_acquire_system_u64_until(8, 1, 2, scope=package.Scope.GRID)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.wait_acquire_system_u64_for(8, 1, 2, scope=package.Scope.BLOCK)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.wait_acquire_system_u64_for_or_abort(
            8, 1, 16, 2, 3, scope=package.Scope.BLOCK
        )
    with pytest.raises(NotImplementedError, match="GRID"):
        package.wait_acquire_gpu_u64_for(8, 1, 2, scope=package.Scope.GRID)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.wait_acquire_gpu_u64(8, 1, scope=package.Scope.BLOCK)
    with pytest.raises(NotImplementedError, match="GRID"):
        package.wait_acquire_gpu_u64_or_abort(8, 1, 16, 1, scope=package.Scope.GRID)
    with pytest.raises(NotImplementedError, match="BLOCK"):
        package.wait_acquire_system_u64_or_abort(8, 1, 16, 1, scope=package.Scope.BLOCK)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.wait_acquire_system_u64_or_abort(8, 1, 12, 1)
    with pytest.raises(NotImplementedError, match="GRID"):
        package.wait_acquire_system_u64_or_aborts(
            8, 1, 16, 1, 24, 1, scope=package.Scope.GRID
        )
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.wait_acquire_system_u64_or_aborts(8, 1, 16, 1, 20, 1)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.store_release_system_u64(3, 1)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.store_release_gpu_u64(3, 1)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.atomic_add_release_gpu_u64(3, 1)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.atomic_max_release_system_u64(3, 1)
    with pytest.raises(ValueError, match="4-byte aligned"):
        package.compare_exchange_status_gpu_i32(2, -1)
    with pytest.raises(ValueError, match="fit in int32"):
        package.compare_exchange_status_gpu_i32(4, 1 << 31)
    with pytest.raises(ValueError, match="source_address must be nonzero"):
        package.mapped_copy_warp(0, remote, 16)
    with pytest.raises(ValueError, match="source_address must be 16-byte aligned"):
        package.mapped_copy_warp(8, remote, 16)
    with pytest.raises(ValueError, match="remote_offset must be 16-byte aligned"):
        package.mapped_copy_warp(0x1000, remote, 16, remote_offset=8)
    with pytest.raises(ValueError, match="multiple of 16"):
        package.mapped_copy_warp(0x1000, remote, 17)
    with pytest.raises(ValueError, match="multiple of 16"):
        package.mapped_copy_warp_readonly(0x1000, remote, 17)
    with pytest.raises(ValueError, match="destination_address must be nonzero"):
        package.mapped_copy_warp_ptr(0x1000, 0, 16)
    with pytest.raises(ValueError, match="destination_address must be 16-byte aligned"):
        package.mapped_copy_warp_ptr(0x1000, 0x2008, 16)
    with pytest.raises(ValueError, match="size"):
        package.put(local, remote, -1)


def test_operations_validate_view_kind_and_literal_bounds(fake_cute_package):
    package, _ = fake_cute_package
    local_prototype = package.MemoryView(
        _FakeHandle(kind="local", descriptor_lengths=(64,))
    )
    remote_prototype = package.MemoryView(
        _FakeHandle(kind="remote", descriptor_lengths=(32, 128))
    )
    local = local_prototype.__new_from_mlir_values__([_FakeValue()])
    remote = remote_prototype.__new_from_mlir_values__([_FakeValue()])

    with pytest.raises(ValueError, match="must be a local"):
        package.put(remote, local, 8)
    with pytest.raises(IndexError, match="local_index"):
        package.put(local, remote, 8, local_index=1)
    with pytest.raises(ValueError, match="local span"):
        package.put(local, remote, 16, local_offset=56)
    with pytest.raises(ValueError, match="remote span"):
        package.put(local, remote, 16, remote_offset=24)
    with pytest.raises(ValueError, match="greater than zero"):
        package.put(local, remote, 0)
    with pytest.raises(ValueError, match="8-byte aligned"):
        package.atomic_add(remote, 1, offset=4)
    with pytest.raises(ValueError, match="needs 8 bytes"):
        package.atomic_add(remote, 1, offset=32)
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        package.put(local, remote, 8, channel=1 << 32)
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        package.put(local, remote, 8, remote_index=1 << 32)
    with pytest.raises(ValueError, match=r"\[0, 18446744073709551615\]"):
        package.put(local, remote, 1 << 64)
    unknown_local = package.make_fake_memory_view("local").__new_from_mlir_values__(
        [_FakeValue()]
    )
    unknown_remote = package.make_fake_memory_view("remote").__new_from_mlir_values__(
        [_FakeValue()]
    )
    with pytest.raises(ValueError, match="overflows uint64"):
        package.put(
            unknown_local,
            unknown_remote,
            8,
            local_offset=(1 << 64) - 4,
        )
    with pytest.raises(ValueError, match="must be a remote"):
        package.get_ptr(local)
