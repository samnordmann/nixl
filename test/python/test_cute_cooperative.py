# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
import sys
import types

import pytest

from examples.python.cute import _cooperative

_REAL_LOAD_CUDA_HANDLE_CONTRACT = _cooperative._load_cuda_handle_contract


class _Result:
    CUDA_SUCCESS = 0


class _DeviceAttribute:
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


class _FunctionAttribute:
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4


_SYMBOL = "_mlir_cute_elastic_kernel"
_LIBRARY_NATIVE = 0xC0FFEE


def _handle_init(self, native):
    self.native = native


def _handle_int(self):
    return self.native


def _handle_get_ptr(_self):
    raise AssertionError("occupancy query must never use handle.getPtr()")


_HANDLE_METHODS = {
    "__init__": _handle_init,
    "__int__": _handle_int,
    "getPtr": _handle_get_ptr,
}


_RuntimeLibrary = type(
    "cudaLibrary_t",
    (),
    {"__module__": "cuda.bindings.runtime", **_HANDLE_METHODS},
)
_DriverLibrary = type(
    "CUlibrary",
    (),
    {"__module__": "cuda.bindings.driver", **_HANDLE_METHODS},
)
_DriverKernel = type(
    "CUkernel",
    (),
    {"__module__": "cuda.bindings.driver", **_HANDLE_METHODS},
)
_DriverFunction = type(
    "CUfunction",
    (),
    {"__module__": "cuda.bindings.driver", **_HANDLE_METHODS},
)


class _Driver:
    CUresult = _Result
    CUdevice_attribute = _DeviceAttribute
    CUfunction_attribute = _FunctionAttribute
    CUlibrary = _DriverLibrary
    CUkernel = _DriverKernel
    CUfunction = _DriverFunction
    kernel_symbol = _SYMBOL
    function_symbol = _SYMBOL
    owner_library_native = _LIBRARY_NATIVE
    requested_symbols = []

    @staticmethod
    def cuGetErrorName(_status):
        return 0, b"CUDA_SUCCESS"

    @staticmethod
    def cuCtxGetDevice():
        return 0, 0

    @staticmethod
    def cuDeviceGet(device):
        return 0, device

    @staticmethod
    def cuDeviceGetAttribute(attribute, _device):
        if attribute == _DeviceAttribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
            return 0, 1
        if attribute == _DeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            return 0, 148
        raise AssertionError(attribute)

    @staticmethod
    def cuLibraryGetKernel(library, symbol):
        assert type(library) is _DriverLibrary
        assert library.native == _LIBRARY_NATIVE
        _Driver.requested_symbols.append(symbol)
        return 0, _DriverKernel(0xBEEF)

    @staticmethod
    def cuKernelGetName(kernel):
        assert type(kernel) is _DriverKernel
        return 0, _Driver.kernel_symbol.encode()

    @staticmethod
    def cuKernelGetLibrary(kernel):
        assert type(kernel) is _DriverKernel
        return 0, _DriverLibrary(_Driver.owner_library_native)

    @staticmethod
    def cuKernelGetFunction(kernel):
        assert type(kernel) is _DriverKernel
        return 0, _DriverFunction(0xF00D)

    @staticmethod
    def cuFuncGetName(function):
        assert type(function) is _DriverFunction
        return 0, _Driver.function_symbol.encode()

    @staticmethod
    def cuLibraryLoadData(*_args):
        raise AssertionError("occupancy query must not load another CUDA library")

    @staticmethod
    def cuLibraryUnload(*_args):
        raise AssertionError("occupancy query must not unload CuTe's borrowed library")

    @staticmethod
    def cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function, block_threads, dynamic_smem_bytes
    ):
        assert type(function) is _DriverFunction
        assert block_threads == 256 and dynamic_smem_bytes == 0
        return 0, 2

    @staticmethod
    def cuFuncGetAttribute(attribute, function):
        assert type(function) is _DriverFunction
        return (
            0,
            {
                _FunctionAttribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 1024,
                _FunctionAttribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: 0,
                _FunctionAttribute.CU_FUNC_ATTRIBUTE_NUM_REGS: 96,
            }[attribute],
        )


_CompiledTarget = type(
    "CudaDialectJitCompiledFunction",
    (),
    {
        "__module__": "cutlass.cutlass_dsl.cuda_jit_executor",
    },
)
_BoundTarget = type("JitExecutor", (), {"__module__": "cutlass.base_dsl.jit_executor"})
_CudaDialectModule = type(
    "CudaDialectJitModule",
    (),
    {
        "__module__": "cutlass.cutlass_dsl.cuda_jit_executor",
        "is_unloaded": lambda self: self._unloaded,
    },
)
_ContractWrapper = type(
    "_ContractCheckedCallable", (), {"__module__": "nixl.device.cute.memory"}
)


@pytest.fixture(autouse=True)
def _pinned_runtime_types(monkeypatch):
    _Driver.kernel_symbol = _SYMBOL
    _Driver.function_symbol = _SYMBOL
    _Driver.owner_library_native = _LIBRARY_NATIVE
    _Driver.requested_symbols = []
    monkeypatch.setattr(
        _cooperative,
        "_load_cute_target_types",
        lambda: (
            _ContractWrapper,
            _CompiledTarget,
            _BoundTarget,
            _CudaDialectModule,
        ),
    )
    monkeypatch.setattr(
        _cooperative,
        "_load_cuda_handle_contract",
        lambda: (_RuntimeLibrary, lambda handle: handle.native),
    )


def _wrapped(target):
    wrapper = _ContractWrapper()
    wrapper.__wrapped__ = target
    return wrapper


def _install_fake_cuda_handle_modules(monkeypatch, *, include_utils: bool):
    runtime = types.ModuleType("cuda.bindings.runtime")
    runtime.cudaLibrary_t = _RuntimeLibrary
    bindings = types.ModuleType("cuda.bindings")
    bindings.__path__ = []
    bindings.runtime = runtime
    cuda = types.ModuleType("cuda")
    cuda.__path__ = []
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime)
    monkeypatch.delitem(sys.modules, "cuda.bindings.utils", raising=False)
    if not include_utils:
        return None

    def get_cuda_native_handle(handle):
        return handle.native

    get_cuda_native_handle.__module__ = "cuda.bindings.utils"
    utils = types.ModuleType("cuda.bindings.utils")
    utils.get_cuda_native_handle = get_cuda_native_handle
    bindings.utils = utils
    monkeypatch.setitem(sys.modules, "cuda.bindings.utils", utils)
    return get_cuda_native_handle


def _compiled_and_bound(*, wrapped: bool):
    module = _CudaDialectModule()
    module._unloaded = False
    module.cuda_library = [_RuntimeLibrary(_LIBRARY_NATIVE)]
    compiled = _CompiledTarget()
    compiled.kernel_info = {_SYMBOL: {}}
    compiled.jit_module = module
    bound = _BoundTarget()
    bound.jit_module = module
    bound.exec_context = None
    bound._kernel_ptrs = None
    if wrapped:
        return _wrapped(compiled), _wrapped(bound)
    return compiled, bound


def _query(planned_ctas: int, *, wrapped: bool = True):
    _Driver.kernel_symbol = _SYMBOL
    _Driver.function_symbol = _SYMBOL
    _Driver.owner_library_native = _LIBRARY_NATIVE
    _Driver.requested_symbols = []
    compiled, bound = _compiled_and_bound(wrapped=wrapped)
    return _cooperative._query_compiled_specialization(
        compiled,
        bound,
        driver=_Driver,
        device_ordinal=0,
        block_threads=256,
        planned_ctas=planned_ctas,
        dynamic_smem_bytes=0,
    )


@pytest.mark.parametrize(
    "version,include_utils,expected_int",
    [("12.8.0", False, True), ("12.9.0", False, True), ("13.3.1", True, False)],
)
def test_cuda_bindings_version_selects_supported_native_bridge(
    monkeypatch, version, include_utils, expected_int
):
    expected_getter = _install_fake_cuda_handle_modules(
        monkeypatch, include_utils=include_utils
    )
    monkeypatch.setattr(
        _cooperative.importlib.metadata,
        "version",
        lambda distribution: version
        if distribution == "cuda-bindings"
        else pytest.fail(distribution),
    )

    runtime_type, getter = _REAL_LOAD_CUDA_HANDLE_CONTRACT()

    assert runtime_type is _RuntimeLibrary
    if expected_int:
        assert getter is int
    else:
        assert getter is expected_getter


@pytest.mark.parametrize("version", ["12.7.9", "14.0.0", "13.3.1+local", "invalid"])
def test_cuda_bindings_version_fails_closed_outside_supported_releases(
    monkeypatch, version
):
    _install_fake_cuda_handle_modules(monkeypatch, include_utils=True)
    monkeypatch.setattr(
        _cooperative.importlib.metadata, "version", lambda _distribution: version
    )

    with pytest.raises(RuntimeError, match="unsupported cuda-bindings version"):
        _REAL_LOAD_CUDA_HANDLE_CONTRACT()


@pytest.mark.parametrize("wrapped", [False, True])
def test_exact_bound_function_occupancy_is_reported_without_load_or_unload(wrapped):
    occupancy = _query(16, wrapped=wrapped)

    assert occupancy.active_ctas_per_sm == 2
    assert occupancy.cooperative_cta_capacity == 296
    assert occupancy.registers_per_thread == 96
    assert occupancy.planned_ctas == 16
    assert occupancy.kernel_symbol == _SYMBOL
    assert occupancy.cu_function_source == (
        "bound_cuda_library->cuLibraryGetKernel(exact_kernel_info_symbol)"
        "->cuKernelGetFunction(current_context)"
    )
    assert _Driver.requested_symbols == [_SYMBOL.encode()]


def test_cuda12_int_handle_bridge_preserves_exact_bound_function(monkeypatch):
    monkeypatch.setattr(
        _cooperative,
        "_load_cuda_handle_contract",
        lambda: (_RuntimeLibrary, int),
    )

    occupancy = _query(16, wrapped=True)

    assert occupancy.kernel_symbol == _SYMBOL
    assert occupancy.cu_function_source == (
        "bound_cuda_library->cuLibraryGetKernel(exact_kernel_info_symbol)"
        "->cuKernelGetFunction(current_context)"
    )
    assert _Driver.requested_symbols == [_SYMBOL.encode()]


def test_oversubscribed_cooperative_grid_fails_closed_without_unload():
    with pytest.raises(RuntimeError, match="297 CTAs"):
        _query(297)


def test_private_adapter_rejects_wrong_wrapper_and_needs_no_retained_file():
    _, exact_bound = _compiled_and_bound(wrapped=False)
    with pytest.raises(RuntimeError, match="compiled target is not exact"):
        _cooperative._query_compiled_specialization(
            exact_bound,
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    no_retained_file, bound = _compiled_and_bound(wrapped=False)
    no_retained_file.artifacts = None
    no_retained_file.ir_module = None
    occupancy = _cooperative._query_compiled_specialization(
        _wrapped(no_retained_file),
        _wrapped(bound),
        driver=_Driver,
        device_ordinal=0,
        block_threads=256,
        planned_ctas=1,
        dynamic_smem_bytes=0,
    )
    assert occupancy.planned_ctas == 1


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda _compiled, bound: setattr(bound, "exec_context", object()),
            "exec_context",
        ),
        (lambda _compiled, bound: setattr(bound, "_kernel_ptrs", []), "_kernel_ptrs"),
        (
            lambda _compiled, bound: setattr(
                bound, "jit_module", _CudaDialectModule()
            ),
            "do not share",
        ),
        (
            lambda compiled, bound: (
                setattr(bound, "jit_module", object()),
                setattr(compiled, "jit_module", bound.jit_module),
            ),
            "bound module is not exact",
        ),
        (
            lambda compiled, bound: (
                setattr(compiled.jit_module, "_unloaded", True),
                setattr(bound.jit_module, "_unloaded", True),
            ),
            "already unloaded",
        ),
        (
            lambda compiled, _bound: setattr(compiled.jit_module, "cuda_library", []),
            "one bound CUDA library",
        ),
        (
            lambda compiled, _bound: setattr(
                compiled.jit_module, "cuda_library", [object()]
            ),
            "bound library is not exact",
        ),
    ],
)
def test_borrowed_library_contract_fails_closed_on_private_layout_drift(
    mutation, error
):
    compiled, bound = _compiled_and_bound(wrapped=False)
    mutation(compiled, bound)
    with pytest.raises(RuntimeError, match=error):
        _cooperative._query_compiled_specialization(
            compiled,
            bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )


def test_borrowed_library_requires_stable_native_handle_round_trip(monkeypatch):
    compiled, bound = _compiled_and_bound(wrapped=False)

    def changed_round_trip(handle):
        if type(handle) is _DriverLibrary:
            return handle.native + 1
        return handle.native

    monkeypatch.setattr(
        _cooperative,
        "_load_cuda_handle_contract",
        lambda: (_RuntimeLibrary, changed_round_trip),
    )
    with pytest.raises(RuntimeError, match="round trip changed value"):
        _cooperative._query_compiled_specialization(
            compiled,
            bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )


@pytest.mark.parametrize("native", [True, 0])
def test_native_handle_rejects_boolean_and_zero(native):
    with pytest.raises(RuntimeError, match="positive integer"):
        _cooperative._native_handle(lambda _handle: native, object(), "test")


def test_native_handle_reports_raising_conversion():
    class _RaisingHandle:
        def __int__(self):
            raise ValueError("cannot convert")

    with pytest.raises(RuntimeError, match="cannot extract test native handle"):
        _cooperative._native_handle(int, _RaisingHandle(), "test")


@pytest.mark.parametrize(
    "kernel_symbol,function_symbol,owner_native,error",
    [
        ("other_kernel", _SYMBOL, _LIBRARY_NATIVE, "does not match compiled"),
        (_SYMBOL, "other_function", _LIBRARY_NATIVE, "does not match bound kernel"),
        (_SYMBOL, _SYMBOL, _LIBRARY_NATIVE + 1, "belongs to a different library"),
    ],
)
def test_exact_kernel_function_and_owner_identity_is_required(
    kernel_symbol, function_symbol, owner_native, error
):
    compiled, bound = _compiled_and_bound(wrapped=False)
    _Driver.kernel_symbol = kernel_symbol
    _Driver.function_symbol = function_symbol
    _Driver.owner_library_native = owner_native
    with pytest.raises(RuntimeError, match=error):
        _cooperative._query_compiled_specialization(
            compiled,
            bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )


def test_private_adapter_rejects_lookalikes_and_extra_wrapper_layers():
    exact_compiled, exact_bound = _compiled_and_bound(wrapped=False)
    with pytest.raises(RuntimeError, match="no immediate __wrapped__ target"):
        _cooperative._query_compiled_specialization(
            _ContractWrapper(),
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    compiled_lookalike = type(
        "CudaDialectJitCompiledFunction",
        (),
        {"__module__": "adversarial.lookalike"},
    )()
    with pytest.raises(RuntimeError, match="compiled target is not exact"):
        _cooperative._query_compiled_specialization(
            compiled_lookalike,
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    same_identity_compiled_impostor = type(
        "CudaDialectJitCompiledFunction",
        (),
        {"__module__": "cutlass.cutlass_dsl.cuda_jit_executor"},
    )()
    with pytest.raises(RuntimeError, match="compiled target is not exact"):
        _cooperative._query_compiled_specialization(
            same_identity_compiled_impostor,
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    same_identity_wrapper_impostor_type = type(
        "_ContractCheckedCallable",
        (),
        {"__module__": "nixl.device.cute.memory"},
    )
    same_identity_wrapper_impostor = same_identity_wrapper_impostor_type()
    same_identity_wrapper_impostor.__wrapped__ = exact_compiled
    with pytest.raises(RuntimeError, match="compiled target is not exact"):
        _cooperative._query_compiled_specialization(
            same_identity_wrapper_impostor,
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    with pytest.raises(RuntimeError, match="compiled target is not exact"):
        _cooperative._query_compiled_specialization(
            _wrapped(_wrapped(exact_compiled)),
            exact_bound,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    bound_lookalike = type("JitExecutor", (), {"__module__": "adversarial.lookalike"})()
    with pytest.raises(RuntimeError, match="bound target is not exact"):
        _cooperative._query_compiled_specialization(
            exact_compiled,
            bound_lookalike,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )

    same_identity_bound_impostor = type(
        "JitExecutor", (), {"__module__": "cutlass.base_dsl.jit_executor"}
    )()
    with pytest.raises(RuntimeError, match="bound target is not exact"):
        _cooperative._query_compiled_specialization(
            exact_compiled,
            same_identity_bound_impostor,
            driver=_Driver,
            device_ordinal=0,
            block_threads=256,
            planned_ctas=1,
            dynamic_smem_bytes=0,
        )


@pytest.mark.parametrize("wrapped", [False, True])
def test_bind_preserves_raw_or_contract_wrapped_to_result(monkeypatch, wrapped):
    raw_bound = _BoundTarget()
    returned_bound = _wrapped(raw_bound) if wrapped else raw_bound
    calls = []

    def bind(_self, device_ordinal):
        calls.append(device_ordinal)
        return returned_bound

    raw_compiled = _CompiledTarget()
    raw_compiled.to = lambda device_ordinal: bind(raw_compiled, device_ordinal)
    if wrapped:
        compiled = _wrapped(raw_compiled)
        compiled.to = lambda device_ordinal: bind(raw_compiled, device_ordinal)
    else:
        compiled = raw_compiled

    driver = types.ModuleType("cuda.bindings.driver")
    driver.CUresult = _Result
    driver.cuInit = lambda _flags: (0,)
    bindings = types.ModuleType("cuda.bindings")
    bindings.__path__ = []
    bindings.driver = driver
    cuda = types.ModuleType("cuda")
    cuda.__path__ = []
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        device=lambda _ordinal: nullcontext(),
        current_stream=lambda _ordinal: object(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        _cooperative.importlib.metadata,
        "version",
        lambda _distribution: _cooperative.PINNED_CUTE_DSL_VERSION,
    )
    observed = {}

    def query(compiled_value, bound_value, **kwargs):
        observed.update(compiled=compiled_value, bound=bound_value, kwargs=kwargs)
        return "occupancy"

    monkeypatch.setattr(_cooperative, "_query_compiled_specialization", query)
    result, occupancy = _cooperative.bind_and_validate_cooperative_launch(
        compiled,
        device_ordinal=0,
        block_threads=32,
        planned_ctas=64,
    )
    assert result is returned_bound
    assert occupancy == "occupancy"
    assert observed["compiled"] is compiled
    assert observed["bound"] is returned_bound
    assert calls == [0]


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("device_ordinal", -1),
        ("block_threads", 0),
        ("planned_ctas", True),
        ("dynamic_smem_bytes", -1),
    ],
)
def test_bind_validation_rejects_invalid_dimensions_before_cuda(keyword, value):
    arguments = {
        "device_ordinal": 0,
        "block_threads": 256,
        "planned_ctas": 1,
        "dynamic_smem_bytes": 0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        _cooperative.bind_and_validate_cooperative_launch(object(), **arguments)
