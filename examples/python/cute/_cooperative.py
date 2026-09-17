# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed occupancy preflight for cooperative CuTe DSL kernels.

CuTe DSL 4.5.1 has no public specialization-occupancy API. This adapter binds
a callable once, borrows the CUDA library already owned by that exact bound
specialization, resolves its CUfunction in the current context, queries
occupancy, and returns the same bound callable for launch. The callable may be
raw CuTe or the exact NIXL MemoryView contract wrapper, depending on its compile
prototype. No compiler artifact is retained and no second module is loaded.
Every private executor assumption is checked so a CuTe upgrade fails during
setup, never as an oversubscribed persistent launch.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import asdict, dataclass
from typing import Any

PINNED_CUTE_DSL_VERSION = "4.5.1"

_NIXL_CONTRACT_WRAPPER_TYPE = (
    "nixl.device.cute.memory",
    "_ContractCheckedCallable",
)
_COMPILED_TARGET_TYPE = (
    "cutlass.cutlass_dsl.cuda_jit_executor",
    "CudaDialectJitCompiledFunction",
)
_BOUND_TARGET_TYPE = ("cutlass.base_dsl.jit_executor", "JitExecutor")
_CUDA_DIALECT_MODULE_TYPE = (
    "cutlass.cutlass_dsl.cuda_jit_executor",
    "CudaDialectJitModule",
)
_CUDA_RUNTIME_LIBRARY_TYPE = ("cuda.bindings.runtime", "cudaLibrary_t")
_CUDA_DRIVER_LIBRARY_TYPE = ("cuda.bindings.driver", "CUlibrary")
_CUDA_DRIVER_KERNEL_TYPE = ("cuda.bindings.driver", "CUkernel")
_CUDA_DRIVER_FUNCTION_TYPE = ("cuda.bindings.driver", "CUfunction")
_CUDA_NATIVE_HANDLE_GETTER = ("cuda.bindings.utils", "get_cuda_native_handle")
_CUDA_BINDINGS_INT_HANDLE_MIN = (12, 8)
_CUDA_BINDINGS_NATIVE_HANDLE_MAJOR = 13


@dataclass(frozen=True, slots=True)
class CooperativeOccupancy:
    """Exact residency evidence for one bound kernel specialization."""

    device_ordinal: int
    kernel_symbol: str
    block_threads: int
    dynamic_smem_bytes: int
    registers_per_thread: int
    static_smem_bytes: int
    max_threads_per_block: int
    active_ctas_per_sm: int
    multiprocessor_count: int
    cooperative_cta_capacity: int
    planned_ctas: int
    cu_function_source: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _check_cuda(driver: Any, operation: str, result: tuple[Any, ...]) -> Any:
    status = result[0]
    if status != driver.CUresult.CUDA_SUCCESS:
        error_name = repr(status)
        name_result = driver.cuGetErrorName(status)
        if name_result[0] == driver.CUresult.CUDA_SUCCESS:
            raw_name = name_result[1]
            error_name = (
                raw_name.decode(errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
        raise RuntimeError(f"{operation} failed with {error_name}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _load_cute_target_types() -> tuple[type, type, type, type]:
    """Load the pinned private classes only after the accelerator stack exists."""

    try:
        from cutlass.base_dsl.jit_executor import (  # pylint: disable=import-outside-toplevel
            JitCompiledFunction,
            JitExecutor,
        )
        from cutlass.cutlass_dsl.cuda_jit_executor import (  # pylint: disable=import-outside-toplevel
            CudaDialectJitCompiledFunction,
            CudaDialectJitModule,
        )
        from nixl.device.cute.memory import (  # pylint: disable=import-outside-toplevel
            _ContractCheckedCallable,
        )
    except ImportError as error:
        raise RuntimeError(
            "cannot load the pinned CuTe/NIXL cooperative callable contract"
        ) from error

    identities = (
        (
            _ContractCheckedCallable,
            _NIXL_CONTRACT_WRAPPER_TYPE,
            "NIXL contract wrapper",
        ),
        (CudaDialectJitCompiledFunction, _COMPILED_TARGET_TYPE, "compiled target"),
        (JitExecutor, _BOUND_TARGET_TYPE, "bound target"),
        (CudaDialectJitModule, _CUDA_DIALECT_MODULE_TYPE, "bound CUDA module"),
    )
    for target_type, expected, label in identities:
        actual = (target_type.__module__, target_type.__name__)
        if actual != expected:
            raise RuntimeError(
                f"CuTe 4.5.1 {label} type identity drift: {actual!r} != "
                f"{expected!r}"
            )
    if not issubclass(CudaDialectJitCompiledFunction, JitCompiledFunction):
        raise RuntimeError(
            "CuTe 4.5.1 CUDA compiled target no longer derives from "
            "JitCompiledFunction"
        )
    return (
        _ContractCheckedCallable,
        CudaDialectJitCompiledFunction,
        JitExecutor,
        CudaDialectJitModule,
    )


def _resolve_cute_target(
    value: object,
    *,
    wrapper_type: type,
    expected_type: type,
    stage: str,
) -> object:
    """Resolve an exact raw CuTe target or one exact NIXL wrapper layer.

    ``nixl.device.cute.compile`` only creates ``_ContractCheckedCallable`` when
    at least one compile prototype is a NIXL ``MemoryView``. The mapped MoE
    dispatch/combine launchers use tensors and scalars and therefore return raw
    CuTe callables; persistent examples with a view prototype return the NIXL
    wrapper. Occupancy preflight supports both contracts without guessing from
    a convenient ``__wrapped__`` attribute or accepting a look-alike class.
    """

    if type(value) is wrapper_type:
        try:
            target = object.__getattribute__(value, "__wrapped__")
        except AttributeError as error:
            raise RuntimeError(
                f"NIXL {stage} contract wrapper has no immediate __wrapped__ target"
            ) from error
    else:
        target = value

    if type(target) is not expected_type:
        actual_type = type(target)
        raise RuntimeError(
            f"CuTe 4.5.1 {stage} target is not exact "
            f"{expected_type.__module__}.{expected_type.__name__}: "
            f"{actual_type.__module__}.{actual_type.__name__}"
        )
    return target


def _load_cuda_handle_contract() -> tuple[type, Any]:
    """Load the versioned runtime-handle bridge without pointer-storage APIs."""

    try:
        from cuda.bindings.runtime import (  # pylint: disable=import-outside-toplevel
            cudaLibrary_t,
        )
    except ImportError as error:
        raise RuntimeError(
            "cooperative occupancy preflight cannot load cuda.bindings.runtime"
        ) from error

    actual_type = (cudaLibrary_t.__module__, cudaLibrary_t.__name__)
    if actual_type != _CUDA_RUNTIME_LIBRARY_TYPE:
        raise RuntimeError(
            "CUDA runtime library handle identity drift: "
            f"{actual_type!r} != {_CUDA_RUNTIME_LIBRARY_TYPE!r}"
        )

    try:
        version = importlib.metadata.version("cuda-bindings")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("cuda-bindings distribution metadata is missing") from error
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.([0-9]+))?", version)
    if match is None:
        raise RuntimeError(f"unsupported cuda-bindings version {version!r}")
    release = tuple(int(component) for component in match.groups(default="0"))

    if release[0] == _CUDA_BINDINGS_INT_HANDLE_MIN[0]:
        if release[:2] < _CUDA_BINDINGS_INT_HANDLE_MIN:
            raise RuntimeError(f"unsupported cuda-bindings version {version!r}")
        # CUDA Python 12.x handle wrappers intentionally expose the native value
        # through __int__. Their getPtr() method instead returns the address of
        # the wrapper's handle-storage slot and must never be used here.
        return cudaLibrary_t, int
    if release[0] != _CUDA_BINDINGS_NATIVE_HANDLE_MAJOR:
        raise RuntimeError(f"unsupported cuda-bindings version {version!r}")

    try:
        from cuda.bindings.utils import (  # pylint: disable=import-outside-toplevel
            get_cuda_native_handle,
        )
    except ImportError as error:
        raise RuntimeError(
            "cuda-bindings 13.x requires get_cuda_native_handle"
        ) from error
    actual_getter = (
        get_cuda_native_handle.__module__,
        get_cuda_native_handle.__name__,
    )
    if actual_getter != _CUDA_NATIVE_HANDLE_GETTER:
        raise RuntimeError(
            "CUDA native-handle getter identity drift: "
            f"{actual_getter!r} != {_CUDA_NATIVE_HANDLE_GETTER!r}"
        )
    return cudaLibrary_t, get_cuda_native_handle


def _require_driver_handle_type(
    driver: Any, attribute: str, expected: tuple[str, str]
) -> type:
    """Return one exact cuda.bindings driver handle wrapper type."""

    handle_type = getattr(driver, attribute, None)
    if not isinstance(handle_type, type):
        raise RuntimeError(f"CUDA driver has no {attribute} handle type")
    actual = (handle_type.__module__, handle_type.__name__)
    if actual != expected:
        raise RuntimeError(
            f"CUDA {attribute} type identity drift: {actual!r} != {expected!r}"
        )
    return handle_type


def _native_handle(get_native_handle: Any, value: object, label: str) -> int:
    """Extract and validate a CUDA native handle value."""

    try:
        native = get_native_handle(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"cannot extract {label} native handle") from error
    if type(native) is not int or native <= 0:
        raise RuntimeError(
            f"{label} native handle must be a positive integer, got {native!r}"
        )
    return native


def _borrow_bound_cuda_library(
    compiled_target: object,
    bound_target: object,
    *,
    module_type: type,
    runtime_library_type: type,
    get_native_handle: Any,
    driver: Any,
) -> tuple[object, int]:
    """Borrow CuTe's live runtime library as a non-owning driver wrapper.

    CUDA runtime ``cudaLibrary_t`` and driver ``CUlibrary`` are handles to the
    same CUDA library object. The selected versioned bridge returns that object
    handle: CUDA Python 12.x intentionally exposes it through ``int(handle)``;
    13.x provides ``get_cuda_native_handle``. Neither path uses ``getPtr()``,
    which returns the address of Python's handle-storage slot. The returned
    driver wrapper is borrowed. Its lifetime remains owned by
    ``CudaDialectJitModule`` and this module never unloads it.
    """

    missing = object()
    exec_context = getattr(bound_target, "exec_context", missing)
    if exec_context is missing or exec_context is not None:
        raise RuntimeError(
            "CuTe 4.5.1 CUDA-dialect JitExecutor must have exec_context=None"
        )
    kernel_ptrs = getattr(bound_target, "_kernel_ptrs", missing)
    if kernel_ptrs is missing or kernel_ptrs is not None:
        raise RuntimeError(
            "CuTe 4.5.1 CUDA-dialect JitExecutor must have _kernel_ptrs=None"
        )

    compiled_module = getattr(compiled_target, "jit_module", missing)
    bound_module = getattr(bound_target, "jit_module", missing)
    if compiled_module is missing or bound_module is missing:
        raise RuntimeError("CuTe specialization has no bound JIT module")
    if compiled_module is not bound_module:
        raise RuntimeError(
            "compiled and bound CuTe callables do not share the same JIT module"
        )
    if type(bound_module) is not module_type:
        actual_type = type(bound_module)
        raise RuntimeError(
            "CuTe bound module is not exact "
            f"{module_type.__module__}.{module_type.__name__}: "
            f"{actual_type.__module__}.{actual_type.__name__}"
        )

    unloaded = bound_module.is_unloaded()
    if type(unloaded) is not bool:
        raise RuntimeError("CuTe bound CUDA module returned a non-boolean unload state")
    if unloaded:
        raise RuntimeError("CuTe bound CUDA module is already unloaded")
    runtime_libraries = getattr(bound_module, "cuda_library", missing)
    if type(runtime_libraries) is not list or len(runtime_libraries) != 1:
        count = len(runtime_libraries) if type(runtime_libraries) is list else "invalid"
        raise RuntimeError(
            "cooperative occupancy preflight requires one bound CUDA library; "
            f"got {count}"
        )
    runtime_library = runtime_libraries[0]
    if type(runtime_library) is not runtime_library_type:
        actual_type = type(runtime_library)
        raise RuntimeError(
            "CuTe bound library is not exact "
            f"{runtime_library_type.__module__}.{runtime_library_type.__name__}: "
            f"{actual_type.__module__}.{actual_type.__name__}"
        )

    native = _native_handle(
        get_native_handle, runtime_library, "CuTe runtime library"
    )
    driver_library_type = _require_driver_handle_type(
        driver, "CUlibrary", _CUDA_DRIVER_LIBRARY_TYPE
    )
    try:
        driver_library = driver_library_type(native)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "cannot construct a borrowed driver CUlibrary wrapper"
        ) from error
    if type(driver_library) is not driver_library_type:
        raise RuntimeError("CUDA CUlibrary constructor returned an unexpected type")
    if _native_handle(
        get_native_handle, driver_library, "borrowed driver library"
    ) != native:
        raise RuntimeError(
            "CUDA runtime-to-driver library handle round trip changed value"
        )
    return driver_library, native


def _decode_cuda_symbol(raw_symbol: object, operation: str) -> str:
    """Decode an exact CUDA-owned function name."""

    if isinstance(raw_symbol, bytes):
        try:
            symbol = raw_symbol.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"{operation} returned a non-UTF-8 symbol") from error
    elif isinstance(raw_symbol, str):
        symbol = raw_symbol
    else:
        raise RuntimeError(
            f"{operation} returned an invalid symbol type {type(raw_symbol).__name__}"
        )
    if not symbol or "\x00" in symbol:
        raise RuntimeError(f"{operation} returned an invalid symbol {symbol!r}")
    return symbol


def _query_compiled_specialization(
    compiled: object,
    bound: object,
    *,
    driver: Any,
    device_ordinal: int,
    block_threads: int,
    planned_ctas: int,
    dynamic_smem_bytes: int,
) -> CooperativeOccupancy:
    wrapper_type, compiled_type, bound_type, module_type = _load_cute_target_types()
    compiled_target = _resolve_cute_target(
        compiled,
        wrapper_type=wrapper_type,
        expected_type=compiled_type,
        stage="compiled",
    )
    bound_target = _resolve_cute_target(
        bound,
        wrapper_type=wrapper_type,
        expected_type=bound_type,
        stage="bound",
    )
    kernel_info = getattr(compiled_target, "kernel_info", None)
    if not isinstance(kernel_info, dict) or len(kernel_info) != 1:
        kernel_count = len(kernel_info) if isinstance(kernel_info, dict) else "invalid"
        raise RuntimeError(
            "cooperative occupancy preflight requires one exact compiled kernel; "
            f"got {kernel_count} kernel records"
        )
    expected_symbol = next(iter(kernel_info))
    if (
        type(expected_symbol) is not str
        or not expected_symbol
        or "\x00" in expected_symbol
    ):
        raise RuntimeError(
            "compiled specialization symbol must be a non-empty NUL-free string, "
            f"got {expected_symbol!r}"
        )
    runtime_library_type, get_native_handle = _load_cuda_handle_contract()

    current_device = _check_cuda(driver, "cuCtxGetDevice", driver.cuCtxGetDevice())
    if int(current_device) != device_ordinal:
        raise RuntimeError(
            f"current CUDA context is device {int(current_device)}, expected "
            f"{device_ordinal}"
        )
    cuda_device = _check_cuda(driver, "cuDeviceGet", driver.cuDeviceGet(device_ordinal))
    cooperative = _check_cuda(
        driver,
        "cuDeviceGetAttribute(COOPERATIVE_LAUNCH)",
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
            cuda_device,
        ),
    )
    if cooperative != 1:
        raise RuntimeError(f"CUDA device {device_ordinal} lacks cooperative launch")
    multiprocessors = _check_cuda(
        driver,
        "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            cuda_device,
        ),
    )
    library, library_native = _borrow_bound_cuda_library(
        compiled_target,
        bound_target,
        module_type=module_type,
        runtime_library_type=runtime_library_type,
        get_native_handle=get_native_handle,
        driver=driver,
    )
    kernel_type = _require_driver_handle_type(
        driver, "CUkernel", _CUDA_DRIVER_KERNEL_TYPE
    )
    function_type = _require_driver_handle_type(
        driver, "CUfunction", _CUDA_DRIVER_FUNCTION_TYPE
    )
    kernel = _check_cuda(
        driver,
        "cuLibraryGetKernel",
        driver.cuLibraryGetKernel(library, expected_symbol.encode("utf-8")),
    )
    if type(kernel) is not kernel_type:
        raise RuntimeError("cuLibraryGetKernel returned an unexpected handle type")
    raw_kernel_symbol = _check_cuda(
        driver, "cuKernelGetName", driver.cuKernelGetName(kernel)
    )
    kernel_symbol = _decode_cuda_symbol(raw_kernel_symbol, "cuKernelGetName")
    if kernel_symbol != expected_symbol:
        raise RuntimeError(
            f"bound CUDA kernel symbol {kernel_symbol!r} does not match compiled "
            f"specialization {expected_symbol!r}"
        )
    owning_library = _check_cuda(
        driver, "cuKernelGetLibrary", driver.cuKernelGetLibrary(kernel)
    )
    if type(owning_library) is not type(library):
        raise RuntimeError("cuKernelGetLibrary returned an unexpected handle type")
    if (
        _native_handle(
            get_native_handle, owning_library, "kernel owner library"
        )
        != library_native
    ):
        raise RuntimeError("resolved CUDA kernel belongs to a different library")

    function = _check_cuda(
        driver, "cuKernelGetFunction", driver.cuKernelGetFunction(kernel)
    )
    if type(function) is not function_type:
        raise RuntimeError("cuKernelGetFunction returned an unexpected handle type")
    raw_function_symbol = _check_cuda(
        driver, "cuFuncGetName", driver.cuFuncGetName(function)
    )
    function_symbol = _decode_cuda_symbol(raw_function_symbol, "cuFuncGetName")
    if function_symbol != kernel_symbol:
        raise RuntimeError(
            f"current-context CUDA function symbol {function_symbol!r} does not "
            f"match bound kernel {kernel_symbol!r}"
        )
    active_per_sm = _check_cuda(
        driver,
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, block_threads, dynamic_smem_bytes
        ),
    )

    def function_attribute(attribute: Any, label: str) -> int:
        return int(
            _check_cuda(
                driver,
                f"cuFuncGetAttribute({label})",
                driver.cuFuncGetAttribute(attribute, function),
            )
        )

    registers = function_attribute(
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, "NUM_REGS"
    )
    static_smem = function_attribute(
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        "SHARED_SIZE_BYTES",
    )
    max_threads = function_attribute(
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        "MAX_THREADS_PER_BLOCK",
    )
    active_per_sm = int(active_per_sm)
    multiprocessors = int(multiprocessors)
    capacity = active_per_sm * multiprocessors
    if block_threads > max_threads:
        raise RuntimeError(
            f"planned block has {block_threads} threads, kernel limit is {max_threads}"
        )
    if active_per_sm <= 0 or capacity <= 0:
        raise RuntimeError(
            f"kernel {function_symbol} has no resident cooperative CTA capacity"
        )
    if planned_ctas > capacity:
        raise RuntimeError(
            f"cooperative grid has {planned_ctas} CTAs but exact kernel capacity "
            f"is {active_per_sm}/SM * {multiprocessors} SMs = {capacity}"
        )
    return CooperativeOccupancy(
        device_ordinal=device_ordinal,
        kernel_symbol=function_symbol,
        block_threads=block_threads,
        dynamic_smem_bytes=dynamic_smem_bytes,
        registers_per_thread=registers,
        static_smem_bytes=static_smem,
        max_threads_per_block=max_threads,
        active_ctas_per_sm=active_per_sm,
        multiprocessor_count=multiprocessors,
        cooperative_cta_capacity=capacity,
        planned_ctas=planned_ctas,
        cu_function_source=(
            "bound_cuda_library->cuLibraryGetKernel(exact_kernel_info_symbol)"
            "->cuKernelGetFunction(current_context)"
        ),
    )


def bind_and_validate_cooperative_launch(
    compiled: object,
    *,
    device_ordinal: int,
    block_threads: int,
    planned_ctas: int,
    dynamic_smem_bytes: int = 0,
) -> tuple[object, CooperativeOccupancy]:
    """Bind, occupancy-check, and return the exact callable to launch."""

    for name, value, allow_zero in (
        ("device_ordinal", device_ordinal, True),
        ("block_threads", block_threads, False),
        ("planned_ctas", planned_ctas, False),
        ("dynamic_smem_bytes", dynamic_smem_bytes, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    actual_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if actual_version != PINNED_CUTE_DSL_VERSION:
        raise RuntimeError(
            "private CuTe executor occupancy adapter is pinned to "
            f"{PINNED_CUTE_DSL_VERSION}, installed version is {actual_version}"
        )

    import cuda.bindings.driver as driver  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    _check_cuda(driver, "cuInit", driver.cuInit(0))
    with torch.cuda.device(device_ordinal):
        # Force Torch's primary context current before CuTe binds its module.
        torch.cuda.current_stream(device_ordinal)
        bind = getattr(compiled, "to", None)
        if not callable(bind):
            raise RuntimeError("compiled CuTe callable has no callable .to() binder")
        bound = bind(device_ordinal)
        occupancy = _query_compiled_specialization(
            compiled,
            bound,
            driver=driver,
            device_ordinal=device_ordinal,
            block_threads=block_threads,
            planned_ctas=planned_ctas,
            dynamic_smem_bytes=dynamic_smem_bytes,
        )
    return bound, occupancy


__all__ = [
    "CooperativeOccupancy",
    "PINNED_CUTE_DSL_VERSION",
    "bind_and_validate_cooperative_launch",
]
