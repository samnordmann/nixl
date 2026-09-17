# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL JIT view over a host-prepared NIXL device memory view."""

from __future__ import annotations

import ctypes
import inspect
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Protocol, cast

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import ir

from ... import nixl_device_view_handle
from ._mlir import LLVMPtr

_UINT64_MAX = (1 << 64) - 1


class MemoryView:
    """Opaque device pointer adapted from ``nixl_device_view_handle``.

    A host-mode instance keeps the owning handle (and therefore its NIXL
    agent and descriptors) alive for an invocation. The application must not
    release the handle, deregister its memory, invalidate remote metadata, or
    destroy the agent until all kernels using the view have completed.

    Locality and descriptor lengths are compile-time state, not kernel ABI
    fields. :func:`compile` retains that state and rejects a substituted handle
    unless its kind, descriptor count, and validated extents are compatible.
    """

    def __init__(self, handle: nixl_device_view_handle):
        if not isinstance(handle, nixl_device_view_handle):
            raise TypeError(
                "MemoryView expects a nixl.nixl_device_view_handle, "
                f"got {type(handle).__name__}"
            )
        if not handle.is_valid:
            raise RuntimeError("cannot adapt a released NIXL device memory view")
        if handle.kind not in {"local", "remote"}:
            raise ValueError(f"unsupported NIXL device view kind: {handle.kind!r}")
        descriptor_lengths = tuple(handle.descriptor_lengths)
        if not descriptor_lengths or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 < length <= _UINT64_MAX
            for length in descriptor_lengths
        ):
            raise ValueError(
                "NIXL device view descriptor lengths must be uint64-sized "
                "positive integers"
            )
        self._handle: nixl_device_view_handle | None = handle
        self._ptr: ir.Value | None = None
        self._kind = handle.kind
        self._descriptor_lengths = descriptor_lengths
        # CuTe's host ABI wants the address of argument storage, not the device
        # pointer value itself. Keep this descriptor alive through invocation.
        self._c_value: ctypes.c_void_p | None = ctypes.c_void_p(handle.handle)
        self._c_pointer: int | None = ctypes.addressof(self._c_value)

    def __c_pointers__(self) -> list[int]:
        if self._handle is None or self._c_pointer is None:
            raise cutlass.DSLRuntimeError(
                "MemoryView.__c_pointers__ is only available in host mode"
            )
        if not self._handle.is_valid:
            raise RuntimeError("NIXL device memory view was released before launch")
        return [self._c_pointer]

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return [LLVMPtr.mlir_type()]

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self.ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> MemoryView:
        obj = object.__new__(type(self))
        obj._handle = None
        obj._ptr = values[0]
        obj._kind = self._kind
        obj._descriptor_lengths = self._descriptor_lengths
        obj._c_value = None
        obj._c_pointer = None
        return obj

    @property
    def ptr(self) -> ir.Value:
        """Opaque ``nixlMemViewH`` pointer, available while tracing."""
        if self._ptr is None:
            raise cutlass.DSLRuntimeError(
                "MemoryView.ptr is only available while tracing CuTe DSL code"
            )
        return self._ptr

    @property
    def kind(self) -> str:
        """Compile-time locality, ``"local"`` or ``"remote"``."""
        return self._kind

    @property
    def descriptor_lengths(self) -> Optional[tuple[int, ...]]:
        """Compile-time descriptor byte lengths, if supplied by the prototype."""
        return self._descriptor_lengths


@cutlass.register_jit_arg_adapter(nixl_device_view_handle)
def _adapt_device_view(handle: nixl_device_view_handle) -> MemoryView:
    return MemoryView(handle)


class _CompileOnlyMemoryView:
    """Typed prototype with a deliberately invalid execution pointer."""

    _real_cls: ClassVar[type] = MemoryView

    def __init__(
        self, kind: str, descriptor_lengths: Optional[tuple[int, ...]]
    ) -> None:
        self._kind = kind
        self._descriptor_lengths = descriptor_lengths

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return self._real_cls

    def __get_mlir_types__(self):
        return self._real_cls.__get_mlir_types__()

    def __new_from_mlir_values__(self, values):
        prototype = object.__new__(self._real_cls)
        prototype._kind = self._kind
        prototype._descriptor_lengths = self._descriptor_lengths
        return self._real_cls.__new_from_mlir_values__(prototype, values)

    def __c_pointers__(self) -> list[object]:
        # A slot-preserving non-pointer makes accidental invocation fail in
        # ctypes argument packing, before a kernel is launched. Returning []
        # would silently shift every following ABI argument.
        return [_INVALID_EXECUTION_POINTER]

    def __repr__(self) -> str:
        return "<compile-only fake MemoryView>"


_INVALID_EXECUTION_POINTER = object()


@dataclass(frozen=True, slots=True)
class _MemoryViewContract:
    argument_index: int
    argument_name: str
    kind: str
    descriptor_lengths: Optional[tuple[int, ...]]


class _ExecutionArguments(Protocol):
    """Subset of CuTe runtime argument metadata used by the contract guard."""

    signature: inspect.Signature

    def get_rectified_args(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[object, ...]:
        raise NotImplementedError

    def get_rectified_args_from_original_args(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[object, ...]:
        raise NotImplementedError


class _CompiledCallable(Protocol):
    """Callable surface exposed by CuTe compiled functions and executors."""

    def __call__(self, *args: object, **kwargs: object) -> Any:
        raise NotImplementedError

    def generate_execution_args(
        self, *args: object, **kwargs: object
    ) -> tuple[list[Any], list[Any]]:
        raise NotImplementedError

    def to(self, *args: object, **kwargs: object) -> _CompiledCallable:
        raise NotImplementedError


_CUTE_COMPILE_CONTROL_KWARGS = frozenset(
    {
        "_name_prefix",
        "compile_only",
        "gpu_module_attrs",
        "no_cache",
        "no_jit_engine",
        "options",
        "pipeline",
    }
)


def make_fake_memory_view(
    kind: str, descriptor_lengths: Optional[tuple[int, ...]] = None
) -> MemoryView:
    """Return a type-only ``MemoryView`` argument for ``cute.compile``.

    Args:
        kind: Required ``"local"`` or ``"remote"`` compile-time locality.
        descriptor_lengths: Optional positive descriptor byte lengths, used to
            reject statically out-of-bounds device operations while tracing.

    The fake is invalid for execution. Its sentinel pointer preserves the ABI
    slot and makes traditional ctypes invocation fail before kernel launch.
    Use :func:`compile` to retain this prototype as a runtime launch contract,
    then invoke the result with a live :class:`nixl_device_view_handle`
    (adapted automatically) or compatible MemoryView.
    """
    if kind not in {"local", "remote"}:
        raise ValueError("kind must be 'local' or 'remote'")
    if descriptor_lengths is not None:
        descriptor_lengths = tuple(descriptor_lengths)
        if not descriptor_lengths or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 < length <= _UINT64_MAX
            for length in descriptor_lengths
        ):
            raise ValueError(
                "descriptor_lengths must contain uint64-sized positive integers"
            )
    return cast(MemoryView, _CompileOnlyMemoryView(kind, descriptor_lengths))


def _is_memory_view_prototype(value: object) -> bool:
    return (
        type(value) is _CompileOnlyMemoryView
        or isinstance(value, MemoryView)
        or isinstance(value, nixl_device_view_handle)
    )


def _prototype_contract(
    value: object, argument_index: int, argument_name: str
) -> Optional[_MemoryViewContract]:
    if type(value) is _CompileOnlyMemoryView:
        kind = value._kind
        descriptor_lengths = value._descriptor_lengths
    elif isinstance(value, MemoryView):
        kind = value._kind
        descriptor_lengths = value._descriptor_lengths
    elif isinstance(value, nixl_device_view_handle):
        # Reuse the public adapter validation, but retain only immutable launch
        # metadata rather than the prototype or its owning handle.
        prototype = MemoryView(value)
        kind = prototype.kind
        descriptor_lengths = prototype.descriptor_lengths
    else:
        return None
    return _MemoryViewContract(
        argument_index,
        argument_name,
        kind,
        descriptor_lengths,
    )


def _full_compile_arguments(
    function: object, prototype_args: tuple[object, ...]
) -> tuple[object, ...]:
    # Match CuTe CompileCallable's treatment of bound methods and callable
    # instances. ``self``/``cls`` is filtered from the runtime signature.
    if inspect.ismethod(function):
        return (function.__self__, *prototype_args)
    if not inspect.isfunction(function) and callable(function):
        return (function, *prototype_args)
    return prototype_args


def _memory_view_contracts(
    compiled: object,
    function: object,
    prototype_args: tuple[object, ...],
    prototype_kwargs: dict[str, object],
) -> tuple[tuple[_MemoryViewContract, ...], _ExecutionArguments | None]:
    function_kwargs = {
        name: value
        for name, value in prototype_kwargs.items()
        if name not in _CUTE_COMPILE_CONTROL_KWARGS
    }
    if not any(
        _is_memory_view_prototype(value) for value in prototype_args
    ) and not any(
        _is_memory_view_prototype(value) for value in function_kwargs.values()
    ):
        return (), None

    execution_args = getattr(compiled, "execution_args", None)
    required_execution_metadata = (
        "signature",
        "get_rectified_args",
        "get_rectified_args_from_original_args",
    )
    if execution_args is None or any(
        not hasattr(execution_args, name) for name in required_execution_metadata
    ):
        raise RuntimeError(
            "CuTe compiled callable does not expose the runtime signature needed "
            "to enforce NIXL MemoryView contracts"
        )
    typed_execution_args = cast(_ExecutionArguments, execution_args)
    runtime_prototypes = typed_execution_args.get_rectified_args_from_original_args(
        _full_compile_arguments(function, prototype_args), function_kwargs
    )
    argument_names = tuple(typed_execution_args.signature.parameters)
    if len(runtime_prototypes) != len(argument_names):
        raise RuntimeError(
            "CuTe returned inconsistent runtime argument metadata while binding "
            "NIXL MemoryView contracts"
        )
    contracts = tuple(
        contract
        for index, (name, value) in enumerate(zip(argument_names, runtime_prototypes))
        if (
            contract := _prototype_contract(
                value,
                argument_index=index,
                argument_name=name,
            )
        )
        is not None
    )
    return contracts, typed_execution_args


def _live_view_metadata(
    value: object, argument_name: str
) -> tuple[str, tuple[int, ...]]:
    if type(value) is _CompileOnlyMemoryView:
        raise RuntimeError(
            f"{argument_name} is a compile-only fake MemoryView, not a live view"
        )
    if isinstance(value, MemoryView):
        handle = value._handle
        if handle is None:
            raise RuntimeError(
                f"{argument_name} is a tracing-only MemoryView, not a live view"
            )
    elif isinstance(value, nixl_device_view_handle):
        handle = value
    else:
        raise TypeError(
            f"{argument_name} must be a live nixl_device_view_handle or MemoryView, "
            f"got {type(value).__name__}"
        )

    if not handle.is_valid:
        raise RuntimeError(
            f"{argument_name} NIXL device memory view was released before launch"
        )
    kind = handle.kind
    descriptor_lengths = tuple(handle.descriptor_lengths)
    if kind not in {"local", "remote"}:
        raise ValueError(
            f"{argument_name} has unsupported NIXL device view kind {kind!r}"
        )
    if not descriptor_lengths or any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 0 < length <= _UINT64_MAX
        for length in descriptor_lengths
    ):
        raise ValueError(
            f"{argument_name} has invalid NIXL device view descriptor lengths"
        )
    return kind, descriptor_lengths


def _validate_memory_view_contract(
    contract: _MemoryViewContract, value: object
) -> None:
    kind, descriptor_lengths = _live_view_metadata(value, contract.argument_name)
    if kind != contract.kind:
        raise ValueError(
            f"{contract.argument_name} NIXL device view kind is {kind!r}; "
            f"compiled prototype requires {contract.kind!r}"
        )
    required = contract.descriptor_lengths
    if required is None:
        return
    if len(descriptor_lengths) != len(required):
        raise ValueError(
            f"{contract.argument_name} NIXL device view has "
            f"{len(descriptor_lengths)} descriptors; compiled prototype requires "
            f"exactly {len(required)}"
        )
    for descriptor_index, (actual, minimum) in enumerate(
        zip(descriptor_lengths, required)
    ):
        if actual < minimum:
            raise ValueError(
                f"{contract.argument_name} NIXL descriptor {descriptor_index} has "
                f"{actual} bytes; compiled prototype requires at least {minimum}"
            )


class _ContractCheckedCallable:
    """Host-only guard around a CuTe compiled callable or bound executor."""

    def __init__(
        self,
        target: _CompiledCallable,
        contracts: tuple[_MemoryViewContract, ...],
        execution_args: _ExecutionArguments,
    ) -> None:
        self._target = target
        self._contracts = contracts
        self._execution_args = execution_args
        self.__wrapped__ = target

    def _validate(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        argument_count = len(self._execution_args.signature.parameters)
        base_args = args[:argument_count]
        if not kwargs and len(base_args) == argument_count:
            runtime_args = base_args
        else:
            runtime_args = self._execution_args.get_rectified_args(base_args, kwargs)
        for contract in self._contracts:
            _validate_memory_view_contract(
                contract, runtime_args[contract.argument_index]
            )

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self._validate(args, kwargs)
        return self._target(*args, **kwargs)

    def generate_execution_args(
        self, *args: object, **kwargs: object
    ) -> tuple[list[Any], list[Any]]:
        self._validate(args, kwargs)
        return self._target.generate_execution_args(*args, **kwargs)

    def to(self, *args: object, **kwargs: object) -> _ContractCheckedCallable:
        return type(self)(
            self._target.to(*args, **kwargs),
            self._contracts,
            self._execution_args,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __repr__(self) -> str:
        return f"<NIXL MemoryView contract guard for {self._target!r}>"


def compile(
    function: object, *prototype_args: object, **prototype_kwargs: object
) -> Any:
    """Compile with host-side contracts for every NIXL memory-view prototype.

    This is a drop-in call-form replacement for ``cute.compile``. A returned
    callable validates each live NIXL view's locality, exact descriptor count,
    minimum descriptor extents, and lifetime before CuTe adapts or marshals any
    launch pointer. The check applies to direct calls and executors returned by
    ``.to()``; it emits no device code and adds nothing to the kernel ABI.
    """

    compiled = cute.compile(function, *prototype_args, **prototype_kwargs)
    contracts, execution_args = _memory_view_contracts(
        compiled,
        function,
        prototype_args,
        prototype_kwargs,
    )
    if not contracts:
        return compiled
    if execution_args is None:
        raise AssertionError("NIXL MemoryView contracts require execution metadata")
    return _ContractCheckedCallable(
        cast(_CompiledCallable, compiled), contracts, execution_args
    )


__all__ = ["MemoryView", "compile", "make_fake_memory_view"]
