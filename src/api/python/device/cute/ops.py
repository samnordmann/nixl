# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level NIXL operations for CuTe DSL device code."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cutlass

from . import _bindings
from .memory import MemoryView
from .types import Flags, Scope

_PUT: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.put_thread_wait,
    Scope.WARP: _bindings.put_warp_wait,
}
_PUT_POST: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.put_thread_post,
    Scope.WARP: _bindings.put_warp_post,
}
_ATOMIC_ADD: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.atomic_add_thread_wait,
    Scope.WARP: _bindings.atomic_add_warp_wait,
}
_ATOMIC_ADD_POST: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.atomic_add_thread_post,
    Scope.WARP: _bindings.atomic_add_warp_post,
}
_WAIT_ACQUIRE_SYSTEM_U64: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_thread,
    Scope.WARP: _bindings.wait_acquire_system_u64_warp,
}
_WAIT_ACQUIRE_GPU_U64: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_gpu_u64_thread,
    Scope.WARP: _bindings.wait_acquire_gpu_u64_warp,
}
_WAIT_ACQUIRE_GPU_U64_OR_ABORT: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_gpu_u64_or_abort_thread,
    Scope.WARP: _bindings.wait_acquire_gpu_u64_or_abort_warp,
}
_WAIT_ACQUIRE_SYSTEM_U64_OR_ABORT: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_or_abort_thread,
    Scope.WARP: _bindings.wait_acquire_system_u64_or_abort_warp,
}
_WAIT_ACQUIRE_SYSTEM_U64_OR_ABORTS: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_or_aborts_thread,
    Scope.WARP: _bindings.wait_acquire_system_u64_or_aborts_warp,
}
_WAIT_ACQUIRE_SYSTEM_U64_UNTIL: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_thread_until,
    Scope.WARP: _bindings.wait_acquire_system_u64_warp_until,
}
_WAIT_ACQUIRE_SYSTEM_U64_FOR: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_thread_for,
    Scope.WARP: _bindings.wait_acquire_system_u64_warp_for,
}
_WAIT_ACQUIRE_SYSTEM_U64_FOR_OR_ABORT: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_system_u64_thread_for_or_abort,
    Scope.WARP: _bindings.wait_acquire_system_u64_warp_for_or_abort,
}
_WAIT_ACQUIRE_GPU_U64_FOR: dict[Scope, Callable[..., Any]] = {
    Scope.THREAD: _bindings.wait_acquire_gpu_u64_thread_for,
    Scope.WARP: _bindings.wait_acquire_gpu_u64_warp_for,
}
_UINT32_MAX = (1 << 32) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1


def _extern_i32(value):
    """Give a signless LLVM extern result CuTe's signed i32 value type."""
    # CuTe DSL 4.5.1 preserves a cached extern result as signless ``i32`` on a
    # later compilation. A same-width constructor is intentionally a no-op;
    # crossing signedness through scalar bitcast materializes the DSL type.
    # LLVM bitcast is representation-preserving; retained-CUBIN qualification
    # guards this boundary against generated-code regressions.
    return cutlass.Uint32(value).bitcast(cutlass.Int32)


def _extern_u64(value):
    """Give a signless LLVM extern result CuTe's unsigned u64 value type."""
    # Preserve the public uint64 timer/counter contract without a numerical
    # conversion; this mirrors the signed retag above at 64-bit width.
    return cutlass.Int64(value).bitcast(cutlass.Uint64)


def _operation(binding: dict[Scope, Callable[..., Any]], scope: Scope):
    if not isinstance(scope, Scope):
        raise TypeError("scope must be a compile-time nixl.device.cute.Scope")
    if scope in (Scope.BLOCK, Scope.GRID):
        raise NotImplementedError(
            f"Scope.{scope.name} is not supported by the UCX-backed ABI; "
            "use THREAD or WARP"
        )
    return binding[scope]


def _wait_flags(value: Flags) -> cutlass.Uint64:
    if not isinstance(value, Flags):
        raise TypeError("flags must be a compile-time nixl.device.cute.Flags")
    if value != Flags.NONE:
        raise ValueError(
            "synchronous NIXL CuTe operations require Flags.NONE; DEFER needs "
            "a later non-deferred post and is not supported by the wait API"
        )
    return cutlass.Uint64(int(value))


def _post_flags(value: Flags) -> cutlass.Uint64:
    if not isinstance(value, Flags):
        raise TypeError("flags must be a compile-time nixl.device.cute.Flags")
    if value not in (Flags.NONE, Flags.DEFER):
        raise ValueError("post operations accept only Flags.NONE or Flags.DEFER")
    return cutlass.Uint64(int(value))


def _unsigned(name: str, value, dtype, maximum: int = _UINT64_MAX):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an unsigned integer, not bool")
    if isinstance(value, int) and not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return dtype(value)


def _view(name: str, value: MemoryView, expected_kind: str) -> MemoryView:
    if not isinstance(value, MemoryView):
        raise TypeError(f"{name} must be a nixl.device.cute.MemoryView")
    if value.kind != expected_kind:
        raise ValueError(
            f"{name} must be a {expected_kind} NIXL device view, " f"got {value.kind!r}"
        )
    return value


def _index(name: str, view: MemoryView, value):
    converted = _unsigned(name, value, cutlass.Uint32, _UINT32_MAX)
    lengths = view.descriptor_lengths
    if isinstance(value, int) and lengths is not None and value >= len(lengths):
        raise IndexError(
            f"{name} {value} is out of range for {len(lengths)} descriptors"
        )
    return converted


def _validate_span(
    name: str, view: MemoryView, index, offset, size, *, require_size: int | None = None
) -> None:
    if isinstance(offset, int) and isinstance(size, int):
        if size > _UINT64_MAX - offset:
            raise ValueError(f"{name} span end overflows uint64")
    if require_size is not None and isinstance(offset, int):
        if require_size > _UINT64_MAX - offset:
            raise ValueError(f"{name} span end overflows uint64")
    lengths = view.descriptor_lengths
    if lengths is None or not isinstance(index, int) or index >= len(lengths):
        return
    length = lengths[index]
    if isinstance(offset, int) and offset > length:
        raise ValueError(f"{name}_offset {offset} exceeds descriptor length {length}")
    if require_size is not None and isinstance(offset, int):
        if require_size > length - offset:
            raise ValueError(
                f"{name} needs {require_size} bytes at offset {offset}, "
                f"but descriptor length is {length}"
            )
    elif isinstance(offset, int) and isinstance(size, int) and size > length - offset:
        raise ValueError(
            f"{name} span [{offset}, {offset + size}) exceeds descriptor "
            f"length {length}"
        )


def put(
    local: MemoryView,
    remote: MemoryView,
    size,
    *,
    local_index=0,
    local_offset=0,
    remote_index=0,
    remote_offset=0,
    channel=0,
    flags: Flags = Flags.NONE,
    scope: Scope = Scope.THREAD,
):
    """Copy bytes from a local view to a remote view and wait for completion.

    ``scope`` is a compile-time specialization. Every thread in the selected
    warp must execute the call with identical arguments and without divergent
    control flow. Completion covers this NIXL request; this function
    does not add a CUDA block barrier before or after the operation. Literal
    indices, offsets, and sizes are bounds-checked against the view prototype;
    dynamic DSL values remain the application's responsibility.

    Returns a ``cutlass.Int32`` NIXL status; the wait ABI never returns
    ``Status.IN_PROGRESS``.
    """
    fn = _operation(_PUT, scope)
    local = _view("local", local, "local")
    remote = _view("remote", remote, "remote")
    local_index_value = _index("local_index", local, local_index)
    remote_index_value = _index("remote_index", remote, remote_index)
    local_offset_value = _unsigned("local_offset", local_offset, cutlass.Uint64)
    remote_offset_value = _unsigned("remote_offset", remote_offset, cutlass.Uint64)
    size_value = _unsigned("size", size, cutlass.Uint64)
    if isinstance(size, int) and size == 0:
        raise ValueError("size must be greater than zero")
    _validate_span("local", local, local_index, local_offset, size)
    _validate_span("remote", remote, remote_index, remote_offset, size)
    return _extern_i32(
        fn(
            local.ptr,
            local_index_value,
            local_offset_value,
            remote.ptr,
            remote_index_value,
            remote_offset_value,
            size_value,
            _unsigned("channel", channel, cutlass.Uint32, _UINT32_MAX),
            _wait_flags(flags),
        )
    )


def put_post(
    local: MemoryView,
    remote: MemoryView,
    size,
    *,
    local_index=0,
    local_offset=0,
    remote_index=0,
    remote_offset=0,
    channel=0,
    flags: Flags = Flags.NONE,
    scope: Scope = Scope.THREAD,
):
    """Post a PUT without allocating or polling a request.

    A successful post returns ``Status.IN_PROGRESS`` because local completion
    is deliberately not tracked. ``Flags.DEFER`` avoids ringing the transport
    doorbell; a later non-deferred operation on the same channel is mandatory.
    The production publish pattern is one or more deferred PUTs followed by an
    ordered completion atomic. The receiver may consume the data only after it
    observes that atomic.
    """
    fn = _operation(_PUT_POST, scope)
    local = _view("local", local, "local")
    remote = _view("remote", remote, "remote")
    local_index_value = _index("local_index", local, local_index)
    remote_index_value = _index("remote_index", remote, remote_index)
    local_offset_value = _unsigned("local_offset", local_offset, cutlass.Uint64)
    remote_offset_value = _unsigned("remote_offset", remote_offset, cutlass.Uint64)
    size_value = _unsigned("size", size, cutlass.Uint64)
    if isinstance(size, int) and size == 0:
        raise ValueError("size must be greater than zero")
    _validate_span("local", local, local_index, local_offset, size)
    _validate_span("remote", remote, remote_index, remote_offset, size)
    return _extern_i32(
        fn(
            local.ptr,
            local_index_value,
            local_offset_value,
            remote.ptr,
            remote_index_value,
            remote_offset_value,
            size_value,
            _unsigned("channel", channel, cutlass.Uint32, _UINT32_MAX),
            _post_flags(flags),
        )
    )


def atomic_add(
    remote: MemoryView,
    value,
    *,
    index=0,
    offset=0,
    channel=0,
    flags: Flags = Flags.NONE,
    scope: Scope = Scope.THREAD,
):
    """Atomically add a 64-bit value remotely and wait for completion.

    For WARP, all participating threads must execute the same call with
    identical arguments. The counter must be suitably aligned and belong
    to the prepared remote view. Literal positions are checked while tracing;
    dynamic DSL values remain the application's responsibility.

    Returns a ``cutlass.Int32`` NIXL status; the wait ABI never returns
    ``Status.IN_PROGRESS``.
    """
    fn = _operation(_ATOMIC_ADD, scope)
    remote = _view("remote", remote, "remote")
    index_value = _index("index", remote, index)
    offset_value = _unsigned("offset", offset, cutlass.Uint64)
    if isinstance(offset, int) and offset % 8:
        raise ValueError("offset must be 8-byte aligned for atomic_add")
    _validate_span("remote atomic", remote, index, offset, None, require_size=8)
    return _extern_i32(
        fn(
            _unsigned("value", value, cutlass.Uint64),
            remote.ptr,
            index_value,
            offset_value,
            _unsigned("channel", channel, cutlass.Uint32, _UINT32_MAX),
            _wait_flags(flags),
        )
    )


def atomic_add_post(
    remote: MemoryView,
    value,
    *,
    index=0,
    offset=0,
    channel=0,
    flags: Flags = Flags.NONE,
    scope: Scope = Scope.THREAD,
):
    """Post an ordered remote 64-bit atomic without local completion polling.

    A non-deferred atomic rings the selected channel and is visible only after
    preceding writes on that channel complete. The receiver can therefore use
    the counter as the completion condition for deferred PUTs. The caller must
    not interpret ``Status.IN_PROGRESS`` as local completion.
    """
    fn = _operation(_ATOMIC_ADD_POST, scope)
    remote = _view("remote", remote, "remote")
    index_value = _index("index", remote, index)
    offset_value = _unsigned("offset", offset, cutlass.Uint64)
    if isinstance(offset, int) and offset % 8:
        raise ValueError("offset must be 8-byte aligned for atomic_add_post")
    _validate_span("remote atomic", remote, index, offset, None, require_size=8)
    return _extern_i32(
        fn(
            _unsigned("value", value, cutlass.Uint64),
            remote.ptr,
            index_value,
            offset_value,
            _unsigned("channel", channel, cutlass.Uint32, _UINT32_MAX),
            _post_flags(flags),
        )
    )


def globaltimer_ns():
    """Read CUDA's system-wide nanosecond timer inside a kernel."""
    return _extern_u64(_bindings.globaltimer_ns())


def _aligned_address(address):
    if isinstance(address, int):
        if address == 0:
            raise ValueError("address must be nonzero")
        if address % 8:
            raise ValueError("address must be 8-byte aligned")
    return _unsigned("address", address, cutlass.Uint64)


def _aligned_i32_address(address):
    if isinstance(address, int):
        if address == 0:
            raise ValueError("address must be nonzero")
        if address % 4:
            raise ValueError("address must be 4-byte aligned")
    return _unsigned("address", address, cutlass.Uint64)


def load_acquire_system_u64(address):
    """Load an aligned GPU address with system-scope acquire semantics.

    Use this to consume a completion counter written by a remote NIXL atomic
    before reading the corresponding payload. ``address`` is normally the
    result of a CuTe pointer's ``toint()`` method.
    """
    return _extern_u64(_bindings.load_acquire_system_u64(_aligned_address(address)))


def load_acquire_gpu_u64(address):
    """Load a same-device counter with GPU-scope acquire semantics."""
    return _extern_u64(_bindings.load_acquire_gpu_u64(_aligned_address(address)))


def wait_acquire_system_u64(address, least, *, scope: Scope = Scope.THREAD):
    """Acquire-poll an aligned GPU counter until it reaches ``least``.

    The WARP specialization requires identical arguments and converged control
    flow from all 32 lanes. Only lane zero polls memory; the terminal value is
    broadcast to the warp before return. This is the preferred receive/credit
    wait for cooperative NIXL operations.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
        )
    )


def wait_acquire_gpu_u64(address, least, *, scope: Scope = Scope.THREAD):
    """Acquire-poll a same-GPU counter without reading ``%globaltimer``.

    Failed observations use relaxed GPU-scope loads; the successful value is
    reloaded with acquire semantics before returning. This is the production
    fast path for persistent cross-CTA handoffs whose liveness is guaranteed
    by the cooperative launch. Peer GPU or NIC publications require the
    system-scope wait instead. WARP callers must be converged.
    """
    fn = _operation(_WAIT_ACQUIRE_GPU_U64, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
        )
    )


def wait_acquire_gpu_u64_or_abort(
    address,
    least,
    abort_address,
    abort_least,
    *,
    scope: Scope = Scope.THREAD,
):
    """Acquire-poll a same-GPU target until ready or a shared abort.

    The returned target value is at least ``least`` on success and smaller on
    abort. The ready path is one acquire load. After a miss, target and abort
    observations use relaxed GPU-scope loads and acquire only on a terminal
    observation; the abort word is sampled once per 256 target misses. Both
    words must be monotonic and WARP callers must be converged.
    """
    fn = _operation(_WAIT_ACQUIRE_GPU_U64_OR_ABORT, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _aligned_address(abort_address),
            _unsigned("abort_least", abort_least, cutlass.Uint64),
        )
    )


def wait_acquire_system_u64_or_abort(
    address,
    least,
    abort_address,
    abort_least,
    *,
    scope: Scope = Scope.THREAD,
):
    """Acquire-poll a peer counter until ready or a local abort.

    The target uses system scope so peer-GPU and NIC publications are visible;
    the abort word uses cheaper GPU scope because it is local.  The healthy
    path is one target acquire.  After a miss, target loads are relaxed and the
    abort word is sampled once per 4096 target misses; terminal observations
    are reloaded with acquire semantics.  No ``%globaltimer`` instruction is
    executed.  A return below ``least`` reports abort.  Both words must be
    monotonic and WARP callers must be converged.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64_OR_ABORT, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _aligned_address(abort_address),
            _unsigned("abort_least", abort_least, cutlass.Uint64),
        )
    )


def wait_acquire_system_u64_or_aborts(
    address,
    least,
    local_abort_address,
    local_abort_least,
    peer_abort_address,
    peer_abort_least,
    *,
    scope: Scope = Scope.THREAD,
):
    """Acquire-poll a peer counter until ready or either abort fires.

    The target and peer abort use system scope; the local abort uses GPU scope.
    The healthy path is one target acquire. After a miss, both abort words are
    sampled once per 4096 target loads and terminal observations are acquired.
    No timer instruction is executed. A return below least reports abort. The
    peer abort must be published at system scope. WARP callers must converge.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64_OR_ABORTS, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _aligned_address(local_abort_address),
            _unsigned("local_abort_least", local_abort_least, cutlass.Uint64),
            _aligned_address(peer_abort_address),
            _unsigned("peer_abort_least", peer_abort_least, cutlass.Uint64),
        )
    )


def wait_acquire_system_u64_until(
    address, least, deadline_ns, *, scope: Scope = Scope.THREAD
):
    """Acquire-poll a GPU counter until ready or an absolute deadline.

    ``deadline_ns`` uses the same system-wide nanosecond clock returned by
    :func:`globaltimer_ns`. The returned value is at least ``least`` on
    success; a smaller value reports timeout without trapping or synchronizing
    the host. The implementation samples the timer only after each 256 failed
    counter loads, preserving the one-load healthy path while bounding an
    elastic peer-loss wait. WARP callers must be converged.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64_UNTIL, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _unsigned("deadline_ns", deadline_ns, cutlass.Uint64),
        )
    )


def wait_acquire_system_u64_for(
    address, least, timeout_ns, *, scope: Scope = Scope.THREAD
):
    """Acquire-poll with a relative device timeout and one-load fast path.

    If the first acquire load reaches ``least``, this variant never reads
    ``%globaltimer``. On a miss it starts the timeout on-device and amortizes
    subsequent timer reads over 256 failed loads. Prefer this form when the
    caller does not already have an absolute timestamp; use the ``_until`` form
    when it does. The returned value and WARP convergence contract are the same
    as :func:`wait_acquire_system_u64_until`.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64_FOR, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _unsigned("timeout_ns", timeout_ns, cutlass.Uint64),
        )
    )


def wait_acquire_system_u64_for_or_abort(
    address,
    least,
    abort_address,
    abort_least,
    timeout_ns,
    *,
    scope: Scope = Scope.THREAD,
):
    """Acquire-poll a peer counter until ready, local abort, or timeout.

    The healthy path is one system-scope acquire of ``address``. Only after a
    miss does the implementation read the device timer; remote failed polls
    are relaxed, and the same-GPU abort word is sampled once per 4096 misses.
    A return below ``least`` means timeout or abort. Acquire-load the abort word
    to distinguish them. WARP callers must be converged.
    """
    fn = _operation(_WAIT_ACQUIRE_SYSTEM_U64_FOR_OR_ABORT, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _aligned_address(abort_address),
            _unsigned("abort_least", abort_least, cutlass.Uint64),
            _unsigned("timeout_ns", timeout_ns, cutlass.Uint64),
        )
    )


def wait_acquire_gpu_u64_for(
    address, least, timeout_ns, *, scope: Scope = Scope.THREAD
):
    """Acquire-poll a counter produced and consumed on the same GPU.

    This has the relative timeout and one-load fast path of the system-scope
    variant, but deliberately uses GPU scope. It is suitable for cross-CTA
    coordination on one device. A counter written by a peer GPU or NIC must
    use :func:`wait_acquire_system_u64_for` instead.
    """
    fn = _operation(_WAIT_ACQUIRE_GPU_U64_FOR, scope)
    return _extern_u64(
        fn(
            _aligned_address(address),
            _unsigned("least", least, cutlass.Uint64),
            _unsigned("timeout_ns", timeout_ns, cutlass.Uint64),
        )
    )


def store_release_system_u64(address, value):
    """Store an aligned GPU address with system-scope release semantics.

    This is the publish operation for data copied through a locally mapped
    peer pointer. It is not a substitute for an ordered NIXL atomic after
    network PUTs.
    """
    return _extern_i32(
        _bindings.store_release_system_u64(
            _aligned_address(address),
            _unsigned("value", value, cutlass.Uint64),
        )
    )


def store_release_gpu_u64(address, value):
    """Release-publish a counter to CTAs on the same GPU.

    Use the system-scope form for any peer GPU or NIC consumer.
    """
    return _extern_i32(
        _bindings.store_release_gpu_u64(
            _aligned_address(address),
            _unsigned("value", value, cutlass.Uint64),
        )
    )


def atomic_add_release_gpu_u64(address, value):
    """Allocate from an aligned same-GPU u64 counter and return its old value.

    The atomic has GPU-scope release ordering. It is intended for device-local
    multi-writer allocators followed by a cooperative/device acquire boundary;
    it is not a NIXL transport atomic and must not target peer memory.
    """
    return _extern_u64(
        _bindings.atomic_add_release_gpu_u64(
            _aligned_address(address),
            _unsigned("value", value, cutlass.Uint64),
        )
    )


def atomic_max_release_gpu_u64(address, value):
    """Atomically publish a monotonic same-GPU epoch with release ordering.

    This is intended for rare multi-writer control paths such as cooperative
    kernel abort propagation. It is not a remote NIXL atomic.
    """
    return _extern_i32(
        _bindings.atomic_max_release_gpu_u64(
            _aligned_address(address),
            _unsigned("value", value, cutlass.Uint64),
        )
    )


def atomic_max_release_system_u64(address, value):
    """Atomically publish an owner-local abort epoch to mapped peer GPUs.

    This rare-path primitive is not a NIXL transport atomic and must not target
    peer memory.
    """
    return _extern_i32(
        _bindings.atomic_max_release_system_u64(
            _aligned_address(address),
            _unsigned("value", value, cutlass.Uint64),
        )
    )


def compare_exchange_status_gpu_i32(address, value):
    """Atomically record the first nonzero same-GPU status value.

    The compare/exchange writes ``value`` only while the aligned status word is
    zero and returns the prior value. This makes concurrent diagnostic writers
    race-free while preserving the first actionable failure.
    """
    if isinstance(value, bool):
        raise TypeError("value must be a signed integer, not bool")
    if isinstance(value, int) and not _INT32_MIN <= value <= _INT32_MAX:
        raise ValueError("value must fit in int32")
    return _extern_i32(
        _bindings.compare_exchange_status_gpu_i32(
            _aligned_i32_address(address), cutlass.Int32(value)
        )
    )


def sync_grid():
    """Synchronize every thread in a cooperatively launched grid.

    Every CTA and thread must execute this call in converged control flow, and
    the kernel must be launched with ``cooperative=True``. This operation never
    synchronizes the host.
    """
    return _extern_i32(_bindings.sync_grid())


def fence_release_system():
    """Order prior GPU writes before a following NIXL source read.

    Call this after a fused/persistent kernel writes a registered source buffer
    and before it posts a NIXL PUT that may be consumed by a NIC. A source made
    visible at an earlier CUDA operation boundary does not need this fence.
    The operation is device-only and never synchronizes the host.
    """
    return _extern_i32(_bindings.fence_release_system())


def get_ptr(view: MemoryView, index=0):
    """Return a mapped pointer for one remote descriptor, or null if unmapped.

    The returned ``!llvm.ptr`` is an address in the calling process. Its numeric
    value need not equal the descriptor owner's address in another process. Use
    this non-null base plus descriptor-relative offsets; never substitute a
    peer-advertised owner address or require numeric equality with it. Use
    ``cute.make_ptr`` to give the untyped pointer an element type before
    constructing a tensor, and do not outlive ``view``. A literal index is
    checked against the view prototype.
    """
    view = _view("view", view, "remote")
    return _bindings.get_ptr(view.ptr, _index("index", view, index))


def _mapped_copy_warp_ptr(binding, source_address, destination_address, size):
    source_value = _unsigned("source_address", source_address, cutlass.Uint64)
    destination_value = _unsigned(
        "destination_address", destination_address, cutlass.Uint64
    )
    size_value = _unsigned("size", size, cutlass.Uint64)
    for name, address in (
        ("source_address", source_address),
        ("destination_address", destination_address),
    ):
        if isinstance(address, int):
            if address == 0:
                raise ValueError(f"{name} must be nonzero")
            if address % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
    if isinstance(size, int):
        if size == 0:
            raise ValueError("size must be greater than zero")
        if size % 16:
            raise ValueError("size must be a multiple of 16 bytes")
        for name, address in (
            ("source", source_address),
            ("destination", destination_address),
        ):
            if isinstance(address, int) and address > _UINT64_MAX - size:
                raise ValueError(f"{name} span end overflows uint64")
    return _extern_i32(binding(source_value, destination_value, size_value))


def mapped_copy_warp_ptr(source_address, destination_address, size):
    """Copy between pre-resolved GPU addresses with a converged warp.

    This is the production inner-loop form: resolve a stable remote descriptor
    once with :func:`get_ptr`, broadcast its address, and pass the per-message
    destination here. It avoids descriptor lookup and pointer broadcast on
    every copy. Coherent 128-bit loads support source data produced or reused
    inside the same fused/persistent kernel. Addresses and size are 16-byte
    aligned and non-overlapping; publish completion separately after this call
    returns.

    ``destination_address`` must be derived from a non-null process-local
    :func:`get_ptr` result plus an offset inside that descriptor. Numeric
    equality with the allocation owner's address in another process is neither
    expected nor required. Protocols using system-scope atomics on peer GPU
    memory must qualify native atomics for every directed device pair before
    launch; the inner loop carries no topology or address-policy branch.
    """
    return _mapped_copy_warp_ptr(
        _bindings.mapped_copy_warp_ptr,
        source_address,
        destination_address,
        size,
    )


def mapped_copy_warp_ptr_readonly(source_address, destination_address, size):
    """Use the read-only-cache fast path between pre-resolved addresses.

    The source span must remain read-only for the entire current kernel. Do not
    use this form for fused packing or persistent source-buffer reuse.
    """
    return _mapped_copy_warp_ptr(
        _bindings.mapped_copy_warp_ptr_readonly,
        source_address,
        destination_address,
        size,
    )


def mapped_copy_warp(
    source_address,
    remote: MemoryView,
    size,
    *,
    remote_index=0,
    remote_offset=0,
):
    """Copy bytes directly to a mapped peer descriptor with one warp.

    All 32 lanes must call this operation in converged control flow with
    identical arguments. ``source_address`` is a raw GPU virtual address,
    normally obtained from a CuTe global-memory pointer's ``toint()`` method.
    Source, destination, offset, and size must be 16-byte aligned, the spans
    must not overlap, and literal spans are checked while tracing. Coherent
    global loads make this the safe choice when a fused or persistent kernel
    writes/reuses the source. Lane zero resolves the descriptor's process-local
    mapped base and every destination access uses that base plus
    ``remote_offset``; the owner's numeric address is irrelevant.

    The result is uniform across the warp. ``Status.SUCCESS`` means the payload
    stores completed the final warp rendezvous. ``Status.NOT_SUPPORTED`` means
    the remote descriptor is not locally mapped and the caller should take its
    ordinary NIXL PUT path. This operation never writes a completion signal;
    publish mapped writes separately with ``store_release_system_u64`` only
    after all related copies have completed.
    """
    remote = _view("remote", remote, "remote")
    index_value = _index("remote_index", remote, remote_index)
    source_value = _unsigned("source_address", source_address, cutlass.Uint64)
    offset_value = _unsigned("remote_offset", remote_offset, cutlass.Uint64)
    size_value = _unsigned("size", size, cutlass.Uint64)

    if isinstance(source_address, int):
        if source_address == 0:
            raise ValueError("source_address must be nonzero")
        if source_address % 16:
            raise ValueError("source_address must be 16-byte aligned")
    if isinstance(remote_offset, int) and remote_offset % 16:
        raise ValueError("remote_offset must be 16-byte aligned")
    if isinstance(size, int):
        if size == 0:
            raise ValueError("size must be greater than zero")
        if size % 16:
            raise ValueError("size must be a multiple of 16 bytes")
        if isinstance(source_address, int) and source_address > _UINT64_MAX - size:
            raise ValueError("source span end overflows uint64")
    _validate_span("remote mapped copy", remote, remote_index, remote_offset, size)

    return _extern_i32(
        _bindings.mapped_copy_warp(
            source_value,
            remote.ptr,
            index_value,
            offset_value,
            size_value,
        )
    )


def mapped_copy_warp_readonly(
    source_address,
    remote: MemoryView,
    size,
    *,
    remote_index=0,
    remote_offset=0,
):
    """Copy an immutable span through the non-coherent read-only cache.

    This is the highest-bandwidth mapped variant and has the same alignment,
    convergence, status, and publication contract as :func:`mapped_copy_warp`.
    Its additional hard precondition is that the complete source span remains
    read-only for the entire lifetime of the current kernel. In particular, do
    not use it for a buffer packed earlier in the same kernel or reused between
    iterations of a persistent kernel; use the coherent variant there.
    """
    remote = _view("remote", remote, "remote")
    index_value = _index("remote_index", remote, remote_index)
    source_value = _unsigned("source_address", source_address, cutlass.Uint64)
    offset_value = _unsigned("remote_offset", remote_offset, cutlass.Uint64)
    size_value = _unsigned("size", size, cutlass.Uint64)

    if isinstance(source_address, int):
        if source_address == 0:
            raise ValueError("source_address must be nonzero")
        if source_address % 16:
            raise ValueError("source_address must be 16-byte aligned")
    if isinstance(remote_offset, int) and remote_offset % 16:
        raise ValueError("remote_offset must be 16-byte aligned")
    if isinstance(size, int):
        if size == 0:
            raise ValueError("size must be greater than zero")
        if size % 16:
            raise ValueError("size must be a multiple of 16 bytes")
        if isinstance(source_address, int) and source_address > _UINT64_MAX - size:
            raise ValueError("source span end overflows uint64")
    _validate_span("remote mapped copy", remote, remote_index, remote_offset, size)

    return _extern_i32(
        _bindings.mapped_copy_warp_readonly(
            source_value,
            remote.ptr,
            index_value,
            offset_value,
            size_value,
        )
    )


__all__ = [
    "atomic_add",
    "atomic_add_post",
    "atomic_add_release_gpu_u64",
    "atomic_max_release_gpu_u64",
    "atomic_max_release_system_u64",
    "compare_exchange_status_gpu_i32",
    "fence_release_system",
    "get_ptr",
    "globaltimer_ns",
    "load_acquire_system_u64",
    "load_acquire_gpu_u64",
    "mapped_copy_warp",
    "mapped_copy_warp_ptr",
    "mapped_copy_warp_ptr_readonly",
    "mapped_copy_warp_readonly",
    "put",
    "put_post",
    "store_release_gpu_u64",
    "store_release_system_u64",
    "sync_grid",
    "wait_acquire_gpu_u64",
    "wait_acquire_gpu_u64_for",
    "wait_acquire_gpu_u64_or_abort",
    "wait_acquire_system_u64",
    "wait_acquire_system_u64_for",
    "wait_acquire_system_u64_for_or_abort",
    "wait_acquire_system_u64_or_abort",
    "wait_acquire_system_u64_or_aborts",
    "wait_acquire_system_u64_until",
]
