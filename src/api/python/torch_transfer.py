# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL provider for the experimental PyTorch endpoint-transfer API.

Call :func:`register_torch_backend` to register the ``nixl`` backend with
``torch.distributed._transfer``.  The module is intentionally not imported by
the normal ``import nixl`` path, so NIXL does not make PyTorch mandatory.

The default ``caller_ready`` CUDA ordering mode matches existing NIXL users:
the application makes memory ready before submission. PyTorch Core resolves
``host_wait`` before it enters serialized provider calls, so this module never
holds a NIXL lock while waiting on CUDA. Neither mode stages payloads.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import weakref
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from torch.distributed import _transfer as transfer


__all__ = [
    "NixlBackend",
    "TORCH_TRANSFER_FACTORY_API_VERSION",
    "register_torch_backend",
]


_DESCRIPTOR_VERSION = 1
_DEFAULT_MAX_PENDING_NOTIFICATIONS = 65_536
_DEFAULT_MAX_PENDING_NOTIFICATION_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_MATERIALIZED_INDICES = 16 * 1024 * 1024
# Cover the largest predeclared injection-depth cell without forcing callers to
# discover a hidden reuse cliff. Applications can lower this handle budget, or
# set it to zero, explicitly.
_DEFAULT_MAX_CACHED_REQUESTS_PER_PLAN = 64
_UINTPTR_BITS = np.dtype(np.uintp).itemsize * 8
_UINTPTR_MAX = (1 << _UINTPTR_BITS) - 1
_SIZE_T_MAX = _UINTPTR_MAX
_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT16_MAX = (1 << 16) - 1
_INT64_MAX = (1 << 63) - 1
_INT32_MAX = int(np.iinfo(np.int32).max)
_REQUEST_IDLE = "idle"
_REQUEST_LEASED = "leased"
_REQUEST_RETIRING = "retiring"
_REQUEST_RETIRED = "retired"
TORCH_TRANSFER_FACTORY_API_VERSION = 2
_REGISTRATION_LOCK = threading.Lock()
_REGISTERED = False


class _NoopLock:
    """Context-manager lock for Core modes that already serialize all entry."""

    __slots__ = ()

    def __enter__(self) -> "_NoopLock":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        return False


_NOOP_LOCK = _NoopLock()


class _BackendCallReservation:
    """Reusable MULTIPLE-call token whose weak identity is visible to close."""

    __slots__ = ("active", "reference", "__weakref__")

    def __init__(self) -> None:
        self.active = False
        self.reference = weakref.ref(self)


@dataclass(eq=False)
class _NixlRegistration:
    registration: object
    descriptor_list: object
    address: int
    nbytes: int
    device_id: int
    memory_type: str
    registration_started: bool = False
    native_registered: bool = False
    deregistration_receipt: object | None = None
    released: bool = False


@dataclass(eq=False)
class _NixlPeer:
    endpoint_id: str
    incarnation: str
    remote_name: str
    grants: tuple[_RemoteGrant, ...]
    native_import_started: bool = False
    released: bool = False
    notification_sender: object | None = None
    notification_send: Callable[[bytes], object] | None = None


@dataclass(frozen=True)
class _RemoteGrant:
    address: int
    nbytes: int
    device_id: int
    memory_type: str


@dataclass(frozen=True)
class _ResolvedRegion:
    region: transfer.BackendRegion
    peer: _NixlPeer | None
    base: int
    registered_nbytes: int
    device_id: int
    memory_type: str


@dataclass
class _NixlPeerState:
    remote_name: str
    payload_digest: bytes
    grants: tuple[_RemoteGrant, ...]
    live_peers: set[_NixlPeer] = field(default_factory=set)
    notification_sender: object | None = None
    notification_send: Callable[[bytes], object] | None = None

    @property
    def references(self) -> int:
        return len(self.live_peers)


@dataclass(slots=True)
class _PendingNotificationBatch:
    """One immutable native source batch plus a bounded egress cursor."""

    batch: transfer.BackendNotificationBatch
    retained_count: int
    retained_bytes: int
    offset: int = 0


@dataclass(eq=False, slots=True)
class _NixlRequest:
    """Plan-owned native request with an explicit single-lease state."""

    handle: object
    bytes_transferred: int
    reuse_token: object | None
    native_handle: object = field(init=False)
    notification_at_post: bool = False
    notification_present: bool | None = None
    state: str = _REQUEST_LEASED
    idle_epoch: int = 0

    def __post_init__(self) -> None:
        # Raw makeXferReqOwned returns a binding-owned RAII object with
        # ``value``; the compatibility API returns nixl_xfer_handle with
        # ``_handle``. Cache the integer once so every post/poll avoids wrapper
        # allocation and repeated attribute discovery.
        native_handle = getattr(self.handle, "_handle", None)
        if native_handle is None:
            native_handle = getattr(self.handle, "value", None)
            if callable(native_handle):
                native_handle = native_handle()
        self.native_handle = native_handle


@dataclass(eq=False)
class _NixlPlan:
    local_handle: object | None
    remote_handle: object | None
    peer: _NixlPeer
    local_regions: tuple[transfer.BackendRegion, ...]
    remote_regions: tuple[transfer.BackendRegion, ...]
    local_resolved: tuple[_ResolvedRegion, ...]
    remote_resolved: tuple[_ResolvedRegion, ...]
    local_starts: tuple[int, ...]
    remote_starts: tuple[int, ...]
    local_block_count: int
    remote_block_count: int
    local_total_nbytes: int
    remote_total_nbytes: int
    full_layout_matches: bool
    uniform_pair_nbytes: int | None
    local_full_nonoverlapping: bool
    remote_full_nonoverlapping: bool
    # Core supplies a stable, opaque identity for one immutable
    # op/indices/notification binding. Keeping idle requests behind that token
    # lets the provider repost native NIXL requests without rebuilding them or
    # re-hashing large selections on every transfer.
    requests: dict[int, _NixlRequest] = field(default_factory=dict)
    # Keys are id(token), and every lookup additionally checks ``is`` against
    # the strongly retained token on the request. This implements the SPI's
    # opaque-identity contract even for unhashable or adversarial tokens.
    cached_requests: dict[int, list[_NixlRequest]] = field(default_factory=dict)
    cached_request_count: int = 0
    idle_epoch: int = 0
    request_pool_dirty: bool = False
    # Persistent request slots retain this plan even while provider-idle.  A
    # per-plan identity registry makes the release guard O(1); the backend also
    # owns a global identity registry for endpoint cleanup.
    request_slots: dict[int, "NixlRequestSlot"] = field(default_factory=dict)
    local_all_indices: np.ndarray | None = None
    remote_all_indices: np.ndarray | None = None
    local_released: bool = False
    remote_released: bool = False
    release_started: bool = False
    released: bool = False


@dataclass(eq=False, slots=True)
class _SubmissionReservation:
    """MULTIPLE-only lifetime guard before submission Work is ready."""

    plan: _NixlPlan
    request: _NixlRequest | None = None
    restore_idle_on_abort: bool = False
    aborting: bool = False
    done: bool = False


class NixlWork(transfer.BackendWork):
    """Pollable wrapper around one posted NIXL transfer request."""

    __slots__ = (
        "_backend",
        "_bytes_transferred",
        "_error",
        "_handle",
        "_in_progress_status",
        "_lock",
        "_native_handle",
        "_pending_error",
        "_plan",
        "_poll_bounded_xfer",
        "_post_and_poll_xfer",
        "_post_xfer",
        "_released",
        "_request",
        "_reuse_token",
        "_state",
        "_status_xfer",
        "_submission_ready",
        "_success_status",
    )

    def __init__(
        self,
        backend: "NixlBackend",
        handle: object | None,
        *,
        state: transfer.WorkState,
        bytes_transferred: int = 0,
        error: BaseException | None = None,
        pending_error: BaseException | None = None,
        plan: _NixlPlan | None = None,
        reuse_token: object | None = None,
        request: _NixlRequest | None = None,
        submission_ready: bool = True,
    ) -> None:
        self._backend = backend
        self._handle = handle
        self._state = state
        self._bytes_transferred = bytes_transferred
        self._error = error
        self._pending_error = pending_error
        self._plan = plan
        self._reuse_token = reuse_token
        self._request = request
        self._submission_ready = submission_ready
        self._released = False
        if handle is None:
            self._post_xfer: Callable[[object], object] | None = None
            self._post_and_poll_xfer: Callable[..., object] | None = None
            self._poll_bounded_xfer: Callable[..., object] | None = None
            self._status_xfer: Callable[[object], object] | None = None
            self._native_handle: object | None = None
            self._success_status: object | None = None
            self._in_progress_status: object | None = None
        else:
            raw_post = backend._raw_post_xfer
            raw_status = backend._raw_get_xfer_status
            if raw_post is None or raw_status is None:
                agent = backend._require_agent()
                self._post_xfer = agent.transfer
                self._post_and_poll_xfer = None
                self._poll_bounded_xfer = None
                self._status_xfer = agent.check_xfer_state
                self._native_handle = handle
                self._success_status = "DONE"
                self._in_progress_status = "PROC"
            else:
                # The plan-owned high-level or binding-owned RAII handle is
                # retained for release, while hot calls use its cached integer.
                self._post_xfer = raw_post
                self._post_and_poll_xfer = backend._raw_post_xfer_and_poll
                self._poll_bounded_xfer = backend._raw_get_xfer_status_batch
                self._status_xfer = raw_status
                if request is None or request.native_handle is None:
                    raise transfer.BackendFailureError(
                        "NIXL request has no native handle value"
                    )
                self._native_handle = request.native_handle
                self._success_status = backend._raw_success
                self._in_progress_status = backend._raw_in_progress
        self._lock = (
            threading.RLock()
            if backend.thread_mode is transfer.ThreadMode.MULTIPLE
            else None
        )

    @property
    def state(self) -> transfer.WorkState:
        lock = self._lock
        if lock is None:
            self._poll_locked()
            return self._state
        with lock:
            self._poll_locked()
            return self._state

    @property
    def error(self) -> BaseException | None:
        lock = self._lock
        if lock is None:
            return self._error
        with lock:
            return self._error

    @property
    def result(self) -> int | None:
        lock = self._lock
        if lock is None:
            if self._state is transfer.WorkState.COMPLETED:
                return self._bytes_transferred
            return None
        with lock:
            if self._state is transfer.WorkState.COMPLETED:
                return self._bytes_transferred
            return None

    def test(self) -> bool:
        lock = self._lock
        if lock is None:
            self._poll_locked()
            return _is_terminal(self._state)
        with lock:
            self._poll_locked()
            return _is_terminal(self._state)

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> transfer.WorkState:
        """Keep a bounded status loop behind one raw pybind crossing."""

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a positive integer")
        if max_polls <= 0 or max_polls > _INT64_MAX:
            raise ValueError(f"max_polls must be in [1, {_INT64_MAX}]")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _INT64_MAX:
                raise ValueError(f"timeout_ns must be in [0, {_INT64_MAX}] or None")
        if isinstance(timeout_check_interval, bool) or not isinstance(
            timeout_check_interval, int
        ):
            raise TypeError("timeout_check_interval must be a positive integer")
        if timeout_check_interval <= 0 or timeout_check_interval > _INT64_MAX:
            raise ValueError(f"timeout_check_interval must be in [1, {_INT64_MAX}]")
        lock = self._lock
        if lock is None:
            return self._poll_bounded_locked(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )
        with lock:
            return self._poll_bounded_locked(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )

    def _poll_bounded_locked(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> transfer.WorkState:
        if _is_terminal(self._state):
            return self._state
        if self._pending_error is not None:
            self._finish_pending_failure_locked()
            return self._state
        if not self._submission_ready or self._handle is None:
            return self._state
        poll_bounded_xfer = self._poll_bounded_xfer
        if poll_bounded_xfer is None:
            deadline_ns = (
                None if timeout_ns is None else time.monotonic_ns() + timeout_ns
            )
            for poll_index in range(max_polls):
                self._poll_locked()
                if _is_terminal(self._state):
                    break
                if (
                    deadline_ns is not None
                    and (
                        poll_index == 0
                        or (poll_index + 1) % timeout_check_interval == 0
                    )
                    and time.monotonic_ns() >= deadline_ns
                ):
                    break
            return self._state
        try:
            status = poll_bounded_xfer(
                self._native_handle,
                max_polls,
                timeout_ns,
                timeout_check_interval,
            )
        except Exception as error:
            self._remember_failure(
                "NIXL transfer failed during bounded status polling", error
            )
            return self._state
        self._apply_polled_status_locked(status)
        return self._state

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        polls = 0
        while not self.test():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
            polls += 1
            if polls >= 16:
                delay = 0 if polls < 128 else 0.0001
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                time.sleep(delay)
        return True

    def cancel(self) -> bool:
        lock = self._lock
        if lock is None:
            return self._cancel_locked()
        with lock:
            return self._cancel_locked()

    def _cancel_locked(self) -> bool:
        self._poll_locked()
        if _is_terminal(self._state) or self._handle is None:
            return False
        # releaseXferReq() can either abort an in-progress request or discover
        # completion in the race before release. Its Python result does not
        # distinguish those outcomes, while Core permits True only for a
        # request known to be cancelled. Keep polling instead.
        return False

    def release(self) -> None:
        self._backend._release_work(self)

    close = release

    def _remember_failure(
        self, message: str, cause: BaseException | None = None
    ) -> None:
        error = transfer.BackendFailureError(message)
        if cause is not None:
            error.__cause__ = cause
        self._pending_error = error
        self._finish_pending_failure_locked()

    def _finish_pending_failure_locked(self) -> None:
        pending_error = self._pending_error
        if pending_error is None:
            return
        # The pending marker is authoritative even for a cold, not-yet-ready
        # Work. Publish readiness before the interruptible release so every
        # subsequent poll can drive this transaction to a terminal state.
        self._submission_ready = True
        handle = self._handle
        if handle is None and self._request is not None:
            # Recover a partially cleared Work from its plan-owned request.
            handle = self._request.handle
        release_interruption: BaseException | None = None
        try:
            if handle is not None:
                handle.release()
        except BaseException as error:
            # The native request may still be active (for example, a remote
            # metadata failure can race an in-flight operation). Keep all Core
            # lifetime anchors until a later release attempt succeeds.
            if not _handle_is_released(handle):
                self._state = transfer.WorkState.RUNNING
                if not isinstance(error, Exception):
                    raise
                return
            # Native ownership committed before the signal surfaced. Finish
            # terminal publication, then preserve asynchronous cancellation.
            if not isinstance(error, Exception):
                release_interruption = error

        # Error precedes the terminal-state commit. From this point an
        # interruption may leave cleanup fields populated, but cannot expose a
        # RUNNING Work backed by a released pointer.
        self._error = pending_error
        self._state = transfer.WorkState.FAILED
        self._handle = None
        self._post_xfer = None
        self._post_and_poll_xfer = None
        self._poll_bounded_xfer = None
        self._status_xfer = None
        self._native_handle = None
        self._success_status = None
        self._in_progress_status = None
        if self._request is not None:
            self._request.state = _REQUEST_RETIRED
            self._request = None
        self._pending_error = None
        if release_interruption is not None:
            raise release_interruption

    def _poll_locked(self) -> None:
        if _is_terminal(self._state):
            return
        if self._pending_error is not None:
            self._finish_pending_failure_locked()
            return
        if not self._submission_ready or self._handle is None:
            return
        try:
            status_xfer = self._status_xfer
            assert status_xfer is not None
            status = status_xfer(self._native_handle)
        except Exception as error:
            # The pinned NIXL binding throws for a negative native status. A
            # release attempt determines whether the request is safely
            # terminal; if not, retain ownership and retry on later polls.
            self._remember_failure("NIXL transfer failed while checking status", error)
            return
        self._apply_polled_status_locked(status)

    def _apply_polled_status_locked(self, status: object) -> None:
        if status == self._success_status:
            self._state = transfer.WorkState.COMPLETED
        elif status == self._in_progress_status:
            self._state = transfer.WorkState.RUNNING
        elif status == "ERR":
            self._remember_failure(f"NIXL returned transfer status {status!r}")
        else:
            self._remember_failure(f"NIXL returned unknown transfer status {status!r}")

    def _submission_succeeded(
        self,
        status: object,
    ) -> None:
        lock = self._lock
        if lock is None:
            self._submission_succeeded_locked(status)
            return
        with lock:
            self._submission_succeeded_locked(status)

    def _submission_succeeded_locked(self, status: object) -> None:
        if status == self._success_status:
            self._submission_ready = True
            self._state = transfer.WorkState.COMPLETED
            return
        if status == self._in_progress_status:
            self._submission_ready = True
            self._state = transfer.WorkState.RUNNING
            return
        self._submission_failed_locked(
            transfer.BackendFailureError("NIXL returned an invalid transfer status")
        )

    def _submission_failed(
        self,
        failure: transfer.BackendFailureError,
        cause: BaseException | None = None,
    ) -> None:
        if cause is not None:
            failure.__cause__ = cause
        lock = self._lock
        if lock is None:
            self._submission_failed_locked(failure)
            return
        with lock:
            self._submission_failed_locked(failure)

    def _submission_failed_locked(self, failure: BaseException) -> None:
        # Publish the retry marker before any native mutation. In particular,
        # cold adopted Work starts with submission_ready=False, so polls must
        # be able to resume cleanup after an interruption at every boundary.
        try:
            self._pending_error = failure
            self._submission_ready = True
            self._finish_pending_failure_locked()
        except BaseException:
            # submit/repost are no-raise-after-adoption operations. The marker,
            # handle ownership, and ready bit make a later poll/close retryable.
            if not _is_terminal(self._state):
                self._pending_error = failure
                self._submission_ready = True
                self._state = transfer.WorkState.RUNNING

    def _finish_synchronous(
        self,
        *,
        error: transfer.BackendFailureError | None = None,
        cause: BaseException | None = None,
    ) -> None:
        lock = self._lock
        if lock is None:
            self._finish_synchronous_locked(error=error, cause=cause)
            return
        with lock:
            self._finish_synchronous_locked(error=error, cause=cause)

    def _finish_synchronous_locked(
        self,
        *,
        error: transfer.BackendFailureError | None,
        cause: BaseException | None,
    ) -> None:
        if error is None:
            # Terminal state commits a handleless synchronous Work and remains
            # observable if readiness publication is interrupted immediately.
            self._state = transfer.WorkState.COMPLETED
        else:
            if cause is not None:
                error.__cause__ = cause
            self._pending_error = error
            self._error = error
            self._state = transfer.WorkState.FAILED
            self._pending_error = None
        self._submission_ready = True


class NixlRequestSlot(transfer.BackendRequestSlot):
    """Single-flight persistent wrapper around one repostable NIXL request.

    A successful generation deliberately leaves its :class:`NixlWork`, native
    request record, and prepared NIXL handle leased to this slot. A clean
    completion publishes one authoritative repostable receipt. Explicit
    ``recycle`` consumes only the active bit; Core may instead defer that
    successful recycle into the next ``start`` or ``close``. A failed,
    cancelled, running, or otherwise ambiguous generation never publishes the
    receipt and therefore cannot be folded into either boundary.
    """

    __slots__ = (
        "_active",
        "_backend",
        "_closed",
        "_dispatch_error",
        "_dispatch_error_epoch",
        "_execution_dispatch",
        "_failure",
        "_local_indices",
        "_notification",
        "_op",
        "_plan",
        "_remote_indices",
        "_repostable",
        "_work",
    )

    def __init__(
        self,
        backend: "NixlBackend",
        op: transfer.TransferOp,
        plan: _NixlPlan,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        execution_dispatch: object | None = None,
    ) -> None:
        self._backend = backend
        self._op = op
        self._plan: _NixlPlan | None = plan
        self._local_indices = (
            local_indices
            if local_indices is None or isinstance(local_indices, tuple)
            else tuple(local_indices)
        )
        self._remote_indices = (
            remote_indices
            if remote_indices is None or isinstance(remote_indices, tuple)
            else tuple(remote_indices)
        )
        self._notification = (
            notification
            if notification is None or isinstance(notification, bytes)
            else bytes(notification)
        )
        self._execution_dispatch = execution_dispatch
        self._dispatch_error_epoch = -1
        self._dispatch_error: transfer.BackendFailureError | None = None
        self._work: NixlWork | None = None
        self._failure: BaseException | None = None
        self._repostable = False
        self._active = False
        self._closed = False

    @property
    def state(self) -> transfer.WorkState | None:
        return self.poll_state()

    def poll_state(self) -> transfer.WorkState | None:
        """Poll through a bound method Core can cache on its hot path."""

        execution = self._execution_dispatch
        if execution is not None:
            return execution.poll_state()

        if self._failure is not None:
            self._repostable = False
            return transfer.WorkState.FAILED
        if not self._active:
            return None
        work = self._work
        if work is None:
            # Once start has acknowledged the generation, no Work and no
            # explicit failure can only mean an asynchronous interruption hit
            # the provider pre-launch prologue. Make that state terminal rather
            # than leaving Core with a permanently pending generation.
            self._failure = transfer.BackendFailureError(
                "NIXL persistent request start was interrupted before launch"
            )
            self._repostable = False
            return transfer.WorkState.FAILED
        # Core serializes calls to one BackendRequestSlot even in MULTIPLE;
        # avoid a duplicate Work lock while retaining concurrency across slots.
        work._poll_locked()
        self._repostable = work._state is transfer.WorkState.COMPLETED
        return work._state

    @property
    def error(self) -> BaseException | None:
        execution = self._execution_dispatch
        if execution is not None:
            if not execution.failed:
                return None
            epoch = execution.failure_epoch
            if epoch != self._dispatch_error_epoch:
                message = execution.failure_message
                self._dispatch_error = transfer.BackendFailureError(
                    message or "NIXL persistent request execution failed"
                )
                self._dispatch_error_epoch = epoch
            return self._dispatch_error
        failure = self._failure
        if failure is not None:
            return failure
        work = self._work
        if work is None:
            return None
        return work._error

    @property
    def completion_event(self) -> object | None:
        return None

    def start(self) -> None:
        # First start enters Backend.submit(), which protects shared request
        # registries. Reposts touch only slot-owned state and the native handle,
        # so a provider-global lock would unnecessarily serialize independent
        # MULTIPLE-mode slots while the pybind call has released the GIL.
        execution = self._execution_dispatch
        if execution is None:
            self._start_locked()
        else:
            execution.start()

    def start_with_notification(self, notification: bytes | None) -> None:
        """Post with an explicit per-generation notification or clear."""

        execution = self._execution_dispatch
        if execution is None:
            raise transfer.UnsupportedError(
                "the loaded NIXL binding has no request-slot notification "
                "override dispatcher"
            )
        execution.start_with_notification(notification)

    def start_and_poll(
        self, *, max_polls: int, timeout_ns: int | None
    ) -> transfer.WorkState:
        """Post once and perform a bounded native completion probe.

        The first generation uses the normal adopt-before-launch transaction.
        Later generations call the fused raw binding, keeping post and up to
        ``max_polls`` status probes behind one GIL release and one pybind
        crossing. Source-only and older-binding configurations retain the same
        contract through a bounded Python fallback.
        """

        # Core's steady execute path always supplies this exact pair. Keep the
        # complete direct-call contract on the slow branch without paying two
        # isinstance calls and range checks on every established generation.
        if (
            type(max_polls) is not int
            or (max_polls != 64 and max_polls != 0)
            or timeout_ns is not None
        ):
            if isinstance(max_polls, bool) or not isinstance(max_polls, int):
                raise TypeError("max_polls must be a non-negative int")
            if max_polls < 0 or max_polls > _INT64_MAX:
                raise ValueError(f"max_polls must be in [0, {_INT64_MAX}]")
            if timeout_ns is not None:
                if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                    raise TypeError("timeout_ns must be an int or None")
                if timeout_ns < 0 or timeout_ns > _INT64_MAX:
                    raise ValueError(f"timeout_ns must be in [0, {_INT64_MAX}] or None")

        execution = self._execution_dispatch
        if execution is not None:
            return execution.start_and_poll(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
            )

        work = self._work
        if work is not None and work._post_and_poll_xfer is not None:
            repostable = self._repostable
            if not repostable:
                # A clean receipt implies an open slot with a completed Work;
                # close invalidates it before release and then clears Work.
                # Keep lifecycle diagnostics on the exceptional slow branch.
                if self._closed:
                    raise transfer.ClosedError("NIXL persistent request slot is closed")
                if self._active:
                    raise transfer.BusyError(
                        "NIXL persistent request slot is already active"
                    )
            self._active = True
            try:
                if not repostable:
                    raise transfer.BackendFailureError(
                        "NIXL persistent request is not safely repostable"
                    )
                # The receipt is the replay barrier. Invalidate it before the
                # GIL-releasing native call; every later mutation remains in
                # this BaseException guard.
                self._repostable = False
                work._state = transfer.WorkState.RUNNING
                # Established Work stays submission-ready. Core serializes the
                # repost, and failure retirement checks `_pending_error` before
                # any later status observation.
                status = work._post_and_poll_xfer(
                    work._native_handle, max_polls, timeout_ns
                )
                if status == work._success_status:
                    work._state = transfer.WorkState.COMPLETED
                    self._repostable = True
                elif status != work._in_progress_status:
                    work._submission_failed_locked(
                        transfer.BackendFailureError(
                            "NIXL returned an invalid transfer status"
                        )
                    )
            except BaseException as error:
                self._repostable = False
                failure = _slot_failure(
                    "NIXL failed to repost and poll a persistent request", error
                )
                work._submission_failed_locked(failure)
            return work._state

        # Cold construction and compatibility fallback are not the steady
        # native fast path. Account their Python-side post time against a
        # finite budget and issue no more than the requested number of polls.
        deadline_ns = (
            None
            if timeout_ns is None or max_polls == 0
            else time.monotonic_ns() + timeout_ns
        )
        self._start_locked()
        failure = self._failure
        if failure is not None:
            return transfer.WorkState.FAILED
        work = self._work
        if work is None:
            self._failure = transfer.BackendFailureError(
                "NIXL did not acknowledge persistent request start"
            )
            return transfer.WorkState.FAILED
        state = self._poll_bounded_locked(work, max_polls, deadline_ns)
        self._repostable = state is transfer.WorkState.COMPLETED
        return state

    def start_and_poll_with_notification(
        self,
        notification: bytes | None,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> transfer.WorkState:
        """Post an explicit tag/clear and poll in one native boundary."""

        if (
            type(max_polls) is not int
            or (max_polls != 64 and max_polls != 0)
            or timeout_ns is not None
        ):
            if isinstance(max_polls, bool) or not isinstance(max_polls, int):
                raise TypeError("max_polls must be a non-negative int")
            if max_polls < 0 or max_polls > _INT64_MAX:
                raise ValueError(f"max_polls must be in [0, {_INT64_MAX}]")
            if timeout_ns is not None:
                if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                    raise TypeError("timeout_ns must be an int or None")
                if timeout_ns < 0 or timeout_ns > _INT64_MAX:
                    raise ValueError(f"timeout_ns must be in [0, {_INT64_MAX}] or None")

        execution = self._execution_dispatch
        if execution is None:
            raise transfer.UnsupportedError(
                "the loaded NIXL binding has no fused request-slot notification "
                "override dispatcher"
            )
        return execution.start_and_poll_with_notification(
            notification,
            max_polls=max_polls,
            timeout_ns=timeout_ns,
        )

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> transfer.WorkState | None:
        """Perform a bounded completion probe for one active generation.

        A rebuilt binding executes the entire batch behind one pybind crossing
        and one GIL release. Source-only and older-binding environments use the
        exact same observation and timeout contract through a Python fallback.
        """

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a positive int")
        if max_polls <= 0 or max_polls > _INT64_MAX:
            raise ValueError(f"max_polls must be in [1, {_INT64_MAX}]")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be an int or None")
            if timeout_ns < 0 or timeout_ns > _INT64_MAX:
                raise ValueError(f"timeout_ns must be in [0, {_INT64_MAX}] or None")
        if isinstance(timeout_check_interval, bool) or not isinstance(
            timeout_check_interval, int
        ):
            raise TypeError("timeout_check_interval must be a positive int")
        if timeout_check_interval <= 0 or timeout_check_interval > _INT64_MAX:
            raise ValueError(f"timeout_check_interval must be in [1, {_INT64_MAX}]")

        execution = self._execution_dispatch
        if execution is not None:
            return execution.poll_bounded(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )

        failure = self._failure
        if failure is not None:
            self._repostable = False
            return transfer.WorkState.FAILED
        if not self._active:
            return None
        work = self._work
        if work is None:
            self._failure = transfer.BackendFailureError(
                "NIXL persistent request start was interrupted before bounded poll"
            )
            self._repostable = False
            return transfer.WorkState.FAILED
        if work._pending_error is not None:
            self._repostable = False
            work._finish_pending_failure_locked()
            return work._state
        if work._state is not transfer.WorkState.RUNNING or not work._submission_ready:
            self._repostable = work._state is transfer.WorkState.COMPLETED
            return work._state

        poll_bounded_xfer = work._poll_bounded_xfer
        if poll_bounded_xfer is None:
            deadline_ns = (
                None if timeout_ns is None else time.monotonic_ns() + timeout_ns
            )
            state = self._poll_bounded_locked(
                work,
                max_polls,
                deadline_ns,
                timeout_check_interval=timeout_check_interval,
            )
            self._repostable = state is transfer.WorkState.COMPLETED
            return state

        # Core serializes calls to a given BackendRequestSlot. There is no
        # provider-global lock here, so independent MULTIPLE-mode slots can
        # execute their native polling loops concurrently while pybind has
        # released the GIL.
        try:
            status = poll_bounded_xfer(
                work._native_handle,
                max_polls,
                timeout_ns,
                timeout_check_interval,
            )
        except BaseException as error:
            self._repostable = False
            work._remember_failure(
                "NIXL transfer failed during bounded status polling", error
            )
            return work._state

        if status == work._success_status:
            work._state = transfer.WorkState.COMPLETED
            self._repostable = True
        elif status == work._in_progress_status:
            self._repostable = False
            work._state = transfer.WorkState.RUNNING
        else:
            self._repostable = False
            work._remember_failure(
                f"NIXL returned invalid bounded-poll status {status!r}"
            )
        return work._state

    def _start_locked(self) -> None:
        if self._closed:
            raise transfer.ClosedError("NIXL persistent request slot is closed")
        if self._active and not self._repostable:
            raise transfer.BusyError("NIXL persistent request slot is already active")

        self._active = True
        self._failure = None
        work = self._work
        if work is not None:
            # An extant Work implies the slot and plan are open: provider close
            # first closes every idle slot, and plan release is blocked by the
            # plan-local slot registry. Avoid rechecking those invariants on the
            # steady hot path.
            self._repost_locked(work)
            return

        self._repostable = False
        plan = self._plan
        try:
            self._backend._check_open()
            if plan is None or plan.release_started:
                raise transfer.ClosedError("NIXL transfer plan is closing or closed")
        except BaseException as error:
            # Core has entered start and therefore conservatively owns this
            # generation. Make even a late lifecycle/precondition failure
            # observable and recyclable instead of leaving Core waiting on an
            # idle-looking provider slot.
            self._failure = _slot_failure(
                "NIXL persistent request cannot be started", error
            )
            return
        assert plan is not None

        # The slot itself was adopted at construction. This existing
        # transaction additionally adopts its stable Work before native post,
        # and pays all lowering/allocation only on the untimed prime (or after
        # a retired failure).
        try:
            self._backend.submit(
                self._op,
                plan,
                local_indices=self._local_indices,
                remote_indices=self._remote_indices,
                notification=self._notification,
                reuse_token=None,
                adopt_work=self._adopt_work,
            )
        except BaseException as error:
            work = self._work
            if work is None:
                self._failure = _slot_failure(
                    "NIXL failed to initialize a persistent request", error
                )
            else:
                work._submission_failed(
                    _slot_failure(
                        "NIXL persistent request start was interrupted", error
                    )
                )
        if self._work is None and self._failure is None:
            self._failure = transfer.BackendFailureError(
                "NIXL did not adopt persistent request work"
            )

    def _adopt_work(self, work: transfer.BackendWork) -> None:
        if not isinstance(work, NixlWork):
            raise TypeError("NIXL request slot requires NixlWork")
        if self._work is not None:
            raise transfer.BusyError("NIXL request slot already owns work")
        self._work = work

    def _repost_locked(self, work: NixlWork) -> None:
        try:
            if not self._repostable:
                raise transfer.BackendFailureError(
                    "NIXL persistent request is not safely repostable"
                )
            # Invalidate before the GIL-releasing post. A signal or post error
            # can then only create a failed/ambiguous active generation; it can
            # never replay this completed receipt a second time.
            self._repostable = False
            work._state = transfer.WorkState.RUNNING
            work._error = None
            work._pending_error = None
            # This established Work is already submission-ready; calls to this
            # slot are serialized across the interruptible native post.
            post_xfer = work._post_xfer
            assert post_xfer is not None
            status = post_xfer(work._native_handle)
            if status == work._success_status:
                work._state = transfer.WorkState.COMPLETED
                self._repostable = True
            elif status != work._in_progress_status:
                work._submission_failed_locked(
                    transfer.BackendFailureError(
                        "NIXL returned an invalid transfer status"
                    )
                )
        except BaseException as error:
            self._repostable = False
            failure = _slot_failure("NIXL failed to repost a persistent request", error)
            work._submission_failed_locked(failure)

    @staticmethod
    def _poll_bounded_locked(
        work: NixlWork,
        max_polls: int,
        deadline_ns: int | None,
        *,
        timeout_check_interval: int = 1,
    ) -> transfer.WorkState:
        state = work._state
        poll = 0
        polls_until_timeout_check = max(1, timeout_check_interval - 1)
        while state is transfer.WorkState.RUNNING and poll < max_polls:
            work._poll_locked()
            state = work._state
            poll += 1
            if state is not transfer.WorkState.RUNNING or poll == max_polls:
                break
            if deadline_ns is None:
                continue
            check_timeout = poll == 1
            if not check_timeout:
                polls_until_timeout_check -= 1
                check_timeout = polls_until_timeout_check == 0
                if check_timeout:
                    polls_until_timeout_check = timeout_check_interval
            if check_timeout and time.monotonic_ns() >= deadline_ns:
                break
        return state

    def cancel(self) -> bool:
        execution = self._execution_dispatch
        if execution is not None:
            return execution.cancel()
        return self._cancel_locked()

    def _cancel_locked(self) -> bool:
        if self._closed or not self._active:
            return False
        # Clear the receipt atomically before cancellation observes native
        # state. NixlWork.cancel() is deliberately observational: it polls and
        # returns False rather than releasing an in-flight native request. A
        # clean completion that wins this race is therefore safe to publish
        # again. Any future destructive cancellation must leave RUNNING or a
        # terminal non-success state when its outcome is ambiguous; it must
        # never report COMPLETED without a repost-safe handle.
        self._repostable = False
        work = self._work
        if work is None:
            return False
        cancelled = work.cancel()
        if not cancelled and work._state is transfer.WorkState.COMPLETED:
            self._repostable = True
        return cancelled

    def recycle(self) -> None:
        execution = self._execution_dispatch
        if execution is not None:
            execution.recycle()
            return
        self._recycle_locked()

    def _recycle_locked(self) -> None:
        if self._closed:
            raise transfer.ClosedError("NIXL persistent request slot is closed")
        if not self._active:
            # Core may retry after an asynchronous interruption landed between
            # this provider commit and its own public-slot state commit.
            return
        if self._failure is not None:
            self._repostable = False
            self._failure = None
            self._active = False
            return

        work = self._work
        if work is None:
            # Retry window after failed-generation recycle cleared either its
            # synthetic failure or retired Work but was interrupted before the
            # final active-bit commit.
            self._repostable = False
            self._active = False
            return
        if work._state is transfer.WorkState.COMPLETED:
            # This is the steady-state fast path: no allocation, registry or
            # request-pool mutation, terminal-set membership test, release
            # call, or synchronization primitive.
            self._repostable = True
            self._active = False
            return
        if not _is_terminal(work._state):
            raise transfer.BusyError("cannot recycle active NIXL request work")

        # Failed/cancelled native handles are never eligible for repost. Work
        # only becomes terminal failure after NIXL release has proved native
        # access stopped, so this close is retry-safe and removes its registry
        # entry without publishing it to the ordinary request cache.
        self._repostable = False
        work.release()
        self._work = None
        self._active = False

    def close(self) -> None:
        lock = self._backend._lock
        if lock is _NOOP_LOCK:
            self._close_locked()
            return
        with lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        execution = self._execution_dispatch
        if execution is not None:
            execution.close()
            plan = self._plan
            if plan is not None:
                plan.request_slots.pop(id(self), None)
            self._backend._slots.pop(id(self), None)
            self._plan = None
            self._dispatch_error = None
            self._execution_dispatch = None
            self._closed = True
            return
        if self._active:
            work = self._work
            if (
                self._failure is not None
                or work is None
                or work._state is not transfer.WorkState.COMPLETED
                or not self._repostable
            ):
                raise transfer.BusyError("cannot close an active NIXL request slot")
            # Core may have made a successful execute generation logically idle
            # without calling recycle. Consume that receipt before releasing;
            # commit provider-idle first so an interrupted close remains retryable.
            self._active = False
        work = self._work
        if work is not None:
            # No reuse token was supplied when the slot created this Work, so
            # release retires its request instead of returning it to the plan's
            # ordinary token cache.
            self._repostable = False
            work.release()
            self._work = None
        plan = self._plan
        if plan is not None:
            plan.request_slots.pop(id(self), None)
        self._backend._slots.pop(id(self), None)
        self._plan = None
        self._failure = None
        self._repostable = False
        self._closed = True

    def _provider_active(self) -> bool:
        execution = self._execution_dispatch
        if execution is not None:
            return bool(execution.active)
        return self._active


class NixlBackend(transfer.Backend):
    """Backend implementation using the public :class:`nixl_agent` API.

    ``max_cached_requests_per_plan`` bounds idle, repostable native requests
    retained by each prepared plan (default: 64; zero disables caching).
    ``nixl_thread_sync_mode`` accepts ``"none"``, ``"strict"``, or ``"rw"``.
    It defaults to ``"none"`` when Core guarantees single-threaded or
    serialized provider entry and to ``"rw"`` for ``ThreadMode.MULTIPLE``.
    NIXL may safely upgrade ``"none"`` when a metadata-owned agent thread is
    enabled. UCX independently selects its worker thread mode for progress.
    """

    _CAPABILITIES = transfer.Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        raw_spans=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_start_poll=True,
        fused_request_slot_poll=True,
        deferred_successful_request_slot_recycle=True,
        grouped_notification_receive=True,
        synchronous_notification_send=True,
        cuda_ordering_modes=frozenset(
            {
                transfer.CudaOrderingMode.CALLER_READY,
                transfer.CudaOrderingMode.HOST_WAIT,
            }
        ),
        standalone_notification_completion=(
            transfer.NotificationCompletion.LOCAL_ACCEPTED
        ),
    )

    def __init__(
        self,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: transfer.ProgressMode,
        thread_mode: transfer.ThreadMode,
        options: Mapping[str, Any],
        *,
        _activate_backend: bool = True,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.name = name
        self.incarnation = incarnation
        self._native_name = _native_agent_name(endpoint_id, incarnation)
        self.progress_mode = progress_mode
        self.thread_mode = thread_mode
        self._options = dict(options)
        self._capabilities = self._CAPABILITIES
        # SINGLE is one-caller by definition. CALLER_SERIALIZED is protected by
        # the application's whole-graph boundary. SERIALIZED is protected by
        # Core's backend-entry lock. Retaking a provider-global RLock in those
        # modes is redundant. MULTIPLE uses short provider-side ownership
        # transactions, but native make/adopt/post operations run without this
        # global lock.
        self._lock = (
            threading.RLock()
            if thread_mode is transfer.ThreadMode.MULTIPLE
            else _NOOP_LOCK
        )
        self._notification_lock = (
            threading.RLock()
            if thread_mode is transfer.ThreadMode.MULTIPLE
            else _NOOP_LOCK
        )
        self._active_backend_calls: (
            set[weakref.ReferenceType[_BackendCallReservation]] | None
        ) = (set() if thread_mode is transfer.ThreadMode.MULTIPLE else None)
        self._backend_call_reservation_pool: list[_BackendCallReservation] | None = (
            [] if thread_mode is transfer.ThreadMode.MULTIPLE else None
        )
        self._available = False
        self._closing = False
        self._closed = False
        self._activation_started = False
        self._activated = False
        self._registrations: list[_NixlRegistration] = []
        self._plans: list[_NixlPlan] = []
        self._works: dict[int, NixlWork] = {}
        # Reservations exist only in MULTIPLE mode. The selected contract
        # excludes concurrent teardown in SINGLE/CALLER_SERIALIZED/SERIALIZED,
        # so their allocation-free submission path remains unchanged.
        self._submission_reservations: dict[int, _SubmissionReservation] | None = (
            {} if thread_mode is transfer.ThreadMode.MULTIPLE else None
        )
        self._slots: dict[int, NixlRequestSlot] = {}
        self._peers: list[_NixlPeer] = []
        self._peer_states: dict[tuple[str, str], _NixlPeerState] = {}
        self._peer_keys_by_name: dict[str, tuple[str, str]] = {}
        self._pending_notification_batches: deque[_PendingNotificationBatch] = deque()
        self._pending_notification_available_count = 0
        self._pending_notification_retained_count = 0
        self._pending_notification_bytes = 0
        self._notification_error: transfer.BackendFailureError | None = None
        self._notification_drain_dirty = False
        # Allocate the fail-stop object before any destructive poll. If a
        # second asynchronous exception interrupts detailed poison creation,
        # the already-published dirty bit can still raise this exact object
        # without allocating on the recovery path.
        self._notification_drain_error = transfer.BackendFailureError(
            "NIXL notification state is ambiguous after a destructive drain; "
            "the endpoint must be closed and recreated"
        )
        self._max_pending_notifications = _option_positive_int(
            self._options.get(
                "max_pending_notifications", _DEFAULT_MAX_PENDING_NOTIFICATIONS
            ),
            "max_pending_notifications",
        )
        self._max_pending_notification_bytes = _option_positive_int(
            self._options.get(
                "max_pending_notification_bytes",
                _DEFAULT_MAX_PENDING_NOTIFICATION_BYTES,
            ),
            "max_pending_notification_bytes",
        )
        self._max_materialized_indices = min(
            _option_positive_int(
                self._options.get(
                    "max_materialized_indices",
                    _DEFAULT_MAX_MATERIALIZED_INDICES,
                ),
                "max_materialized_indices",
            ),
            _INT32_MAX,
        )
        self._max_cached_requests_per_plan = _option_nonnegative_int(
            self._options.get(
                "max_cached_requests_per_plan",
                _DEFAULT_MAX_CACHED_REQUESTS_PER_PLAN,
            ),
            "max_cached_requests_per_plan",
        )
        self._nixl_thread_sync_mode = _nixl_thread_sync_mode(
            self._options.get("nixl_thread_sync_mode"), self.thread_mode
        )
        self._use_native_request_slot_execution = self._options.get(
            "use_native_request_slot_execution", True
        )
        if not isinstance(self._use_native_request_slot_execution, bool):
            raise TypeError("use_native_request_slot_execution must be a bool")
        self._orphan_dlists: list[object] = []
        self._orphan_remote_names: set[str] = set()
        self._configured_backends = _ucx_backends(
            self._options.get("backends", ("UCX",)), "backends"
        )
        self._transfer_backends = _ucx_backends(
            self._options.get("transfer_backends", self._configured_backends),
            "transfer_backends",
        )

        self._agent: object | None = None
        self._raw_post_xfer: Callable[..., object] | None = None
        self._raw_post_xfer_with_notification_override: Callable[..., object] | None = (
            None
        )
        self._raw_post_xfer_and_poll: Callable[..., object] | None = None
        self._raw_get_xfer_status_batch: Callable[..., object] | None = None
        self._raw_get_xfer_status: Callable[..., object] | None = None
        self._raw_notification_receiver: object | None = None
        self._raw_notification_sender_factory: Callable[..., object] | None = None
        self._raw_poll_notification_batches: Callable[[], object] | None = None
        self._raw_poll_notification_batches_bounded: Callable[..., object] | None = None
        self._raw_get_notification_batches: Callable[..., object] | None = None
        self._raw_get_notifications: Callable[..., object] | None = None
        self._raw_notification_buffer: dict[str, list[bytes]] = {}
        self._raw_request_slot_factory: Callable[..., object] | None = None
        self._raw_make_xfer_req_owned: Callable[..., object] | None = None
        self._raw_transfer_backend_handles: tuple[int, ...] | None = None
        self._raw_read_operation: object | None = None
        self._raw_write_operation: object | None = None
        self._raw_success: object | None = None
        self._raw_in_progress: object | None = None
        if not _activate_backend:
            return
        try:
            self._activate()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                _add_exception_note(
                    error,
                    f"NIXL backend activation cleanup failed: {cleanup_error!r}",
                )
            raise

    @classmethod
    def _new_unactivated(
        cls,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: transfer.ProgressMode,
        thread_mode: transfer.ThreadMode,
        options: Mapping[str, Any],
    ) -> NixlBackend:
        backend = cls.__new__(cls)
        # Initialize every recovery field without creating or publishing a
        # native agent. FactoryV2 transfers this exact record to Core first.
        backend._initialize_record(
            endpoint_id,
            name,
            incarnation,
            progress_mode,
            thread_mode,
            options,
        )
        return backend

    def _initialize_record(
        self,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: transfer.ProgressMode,
        thread_mode: transfer.ThreadMode,
        options: Mapping[str, Any],
    ) -> None:
        # Reuse the direct constructor's pure phase without duplicating the
        # field contract; its private flag returns before native activation.
        self.__init__(
            endpoint_id,
            name,
            incarnation,
            progress_mode,
            thread_mode,
            options,
            _activate_backend=False,
        )

    def _activate(self) -> None:
        if self._closed:
            raise transfer.ClosedError("cannot activate a closed NIXL backend")
        if self._activated:
            return
        if self._activation_started:
            raise transfer.BusyError("NIXL backend activation is already in progress")
        self._activation_started = True
        supplied_agent = self._options.get("_agent")
        if supplied_agent is not None:
            self._agent = supplied_agent
        else:
            self._agent = self._create_agent()
        raw_agent = getattr(self._agent, "agent", None)
        if raw_agent is not None:
            # Keep synthetic agents and source-only tests independent of the
            # compiled module. A real nixl_agent exposes its owning raw pybind
            # object here; cache its bound calls so every post/poll avoids the
            # high-level enum -> string -> WorkState round trip.
            from . import _bindings as nixl_bind

            if isinstance(raw_agent, nixl_bind.nixlAgent):
                # Resolve every immutable native request-creation input once.
                # This is independent of persistent-slot execution: a fresh
                # request can use the raw owned factory when no compiled slot
                # dispatcher is present (or that dispatcher is disabled).
                try:
                    raw_backend_handles = tuple(
                        self._agent.backends[name] for name in self._transfer_backends
                    )
                except (AttributeError, KeyError, TypeError):
                    raw_backend_handles = None
                self._raw_transfer_backend_handles = raw_backend_handles
                self._raw_read_operation = getattr(nixl_bind, "NIXL_READ", None)
                self._raw_write_operation = getattr(nixl_bind, "NIXL_WRITE", None)
                raw_make_xfer_req_owned = getattr(raw_agent, "makeXferReqOwned", None)
                if (
                    callable(raw_make_xfer_req_owned)
                    and raw_backend_handles is not None
                    and self._raw_read_operation is not None
                    and self._raw_write_operation is not None
                ):
                    self._raw_make_xfer_req_owned = raw_make_xfer_req_owned
                self._raw_post_xfer = raw_agent.postXferReq
                self._raw_post_xfer_with_notification_override = getattr(
                    raw_agent, "postXferReqWithNotifOverride", None
                )
                self._raw_post_xfer_and_poll = getattr(
                    raw_agent, "postXferReqAndPoll", None
                )
                self._raw_get_xfer_status_batch = getattr(
                    raw_agent, "getXferStatusBatch", None
                )
                self._raw_get_xfer_status = raw_agent.getXferStatus
                if raw_backend_handles is not None:
                    raw_notification_sender_factory = getattr(
                        raw_agent, "createNotifSender", None
                    )
                    if callable(raw_notification_sender_factory):
                        self._raw_notification_sender_factory = (
                            raw_notification_sender_factory
                        )
                    raw_notification_receiver_factory = getattr(
                        raw_agent, "createNotifReceiver", None
                    )
                    raw_get_notification_batches = getattr(
                        raw_agent, "getNotifsGrouped", None
                    )
                    raw_get_notifications = getattr(raw_agent, "getNotifs", None)
                    if callable(raw_notification_receiver_factory):
                        receiver = raw_notification_receiver_factory(
                            raw_backend_handles
                        )
                        poll = getattr(receiver, "poll", None)
                        if not callable(poll):
                            raise TypeError(
                                "NIXL notification receiver must expose poll()"
                            )
                        self._raw_notification_receiver = receiver
                        self._raw_poll_notification_batches = poll
                        bounded_poll = getattr(receiver, "poll_bounded", None)
                        if callable(bounded_poll):
                            self._raw_poll_notification_batches_bounded = bounded_poll
                    elif callable(raw_get_notification_batches):
                        self._raw_get_notification_batches = (
                            raw_get_notification_batches
                        )
                    elif callable(raw_get_notifications):
                        self._raw_get_notifications = raw_get_notifications
                self._raw_success = nixl_bind.NIXL_SUCCESS
                self._raw_in_progress = nixl_bind.NIXL_IN_PROG
                self._capabilities = replace(
                    self._capabilities,
                    fused_prevalidated_submit_poll=callable(
                        self._raw_post_xfer_and_poll
                    ),
                    fused_work_poll=callable(self._raw_get_xfer_status_batch),
                )
                if self._use_native_request_slot_execution:
                    self._raw_request_slot_factory = getattr(
                        raw_agent, "createXferRequestSlotExecution", None
                    )
                    if self._raw_request_slot_factory is not None:
                        execution_type = getattr(
                            nixl_bind, "nixlRequestSlotExecution", None
                        )
                        if (
                            execution_type is not None
                            and callable(
                                getattr(
                                    execution_type,
                                    "start_with_notification",
                                    None,
                                )
                            )
                            and callable(
                                getattr(
                                    execution_type,
                                    "start_and_poll_with_notification",
                                    None,
                                )
                            )
                        ):
                            self._capabilities = replace(
                                self._capabilities,
                                request_slot_notification_overrides=True,
                            )
        self._activated = True
        # Publish availability last. Every public/hot operation uses this as
        # its sole lifecycle gate.
        self._available = True

    @property
    def capabilities(self) -> transfer.Capabilities:
        return self._capabilities

    @property
    def supported_thread_modes(self) -> frozenset[transfer.ThreadMode]:
        return frozenset(
            {
                transfer.ThreadMode.SINGLE,
                transfer.ThreadMode.CALLER_SERIALIZED,
                transfer.ThreadMode.SERIALIZED,
                transfer.ThreadMode.MULTIPLE,
            }
        )

    def _create_agent(self) -> object:
        # Keep PyTorch integration out of the ordinary NIXL import path and
        # retain a test seam that does not require compiled plugins or hardware.
        from ._api import (
            DEFAULT_COMM_PORT,
            nixl_agent,
            nixl_agent_config,
            nixl_thread_sync_t,
        )

        configured_backends = self._configured_backends
        raw_init_params = self._options.get("backend_init_params")
        if raw_init_params is not None and not isinstance(raw_init_params, Mapping):
            raise TypeError("backend_init_params must be a mapping or None")
        init_params: dict[str, dict[str, str]] = {}
        for backend in configured_backends:
            values = {} if raw_init_params is None else raw_init_params.get(backend, {})
            if not isinstance(values, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in values.items()
            ):
                raise TypeError(
                    "backend_init_params values must map strings to strings"
                )
            init_params[backend] = dict(values)
        native_background = self.progress_mode is transfer.ProgressMode.BACKGROUND
        requested_background = self._options.get("enable_nixl_progress_thread")
        if requested_background is not None:
            requested_background = _option_bool(
                requested_background, "enable_nixl_progress_thread"
            )
            if requested_background is not native_background:
                raise ValueError(
                    "enable_nixl_progress_thread must match Endpoint progress_mode"
                )
        enable_listen_thread = _option_bool(
            self._options.get("enable_listen_thread", False), "enable_listen_thread"
        )
        listen_port = _option_bounded_nonnegative_int(
            self._options.get("listen_port", DEFAULT_COMM_PORT),
            "listen_port",
            _UINT16_MAX,
        )
        capture_telemetry = _option_bool(
            self._options.get("capture_telemetry", False), "capture_telemetry"
        )
        num_threads = _option_bounded_nonnegative_int(
            self._options.get("num_threads", 0), "num_threads", _UINT32_MAX
        )
        sync_modes = {
            "none": nixl_thread_sync_t.NIXL_THREAD_SYNC_NONE,
            "strict": nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
            "rw": nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        }
        config = nixl_agent_config(
            # BACKGROUND is provider-owned by contract, so use NIXL's native
            # progress thread. MANUAL leaves progression to explicit Core
            # progress, status, and notification polls.
            enable_prog_thread=native_background,
            enable_listen_thread=enable_listen_thread,
            listen_port=listen_port,
            capture_telemetry=capture_telemetry,
            num_threads=num_threads,
            # NIXL's config cannot carry per-backend initialization parameters.
            # When they are requested, instantiate the plugins explicitly below.
            backends=[] if raw_init_params is not None else configured_backends,
            # Core/caller policy already enforces single entry for
            # SINGLE/CALLER_SERIALIZED/SERIALIZED. Avoid redundant native agent
            # locking there; MULTIPLE selects RW by default.
            # NIXL itself upgrades NONE if a metadata listener needs a thread.
            sync_mode=sync_modes[self._nixl_thread_sync_mode],
        )
        custom_factory = "_agent_factory" in self._options
        factory = self._options.get("_agent_factory", nixl_agent)
        agent = None
        try:
            agent = factory(self._native_name, config)
            if raw_init_params is not None:
                for backend in configured_backends:
                    agent.create_backend(backend, init_params[backend])
            if not custom_factory and "UCX" not in getattr(agent, "backends", {}):
                raise transfer.UnsupportedError(
                    "NIXL UCX backend is unavailable; the experimental provider "
                    "cannot satisfy its advertised capabilities"
                )
            return agent
        except BaseException as error:
            # Backend construction can publish the native agent name and start
            # threads before Python regains control. Explicitly close custom
            # agents when possible, then drop the last local reference so the
            # pybind-owned nixlAgent destructor retracts that publication.
            if agent is not None:
                close = getattr(agent, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as cleanup_error:
                        _add_exception_note(
                            error,
                            f"unpublished NIXL agent cleanup failed: {cleanup_error!r}",
                        )
                agent = None
            raise

    def register(
        self,
        registration: transfer.RegisteredMemory,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        with self._lock:
            self._check_open()
            address = _positive_int_attr(registration, "address")
            nbytes = _positive_int_attr(registration, "nbytes")
            _validate_native_span(address, nbytes, "registration")
            device = str(getattr(registration, "device"))
            memory_type = _normalize_memory_type(
                str(getattr(registration, "memory_type", device.split(":", 1)[0]))
            )
            device_id = _device_id(device)
            agent = self._require_agent()
            descriptors = [(address, nbytes, device_id, "")]
            try:
                descriptor_list = agent.get_reg_descs(descriptors, mem_type=memory_type)
            except Exception as error:
                raise transfer.BackendFailureError(
                    "NIXL failed to create registration descriptors"
                ) from error
            handle = _NixlRegistration(
                registration=registration,
                descriptor_list=descriptor_list,
                address=address,
                nbytes=nbytes,
                device_id=device_id,
                memory_type=memory_type,
            )
            # Pre-track the allocation owner and native descriptor before the
            # GIL-releasing registration mutator can make memory accessible.
            self._registrations.append(handle)
            try:
                adopt_handle(handle)
            except BaseException:
                _discard_identity(self._registrations, handle)
                raise
            handle.registration_started = True
            try:
                registered_list = agent.register_memory(
                    descriptor_list,
                    mem_type=memory_type,
                    backends=self._transfer_backends,
                )
                if registered_list is not descriptor_list:
                    raise RuntimeError(
                        "NIXL replaced a preconstructed registration descriptor list"
                    )
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                raise transfer.BackendFailureError(
                    "NIXL failed to register memory"
                ) from error
            handle.native_registered = True

    def deregister(self, handle: object) -> None:
        registration = _require_type(handle, _NixlRegistration, "registration")
        with self._lock:
            if registration.released:
                _discard_identity(self._registrations, registration)
                return
            if not registration.registration_started:
                registration.released = True
                _discard_identity(self._registrations, registration)
                return
            if tuple(self._transfer_backends) != ("UCX",):
                raise transfer.BackendFailureError(
                    "NIXL durable deregistration requires exactly one UCX backend"
                )
            agent = self._require_agent()
            receipt = registration.deregistration_receipt
            execute_receipt = getattr(agent, "execute_deregister_memory", None)
            prepare_receipt = getattr(agent, "prepare_deregister_memory", None)
            if not callable(prepare_receipt) or not callable(execute_receipt):
                raise transfer.BackendFailureError(
                    "PyTorch NIXL integration requires durable deregistration "
                    "receipt support"
                )
            if receipt is None:
                receipt = prepare_receipt(
                    registration.descriptor_list,
                    backends=self._transfer_backends,
                )
                registration.deregistration_receipt = receipt
            try:
                execute_receipt(receipt)
            except BaseException as error:
                completed = bool(receipt.completed)
                if completed:
                    registration.released = True
                    _discard_identity(self._registrations, registration)
                    if isinstance(error, Exception) and _is_native_not_found(error):
                        return
                if not isinstance(error, Exception):
                    raise
                raise transfer.BackendFailureError(
                    "NIXL failed to deregister memory"
                ) from error
            if not bool(receipt.completed):
                raise transfer.BackendFailureError(
                    "NIXL deregistration returned without a completion receipt"
                )
            registration.released = True
            _discard_identity(self._registrations, registration)

    def describe_registration(self, handle: object) -> bytes:
        registration = _require_type(handle, _NixlRegistration, "registration")
        if registration.released:
            raise transfer.ClosedError("NIXL registration is closed")
        return _encode_descriptor(registration)

    def endpoint_payload(self, registration_handles: Sequence[object]) -> bytes:
        """Export metadata for exactly the registrations selected by PyTorch.

        NIXL's full metadata path is used when the selection contains every
        live registration owned by this backend.  Proper subsets use NIXL's
        public partial-metadata API and include connection information.  That
        API accepts one memory type per descriptor list, so a mixed-memory
        proper subset cannot be represented without leaking registrations and
        is rejected explicitly.
        """

        with self._lock:
            self._check_open()
            registrations = self._validate_registration_selection(registration_handles)
            if _same_identity_set(registrations, self._registrations):
                try:
                    return bytes(self._require_agent().get_agent_metadata())
                except Exception as error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to export full endpoint metadata"
                    ) from error

            memory_types = {registration.memory_type for registration in registrations}
            if len(memory_types) > 1:
                raise transfer.UnsupportedError(
                    "NIXL cannot export a mixed-memory registration subset "
                    "without including unselected registrations"
                )

            # get_reg_descs currently requires a non-empty input.  For an
            # explicitly empty selection, construct a public descriptor list
            # from one live registration and immediately clear the temporary
            # list before asking for connection-only metadata.
            if registrations:
                memory_type = registrations[0].memory_type
                descriptors = [
                    (
                        registration.address,
                        registration.nbytes,
                        registration.device_id,
                        "",
                    )
                    for registration in registrations
                ]
            else:
                seed = self._registrations[0]
                memory_type = seed.memory_type
                descriptors = [(seed.address, seed.nbytes, seed.device_id, "")]
            try:
                agent = self._require_agent()
                descriptor_list = agent.get_reg_descs(descriptors, mem_type=memory_type)
                if descriptor_list is None:
                    raise RuntimeError("NIXL did not create a descriptor list")
                if not registrations:
                    descriptor_list.clear()
                return bytes(
                    agent.get_partial_agent_metadata(
                        descriptor_list,
                        inc_conn_info=True,
                        backends=self._transfer_backends,
                    )
                )
            except Exception as error:
                raise transfer.BackendFailureError(
                    "NIXL failed to export partial endpoint metadata"
                ) from error

    def _validate_registration_selection(
        self, registration_handles: Sequence[object]
    ) -> tuple[_NixlRegistration, ...]:
        registrations: list[_NixlRegistration] = []
        identities: set[int] = set()
        live_identities = {id(registration) for registration in self._registrations}
        for handle in registration_handles:
            registration = _require_type(
                handle, _NixlRegistration, "registration selection"
            )
            identity = id(registration)
            if registration.released or identity not in live_identities:
                raise transfer.ClosedError(
                    "NIXL registration selection contains a closed registration"
                )
            if identity in identities:
                raise transfer.InvalidMetadataError(
                    "NIXL registration selection contains duplicates"
                )
            identities.add(identity)
            registrations.append(registration)
        return tuple(registrations)

    def import_peer(
        self,
        metadata: transfer.EndpointMetadata,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        with self._lock:
            self._check_open()
            endpoint_id = metadata.endpoint_id
            incarnation = metadata.incarnation
            payload = metadata.backend_payload
            grants = _decode_remote_grants(metadata.registrations)
            key = (endpoint_id, incarnation)
            expected_remote_name = _native_agent_name(endpoint_id, incarnation)
            payload_digest = hashlib.sha256(payload).digest()
            state = self._peer_states.get(key)

            try:
                inspected_name = self._require_agent().inspect_remote_agent(
                    bytes(payload)
                )
            except Exception as error:
                raise transfer.InvalidMetadataError(
                    "NIXL failed to inspect remote metadata"
                ) from error
            if isinstance(inspected_name, bytes):
                inspected_name = inspected_name.decode("utf-8")
            if inspected_name != expected_remote_name:
                raise transfer.InvalidMetadataError(
                    "NIXL metadata identity does not match the Core endpoint "
                    "identity and incarnation"
                )

            if state is not None:
                if not state.live_peers:
                    raise transfer.BusyError(
                        "NIXL peer cleanup is incomplete; retry release before "
                        "importing the same identity"
                    )
                if state.payload_digest != payload_digest or state.grants != grants:
                    raise transfer.InvalidMetadataError(
                        "NIXL metadata grants are immutable for a live Peer; "
                        "close it before importing a replacement snapshot"
                    )
                peer = _NixlPeer(
                    endpoint_id,
                    incarnation,
                    state.remote_name,
                    state.grants,
                    notification_sender=state.notification_sender,
                    notification_send=state.notification_send,
                )
                self._peers.append(peer)
                try:
                    adopt_handle(peer)
                except BaseException:
                    _discard_identity(self._peers, peer)
                    raise
                # Identity membership is the reference transaction. If a
                # signal follows set.add(), release_peer() discards this exact
                # handle idempotently instead of repeating an integer decrement.
                state.live_peers.add(peer)
                return

            bound_key = self._peer_keys_by_name.get(expected_remote_name)
            if bound_key is not None and bound_key != key:
                raise transfer.BusyError(
                    "NIXL native peer identity is already bound to another "
                    "Core endpoint incarnation"
                )

            state = _NixlPeerState(
                remote_name=expected_remote_name,
                payload_digest=payload_digest,
                grants=grants,
            )
            peer = _NixlPeer(
                endpoint_id,
                incarnation,
                state.remote_name,
                state.grants,
            )
            self._peers.append(peer)
            try:
                adopt_handle(peer)
            except BaseException:
                _discard_identity(self._peers, peer)
                raise
            # Core now owns the provisional peer. Establish provider state
            # before the first native metadata mutation.
            state.live_peers.add(peer)
            self._peer_states[key] = state
            peer_keys_by_name = dict(self._peer_keys_by_name)
            peer_keys_by_name[expected_remote_name] = key
            self._peer_keys_by_name = peer_keys_by_name
            peer.native_import_started = True
            try:
                remote_name = self._require_agent().add_remote_agent(bytes(payload))
                if isinstance(remote_name, bytes):
                    remote_name = remote_name.decode("utf-8")
                if remote_name != expected_remote_name:
                    # The cleanup handle must learn the actually loaded name
                    # first; a signal between these assignments can then still
                    # invalidate the native entry.
                    peer.remote_name = remote_name
                    state.remote_name = remote_name
                    raise transfer.InvalidMetadataError(
                        "NIXL loaded metadata identity changed after preflight"
                    )
                factory = self._raw_notification_sender_factory
                handles = self._raw_transfer_backend_handles
                if factory is not None and handles is not None:
                    sender = factory(state.remote_name, handles)
                    send = getattr(sender, "send", None)
                    if not callable(send):
                        raise TypeError("NIXL notification sender must expose send()")
                    state.notification_sender = sender
                    state.notification_send = send
                    peer.notification_sender = sender
                    peer.notification_send = send
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                if isinstance(error, transfer.InvalidMetadataError):
                    raise
                raise transfer.BackendFailureError(
                    "NIXL failed to import remote metadata"
                ) from error
            return

    def release_peer(self, handle: object) -> None:
        peer = _require_type(handle, _NixlPeer, "peer")
        with self._lock:
            # Identity discard is the release transaction. Every later step is
            # idempotent, so interruption cannot decrement another live handle.
            # Preserve the established retry contract: a failed native removal
            # leaves this handle open for the safe one-shot fallback.
            key = (peer.endpoint_id, peer.incarnation)
            state = self._peer_states.get(key)
            if state is not None:
                state.live_peers.discard(peer)
                if not state.live_peers:
                    peer.notification_send = None
                    peer.notification_sender = None
                    state.notification_send = None
                    state.notification_sender = None
                    try:
                        self._require_agent().remove_remote_agent(peer.remote_name)
                    except Exception as error:
                        if not _is_native_not_found(error):
                            raise transfer.BackendFailureError(
                                "NIXL failed to release remote metadata"
                            ) from error
                    if self._peer_states.get(key) is state:
                        self._peer_states.pop(key, None)
                    peer_keys_by_name = dict(self._peer_keys_by_name)
                    peer_keys_by_name.pop(state.remote_name, None)
                    peer_keys_by_name.pop(
                        _native_agent_name(peer.endpoint_id, peer.incarnation), None
                    )
                    self._peer_keys_by_name = peer_keys_by_name
            peer.notification_send = None
            peer.notification_sender = None
            peer.released = True
            _discard_identity(self._peers, peer)

    def prepare(
        self,
        local: Sequence[transfer.BackendRegion],
        remote: Sequence[transfer.BackendRegion],
        *,
        indexed: bool,
        adopt_handle: Callable[[object], None],
    ) -> None:
        del indexed  # NIXL prepared lists support both full and indexed requests.
        with self._lock:
            self._check_open()
            if not local or not remote:
                raise transfer.InvalidRegionError("transfer plans cannot be empty")
            local_regions = tuple(local)
            remote_regions = tuple(remote)
            local_starts = _catalog_starts(local_regions)
            remote_starts = _catalog_starts(remote_regions)
            local_block_count = _logical_block_count(local_regions)
            remote_block_count = _logical_block_count(remote_regions)
            local_descs, local_memory_type, local_resolved = _make_descriptors(
                local_regions
            )
            remote_descs, remote_memory_type, remote_resolved = _make_descriptors(
                remote_regions
            )
            peer = _single_remote_peer(remote_resolved)
            plan = _NixlPlan(
                local_handle=None,
                remote_handle=None,
                peer=peer,
                local_regions=local_regions,
                remote_regions=remote_regions,
                local_resolved=local_resolved,
                remote_resolved=remote_resolved,
                local_starts=local_starts,
                remote_starts=remote_starts,
                local_block_count=local_block_count,
                remote_block_count=remote_block_count,
                local_total_nbytes=_full_catalog_nbytes(local_regions),
                remote_total_nbytes=_full_catalog_nbytes(remote_regions),
                full_layout_matches=_same_full_block_layout(
                    local_regions, remote_regions
                ),
                uniform_pair_nbytes=_uniform_pair_nbytes(local_regions, remote_regions),
                local_full_nonoverlapping=_resolved_regions_are_nonoverlapping(
                    local_resolved
                ),
                remote_full_nonoverlapping=_resolved_regions_are_nonoverlapping(
                    remote_resolved
                ),
            )
            # Track the fully allocated Python transaction before acquiring any
            # native handle. After this point rollback needs no plan allocation.
            self._plans.append(plan)
            try:
                adopt_handle(plan)
            except BaseException:
                _discard_identity(self._plans, plan)
                raise
            try:
                plan.local_handle = self._require_agent().prep_xfer_dlist(
                    "NIXL_INIT_AGENT",
                    local_descs,
                    mem_type=local_memory_type,
                    backends=self._transfer_backends,
                )
                plan.remote_handle = self._require_agent().prep_xfer_dlist(
                    peer.remote_name,
                    remote_descs,
                    mem_type=remote_memory_type,
                    backends=self._transfer_backends,
                )
            except BaseException as prepare_error:
                if not isinstance(prepare_error, Exception):
                    raise
                raise transfer.BackendFailureError(
                    "NIXL failed to prepare transfer descriptors"
                ) from prepare_error

    def release_plan(self, handle: object) -> None:
        plan = _require_type(handle, _NixlPlan, "plan")
        with self._lock:
            if plan.released:
                return
            self._reconcile_submission_reservations_locked()
            if self._has_active_submission_locked(plan):
                raise transfer.BusyError(
                    "cannot release a NIXL plan with an active submission"
                )
            self._reconcile_request_pool_locked(plan)
            if plan.request_slots:
                raise transfer.BusyError(
                    "cannot release a NIXL plan with persistent request slots"
                )
            if any(
                work._plan is plan and not work._released
                for work in self._works.values()
            ):
                raise transfer.BusyError(
                    "cannot release a NIXL plan with unreleased work"
                )
            # Once native teardown starts, one side may already be gone even
            # if another side fails. The plan can only be retried for close; it
            # must never become submit-capable again.
            plan.release_started = True
            request_errors = self._drain_cached_requests_locked(plan)
            if request_errors:
                # Prepared descriptor lists remain dependencies of every
                # request whose release failed. Do not touch either list until
                # the idle request pool is completely drained on a later retry.
                raise transfer.BackendFailureError(
                    "NIXL failed to release cached transfer requests"
                ) from request_errors[0]
            errors: list[BaseException] = []
            for side, dlist in (
                ("remote", plan.remote_handle),
                ("local", plan.local_handle),
            ):
                released_attribute = f"{side}_released"
                if getattr(plan, released_attribute):
                    continue
                if dlist is None:
                    setattr(plan, released_attribute, True)
                    continue
                try:
                    dlist.release()
                except Exception as error:
                    errors.append(error)
                else:
                    setattr(plan, released_attribute, True)
            if errors:
                raise transfer.BackendFailureError(
                    "NIXL failed to release prepared transfer descriptors"
                ) from errors[0]
            plan.released = True
            _discard_identity(self._plans, plan)

    def _request_selection_locked(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        local_is_full = local_indices is None
        remote_is_full = remote_indices is None
        if local_is_full and plan.local_block_count > self._max_materialized_indices:
            raise transfer.UnsupportedError(
                "full local NIXL selection exceeds max_materialized_indices; "
                "submit explicit bounded index chunks"
            )
        if remote_is_full and plan.remote_block_count > self._max_materialized_indices:
            raise transfer.UnsupportedError(
                "full remote NIXL selection exceeds max_materialized_indices; "
                "submit explicit bounded index chunks"
            )
        local_selection = (
            _materialize_full_indices(plan, "local")
            if local_is_full
            else _indices(local_indices, plan.local_block_count, "local")
        )
        remote_selection = (
            _materialize_full_indices(plan, "remote")
            if remote_is_full
            else _indices(remote_indices, plan.remote_block_count, "remote")
        )
        if local_is_full and remote_is_full:
            layout_matches = plan.full_layout_matches
            bytes_transferred = plan.local_total_nbytes
            destination_nonoverlapping = (
                plan.remote_full_nonoverlapping
                if op is transfer.TransferOp.WRITE
                else plan.local_full_nonoverlapping
            )
            if not destination_nonoverlapping:
                raise transfer.InvalidRegionError(
                    "NIXL rejects overlapping destination regions"
                )
        else:
            layout_matches, bytes_transferred = _selected_layout(
                plan.local_resolved,
                plan.local_starts,
                local_selection,
                plan.remote_resolved,
                plan.remote_starts,
                remote_selection,
            )
            if op is transfer.TransferOp.WRITE:
                _reject_overlapping_destination(
                    plan.remote_resolved,
                    plan.remote_starts,
                    remote_selection,
                )
            else:
                _reject_overlapping_destination(
                    plan.local_resolved,
                    plan.local_starts,
                    local_selection,
                )
        if not layout_matches:
            raise transfer.UnsupportedError(
                "NIXL requires selected local and remote regions to have the "
                "same per-block layout"
            )
        if notification is not None and not notification:
            raise transfer.UnsupportedError(
                "the NIXL Python API cannot attach an empty notification"
            )
        return local_selection, remote_selection, bytes_transferred

    def _prevalidated_request_selection_locked(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        selection: transfer.BackendIndexSelection,
    ) -> tuple[memoryview, memoryview, int]:
        # The authenticated Core object already owns exact read-only native-int32
        # views. Forward them unchanged so the binding can span their PEP 3118
        # buffers directly instead of allocating two NumPy wrapper objects.
        local_selection = selection.local_indices
        remote_selection = selection.remote_indices

        # Core's uniform proof is exact for its plan view; NIXL independently
        # resolves the provider plan and must agree before eliding pair walks.
        if (
            selection.uniform_pair_layout
            and plan.uniform_pair_nbytes is not None
            and len(local_selection) == len(remote_selection)
        ):
            bytes_transferred = len(local_selection) * plan.uniform_pair_nbytes
            layout_matches = True
        else:
            layout_matches, bytes_transferred = _selected_layout(
                plan.local_resolved,
                plan.local_starts,
                local_selection,
                plan.remote_resolved,
                plan.remote_starts,
                remote_selection,
            )
        if not layout_matches:
            raise transfer.UnsupportedError(
                "NIXL requires selected local and remote regions to have the "
                "same per-block layout"
            )
        if bytes_transferred != selection.selected_nbytes:
            raise transfer.BackendFailureError(
                "Core and NIXL selected-byte proofs disagree"
            )

        destination_full_nonoverlapping = (
            plan.remote_full_nonoverlapping
            if op is transfer.TransferOp.WRITE
            else plan.local_full_nonoverlapping
        )
        if not (
            selection.destination_catalog_nonoverlapping
            and selection.destination_indices_unique
            and destination_full_nonoverlapping
        ):
            if op is transfer.TransferOp.WRITE:
                _reject_overlapping_destination(
                    plan.remote_resolved,
                    plan.remote_starts,
                    remote_selection,
                )
            else:
                _reject_overlapping_destination(
                    plan.local_resolved,
                    plan.local_starts,
                    local_selection,
                )
        return local_selection, remote_selection, bytes_transferred

    def create_request_slot(
        self,
        op: transfer.TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        adopt_slot: Callable[..., None],
    ) -> None:
        plan = _require_type(plan_handle, _NixlPlan, "plan")
        with self._lock:
            self._check_open()
            if plan.release_started:
                raise transfer.ClosedError("NIXL transfer plan is closing or closed")
            if not callable(adopt_slot):
                raise TypeError("adopt_slot must be callable")
            execution_dispatch = None
            factory = self._raw_request_slot_factory
            if factory is not None:
                local_selection, remote_selection, _ = self._request_selection_locked(
                    op,
                    plan,
                    local_indices,
                    remote_indices,
                    notification,
                )
                agent = self._require_agent()
                operation = (
                    self._raw_write_operation
                    if op is transfer.TransferOp.WRITE
                    else self._raw_read_operation
                )
                local_handle = _require_native_plan_handle(plan.local_handle)
                remote_handle = _require_native_plan_handle(plan.remote_handle)
                backend_handles = self._raw_transfer_backend_handles
                if backend_handles is None:
                    # Compatibility seam for synthetic tests and callers that
                    # install a slot factory after backend construction.
                    backend_handles = [
                        agent.backends[name] for name in self._transfer_backends
                    ]
                execution_dispatch = factory(
                    operation,
                    local_handle._handle,
                    local_selection,
                    remote_handle._handle,
                    remote_selection,
                    b"" if notification is None else bytes(notification),
                    backend_handles,
                    transfer.WorkState.RUNNING,
                    transfer.WorkState.COMPLETED,
                    transfer.WorkState.FAILED,
                    getattr(local_handle, "_owner", local_handle),
                    getattr(remote_handle, "_owner", remote_handle),
                )
            slot = NixlRequestSlot(
                self,
                op,
                plan,
                local_indices=local_indices,
                remote_indices=remote_indices,
                notification=notification,
                execution_dispatch=execution_dispatch,
            )
            try:
                self._slots[id(slot)] = slot
                plan.request_slots[id(slot)] = slot
                if execution_dispatch is None:
                    adopt_slot(slot)
                else:
                    adopt_slot(slot, execution_dispatch=execution_dispatch)
            except BaseException:
                # The callback may have published the slot immediately before
                # interruption. No native request exists, so remove provider
                # registry ownership, but leave the Python slot itself usable:
                # Core can close (or use) it if adoption actually took effect;
                # otherwise the unreferenced record is harmlessly collected.
                plan.request_slots.pop(id(slot), None)
                self._slots.pop(id(slot), None)
                return

    def _publish_submission_reservation_locked(
        self, reservation: _SubmissionReservation
    ) -> None:
        reservations = self._submission_reservations
        assert reservations is not None
        reservations[id(reservation)] = reservation

    def _finish_submission_reservation_locked(
        self, reservation: _SubmissionReservation
    ) -> None:
        # The plain attribute store is the transaction commit. If an
        # asynchronous exception lands in dict.pop() after mutation, close and
        # release_plan can distinguish the stale record from an active submit.
        reservation.done = True
        reservations = self._submission_reservations
        if reservations is not None:
            reservations.pop(id(reservation), None)

    def _reconcile_submission_reservations_locked(self) -> None:
        reservations = self._submission_reservations
        if not reservations:
            return
        for key, reservation in tuple(reservations.items()):
            if reservation.done:
                reservations.pop(key, None)
            elif reservation.aborting:
                # The submitting thread may have been interrupted while
                # acquiring the lock for rollback. Teardown can safely finish
                # that durably declared, pre-Work abort.
                self._abort_submission_reservation_locked(reservation, None)

    def _has_active_submission_locked(self, plan: _NixlPlan | None = None) -> bool:
        reservations = self._submission_reservations
        if not reservations:
            return False
        return any(
            not reservation.done and (plan is None or reservation.plan is plan)
            for reservation in reservations.values()
        )

    def _record_fresh_request_locked(
        self, plan: _NixlPlan, request: _NixlRequest
    ) -> None:
        # The master request record is authoritative if dict publication is
        # interrupted after mutation.
        plan.request_pool_dirty = True
        plan.requests[id(request)] = request
        plan.request_pool_dirty = False

    def _abort_submission_reservation_locked(
        self,
        reservation: _SubmissionReservation,
        error: BaseException | None,
    ) -> None:
        request = reservation.request
        try:
            if request is not None:
                plan = reservation.plan
                plan.request_pool_dirty = True
                if reservation.restore_idle_on_abort:
                    # No native retirement may begin while this flag is set.
                    # Reconciliation reconstructs the possibly interrupted
                    # cache bucket from the plan's master record.
                    request.state = _REQUEST_IDLE
                else:
                    release_error: BaseException | None = None
                    try:
                        if not _handle_is_released(request.handle):
                            request.handle.release()
                    except BaseException as cleanup_error:
                        release_error = cleanup_error
                    if _handle_is_released(request.handle):
                        request.state = _REQUEST_RETIRED
                        plan.requests.pop(id(request), None)
                    else:
                        # A failed native retirement remains plan-owned so
                        # release_plan/backend.close can retry it.
                        plan.requests[id(request)] = request
                    if release_error is not None and error is not None:
                        _add_exception_note(
                            error,
                            "NIXL request cleanup during submission rollback "
                            f"reported {release_error!r}",
                        )
        finally:
            reservation.restore_idle_on_abort = False
            self._finish_submission_reservation_locked(reservation)

    def _abort_submission_reservation(
        self,
        reservation: _SubmissionReservation,
        error: BaseException,
    ) -> None:
        # Publish abort intent before the interruptible lock acquisition. A
        # later close/release_plan reconciliation can take over if this thread
        # never enters the critical section.
        reservation.aborting = True
        # A signal can arrive while the provider lock is being acquired. Retry
        # one interrupted entry so a pre-Work reservation cannot become a
        # permanent active-looking close blocker.
        for _ in range(2):
            try:
                with self._lock:
                    if reservation.done:
                        return
                    self._abort_submission_reservation_locked(reservation, error)
                return
            except BaseException as cleanup_error:
                _add_exception_note(
                    error,
                    "NIXL submission-reservation rollback reported "
                    f"{cleanup_error!r}",
                )
                if reservation.done:
                    return

    def _finish_submission_reservation_best_effort(
        self,
        reservation: _SubmissionReservation,
        error: BaseException,
    ) -> None:
        # A registered cold Work is already the lifetime anchor. Commit before
        # the potentially interrupted lock acquisition; a stale dict entry is
        # then distinguishable and removable by close/release reconciliation.
        reservation.done = True
        try:
            with self._lock:
                self._finish_submission_reservation_locked(reservation)
        except BaseException as cleanup_error:
            reservation.done = True
            _add_exception_note(
                error,
                "NIXL submission-reservation completion reported " f"{cleanup_error!r}",
            )

    def submit(
        self,
        op: transfer.TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        reuse_token: object | None = None,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        plan = _require_type(plan_handle, _NixlPlan, "plan")
        if self._submission_reservations is not None:
            self._submit_multiple(
                op,
                plan,
                local_indices=local_indices,
                remote_indices=remote_indices,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
            )
            return
        with self._lock:
            self._check_open()
            if plan.release_started:
                raise transfer.ClosedError("NIXL transfer plan is closing or closed")
            if not callable(adopt_work):
                raise TypeError("adopt_work must be callable")

            # The token is a Core-owned capability for one immutable
            # op/indices/notification binding. On a hit, the request already
            # contains the fully validated native descriptors and notification,
            # so avoid rematerializing or revalidating those selections.
            cached = (
                None
                if reuse_token is None
                else self._take_cached_request_locked(plan, reuse_token)
            )
            fresh_request = cached is None
            if cached is not None:
                request = cached
                handle = request.handle
                bytes_transferred = request.bytes_transferred
            else:
                (
                    local_selection,
                    remote_selection,
                    bytes_transferred,
                ) = self._request_selection_locked(
                    op,
                    plan,
                    local_indices,
                    remote_indices,
                    notification,
                )
                try:
                    handle = self._make_prepped_xfer_request(
                        op,
                        plan,
                        local_selection,
                        remote_selection,
                        (b"" if notification is None else bytes(notification)),
                    )
                except Exception as error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to create a transfer request"
                    ) from error
                request = _NixlRequest(
                    handle,
                    bytes_transferred,
                    reuse_token,
                )
            self._publish_and_post_request_locked(
                plan,
                request,
                handle,
                bytes_transferred,
                reuse_token,
                adopt_work,
                fresh_request=fresh_request,
                notification=notification,
                notification_at_post=False,
            )

    def _make_prepped_xfer_request(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        local_selection: Sequence[int],
        remote_selection: Sequence[int],
        notification: bytes,
    ) -> object:
        """Create one plan-based request at the strongest loaded NIXL layer."""

        local_handle = _require_native_plan_handle(plan.local_handle)
        remote_handle = _require_native_plan_handle(plan.remote_handle)
        raw_make_xfer_req_owned = self._raw_make_xfer_req_owned
        raw_backend_handles = self._raw_transfer_backend_handles
        if raw_make_xfer_req_owned is not None and raw_backend_handles is not None:
            operation = (
                self._raw_write_operation
                if op is transfer.TransferOp.WRITE
                else self._raw_read_operation
            )
            owned_handle = raw_make_xfer_req_owned(
                operation,
                local_handle._handle,
                local_selection,
                remote_handle._handle,
                remote_selection,
                notification,
                raw_backend_handles,
                False,
            )
            # The plan/request registry is already the durable lifetime owner.
            # Retain the binding RAII object directly instead of allocating a
            # public wrapper and mutating its global retry list.
            return owned_handle
        return self._require_agent().make_prepped_xfer(
            op.value.upper(),
            local_handle,
            local_selection,
            remote_handle,
            remote_selection,
            notif_msg=notification,
            backends=self._transfer_backends,
        )

    def submit_prevalidated(
        self,
        op: transfer.TransferOp,
        plan_handle: object,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None = None,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        """Consume Core's packed int32 snapshot without materializing it."""

        self._submit_prevalidated_impl(
            op,
            plan_handle,
            selection=selection,
            notification=notification,
            reuse_token=reuse_token,
            adopt_work=adopt_work,
            initial_max_polls=None,
            initial_timeout_ns=None,
        )

    def submit_prevalidated_and_poll(
        self,
        op: transfer.TransferOp,
        plan_handle: object,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        max_polls: int,
        timeout_ns: int | None,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> transfer.WorkState:
        """Create, adopt, post, and bounded-poll one fresh selection."""

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a non-negative integer")
        if max_polls < 0 or max_polls > _INT64_MAX:
            raise ValueError(f"max_polls must be in [0, {_INT64_MAX}]")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _INT64_MAX:
                raise ValueError(f"timeout_ns must be in [0, {_INT64_MAX}] or None")
        if self._raw_post_xfer_and_poll is None:
            raise transfer.UnsupportedError(
                "loaded NIXL binding lacks fused prevalidated submit/poll"
            )
        return self._submit_prevalidated_impl(
            op,
            plan_handle,
            selection=selection,
            notification=notification,
            reuse_token=None,
            adopt_work=adopt_work,
            initial_max_polls=max_polls,
            initial_timeout_ns=timeout_ns,
        )

    def _submit_prevalidated_impl(
        self,
        op: transfer.TransferOp,
        plan_handle: object,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[transfer.BackendWork], None],
        initial_max_polls: int | None,
        initial_timeout_ns: int | None,
    ) -> transfer.WorkState:
        """Shared one-shot/reusable preparation and launch transaction."""

        plan = _require_type(plan_handle, _NixlPlan, "plan")
        if type(selection) is not transfer.BackendIndexSelection:
            raise TypeError("selection must be a Core BackendIndexSelection")
        selection._validate_binding(op, plan)
        if self._submission_reservations is not None:
            return self._submit_prevalidated_multiple(
                op,
                plan,
                selection=selection,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
                initial_max_polls=initial_max_polls,
                initial_timeout_ns=initial_timeout_ns,
            )
        lock = self._lock
        if lock is _NOOP_LOCK:
            if reuse_token is None:
                return self._submit_prevalidated_one_shot_locked(
                    op,
                    plan,
                    selection=selection,
                    notification=notification,
                    adopt_work=adopt_work,
                    initial_max_polls=initial_max_polls,
                    initial_timeout_ns=initial_timeout_ns,
                )
            return self._submit_prevalidated_locked(
                op,
                plan,
                selection=selection,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
                initial_max_polls=initial_max_polls,
                initial_timeout_ns=initial_timeout_ns,
            )
        with lock:
            if reuse_token is None:
                return self._submit_prevalidated_one_shot_locked(
                    op,
                    plan,
                    selection=selection,
                    notification=notification,
                    adopt_work=adopt_work,
                    initial_max_polls=initial_max_polls,
                    initial_timeout_ns=initial_timeout_ns,
                )
            return self._submit_prevalidated_locked(
                op,
                plan,
                selection=selection,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
                initial_max_polls=initial_max_polls,
                initial_timeout_ns=initial_timeout_ns,
            )

    def _submit_prevalidated_one_shot_locked(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        adopt_work: Callable[[transfer.BackendWork], None],
        initial_max_polls: int | None,
        initial_timeout_ns: int | None,
    ) -> transfer.WorkState:
        """Prepare one fresh request without entering reusable-cache logic."""

        self._check_open()
        if plan.release_started:
            raise transfer.ClosedError("NIXL transfer plan is closing or closed")
        if not callable(adopt_work):
            raise TypeError("adopt_work must be callable")
        if notification is not None and not notification:
            raise transfer.UnsupportedError(
                "the NIXL Python API cannot attach an empty notification"
            )
        local_selection, remote_selection, bytes_transferred = (
            self._prevalidated_request_selection_locked(
                op,
                plan,
                selection,
            )
        )
        try:
            handle = self._make_prepped_xfer_request(
                op,
                plan,
                local_selection,
                remote_selection,
                b"" if notification is None else bytes(notification),
            )
        except Exception as error:
            raise transfer.BackendFailureError(
                "NIXL failed to create a prevalidated transfer request"
            ) from error
        request = _NixlRequest(
            handle,
            bytes_transferred,
            None,
            notification_at_post=False,
            notification_present=notification is not None,
        )
        return self._publish_and_post_request_locked(
            plan,
            request,
            handle,
            bytes_transferred,
            None,
            adopt_work,
            fresh_request=True,
            notification=notification,
            notification_at_post=False,
            initial_max_polls=initial_max_polls,
            initial_timeout_ns=initial_timeout_ns,
        )

    def _submit_prevalidated_locked(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[transfer.BackendWork], None],
        initial_max_polls: int | None,
        initial_timeout_ns: int | None,
    ) -> transfer.WorkState:
        self._check_open()
        if plan.release_started:
            raise transfer.ClosedError("NIXL transfer plan is closing or closed")
        if not callable(adopt_work):
            raise TypeError("adopt_work must be callable")
        if notification is not None and not notification:
            raise transfer.UnsupportedError(
                "the NIXL Python API cannot attach an empty notification"
            )
        # The dispatcher routes one-shot requests to the straight-line sibling.
        # This body is exclusively for a reusable Core selection token.
        assert reuse_token is not None
        cached = self._take_cached_request_locked(plan, reuse_token)
        if cached is not None:
            request = cached
            if not request.notification_at_post:
                raise transfer.BackendFailureError(
                    "NIXL selection cache returned an incompatible request"
                )
            if (
                self._raw_post_xfer_with_notification_override is None
                and request.notification_present != (notification is not None)
            ):
                # Older bindings cannot clear a tag retained by a prior
                # post. Retire on a presence transition; the Core token is
                # provider-neutral and deliberately excludes notification.
                try:
                    request.handle.release()
                except Exception as error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to retire a notification-incompatible "
                        "transfer request"
                    ) from error
                request.state = _REQUEST_RETIRED
                plan.requests.pop(id(request), None)
                cached = None
        fresh_request = cached is None
        if cached is None:
            local_selection, remote_selection, bytes_transferred = (
                self._prevalidated_request_selection_locked(
                    op,
                    plan,
                    selection,
                )
            )
            try:
                handle = self._make_prepped_xfer_request(
                    op,
                    plan,
                    local_selection,
                    remote_selection,
                    # Reusable requests are notification-neutral because
                    # each generation may carry a different payload.
                    b"",
                )
            except Exception as error:
                raise transfer.BackendFailureError(
                    "NIXL failed to create a prevalidated transfer request"
                ) from error
            request = _NixlRequest(
                handle,
                bytes_transferred,
                reuse_token,
                notification_at_post=True,
                notification_present=notification is not None,
            )
        else:
            handle = request.handle
            bytes_transferred = request.bytes_transferred
        return self._publish_and_post_request_locked(
            plan,
            request,
            handle,
            bytes_transferred,
            reuse_token,
            adopt_work,
            fresh_request=fresh_request,
            notification=notification,
            notification_at_post=request.notification_at_post,
            initial_max_polls=initial_max_polls,
            initial_timeout_ns=initial_timeout_ns,
        )

    def _publish_and_post_request_locked(
        self,
        plan: _NixlPlan,
        request: _NixlRequest,
        handle: object,
        bytes_transferred: int,
        reuse_token: object | None,
        adopt_work: Callable[[transfer.BackendWork], None],
        *,
        fresh_request: bool,
        notification: bytes | None,
        notification_at_post: bool,
        initial_max_polls: int | None = None,
        initial_timeout_ns: int | None = None,
    ) -> transfer.WorkState:
        work: NixlWork | None = None
        adoption_may_have_started = False
        try:
            if fresh_request:
                # Publish native ownership before any Python wrapper can be
                # partially constructed. Every later ambiguous boundary is
                # therefore recoverable from the plan's master registry.
                self._record_fresh_request_locked(plan, request)
            work = NixlWork(
                self,
                handle,
                state=transfer.WorkState.RUNNING,
                bytes_transferred=bytes_transferred,
                plan=plan,
                reuse_token=reuse_token,
                request=request,
                submission_ready=False,
            )
            # Registry publication and Core adoption are one guarded
            # transaction. If an asynchronous exception lands after the
            # dict mutation but before callback entry, the except path
            # retires the unposted native request instead of stranding an
            # invisible, not-ready Work in the backend registry.
            self._works[id(work)] = work
            # From this store onward a signal may cross immediately before,
            # during, or after callback entry. Conservatively treat all three
            # as possible Core adoption and honor the no-raise handshake.
            adoption_may_have_started = True
            adopt_work(work)
        except BaseException as adoption_error:
            if not adoption_may_have_started:
                # Callback entry was impossible. Remove a possibly published
                # wrapper and roll a fresh unposted request back completely,
                # or return an authenticated cached request to idle.
                if work is not None:
                    self._works.pop(id(work), None)
                reservation = _SubmissionReservation(
                    plan,
                    request=request,
                    restore_idle_on_abort=not fresh_request,
                )
                try:
                    self._abort_submission_reservation_locked(
                        reservation, adoption_error
                    )
                except BaseException as cleanup_error:
                    _add_exception_note(
                        adoption_error,
                        "NIXL pre-adoption request rollback reported "
                        f"{cleanup_error!r}",
                    )
                raise
            # Callback entry may already have committed Core ownership before
            # an asynchronous exception was delivered. Resolve through the
            # retained Work and do not raise across that ambiguous boundary.
            adoption_failure = transfer.BackendFailureError(
                "Core failed to adopt a prepared NIXL transfer request"
            )
            adoption_failure.__cause__ = adoption_error
            work._submission_failed(adoption_failure)
            if work._handle is None:
                try:
                    work.release()
                except BaseException:
                    # The terminal registry entry remains provider-owned;
                    # endpoint close can finish the idempotent retirement.
                    pass
            else:
                _add_exception_note(
                    adoption_error,
                    "unposted NIXL request cleanup is pending and will be "
                    "retried by endpoint close",
                )
            # The callback may have stored Work immediately before the
            # interruption. Returning a terminal failure lets Core inspect
            # that adoption atomically; raising would violate the
            # no-raise-after-adoption side of the SPI handshake.
            return work._state
        assert work is not None
        # Core now owns every memory and plan dependency. From this point
        # submission must complete through Work, even when a signal arrives
        # after the native post has started.
        try:
            raw_post = self._raw_post_xfer
            raw_notification_override = self._raw_post_xfer_with_notification_override
            if notification_at_post and raw_notification_override is not None:
                status = raw_notification_override(request.native_handle, notification)
            elif (
                initial_max_polls == 0
                and initial_timeout_ns is None
                and not notification_at_post
                and raw_post is not None
            ):
                # A zero/no-timeout fused request is post-only.  One-shot
                # selections already carry their final notification on the
                # native request, so use the same plain raw post as direct NIXL
                # and retain its exact immediate DONE/PROC result on Work.
                status = raw_post(request.native_handle)
            elif initial_max_polls is not None:
                raw_post_and_poll = self._raw_post_xfer_and_poll
                if raw_post_and_poll is None or notification_at_post:
                    raise transfer.BackendFailureError(
                        "NIXL fused one-shot submission is unavailable"
                    )
                status = raw_post_and_poll(
                    request.native_handle,
                    initial_max_polls,
                    initial_timeout_ns,
                )
            elif raw_post is None:
                if notification_at_post:
                    status = self._require_agent().transfer(
                        handle, b"" if notification is None else notification
                    )
                else:
                    status = self._require_agent().transfer(handle)
            else:
                if notification_at_post:
                    status = raw_post(
                        request.native_handle,
                        b"" if notification is None else notification,
                    )
                else:
                    status = raw_post(request.native_handle)
            if notification_at_post:
                request.notification_present = notification is not None
            # Keep terminal/readiness publication inside the same
            # no-raise-after-adoption guard as the native post.
            work._submission_succeeded(status)
        except BaseException as error:
            work._submission_failed(
                transfer.BackendFailureError(
                    "NIXL failed to submit a transfer request"
                ),
                error,
            )
        return work._state

    def _submit_multiple(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        """Submit without serializing independent native MULTIPLE calls."""

        reservation = _SubmissionReservation(plan)
        try:
            with self._lock:
                self._check_open()
                if plan.release_started:
                    raise transfer.ClosedError(
                        "NIXL transfer plan is closing or closed"
                    )
                if not callable(adopt_work):
                    raise TypeError("adopt_work must be callable")
                self._publish_submission_reservation_locked(reservation)
                request = (
                    None
                    if reuse_token is None
                    else self._take_cached_request_locked(
                        plan, reuse_token, reservation
                    )
                )
            if request is None:
                (
                    local_selection,
                    remote_selection,
                    bytes_transferred,
                ) = self._request_selection_locked(
                    op,
                    plan,
                    local_indices,
                    remote_indices,
                    notification,
                )
                try:
                    handle = self._make_prepped_xfer_request(
                        op,
                        plan,
                        local_selection,
                        remote_selection,
                        (b"" if notification is None else bytes(notification)),
                    )
                except Exception as make_error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to create a transfer request"
                    ) from make_error
                request = _NixlRequest(handle, bytes_transferred, reuse_token)
                reservation.request = request
                with self._lock:
                    self._record_fresh_request_locked(plan, request)
            work = NixlWork(
                self,
                request.handle,
                state=transfer.WorkState.RUNNING,
                bytes_transferred=request.bytes_transferred,
                plan=plan,
                reuse_token=reuse_token,
                request=request,
                submission_ready=False,
            )
        except BaseException as error:
            self._abort_submission_reservation(reservation, error)
            raise
        self._publish_and_post_request_multiple(
            plan,
            request,
            work,
            reservation,
            adopt_work,
            notification=notification,
            notification_at_post=False,
        )

    def _submit_prevalidated_multiple(
        self,
        op: transfer.TransferOp,
        plan: _NixlPlan,
        *,
        selection: transfer.BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[transfer.BackendWork], None],
        initial_max_polls: int | None,
        initial_timeout_ns: int | None,
    ) -> transfer.WorkState:
        reservation = _SubmissionReservation(plan)
        try:
            with self._lock:
                self._check_open()
                if plan.release_started:
                    raise transfer.ClosedError(
                        "NIXL transfer plan is closing or closed"
                    )
                if not callable(adopt_work):
                    raise TypeError("adopt_work must be callable")
                if notification is not None and not notification:
                    raise transfer.UnsupportedError(
                        "the NIXL Python API cannot attach an empty notification"
                    )
                self._publish_submission_reservation_locked(reservation)
                request = (
                    None
                    if reuse_token is None
                    else self._take_cached_request_locked(
                        plan, reuse_token, reservation
                    )
                )
                if request is not None:
                    if not request.notification_at_post:
                        reservation.restore_idle_on_abort = False
                        raise transfer.BackendFailureError(
                            "NIXL selection cache returned an incompatible request"
                        )
                    if (
                        self._raw_post_xfer_with_notification_override is None
                        and request.notification_present != (notification is not None)
                    ):
                        # Once retirement begins this request must never return
                        # to IDLE, even if native release commits before a
                        # KeyboardInterrupt is delivered.
                        reservation.restore_idle_on_abort = False
                        try:
                            request.handle.release()
                        except Exception as retire_error:
                            raise transfer.BackendFailureError(
                                "NIXL failed to retire a "
                                "notification-incompatible transfer request"
                            ) from retire_error
                        request.state = _REQUEST_RETIRED
                        plan.request_pool_dirty = True
                        plan.requests.pop(id(request), None)
                        reservation.request = None
                        request = None
            reusable = reuse_token is not None
            if request is None:
                local_selection, remote_selection, bytes_transferred = (
                    self._prevalidated_request_selection_locked(
                        op,
                        plan,
                        selection,
                    )
                )
                try:
                    handle = self._make_prepped_xfer_request(
                        op,
                        plan,
                        local_selection,
                        remote_selection,
                        (
                            b""
                            if reusable or notification is None
                            else bytes(notification)
                        ),
                    )
                except Exception as make_error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to create a prevalidated transfer request"
                    ) from make_error
                request = _NixlRequest(
                    handle,
                    bytes_transferred,
                    reuse_token,
                    notification_at_post=reusable,
                    notification_present=notification is not None,
                )
                reservation.request = request
                with self._lock:
                    self._record_fresh_request_locked(plan, request)
            work = NixlWork(
                self,
                request.handle,
                state=transfer.WorkState.RUNNING,
                bytes_transferred=request.bytes_transferred,
                plan=plan,
                reuse_token=reuse_token,
                request=request,
                submission_ready=False,
            )
        except BaseException as error:
            self._abort_submission_reservation(reservation, error)
            raise
        return self._publish_and_post_request_multiple(
            plan,
            request,
            work,
            reservation,
            adopt_work,
            notification=notification,
            notification_at_post=request.notification_at_post,
            initial_max_polls=initial_max_polls,
            initial_timeout_ns=initial_timeout_ns,
        )

    def _publish_and_post_request_multiple(
        self,
        plan: _NixlPlan,
        request: _NixlRequest,
        work: NixlWork,
        reservation: _SubmissionReservation,
        adopt_work: Callable[[transfer.BackendWork], None],
        *,
        notification: bytes | None,
        notification_at_post: bool,
        initial_max_polls: int | None = None,
        initial_timeout_ns: int | None = None,
    ) -> transfer.WorkState:
        try:
            with self._lock:
                self._works[id(work)] = work
        except BaseException as publication_error:
            # Work publication has started, so failure must retire rather than
            # restore a possibly cached native request. Commit that rule before
            # the interruptible registry inspection.
            reservation.restore_idle_on_abort = False
            registered = False
            try:
                with self._lock:
                    registered = self._works.get(id(work)) is work
            except BaseException as inspection_error:
                _add_exception_note(
                    publication_error,
                    "NIXL Work publication inspection reported "
                    f"{inspection_error!r}",
                )
            failure = transfer.BackendFailureError(
                "Core failed to adopt a prepared NIXL transfer request"
            )
            if registered:
                self._finish_submission_reservation_best_effort(
                    reservation, publication_error
                )
                work._submission_failed(failure, publication_error)
                if work._handle is None:
                    try:
                        work.release()
                    except BaseException:
                        pass
            else:
                # The reservation remains the only teardown blocker while the
                # invisible Work retires native ownership.
                work._submission_failed(failure, publication_error)
                self._abort_submission_reservation(reservation, publication_error)
            return work._state

        try:
            adopt_work(work)
        except BaseException as adoption_error:
            self._finish_submission_reservation_best_effort(reservation, adoption_error)
            failure = transfer.BackendFailureError(
                "Core failed to adopt a prepared NIXL transfer request"
            )
            work._submission_failed(failure, adoption_error)
            if work._handle is None:
                try:
                    work.release()
                except BaseException:
                    pass
            else:
                _add_exception_note(
                    adoption_error,
                    "unposted NIXL request cleanup is pending and will be "
                    "retried by endpoint close",
                )
            return work._state

        try:
            raw_post = self._raw_post_xfer
            raw_notification_override = self._raw_post_xfer_with_notification_override
            handle = request.handle
            if notification_at_post and raw_notification_override is not None:
                status = raw_notification_override(request.native_handle, notification)
            elif (
                initial_max_polls == 0
                and initial_timeout_ns is None
                and not notification_at_post
                and raw_post is not None
            ):
                # Match the single-entry post-only route without adding a fused
                # binding argument conversion to independent MULTIPLE submissions.
                status = raw_post(request.native_handle)
            elif initial_max_polls is not None:
                raw_post_and_poll = self._raw_post_xfer_and_poll
                if raw_post_and_poll is None or notification_at_post:
                    raise transfer.BackendFailureError(
                        "NIXL fused one-shot submission is unavailable"
                    )
                status = raw_post_and_poll(
                    request.native_handle,
                    initial_max_polls,
                    initial_timeout_ns,
                )
            elif raw_post is None:
                if notification_at_post:
                    status = self._require_agent().transfer(
                        handle, b"" if notification is None else notification
                    )
                else:
                    status = self._require_agent().transfer(handle)
            elif notification_at_post:
                status = raw_post(
                    request.native_handle,
                    b"" if notification is None else notification,
                )
            else:
                status = raw_post(request.native_handle)
            if notification_at_post:
                request.notification_present = notification is not None
            # Handoff teardown protection from the reservation to the cold
            # registered Work before publishing readiness under the Work lock.
            reservation.done = True
            with self._lock:
                self._finish_submission_reservation_locked(reservation)
            work._submission_succeeded(status)
        except BaseException as post_error:
            self._finish_submission_reservation_best_effort(reservation, post_error)
            work._submission_failed(
                transfer.BackendFailureError(
                    "NIXL failed to submit a transfer request"
                ),
                post_error,
            )
        return work._state

    def send_notification(
        self,
        peer_handle: object,
        payload: bytes,
        *,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        peer = _require_type(peer_handle, _NixlPeer, "peer")
        lock = self._lock
        if lock is _NOOP_LOCK:
            self._send_notification_locked(peer, payload, adopt_work)
            return
        self._send_notification_multiple(peer, payload, adopt_work)

    def _send_notification_multiple(
        self,
        peer: _NixlPeer,
        payload: bytes,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        """Adopt a close-visible Work, then enter the native send unlocked."""

        with self._lock:
            self._check_open()
            if peer.released:
                raise transfer.ClosedError("NIXL peer is closed")
            if not callable(adopt_work):
                raise TypeError("adopt_work must be callable")
            send_failure = transfer.BackendFailureError(
                "NIXL failed to send a notification"
            )
            adoption_failure = transfer.BackendFailureError(
                "Core failed to adopt a NIXL notification request"
            )
            work = NixlWork(
                self,
                None,
                state=transfer.WorkState.RUNNING,
                submission_ready=False,
            )
            try:
                self._works[id(work)] = work
                adopt_work(work)
            except BaseException as error:
                work._submission_failed(adoption_failure, error)
                try:
                    work.release()
                except BaseException:
                    pass
                return
            send = peer.notification_send
            fallback_send = self._require_agent().send_notif if send is None else None
            remote_name = peer.remote_name

        # The adopted Work pins provider teardown, and Core pins the Peer. The
        # captured bound callable strongly owns its native sender/agent while the
        # provider-global ownership lock is released.
        try:
            if send is None:
                assert fallback_send is not None
                fallback_send(remote_name, payload)
            else:
                send(payload)
            work._finish_synchronous()
        except BaseException as error:
            work._submission_failed(send_failure, error)

    def _send_notification_locked(
        self,
        peer: _NixlPeer,
        payload: bytes,
        adopt_work: Callable[[transfer.BackendWork], None],
    ) -> None:
        self._check_open()
        if peer.released:
            raise transfer.ClosedError("NIXL peer is closed")
        if not callable(adopt_work):
            raise TypeError("adopt_work must be callable")
        # These sentinels deliberately predate adoption. Once the callback may
        # have committed Core ownership, even an asynchronous exception while
        # constructing failure state must not escape the no-raise handshake.
        send_failure = transfer.BackendFailureError(
            "NIXL failed to send a notification"
        )
        adoption_failure = transfer.BackendFailureError(
            "Core failed to adopt a NIXL notification request"
        )
        work = NixlWork(
            self,
            None,
            state=transfer.WorkState.RUNNING,
            submission_ready=False,
        )
        try:
            self._works[id(work)] = work
            adopt_work(work)
        except BaseException as error:
            work._submission_failed(adoption_failure, error)
            try:
                work.release()
            except BaseException:
                pass
            return
        try:
            send = peer.notification_send
            if send is None:
                self._require_agent().send_notif(peer.remote_name, payload)
            else:
                send(payload)
            work._finish_synchronous()
        except BaseException as error:
            work._submission_failed(send_failure, error)

    def send_notification_sync(self, peer_handle: object, payload: bytes) -> None:
        peer = _require_type(peer_handle, _NixlPeer, "peer")
        lock = self._lock
        if lock is _NOOP_LOCK:
            self._check_open()
            if peer.released:
                raise transfer.ClosedError("NIXL peer is closed")
            send = peer.notification_send
            if send is None:
                self._require_agent().send_notif(peer.remote_name, payload)
            else:
                send(payload)
            return
        with lock:
            self._check_open()
            if peer.released:
                raise transfer.ClosedError("NIXL peer is closed")
            send = peer.notification_send
            fallback_send = self._require_agent().send_notif if send is None else None
            remote_name = peer.remote_name
        # Core owns the MULTIPLE-mode lifetime reservation required by the SPI.
        # The provider lock protects only callable capture; keeping it out of
        # native entry preserves independent reader/writer sends.
        if send is None:
            assert fallback_send is not None
            fallback_send(remote_name, payload)
        else:
            send(payload)

    def poll_notifications(
        self, max_items: int | None = None
    ) -> list[transfer.BackendNotification]:
        if max_items is not None:
            if isinstance(max_items, bool) or not isinstance(max_items, int):
                raise TypeError("max_items must be a non-negative integer or None")
            if max_items < 0:
                raise ValueError("max_items must be non-negative or None")
        if max_items == 0:
            return []
        lock = self._lock
        if lock is _NOOP_LOCK:
            self._check_open()
            batches = self._poll_notification_batches_locked(max_items)
        else:
            batches = self._poll_notification_batches_multiple(max_items)
        return [
            transfer.BackendNotification(
                payload=payload,
                source_endpoint_id=batch.source_endpoint_id,
                source_incarnation=batch.source_incarnation,
            )
            for batch in batches
            for payload in batch.payloads
        ]

    def poll_notification_batches(
        self, max_items: int | None = None
    ) -> list[transfer.BackendNotificationBatch]:
        if max_items is not None:
            if isinstance(max_items, bool) or not isinstance(max_items, int):
                raise TypeError("max_items must be a non-negative integer or None")
            if max_items < 0:
                raise ValueError("max_items must be non-negative or None")
        if max_items == 0:
            return []
        lock = self._lock
        if lock is _NOOP_LOCK:
            self._check_open()
            return self._poll_notification_batches_locked(max_items)
        return self._poll_notification_batches_multiple(max_items)

    def _poll_notification_batches_multiple(
        self, max_items: int | None
    ) -> list[transfer.BackendNotificationBatch]:
        reservation: _BackendCallReservation | None = None
        # Both public receive shapes share this destructive queue boundary.
        # Waiting consumers are not admitted, so one pooled token covers the
        # ordinary sequential receive path after warm-up.
        with self._notification_lock:
            try:
                with self._lock:
                    self._check_open()
                    reservation = self._reserve_backend_call_locked()
                return self._poll_notification_batches_locked(max_items)
            finally:
                if reservation is not None:
                    self._finish_backend_call(reservation)

    def _poll_notification_batches_locked(
        self, max_items: int | None
    ) -> list[transfer.BackendNotificationBatch]:
        pending = self._pending_notification_batches
        head_is_partial = bool(pending and pending[0].offset)
        if not head_is_partial and (
            max_items is None or self._pending_notification_available_count < max_items
        ):
            self._collect_notifications_locked(max_items)

        remaining = (
            self._pending_notification_available_count
            if max_items is None
            else min(max_items, self._pending_notification_available_count)
        )
        result: list[transfer.BackendNotificationBatch] = []
        try:
            while remaining:
                entry = pending[0]
                batch = entry.batch
                available = len(batch.payloads) - entry.offset
                take = min(remaining, available)
                if entry.offset == 0 and take == available:
                    returned = batch
                else:
                    returned = transfer.BackendNotificationBatch(
                        payloads=batch.payloads[entry.offset : entry.offset + take],
                        source_endpoint_id=batch.source_endpoint_id,
                        source_incarnation=batch.source_incarnation,
                    )
                result.append(returned)
                entry.offset += take
                self._pending_notification_available_count -= take
                remaining -= take
                if entry.offset == len(batch.payloads):
                    pending.popleft()
                    self._pending_notification_retained_count -= entry.retained_count
                    self._pending_notification_bytes -= entry.retained_bytes
            return result
        except BaseException as error:
            # Cursor/count/deque mutation spans multiple Python bytecodes. Once
            # interrupted, exact accounting cannot be reconstructed safely.
            self._poison_notifications_locked(error)
            if not isinstance(error, Exception):
                raise
            raise self._notification_error

    def _collect_notifications_locked(self, max_items: int | None = None) -> None:
        # New bindings destructively drain one native batch into bounded C++
        # spill storage and materialize only the requested prefix. The limits do
        # not bound notifications still resident in a backend before getNotifs.
        # A partially consumed Python head intentionally applies backpressure.
        raw_poll_notification_batches_bounded = (
            self._raw_poll_notification_batches_bounded
        )
        raw_poll_notification_batches = self._raw_poll_notification_batches
        raw_get_notification_batches = self._raw_get_notification_batches
        raw_get_notifications = self._raw_get_notifications
        exact_raw_batch_contract = False
        try:
            if raw_poll_notification_batches_bounded is not None:
                batch_item_capacity = (
                    self._max_pending_notifications
                    - self._pending_notification_retained_count
                )
                batch_byte_capacity = (
                    self._max_pending_notification_bytes
                    - self._pending_notification_bytes
                )
                if batch_item_capacity == 0 or batch_byte_capacity == 0:
                    return
                materialize_items = batch_item_capacity
                if max_items is not None:
                    materialize_items = min(
                        materialize_items,
                        max_items - self._pending_notification_available_count,
                    )
                if materialize_items <= 0:
                    return
                self._begin_notification_drain_locked()
                notifications = raw_poll_notification_batches_bounded(
                    max_items=materialize_items,
                    max_batch_items=batch_item_capacity,
                    max_batch_bytes=batch_byte_capacity,
                    max_payload_bytes=(
                        transfer.BackendNotificationBatch.MAX_PAYLOAD_BYTES
                    ),
                )
                exact_raw_batch_contract = True
            elif raw_poll_notification_batches is not None:
                self._begin_notification_drain_locked()
                notifications = raw_poll_notification_batches()
                exact_raw_batch_contract = True
            elif raw_get_notification_batches is not None:
                self._begin_notification_drain_locked()
                notifications = raw_get_notification_batches(
                    self._raw_transfer_backend_handles
                )
                exact_raw_batch_contract = True
            elif raw_get_notifications is not None:
                self._begin_notification_drain_locked()
                notifications = raw_get_notifications(
                    self._raw_notification_buffer,
                    self._raw_transfer_backend_handles,
                )
            else:
                agent = self._require_agent()
                get_new_notif_batches = getattr(agent, "get_new_notif_batches", None)
                self._begin_notification_drain_locked()
                if callable(get_new_notif_batches):
                    notifications = get_new_notif_batches(
                        backends=self._transfer_backends
                    )
                else:
                    notifications = agent.get_new_notifs(
                        backends=self._transfer_backends
                    )
        except BaseException as error:
            # getNotifs drains the native batch before pybind converts its
            # strings to Python objects. Conversion can therefore fail after
            # the native queue has irreversibly advanced. Poison even on an
            # ambiguous native/binding failure so a later poll can never
            # silently resume after potentially losing protocol messages.
            self._poison_notifications_locked(error)
            if not isinstance(error, Exception):
                raise
            raise self._notification_error
        try:
            exact_raw_batch_contract = (
                exact_raw_batch_contract and type(notifications) is dict
            )
            # The prepared binding returns an exact dict. Empty polls dominate
            # idle progress, so avoid allocating a staging list there. Do not
            # trust custom Mapping truthiness on compatibility paths.
            if type(notifications) is dict and not notifications:
                self._notification_drain_dirty = False
                return
            new_batches: list[_PendingNotificationBatch] = []
            projected_count = self._pending_notification_retained_count
            projected_bytes = self._pending_notification_bytes
            # Peer-name publication is copy-on-write. One local reference gives
            # this destructive drain a coherent identity snapshot without
            # holding the provider-global lock across native polling.
            peer_keys_by_name = self._peer_keys_by_name
            for remote_name, messages in notifications.items():
                peer_key = peer_keys_by_name.get(remote_name)
                endpoint_id = None if peer_key is None else peer_key[0]
                incarnation = None if peer_key is None else peer_key[1]
                exact_raw_batch = exact_raw_batch_contract and type(messages) is tuple
                if exact_raw_batch:
                    payloads = messages
                else:
                    try:
                        payloads = tuple(messages)
                    except TypeError as error:
                        raise TypeError(
                            "notification source values must be iterable"
                        ) from error
                if not payloads:
                    continue
                next_count = projected_count + len(payloads)
                if next_count > self._max_pending_notifications:
                    raise ValueError("notification count limit exceeded")
                if not exact_raw_batch:
                    # Compatibility producers may expose mutable bytes-like
                    # values whose snapshot is expensive or whose __bytes__
                    # result is larger than the reported view. Reject against
                    # the remaining endpoint budget before every copy, then
                    # recheck the immutable result. The prepared/one-shot raw
                    # binding contract is exact dict[str, tuple[bytes, ...]],
                    # so its normal path skips this duplicate walk and copy.
                    max_payload_bytes = (
                        transfer.BackendNotificationBatch.MAX_PAYLOAD_BYTES
                    )
                    remaining_bytes = (
                        self._max_pending_notification_bytes - projected_bytes
                    )
                    normalized_payloads = []
                    normalized_bytes = 0
                    for payload in payloads:
                        if not isinstance(payload, (bytes, bytearray, memoryview)):
                            raise TypeError("notification payload must be bytes-like")
                        reported_nbytes = (
                            payload.nbytes
                            if isinstance(payload, memoryview)
                            else len(payload)
                        )
                        if reported_nbytes > max_payload_bytes:
                            raise ValueError(
                                "notification payload exceeds the wire size limit"
                            )
                        if normalized_bytes + reported_nbytes > remaining_bytes:
                            raise ValueError("notification byte limit exceeded")
                        immutable_payload = bytes(payload)
                        payload_nbytes = len(immutable_payload)
                        if payload_nbytes > max_payload_bytes:
                            raise ValueError(
                                "notification payload exceeds the wire size limit"
                            )
                        next_normalized_bytes = normalized_bytes + payload_nbytes
                        if next_normalized_bytes > remaining_bytes:
                            raise ValueError("notification byte limit exceeded")
                        normalized_payloads.append(immutable_payload)
                        normalized_bytes = next_normalized_bytes
                    payloads = tuple(normalized_payloads)
                batch = transfer.BackendNotificationBatch(
                    payloads=payloads,
                    source_endpoint_id=endpoint_id,
                    source_incarnation=incarnation,
                )
                batch_bytes = batch.total_nbytes
                if projected_bytes + batch_bytes > self._max_pending_notification_bytes:
                    raise ValueError("notification byte limit exceeded")
                new_batches.append(
                    _PendingNotificationBatch(
                        batch=batch,
                        retained_count=len(batch.payloads),
                        retained_bytes=batch_bytes,
                    )
                )
                projected_count = next_count
                projected_bytes += batch_bytes
            self._pending_notification_batches.extend(new_batches)
            self._pending_notification_available_count = projected_count
            self._pending_notification_retained_count = projected_count
            self._pending_notification_bytes = projected_bytes
            if notifications is self._raw_notification_buffer:
                self._raw_notification_buffer.clear()
            # Commit last: every payload, queue cursor, and accounting field is
            # now owned by Python. A signal before this store leaves fail-stop
            # set; a signal after it cannot lose a native-drained payload.
            self._notification_drain_dirty = False
        except BaseException as error:
            self._poison_notifications_locked(error)
            if not isinstance(error, Exception):
                raise
            raise self._notification_error

    def _poison_notifications_locked(self, cause: BaseException) -> None:
        # This store must precede error allocation and cause publication. It is
        # the durable fallback if either of those Python operations is itself
        # interrupted by a second asynchronous exception.
        self._notification_drain_dirty = True
        if self._notification_error is None:
            self._notification_error = transfer.BackendFailureError(
                "NIXL returned malformed data or its notification backlog "
                "exceeded the configured limit; "
                "the endpoint must be closed and recreated"
            )
            self._notification_error.__cause__ = cause

    def _begin_notification_drain_locked(self) -> None:
        if self._notification_drain_dirty:
            raise self._notification_failure_locked()
        self._notification_drain_dirty = True

    def _notification_failure_locked(self) -> transfer.BackendFailureError:
        return self._notification_error or self._notification_drain_error

    def _reserve_backend_call_locked(self) -> _BackendCallReservation:
        active_calls = self._active_backend_calls
        pool = self._backend_call_reservation_pool
        assert active_calls is not None and pool is not None
        reservation = pool.pop() if pool else _BackendCallReservation()
        try:
            reservation.active = True
            active_calls.add(reservation.reference)
        except BaseException:
            reservation.active = False
            active_calls.discard(reservation.reference)
            pool.append(reservation)
            raise
        return reservation

    def _finish_backend_call(self, reservation: _BackendCallReservation) -> None:
        # Publish native exit before interruptible cleanup. Close also examines
        # this bit, so a retained exception traceback is not a false blocker.
        reservation.active = False
        with self._lock:
            active_calls = self._active_backend_calls
            pool = self._backend_call_reservation_pool
            assert active_calls is not None and pool is not None
            active_calls.discard(reservation.reference)
            if not self._closing and not self._closed:
                pool.append(reservation)

    def _reconcile_backend_calls_locked(self) -> None:
        active_calls = self._active_backend_calls
        if not active_calls:
            return
        for reference in tuple(active_calls):
            reservation = reference()
            if reservation is None or not reservation.active:
                active_calls.discard(reference)

    def _has_active_backend_calls_locked(self) -> bool:
        self._reconcile_backend_calls_locked()
        return bool(self._active_backend_calls)

    def progress(self) -> None:
        lock = self._lock
        if lock is _NOOP_LOCK:
            self._progress_locked()
            return

        reservation: _BackendCallReservation | None = None
        with self._notification_lock:
            try:
                with lock:
                    self._check_open()
                    reservation = self._reserve_backend_call_locked()
                self._progress_locked(check_open=False)
            finally:
                if reservation is not None:
                    self._finish_backend_call(reservation)

    def _progress_locked(self, *, check_open: bool = True) -> None:
        if check_open:
            self._check_open()
        # NIXL has no separate public progress call. get_new_notifs drives
        # notification-capable engines (including an otherwise idle UCX
        # responder), so preserve anything it receives for the caller.
        pending = self._pending_notification_batches
        if not pending or pending[0].offset == 0:
            self._collect_notifications_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._closing:
                if self._has_active_backend_calls_locked():
                    raise transfer.BusyError(
                        "cannot close NIXL backend with active provider calls"
                    )
                self._reconcile_submission_reservations_locked()
                if self._has_active_submission_locked():
                    raise transfer.BusyError(
                        "cannot close NIXL backend with active submissions"
                    )
                self._retry_orphan_cleanup_locked()
                active_slots = [
                    slot for slot in self._slots.values() if slot._provider_active()
                ]
                if active_slots:
                    raise transfer.BusyError(
                        "cannot close NIXL backend with "
                        f"{len(active_slots)} active request slot(s)"
                    )
                active = [work for work in self._works.values() if not work.test()]
                if active:
                    raise transfer.BusyError(
                        "cannot close NIXL backend with "
                        f"{len(active)} active transfer(s)"
                    )
                # Reject public/hot operations before entering retry-only
                # teardown. A cut between these stores is safe: close reruns
                # the non-destructive admission checks and operations still see
                # the single false availability gate.
                self._available = False
                self._commit_closing()
            for slot in list(self._slots.values()):
                slot.close()
            for work in list(self._works.values()):
                work.release()
            for plan in reversed(self._plans.copy()):
                self.release_plan(plan)
            for peer in reversed(self._peers.copy()):
                self.release_peer(peer)
            for registration in reversed(self._registrations.copy()):
                self.deregister(registration)
            self._pending_notification_batches.clear()
            self._pending_notification_available_count = 0
            self._pending_notification_retained_count = 0
            self._pending_notification_bytes = 0
            self._raw_notification_buffer.clear()
            self._notification_drain_dirty = False
            self._notification_error = None
            # Cached bound pybind methods retain the native agent owner.
            self._raw_post_xfer = None
            self._raw_post_xfer_with_notification_override = None
            self._raw_post_xfer_and_poll = None
            self._raw_get_xfer_status_batch = None
            self._raw_get_xfer_status = None
            self._raw_notification_sender_factory = None
            self._raw_poll_notification_batches = None
            self._raw_poll_notification_batches_bounded = None
            self._raw_get_notification_batches = None
            self._raw_get_notifications = None
            self._raw_notification_receiver = None
            self._raw_request_slot_factory = None
            self._raw_make_xfer_req_owned = None
            self._raw_transfer_backend_handles = None
            self._raw_read_operation = None
            self._raw_write_operation = None
            self._raw_success = None
            self._raw_in_progress = None
            # Publish CLOSED only after every strong native-owner reference is
            # gone. If Python interruption lands during reference teardown, a
            # retry still enters close and completes the remaining cleanup.
            self._agent = None
            self._activated = False
            self._closed = True
            active_calls = self._active_backend_calls
            if active_calls is not None:
                active_calls.clear()
            reservation_pool = self._backend_call_reservation_pool
            if reservation_pool is not None:
                reservation_pool.clear()

    def _commit_closing(self) -> None:
        self._closing = True

    def _take_cached_request_locked(
        self,
        plan: _NixlPlan,
        reuse_token: object | None,
        reservation: _SubmissionReservation | None = None,
    ) -> _NixlRequest | None:
        if reuse_token is None:
            return None
        self._reconcile_request_pool_locked(plan)
        token_key = id(reuse_token)
        requests = plan.cached_requests.get(token_key)
        if not requests:
            return None
        if requests[-1].reuse_token is not reuse_token:
            # Retained tokens make an id collision impossible during normal
            # operation. Treat corrupted internal state as a miss, never as an
            # authorization to reuse another binding's request.
            return None
        # request_pool_dirty makes the plan-owned record state authoritative if
        # an asynchronous exception lands between pop/state/count updates.
        plan.request_pool_dirty = True
        if reservation is None:
            # Preserve the exact single-entry transaction and hot path.
            request = requests.pop()
            request.state = _REQUEST_LEASED
        else:
            # Anchor the candidate before changing authoritative ownership.
            # The reservation can restore IDLE after an interruption even if
            # the caller never receives the helper's return value.
            request = requests[-1]
            reservation.request = request
            reservation.restore_idle_on_abort = True
            request.state = _REQUEST_LEASED
            requests.pop()
        plan.cached_request_count -= 1
        if not requests:
            del plan.cached_requests[token_key]
        plan.request_pool_dirty = False
        return request

    def _release_work(self, work: NixlWork) -> None:
        # Every non-MULTIPLE mode is already serialized by Core or the caller.
        # Dispatch directly to retirement: its NixlWork is lock-free by
        # construction, so no second helper frame or lock branch is required.
        backend_lock = self._lock
        if backend_lock is _NOOP_LOCK:
            if work._reuse_token is None:
                self._release_one_shot_work_locked(work)
            else:
                self._release_work_locked(work)
            return
        # MULTIPLE retains the established backend-then-Work lock order.
        with backend_lock:
            work_lock = work._lock
            if work_lock is None:
                raise transfer.BackendFailureError(
                    "MULTIPLE NIXL work has no provider lock"
                )
            with work_lock:
                self._release_work_locked(work)

    def _release_one_shot_work_locked(self, work: NixlWork) -> None:
        """Retire a fresh request without reusable-cache reconciliation."""

        if work._released:
            self._finish_released_work_locked(work)
            return
        if not _is_terminal(work._state):
            raise transfer.BusyError("cannot release active NIXL work")
        handle = work._handle
        request = work._request
        plan = work._plan
        if handle is not None:
            try:
                handle.release()
            except Exception as error:
                raise transfer.BackendFailureError(
                    "NIXL failed to release a transfer request"
                ) from error
            if request is not None:
                request.state = _REQUEST_RETIRED
                if plan is not None:
                    plan.requests.pop(id(request), None)
        work._released = True
        self._finish_released_work_locked(work)

    def _release_work_locked(self, work: NixlWork) -> None:
        plan = work._plan
        if plan is not None:
            self._reconcile_request_pool_locked(plan)
        if work._released:
            self._finish_released_work_locked(work)
            return
        if not _is_terminal(work._state):
            raise transfer.BusyError("cannot release active NIXL work")
        handle = work._handle
        if handle is not None:
            reuse_token = work._reuse_token
            request = work._request
            may_cache = (
                work._state is transfer.WorkState.COMPLETED
                and plan is not None
                and request is not None
                and reuse_token is not None
                and not plan.release_started
                and self._max_cached_requests_per_plan > 0
            )
            if may_cache:
                token_key = id(reuse_token)
                same_token_cached = bool(plan.cached_requests.get(token_key))
                if (
                    plan.cached_request_count >= self._max_cached_requests_per_plan
                    and not same_token_cached
                ):
                    # Admit a newly active binding by evicting the least
                    # recently returned idle request. Without this, old
                    # closed bindings can permanently crowd a hot token
                    # out of a bounded plan cache.
                    idle = min(
                        (
                            candidate
                            for candidate in plan.requests.values()
                            if candidate.state == _REQUEST_IDLE
                        ),
                        key=lambda candidate: candidate.idle_epoch,
                    )
                    # Quarantine before the GIL-releasing native destructor.
                    # Dirty reconciliation can distinguish a pre-release cut
                    # (still IDLE) from an ambiguous/post-release cut
                    # (RETIRING) without ever recaching the latter.
                    plan.request_pool_dirty = True
                    idle.state = _REQUEST_RETIRING
                    try:
                        idle.handle.release()
                    except BaseException as error:
                        if not isinstance(error, Exception):
                            raise
                        raise transfer.BackendFailureError(
                            "NIXL failed to evict an idle transfer request"
                        ) from error
                    idle.state = _REQUEST_RETIRED
                    self._reconcile_request_pool_locked(plan)
                if plan.cached_request_count >= self._max_cached_requests_per_plan:
                    may_cache = False
            if may_cache:
                # Mark the old Work released before publishing IDLE.
                # A dirty transition is reconstructed from the plan's
                # master records after KeyboardInterrupt at any boundary.
                plan.request_pool_dirty = True
                work._released = True
                request.state = _REQUEST_IDLE
                plan.idle_epoch += 1
                request.idle_epoch = plan.idle_epoch
                requests = plan.cached_requests.setdefault(token_key, [])
                requests.append(request)
                plan.cached_request_count += 1
                plan.request_pool_dirty = False
            else:
                try:
                    handle.release()
                except Exception as error:
                    raise transfer.BackendFailureError(
                        "NIXL failed to release a transfer request"
                    ) from error
                if request is not None:
                    request.state = _REQUEST_RETIRED
                    if plan is not None:
                        plan.requests.pop(id(request), None)
                work._released = True
        else:
            work._released = True
        self._finish_released_work_locked(work)

    def _finish_released_work_locked(self, work: NixlWork) -> None:
        work._handle = None
        work._post_xfer = None
        work._post_and_poll_xfer = None
        work._poll_bounded_xfer = None
        work._status_xfer = None
        work._native_handle = None
        work._success_status = None
        work._in_progress_status = None
        work._plan = None
        work._reuse_token = None
        work._request = None
        self._works.pop(id(work), None)

    def _reconcile_request_pool_locked(self, plan: _NixlPlan) -> None:
        if not plan.request_pool_dirty:
            return
        # RETIRING is an interruption-durable quarantine. Owned native request
        # release is idempotent, so replay it before rebuilding any cache view.
        for request in tuple(plan.requests.values()):
            if request.state != _REQUEST_RETIRING:
                continue
            try:
                request.handle.release()
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                raise transfer.BackendFailureError(
                    "NIXL failed to reconcile a retiring transfer request"
                ) from error
            request.state = _REQUEST_RETIRED
        live_requests: dict[int, _NixlRequest] = {}
        cached_requests: dict[int, list[_NixlRequest]] = {}
        cached_request_count = 0
        for request in plan.requests.values():
            if request.state == _REQUEST_RETIRED:
                continue
            live_requests[id(request)] = request
            if request.state == _REQUEST_IDLE:
                assert request.reuse_token is not None
                cached_requests.setdefault(id(request.reuse_token), []).append(request)
                cached_request_count += 1
        plan.requests = live_requests
        plan.cached_requests = cached_requests
        plan.cached_request_count = cached_request_count
        plan.request_pool_dirty = False

    def _drain_cached_requests_locked(self, plan: _NixlPlan) -> list[BaseException]:
        self._reconcile_request_pool_locked(plan)
        errors: list[BaseException] = []
        plan.request_pool_dirty = True
        for request in tuple(plan.requests.values()):
            if request.state == _REQUEST_RETIRED:
                continue
            request.state = _REQUEST_RETIRING
            try:
                request.handle.release()
            except BaseException as error:
                # Retain both the native request wrapper and its descriptor
                # dependencies so release_plan() can retry safely.
                if not isinstance(error, Exception):
                    raise
                errors.append(error)
            else:
                request.state = _REQUEST_RETIRED
        self._reconcile_request_pool_locked(plan)
        return errors

    def _discard_work(self, work: NixlWork) -> None:
        with self._lock:
            self._works.pop(id(work), None)

    def _check_open(self) -> None:
        if not self._available:
            if self._closed:
                raise transfer.ClosedError("NIXL backend is closed")
            if self._closing:
                raise transfer.ClosedError(
                    "NIXL backend is closing; only close() may be retried"
                )
            raise transfer.ClosedError("NIXL backend is not active")
        if self._orphan_dlists or self._orphan_remote_names:
            self._retry_orphan_cleanup_locked()
        if self._notification_drain_dirty or self._notification_error is not None:
            raise self._notification_failure_locked()

    def _retry_orphan_cleanup_locked(self) -> None:
        errors: list[BaseException] = []
        for handle in tuple(self._orphan_dlists):
            try:
                handle.release()
            except Exception as error:
                errors.append(error)
            else:
                _discard_identity(self._orphan_dlists, handle)
        for remote_name in tuple(self._orphan_remote_names):
            try:
                self._require_agent().remove_remote_agent(remote_name)
            except Exception as error:
                if _is_native_not_found(error):
                    self._orphan_remote_names.discard(remote_name)
                else:
                    errors.append(error)
            else:
                self._orphan_remote_names.discard(remote_name)
        if errors:
            raise transfer.BackendFailureError(
                "NIXL has native resources awaiting rollback; retry cleanup before "
                "using the endpoint"
            ) from errors[0]

    def _require_agent(self) -> object:
        if self._agent is None:
            raise transfer.ClosedError("NIXL backend is closed")
        return self._agent


def register_torch_backend() -> None:
    """Register the NIXL provider with the experimental PyTorch registry.

    Core treats an exact same-factory/same-version retry as idempotent.  Thus a
    retry converges if Core committed the registry write but an asynchronous
    exception prevented this module from recording ``_REGISTERED``.
    """

    global _REGISTERED
    with _REGISTRATION_LOCK:
        core_factory_version = getattr(transfer, "BACKEND_FACTORY_API_VERSION", 0)
        if (
            type(core_factory_version) is not int
            or core_factory_version < TORCH_TRANSFER_FACTORY_API_VERSION
        ):
            raise RuntimeError(
                "nixl.torch_transfer requires PyTorch transfer factory API "
                f">={TORCH_TRANSFER_FACTORY_API_VERSION}"
            )
        transfer.register_backend(
            "nixl",
            _backend_factory,
            factory_api_version=TORCH_TRANSFER_FACTORY_API_VERSION,
        )
        _REGISTERED = True


def _backend_factory(
    endpoint_id: str,
    name: str,
    incarnation: str,
    progress_mode: transfer.ProgressMode,
    thread_mode: transfer.ThreadMode,
    options: Mapping[str, Any],
    *,
    adopt_backend: Callable[[transfer.Backend], None],
) -> None:
    backend = NixlBackend._new_unactivated(
        endpoint_id,
        name,
        incarnation,
        progress_mode,
        thread_mode,
        options,
    )
    adopt_backend(backend)
    backend._activate()


def _encode_descriptor(registration: _NixlRegistration) -> bytes:
    return json.dumps(
        {
            "address": registration.address,
            "device_id": registration.device_id,
            "memory_type": registration.memory_type,
            "nbytes": registration.nbytes,
            "version": _DESCRIPTOR_VERSION,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _decode_descriptor(value: object) -> tuple[int, int, int, str]:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise transfer.InvalidMetadataError("NIXL region descriptor must be bytes-like")
    try:
        decoded = json.loads(
            bytes(value).decode("ascii"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise transfer.InvalidMetadataError("invalid NIXL region descriptor") from error
    expected = {"address", "device_id", "memory_type", "nbytes", "version"}
    if not isinstance(decoded, dict) or set(decoded) != expected:
        raise transfer.InvalidMetadataError("invalid NIXL region descriptor fields")
    if (
        isinstance(decoded["version"], bool)
        or not isinstance(decoded["version"], int)
        or decoded["version"] != _DESCRIPTOR_VERSION
    ):
        raise transfer.InvalidMetadataError(
            "unsupported NIXL region descriptor version"
        )
    address = _positive_int(decoded["address"], "descriptor address")
    nbytes = _positive_int(decoded["nbytes"], "descriptor nbytes")
    try:
        _validate_native_span(address, nbytes, "descriptor")
        device_id = _bounded_nonnegative_int(
            decoded["device_id"], "descriptor device_id", _UINT64_MAX
        )
    except transfer.InvalidRegionError as error:
        raise transfer.InvalidMetadataError("invalid NIXL descriptor range") from error
    memory_type = _normalize_memory_type(decoded["memory_type"])
    return address, nbytes, device_id, memory_type


def _decode_remote_grants(
    registrations: Sequence[transfer.RegistrationMetadata],
) -> tuple[_RemoteGrant, ...]:
    grants: list[_RemoteGrant] = []
    for registration in registrations:
        try:
            address, nbytes, device_id, memory_type = _decode_descriptor(
                registration.backend_descriptor
            )
            expected_device_id = _device_id(registration.device)
            expected_memory_type = _normalize_memory_type(registration.memory_type)
        except transfer.TransferError as error:
            raise transfer.InvalidMetadataError(
                f"invalid NIXL descriptor for registration {registration.name!r}"
            ) from error
        if nbytes != registration.nbytes:
            raise transfer.InvalidMetadataError(
                f"NIXL descriptor size for registration {registration.name!r} "
                "does not match Core metadata"
            )
        if device_id != expected_device_id:
            raise transfer.InvalidMetadataError(
                f"NIXL descriptor device for registration {registration.name!r} "
                "does not match Core metadata"
            )
        if memory_type != expected_memory_type:
            raise transfer.InvalidMetadataError(
                f"NIXL descriptor memory type for registration "
                f"{registration.name!r} does not match Core metadata"
            )
        grants.append(_RemoteGrant(address, nbytes, device_id, memory_type))
    return tuple(grants)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise transfer.InvalidMetadataError(
                f"duplicate NIXL descriptor field {key!r}"
            )
        result[key] = value
    return result


def _native_agent_name(endpoint_id: str, incarnation: str) -> str:
    endpoint_bytes = endpoint_id.encode("utf-8")
    incarnation_bytes = incarnation.encode("utf-8")
    identity = (
        len(endpoint_bytes).to_bytes(8, "big")
        + endpoint_bytes
        + len(incarnation_bytes).to_bytes(8, "big")
        + incarnation_bytes
    )
    return f"torch-transfer-{hashlib.sha256(identity).hexdigest()[:32]}"


def _make_descriptors(
    regions: Sequence[transfer.BackendRegion],
) -> tuple[
    list[tuple[int, int, int]] | np.ndarray,
    str,
    tuple[_ResolvedRegion, ...],
]:
    rows: list[tuple[int, int, int, int, int]] = []
    resolved_regions: list[_ResolvedRegion] = []
    memory_type: str | None = None
    has_stride = False
    for region in regions:
        if region.remote:
            (
                peer,
                base,
                registered_nbytes,
                device_id,
                descriptor_memory_type,
            ) = _remote_region_parts(region)
            current_memory_type = _normalize_memory_type(region.memory_type)
            if current_memory_type != descriptor_memory_type:
                raise transfer.InvalidMetadataError(
                    "remote region memory type does not match its NIXL descriptor"
                )
            if _device_id(str(region.device)) != device_id:
                raise transfer.InvalidMetadataError(
                    "remote region device does not match its NIXL descriptor"
                )
        else:
            owner = _require_type(
                region.backend_descriptor,
                _NixlRegistration,
                "local region descriptor",
            )
            if owner.released:
                raise transfer.ClosedError("NIXL registration is closed")
            base = owner.address
            registered_nbytes = owner.nbytes
            peer = None
            device_id = owner.device_id
            current_memory_type = owner.memory_type
            if _normalize_memory_type(region.memory_type) != current_memory_type:
                raise transfer.InvalidRegionError(
                    "local region memory type does not match its registration"
                )
            if _device_id(str(region.device)) != device_id:
                raise transfer.InvalidRegionError(
                    "local region device does not match its registration"
                )
        address, nbytes, stride, count = _native_region_geometry(
            region,
            base,
            registered_nbytes,
            "remote" if region.remote else "local",
        )
        if memory_type is None:
            memory_type = current_memory_type
        elif memory_type != current_memory_type:
            raise transfer.UnsupportedError(
                "one NIXL prepared descriptor list cannot mix memory types"
            )
        has_stride = has_stride or region.count is not None
        rows.append((address, nbytes, device_id, stride, count))
        resolved_regions.append(
            _ResolvedRegion(
                region=region,
                peer=peer,
                base=base,
                registered_nbytes=registered_nbytes,
                device_id=device_id,
                memory_type=current_memory_type,
            )
        )
    assert memory_type is not None
    if has_stride:
        descriptors: list[tuple[int, int, int]] | np.ndarray = np.asarray(
            rows, dtype=np.uint64
        )
    else:
        descriptors = [
            (address, nbytes, device_id) for address, nbytes, device_id, _, _ in rows
        ]
    return descriptors, memory_type, tuple(resolved_regions)


def _single_remote_peer(regions: Sequence[_ResolvedRegion]) -> _NixlPeer:
    peers: list[_NixlPeer] = []
    for resolved in regions:
        if not resolved.region.remote or resolved.peer is None:
            raise transfer.InvalidRegionError("remote plan regions must be remote")
        peer = resolved.peer
        if peer.released:
            raise transfer.ClosedError("NIXL peer is closed")
        if all(existing is not peer for existing in peers):
            peers.append(peer)
    if len(peers) != 1:
        raise transfer.UnsupportedError(
            "a NIXL transfer plan must target exactly one peer"
        )
    return peers[0]


def _reject_overlapping_destination(
    regions: Sequence[_ResolvedRegion],
    starts: Sequence[int],
    indices: Sequence[int],
) -> None:
    intervals: dict[tuple[object, int, str], list[tuple[int, int]]] = {}
    for flat_index in indices:
        resolved, block_index = _selected_block(regions, starts, int(flat_index))
        region = resolved.region
        address_space: object = (
            id(resolved.peer) if resolved.peer is not None else "local"
        )
        stride = region.nbytes if region.stride is None else region.stride
        start = resolved.base + region.offset + block_index * stride
        intervals.setdefault(
            (address_space, resolved.device_id, resolved.memory_type), []
        ).append((start, start + region.nbytes))
    for values in intervals.values():
        values.sort()
        for previous, current in zip(values, values[1:]):
            if current[0] < previous[1]:
                raise transfer.InvalidRegionError(
                    "NIXL rejects overlapping destination regions"
                )


def _remote_region_parts(
    region: transfer.BackendRegion,
) -> tuple[_NixlPeer, int, int, int, str]:
    descriptor = region.backend_descriptor
    if isinstance(descriptor, transfer.ImportedRegistration):
        imported = descriptor
        peer = _require_type(imported.peer_handle, _NixlPeer, "remote peer")
        base, descriptor_nbytes, device_id, memory_type = _decode_descriptor(
            imported.descriptor
        )
        if descriptor_nbytes != imported.nbytes:
            raise transfer.InvalidMetadataError(
                "remote registration size does not match its NIXL descriptor"
            )
        if not any(
            grant == _RemoteGrant(base, imported.nbytes, device_id, memory_type)
            for grant in peer.grants
        ):
            raise transfer.InvalidMetadataError(
                "remote registration is not covered by the imported NIXL grant"
            )
        return peer, base, imported.nbytes, device_id, memory_type

    if isinstance(descriptor, transfer.ImportedRawMemory):
        raw = descriptor
        peer = _require_type(raw.peer_handle, _NixlPeer, "remote raw-memory peer")
        address = _positive_int(raw.address, "remote raw address")
        nbytes = _positive_int(raw.nbytes, "remote raw nbytes")
        _validate_native_span(address, nbytes, "remote raw span")
        device_id = _device_id(raw.device)
        memory_type = _normalize_memory_type(raw.memory_type)
        if not _span_is_granted(peer, address, nbytes, device_id, memory_type):
            raise transfer.InvalidRegionError(
                "remote raw span is not covered by the imported NIXL grant"
            )
        return peer, address, nbytes, device_id, memory_type

    raise TypeError("remote region does not carry an imported NIXL registration")


def _span_is_granted(
    peer: _NixlPeer,
    address: int,
    nbytes: int,
    device_id: int,
    memory_type: str,
) -> bool:
    return any(
        grant.device_id == device_id
        and grant.memory_type == memory_type
        and address >= grant.address
        and address - grant.address <= grant.nbytes
        and nbytes <= grant.nbytes - (address - grant.address)
        for grant in peer.grants
    )


def _indices(indices: Sequence[int], size: int, side: str) -> np.ndarray:
    result = list(indices)
    for index in result:
        if isinstance(index, bool) or not isinstance(index, int):
            raise transfer.InvalidRegionError(f"{side} indices must be integers")
        if index < 0 or index >= size:
            raise transfer.InvalidRegionError(f"{side} index {index} is out of range")
    return np.asarray(result, dtype=np.int32)


def _logical_block_count(regions: Sequence[transfer.BackendRegion]) -> int:
    count = sum(1 if region.count is None else region.count for region in regions)
    if count > _INT32_MAX:
        raise transfer.UnsupportedError(
            "NIXL descriptor catalog exceeds int32 indexing"
        )
    return count


def _catalog_starts(
    regions: Sequence[transfer.BackendRegion],
) -> tuple[int, ...]:
    starts = []
    next_index = 0
    for region in regions:
        starts.append(next_index)
        next_index += 1 if region.count is None else region.count
    return tuple(starts)


def _selected_block(
    regions: Sequence[_ResolvedRegion],
    starts: Sequence[int],
    index: int,
) -> tuple[_ResolvedRegion, int]:
    region_index = bisect_right(starts, index) - 1
    return regions[region_index], index - starts[region_index]


def _selected_layout(
    local_regions: Sequence[_ResolvedRegion],
    local_starts: Sequence[int],
    local_indices: Sequence[int],
    remote_regions: Sequence[_ResolvedRegion],
    remote_starts: Sequence[int],
    remote_indices: Sequence[int],
) -> tuple[bool, int]:
    if len(local_indices) != len(remote_indices):
        return False, 0
    total = 0
    for local_index, remote_index in zip(local_indices, remote_indices):
        local, _ = _selected_block(local_regions, local_starts, int(local_index))
        remote, _ = _selected_block(remote_regions, remote_starts, int(remote_index))
        if local.region.nbytes != remote.region.nbytes:
            return False, 0
        total += local.region.nbytes
        if total > _SIZE_T_MAX:
            raise transfer.UnsupportedError(
                "selected NIXL byte count exceeds native size_t"
            )
    return True, total


def _full_catalog_nbytes(regions: Sequence[transfer.BackendRegion]) -> int:
    total = 0
    for region in regions:
        count = 1 if region.count is None else region.count
        total += region.nbytes * count
        if total > _SIZE_T_MAX:
            raise transfer.UnsupportedError(
                "NIXL descriptor catalog byte count exceeds native size_t"
            )
    return total


def _same_full_block_layout(
    local: Sequence[transfer.BackendRegion],
    remote: Sequence[transfer.BackendRegion],
) -> bool:
    local_index = remote_index = 0
    local_remaining = remote_remaining = 0
    while local_index < len(local) and remote_index < len(remote):
        if local_remaining == 0:
            local_region = local[local_index]
            local_remaining = 1 if local_region.count is None else local_region.count
        if remote_remaining == 0:
            remote_region = remote[remote_index]
            remote_remaining = 1 if remote_region.count is None else remote_region.count
        if local_region.nbytes != remote_region.nbytes:
            return False
        consumed = min(local_remaining, remote_remaining)
        local_remaining -= consumed
        remote_remaining -= consumed
        if local_remaining == 0:
            local_index += 1
        if remote_remaining == 0:
            remote_index += 1
    return local_index == len(local) and remote_index == len(remote)


def _uniform_pair_nbytes(
    local: Sequence[transfer.BackendRegion],
    remote: Sequence[transfer.BackendRegion],
) -> int | None:
    local_nbytes = local[0].nbytes
    remote_nbytes = remote[0].nbytes
    if local_nbytes != remote_nbytes:
        return None
    if any(region.nbytes != local_nbytes for region in local[1:]):
        return None
    if any(region.nbytes != remote_nbytes for region in remote[1:]):
        return None
    return local_nbytes


def _resolved_regions_are_nonoverlapping(
    regions: Sequence[_ResolvedRegion],
) -> bool:
    """Check full catalogs in O(region-count) space without expansion.

    A strided region is represented by its bounding envelope. This is
    deliberately conservative when sparse envelopes from different regions
    interleave; callers can use explicit bounded index selections in that case.
    """

    intervals: dict[tuple[object, int, str], list[tuple[int, int]]] = {}
    for resolved in regions:
        region = resolved.region
        count = 1 if region.count is None else region.count
        stride = region.nbytes if region.stride is None else region.stride
        start = resolved.base + region.offset
        end = start + (count - 1) * stride + region.nbytes
        address_space: object = (
            id(resolved.peer) if resolved.peer is not None else "local"
        )
        intervals.setdefault(
            (address_space, resolved.device_id, resolved.memory_type), []
        ).append((start, end))
    for values in intervals.values():
        values.sort()
        if any(
            current[0] < previous[1] for previous, current in zip(values, values[1:])
        ):
            return False
    return True


def _materialize_full_indices(plan: _NixlPlan, side: str) -> np.ndarray:
    attribute = f"{side}_all_indices"
    cached = getattr(plan, attribute)
    if cached is not None:
        return cached
    count = getattr(plan, f"{side}_block_count")
    materialized = np.arange(count, dtype=np.int32)
    setattr(plan, attribute, materialized)
    return materialized


def _native_region_geometry(
    region: transfer.BackendRegion,
    base: int,
    registered_nbytes: int,
    side: str,
) -> tuple[int, int, int, int]:
    offset = _bounded_nonnegative_int(
        region.offset, f"{side} region offset", _SIZE_T_MAX
    )
    nbytes = _bounded_positive_int(region.nbytes, f"{side} region nbytes", _SIZE_T_MAX)
    if region.count is None:
        count = 1
        stride = nbytes
    else:
        count = _bounded_positive_int(region.count, f"{side} region count", _SIZE_T_MAX)
        stride = _bounded_positive_int(
            region.stride, f"{side} region stride", _SIZE_T_MAX
        )
        if stride < nbytes:
            raise transfer.InvalidRegionError(
                f"{side} region stride must be at least nbytes"
            )
    repeated = count - 1
    if repeated and stride > (_SIZE_T_MAX - nbytes) // repeated:
        raise transfer.InvalidRegionError(
            f"{side} strided region extent exceeds native size_t"
        )
    extent = repeated * stride + nbytes
    if offset > registered_nbytes or extent > registered_nbytes - offset:
        raise transfer.InvalidRegionError(
            f"{side} region exceeds its advertised NIXL registration"
        )
    if offset > _UINTPTR_MAX - base:
        raise transfer.InvalidRegionError(
            f"{side} region address exceeds native uintptr_t"
        )
    address = base + offset
    if extent > _UINTPTR_MAX - address:
        raise transfer.InvalidRegionError(f"{side} region end exceeds native uintptr_t")
    return address, nbytes, stride, count


def _normalize_memory_type(value: object) -> str:
    if not isinstance(value, str):
        raise transfer.InvalidRegionError("memory_type must be a string")
    normalized = {
        "cpu": "DRAM",
        "dram": "DRAM",
        "host": "DRAM",
        "host-pinned": "DRAM",
        "cuda": "VRAM",
        "vram": "VRAM",
    }.get(value.lower())
    if normalized is None:
        raise transfer.UnsupportedError(f"NIXL memory type {value!r} is unsupported")
    return normalized


def _device_id(device: str) -> int:
    kind, separator, index = device.partition(":")
    if kind == "cpu":
        return 0
    if kind != "cuda":
        raise transfer.UnsupportedError(f"NIXL device {device!r} is unsupported")
    if not separator:
        raise transfer.InvalidRegionError(
            "CUDA registrations require an explicit index"
        )
    try:
        parsed_index = int(index)
    except ValueError as error:
        raise transfer.InvalidRegionError("invalid CUDA device index") from error
    return _bounded_nonnegative_int(parsed_index, "CUDA device index", _UINT64_MAX)


def _ucx_backends(value: object, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of backend names")
    result = list(value)
    if result != ["UCX"]:
        raise transfer.UnsupportedError(
            f"experimental NIXL transfer supports exactly {name}=['UCX']"
        )
    return result


def _option_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _option_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _option_bounded_nonnegative_int(value: object, name: str, maximum: int) -> int:
    result = _option_nonnegative_int(value, name)
    if result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _option_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _nixl_thread_sync_mode(value: object, thread_mode: transfer.ThreadMode) -> str:
    if value is None:
        return "rw" if thread_mode is transfer.ThreadMode.MULTIPLE else "none"
    if not isinstance(value, str):
        raise TypeError("nixl_thread_sync_mode must be a string")
    if value not in {"none", "strict", "rw"}:
        raise ValueError(
            "nixl_thread_sync_mode must be one of 'none', 'strict', or 'rw'"
        )
    if value == "none" and thread_mode is transfer.ThreadMode.MULTIPLE:
        raise ValueError(
            "nixl_thread_sync_mode='none' is unsafe with ThreadMode.MULTIPLE"
        )
    return value


def _is_native_not_found(error: BaseException) -> bool:
    return (
        error.__class__.__name__ == "nixlNotFoundError"
        or "NIXL_ERR_NOT_FOUND" in str(error)
    )


def _positive_int_attr(value: object, name: str) -> int:
    try:
        attribute = getattr(value, name)
    except AttributeError as error:
        raise transfer.InvalidRegionError(
            f"registration does not expose a backend-safe {name}"
        ) from error
    return _positive_int(attribute, name)


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise transfer.InvalidRegionError(f"{name} must be positive")
    return result


def _bounded_positive_int(value: object, name: str, maximum: int) -> int:
    result = _positive_int(value, name)
    if result > maximum:
        raise transfer.InvalidRegionError(f"{name} exceeds its native integer range")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise transfer.InvalidRegionError(f"{name} must be a non-negative integer")
    return value


def _bounded_nonnegative_int(value: object, name: str, maximum: int) -> int:
    result = _nonnegative_int(value, name)
    if result > maximum:
        raise transfer.InvalidRegionError(f"{name} exceeds its native integer range")
    return result


def _validate_native_span(address: int, nbytes: int, name: str) -> None:
    _bounded_positive_int(address, f"{name} address", _UINTPTR_MAX)
    _bounded_positive_int(nbytes, f"{name} nbytes", _SIZE_T_MAX)
    if nbytes > _UINTPTR_MAX - address:
        raise transfer.InvalidRegionError(f"{name} end exceeds native uintptr_t")


def _require_native_plan_handle(handle: object | None) -> object:
    if handle is None:
        raise transfer.ClosedError("NIXL transfer plan has no native descriptors")
    return handle


def _require_type(value: object, expected: type[Any], name: str) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")
    return value


def _discard_identity(values: list[Any], target: object) -> None:
    for index, value in enumerate(values):
        if value is target:
            del values[index]
            return


def _add_exception_note(error: BaseException, note: str) -> None:
    """Preserve rollback context where Python 3.11 exception notes exist."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def _same_identity_set(left: Sequence[object], right: Sequence[object]) -> bool:
    return len(left) == len(right) and {id(value) for value in left} == {
        id(value) for value in right
    }


def _is_terminal(state: transfer.WorkState) -> bool:
    return (
        state is transfer.WorkState.COMPLETED
        or state is transfer.WorkState.FAILED
        or state is transfer.WorkState.CANCELLED
    )


def _handle_is_released(handle: object | None) -> bool:
    if handle is None:
        return True
    released = getattr(handle, "released", None)
    if released is None:
        released = getattr(handle, "_released", False)
    return bool(released)


def _slot_failure(message: str, cause: BaseException) -> transfer.BackendFailureError:
    failure = transfer.BackendFailureError(message)
    failure.__cause__ = cause
    return failure
