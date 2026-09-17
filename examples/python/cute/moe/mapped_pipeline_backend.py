# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-node, all-mapped NIXL/CuTe backend for :mod:`moe.pipeline`.

This module is the concrete bridge between the import-free host state machine
and the split dispatch/combine kernels.  Its performance contract is narrow on
purpose:

* one stable, fixed-capacity, UCX-registered ``uint8`` arena per rank;
* one process per GPU and a live process for every capacity slot;
* a process-local mapped pointer for every active peer and native peer atomics
  for every directed GPU pair; and
* all requested live-token shapes compiled and occupancy-qualified before the
  communicator becomes active.

Every backend-owned tensor is validated and converted to a strongly-owned CuTe
descriptor during admission.  The two bank descriptor sets are rebound to an
explicit generation/incarnation token only after a successful activation
quorum.  Caller tensors are deliberately not cached implicitly: PyTorch can
rebind their storage without changing Python object identity, so each enqueue
still validates their current metadata/span before converting them.
Operation-specific events are never retained in a descriptor cache.  Stable
eager streams may be explicitly bound by
:meth:`MappedPipelineBackend.prepare_submission_context`; the binding strongly
owns each exact stream object and caches its immutable CUstream wrapper plus a
bounded range of by-value operation-step wrappers.  Unprepared execution keeps
the validating wrapper-construction fallback.
Applications with a bounded set of stable operands may explicitly seal them
with :meth:`MappedPipelineBackend.prepare_external_tensor`.  A sealed hit uses
its cached shape, span, and descriptor without tensor metadata queries.  The
backend retains a strong reference; rebinding or resizing that tensor before
close violates the seal and is detectable with the cold
:meth:`MappedPipelineBackend.validate_prepared_bindings` hook.
For the minimum-overhead zero-copy path,
:meth:`MappedPipelineBackend.prepare_operation` additionally validates the
whole operand alias matrix and returns an opaque token.  Passing that token to
both enqueues removes even per-role identity-map lookups and hot span checks.
CUDA Graph replay is rejected: the mapped wire protocol embeds the operation
step by value, so replay would create an ABA publication.

The steady-state methods enqueue only GPU waits, retained CuTe launchers, and
preallocated CUDA events.  They do not allocate device storage, query an event,
read a device scalar, synchronize a stream, or call a NIXL host progress API.
Membership staging and commit are deliberately synchronous control-plane
operations.  Staging resolves and validates a coexisting candidate view;
commit drains both banks, converges every fixed-capacity process, rebases all
sequence/credit planes from precreated controls, and releases the old view only
after an all-rank activation quorum.

This is not an abrupt-failure recovery protocol.  A mapped load/store can fault
when its owner exits, and posted requestless work cannot be cancelled.  The
production ``fail_stop`` path therefore terminates the process without
releasing the view or registration.  Tests may inject a nonterminal terminator
to inspect that quarantine state.

Torch, CUDA, CuTe, and NIXL are imported lazily in ``preallocate`` so topology
planning and CPU-only unit tests can import this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

try:
    from .._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from .ll_protocol import (
        NUM_BANKS,
        UINT32_MAX,
        OperationEpoch,
        PipelineLLArenaLayout,
        StableSparseTopology,
    )
    from .pipeline import (
        GENERATION_REBASE_PLANES,
        CombineEvents,
        GenerationCommit,
        GenerationDrain,
        PipelineBankBuffers,
        PipelineError,
        PipelineFailed,
    )
except ImportError:  # Direct ``python examples/python/cute/...`` execution.
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from moe.ll_protocol import (  # type: ignore[no-redef]
        NUM_BANKS,
        UINT32_MAX,
        OperationEpoch,
        PipelineLLArenaLayout,
        StableSparseTopology,
    )
    from moe.pipeline import (  # type: ignore[no-redef]
        GENERATION_REBASE_PLANES,
        CombineEvents,
        GenerationCommit,
        GenerationDrain,
        PipelineBankBuffers,
        PipelineError,
        PipelineFailed,
    )


_TAG_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_UCX_POST_THREADS = 0
_UCX_NUM_WORKERS = 2
_DEFAULT_EVENT_POOL_DEPTH = 64
_FAIL_STOP_EXIT_CODE = 70
_FAIL_STOP_MESSAGE_CHARS = 512
_FAIL_STOP_DIAGNOSTIC_FALLBACK = b"NIXL_CUTE_FAIL_STOP diagnostic_unavailable=1\n"
# Caching one Python scalar wrapper per operation is intentionally bounded.
# Longer generations retain the uncached API or can be split at a quiescent
# process restart; silently attempting a multi-billion-object tuple is unsafe.
_MAX_PREPARED_STEP_COUNT = 1 << 20
_EXTERNAL_TENSOR_ROLES = (
    "activations",
    "topk_indices",
    "topk_weights",
    "combine_output",
    "expert_output",
)
WARP_SIZE = 32


def _emit_fail_stop_diagnostic(
    *, rank: int | None, state: str, error: BaseException
) -> None:
    """Best-effort single-write cause record for the terminating failure path."""

    try:
        error_type = f"{type(error).__module__}.{type(error).__qualname__}"[
            :_FAIL_STOP_MESSAGE_CHARS
        ]
        error_message = str(error)[:_FAIL_STOP_MESSAGE_CHARS]
        payload = {
            "schema_version": 1,
            "rank": rank,
            "state": state,
            "error_type": error_type,
            "error": error_message,
        }
        record = (
            "NIXL_CUTE_FAIL_STOP "
            + json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii", errors="strict")
        os.write(2, record)
    except BaseException:
        try:
            os.write(2, _FAIL_STOP_DIAGNOSTIC_FALLBACK)
        except BaseException:
            pass


class MappedBackendState(str, Enum):
    """Observable lifecycle of :class:`MappedPipelineBackend`."""

    NEW = "new"
    ACTIVE = "active"
    STAGED = "staged"
    DRAINING = "draining"
    COMMITTING = "committing"
    CLOSING = "closing"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class _DeviceDependencies:
    torch: Any
    cuda: Any
    cutlass: Any
    from_dlpack: Callable[[Any], Any]
    nixl_cute: Any
    kernels: Any
    bind_cooperative: Callable[..., tuple[Any, Any]]
    agent_factory: Callable[..., Any]
    agent_config_factory: Callable[..., Any]
    sync_mode_default: Any


@dataclass(frozen=True, slots=True)
class _EventSet:
    dispatch_ready: Any
    expert_ready: Any
    output_ready: Any
    bank_reusable: Any
    combine_events: CombineEvents = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "combine_events",
            CombineEvents(
                output_ready=self.output_ready,
                bank_reusable=self.bank_reusable,
            ),
        )


@dataclass(frozen=True, slots=True)
class _PreparedCandidateDescriptors:
    """Cold-path descriptors for a staged generation's private controls."""

    rank_mask: Any
    rank_incarnations: Any
    peer_bases: Any
    statuses: Any


@dataclass(slots=True)
class _CandidateGeneration:
    topology: StableSparseTopology
    view: Any
    rank_mask: Any
    rank_incarnations: Any
    peer_bases: Any
    statuses: Any
    prepared: _PreparedCandidateDescriptors | None = None
    mapped_pointer_evidence: dict[str, Any] | None = None
    resolved: bool = False
    released: bool = False


@dataclass(frozen=True, slots=True)
class _DrainToken:
    candidate: _CandidateGeneration
    event: Any


@dataclass(frozen=True, slots=True)
class _PreparedTensor:
    """One storage-stable tensor and its reusable CuTe argument."""

    name: str
    owner: Any
    argument: Any
    span: tuple[int, int]
    shape: tuple[int, ...]
    dtype: Any
    device: int
    storage_identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PreparedBankStorage:
    """CuTe descriptors for every tensor owned by one physical bank."""

    bank: int
    expert_input: _PreparedTensor
    expert_counts: _PreparedTensor
    source_info: _PreparedTensor
    layout_ranges: _PreparedTensor
    combine_stage: _PreparedTensor
    route_counts: _PreparedTensor


@dataclass(frozen=True, slots=True)
class _PreparedOwnedStorage:
    """Admission-time descriptors whose storage never changes while active."""

    arena: _PreparedTensor
    rank_mask: _PreparedTensor
    rank_incarnations: _PreparedTensor
    peer_bases: _PreparedTensor
    statuses: _PreparedTensor
    banks: tuple[_PreparedBankStorage, ...]
    dummy_activations: _PreparedTensor
    dummy_topk_indices: _PreparedTensor
    dummy_topk_weights: _PreparedTensor
    dummy_combine_output: _PreparedTensor


@dataclass(frozen=True, slots=True)
class _PreparedBankGeneration:
    """Fail-closed binding of stable descriptors to one active generation."""

    membership_generation: int
    source_incarnation: int
    generation_arg: Any
    source_incarnation_arg: Any
    owned: _PreparedOwnedStorage
    storage: _PreparedBankStorage


@dataclass(frozen=True, slots=True)
class _PreparedGeneration:
    membership_generation: int
    banks: tuple[_PreparedBankGeneration, ...]


@dataclass(frozen=True, slots=True)
class _PreparedLaunchStream:
    """Cold binding of one exact CUDA stream object and native handle."""

    owner: Any
    argument: Any
    wait_event: Any
    handle: int
    device: int


@dataclass(frozen=True, slots=True)
class _PreparedSubmissionContext:
    """Finite eager-launch plan built entirely before submission begins."""

    first_step: int
    steps: tuple["_PreparedStepLaunch", ...]
    bindings: dict[int, _PreparedLaunchStream]


@dataclass(frozen=True, slots=True)
class _PreparedStepLaunch:
    """Cold scalar/bank/event binding for one immutable operation step."""

    step: int
    bank: int
    step_argument: Any
    events: _EventSet


@dataclass(frozen=True, slots=True)
class _PreparedMappedLaunch:
    """All stable arguments needed by one prepared dispatch/expert/combine."""

    backend_cookie: object
    submission: _PreparedSubmissionContext
    operation: OperationEpoch
    generation: _PreparedGeneration
    bank_generation: _PreparedBankGeneration
    operation_binding: "PreparedMappedOperation"
    communication_stream: _PreparedLaunchStream
    expert_stream: _PreparedLaunchStream
    step_argument: Any
    events: _EventSet
    bank: int
    dispatch_launcher: Any
    combine_launcher: Any
    expert_launcher: Any
    activation_argument: Any
    topk_indices_argument: Any
    topk_weights_argument: Any
    combine_output_argument: Any


@dataclass(frozen=True, slots=True)
class PreparedMappedOperation:
    """Opaque, backend-issued fast-path binding for one stable operand set.

    Instances are created only by :meth:`MappedPipelineBackend.prepare_operation`.
    Their tensors are strongly owned by the backend's sealed-descriptor cache;
    callers must treat every field except :attr:`live_tokens` as private.
    """

    _backend_cookie: object
    _activations: _PreparedTensor
    _topk_indices: _PreparedTensor
    _topk_weights: _PreparedTensor
    _combine_output: _PreparedTensor

    @property
    def live_tokens(self) -> int:
        """Logical token count of this fully validated specialization."""

        return self._activations.shape[0]


def _load_device_dependencies() -> _DeviceDependencies:
    """Import the GPU stack only when a concrete backend is initialized."""

    import cuda.bindings.driver as cuda  # pylint: disable=import-outside-toplevel
    import cutlass  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from cutlass.cute.runtime import (  # pylint: disable=import-outside-toplevel
        from_dlpack,
    )

    import nixl.device.cute as nixl_cute  # pylint: disable=import-outside-toplevel
    from nixl import (  # pylint: disable=import-outside-toplevel
        nixl_agent,
        nixl_agent_config,
        nixl_thread_sync_t,
    )

    try:
        from .._cooperative import (  # pylint: disable=import-outside-toplevel
            bind_and_validate_cooperative_launch,
        )
        from . import pipeline_kernels  # pylint: disable=import-outside-toplevel
    except ImportError:  # Direct example execution.
        import moe.pipeline_kernels as pipeline_kernels  # type: ignore[no-redef]  # pylint: disable=import-outside-toplevel
        from _cooperative import (  # type: ignore[no-redef]  # pylint: disable=import-outside-toplevel
            bind_and_validate_cooperative_launch,
        )

    return _DeviceDependencies(
        torch=torch,
        cuda=cuda,
        cutlass=cutlass,
        from_dlpack=from_dlpack,
        nixl_cute=nixl_cute,
        kernels=pipeline_kernels,
        bind_cooperative=bind_and_validate_cooperative_launch,
        agent_factory=nixl_agent,
        agent_config_factory=nixl_agent_config,
        sync_mode_default=nixl_thread_sync_t.NIXL_THREAD_SYNC_DEFAULT,
    )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _device_region(tensor: Any) -> DeviceRegion:
    return DeviceRegion(
        int(tensor.data_ptr()),
        int(tensor.numel()) * int(tensor.element_size()),
        int(tensor.get_device()),
    )


def _decode_coordinates(
    gathered: dict[int, bytes], max_ranks: int
) -> tuple[PeerCoordinates, ...]:
    expected = set(range(max_ranks))
    if set(gathered) != expected:
        raise RuntimeError(
            "arena coordinate exchange did not cover every stable slot: "
            f"missing={sorted(expected - set(gathered))}, "
            f"extra={sorted(set(gathered) - expected)}"
        )
    result = tuple(
        PeerCoordinates.from_bytes(gathered[rank]) for rank in range(max_ranks)
    )
    if any(len(coordinates.regions) != 1 for coordinates in result):
        raise RuntimeError("each rank must publish exactly one registered arena")
    return result


class MappedPipelineBackend:
    """Production-shaped same-node implementation of ``PipelineBackend``.

    Parameters are intentionally admission-time facts.  ``devices`` maps every
    stable rank slot to the CUDA ordinal visible in each worker process.
    ``live_token_specializations`` is the complete set of logical batch sizes
    accepted by the hot path. ``worker_ctas`` is checked against each exact
    bound dispatch/combine specialization through its already-loaded CUDA
    library on the selected device.

    CUDA events come from a fixed ring.  Consumers must enqueue their event wait
    before ``event_pool_depth`` later operations reuse the same physical bank.
    The default permits 128 total operations of lag.  Tensor results themselves
    are caller-owned and have no such bank lifetime restriction.
    """

    def __init__(
        self,
        *,
        control: FileControlPlane,
        devices: Sequence[int],
        live_token_specializations: Iterable[int],
        worker_ctas: int,
        run_id: str,
        event_pool_depth: int = _DEFAULT_EVENT_POOL_DEPTH,
        timeout_s: float | None = None,
        terminator: Callable[[int], Any] | None = None,
        _dependencies: _DeviceDependencies | None = None,
    ) -> None:
        if not isinstance(control, FileControlPlane):
            raise TypeError("control must be a FileControlPlane")
        if isinstance(devices, (str, bytes)):
            raise TypeError("devices must be a sequence of CUDA ordinals")
        device_tuple = tuple(devices)
        if len(device_tuple) != control.world_size:
            raise ValueError("devices must contain one CUDA ordinal per stable rank")
        for ordinal in device_tuple:
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise ValueError("devices must contain non-negative CUDA ordinals")
        if len(set(device_tuple)) != len(device_tuple):
            raise ValueError("all-mapped workers require one distinct GPU per rank")
        if isinstance(live_token_specializations, (str, bytes)):
            raise TypeError("live_token_specializations must contain integers")
        specializations = tuple(sorted(set(live_token_specializations) | {0}))
        for value in specializations:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "live_token_specializations must contain non-negative integers"
                )
        _positive_int("worker_ctas", worker_ctas)
        _positive_int("event_pool_depth", event_pool_depth)
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        safe_run_id = _TAG_SAFE.sub("-", run_id.strip()).strip("-.")
        if not safe_run_id:
            raise ValueError("run_id must contain an alphanumeric character")
        actual_timeout = control.timeout_s if timeout_s is None else timeout_s
        if (
            isinstance(actual_timeout, bool)
            or not isinstance(actual_timeout, (int, float))
            or actual_timeout <= 0
        ):
            raise ValueError("timeout_s must be positive")
        if terminator is not None and not callable(terminator):
            raise TypeError("terminator must be callable")

        self.control = control
        self.devices = device_tuple
        self.live_token_specializations = specializations
        self.worker_ctas = worker_ctas
        self.run_id = safe_run_id
        self.event_pool_depth = event_pool_depth
        self.timeout_s = float(actual_timeout)
        self._terminator = os._exit if terminator is None else terminator
        self._dependencies_override = _dependencies

        self._state = MappedBackendState.NEW
        self._failed_error: BaseException | None = None
        self._deps: _DeviceDependencies | None = None
        self._rank: int | None = None
        self._layout: PipelineLLArenaLayout | None = None
        self._contract: Any = None
        self._topology: StableSparseTopology | None = None
        self._device: int | None = None
        self._arena: Any = None
        self._registration: Any = None
        self._agent: Any = None
        self._coordinates: tuple[PeerCoordinates, ...] = ()
        self._peer_names: tuple[str, ...] = ()
        self._active_candidate: _CandidateGeneration | None = None
        self._staged_candidate: _CandidateGeneration | None = None
        self._rank_mask: Any = None
        self._rank_incarnations: Any = None
        self._peer_bases: Any = None
        self._statuses: Any = None
        self._route_counts: Any = None
        self._rebase_planes: tuple[Any, ...] = ()
        self._control_stream: Any = None
        self._buffers: tuple[PipelineBankBuffers, ...] = ()
        self._event_pool: tuple[tuple[_EventSet, ...], ...] = ()
        self._drain_events: tuple[Any, ...] = ()
        self._latest_bank_events: list[Any | None] = [None] * NUM_BANKS
        self._compiled_resolve: Any = None
        self._compiled_dispatch: dict[int, Any] = {}
        self._compiled_combine: dict[int, Any] = {}
        self._compiled_expert: Any = None
        self._occupancy: dict[int, dict[str, dict[str, Any]]] = {}
        self._dummy: dict[str, Any] = {}
        self._prepared_owned: _PreparedOwnedStorage | None = None
        self._prepared_generation: _PreparedGeneration | None = None
        self._prepared_external: dict[str, dict[int, _PreparedTensor]] = {
            role: {} for role in _EXTERNAL_TENSOR_ROLES
        }
        self._prepared_submission: _PreparedSubmissionContext | None = None
        self._operation_cookie = object()
        self._native_atomic_evidence: Any = None
        self._mapped_pointer_evidence: dict[str, Any] | None = None

    @property
    def state(self) -> MappedBackendState:
        return self._state

    @property
    def arena(self) -> Any:
        """The single registered arena, available after ``preallocate``."""

        if self._arena is None:
            raise PipelineError("backend has not been initialized")
        return self._arena

    @property
    def occupancy_evidence(self) -> dict[int, dict[str, dict[str, Any]]]:
        """Return JSON-ready exact compiled-CUBIN occupancy evidence."""

        return {
            live: {kind: dict(values) for kind, values in by_kind.items()}
            for live, by_kind in self._occupancy.items()
        }

    @property
    def native_atomic_evidence(self) -> Any:
        return self._native_atomic_evidence

    @property
    def mapped_pointer_evidence(self) -> dict[str, Any] | None:
        """JSON-safe setup evidence without exposing process-local addresses."""

        if self._mapped_pointer_evidence is None:
            return None
        # Return a detached JSON-shaped copy so callers cannot rewrite the
        # backend's admission evidence in-place.
        return json.loads(json.dumps(self._mapped_pointer_evidence))

    def prepare_submission_context(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> _PreparedSubmissionContext:
        """Cold-bind an exact eager-stream set and finite operation-step range.

        Every operation-specific prepared launch made from the returned context
        must use one of the exact stream objects in ``streams`` and a step in
        ``[first_step, first_step + step_count)``. The backend strongly owns
        those objects and caches their CUstream wrappers, bound wait callables,
        and every Cutlass ``Uint32`` step wrapper. Native handles and device
        bindings must remain immutable until healthy :meth:`close`; values
        outside the admitted range are rejected while constructing the cold
        launch record, before any event wait or kernel launch. The direct/raw
        compatibility entry points remain independent of this finite context.

        The finite cache is capped at :data:`_MAX_PREPARED_STEP_COUNT` to avoid
        turning an adversarial range into an unbounded host allocation. A
        fully posted finite plan may be followed by another plan: a different
        context transactionally replaces the prior wrappers, which have
        already been copied into their CUDA launches. An unconsumed pipeline
        proof retains the old context identity and therefore fails closed if a
        later preparation supersedes it. Omit this preparation to retain the
        ordinary per-launch fallback.
        """

        self._require_active_backend()
        unique_streams = self._normalize_streams(streams)
        if isinstance(first_step, bool) or not isinstance(first_step, int):
            raise TypeError("first_step must be an integer")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("step_count must be an integer")
        if first_step < 0 or first_step > UINT32_MAX:
            raise ValueError("first_step must fit uint32")
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        if step_count > _MAX_PREPARED_STEP_COUNT:
            raise ValueError(
                "step_count exceeds the bounded prepared-wrapper capacity "
                f"{_MAX_PREPARED_STEP_COUNT}"
            )
        stop_step = first_step + step_count
        if stop_step > UINT32_MAX + 1:
            raise ValueError("prepared operation-step range exceeds uint32")

        current = self._prepared_submission
        if current is not None:
            for binding in current.bindings.values():
                if not self._prepared_stream_binding_matches(binding):
                    self._fail_changed_prepared_stream(binding)
            same_range = (
                current.first_step == first_step and len(current.steps) == step_count
            )
            same_streams = len(current.bindings) == len(unique_streams) and all(
                (binding := current.bindings.get(id(stream))) is not None
                and binding.owner is stream
                for stream in unique_streams
            )
            if same_range and same_streams:
                return current

        bindings: dict[int, _PreparedLaunchStream] = {}
        native_handles: dict[int, Any] = {}
        for stream in unique_streams:
            binding = self._prepare_launch_stream(stream)
            other = native_handles.get(binding.handle)
            if other is not None and other is not stream:
                raise ValueError(
                    "distinct stream objects in one submission context must not "
                    "alias the same native CUDA stream"
                )
            native_handles[binding.handle] = stream
            bindings[id(stream)] = binding
        steps = tuple(
            _PreparedStepLaunch(
                step=step,
                bank=step & 1,
                step_argument=self._deps.cutlass.Uint32(step),
                events=self._event_pool[step & 1][
                    (step // NUM_BANKS) % self.event_pool_depth
                ],
            )
            for step in range(first_step, stop_step)
        )
        prepared = _PreparedSubmissionContext(
            first_step=first_step,
            steps=steps,
            bindings=bindings,
        )
        self._prepared_submission = prepared
        return prepared

    def validate_prepared_submission_context(
        self, context: _PreparedSubmissionContext
    ) -> None:
        """Identity-check one pipeline proof without rebuilding cold bindings."""

        self._require_active_backend()
        if context is not self._prepared_submission:
            raise PipelineError(
                "submission preflight backend context was replaced before use"
            )

    def validate_submission_context(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> None:
        """Reject an invalid eager operation range or CUDA capture before post.

        This is deliberately a cold/batch check rather than a per-kernel query.
        The entire half-open operation range is checked against the finite
        prepared step-wrapper plan before any stream binding or capture query.
        It probes both PyTorch's current-stream capture state and every unique
        scheduled stream through ``cuStreamIsCapturing``.  CUDA Graph replay is
        unsupported because replaying a kernel with a by-value operation step
        would violate the publication protocol's monotonic epoch contract. This
        is a point-in-time observation, not a stream lock; a deferred pipeline
        token requires exclusive submission ownership and no intervening CUDA
        operation on these streams before consumption.
        """

        self._require_active_backend()
        if isinstance(first_step, bool) or not isinstance(first_step, int):
            raise TypeError("first_step must be an integer")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("step_count must be an integer")
        if first_step < 0 or first_step > UINT32_MAX:
            raise ValueError("first_step must fit uint32")
        if step_count <= 0:
            raise ValueError("step_count must be a positive integer")
        stop_step = first_step + step_count
        if stop_step > UINT32_MAX + 1:
            raise ValueError("submission operation-step range exceeds uint32")

        prepared = self._prepared_submission
        if prepared is not None:
            prepared_stop = prepared.first_step + len(prepared.steps)
            if first_step < prepared.first_step or stop_step > prepared_stop:
                raise PipelineError(
                    "submission operation-step range is outside the prepared context"
                )

        unique_streams = self._normalize_streams(streams)
        try:
            current_capture = self._deps.torch.cuda.is_current_stream_capturing()
        except Exception as error:
            raise RuntimeError(
                "torch.cuda.is_current_stream_capturing failed"
            ) from error
        if not isinstance(current_capture, bool):
            raise RuntimeError(
                "torch.cuda.is_current_stream_capturing returned a malformed result"
            )
        if current_capture:
            raise PipelineError("CUDA Graph capture/replay is unsupported")

        query_bindings: list[_PreparedLaunchStream] = []
        if prepared is not None:
            for stream in unique_streams:
                binding = prepared.bindings.get(id(stream))
                if binding is None or binding.owner is not stream:
                    raise PipelineError(
                        "submission validation received an unprepared stream"
                    )
                if not self._prepared_stream_binding_matches(binding):
                    self._fail_changed_prepared_stream(binding)
                query_bindings.append(binding)
        else:
            query_bindings.extend(
                self._prepare_launch_stream(stream) for stream in unique_streams
            )

        for binding in query_bindings:
            capture_status = self._checked_stream_capture_status(binding.argument)
            if capture_status != self._capture_status_none():
                raise PipelineError(
                    "CUDA Graph capture/replay is unsupported; scheduled stream "
                    f"handle {binding.handle} has capture status {capture_status!r}"
                )

    def prepare_external_tensor(self, tensor: Any, *, role: str) -> None:
        """Seal one stable caller tensor and cache its CuTe descriptor.

        ``role`` must be one of ``activations``, ``topk_indices``,
        ``topk_weights``, ``combine_output``, or ``expert_output``. The tensor
        object, its storage, dtype, shape, device, and contiguous layout must
        remain unchanged until :meth:`close`. Values may change, with normal
        stream/event ordering. The backend keeps a strong owner reference.

        This is a cold preparation API: call it for a bounded operand set
        before entering a measured eager loop. A cache hit in enqueue
        uses the retained descriptor and span without querying tensor metadata.
        Ordinary unsealed tensors continue through full per-call validation.
        """

        self._require_active_backend()
        prepared, is_new = self._prepare_or_validate_external(tensor, role)
        if is_new:
            self._prepared_external[role][id(tensor)] = prepared

    def prepare_operation(
        self,
        activations: Any,
        topk_indices: Any,
        topk_weights: Any,
        combine_output: Any,
    ) -> PreparedMappedOperation:
        """Seal and jointly validate one reusable zero-copy operation.

        This cold API converts each external operand exactly once and validates
        its complete alias matrix before publishing an opaque token.  Passing
        the token back through ``prepared_operation=`` selects the smallest hot
        path: backend/caller identity checks followed directly by GPU event
        waits and the retained launcher.  Combine token use intentionally
        requires the exact backend-owned ``combine_stage`` zero-copy output;
        callers with an external expert output use the ordinary validated path.

        The exact tensor objects and their storage, dtype, shape, device, and
        contiguous layout are sealed until healthy :meth:`close`.  Tensor
        values may change when ordered by the normal readiness events.
        """

        self._require_active_backend()
        requested = (
            ("activations", activations),
            ("topk_indices", topk_indices),
            ("topk_weights", topk_weights),
            ("combine_output", combine_output),
        )
        bindings: dict[str, _PreparedTensor] = {}
        additions: list[tuple[str, int, _PreparedTensor]] = []
        for role, tensor in requested:
            prepared, is_new = self._prepare_or_validate_external(tensor, role)
            bindings[role] = prepared
            if is_new:
                additions.append((role, id(tensor), prepared))

        live_tokens = bindings["activations"].shape[0]
        expected_shapes = {
            "topk_indices": (live_tokens, self._contract.top_k),
            "topk_weights": (live_tokens, self._contract.top_k),
            "combine_output": (live_tokens, self._contract.hidden_size),
        }
        for role, expected in expected_shapes.items():
            if bindings[role].shape != expected:
                raise ValueError(
                    f"{role} shape {bindings[role].shape} does not match "
                    f"operation shape {expected}"
                )
        self._require_disjoint_spans(
            tuple((role, bindings[role].span) for role, _ in requested),
            permit_empty=True,
        )
        # Publish only after every descriptor and the joint alias contract have
        # succeeded, so a rejected operation does not leave a partial seal.
        for role, key, prepared in additions:
            self._prepared_external[role][key] = prepared
        return PreparedMappedOperation(
            _backend_cookie=self._operation_cookie,
            _activations=bindings["activations"],
            _topk_indices=bindings["topk_indices"],
            _topk_weights=bindings["topk_weights"],
            _combine_output=bindings["combine_output"],
        )

    def prepare_operation_launch(
        self,
        *,
        submission_context: Any,
        operation: OperationEpoch,
        source_incarnation: int,
        topology: StableSparseTopology,
        buffers: PipelineBankBuffers,
        prepared_operation: Any,
        communication_stream: Any,
        expert_stream: Any,
    ) -> _PreparedMappedLaunch:
        """Cold-bind every stable scalar, stream, bank, event, and launcher.

        The returned token is deliberately operation-specific. Its stream and
        scalar wrappers, physical bank, ring events, and live-shape launchers
        are selected here so the post-D0 path never derives them from a Python
        object identity or operation-step arithmetic.
        """

        self._require_active_backend()
        context = self._prepared_submission
        if submission_context is not context or context is None:
            raise PipelineError("operation launch used a stale submission context")
        if not isinstance(prepared_operation, PreparedMappedOperation):
            raise TypeError("prepared_operation was not issued by this backend")
        if prepared_operation._backend_cookie is not self._operation_cookie:
            raise PipelineError(
                "prepared_operation is stale or belongs to another backend"
            )
        bank_generation = self._require_hot(
            operation,
            source_incarnation,
            topology,
            buffers,
        )
        generation = self._prepared_generation
        if generation is None:
            raise PipelineError("active generation has no prepared CuTe descriptors")
        offset = operation.step - context.first_step
        if not 0 <= offset < len(context.steps):
            raise PipelineError("operation step is outside the prepared context")
        step = context.steps[offset]
        if step.step != operation.step or step.bank != operation.bank:
            raise AssertionError("prepared step/bank binding is inconsistent")

        communication = context.bindings.get(id(communication_stream))
        expert = context.bindings.get(id(expert_stream))
        if communication is None or communication.owner is not communication_stream:
            raise PipelineError(
                "operation launch used an unprepared communication stream"
            )
        if expert is None or expert.owner is not expert_stream:
            raise PipelineError("operation launch used an unprepared expert stream")

        live_tokens = prepared_operation.live_tokens
        dispatch_launcher = self._compiled_dispatch.get(live_tokens)
        combine_launcher = self._compiled_combine.get(live_tokens)
        if dispatch_launcher is None or combine_launcher is None:
            raise PipelineError(
                f"live token count {live_tokens} was not compiled before activation"
            )
        owned = bank_generation.owned
        return _PreparedMappedLaunch(
            backend_cookie=self._operation_cookie,
            submission=context,
            operation=operation,
            generation=generation,
            bank_generation=bank_generation,
            operation_binding=prepared_operation,
            communication_stream=communication,
            expert_stream=expert,
            step_argument=step.step_argument,
            events=step.events,
            bank=step.bank,
            dispatch_launcher=dispatch_launcher,
            combine_launcher=combine_launcher,
            expert_launcher=self._compiled_expert,
            activation_argument=(
                owned.dummy_activations.argument
                if live_tokens == 0
                else prepared_operation._activations.argument
            ),
            topk_indices_argument=(
                owned.dummy_topk_indices.argument
                if live_tokens == 0
                else prepared_operation._topk_indices.argument
            ),
            topk_weights_argument=(
                owned.dummy_topk_weights.argument
                if live_tokens == 0
                else prepared_operation._topk_weights.argument
            ),
            combine_output_argument=(
                owned.dummy_combine_output.argument
                if live_tokens == 0
                else prepared_operation._combine_output.argument
            ),
        )

    def _prepare_or_validate_external(
        self, tensor: Any, role: str
    ) -> tuple[_PreparedTensor, bool]:
        """Return a cold validated binding and whether it still needs publish."""

        if role not in self._prepared_external:
            raise ValueError(f"role must be one of {', '.join(_EXTERNAL_TENSOR_ROLES)}")
        key = id(tensor)
        existing = self._prepared_external[role].get(key)
        if existing is not None:
            if existing.owner is not tensor:
                raise AssertionError("sealed tensor identity key was reused")
            if not self._prepared_binding_matches(existing):
                self._fail_changed_prepared_binding(existing)
            return existing, False
        return self._prepare_external_descriptor(tensor, role), True

    def validate_prepared_bindings(self) -> None:
        """Cold/debug check for forbidden storage rebinding.

        Invoke only at a quiescent control boundary. Any mismatch is terminal:
        a cached pointer may already have reached asynchronous device work, so
        the backend quarantines resources through :meth:`fail_stop`.
        """

        self._require_active_backend()
        owned = self._require_prepared_owned()
        for prepared in self._iter_owned_descriptors(owned):
            if not self._prepared_binding_matches(prepared):
                self._fail_changed_prepared_binding(prepared)
        for by_identity in self._prepared_external.values():
            for prepared in by_identity.values():
                if not self._prepared_binding_matches(prepared):
                    self._fail_changed_prepared_binding(prepared)
        submission = self._prepared_submission
        if submission is not None:
            for binding in submission.bindings.values():
                if not self._prepared_stream_binding_matches(binding):
                    self._fail_changed_prepared_stream(binding)

    def _prepare_external_descriptor(self, tensor: Any, role: str) -> _PreparedTensor:
        torch = self._deps.torch
        contract = self._contract
        shape = tuple(tensor.shape)
        if role == "expert_output":
            dtype = torch.bfloat16
            expected = (
                contract.experts_per_rank,
                contract.max_ranks * contract.token_capacity,
                contract.hidden_size,
            )
            live_tokens = None
        else:
            if len(shape) != 2:
                raise ValueError(f"{role} must be a rank-2 tensor, got {shape}")
            live_tokens = shape[0]
            if live_tokens not in self.live_token_specializations:
                raise PipelineError(
                    f"{role} live token count {live_tokens} was not compiled"
                )
            if role in ("activations", "combine_output"):
                dtype = torch.bfloat16
                expected = (live_tokens, contract.hidden_size)
            elif role == "topk_indices":
                dtype = torch.int32
                expected = (live_tokens, contract.top_k)
            else:
                dtype = torch.float32
                expected = (live_tokens, contract.top_k)
        self._validate_tensor(tensor, role, dtype)
        self._validate_shape(tensor, role, expected)
        if role in ("activations", "combine_output", "expert_output"):
            self._require_aligned(role, tensor)
        span = self._memory_span(tensor)
        self._require_outside_owned_storage(role, span)
        argument = None if live_tokens == 0 else self._tensor_arg(tensor)
        return _PreparedTensor(
            name=f"sealed {role}",
            owner=tensor,
            argument=argument,
            span=span,
            shape=expected,
            dtype=dtype,
            device=int(tensor.get_device()),
            storage_identity=self._tensor_storage_identity(tensor),
        )

    @staticmethod
    def _iter_owned_descriptors(
        owned: _PreparedOwnedStorage,
    ) -> Iterable[_PreparedTensor]:
        yield owned.arena
        yield owned.rank_mask
        yield owned.rank_incarnations
        yield owned.peer_bases
        yield owned.statuses
        for bank in owned.banks:
            yield bank.expert_input
            yield bank.expert_counts
            yield bank.source_info
            yield bank.layout_ranges
            yield bank.combine_stage
            yield bank.route_counts
        yield owned.dummy_activations
        yield owned.dummy_topk_indices
        yield owned.dummy_topk_weights
        yield owned.dummy_combine_output

    def _fail_changed_prepared_binding(self, prepared: _PreparedTensor) -> None:
        error = PipelineError(
            f"{prepared.name} storage changed after its CuTe descriptor was sealed"
        )
        self.fail_stop(error)
        raise error

    def preallocate(
        self,
        *,
        rank: int,
        layout: PipelineLLArenaLayout,
        topology: StableSparseTopology,
    ) -> Sequence[PipelineBankBuffers]:
        """Collectively allocate, register, connect, compile, and activate."""

        if self._state != MappedBackendState.NEW:
            raise PipelineError("MappedPipelineBackend.preallocate is one-shot")
        if rank != self.control.rank:
            raise ValueError("rank must match the FileControlPlane rank")
        if layout.max_ranks != self.control.world_size:
            raise ValueError("layout capacity must match control-plane world size")
        if topology.max_ranks != layout.max_ranks:
            raise ValueError("topology and layout rank capacity differ")
        if topology.experts_per_rank != layout.experts_per_rank:
            raise ValueError("topology and layout expert geometry differ")
        if layout.element_size != 2:
            raise ValueError("the mapped kernel specialization requires BF16 elements")
        if any(value > layout.num_tokens for value in self.live_token_specializations):
            raise ValueError("a live-token specialization exceeds arena capacity")
        self._validate_live_standby_topology(topology, initial=True)

        deps = self._dependencies_override or _load_device_dependencies()
        contract = deps.kernels.KernelContract(
            max_ranks=layout.max_ranks,
            experts_per_rank=layout.experts_per_rank,
            token_capacity=layout.num_tokens,
            top_k=layout.top_k,
            hidden_size=layout.hidden_size,
        )
        torch = deps.torch
        device = self.devices[rank]

        self._deps = deps
        self._rank = rank
        self._layout = layout
        self._contract = contract
        self._topology = topology
        self._device = device
        self._validate_configuration_quorum(topology)
        torch.cuda.set_device(device)
        self._native_atomic_evidence = deps.nixl_cute.require_peer_native_atomics(
            self.devices, accessing_devices=(device,)
        )

        try:
            self._initialize_storage()
            self._prepare_owned_descriptors()
            self._initialize_agent_and_metadata()
            self._compile_specializations()
            candidate = self._make_candidate(topology)
            self._resolve_candidate(candidate)
            self._install_initial_candidate(candidate)
        except BaseException as error:
            # Registration/view state can be uncertain once setup reaches the
            # device.  Do not attempt broad rollback here; external process
            # teardown is the only universally safe cleanup after setup error.
            self.fail_stop(error)
            raise
        self._state = MappedBackendState.ACTIVE
        return self._buffers

    def _initialize_storage(self) -> None:
        deps, layout, contract, device = self._required_setup()
        torch = deps.torch
        with torch.cuda.device(device):
            self._control_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(self._control_stream):
                self._arena = torch.empty(
                    layout.arena_nbytes, dtype=torch.uint8, device=device
                )
                self._arena.zero_()
                self._peer_bases = torch.zeros(
                    contract.max_ranks, dtype=torch.uint64, device=device
                )
                self._statuses = torch.zeros(
                    contract.max_ranks, dtype=torch.int32, device=device
                )
                expert_counts = torch.zeros(
                    (NUM_BANKS, contract.experts_per_rank),
                    dtype=torch.int32,
                    device=device,
                )
                self._route_counts = torch.zeros(
                    (NUM_BANKS, contract.max_ranks * contract.experts_per_rank),
                    dtype=torch.uint64,
                    device=device,
                )
                self._dummy = {
                    "activations": torch.empty(
                        (1, contract.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    "topk_indices": torch.empty(
                        (1, contract.top_k), dtype=torch.int32, device=device
                    ),
                    "topk_weights": torch.empty(
                        (1, contract.top_k), dtype=torch.float32, device=device
                    ),
                    "combine_output": torch.empty(
                        (1, contract.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                }
            self._control_stream.synchronize()

            self._rank_mask = self._arena_view(
                "rank_mask", torch.int64, (contract.max_ranks,)
            )
            self._rank_incarnations = self._arena_view(
                "rank_incarnation", torch.uint64, (contract.max_ranks,)
            )
            self._rebase_planes = tuple(
                self._arena.narrow(
                    0, layout.region(name).offset, layout.region(name).nbytes
                )
                for name in GENERATION_REBASE_PLANES
            )
            buffers: list[PipelineBankBuffers] = []
            dense_rows = contract.max_ranks * contract.token_capacity
            for bank in range(NUM_BANKS):
                buffers.append(
                    PipelineBankBuffers(
                        bank=bank,
                        expert_input=self._arena_bank_view(
                            "expert_input",
                            bank,
                            torch.bfloat16,
                            (
                                contract.experts_per_rank,
                                dense_rows,
                                contract.hidden_size,
                            ),
                        ),
                        expert_counts=expert_counts[bank],
                        source_info=self._arena_bank_view(
                            "dispatch_src_info",
                            bank,
                            torch.int32,
                            (contract.experts_per_rank, dense_rows, 4),
                        ),
                        layout_ranges=self._arena_bank_view(
                            "dispatch_layout_range",
                            bank,
                            torch.uint64,
                            (contract.experts_per_rank, contract.max_ranks),
                        ),
                        combine_stage=self._arena_bank_view(
                            "combine_stage",
                            bank,
                            torch.bfloat16,
                            (
                                contract.experts_per_rank,
                                dense_rows,
                                contract.hidden_size,
                            ),
                        ),
                    )
                )
            self._buffers = tuple(buffers)
            self._event_pool = tuple(
                tuple(
                    _EventSet(
                        dispatch_ready=torch.cuda.Event(enable_timing=False),
                        expert_ready=torch.cuda.Event(enable_timing=False),
                        output_ready=torch.cuda.Event(enable_timing=False),
                        bank_reusable=torch.cuda.Event(enable_timing=False),
                    )
                    for _ in range(self.event_pool_depth)
                )
                for _ in range(NUM_BANKS)
            )
            self._drain_events = tuple(
                torch.cuda.Event(enable_timing=False) for _ in range(NUM_BANKS)
            )
            # torch.cuda.Event creates the native event lazily. Materialize
            # every ring/drain event now so the first hot record cannot allocate
            # a hidden CUDA object. CUDA Graph replay is explicitly unsupported:
            # by-value operation epochs would otherwise repeat stale sequences.
            with torch.cuda.stream(self._control_stream):
                for bank_events in self._event_pool:
                    for events in bank_events:
                        events.dispatch_ready.record(self._control_stream)
                        events.expert_ready.record(self._control_stream)
                        events.output_ready.record(self._control_stream)
                        events.bank_reusable.record(self._control_stream)
                for event in self._drain_events:
                    event.record(self._control_stream)
            self._control_stream.synchronize()

    def _arena_view(self, name: str, dtype: Any, shape: tuple[int, ...]) -> Any:
        layout = self._require_layout()
        span = layout.region(name)
        view = self._arena.narrow(0, span.offset, span.nbytes).view(dtype)
        expected = 1
        for extent in shape:
            expected *= extent
        if int(view.numel()) != expected:
            raise AssertionError(f"arena region {name} does not match typed shape")
        return view.reshape(shape)

    def _arena_bank_view(
        self, name: str, bank: int, dtype: Any, shape: tuple[int, ...]
    ) -> Any:
        layout = self._require_layout()
        span = layout.bank_region(name, bank)
        view = self._arena.narrow(0, span.offset, span.nbytes).view(dtype)
        expected = 1
        for extent in shape:
            expected *= extent
        if int(view.numel()) != expected:
            raise AssertionError(f"arena bank region {name}[{bank}] has wrong shape")
        return view.reshape(shape)

    def _prepare_tensor_descriptor(
        self,
        name: str,
        tensor: Any,
        dtype: Any,
        shape: tuple[int, ...],
    ) -> _PreparedTensor:
        """Validate backend-owned storage once, then retain its CuTe wrapper."""

        self._validate_tensor(tensor, name, dtype)
        self._validate_shape(tensor, name, shape)
        return _PreparedTensor(
            name=name,
            owner=tensor,
            argument=self._tensor_arg(tensor),
            span=self._memory_span(tensor),
            shape=shape,
            dtype=dtype,
            device=int(tensor.get_device()),
            storage_identity=self._tensor_storage_identity(tensor),
        )

    @staticmethod
    def _require_exact_arena_region(
        name: str,
        prepared: _PreparedTensor,
        arena: _PreparedTensor,
        region: Any,
    ) -> None:
        expected = (
            arena.span[0] + int(region.offset),
            arena.span[0] + int(region.offset) + int(region.nbytes),
        )
        if prepared.span != expected:
            raise AssertionError(
                f"backend-owned {name} view does not exactly cover its arena region"
            )

    def _prepare_owned_descriptors(self) -> None:
        """Prepare all stable CuTe tensor arguments on the cold admission path.

        PyTorch tensors can be rebound to new storage without changing their
        Python identity, so caller-owned operands are intentionally excluded.
        Every tensor captured here is privately allocated by this backend and
        remains strongly owned until a healthy close or fail-stop teardown.
        """

        if self._prepared_owned is not None:
            raise PipelineError("backend-owned descriptors were already prepared")
        deps, layout, contract, _ = self._required_setup()
        torch = deps.torch
        dense_rows = contract.max_ranks * contract.token_capacity
        arena = self._prepare_tensor_descriptor(
            "arena", self._arena, torch.uint8, (layout.arena_nbytes,)
        )
        self._require_aligned("arena", self._arena)
        rank_mask = self._prepare_tensor_descriptor(
            "rank_mask",
            self._rank_mask,
            torch.int64,
            (contract.max_ranks,),
        )
        rank_incarnations = self._prepare_tensor_descriptor(
            "rank_incarnations",
            self._rank_incarnations,
            torch.uint64,
            (contract.max_ranks,),
        )
        self._require_exact_arena_region(
            "rank_mask", rank_mask, arena, layout.region("rank_mask")
        )
        self._require_exact_arena_region(
            "rank_incarnations",
            rank_incarnations,
            arena,
            layout.region("rank_incarnation"),
        )
        peer_bases = self._prepare_tensor_descriptor(
            "peer_bases",
            self._peer_bases,
            torch.uint64,
            (contract.max_ranks,),
        )
        statuses = self._prepare_tensor_descriptor(
            "statuses", self._statuses, torch.int32, (contract.max_ranks,)
        )

        prepared_banks: list[_PreparedBankStorage] = []
        outside_arena: list[tuple[str, Any]] = [
            ("peer_bases", self._peer_bases),
            ("statuses", self._statuses),
        ]
        for bank, buffers in enumerate(self._buffers):
            expert_input = self._prepare_tensor_descriptor(
                f"expert_input[{bank}]",
                buffers.expert_input,
                torch.bfloat16,
                (
                    contract.experts_per_rank,
                    dense_rows,
                    contract.hidden_size,
                ),
            )
            expert_counts = self._prepare_tensor_descriptor(
                f"expert_counts[{bank}]",
                buffers.expert_counts,
                torch.int32,
                (contract.experts_per_rank,),
            )
            source_info = self._prepare_tensor_descriptor(
                f"source_info[{bank}]",
                buffers.source_info,
                torch.int32,
                (contract.experts_per_rank, dense_rows, 4),
            )
            layout_ranges = self._prepare_tensor_descriptor(
                f"layout_ranges[{bank}]",
                buffers.layout_ranges,
                torch.uint64,
                (contract.experts_per_rank, contract.max_ranks),
            )
            combine_stage = self._prepare_tensor_descriptor(
                f"combine_stage[{bank}]",
                buffers.combine_stage,
                torch.bfloat16,
                (
                    contract.experts_per_rank,
                    dense_rows,
                    contract.hidden_size,
                ),
            )
            route_owner = self._route_counts[bank]
            route_counts = self._prepare_tensor_descriptor(
                f"route_counts[{bank}]",
                route_owner,
                torch.uint64,
                (contract.max_ranks * contract.experts_per_rank,),
            )
            self._require_aligned(f"route_counts[{bank}]", route_owner)
            for region_name, prepared in (
                ("expert_input", expert_input),
                ("dispatch_src_info", source_info),
                ("dispatch_layout_range", layout_ranges),
                ("combine_stage", combine_stage),
            ):
                self._require_exact_arena_region(
                    f"{region_name}[{bank}]",
                    prepared,
                    arena,
                    layout.bank_region(region_name, bank),
                )
            outside_arena.extend(
                (
                    (f"expert_counts[{bank}]", buffers.expert_counts),
                    (f"route_counts[{bank}]", route_owner),
                )
            )
            prepared_banks.append(
                _PreparedBankStorage(
                    bank=bank,
                    expert_input=expert_input,
                    expert_counts=expert_counts,
                    source_info=source_info,
                    layout_ranges=layout_ranges,
                    combine_stage=combine_stage,
                    route_counts=route_counts,
                )
            )

        dummy_activations = self._prepare_tensor_descriptor(
            "dummy activations",
            self._dummy["activations"],
            torch.bfloat16,
            (1, contract.hidden_size),
        )
        dummy_topk_indices = self._prepare_tensor_descriptor(
            "dummy topk_indices",
            self._dummy["topk_indices"],
            torch.int32,
            (1, contract.top_k),
        )
        dummy_topk_weights = self._prepare_tensor_descriptor(
            "dummy topk_weights",
            self._dummy["topk_weights"],
            torch.float32,
            (1, contract.top_k),
        )
        dummy_combine_output = self._prepare_tensor_descriptor(
            "dummy combine_output",
            self._dummy["combine_output"],
            torch.bfloat16,
            (1, contract.hidden_size),
        )
        outside_arena.extend(
            (
                ("dummy activations", dummy_activations.owner),
                ("dummy topk_indices", dummy_topk_indices.owner),
                ("dummy topk_weights", dummy_topk_weights.owner),
                ("dummy combine_output", dummy_combine_output.owner),
            )
        )
        for name, tensor in outside_arena:
            self._require_outside_arena(name, tensor)
        self._require_disjoint(tuple(outside_arena), permit_empty=False)

        self._prepared_owned = _PreparedOwnedStorage(
            arena=arena,
            rank_mask=rank_mask,
            rank_incarnations=rank_incarnations,
            peer_bases=peer_bases,
            statuses=statuses,
            banks=tuple(prepared_banks),
            dummy_activations=dummy_activations,
            dummy_topk_indices=dummy_topk_indices,
            dummy_topk_weights=dummy_topk_weights,
            dummy_combine_output=dummy_combine_output,
        )

    def _activate_prepared_generation(self, topology: StableSparseTopology) -> None:
        """Bind immutable descriptors and scalar wrappers to one generation."""

        if self._prepared_generation is not None:
            raise PipelineError(
                "prepared generation must be invalidated before activation"
            )
        owned = self._require_prepared_owned()
        membership_generation = topology.membership_generation
        source_incarnation = topology.incarnation(self._require_rank())
        generation_arg = self._deps.cutlass.Uint32(membership_generation)
        source_incarnation_arg = self._deps.cutlass.Uint64(source_incarnation)
        self._prepared_generation = _PreparedGeneration(
            membership_generation=membership_generation,
            banks=tuple(
                _PreparedBankGeneration(
                    membership_generation=membership_generation,
                    source_incarnation=source_incarnation,
                    generation_arg=generation_arg,
                    source_incarnation_arg=source_incarnation_arg,
                    owned=owned,
                    storage=storage,
                )
                for storage in owned.banks
            ),
        )

    def _initialize_agent_and_metadata(self) -> None:
        deps, layout, _, _ = self._required_setup()
        rank = self._require_rank()
        name = self._agent_name(rank)
        config = deps.agent_config_factory(
            enable_prog_thread=False,
            num_threads=_UCX_POST_THREADS,
            backends=[],
            sync_mode=deps.sync_mode_default,
        )
        agent = deps.agent_factory(name, config)
        agent.create_backend(
            "UCX",
            {
                "num_threads": str(_UCX_POST_THREADS),
                "num_workers": str(_UCX_NUM_WORKERS),
                "ucx_num_device_channels": str(max(1, layout.experts_per_rank)),
            },
        )
        self._agent = agent
        self._registration = agent.register_memory([self._arena], backends=["UCX"])

        prefix = self._generation_tag("setup", self._topology)
        metadata = self.control.exchange(
            f"{prefix}.metadata", agent.get_agent_metadata()
        )
        local_coordinates = PeerCoordinates(name, (_device_region(self._arena),))
        coordinates = _decode_coordinates(
            self.control.exchange(
                f"{prefix}.coordinates", local_coordinates.to_bytes()
            ),
            layout.max_ranks,
        )
        self._validate_coordinates(coordinates)
        expected_names = {
            peer_rank: self._agent_name(peer_rank)
            for peer_rank in range(layout.max_ranks)
        }
        peer_names: list[str] = []
        for peer_rank, peer_coordinates in enumerate(coordinates):
            expected_name = expected_names[peer_rank]
            if peer_coordinates.agent_name != expected_name:
                raise RuntimeError(
                    f"stable rank {peer_rank} published unexpected agent "
                    f"{peer_coordinates.agent_name!r}"
                )
            if peer_rank == rank:
                continue
            loaded_name = normalize_agent_name(
                agent.add_remote_agent(metadata[peer_rank])
            )
            if loaded_name != expected_name:
                raise RuntimeError(
                    f"loaded agent {loaded_name!r}, expected {expected_name!r}"
                )
            peer_names.append(expected_name)
        self._coordinates = coordinates
        self._peer_names = tuple(peer_names)
        self.control.barrier(f"{prefix}.metadata-loaded")
        for peer_name in self._peer_names:
            agent.make_connection(peer_name, backends=["UCX"])
        complete_ucx_setup_handshake(
            agent,
            self.control,
            expected_names,
            generation=self._topology.membership_generation,
            nonce=(f"{self.run_id}-" f"{self._configuration_digest(self._topology)}"),
            timeout_s=self.timeout_s,
        )

    def _compile_specializations(self) -> None:
        """Compile and occupancy-admit every configured logical shape."""

        deps, layout, contract, device = self._required_setup()
        torch = deps.torch
        owned = self._require_prepared_owned()
        bank_zero = owned.banks[0]
        stream_arg = self._stream_arg(self._control_stream)
        tensor = self._tensor_arg
        fake_remote = deps.nixl_cute.make_fake_memory_view(
            "remote", (layout.arena_nbytes,) * layout.max_ranks
        )
        self._compiled_resolve = deps.nixl_cute.compile(
            deps.kernels.launch_resolve_mapped_peers,
            fake_remote,
            owned.arena.argument,
            owned.rank_mask.argument,
            owned.peer_bases.argument,
            owned.statuses.argument,
            stream_arg,
            self._require_rank(),
            contract.max_ranks,
        )

        self._compiled_expert = deps.nixl_cute.compile(
            deps.kernels.launch_standin_expert,
            owned.arena.argument,
            bank_zero.combine_stage.argument,
            stream_arg,
            deps.cutlass.Uint32(0),
            self._require_rank(),
            contract.max_ranks,
            contract.experts_per_rank,
            contract.token_capacity,
            contract.hidden_size,
            layout.region("expert_input").offset,
            layout.region("dispatch_layout_range").offset,
            layout.payload_stride,
        )

        for live_tokens in self.live_token_specializations:
            physical_tokens = max(1, live_tokens)
            with torch.cuda.stream(self._control_stream):
                activation_template = torch.empty(
                    (physical_tokens, contract.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                index_template = torch.empty(
                    (physical_tokens, contract.top_k),
                    dtype=torch.int32,
                    device=device,
                )
                weight_template = torch.empty(
                    (physical_tokens, contract.top_k),
                    dtype=torch.float32,
                    device=device,
                )
                output_template = torch.empty(
                    (physical_tokens, contract.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
            dispatch = deps.nixl_cute.compile(
                deps.kernels.launch_mapped_dispatch,
                owned.arena.argument,
                tensor(activation_template),
                tensor(index_template),
                bank_zero.expert_counts.argument,
                bank_zero.route_counts.argument,
                owned.rank_mask.argument,
                owned.rank_incarnations.argument,
                owned.peer_bases.argument,
                owned.statuses.argument,
                stream_arg,
                deps.cutlass.Uint32(0),
                deps.cutlass.Uint32(0),
                deps.cutlass.Uint64(1),
                self._require_rank(),
                contract.max_ranks,
                contract.experts_per_rank,
                contract.token_capacity,
                live_tokens,
                self.worker_ctas,
                contract.top_k,
                contract.hidden_size,
                layout.region("dispatch_recv").offset,
                layout.region("dispatch_recv_src_info").offset,
                layout.region("expert_input").offset,
                layout.region("dispatch_src_info").offset,
                layout.region("dispatch_stamp").offset,
                layout.region("dispatch_ready_seq").offset,
                layout.region("dispatch_credit").offset,
                layout.region("dispatch_layout_range").offset,
                layout.payload_stride,
            )
            combine = deps.nixl_cute.compile(
                deps.kernels.launch_mapped_combine,
                owned.arena.argument,
                bank_zero.combine_stage.argument,
                tensor(index_template),
                tensor(weight_template),
                tensor(output_template),
                bank_zero.route_counts.argument,
                owned.rank_mask.argument,
                owned.rank_incarnations.argument,
                owned.peer_bases.argument,
                owned.statuses.argument,
                stream_arg,
                deps.cutlass.Uint32(0),
                deps.cutlass.Uint32(0),
                deps.cutlass.Uint64(1),
                self._require_rank(),
                contract.max_ranks,
                contract.experts_per_rank,
                contract.token_capacity,
                live_tokens,
                self.worker_ctas,
                contract.top_k,
                contract.hidden_size,
                layout.region("dispatch_src_info").offset,
                layout.region("dispatch_layout_range").offset,
                layout.region("dispatch_credit").offset,
                layout.region("combine_recv").offset,
                layout.region("combine_stamp").offset,
                layout.region("combine_ready_seq").offset,
                layout.region("combine_credit").offset,
                layout.payload_stride,
            )
            bound_dispatch, dispatch_occupancy = deps.bind_cooperative(
                dispatch,
                device_ordinal=device,
                block_threads=WARP_SIZE,
                planned_ctas=self.worker_ctas,
            )
            bound_combine, combine_occupancy = deps.bind_cooperative(
                combine,
                device_ordinal=device,
                block_threads=WARP_SIZE,
                planned_ctas=self.worker_ctas,
            )
            contract.validate_cooperative_grid(
                dispatch_resident_limit=dispatch_occupancy.cooperative_cta_capacity,
                worker_ctas=self.worker_ctas,
                combine_resident_limit=combine_occupancy.cooperative_cta_capacity,
            )
            self._compiled_dispatch[live_tokens] = bound_dispatch
            self._compiled_combine[live_tokens] = bound_combine
            self._occupancy[live_tokens] = {
                "dispatch": dispatch_occupancy.as_dict(),
                "combine": combine_occupancy.as_dict(),
            }
        # Occupancy is read from each exact CUDA library already loaded by its
        # bound specialization. Normal execution therefore needs no retained
        # files or path-bearing compile option; explicit runner evidence mode
        # can still retain rank-private PTX/CUBIN without changing admission.
        self.control.barrier(
            f"{self._generation_tag('setup', self._topology)}.compiled"
        )

    def _make_candidate(self, topology: StableSparseTopology) -> _CandidateGeneration:
        deps, _, contract, device = self._required_setup()
        torch = deps.torch
        coordinates = self._composite_coordinates(topology)
        view = self._agent.prepare_device_view(
            coordinates,
            mem_type="VRAM",
            backend="UCX",
            connection_timeout_ms=max(1, int(self.timeout_s * 1000)),
        )
        try:
            with torch.cuda.stream(self._control_stream):
                rank_mask = torch.tensor(
                    topology.nixl_mask, dtype=torch.int64, device=device
                )
                rank_incarnations = torch.tensor(
                    topology.rank_incarnations, dtype=torch.uint64, device=device
                )
                peer_bases = torch.zeros(
                    contract.max_ranks, dtype=torch.uint64, device=device
                )
                statuses = torch.zeros(
                    contract.max_ranks, dtype=torch.int32, device=device
                )
            candidate = _CandidateGeneration(
                topology=topology,
                view=view,
                rank_mask=rank_mask,
                rank_incarnations=rank_incarnations,
                peer_bases=peer_bases,
                statuses=statuses,
            )
            candidate.prepared = self._prepare_candidate_descriptors(candidate)
            return candidate
        except BaseException:
            view.release()
            raise

    def _prepare_candidate_descriptors(
        self, candidate: _CandidateGeneration
    ) -> _PreparedCandidateDescriptors:
        """Prepare staged controls once; candidate resolution is a cold path."""

        contract = self._contract
        torch = self._deps.torch
        tensors = (
            ("candidate rank_mask", candidate.rank_mask, torch.int64),
            (
                "candidate rank_incarnations",
                candidate.rank_incarnations,
                torch.uint64,
            ),
            ("candidate peer_bases", candidate.peer_bases, torch.uint64),
            ("candidate statuses", candidate.statuses, torch.int32),
        )
        for name, tensor, dtype in tensors:
            self._validate_tensor(tensor, name, dtype)
            self._validate_shape(tensor, name, (contract.max_ranks,))
            self._require_outside_arena(name, tensor)
        self._require_disjoint(
            tuple((name, tensor) for name, tensor, _ in tensors),
            permit_empty=False,
        )
        return _PreparedCandidateDescriptors(
            rank_mask=self._tensor_arg(candidate.rank_mask),
            rank_incarnations=self._tensor_arg(candidate.rank_incarnations),
            peer_bases=self._tensor_arg(candidate.peer_bases),
            statuses=self._tensor_arg(candidate.statuses),
        )

    def _resolve_candidate(self, candidate: _CandidateGeneration) -> None:
        deps, _, contract, _ = self._required_setup()
        if candidate.released:
            raise PipelineError("cannot resolve a released generation candidate")
        if candidate.resolved:
            raise PipelineError("generation candidate was already resolved")
        if candidate.prepared is None:
            candidate.prepared = self._prepare_candidate_descriptors(candidate)
        prepared = candidate.prepared
        owned = self._require_prepared_owned()
        self._compiled_resolve(
            candidate.view,
            owned.arena.argument,
            prepared.rank_mask,
            prepared.peer_bases,
            prepared.statuses,
            self._stream_arg(self._control_stream),
        )
        # Candidate resolution is a control-plane operation.  This is the only
        # place where peer pointer status reaches the CPU.
        self._control_stream.synchronize()
        statuses = [int(value) for value in candidate.statuses.cpu().tolist()]
        bases = [int(value) for value in candidate.peer_bases.cpu().tolist()]
        if len(statuses) != contract.max_ranks or len(bases) != contract.max_ranks:
            raise AssertionError(
                "candidate pointer result shape changed during resolution"
            )
        for peer in candidate.topology.active_ranks:
            if statuses[peer] != 0:
                raise RuntimeError(
                    f"mapped pointer resolution failed for active peer {peer}: "
                    f"status={statuses[peer]}"
                )
            if bases[peer] == 0:
                raise RuntimeError(
                    f"active peer {peer} has no process-local mapped address"
                )
        topology = candidate.topology
        active = set(topology.active_ranks)
        rank = self._require_rank()
        candidate.mapped_pointer_evidence = {
            "schema_version": 1,
            "backend": "UCX",
            "memory_type": "VRAM",
            "rank": rank,
            "membership_generation": topology.membership_generation,
            "configuration_digest": self._configuration_digest(topology),
            "arena_nbytes": self._layout.arena_nbytes,
            "composite_view_descriptors": topology.max_ranks,
            "stable_slot_order": list(range(topology.max_ranks)),
            "active_ranks": list(topology.active_ranks),
            "local_loopback_slots": [
                peer
                for peer in range(topology.max_ranks)
                if peer == rank or peer not in active
            ],
            "peers": [
                {
                    "rank": peer,
                    "owner_device": self.devices[peer],
                    "active": peer in active,
                    "resolve_status": statuses[peer],
                    "nonzero_process_local_base": bases[peer] != 0,
                    "classification": (
                        "local_registered_arena"
                        if peer == rank and peer in active
                        else (
                            "mapped_process_local_peer"
                            if peer in active
                            else "masked_loopback_not_resolved"
                        )
                    ),
                }
                for peer in range(topology.max_ranks)
            ],
        }
        candidate.resolved = True

    def _install_initial_candidate(self, candidate: _CandidateGeneration) -> None:
        if not candidate.resolved:
            raise PipelineError("initial generation candidate was not resolved")
        topology = candidate.topology
        with self._deps.torch.cuda.stream(self._control_stream):
            self._rebase_arena(candidate)
            self._peer_bases.copy_(candidate.peer_bases, non_blocking=True)
            self._statuses.zero_()
        self._control_stream.synchronize()
        self._activate_prepared_generation(topology)
        self.control.barrier(f"{self._generation_tag('generation', topology)}.active")
        self._active_candidate = candidate
        self._mapped_pointer_evidence = candidate.mapped_pointer_evidence

    def enqueue_prepared_dispatch(
        self,
        prepared_launch: Any,
        *,
        input_ready_event: Any | None,
        communication_predecessor_event: Any | None,
    ) -> Any:
        """Post a cold-bound rolling dispatch without host scalar derivation."""

        launch = self._require_prepared_mapped_launch(prepared_launch)
        stream = launch.communication_stream
        prepared = launch.bank_generation
        owned = prepared.owned
        storage = prepared.storage
        if communication_predecessor_event is not None:
            stream.wait_event(communication_predecessor_event)
        if input_ready_event is not None:
            stream.wait_event(input_ready_event)
        launch.dispatch_launcher(
            owned.arena.argument,
            launch.activation_argument,
            launch.topk_indices_argument,
            storage.expert_counts.argument,
            storage.route_counts.argument,
            owned.rank_mask.argument,
            owned.rank_incarnations.argument,
            owned.peer_bases.argument,
            owned.statuses.argument,
            launch.communication_stream.argument,
            prepared.generation_arg,
            launch.step_argument,
            prepared.source_incarnation_arg,
        )
        launch.events.dispatch_ready.record(stream.owner)
        return launch.events.dispatch_ready

    def enqueue_prepared_combine(
        self,
        prepared_launch: Any,
        *,
        expert_ready_event: Any | None,
        communication_predecessor_event: Any | None,
    ) -> CombineEvents:
        """Post a cold-bound rolling combine without bank/event/step arithmetic."""

        launch = self._require_prepared_mapped_launch(prepared_launch)
        stream = launch.communication_stream
        prepared = launch.bank_generation
        owned = prepared.owned
        storage = prepared.storage
        if communication_predecessor_event is not None:
            stream.wait_event(communication_predecessor_event)
        if expert_ready_event is not None:
            stream.wait_event(expert_ready_event)
        launch.combine_launcher(
            owned.arena.argument,
            storage.combine_stage.argument,
            launch.topk_indices_argument,
            launch.topk_weights_argument,
            launch.combine_output_argument,
            storage.route_counts.argument,
            owned.rank_mask.argument,
            owned.rank_incarnations.argument,
            owned.peer_bases.argument,
            owned.statuses.argument,
            launch.communication_stream.argument,
            prepared.generation_arg,
            launch.step_argument,
            prepared.source_incarnation_arg,
        )
        launch.events.output_ready.record(stream.owner)
        launch.events.bank_reusable.record(stream.owner)
        self._latest_bank_events[launch.bank] = launch.events.bank_reusable
        return launch.events.combine_events

    def _enqueue_prepared_standin_expert(
        self,
        prepared_launch: Any,
        *,
        handle: Any,
        output: Any,
        input_ready_event: Any | None,
    ) -> Any:
        """Post the correctness-only expert using its cold launch record."""

        launch = self._require_prepared_mapped_launch(prepared_launch)
        stream = launch.expert_stream
        if handle._backend_launch is not launch:
            raise PipelineError("stand-in expert handle/launch binding does not match")
        if output is not launch.bank_generation.storage.combine_stage.owner:
            raise ValueError("stand-in expert must target its cold-bound bank output")
        if input_ready_event is not None:
            stream.wait_event(input_ready_event)
        launch.expert_launcher(
            launch.bank_generation.owned.arena.argument,
            launch.bank_generation.storage.combine_stage.argument,
            launch.expert_stream.argument,
            launch.step_argument,
        )
        launch.events.expert_ready.record(stream.owner)
        return launch.events.expert_ready

    def enqueue_dispatch(
        self,
        *,
        operation: OperationEpoch,
        source_incarnation: int,
        topology: StableSparseTopology,
        activations: Any,
        topk_indices: Any,
        input_ready_event: Any | None,
        bank_reuse_event: Any | None,
        communication_predecessor_event: Any | None,
        buffers: PipelineBankBuffers,
        stream: Any,
        prepared_operation: PreparedMappedOperation | None = None,
    ) -> Any:
        """Enqueue the mapped dispatch specialization without a CPU wait."""

        prepared = self._require_hot(operation, source_incarnation, topology, buffers)
        operation_binding = (
            None
            if prepared_operation is None
            else self._require_dispatch_operation(
                prepared_operation, activations, topk_indices
            )
        )
        if operation_binding is None:
            sealed_activations = self._find_prepared_external(
                "activations", activations
            )
            sealed_indices = self._find_prepared_external("topk_indices", topk_indices)
            live_tokens = (
                sealed_activations.shape[0]
                if sealed_activations is not None
                else int(activations.shape[0])
            )
            contract = self._contract
            activation_shape = (live_tokens, contract.hidden_size)
            index_shape = (live_tokens, contract.top_k)
            if sealed_activations is None:
                self._validate_tensor(
                    activations, "activations", self._deps.torch.bfloat16
                )
                self._validate_shape(activations, "activations", activation_shape)
                self._require_aligned("activations", activations)
                activation_span = self._memory_span(activations)
            else:
                activation_span = sealed_activations.span
            if sealed_indices is None:
                self._validate_tensor(
                    topk_indices, "topk_indices", self._deps.torch.int32
                )
                self._validate_shape(topk_indices, "topk_indices", index_shape)
                index_span = self._memory_span(topk_indices)
            else:
                if sealed_indices.shape != index_shape:
                    raise ValueError(
                        "sealed topk_indices shape does not match activations"
                    )
                index_span = sealed_indices.span
            self._require_outside_owned_storage("activations", activation_span)
            self._require_outside_owned_storage("topk_indices", index_span)
            self._require_disjoint_spans(
                (
                    ("activations", activation_span),
                    ("topk_indices", index_span),
                ),
                permit_empty=True,
            )
            activation_arg = (
                prepared.owned.dummy_activations.argument
                if live_tokens == 0
                else (
                    sealed_activations.argument
                    if sealed_activations is not None
                    else self._tensor_arg(activations)
                )
            )
            index_arg = (
                prepared.owned.dummy_topk_indices.argument
                if live_tokens == 0
                else (
                    sealed_indices.argument
                    if sealed_indices is not None
                    else self._tensor_arg(topk_indices)
                )
            )
        else:
            live_tokens = operation_binding.live_tokens
            sealed_activations = operation_binding._activations
            sealed_indices = operation_binding._topk_indices
            activation_arg = (
                prepared.owned.dummy_activations.argument
                if live_tokens == 0
                else sealed_activations.argument
            )
            index_arg = (
                prepared.owned.dummy_topk_indices.argument
                if live_tokens == 0
                else sealed_indices.argument
            )
        launcher = self._compiled_dispatch.get(live_tokens)
        if launcher is None:
            raise PipelineError(
                f"live token count {live_tokens} was not compiled before activation"
            )
        # The direct/raw API is deliberately independent of any finite rolling
        # context left installed by an earlier preflight. Prepared rolling
        # calls use ``enqueue_prepared_dispatch`` and their operation-specific
        # launch record; direct calls retain their original wrapper fallback.
        stream_arg = self._stream_arg(stream)
        step_arg = self._deps.cutlass.Uint32(operation.step)
        events = self._events_for(operation)
        if communication_predecessor_event is not None:
            stream.wait_event(communication_predecessor_event)
        if input_ready_event is not None:
            stream.wait_event(input_ready_event)
        if bank_reuse_event is not None:
            stream.wait_event(bank_reuse_event)
        launcher(
            prepared.owned.arena.argument,
            activation_arg,
            index_arg,
            prepared.storage.expert_counts.argument,
            prepared.storage.route_counts.argument,
            prepared.owned.rank_mask.argument,
            prepared.owned.rank_incarnations.argument,
            prepared.owned.peer_bases.argument,
            prepared.owned.statuses.argument,
            stream_arg,
            prepared.generation_arg,
            step_arg,
            prepared.source_incarnation_arg,
        )
        events.dispatch_ready.record(stream)
        if sealed_activations is None:
            activations.record_stream(stream)
        if sealed_indices is None:
            topk_indices.record_stream(stream)
        return events.dispatch_ready

    def enqueue_combine(
        self,
        *,
        operation: OperationEpoch,
        source_incarnation: int,
        topology: StableSparseTopology,
        expert_output: Any,
        zero_copy: bool,
        topk_indices: Any,
        topk_weights: Any,
        combine_output: Any,
        expert_ready_event: Any | None,
        communication_predecessor_event: Any | None,
        buffers: PipelineBankBuffers,
        stream: Any,
        prepared_operation: PreparedMappedOperation | None = None,
    ) -> CombineEvents:
        """Enqueue reverse scatter, reduction, and peer credits."""

        prepared = self._require_hot(operation, source_incarnation, topology, buffers)
        operation_binding = (
            None
            if prepared_operation is None
            else self._require_combine_operation(
                prepared_operation,
                topk_indices,
                topk_weights,
                combine_output,
            )
        )
        if operation_binding is None:
            sealed_indices = self._find_prepared_external("topk_indices", topk_indices)
            sealed_weights = self._find_prepared_external("topk_weights", topk_weights)
            sealed_output = self._find_prepared_external(
                "combine_output", combine_output
            )
            live_tokens = (
                sealed_indices.shape[0]
                if sealed_indices is not None
                else int(topk_indices.shape[0])
            )
            torch = self._deps.torch
            backend_expert_output = expert_output is buffers.combine_stage
            sealed_expert = (
                None
                if backend_expert_output
                else self._find_prepared_external("expert_output", expert_output)
            )
            if not backend_expert_output and sealed_expert is None:
                self._validate_tensor(expert_output, "expert_output", torch.bfloat16)
            contract = self._contract
            dense_rows = contract.max_ranks * contract.token_capacity
            expert_shape = (
                contract.experts_per_rank,
                dense_rows,
                contract.hidden_size,
            )
            index_shape = (live_tokens, contract.top_k)
            output_shape = (live_tokens, contract.hidden_size)
            if not backend_expert_output and sealed_expert is None:
                self._validate_shape(
                    expert_output,
                    "expert_output",
                    expert_shape,
                )
            if sealed_indices is None:
                self._validate_tensor(topk_indices, "topk_indices", torch.int32)
                self._validate_shape(topk_indices, "topk_indices", index_shape)
                index_span = self._memory_span(topk_indices)
            else:
                index_span = sealed_indices.span
            if sealed_weights is None:
                self._validate_tensor(topk_weights, "topk_weights", torch.float32)
                self._validate_shape(topk_weights, "topk_weights", index_shape)
                weight_span = self._memory_span(topk_weights)
            else:
                if sealed_weights.shape != index_shape:
                    raise ValueError("sealed topk_weights shape does not match indices")
                weight_span = sealed_weights.span
            if sealed_output is None:
                self._validate_tensor(combine_output, "combine_output", torch.bfloat16)
                self._validate_shape(combine_output, "combine_output", output_shape)
                self._require_aligned("combine_output", combine_output)
                output_span = self._memory_span(combine_output)
            else:
                if sealed_output.shape != output_shape:
                    raise ValueError(
                        "sealed combine_output shape does not match indices"
                    )
                output_span = sealed_output.span
            if not backend_expert_output and sealed_expert is None:
                self._require_aligned("expert_output", expert_output)
            expert_span = (
                prepared.storage.combine_stage.span
                if backend_expert_output
                else (
                    sealed_expert.span
                    if sealed_expert is not None
                    else self._memory_span(expert_output)
                )
            )
            self._require_disjoint_spans(
                (
                    ("expert_output", expert_span),
                    ("topk_indices", index_span),
                    ("topk_weights", weight_span),
                    ("combine_output", output_span),
                    ("route_counts", prepared.storage.route_counts.span),
                ),
                permit_empty=True,
            )
            self._require_outside_owned_storage("topk_indices", index_span)
            self._require_outside_owned_storage("topk_weights", weight_span)
            self._require_outside_owned_storage("combine_output", output_span)
            combine_stage_span = prepared.storage.combine_stage.span
            if zero_copy and expert_span != combine_stage_span:
                raise PipelineError(
                    "zero_copy requires the exact registered combine_stage byte span"
                )
            expert_overlaps_arena = self._spans_overlap(
                expert_span, prepared.owned.arena.span
            )
            if expert_overlaps_arena and (
                not zero_copy or expert_span != combine_stage_span
            ):
                raise ValueError(
                    "expert_output may overlap the registered arena only as the exact "
                    "bank combine_stage zero-copy view"
                )
            if not expert_overlaps_arena:
                self._require_outside_owned_storage("expert_output", expert_span)
            expert_arg = (
                prepared.storage.combine_stage.argument
                if zero_copy
                else (
                    sealed_expert.argument
                    if sealed_expert is not None
                    else self._tensor_arg(expert_output)
                )
            )
            index_arg = (
                prepared.owned.dummy_topk_indices.argument
                if live_tokens == 0
                else (
                    sealed_indices.argument
                    if sealed_indices is not None
                    else self._tensor_arg(topk_indices)
                )
            )
            weight_arg = (
                prepared.owned.dummy_topk_weights.argument
                if live_tokens == 0
                else (
                    sealed_weights.argument
                    if sealed_weights is not None
                    else self._tensor_arg(topk_weights)
                )
            )
            output_arg = (
                prepared.owned.dummy_combine_output.argument
                if live_tokens == 0
                else (
                    sealed_output.argument
                    if sealed_output is not None
                    else self._tensor_arg(combine_output)
                )
            )
        else:
            if not zero_copy or expert_output is not buffers.combine_stage:
                raise PipelineError(
                    "prepared_operation combine requires the exact backend-owned "
                    "combine_stage with zero_copy=True"
                )
            live_tokens = operation_binding.live_tokens
            sealed_expert = None
            sealed_indices = operation_binding._topk_indices
            sealed_weights = operation_binding._topk_weights
            sealed_output = operation_binding._combine_output
            expert_arg = prepared.storage.combine_stage.argument
            index_arg = (
                prepared.owned.dummy_topk_indices.argument
                if live_tokens == 0
                else sealed_indices.argument
            )
            weight_arg = (
                prepared.owned.dummy_topk_weights.argument
                if live_tokens == 0
                else sealed_weights.argument
            )
            output_arg = (
                prepared.owned.dummy_combine_output.argument
                if live_tokens == 0
                else sealed_output.argument
            )
        launcher = self._compiled_combine.get(live_tokens)
        if launcher is None:
            raise PipelineError(
                f"live token count {live_tokens} was not compiled before activation"
            )
        # Do not let a prior finite rolling context constrain this direct/raw
        # compatibility path. The prepared rolling entry point already owns
        # exact cold stream and step arguments.
        stream_arg = self._stream_arg(stream)
        step_arg = self._deps.cutlass.Uint32(operation.step)
        if communication_predecessor_event is not None:
            stream.wait_event(communication_predecessor_event)
        if expert_ready_event is not None:
            stream.wait_event(expert_ready_event)
        events = self._events_for(operation)
        launcher(
            prepared.owned.arena.argument,
            expert_arg,
            index_arg,
            weight_arg,
            output_arg,
            prepared.storage.route_counts.argument,
            prepared.owned.rank_mask.argument,
            prepared.owned.rank_incarnations.argument,
            prepared.owned.peer_bases.argument,
            prepared.owned.statuses.argument,
            stream_arg,
            prepared.generation_arg,
            step_arg,
            prepared.source_incarnation_arg,
        )
        events.output_ready.record(stream)
        events.bank_reusable.record(stream)
        if not zero_copy and sealed_expert is None:
            # Registered zero-copy storage is strongly owned through the final
            # bank drain; only caller storage needs allocator lifetime metadata.
            expert_output.record_stream(stream)
        if sealed_indices is None:
            topk_indices.record_stream(stream)
        if sealed_weights is None:
            topk_weights.record_stream(stream)
        if sealed_output is None:
            combine_output.record_stream(stream)
        self._latest_bank_events[operation.bank] = events.bank_reusable
        return events.combine_events

    def enqueue_standin_expert(
        self,
        *,
        handle: Any,
        output: Any,
        stream: Any,
        input_ready_event: Any | None,
    ) -> Any:
        """Example-only expert boundary; production replaces this with GEMM."""

        if handle._backend_launch is not None:
            return self._enqueue_prepared_standin_expert(
                handle._backend_launch,
                handle=handle,
                output=output,
                input_ready_event=input_ready_event,
            )
        self._require_active_backend()
        prepared = self._require_prepared_bank(handle.operation)
        if output is not handle.combine_buffer:
            raise ValueError("stand-in expert must write the handle's combine buffer")
        if output is not prepared.storage.combine_stage.owner:
            raise ValueError(
                "stand-in expert must target this backend generation's exact bank"
            )
        stream_arg = self._stream_arg(stream)
        step_arg = self._deps.cutlass.Uint32(handle.operation.step)
        if input_ready_event is not None:
            stream.wait_event(input_ready_event)
        self._compiled_expert(
            prepared.owned.arena.argument,
            prepared.storage.combine_stage.argument,
            stream_arg,
            step_arg,
        )
        event = self._events_for(handle.operation).expert_ready
        event.record(stream)
        return event

    def stage_generation(
        self,
        *,
        current: StableSparseTopology,
        next_topology: StableSparseTopology,
    ) -> Any:
        """Prepare and pointer-resolve a hidden same-incarnation view."""

        self._require_active_backend()
        self._require_current_topology(current)
        self.validate_prepared_bindings()
        if self._staged_candidate is not None:
            raise PipelineError("a generation candidate is already staged")
        if next_topology.rank_incarnations != current.rank_incarnations:
            raise PipelineError(
                "all-mapped elastic mode supports live standby membership changes, "
                "not process incarnation replacement"
            )
        self._validate_live_standby_topology(next_topology, initial=False)
        self._validate_configuration_quorum(next_topology)
        candidate = self._make_candidate(next_topology)
        try:
            self._resolve_candidate(candidate)
            self.control.barrier(
                f"{self._generation_tag('generation', next_topology)}.mapped"
            )
        except BaseException as error:
            try:
                self._release_candidate(candidate)
            except BaseException as release_error:
                self.fail_stop(release_error)
                raise release_error from error
            raise
        self._staged_candidate = candidate
        self._state = MappedBackendState.STAGED
        return candidate

    def enqueue_generation_drain(
        self,
        *,
        current: StableSparseTopology,
        staged_token: Any,
        completion_events: Sequence[Any],
        stream: Any,
    ) -> GenerationDrain:
        """Put a CUDA drain behind every completed bank on ``stream``."""

        if self._state != MappedBackendState.STAGED:
            raise PipelineError("no staged generation is available to drain")
        self._require_current_topology(current)
        candidate = self._require_candidate(staged_token)
        for event in completion_events:
            stream.wait_event(event)
        # The mapped specialization has no host-visible request handles: the
        # bank-reusable event follows all direct stores, publications, and
        # returned credits.  Reaching this event is its cumulative drain.
        event = self._drain_events[current.membership_generation % NUM_BANKS]
        event.record(stream)
        self._state = MappedBackendState.DRAINING
        return GenerationDrain(_DrainToken(candidate=candidate, event=event))

    def commit_generation(
        self,
        *,
        current: StableSparseTopology,
        next_topology: StableSparseTopology,
        staged_token: Any,
        drain: GenerationDrain,
    ) -> GenerationCommit:
        """Synchronously drain, rebase, and activate one resolved change."""

        if self._state != MappedBackendState.DRAINING:
            raise PipelineError("backend has no staged generation to commit")
        self._require_current_topology(current)
        candidate = self._require_candidate(staged_token)
        if candidate.topology != next_topology:
            raise PipelineError("staged candidate does not match next_topology")
        if not isinstance(drain, GenerationDrain) or not isinstance(
            drain.token, _DrainToken
        ):
            raise TypeError("drain was not produced by this mapped backend")
        if drain.token.candidate is not candidate:
            raise PipelineError("drain belongs to another generation candidate")

        self._state = MappedBackendState.COMMITTING
        # No hot launch may retain the old generation binding once rebasing
        # begins. A failed commit therefore cannot accidentally reactivate it.
        self._prepared_generation = None
        try:
            drain.token.event.synchronize()
            generation = next_topology.membership_generation
            prefix = self._generation_tag("generation", next_topology)
            self.control.barrier(f"{prefix}.drained")
            if not candidate.resolved:
                raise PipelineError("staged generation was not pointer-resolved")

            with self._deps.torch.cuda.stream(self._control_stream):
                self._rebase_arena(candidate)
                self._peer_bases.copy_(candidate.peer_bases, non_blocking=True)
                self._statuses.zero_()
            self._control_stream.synchronize()
            self.control.barrier(f"{prefix}.rebased")

            old = self._active_candidate
            self._active_candidate = candidate
            self._staged_candidate = None
            self._topology = next_topology
            self._latest_bank_events = [None] * NUM_BANKS
            self._activate_prepared_generation(next_topology)
            self.control.barrier(f"{prefix}.active")
            self._mapped_pointer_evidence = candidate.mapped_pointer_evidence
            if old is not None:
                self._release_candidate(old)
            self.control.barrier(f"{prefix}.old-view-released")
        except BaseException as error:
            self.fail_stop(error)
            raise
        self._state = MappedBackendState.ACTIVE
        return GenerationCommit(
            membership_generation=generation,
            rebased_planes=GENERATION_REBASE_PLANES,
        )

    def fail_stop(self, error: BaseException) -> None:
        """Quarantine mappings and terminate the process/CUDA context."""

        if self._state == MappedBackendState.FAILED:
            return
        self._failed_error = error
        # Publish the terminal state and invalidate every operation binding
        # before any cleanup-like action.  This path must still work when the
        # triggering exception is MemoryError after a device enqueue, so it
        # creates no replacement cookie or candidate tuple.
        self._state = MappedBackendState.FAILED
        self._prepared_generation = None
        self._prepared_owned = None
        self._operation_cookie = None
        active_candidate = self._active_candidate
        if isinstance(active_candidate, _CandidateGeneration):
            active_candidate.prepared = None
        staged_candidate = self._staged_candidate
        if isinstance(staged_candidate, _CandidateGeneration):
            staged_candidate.prepared = None
        # ``os._exit`` deliberately bypasses Python traceback flushing.  Emit
        # one bounded failure-only record after quarantine is installed and
        # before termination.  This performs no synchronization or cleanup and
        # is absent from every successful setup and enqueue path.
        try:
            _emit_fail_stop_diagnostic(
                rank=self._rank,
                state=self._state.value,
                error=error,
            )
        finally:
            # Deliberately do not synchronize, release a view, remove metadata,
            # or deregister the arena. Sealed external tensor and prepared
            # stream owners are retained as part of the same async-lifetime
            # quarantine. ``os._exit`` is the production default; a test
            # terminator may return so quarantine can be asserted. The finally
            # is essential under allocation failure: diagnostics can never
            # turn fail-stop into Python unwinding through live GPU owners.
            self._terminator(_FAIL_STOP_EXIT_CODE)

    def close(self) -> None:
        """Collectively drain and release view -> metadata -> registration."""

        if self._state == MappedBackendState.CLOSED:
            return
        if self._state == MappedBackendState.FAILED:
            raise PipelineFailed(
                "cannot close a fail-stopped mapped backend; its registered "
                "arena is quarantined until process teardown"
            ) from self._failed_error
        if self._state not in (MappedBackendState.ACTIVE, MappedBackendState.STAGED):
            raise PipelineError(f"cannot close backend in state {self._state.value}")
        self._state = MappedBackendState.CLOSING
        self._prepared_generation = None
        try:
            for event in self._latest_bank_events:
                if event is not None:
                    self._control_stream.wait_event(event)
            final_drain = self._drain_events[0]
            final_drain.record(self._control_stream)
            final_drain.synchronize()
            prefix = self._generation_tag("close", self._topology)
            self.control.barrier(f"{prefix}.drained")

            if self._staged_candidate is not None:
                self._release_candidate(self._staged_candidate)
                self._staged_candidate = None
            if self._active_candidate is not None:
                self._release_candidate(self._active_candidate)
                self._active_candidate = None
            self.control.barrier(f"{prefix}.views-released")
            for peer_name in self._peer_names:
                self._agent.remove_remote_agent(peer_name)
            self.control.barrier(f"{prefix}.metadata-removed")
            self._agent.deregister_memory(self._registration, backends=["UCX"])
            self._registration = None
            self.control.barrier(f"{prefix}.deregistered")
        except BaseException as error:
            self.fail_stop(error)
            raise
        self._drop_healthy_resources()
        self._state = MappedBackendState.CLOSED

    def poison_data_planes(self, value: int = 0xA5) -> None:
        """Cold-path test helper that poisons payload regions after a drain."""

        self._require_active_backend()
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 256
        ):
            raise ValueError("poison value must be one byte")
        with self._deps.torch.cuda.stream(self._control_stream):
            for name in (
                "dispatch_recv",
                "dispatch_recv_src_info",
                "expert_input",
                "dispatch_src_info",
                "combine_stage",
                "combine_recv",
            ):
                span = self._layout.region(name)
                self._arena.narrow(0, span.offset, span.nbytes).fill_(value)
        self._control_stream.synchronize()

    def _drop_healthy_resources(self) -> None:
        """Drop backend owners only after the final deregistration quorum."""

        if self._registration is not None:
            raise AssertionError("healthy resource drop preceded deregistration")
        self._compiled_resolve = None
        self._compiled_expert = None
        self._prepared_generation = None
        self._prepared_owned = None
        for by_identity in self._prepared_external.values():
            by_identity.clear()
        self._prepared_submission = None
        self._operation_cookie = object()
        self._compiled_dispatch.clear()
        self._compiled_combine.clear()
        self._event_pool = ()
        self._drain_events = ()
        self._latest_bank_events = [None] * NUM_BANKS
        self._buffers = ()
        self._dummy.clear()
        self._rank_mask = None
        self._rank_incarnations = None
        self._peer_bases = None
        self._statuses = None
        self._route_counts = None
        self._rebase_planes = ()
        self._control_stream = None
        self._arena = None
        self._coordinates = ()
        self._peer_names = ()
        self._agent = None
        self._deps = None

    def _rebase_arena(self, candidate: _CandidateGeneration) -> None:
        if not candidate.resolved:
            raise PipelineError("cannot rebase from an unresolved generation")
        for plane in self._rebase_planes:
            plane.zero_()
        self._rank_mask.copy_(candidate.rank_mask, non_blocking=True)
        self._rank_incarnations.copy_(candidate.rank_incarnations, non_blocking=True)

    @staticmethod
    def _validate_live_standby_topology(
        topology: StableSparseTopology, *, initial: bool
    ) -> None:
        """Require stable, already-live identities for every capacity slot."""

        missing = tuple(
            rank
            for rank, incarnation in enumerate(topology.rank_incarnations)
            if incarnation == 0
        )
        if missing:
            phase = "initial admission" if initial else "generation staging"
            raise PipelineError(
                f"{phase} requires a live standby process incarnation for every "
                f"fixed slot; zero incarnation slots={missing}. This backend can "
                "mask/unmask an existing process but cannot replace one."
            )

    def _configuration_document(self, topology: StableSparseTopology) -> dict[str, Any]:
        """Canonical full admission contract shared by every stable rank."""

        layout = self._require_layout()
        contract = self._contract
        return {
            "schema_version": 1,
            "devices": list(self.devices),
            "layout": {
                "max_ranks": layout.max_ranks,
                "experts_per_rank": layout.experts_per_rank,
                "num_tokens": layout.num_tokens,
                "top_k": layout.top_k,
                "hidden_size": layout.hidden_size,
                "element_size": layout.element_size,
                "record_alignment": layout.record_alignment,
                "region_alignment": layout.region_alignment,
                "arena_alignment": layout.arena_alignment,
                "address_limit": layout.address_limit,
                "route_capacity": layout.route_capacity,
                "payload_nbytes": layout.payload_nbytes,
                "payload_stride": layout.payload_stride,
                "arena_nbytes": layout.arena_nbytes,
                "regions": [
                    {
                        "name": region.name,
                        "offset": region.offset,
                        "nbytes": region.nbytes,
                        "alignment": region.alignment,
                    }
                    for region in layout.regions
                ],
            },
            "kernel_contract": {
                "max_ranks": contract.max_ranks,
                "experts_per_rank": contract.experts_per_rank,
                "token_capacity": contract.token_capacity,
                "top_k": contract.top_k,
                "hidden_size": contract.hidden_size,
                "live_token_specializations": list(self.live_token_specializations),
                "worker_ctas": self.worker_ctas,
            },
            "topology": {
                "max_ranks": topology.max_ranks,
                "experts_per_rank": topology.experts_per_rank,
                "active_ranks": list(topology.active_ranks),
                "membership_generation": topology.membership_generation,
                "rank_incarnations": list(topology.rank_incarnations),
            },
        }

    def _configuration_digest(self, topology: StableSparseTopology) -> str:
        encoded = json.dumps(
            self._configuration_document(topology),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_configuration_quorum(self, topology: StableSparseTopology) -> str:
        """Bounded all-slot agreement before a NIXL view can become active."""

        document = self._configuration_document(topology)
        digest = self._configuration_digest(topology)
        payload = json.dumps(
            {"digest": digest, "document": document},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        tag = f"{self._generation_tag('contract', topology)}.unanimous"
        gathered = self.control.exchange(tag, payload)
        if set(gathered) != set(range(topology.max_ranks)):
            raise RuntimeError("configuration quorum omitted a stable rank slot")
        for rank in range(topology.max_ranks):
            try:
                peer = json.loads(gathered[rank].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"rank {rank} published malformed configuration evidence"
                ) from error
            if peer.get("digest") != digest or peer.get("document") != document:
                raise RuntimeError(
                    f"rank {rank} disagrees with configuration digest {digest}"
                )
        return digest

    def _validate_coordinates(self, coordinates: Sequence[PeerCoordinates]) -> None:
        """Reject malformed owner coordinates before preparing any view."""

        layout = self._require_layout()
        if len(coordinates) != layout.max_ranks:
            raise RuntimeError("coordinate vector does not match stable rank capacity")
        for owner_rank, peer in enumerate(coordinates):
            if len(peer.regions) != 1:
                raise RuntimeError(
                    f"rank {owner_rank} must publish exactly one arena region"
                )
            region = peer.regions[0]
            if region.length != layout.arena_nbytes:
                raise RuntimeError(
                    f"rank {owner_rank} arena length {region.length} does not match "
                    f"{layout.arena_nbytes}"
                )
            expected_device = self.devices[owner_rank]
            if region.device_id != expected_device:
                raise RuntimeError(
                    f"rank {owner_rank} published device {region.device_id}, "
                    f"expected {expected_device}"
                )
            if region.address == 0 or region.address % 16:
                raise RuntimeError(
                    f"rank {owner_rank} arena base must be nonzero and 16-byte aligned"
                )

    def _composite_coordinates(
        self, topology: StableSparseTopology
    ) -> list[tuple[int, int, int, str]]:
        rank = self._require_rank()
        local = self._coordinates[rank]
        active = set(topology.active_ranks)
        result: list[tuple[int, int, int, str]] = []
        for slot in range(topology.max_ranks):
            peer = self._coordinates[slot] if slot in active and slot != rank else local
            region = peer.regions[0]
            result.append(
                (
                    region.address,
                    region.length,
                    region.device_id,
                    peer.agent_name,
                )
            )
        return result

    def _release_candidate(self, candidate: _CandidateGeneration) -> None:
        if candidate.released:
            return
        candidate.view.release()
        if bool(getattr(candidate.view, "valid", False)):
            raise RuntimeError("NIXL device view remained valid after release")
        candidate.prepared = None
        candidate.released = True

    def _events_for(self, operation: OperationEpoch) -> _EventSet:
        ordinal = operation.step // NUM_BANKS
        return self._event_pool[operation.bank][ordinal % self.event_pool_depth]

    def _validate_tensor(self, tensor: Any, name: str, dtype: Any) -> None:
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not bool(tensor.is_cuda):
            raise ValueError(f"{name} must be CUDA storage")
        if int(tensor.get_device()) != self._device:
            raise ValueError(f"{name} resides on the wrong CUDA device")
        if not bool(tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous")

    @staticmethod
    def _validate_shape(tensor: Any, name: str, expected: tuple[int, ...]) -> None:
        actual = tuple(tensor.shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}")

    def _tensor_arg(self, tensor: Any) -> Any:
        return self._deps.from_dlpack(tensor).mark_layout_dynamic()

    def _stream_arg(self, stream: Any) -> Any:
        return self._deps.cuda.CUstream(stream.cuda_stream)

    @staticmethod
    def _normalize_streams(streams: Sequence[Any]) -> tuple[Any, ...]:
        if isinstance(streams, (str, bytes)) or not isinstance(streams, Sequence):
            raise TypeError("streams must be a sequence of CUDA stream objects")
        unique: list[Any] = []
        by_identity: dict[int, Any] = {}
        for stream in streams:
            if stream is None:
                raise TypeError("streams must not contain None")
            key = id(stream)
            prior = by_identity.get(key)
            if prior is None:
                by_identity[key] = stream
                unique.append(stream)
            elif prior is not stream:
                raise AssertionError("live stream objects reused one Python identity")
        if not unique:
            raise ValueError("streams must contain at least one CUDA stream")
        return tuple(unique)

    @staticmethod
    def _stream_handle(stream: Any) -> int:
        try:
            handle = stream.cuda_stream
        except AttributeError as error:
            raise TypeError(
                "stream must expose an integer cuda_stream handle"
            ) from error
        if isinstance(handle, bool) or not isinstance(handle, int):
            raise TypeError("stream.cuda_stream must be an integer")
        if handle < 0:
            raise ValueError("stream.cuda_stream must be non-negative")
        return handle

    @staticmethod
    def _stream_device(stream: Any) -> int:
        try:
            device = stream.device
        except AttributeError as error:
            raise TypeError("stream must expose its CUDA device") from error
        device_type = getattr(device, "type", None)
        if device_type is not None and device_type != "cuda":
            raise ValueError("prepared stream must be a CUDA stream")
        ordinal = getattr(device, "index", None)
        if ordinal is None and isinstance(device, int) and not isinstance(device, bool):
            ordinal = device
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("stream must expose a non-negative CUDA device ordinal")
        return ordinal

    def _prepare_launch_stream(self, stream: Any) -> _PreparedLaunchStream:
        handle = self._stream_handle(stream)
        device = self._stream_device(stream)
        if device != self._device:
            raise ValueError("prepared stream resides on the wrong CUDA device")
        try:
            wait_event = stream.wait_event
        except AttributeError as error:
            raise TypeError("prepared stream must expose wait_event(event)") from error
        if not callable(wait_event):
            raise TypeError("prepared stream wait_event attribute must be callable")
        return _PreparedLaunchStream(
            owner=stream,
            argument=self._deps.cuda.CUstream(handle),
            wait_event=wait_event,
            handle=handle,
            device=device,
        )

    def _prepared_stream_binding_matches(self, binding: _PreparedLaunchStream) -> bool:
        """Cold/debug validation; never called by a kernel enqueue."""

        try:
            return (
                self._stream_handle(binding.owner) == binding.handle
                and self._stream_device(binding.owner) == binding.device
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _fail_changed_prepared_stream(self, binding: _PreparedLaunchStream) -> None:
        error = PipelineError(
            "CUDA stream handle/device changed after submission-context preparation"
        )
        self.fail_stop(error)
        raise error

    def _capture_status_none(self) -> Any:
        try:
            return self._deps.cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE
        except AttributeError as error:
            raise RuntimeError(
                "cuda.bindings.driver has no CU_STREAM_CAPTURE_STATUS_NONE"
            ) from error

    def _checked_stream_capture_status(self, stream_argument: Any) -> Any:
        cuda = self._deps.cuda
        try:
            result = cuda.cuStreamIsCapturing(stream_argument)
        except Exception as error:
            raise RuntimeError("cuStreamIsCapturing raised an exception") from error
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("cuStreamIsCapturing returned a malformed result")
        try:
            success = cuda.CUresult.CUDA_SUCCESS
            statuses = cuda.CUstreamCaptureStatus
            valid = (
                statuses.CU_STREAM_CAPTURE_STATUS_NONE,
                statuses.CU_STREAM_CAPTURE_STATUS_ACTIVE,
                statuses.CU_STREAM_CAPTURE_STATUS_INVALIDATED,
            )
        except AttributeError as error:
            raise RuntimeError("CUDA stream-capture enums are unavailable") from error
        if result[0] != success:
            raise RuntimeError(f"cuStreamIsCapturing failed with {result[0]!r}")
        if result[1] not in valid:
            raise RuntimeError("cuStreamIsCapturing returned an unknown capture status")
        return result[1]

    def _require_prepared_mapped_launch(
        self, prepared_launch: Any
    ) -> _PreparedMappedLaunch:
        """Validate a cold launch record using identity comparisons only."""

        self._require_active_backend()
        if not isinstance(prepared_launch, _PreparedMappedLaunch):
            raise TypeError("prepared launch was not issued by this mapped backend")
        if prepared_launch.backend_cookie is not self._operation_cookie:
            raise PipelineError(
                "prepared launch is stale or belongs to another backend"
            )
        if prepared_launch.submission is not self._prepared_submission:
            raise PipelineError("prepared launch belongs to a replaced submission")
        if prepared_launch.generation is not self._prepared_generation:
            raise PipelineError("prepared launch belongs to a stale generation")
        return prepared_launch

    def _require_hot(
        self,
        operation: OperationEpoch,
        source_incarnation: int,
        topology: StableSparseTopology,
        buffers: PipelineBankBuffers,
    ) -> _PreparedBankGeneration:
        self._require_active_backend()
        self._require_current_topology(topology)
        if operation.membership_generation != topology.membership_generation:
            raise PipelineError("operation belongs to a stale membership generation")
        rank = self._require_rank()
        if not topology.is_active(rank):
            raise PipelineError(f"rank {rank} is masked in the committed generation")
        if source_incarnation != topology.incarnation(rank):
            raise PipelineError(
                "source incarnation does not match the committed stable-rank identity"
            )
        if buffers is not self._buffers[operation.bank]:
            raise PipelineError("buffers do not own the operation's physical bank")
        prepared = self._require_prepared_bank(operation)
        if prepared.source_incarnation != source_incarnation:
            raise PipelineError("prepared descriptors have a stale source incarnation")
        return prepared

    def _require_prepared_owned(self) -> _PreparedOwnedStorage:
        prepared = self._prepared_owned
        if prepared is None:
            raise PipelineError("backend-owned CuTe descriptors are unavailable")
        return prepared

    def _find_prepared_external(self, role: str, tensor: Any) -> _PreparedTensor | None:
        """Lookup a sealed tensor by identity without touching its metadata."""

        prepared = self._prepared_external[role].get(id(tensor))
        if prepared is not None and prepared.owner is not tensor:
            raise AssertionError("sealed tensor identity key was reused")
        return prepared

    def _require_dispatch_operation(
        self,
        token: Any,
        activations: Any,
        topk_indices: Any,
    ) -> PreparedMappedOperation:
        """Validate an opaque dispatch token using identity checks only."""

        if not isinstance(token, PreparedMappedOperation):
            raise TypeError("prepared_operation was not issued by this backend")
        if token._backend_cookie is not self._operation_cookie:
            raise PipelineError(
                "prepared_operation is stale or belongs to another backend"
            )
        if token._activations.owner is not activations:
            raise ValueError("prepared_operation activations object does not match")
        if token._topk_indices.owner is not topk_indices:
            raise ValueError("prepared_operation topk_indices object does not match")
        return token

    def _require_combine_operation(
        self,
        token: Any,
        topk_indices: Any,
        topk_weights: Any,
        combine_output: Any,
    ) -> PreparedMappedOperation:
        """Validate an opaque combine token using identity checks only."""

        if not isinstance(token, PreparedMappedOperation):
            raise TypeError("prepared_operation was not issued by this backend")
        if token._backend_cookie is not self._operation_cookie:
            raise PipelineError(
                "prepared_operation is stale or belongs to another backend"
            )
        if token._topk_indices.owner is not topk_indices:
            raise ValueError("prepared_operation topk_indices object does not match")
        if token._topk_weights.owner is not topk_weights:
            raise ValueError("prepared_operation topk_weights object does not match")
        if token._combine_output.owner is not combine_output:
            raise ValueError("prepared_operation combine_output object does not match")
        return token

    def _require_prepared_bank(
        self, operation: OperationEpoch
    ) -> _PreparedBankGeneration:
        prepared = self._prepared_generation
        if prepared is None:
            raise PipelineError("active generation has no prepared CuTe descriptors")
        if prepared.membership_generation != operation.membership_generation:
            raise PipelineError(
                "prepared CuTe descriptors belong to a stale generation"
            )
        try:
            bank = prepared.banks[operation.bank]
        except IndexError as error:
            raise PipelineError(
                "operation selected an unavailable prepared bank"
            ) from error
        if bank.storage.bank != operation.bank:
            raise AssertionError("prepared CuTe bank descriptor order changed")
        return bank

    def _require_active_backend(self) -> None:
        # Staging prepares a hidden view while the current generation remains
        # hot.  Work stops only when MoEPipeline begins the explicit drain.
        if (
            self._state is not MappedBackendState.ACTIVE
            and self._state is not MappedBackendState.STAGED
        ):
            if self._state == MappedBackendState.FAILED:
                raise PipelineFailed(
                    "mapped backend is fail-stopped"
                ) from self._failed_error
            raise PipelineError(f"mapped backend is {self._state.value}")

    @staticmethod
    def _memory_span(tensor: Any) -> tuple[int, int]:
        begin = int(tensor.data_ptr())
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        return begin, begin + nbytes

    @staticmethod
    def _tensor_storage_identity(tensor: Any) -> tuple[int, ...]:
        """Return a cold-path storage identity stronger than tensor identity."""

        untyped_storage = getattr(tensor, "untyped_storage", None)
        if not callable(untyped_storage):
            return (id(tensor),)
        storage = untyped_storage()
        cdata = int(getattr(storage, "_cdata", 0))
        data_ptr = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
        return (cdata, data_ptr, nbytes)

    def _prepared_binding_matches(self, prepared: _PreparedTensor) -> bool:
        """Cold/debug validation; never called by a steady-state enqueue."""

        tensor = prepared.owner
        try:
            return (
                tensor.dtype == prepared.dtype
                and bool(tensor.is_cuda)
                and int(tensor.get_device()) == prepared.device
                and bool(tensor.is_contiguous())
                and tuple(tensor.shape) == prepared.shape
                and self._memory_span(tensor) == prepared.span
                and self._tensor_storage_identity(tensor) == prepared.storage_identity
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    def _require_disjoint(
        self,
        tensors: Sequence[tuple[str, Any]],
        *,
        permit_empty: bool,
    ) -> None:
        spans = tuple((name, self._memory_span(tensor)) for name, tensor in tensors)
        self._require_disjoint_spans(spans, permit_empty=permit_empty)

    @classmethod
    def _require_disjoint_spans(
        cls,
        spans: Sequence[tuple[str, tuple[int, int]]],
        *,
        permit_empty: bool,
    ) -> None:
        for index, (left_name, left) in enumerate(spans):
            if permit_empty and left[0] == left[1]:
                continue
            for right_name, right in spans[index + 1 :]:
                if permit_empty and right[0] == right[1]:
                    continue
                if cls._spans_overlap(left, right):
                    raise ValueError(
                        f"{left_name} and {right_name} storage must not overlap"
                    )

    def _require_outside_arena(self, name: str, tensor: Any) -> None:
        self._require_outside_span(
            name, self._memory_span(tensor), self._memory_span(self._arena)
        )

    def _require_outside_owned_storage(self, name: str, span: tuple[int, int]) -> None:
        """Reject aliases with the arena and every separate control allocation."""

        if span[0] == span[1]:
            return
        for prepared in self._iter_owned_descriptors(self._require_prepared_owned()):
            if self._spans_overlap(span, prepared.span):
                raise ValueError(
                    f"{name} storage must not overlap backend-owned {prepared.name}"
                )

    @classmethod
    def _require_outside_span(
        cls,
        name: str,
        span: tuple[int, int],
        arena_span: tuple[int, int],
    ) -> None:
        if span[0] == span[1]:
            return
        if cls._spans_overlap(span, arena_span):
            raise ValueError(f"{name} storage must not overlap the registered arena")

    @staticmethod
    def _require_aligned(name: str, tensor: Any) -> None:
        address = int(tensor.data_ptr())
        if int(tensor.numel()) != 0 and address % 16:
            raise ValueError(f"{name} must be 16-byte aligned")

    def _require_current_topology(self, topology: StableSparseTopology) -> None:
        # The pipeline normally forwards the retained object, so the common
        # path is one pointer comparison. Keep equality as a compatibility
        # fallback for callers that reconstructed the same immutable value.
        if topology is not self._topology and topology != self._topology:
            raise PipelineError(
                "backend topology does not match the committed generation"
            )

    def _require_candidate(self, token: Any) -> _CandidateGeneration:
        if not isinstance(token, _CandidateGeneration):
            raise TypeError("staged token was not created by MappedPipelineBackend")
        if token is not self._staged_candidate:
            raise PipelineError("staged token belongs to another backend or generation")
        return token

    def _required_setup(
        self,
    ) -> tuple[_DeviceDependencies, PipelineLLArenaLayout, Any, int]:
        if (
            self._deps is None
            or self._layout is None
            or self._contract is None
            or self._device is None
        ):
            raise PipelineError("backend setup facts are incomplete")
        return self._deps, self._layout, self._contract, self._device

    def _require_layout(self) -> PipelineLLArenaLayout:
        if self._layout is None:
            raise PipelineError("backend layout is unavailable")
        return self._layout

    def _require_rank(self) -> int:
        if self._rank is None:
            raise PipelineError("backend rank is unavailable")
        return self._rank

    def _agent_name(self, rank: int) -> str:
        return f"cute_pipe_{self.run_id}_{rank}"

    def _tag(self, group: str) -> str:
        return f"pipe-{self.run_id}-{group}"

    def _generation_tag(self, group: str, topology: StableSparseTopology) -> str:
        digest = self._configuration_digest(topology)
        return f"{self._tag(group)}-g{topology.membership_generation}-d{digest}"


__all__ = [
    "MappedBackendState",
    "MappedPipelineBackend",
    "PreparedMappedOperation",
]
