# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-phase host contract for a banked NIXL/CuTe MoE pipeline.

The optimized communication implementation belongs in a device backend.  This
module defines the small host API around it:

``dispatch(x, topk_idx) -> handle``
    Pack *live* device activations, transfer them to their stable expert owners,
    and expose preallocated expert-major device buffers.

``combine(expert_output, topk_weight, handle, output=out) -> out``
    Transfer actual expert outputs back to their token owners and enqueue the
    weighted reduction into a preallocated output buffer.

There is deliberately no expert computation between those calls.  A grouped
GEMM implementation consumes the tensors on :class:`DispatchHandle`, either
uses the same stream or waits on its GPU event, and passes its output to
``combine``. Keeping that
boundary explicit lets the communication runtime compose with different expert
kernels without putting a stand-in operation on the production path.  A
same-stream call sequence is the convenience case; explicit GPU events permit
dispatch of bank ``n + 1`` to overlap expert compute for bank ``n``.

The module has no Torch, CUDA, CuTe, or NIXL import.  Tensor objects are opaque;
only immutable shape/device metadata is inspected.  A backend must preallocate
both physical banks and enqueue all transport waits, release/acquire operations,
and returned credits on the GPU.  The hot calls never read a tensor value and
never synchronize the host.  The CPU lifecycle below is therefore a command
submission and ownership model, not a claim that Python observes device-side
credit completion.

Membership changes are intentionally separate from the steady-state calls.
Views for a new generation may be staged while the old generation runs, but a
commit requires every dispatch handle to have reached ``combine`` and a
request-backed cumulative device drain to complete.  Abrupt loss after traffic
has been posted is fail-stop: this API does not relabel an unflushed bank as
free and does not claim in-place survivor recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

from .ll_protocol import (
    NUM_BANKS,
    UINT32_MAX,
    Direction,
    OperationEpoch,
    PipelineLLArenaLayout,
    ProtocolError,
    StableSparseTopology,
    TwoBankCreditLifecycle,
)

MAX_PREPARED_SUBMISSION_OPERATIONS = 1 << 16
"""Maximum operations materialized by one allocation-free submission plan.

Each admitted operation owns a distinct epoch and dispatch handle.  Bounding
the cold O(N) plan prevents an untrusted count from turning validation into an
unbounded host-memory allocation.  Longer services submit multiple finite
plans; an operation handle is never recycled across those plans.
"""
_EMPTY_REFERENCES: tuple[Any, ...] = ()


class PipelineError(ProtocolError):
    """Base error for host-side pipeline contract violations."""


class PipelineFailed(PipelineError):
    """The communicator is fail-stopped and cannot submit more traffic."""


class AbruptPeerLoss(PipelineFailed):
    """An active peer disappeared after traffic could have been posted."""

    def __init__(
        self,
        *,
        peer_rank: int,
        membership_generation: int,
        peer_incarnation: int,
        detail: str,
    ) -> None:
        self.peer_rank = peer_rank
        self.membership_generation = membership_generation
        self.peer_incarnation = peer_incarnation
        self.detail = detail
        super().__init__(
            f"active peer {peer_rank} incarnation {peer_incarnation} was lost "
            f"in membership generation {membership_generation}: {detail}; "
            "the communicator is fail-stopped because outstanding requestless "
            "device operations cannot be cancelled or flushed safely"
        )


class PipelineState(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSING = "closing"
    FAILED = "failed"
    CLOSED = "closed"


class HandleState(str, Enum):
    DISPATCHED = "dispatched"
    COMBINE_ENQUEUED = "combine_enqueued"


class GenerationChangeState(str, Enum):
    STAGED = "staged"
    DRAIN_ENQUEUED = "drain_enqueued"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PipelineBankBuffers:
    """Preallocated device tensors for one physical communication bank.

    The backend owns storage and constructs typed views over its registered
    arena.  Expected logical shapes are:

    * ``expert_input``: dense
      ``[experts_per_rank, max_ranks * T, hidden_size]`` scratch;
    * ``expert_counts``: contiguous int32 ``[experts_per_rank]`` dense-prefix
      lengths, allocated once per bank outside the registered transport arena;
    * ``source_info``: dense ``[experts_per_rank, max_ranks * T, 4]`` u32 words
      ``(origin_rank, origin_token, route_slot, reserved=0)``;
    * ``layout_ranges``: ``[experts_per_rank, max_ranks]`` u64 per-source
      ranges packed as ``(begin << 32) | count``;
    * ``combine_stage``: registered
      ``[experts_per_rank, max_ranks * T, hidden_size]`` output target.

    ``T`` is fixed maximum token capacity.  The registered incoming arena is
    source-sharded and remains internal to the backend; dispatch compacts it
    into the exposed dense planes.  The receive path is therefore not claimed
    as zero-copy.  Counts and ranges remain device values; the host must not
    materialize them to launch the expert kernel.  Expert epilogues can write
    directly to ``combine_stage`` to avoid a second outgoing copy. The final
    token-major output is caller-owned rather than bank-owned, so later bank
    reuse cannot overwrite a tensor still consumed by a downstream GPU stream.
    These views have stable storage for the backend lifetime. Expert adapters
    may consume them but must never call ``set_()``, ``resize_()``, or any
    operation that rebinds their storage; cached device descriptors deliberately
    avoid re-reading tensor metadata in the launch path.
    """

    bank: int
    expert_input: Any
    expert_counts: Any
    source_info: Any
    layout_ranges: Any
    combine_stage: Any

    def __post_init__(self) -> None:
        if isinstance(self.bank, bool) or not isinstance(self.bank, int):
            raise TypeError("bank must be an integer")
        if not 0 <= self.bank < NUM_BANKS:
            raise ValueError(f"bank must be in [0, {NUM_BANKS})")
        for name in (
            "expert_input",
            "expert_counts",
            "source_info",
            "layout_ranges",
            "combine_stage",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must be a preallocated device tensor")


class PreparedOperation:
    """Opaque pipeline-owned binding for one stable caller operand set.

    Construction performs every shape/device/backend descriptor check once.
    The exact tensor objects and their storage remain immutable until pipeline
    close; values may change under the normal GPU readiness dependencies. Hot
    dispatch/combine then perform identity/state checks only.
    """

    __slots__ = (
        "_activations",
        "_backend_token",
        "_combine_references",
        "_dispatch_references",
        "_num_tokens",
        "_output",
        "_owner",
        "_topk_indices",
        "_topk_weights",
    )

    def __init__(
        self,
        *,
        owner: object,
        activations: Any,
        topk_indices: Any,
        topk_weights: Any,
        output: Any,
        num_tokens: int,
        backend_token: Any,
        combine_stages: tuple[Any, Any],
    ) -> None:
        self._owner = owner
        self._activations = activations
        self._topk_indices = topk_indices
        self._topk_weights = topk_weights
        self._output = output
        self._num_tokens = num_tokens
        self._backend_token = backend_token
        # These exact immutable tuples are part of the cold binding.  The
        # allocation-free rolling path installs them directly before crossing
        # its asynchronous backend boundaries.
        self._dispatch_references = (activations, topk_indices)
        self._combine_references = (
            (
                activations,
                topk_indices,
                combine_stages[0],
                topk_weights,
                output,
            ),
            (
                activations,
                topk_indices,
                combine_stages[1],
                topk_weights,
                output,
            ),
        )

    @property
    def num_tokens(self) -> int:
        return self._num_tokens


class SubmissionPreflight:
    """Opaque one-shot proof for one exact eager schedule boundary.

    A token is issued only after the backend validates the complete operation
    range and rejects CUDA capture on every scheduled stream. Public schedule
    helpers consume it exactly once immediately before their first post. This
    historical observation cannot lock a CUDA stream: validity requires
    exclusive single-host-submitter ownership and no capture or intervening
    CUDA operation on an admitted stream between issue and consumption.
    """

    __slots__ = (
        "_binding_owner",
        "_backend_context",
        "_consumed",
        "_first_step",
        "_frames",
        "_owner",
        "_step_count",
        "_streams",
        "_topology",
    )

    def __init__(
        self,
        *,
        owner: object,
        topology: StableSparseTopology,
        streams: tuple[Any, ...],
        first_step: int,
        step_count: int,
        binding_owner: object | None = None,
        backend_context: Any | None = None,
        frames: tuple["_PreparedSubmissionFrame", ...] = (),
    ) -> None:
        self._owner = owner
        self._topology = topology
        self._streams = streams
        self._first_step = first_step
        self._step_count = step_count
        self._consumed = False
        self._binding_owner = binding_owner
        self._backend_context = backend_context
        self._frames = frames

    def __repr__(self) -> str:
        return "<SubmissionPreflight opaque>"


@dataclass(frozen=True, slots=True)
class GenerationDrain:
    """Opaque cumulative transport drain returned by a backend."""

    token: Any


GENERATION_REBASE_PLANES = frozenset(
    {
        "control",
        "rank_mask",
        "rank_incarnation",
        "dispatch_ready_seq",
        "dispatch_credit",
        "combine_ready_seq",
        "combine_credit",
        "bank_state",
        "abort_state",
    }
)


@dataclass(frozen=True, slots=True)
class GenerationCommit:
    """Backend attestation that old sequence/credit state cannot alias."""

    membership_generation: int
    rebased_planes: frozenset[str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.membership_generation, bool)
            or not isinstance(self.membership_generation, int)
            or self.membership_generation < 0
        ):
            raise ValueError("membership_generation must be a non-negative integer")
        object.__setattr__(self, "rebased_planes", frozenset(self.rebased_planes))


@dataclass(frozen=True, slots=True)
class CombineEvents:
    """Two device-side milestones of an asynchronous combine operation."""

    output_ready: Any
    bank_reusable: Any

    def __post_init__(self) -> None:
        if self.output_ready is None:
            raise ValueError("output_ready must be a device event")
        if self.bank_reusable is None:
            raise ValueError("bank_reusable must be a device event")


@runtime_checkable
class PipelineBackend(Protocol):
    """Device backend required by :class:`MoEPipeline`.

    ``enqueue_dispatch`` and ``enqueue_combine`` must be nonblocking host APIs.
    Dispatch inserts device waits for ``input_ready_event`` and a cross-stream
    ``communication_predecessor_event`` before touching its bank, then records
    the returned event after receive compaction. Combine likewise waits for its
    communication predecessor and ``expert_ready_event``. This total order is
    mandatory: peer-polling cooperative kernels on independent streams could
    otherwise execute in opposite orders on two ranks and deadlock. Its
    ``output_ready`` event follows origin reduction;
    ``bank_reusable`` additionally follows every peer-originated device credit.
    Same-peer data and publication operations use one ordered NIXL channel and
    receive polling uses system acquire semantics. No device event may be
    queried by the CPU and no host progress loop may enter a generation.

    The backend also validates dtype and storage contracts from host metadata:
    routing indices and ``expert_counts`` must be int32, activations/outputs must
    match ``layout.element_size``, and all operands must be contiguous device
    storage. Packed ``layout_ranges`` remain u64.
    Before an enqueue method returns it must also record allocator lifetime on
    every stream that consumed a tensor (or retain an equivalent strong
    reference); the host model deliberately never synchronizes before rotating
    its per-bank reference set. Enqueue failure means posting state is unknown:
    the host quarantines both the old and new references, and production
    ``fail_stop`` must promptly terminate the process/CUDA context instead of
    deregistering an arena that device work might still reference.

    ``commit_generation`` is the one intentionally control-plane operation. It
    must wait for the cumulative drain, globally converge the membership, and
    rebase every ready sequence, returned credit, bank-state, and fatal-state
    word before activating the staged GPU mask/views. This rebase is mandatory
    because host operation steps restart at zero in the new generation. The
    returned :class:`GenerationCommit` attests that ordering; it is not merely
    descriptive metadata.

    ``validate_submission_context`` checks a complete half-open step range once
    before a public schedule posts anything. The mapped protocol rejects CUDA
    Graph capture because replaying a by-value operation epoch would permit
    stale ready/credit ABA. This point-in-time check belongs at the schedule
    boundary so eager execution pays no per-kernel capture-query overhead; it
    does not provide cross-thread exclusion or lock a CUDA stream.
    """

    def preallocate(  # noqa: E704 - Protocol ellipsis body is Black's format.
        self,
        *,
        rank: int,
        layout: PipelineLLArenaLayout,
        topology: StableSparseTopology,
    ) -> Sequence[PipelineBankBuffers]: ...

    def prepare_operation(
        self,
        activations: Any,
        topk_indices: Any,
        topk_weights: Any,
        combine_output: Any,
    ) -> Any:
        """Cold-seal one reusable caller operand set for direct launch."""
        ...

    def prepare_submission_context(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> Any:
        """Cold-bind a finite stream/step context and return its identity."""
        ...

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
    ) -> Any:
        """Cold-bind one operation's immutable backend launch arguments."""
        ...

    def validate_prepared_submission_context(self, context: Any) -> None:
        """Fail if a one-shot proof's backend context was superseded."""
        ...

    def validate_submission_context(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> None:
        """Reject a schedule context that cannot preserve protocol epochs."""
        ...

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
        prepared_operation: Any | None = None,
    ) -> Any:
        """Return a GPU event recorded after expert inputs become visible."""
        ...

    def enqueue_prepared_dispatch(
        self,
        prepared_launch: Any,
        *,
        input_ready_event: Any | None,
        communication_predecessor_event: Any | None,
    ) -> Any:
        """Post one fully cold-bound rolling dispatch."""
        ...

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
        prepared_operation: Any | None = None,
    ) -> CombineEvents:
        """Return output-ready and full-bank-reusable GPU events."""
        ...

    def enqueue_prepared_combine(
        self,
        prepared_launch: Any,
        *,
        expert_ready_event: Any | None,
        communication_predecessor_event: Any | None,
    ) -> CombineEvents:
        """Post one fully cold-bound rolling combine."""
        ...

    def stage_generation(  # noqa: E704 - Protocol ellipsis body is Black's format.
        self,
        *,
        current: StableSparseTopology,
        next_topology: StableSparseTopology,
    ) -> Any: ...

    def enqueue_generation_drain(
        self,
        *,
        current: StableSparseTopology,
        staged_token: Any,
        completion_events: Sequence[Any],
        stream: Any,
    ) -> GenerationDrain:
        """Device-wait all bank events, then enqueue cumulative channel drains."""
        ...

    def commit_generation(
        self,
        *,
        current: StableSparseTopology,
        next_topology: StableSparseTopology,
        staged_token: Any,
        drain: GenerationDrain,
    ) -> GenerationCommit:
        """Drain, rebase every generation plane, activate, then attest."""
        ...

    def fail_stop(self, error: BaseException) -> None:
        """Terminate production execution; returning is valid only in a model."""
        ...

    def close(self) -> None:
        """Gracefully drain before release, or retain uncertain failed resources."""
        ...


class DispatchHandle:
    """One-shot ownership token spanning dispatch, expert work, and combine.

    Applications may read the expert-facing device tensors, but only the
    creating pipeline can consume the handle. The retained activation/routing
    references keep enqueue inputs alive until the backend records combine.
    Storage lifetime is not mutation safety: callers must not modify
    ``topk_indices`` until an output wait enqueued through
    :meth:`enqueue_output_wait` has completed because combine deliberately
    rereads the route tensor on the GPU.
    """

    __slots__ = (
        "_activations",
        "_buffers",
        "_combine_completion_event",
        "_completion_events_valid",
        "_bank_reuse_event",
        "_backend_launch",
        "_dispatch_ready_event",
        "_dispatch_event_valid",
        "_dispatch_stream",
        "_num_tokens",
        "_operation",
        "_owner",
        "_source_incarnation",
        "_state",
        "_topk_indices",
    )

    def __init__(
        self,
        *,
        owner: object,
        operation: OperationEpoch,
        source_incarnation: int,
        num_tokens: int,
        activations: Any,
        topk_indices: Any,
        buffers: PipelineBankBuffers,
        dispatch_stream: Any,
        dispatch_ready_event: Any,
        backend_launch: Any | None = None,
    ) -> None:
        self._owner = owner
        self._operation = operation
        self._source_incarnation = source_incarnation
        self._num_tokens = num_tokens
        self._activations = activations
        self._topk_indices = topk_indices
        self._buffers = buffers
        self._backend_launch = backend_launch
        self._dispatch_stream = dispatch_stream
        self._dispatch_ready_event = dispatch_ready_event
        self._dispatch_event_valid = dispatch_ready_event is not None
        self._combine_completion_event: Any = None
        self._bank_reuse_event: Any = None
        self._completion_events_valid = False
        self._state = HandleState.DISPATCHED

    @property
    def operation(self) -> OperationEpoch:
        return self._operation

    @property
    def bank(self) -> int:
        return self._operation.bank

    @property
    def membership_generation(self) -> int:
        return self._operation.membership_generation

    @property
    def source_incarnation(self) -> int:
        return self._source_incarnation

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def state(self) -> HandleState:
        return self._state

    @property
    def expert_input(self) -> Any:
        return self._buffers.expert_input

    @property
    def expert_counts(self) -> Any:
        return self._buffers.expert_counts

    @property
    def source_info(self) -> Any:
        return self._buffers.source_info

    @property
    def layout_ranges(self) -> Any:
        return self._buffers.layout_ranges

    @property
    def combine_buffer(self) -> Any:
        """Registered expert-output target for zero-copy combine staging."""

        return self._buffers.combine_stage

    def enqueue_dispatch_wait(self, stream: Any) -> None:
        """Make ``stream`` wait for this dispatch while the event lease is live.

        The raw CUDA event is intentionally not exposed.  Backends may reuse a
        bounded event ring after the handle reaches combine; CUDA guarantees an
        already-enqueued wait keeps the captured record, whereas a late wait on
        a re-recorded event would silently target another operation.
        """

        if not self._dispatch_event_valid:
            raise PipelineError(
                "dispatch-ready event lease expired; enqueue every expert-stream "
                "wait before submitting combine"
            )
        self._enqueue_wait(stream, self._dispatch_ready_event)

    def enqueue_output_wait(self, stream: Any) -> None:
        """Make a downstream stream wait for origin reduction completion."""

        if not self._completion_events_valid:
            raise PipelineError(
                "output-ready event is unavailable or its lease expired; enqueue "
                "downstream waits before reusing this physical bank"
            )
        self._enqueue_wait(stream, self._combine_completion_event)

    def enqueue_bank_reuse_wait(self, stream: Any) -> None:
        """Make an observer stream wait for all returned peer credits."""

        if not self._completion_events_valid:
            raise PipelineError(
                "bank-reuse event is unavailable or its lease expired; enqueue "
                "observer waits before reusing this physical bank"
            )
        self._enqueue_wait(stream, self._bank_reuse_event)

    @staticmethod
    def _enqueue_wait(stream: Any, event: Any) -> None:
        if stream is None:
            raise ValueError("stream must be an explicit device stream")
        wait_event = getattr(stream, "wait_event", None)
        if not callable(wait_event):
            raise TypeError("stream must expose wait_event(event)")
        wait_event(event)

    def _expire_dispatch_event(self) -> None:
        self._dispatch_event_valid = False

    def _expire_completion_events(self) -> None:
        self._completion_events_valid = False


@dataclass(frozen=True, slots=True)
class _PreparedSubmissionFrame:
    """Cold-owned state for one never-reused rolling operation."""

    prepared_operation: PreparedOperation
    handle: DispatchHandle
    bank: int
    backend_launch: Any
    dispatch_references: tuple[Any, Any]
    combine_references: tuple[Any, Any, Any, Any, Any]
    next_step: int


class StagedGeneration:
    """A fully prepared but not yet visible membership generation."""

    __slots__ = ("_backend_token", "_drain", "_next", "_owner", "_state")

    def __init__(
        self,
        *,
        owner: object,
        next_topology: StableSparseTopology,
        backend_token: Any,
    ) -> None:
        self._owner = owner
        self._next = next_topology
        self._backend_token = backend_token
        self._drain: GenerationDrain | None = None
        self._state = GenerationChangeState.STAGED

    @property
    def topology(self) -> StableSparseTopology:
        return self._next

    @property
    def state(self) -> GenerationChangeState:
        return self._state


def _shape(tensor: Any, name: str) -> tuple[int, ...]:
    """Read framework tensor metadata without touching device storage."""

    try:
        shape = tensor.shape
        result = tuple(shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must expose an immutable tensor shape") from error
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{name} shape must contain Python integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} shape cannot contain a negative extent")
    return result


def _device(tensor: Any, name: str) -> Any:
    try:
        return tensor.device
    except AttributeError as error:
        raise TypeError(f"{name} must expose a device") from error


def _capacity(layout: PipelineLLArenaLayout, name: str) -> int:
    try:
        value = getattr(layout, name)
    except AttributeError as error:
        raise TypeError(f"layout must expose {name}") from error
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"layout.{name} must be a positive integer")
    return value


class MoEPipeline:
    """Submit a two-bank, split-phase MoE communication pipeline.

    The class allocates no tensor storage after construction.  At most two
    dispatch handles may be outstanding, one per physical bank.  Calling
    ``combine`` releases the *host submission lease*; the backend must still
    make the next same-bank device kernel wait for its returned credit before
    overwriting storage. Every active rank must submit the exact same total
    sequence of dispatch and combine collective types, generations, and steps;
    the local CUDA-event chain prevents stream overtaking but cannot reconcile
    divergent programs on different ranks. Prefer the canonical rolling helper
    in ``elastic_moe_pipeline.py``. An active rank with zero local tokens still
    calls dispatch and combine so it can publish empty buckets and return
    credits. Submission APIs and handle event
    leases are single-host-thread objects: one serialized submission thread
    must own each pipeline. Concurrent dispatch/combine/wait calls require an
    application command sequencer and are otherwise invalid.
    """

    def __init__(
        self,
        *,
        rank: int,
        topology: StableSparseTopology,
        layout: PipelineLLArenaLayout,
        backend: PipelineBackend,
    ) -> None:
        if not isinstance(topology, StableSparseTopology):
            raise TypeError("topology must be a StableSparseTopology")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < topology.max_ranks:
            raise ValueError(f"rank must be in [0, {topology.max_ranks})")
        if not isinstance(backend, PipelineBackend):
            raise TypeError("backend does not implement PipelineBackend")
        if not isinstance(layout, PipelineLLArenaLayout):
            raise TypeError("layout must be a PipelineLLArenaLayout")
        if _capacity(layout, "max_ranks") != topology.max_ranks:
            raise ValueError("layout and topology max_ranks differ")
        if _capacity(layout, "experts_per_rank") != topology.experts_per_rank:
            raise ValueError("layout and topology experts_per_rank differ")
        _capacity(layout, "num_tokens")
        _capacity(layout, "top_k")
        _capacity(layout, "hidden_size")

        bank_buffers = tuple(
            backend.preallocate(rank=rank, layout=layout, topology=topology)
        )
        if len(bank_buffers) != NUM_BANKS:
            raise ValueError(f"backend must preallocate exactly {NUM_BANKS} banks")
        if any(not isinstance(item, PipelineBankBuffers) for item in bank_buffers):
            raise TypeError("backend preallocate returned an invalid bank buffer")
        by_bank = {item.bank: item for item in bank_buffers}
        if set(by_bank) != set(range(NUM_BANKS)):
            raise ValueError("backend must return each physical bank exactly once")
        bank_devices = tuple(
            self._validate_preallocated_buffers(layout, by_bank[bank])
            for bank in range(NUM_BANKS)
        )
        if bank_devices[0] != bank_devices[1]:
            raise ValueError("both pipeline banks must reside on one device")

        self.rank = rank
        self.layout = layout
        self.backend = backend
        self._device = bank_devices[0]
        self._token_capacity = layout.num_tokens
        self._top_k = layout.top_k
        self._hidden_size = layout.hidden_size
        self._experts_per_rank = layout.experts_per_rank
        self._padded_expert_rows = layout.max_ranks * layout.num_tokens
        self._topology = topology
        self._buffers = tuple(by_bank[bank] for bank in range(NUM_BANKS))
        self._owner = object()
        self._state = PipelineState.ACTIVE
        self._failure: BaseException | None = None
        self._next_step = 0
        self._handles: list[DispatchHandle | None] = [None] * NUM_BANKS
        self._completed_handles: list[DispatchHandle | None] = [None] * NUM_BANKS
        self._retained: list[tuple[Any, ...]] = [(), ()]
        # When a same-bank successor crosses the backend enqueue boundary, the
        # old and new operand sets must both survive an uncertain failure.  A
        # fixed cold-allocated slot avoids tuple concatenation (and therefore
        # allocation failure) on the fail-stop path.
        self._failed_previous_references: list[tuple[Any, ...]] = [(), ()]
        self._completion_events: list[Any | None] = [None] * NUM_BANKS
        self._communication_tail_event: Any | None = None
        self._communication_tail_stream: Any | None = None
        self._staged: StagedGeneration | None = None

        # These are the checked executable model for the device-side credit
        # domains.  Device backends use the same events in kernels; the Python
        # hot path does not fake completion by advancing them on enqueue.
        self.dispatch_credit_model = TwoBankCreditLifecycle(
            Direction.DISPATCH, topology.max_ranks
        )
        self.combine_credit_model = TwoBankCreditLifecycle(
            Direction.COMBINE, topology.max_ranks
        )

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def topology(self) -> StableSparseTopology:
        return self._topology

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def outstanding_handles(self) -> tuple[DispatchHandle, ...]:
        return tuple(handle for handle in self._handles if handle is not None)

    @property
    def next_operation_step(self) -> int:
        """Return the first operation step a public schedule would submit."""

        return self._next_step

    def prepare_operation(
        self,
        activations: Any,
        topk_indices: Any,
        topk_weights: Any,
        output: Any,
    ) -> PreparedOperation:
        """Cold-validate and seal one exact reusable operation binding."""

        self._require_active()
        x_shape = _shape(activations, "activations")
        if len(x_shape) != 2:
            raise ValueError("activations must have shape [tokens, hidden_size]")
        num_tokens, hidden_size = x_shape
        if not 0 <= num_tokens <= self._token_capacity:
            raise ValueError("activation token count exceeds fixed pipeline capacity")
        if hidden_size != self._hidden_size:
            raise ValueError("activation hidden size does not match pipeline layout")
        route_shape = (num_tokens, self._top_k)
        for name, tensor in (
            ("topk_indices", topk_indices),
            ("topk_weights", topk_weights),
        ):
            actual = _shape(tensor, name)
            if actual != route_shape:
                raise ValueError(f"{name} must have shape {route_shape}, got {actual}")
        output_shape = (num_tokens, self._hidden_size)
        actual_output = _shape(output, "output")
        if actual_output != output_shape:
            raise ValueError(
                f"output must have shape {output_shape}, got {actual_output}"
            )
        device = _device(activations, "activations")
        for name, tensor in (
            ("topk_indices", topk_indices),
            ("topk_weights", topk_weights),
            ("output", output),
        ):
            if _device(tensor, name) != device:
                raise ValueError("prepared operation tensors must share one device")
        if device != self._device:
            raise ValueError(
                "prepared operation tensors must reside on pipeline device"
            )
        backend_token = self.backend.prepare_operation(
            activations,
            topk_indices,
            topk_weights,
            output,
        )
        if backend_token is None:
            raise TypeError("backend prepare_operation must return an opaque token")
        return PreparedOperation(
            owner=self._owner,
            activations=activations,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            output=output,
            num_tokens=num_tokens,
            backend_token=backend_token,
            combine_stages=(
                self._buffers[0].combine_stage,
                self._buffers[1].combine_stage,
            ),
        )

    def validate_submission_context(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> SubmissionPreflight:
        """Validate one complete eager operation range before work is posted.

        This is called once per public schedule, not once per kernel. The mapped
        backend rejects CUDA Graph capture because graph replay would otherwise
        reuse the by-value operation epoch and permit ready/credit ABA. The
        half-open range must start at this pipeline's next operation so the
        preflight cannot accidentally attest a different sequence. The returned
        opaque token is bound to this pipeline, topology object, exact stream
        sequence, and range, and can be consumed by one public helper only.
        It does not lock the streams. The caller must exclusively own submission
        and issue no capture or intervening CUDA operation on them before use.
        """

        normalized_streams = self._validate_submission_boundary(
            streams=streams,
            first_step=first_step,
            step_count=step_count,
        )
        backend_context = self.backend.prepare_submission_context(
            streams=normalized_streams,
            first_step=first_step,
            step_count=step_count,
        )
        if backend_context is None:
            raise TypeError(
                "backend prepare_submission_context must return an opaque context"
            )
        # Allocate the one-shot proof before the backend makes its point-in-time
        # capture observation.  On return from that observation only returning
        # this already-built object remains before caller consumption.
        preflight = SubmissionPreflight(
            owner=self._owner,
            topology=self._topology,
            streams=normalized_streams,
            first_step=first_step,
            step_count=step_count,
            backend_context=backend_context,
        )
        self.backend.validate_submission_context(
            streams=normalized_streams,
            first_step=first_step,
            step_count=step_count,
        )
        return preflight

    def _validate_submission_boundary(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
    ) -> tuple[Any, ...]:
        """Pure host validation shared by generic and materialized proofs."""

        self._require_active()
        if not isinstance(streams, Sequence) or not streams:
            raise ValueError("streams must be a non-empty sequence")
        if any(stream is None for stream in streams):
            raise ValueError("submission streams must be explicit")
        if isinstance(first_step, bool) or not isinstance(first_step, int):
            raise TypeError("first_step must be an integer")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("step_count must be an integer")
        if step_count <= 0:
            raise ValueError("step_count must be a positive integer")
        if first_step != self._next_step:
            raise PipelineError(
                "submission range must start at the pipeline's next operation step"
            )
        if first_step < 0 or first_step > UINT32_MAX:
            raise ValueError("first_step must fit uint32")
        if first_step + step_count > UINT32_MAX + 1:
            raise ValueError("submission operation-step range exceeds uint32")
        return tuple(streams)

    def _materialize_bound_submission_frames(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        prepared_operations: Sequence[PreparedOperation],
        dispatch_streams: tuple[Any, Any],
        expert_streams: tuple[Any, Any],
        backend_context: Any,
    ) -> tuple[_PreparedSubmissionFrame, ...]:
        """Cold-materialize a finite rolling submission before capture admission.

        Only the pipeline can construct valid handles.  The returned frame set
        has one immutable :class:`OperationEpoch` and one unique, never-reused
        :class:`DispatchHandle` per operation. Allocation is O(N), cold, and
        capped by
        :data:`MAX_PREPARED_SUBMISSION_OPERATIONS`.

        This method deliberately does *not* query capture state.  The elastic
        wrapper first allocates its expert slots and terminal leases, builds the
        proof, then calls :meth:`_admit_bound_submission_preflight` so the
        capture observation is the final cold action before consumption.
        """

        step_count = len(prepared_operations)
        if step_count > MAX_PREPARED_SUBMISSION_OPERATIONS:
            raise ValueError(
                "rolling operation_count exceeds the cold-plan capacity "
                f"{MAX_PREPARED_SUBMISSION_OPERATIONS}"
            )
        if len(dispatch_streams) != NUM_BANKS or any(
            stream is None for stream in dispatch_streams
        ):
            raise ValueError("dispatch_streams must contain two explicit streams")
        if len(expert_streams) != NUM_BANKS or any(
            stream is None for stream in expert_streams
        ):
            raise ValueError("expert_streams must contain two explicit streams")
        self._validate_submission_boundary(
            streams=streams,
            first_step=first_step,
            step_count=step_count,
        )
        self._require_local_active()
        for prepared in prepared_operations:
            self._validate_prepared_operation(prepared)

        generation = self._topology.membership_generation
        incarnation = self._topology.incarnation(self.rank)
        frames: list[_PreparedSubmissionFrame] = []
        for offset, prepared in enumerate(prepared_operations):
            operation = OperationEpoch(generation, first_step + offset)
            bank = operation.bank
            backend_launch = self.backend.prepare_operation_launch(
                submission_context=backend_context,
                operation=operation,
                source_incarnation=incarnation,
                topology=self._topology,
                buffers=self._buffers[bank],
                prepared_operation=prepared._backend_token,
                communication_stream=dispatch_streams[bank],
                expert_stream=expert_streams[bank],
            )
            if backend_launch is None:
                raise TypeError(
                    "backend prepare_operation_launch must return an opaque token"
                )
            handle = DispatchHandle(
                owner=self._owner,
                operation=operation,
                source_incarnation=incarnation,
                num_tokens=prepared._num_tokens,
                activations=prepared._activations,
                topk_indices=prepared._topk_indices,
                buffers=self._buffers[bank],
                dispatch_stream=dispatch_streams[bank],
                dispatch_ready_event=None,
                backend_launch=backend_launch,
            )
            frames.append(
                _PreparedSubmissionFrame(
                    prepared_operation=prepared,
                    handle=handle,
                    bank=bank,
                    backend_launch=backend_launch,
                    dispatch_references=prepared._dispatch_references,
                    combine_references=prepared._combine_references[bank],
                    next_step=first_step + offset + 1,
                )
            )
        return tuple(frames)

    def _build_bound_submission_preflight(
        self,
        *,
        streams: Sequence[Any],
        first_step: int,
        frames: tuple[_PreparedSubmissionFrame, ...],
        binding_owner: object,
        backend_context: Any,
    ) -> SubmissionPreflight:
        """Build, but do not yet capture-admit, a materialized plan proof."""

        normalized_streams = self._validate_submission_boundary(
            streams=streams,
            first_step=first_step,
            step_count=len(frames),
        )
        return SubmissionPreflight(
            owner=self._owner,
            topology=self._topology,
            streams=normalized_streams,
            first_step=first_step,
            step_count=len(frames),
            binding_owner=binding_owner,
            backend_context=backend_context,
            frames=frames,
        )

    def _admit_bound_submission_preflight(
        self,
        preflight: SubmissionPreflight,
    ) -> None:
        """Perform the final point-in-time capture query for a built proof."""

        self._require_active()
        if preflight._owner is not self._owner:
            raise PipelineError("submission preflight belongs to another pipeline")
        if preflight._topology is not self._topology:
            raise PipelineError("submission preflight belongs to a stale generation")
        if preflight._first_step != self._next_step:
            raise PipelineError(
                "submission range no longer starts at the next operation step"
            )
        self.backend.validate_submission_context(
            streams=preflight._streams,
            first_step=preflight._first_step,
            step_count=preflight._step_count,
        )

    def _consume_submission_preflight(
        self,
        preflight: SubmissionPreflight,
        *,
        streams: Sequence[Any],
        first_step: int,
        step_count: int,
        binding_owner: object | None = None,
    ) -> None:
        """Consume a schedule proof using state and identity checks only."""

        self._require_active()
        if not isinstance(preflight, SubmissionPreflight):
            raise TypeError("submission_preflight must be a SubmissionPreflight")
        if preflight._owner is not self._owner:
            raise PipelineError("submission preflight belongs to another pipeline")
        if preflight._consumed:
            raise PipelineError("submission preflight was already consumed")
        if preflight._topology is not self._topology:
            raise PipelineError(
                "submission preflight belongs to a stale membership generation"
            )
        if first_step != self._next_step:
            raise PipelineError(
                "submission range no longer starts at the next operation step"
            )
        if preflight._first_step != first_step or preflight._step_count != step_count:
            raise PipelineError("submission preflight operation range does not match")
        if preflight._binding_owner is not binding_owner:
            raise PipelineError("submission preflight binding does not match")
        if len(streams) != len(preflight._streams) or any(
            actual is not expected
            for actual, expected in zip(streams, preflight._streams)
        ):
            raise PipelineError("submission preflight stream sequence does not match")
        if preflight._backend_context is None:
            raise PipelineError("submission preflight has no prepared backend context")
        self.backend.validate_prepared_submission_context(preflight._backend_context)
        preflight._consumed = True

    def dispatch_prepared(
        self,
        prepared: PreparedOperation,
        *,
        stream: Any,
        input_ready_event: Any | None = None,
    ) -> DispatchHandle:
        """Enqueue a sealed dispatch with no tensor metadata inspection.

        Direct dispatch/combine calls are eager-only. Use a public layer or
        rolling helper to obtain the once-per-schedule CUDA-capture rejection.
        """

        self._require_active()
        self._require_stream(stream)
        self._require_local_active()
        self._validate_prepared_operation(prepared)
        return self._dispatch_validated(
            prepared._activations,
            prepared._topk_indices,
            num_tokens=prepared._num_tokens,
            stream=stream,
            input_ready_event=input_ready_event,
            prepared_operation=prepared._backend_token,
        )

    def dispatch(
        self,
        activations: Any,
        topk_indices: Any,
        *,
        stream: Any,
        input_ready_event: Any | None = None,
    ) -> DispatchHandle:
        """Enqueue live-token dispatch without a host tensor read or wait.

        ``activations`` must remain immutable until the dispatch-ready milestone;
        ``topk_indices`` must remain immutable through the later combine milestone.
        """

        self._require_active()
        self._require_stream(stream)
        self._require_local_active()
        x_shape = _shape(activations, "activations")
        route_shape = _shape(topk_indices, "topk_indices")
        if len(x_shape) != 2:
            raise ValueError("activations must have shape [tokens, hidden_size]")
        num_tokens, hidden_size = x_shape
        if not 0 <= num_tokens <= self._token_capacity:
            raise ValueError("activation token count exceeds fixed pipeline capacity")
        if hidden_size != self._hidden_size:
            raise ValueError("activation hidden size does not match pipeline layout")
        expected_route_shape = (num_tokens, self._top_k)
        if route_shape != expected_route_shape:
            raise ValueError(
                f"topk_indices must have shape {expected_route_shape}, got "
                f"{route_shape}"
            )
        activation_device = _device(activations, "activations")
        if activation_device != _device(topk_indices, "topk_indices"):
            raise ValueError("activations and topk_indices must share one device")
        if activation_device != self._device:
            raise ValueError("dispatch tensors must reside on the pipeline device")

        return self._dispatch_validated(
            activations,
            topk_indices,
            num_tokens=num_tokens,
            stream=stream,
            input_ready_event=input_ready_event,
            prepared_operation=None,
        )

    def _dispatch_validated(
        self,
        activations: Any,
        topk_indices: Any,
        *,
        num_tokens: int,
        stream: Any,
        input_ready_event: Any | None,
        prepared_operation: Any | None,
    ) -> DispatchHandle:
        """Post an already validated raw or prepared dispatch."""

        operation = OperationEpoch(
            self._topology.membership_generation, self._next_step
        )
        bank = operation.bank
        if self._handles[bank] is not None:
            active = self._handles[bank]
            assert active is not None
            raise PipelineError(
                f"bank {bank} still has uncombined operation "
                f"{active.operation.wire_value}"
            )
        # A CUDA stream wait captures the event's current record.  Once the
        # successor starts using this bank, no external code may enqueue a late
        # wait through the predecessor handle because a bounded backend event
        # ring can eventually re-record the same native event.
        previous = self._completed_handles[bank]
        if previous is not None:
            previous._expire_completion_events()
            self._completed_handles[bank] = None
        incarnation = self._topology.incarnation(self.rank)
        buffers = self._buffers[bank]
        # The backend may raise after posting work but before it can attach
        # allocator lifetime to every consumed stream. Quarantine both the
        # preceding same-bank operands and this attempt before crossing that
        # uncertainty boundary. A successful enqueue has recorded stream
        # lifetime and can rotate the strong-reference set below.
        prior_references = self._retained[bank]
        attempt_references = (activations, topk_indices)
        # Allocate every Python object required by successful publication
        # before device work can be posted.  In particular, constructing the
        # public handle after enqueue would leave an unrepresentable in-flight
        # operation if Python allocation failed at that point.
        next_step = self._next_step + 1
        handle = DispatchHandle(
            owner=self._owner,
            operation=operation,
            source_incarnation=incarnation,
            num_tokens=num_tokens,
            activations=activations,
            topk_indices=topk_indices,
            buffers=buffers,
            dispatch_stream=stream,
            dispatch_ready_event=None,
        )
        communication_predecessor_event = (
            self._communication_tail_event
            if self._communication_tail_stream is not stream
            else None
        )
        # Both assignments target fixed-size lists allocated at construction.
        # They make the failure quarantine allocation-free while the healthy
        # path still performs no tuple concatenation.
        self._failed_previous_references[bank] = prior_references
        self._retained[bank] = attempt_references
        try:
            dispatch_ready_event = self.backend.enqueue_dispatch(
                operation=operation,
                source_incarnation=incarnation,
                topology=self._topology,
                activations=activations,
                topk_indices=topk_indices,
                input_ready_event=input_ready_event,
                # The total communication tail is already transitive over the
                # prior same-bank combine. A same-stream launch has FIFO; a
                # cross-stream launch waits that tail. Adding the older bank
                # event would only duplicate a CUDA wait node.
                bank_reuse_event=None,
                communication_predecessor_event=communication_predecessor_event,
                buffers=buffers,
                stream=stream,
                prepared_operation=prepared_operation,
            )
            if dispatch_ready_event is None:
                raise TypeError("backend must return a dispatch-ready device event")

            handle._dispatch_ready_event = dispatch_ready_event
            handle._dispatch_event_valid = True
            self._handles[bank] = handle
            # Replacing this only after a valid response keeps the preceding
            # same-bank inputs alive until the successor launch is recorded.
            self._communication_tail_event = dispatch_ready_event
            self._communication_tail_stream = stream
            self._next_step = next_step
            self._failed_previous_references[bank] = _EMPTY_REFERENCES
            return handle
        except BaseException as error:
            # The handle is deliberately not published until all post-enqueue
            # state is installed.  If anything after the backend call fails,
            # expire its private lease and irreversibly quarantine both operand
            # generations through the two preallocated reference lists.
            handle._expire_dispatch_event()
            self._enter_failed(error)
            raise

    def _dispatch_preallocated(
        self,
        frame: _PreparedSubmissionFrame,
        *,
        input_ready_event: Any | None,
    ) -> DispatchHandle:
        """Post one cold-materialized rolling dispatch without constructing state."""

        self._require_active()
        handle = frame.handle
        operation = handle._operation
        if operation.step != self._next_step:
            raise PipelineError("prepared dispatch frame is not the next operation")
        bank = frame.bank
        if self._handles[bank] is not None:
            raise PipelineError(f"bank {bank} still has an uncombined operation")
        if handle._dispatch_event_valid:
            raise PipelineError("prepared dispatch frame is invalid or already used")
        if handle._state != HandleState.DISPATCHED:
            raise PipelineError("prepared dispatch handle was already consumed")

        previous = self._completed_handles[bank]
        if previous is not None:
            previous._expire_completion_events()
            self._completed_handles[bank] = None
        prior_references = self._retained[bank]
        communication_predecessor_event = (
            self._communication_tail_event
            if self._communication_tail_stream is not handle._dispatch_stream
            else None
        )
        self._failed_previous_references[bank] = prior_references
        self._retained[bank] = frame.dispatch_references
        try:
            dispatch_ready_event = self.backend.enqueue_prepared_dispatch(
                frame.backend_launch,
                input_ready_event=input_ready_event,
                communication_predecessor_event=communication_predecessor_event,
            )
            if dispatch_ready_event is None:
                raise TypeError("backend must return a dispatch-ready device event")
            handle._dispatch_ready_event = dispatch_ready_event
            handle._dispatch_event_valid = True
            self._handles[bank] = handle
            self._communication_tail_event = dispatch_ready_event
            self._communication_tail_stream = handle._dispatch_stream
            self._next_step = frame.next_step
            self._failed_previous_references[bank] = _EMPTY_REFERENCES
            return handle
        except BaseException as error:
            handle._expire_dispatch_event()
            self._enter_failed(error)
            raise

    def combine_prepared(
        self,
        expert_output: Any,
        handle: DispatchHandle,
        prepared: PreparedOperation,
        *,
        stream: Any,
        expert_ready_event: Any | None = None,
    ) -> Any:
        """Enqueue a sealed zero-copy combine without tensor metadata reads."""

        self._require_active()
        self._require_stream(stream)
        self._validate_handle(handle)
        self._validate_prepared_operation(prepared)
        if (
            handle._activations is not prepared._activations
            or handle._topk_indices is not prepared._topk_indices
            or handle.num_tokens != prepared._num_tokens
        ):
            raise ValueError("prepared operation does not belong to this dispatch")
        if expert_output is not handle._buffers.combine_stage:
            raise ValueError(
                "prepared combine requires the backend-owned zero-copy output; "
                "use combine() for an external expert output"
            )
        self._require_expert_dependency(handle, stream, expert_ready_event)
        return self._combine_validated(
            expert_output,
            prepared._topk_weights,
            handle,
            output=prepared._output,
            stream=stream,
            expert_ready_event=expert_ready_event,
            prepared_operation=prepared._backend_token,
        )

    def combine(
        self,
        expert_output: Any,
        topk_weights: Any,
        handle: DispatchHandle,
        *,
        output: Any,
        stream: Any,
        expert_ready_event: Any | None = None,
    ) -> Any:
        """Enqueue reverse transfer/reduction into caller-owned ``output``.

        ``output`` must be a preallocated ``[live_tokens, hidden_size]`` device
        tensor. Requiring caller ownership removes an otherwise implicit
        two-bank lifetime limit: a later same-bank operation can never overwrite
        a result that another GPU stream is still consuming. ``expert_output``,
        ``topk_weights``, and the handle's route tensor remain immutable until
        the returned combine completion event.
        """

        self._require_active()
        self._require_stream(stream)
        self._validate_handle(handle)
        self._require_expert_dependency(handle, stream, expert_ready_event)
        weight_shape = _shape(topk_weights, "topk_weights")
        expected_weight_shape = (handle.num_tokens, self._top_k)
        if weight_shape != expected_weight_shape:
            raise ValueError(
                f"topk_weights must have shape {expected_weight_shape}, got "
                f"{weight_shape}"
            )
        output_shape = _shape(expert_output, "expert_output")
        expected_output_shape = (
            self._experts_per_rank,
            self._padded_expert_rows,
            self._hidden_size,
        )
        if output_shape != expected_output_shape:
            raise ValueError(
                f"expert_output must have shape {expected_output_shape}, got "
                f"{output_shape}"
            )
        combine_shape = _shape(output, "output")
        expected_combine_shape = (handle.num_tokens, self._hidden_size)
        if combine_shape != expected_combine_shape:
            raise ValueError(
                f"output must have shape {expected_combine_shape}, got "
                f"{combine_shape}"
            )
        device = _device(expert_output, "expert_output")
        if (
            _device(topk_weights, "topk_weights") != device
            or _device(handle._topk_indices, "topk_indices") != device
            or _device(output, "output") != device
        ):
            raise ValueError("combine tensors must share one device")
        if device != self._device:
            raise ValueError("combine tensors must reside on the pipeline device")

        return self._combine_validated(
            expert_output,
            topk_weights,
            handle,
            output=output,
            stream=stream,
            expert_ready_event=expert_ready_event,
            prepared_operation=None,
        )

    def _combine_validated(
        self,
        expert_output: Any,
        topk_weights: Any,
        handle: DispatchHandle,
        *,
        output: Any,
        stream: Any,
        expert_ready_event: Any | None,
        prepared_operation: Any | None,
    ) -> Any:
        """Post an already validated raw or prepared combine."""

        # Pessimistically quarantine all operands before the backend can post.
        # On failure this tuple is intentionally never rotated or released by
        # close(); process teardown owns recovery from uncertain device work.
        result_references = (
            handle._activations,
            handle._topk_indices,
            expert_output,
            topk_weights,
            output,
        )
        communication_predecessor_event = (
            self._communication_tail_event
            if self._communication_tail_stream is not stream
            else None
        )
        # This superset also retains the dispatch operands, so installing it in
        # the fixed-size list before enqueue makes every uncertain return path
        # allocation-free without losing the prior dispatch lifetime.
        self._retained[handle.bank] = result_references
        try:
            combine_events = self.backend.enqueue_combine(
                operation=handle.operation,
                source_incarnation=handle.source_incarnation,
                topology=self._topology,
                expert_output=expert_output,
                zero_copy=expert_output is handle._buffers.combine_stage,
                topk_indices=handle._topk_indices,
                topk_weights=topk_weights,
                combine_output=output,
                expert_ready_event=expert_ready_event,
                communication_predecessor_event=communication_predecessor_event,
                buffers=handle._buffers,
                stream=stream,
                prepared_operation=prepared_operation,
            )
            if not isinstance(combine_events, CombineEvents):
                raise TypeError("backend must return CombineEvents")

            handle._expire_dispatch_event()
            handle._state = HandleState.COMBINE_ENQUEUED
            handle._combine_completion_event = combine_events.output_ready
            handle._bank_reuse_event = combine_events.bank_reusable
            handle._completion_events_valid = True
            self._handles[handle.bank] = None
            self._completed_handles[handle.bank] = handle
            self._completion_events[handle.bank] = combine_events.bank_reusable
            self._communication_tail_event = combine_events.bank_reusable
            self._communication_tail_stream = stream
            return output
        except BaseException as error:
            # This boundary includes response validation and every publication
            # write: after a combine may have posted, retry is never safe.  The
            # handle is already external, so expire both leases explicitly in
            # case a partial publication cleared it from the pipeline lists.
            handle._expire_dispatch_event()
            handle._expire_completion_events()
            self._enter_failed(error)
            raise

    def _combine_preallocated(
        self,
        frame: _PreparedSubmissionFrame,
        *,
        expert_output: Any,
        expert_ready_event: Any | None,
    ) -> Any:
        """Post one cold-materialized rolling combine without retention allocation."""

        self._require_active()
        handle = frame.handle
        self._validate_preallocated_handle(frame)
        prepared = frame.prepared_operation
        if expert_output is not handle._buffers.combine_stage:
            raise ValueError(
                "prepared rolling combine requires the bank-owned output target"
            )
        self._require_expert_dependency(
            handle,
            handle._dispatch_stream,
            expert_ready_event,
        )
        communication_predecessor_event = (
            self._communication_tail_event
            if self._communication_tail_stream is not handle._dispatch_stream
            else None
        )
        bank = frame.bank
        self._retained[bank] = frame.combine_references
        try:
            combine_events = self.backend.enqueue_prepared_combine(
                frame.backend_launch,
                expert_ready_event=expert_ready_event,
                communication_predecessor_event=communication_predecessor_event,
            )
            if not isinstance(combine_events, CombineEvents):
                raise TypeError("backend must return CombineEvents")
            handle._expire_dispatch_event()
            handle._state = HandleState.COMBINE_ENQUEUED
            handle._combine_completion_event = combine_events.output_ready
            handle._bank_reuse_event = combine_events.bank_reusable
            handle._completion_events_valid = True
            self._handles[bank] = None
            self._completed_handles[bank] = handle
            self._completion_events[bank] = combine_events.bank_reusable
            self._communication_tail_event = combine_events.bank_reusable
            self._communication_tail_stream = handle._dispatch_stream
            return prepared._output
        except BaseException as error:
            handle._expire_dispatch_event()
            handle._expire_completion_events()
            self._enter_failed(error)
            raise

    def get_combine_buffer(self, handle: DispatchHandle) -> Any:
        """Return the registered output target owned by ``handle``'s bank.

        This metadata-only call is intended before launching the expert GEMM.
        Passing the returned tensor back to :meth:`combine` selects the
        zero-copy outgoing path.
        """

        self._require_active()
        self._validate_handle(handle)
        return handle._buffers.combine_stage

    def stage_generation(self, next_topology: StableSparseTopology) -> StagedGeneration:
        """Prepare peer metadata/views without activating them."""

        self._require_active()
        if self._staged is not None:
            raise PipelineError("a membership generation is already staged")
        self._validate_next_topology(next_topology)
        # Allocate the public control object before the backend may retain any
        # staged registrations or peer views.  Its token is filled only after
        # the backend's transactional call succeeds.
        staged = StagedGeneration(
            owner=self._owner,
            next_topology=next_topology,
            backend_token=None,
        )
        try:
            token = self.backend.stage_generation(
                current=self._topology, next_topology=next_topology
            )
            staged._backend_token = token
            self._staged = staged
            return staged
        except BaseException as error:
            # Even a nominally transactional backend cannot attest whether an
            # asynchronous BaseException arrived before or after it installed
            # candidate views. Never leave backend STAGED while the public
            # pipeline still appears ACTIVE with no staged handle.
            self._enter_failed(error)
            raise

    def begin_generation_drain(
        self, staged: StagedGeneration, *, stream: Any
    ) -> GenerationDrain:
        """Enqueue a cumulative drain after all old-generation combines."""

        self._require_active()
        self._require_stream(stream)
        self._validate_staged(staged, GenerationChangeState.STAGED)
        if self.outstanding_handles:
            raise PipelineError(
                "every dispatch handle must reach combine before generation drain"
            )
        try:
            drain = self.backend.enqueue_generation_drain(
                current=self._topology,
                staged_token=staged._backend_token,
                completion_events=tuple(
                    event for event in self._completion_events if event is not None
                ),
                stream=stream,
            )
            if not isinstance(drain, GenerationDrain):
                raise TypeError("backend returned an invalid GenerationDrain")
            for handle in self._completed_handles:
                if handle is not None:
                    handle._expire_completion_events()
            staged._state = GenerationChangeState.DRAIN_ENQUEUED
            staged._drain = drain
            self._state = PipelineState.DRAINING
            return drain
        except BaseException as error:
            # Drain is an asynchronous enqueue boundary.  Response validation
            # and publication therefore belong to the same fail-stop region.
            self._enter_failed(error)
            raise

    def commit_generation(
        self, staged: StagedGeneration, drain: GenerationDrain
    ) -> None:
        """Synchronously commit a graceful membership change.

        Unlike ``dispatch`` and ``combine``, this control-plane method may block:
        the backend must wait for the request-backed cumulative drain and global
        membership convergence before returning.
        """

        if self._state != PipelineState.DRAINING:
            self._raise_for_state("commit a membership generation")
        self._validate_staged(staged, GenerationChangeState.DRAIN_ENQUEUED)
        if not isinstance(drain, GenerationDrain):
            raise TypeError("drain must be a GenerationDrain")
        if staged._drain is not drain:
            raise PipelineError("drain does not belong to this staged generation")
        next_topology = staged.topology
        try:
            commit = self.backend.commit_generation(
                current=self._topology,
                next_topology=next_topology,
                staged_token=staged._backend_token,
                drain=drain,
            )
            if not isinstance(commit, GenerationCommit):
                raise TypeError("backend must return a GenerationCommit")
            if commit.membership_generation != next_topology.membership_generation:
                raise PipelineError("backend committed the wrong membership generation")
            missing_planes = GENERATION_REBASE_PLANES - commit.rebased_planes
            if missing_planes:
                raise PipelineError(
                    "backend activated a generation without rebasing planes "
                    f"{sorted(missing_planes)}"
                )

            # All containers are fixed at two banks and are reset in place so
            # no successful backend commit is followed by a Python allocation.
            self._topology = next_topology
            self._next_step = 0
            self._retained[0] = ()
            self._retained[1] = ()
            self._failed_previous_references[0] = ()
            self._failed_previous_references[1] = ()
            self._completion_events[0] = None
            self._completion_events[1] = None
            self._completed_handles[0] = None
            self._completed_handles[1] = None
            self._communication_tail_event = None
            self._communication_tail_stream = None
            staged._state = GenerationChangeState.COMMITTED
            self._staged = None
            self._state = PipelineState.ACTIVE
        except BaseException as error:
            # The backend may already have activated the new generation.  Any
            # validation or local publication failure is therefore terminal.
            self._enter_failed(error)
            raise

    def report_abrupt_peer_loss(self, peer_rank: int, detail: str) -> None:
        """Fail-stop; never reuse possibly outstanding requestless WQEs."""

        if self._state not in (PipelineState.ACTIVE, PipelineState.DRAINING):
            self._raise_for_state("report an abrupt peer loss")
        if isinstance(peer_rank, bool) or not isinstance(peer_rank, int):
            raise TypeError("peer_rank must be an integer")
        if not 0 <= peer_rank < self._topology.max_ranks:
            raise ValueError(f"peer_rank must be in [0, {self._topology.max_ranks})")
        if peer_rank == self.rank:
            raise ValueError("peer_rank must identify a remote rank")
        if not self._topology.is_active(peer_rank):
            raise ValueError(f"peer rank {peer_rank} is already masked")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("detail must be a non-empty string")
        error = AbruptPeerLoss(
            peer_rank=peer_rank,
            membership_generation=self._topology.membership_generation,
            peer_incarnation=self._topology.incarnation(peer_rank),
            detail=detail.strip(),
        )
        self._enter_failed(error)
        raise error

    def fail_stop(self, error: BaseException) -> None:
        """Stop after a local/expert failure whose device state may be uncertain."""

        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        if self._state is PipelineState.FAILED or self._state is PipelineState.CLOSED:
            self._raise_for_state("fail-stop the pipeline")
        self._enter_failed(error)

    def close(self) -> None:
        """Close resources through the backend's synchronous control path.

        With no open handle, a healthy backend cumulatively drains its latest
        bank events before releasing views/registrations. A failed pipeline
        cannot be closed: its quarantined tensors and registered arena must stay
        alive until process-level CUDA-context teardown.
        """

        if self._state == PipelineState.CLOSED:
            return
        if self._state == PipelineState.FAILED:
            raise PipelineFailed(
                "cannot close a fail-stopped communicator: uncertain device "
                "work keeps its tensors and registrations quarantined until "
                "process teardown"
            ) from self._failure
        if self._state == PipelineState.ACTIVE and self.outstanding_handles:
            raise PipelineError("cannot close with outstanding dispatch handles")
        if self._state == PipelineState.DRAINING:
            raise PipelineError("finish or fail-stop the generation drain before close")
        self._state = PipelineState.CLOSING
        try:
            # Revoke every external completion lease before backend teardown;
            # a completed close may release or recycle the native events.
            first_completed = self._completed_handles[0]
            if first_completed is not None:
                first_completed._expire_completion_events()
            second_completed = self._completed_handles[1]
            if second_completed is not None:
                second_completed._expire_completion_events()
            self.backend.close()
            if self._staged is not None:
                self._staged._state = GenerationChangeState.CANCELLED
                self._staged = None
            self._completed_handles[0] = None
            self._completed_handles[1] = None
            self._completion_events[0] = None
            self._completion_events[1] = None
            self._communication_tail_event = None
            self._communication_tail_stream = None
            self._retained[0] = ()
            self._retained[1] = ()
            self._failed_previous_references[0] = ()
            self._failed_previous_references[1] = ()
            # Drop the pipeline's strong references to registered arena views
            # after the backend completed release/deregistration. Stale handles
            # remain invalid leases but may own tensor views until released.
            self._buffers = ()
            self._state = PipelineState.CLOSED
        except BaseException as error:
            self._enter_failed(error)
            raise

    def _validate_handle(self, handle: DispatchHandle) -> None:
        if not isinstance(handle, DispatchHandle):
            raise TypeError("handle must be a DispatchHandle")
        if handle._owner is not self._owner:
            raise PipelineError("dispatch handle belongs to another pipeline")
        if handle.state != HandleState.DISPATCHED:
            raise PipelineError("dispatch handle was already consumed by combine")
        if handle.membership_generation != self._topology.membership_generation:
            raise PipelineError("dispatch handle belongs to a stale generation")
        if handle.source_incarnation != self._topology.incarnation(self.rank):
            raise PipelineError("dispatch handle belongs to a stale rank incarnation")
        if self._handles[handle.bank] is not handle:
            raise PipelineError("dispatch handle does not own its physical bank")

    def _validate_preallocated_handle(self, frame: _PreparedSubmissionFrame) -> None:
        """Validate a cold frame without deriving its bank on the hot path."""

        handle = frame.handle
        if handle._state != HandleState.DISPATCHED:
            raise PipelineError("prepared dispatch handle was already consumed")
        if self._handles[frame.bank] is not handle:
            raise PipelineError(
                "prepared dispatch handle does not own its physical bank"
            )

    def _validate_prepared_operation(self, prepared: PreparedOperation) -> None:
        if not isinstance(prepared, PreparedOperation):
            raise TypeError("prepared must be a PreparedOperation")
        if prepared._owner is not self._owner:
            raise PipelineError("prepared operation belongs to another pipeline")

    def _require_local_active(self) -> None:
        if not self._topology.is_active(self.rank):
            raise PipelineError(
                f"rank {self.rank} is a standby in generation "
                f"{self._topology.membership_generation}"
            )

    @staticmethod
    def _require_expert_dependency(
        handle: DispatchHandle,
        stream: Any,
        expert_ready_event: Any | None,
    ) -> None:
        if expert_ready_event is None and stream is not handle._dispatch_stream:
            raise PipelineError(
                "cross-stream combine requires an expert_ready_event; same-stream "
                "submission may omit it"
            )

    @staticmethod
    def _validate_preallocated_buffers(
        layout: PipelineLLArenaLayout, buffers: PipelineBankBuffers
    ) -> Any:
        experts = layout.experts_per_rank
        padded_rows = layout.max_ranks * layout.num_tokens
        expected_shapes = {
            "expert_input": (experts, padded_rows, layout.hidden_size),
            "expert_counts": (experts,),
            "source_info": (experts, padded_rows, 4),
            "layout_ranges": (experts, layout.max_ranks),
            "combine_stage": (experts, padded_rows, layout.hidden_size),
        }
        device: Any = None
        for name, expected in expected_shapes.items():
            tensor = getattr(buffers, name)
            actual = _shape(tensor, f"bank {buffers.bank} {name}")
            if actual != expected:
                raise ValueError(
                    f"bank {buffers.bank} {name} must have shape {expected}, "
                    f"got {actual}"
                )
            tensor_device = _device(tensor, f"bank {buffers.bank} {name}")
            if device is None:
                device = tensor_device
            elif tensor_device != device:
                raise ValueError(
                    f"all tensors in bank {buffers.bank} must share one device"
                )
        return device

    def _validate_next_topology(self, next_topology: StableSparseTopology) -> None:
        if not isinstance(next_topology, StableSparseTopology):
            raise TypeError("next_topology must be a StableSparseTopology")
        current = self._topology
        if next_topology.max_ranks != current.max_ranks:
            raise ValueError("membership change cannot resize fixed rank capacity")
        if next_topology.experts_per_rank != current.experts_per_rank:
            raise ValueError("membership change cannot renumber fixed experts")
        if next_topology.membership_generation <= current.membership_generation:
            raise ValueError("membership generation must advance")
        for rank, (old, new) in enumerate(
            zip(current.rank_incarnations, next_topology.rank_incarnations)
        ):
            if new < old:
                raise ValueError(
                    f"rank {rank} incarnation moved backwards from {old} to {new}"
                )

    def _validate_staged(
        self, staged: StagedGeneration, expected: GenerationChangeState
    ) -> None:
        if not isinstance(staged, StagedGeneration):
            raise TypeError("staged must be a StagedGeneration")
        if staged._owner is not self._owner or self._staged is not staged:
            raise PipelineError("staged generation belongs to another pipeline")
        if staged.state != expected:
            raise PipelineError(
                f"staged generation is {staged.state.value}, expected "
                f"{expected.value}"
            )

    def _require_active(self) -> None:
        if self._state != PipelineState.ACTIVE:
            self._raise_for_state("submit pipeline work")

    @staticmethod
    def _require_stream(stream: Any) -> None:
        if stream is None:
            raise ValueError("stream must be an explicit device stream")

    def _raise_for_state(self, action: str) -> None:
        if self._state == PipelineState.FAILED:
            raise PipelineFailed(
                f"cannot {action}: communicator is fail-stopped"
            ) from self._failure
        raise PipelineError(f"cannot {action}: pipeline is {self._state.value}")

    def _enter_failed(self, error: BaseException) -> None:
        self._failure = error
        self._state = PipelineState.FAILED
        # Invoke the backend terminator before even Python iterator creation.
        # A test backend may return; production mapped execution exits here.
        try:
            self.backend.fail_stop(error)
        except BaseException:  # noqa: B036 - backend failure cannot replace cause.
            # The original transport error is authoritative.  A backend must
            # retain uncertain registrations for external allocation cleanup.
            pass
        # Fixed-index invalidation remains allocation-free if a test terminator
        # returns. The combine exception path separately expires its possibly
        # half-published external handle before entering this method.
        first_active = self._handles[0]
        if first_active is not None:
            first_active._expire_dispatch_event()
            first_active._expire_completion_events()
        second_active = self._handles[1]
        if second_active is not None:
            second_active._expire_dispatch_event()
            second_active._expire_completion_events()
        first_completed = self._completed_handles[0]
        if first_completed is not None:
            first_completed._expire_dispatch_event()
            first_completed._expire_completion_events()
        second_completed = self._completed_handles[1]
        if second_completed is not None:
            second_completed._expire_dispatch_event()
            second_completed._expire_completion_events()


__all__ = [
    "AbruptPeerLoss",
    "CombineEvents",
    "DispatchHandle",
    "GenerationChangeState",
    "GenerationCommit",
    "GenerationDrain",
    "GENERATION_REBASE_PLANES",
    "HandleState",
    "MAX_PREPARED_SUBMISSION_OPERATIONS",
    "MoEPipeline",
    "PipelineBackend",
    "PipelineBankBuffers",
    "PipelineError",
    "PipelineFailed",
    "PipelineState",
    "PreparedOperation",
    "StagedGeneration",
    "SubmissionPreflight",
]
