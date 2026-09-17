#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production-shaped split-phase elastic MoE API example.

This file shows the application boundary that the fused
``elastic_moe_ll.py`` benchmark intentionally does not expose:

1. dispatch live ``x`` and ``topk_idx`` device tensors;
2. run the application's real grouped expert kernels on expert-major buffers;
3. combine the resulting expert output with live gate weights; and
4. overlap preparation of a later sparse membership, then drain and commit it
   outside the steady-state loop.

``enqueue_moe_layer`` performs no CPU tensor read and no host synchronization.
Its backend is expected to launch the CuTe/NIXL dispatch and combine kernels on
communication streams and connect an optional expert stream with GPU events.
The expert executor is deliberately application-owned;
embedding a toy BF16 transform here would hide the real compute/communication
handoff and make performance numbers misleading.

The command-line mode prints and validates a fixed-capacity membership plan on
any machine.  It is an executable control-plane reference, not a transport
benchmark.  A concrete device run requires a backend implementing
``moe.pipeline.PipelineBackend`` and registered NIXL device views.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

try:
    from .moe.ll_protocol import StableSparseTopology
    from .moe.pipeline import (
        MAX_PREPARED_SUBMISSION_OPERATIONS,
        DispatchHandle,
        GenerationChangeState,
        MoEPipeline,
        PipelineError,
        PipelineState,
        PreparedOperation,
        StagedGeneration,
        SubmissionPreflight,
    )
except ImportError:  # Direct ``python examples/python/cute/...`` execution.
    from moe.ll_protocol import StableSparseTopology  # type: ignore[no-redef]
    from moe.pipeline import (  # type: ignore[no-redef]
        MAX_PREPARED_SUBMISSION_OPERATIONS,
        DispatchHandle,
        GenerationChangeState,
        MoEPipeline,
        PipelineError,
        PipelineState,
        PreparedOperation,
        StagedGeneration,
        SubmissionPreflight,
    )


class ExpertExecutor(Protocol):
    """Application adapter for grouped GEMM or another real expert kernel."""

    def enqueue(
        self,
        *,
        expert_input: Any,
        expert_counts: Any,
        source_info: Any,
        layout_ranges: Any,
        output: Any,
        handle: DispatchHandle,
        stream: Any,
    ) -> "ExpertSubmission":
        """Write padded expert output and return its device-ready event."""


class PreparedExpertExecutor(ExpertExecutor, Protocol):
    """Optional allocation-audited adapter used by prepared rolling plans.

    The method must enqueue into the exact supplied ``output`` and return its
    device-ready event without constructing a per-operation result wrapper.
    Production adapters should cold-cache grouped-GEMM descriptors, launch
    configuration, and a bounded event pool. Python/native allocations inside
    an application adapter are outside the pipeline's control and must be
    audited independently.
    """

    def enqueue_prepared(
        self,
        *,
        expert_input: Any,
        expert_counts: Any,
        source_info: Any,
        layout_ranges: Any,
        output: Any,
        handle: DispatchHandle,
        stream: Any,
    ) -> Any | None:
        """Post one cold-bound expert launch and return its ready event."""


@dataclass(frozen=True, slots=True)
class ExpertSubmission:
    """Expert output plus its cross-stream device dependency."""

    output: Any
    ready_event: Any | None


@dataclass(frozen=True, slots=True)
class LayerSubmission:
    """Asynchronously submitted layer result and bounded event-wait leases.

    Raw backend events are deliberately not exposed: a production backend may
    use a bounded event ring.  Enqueue downstream waits through these methods
    before the same physical bank is submitted again.  CUDA snapshots an
    event's current record when the wait is enqueued, so later backend reuse
    cannot retarget that already-queued dependency.
    """

    output: Any
    handle: DispatchHandle

    def enqueue_output_wait(self, stream: Any) -> None:
        """Wait on weighted output completion without exposing the raw event."""

        self.handle.enqueue_output_wait(stream)

    def enqueue_bank_reuse_wait(self, stream: Any) -> None:
        """Wait until every peer credit has made the bank reusable."""

        self.handle.enqueue_bank_reuse_wait(stream)


class _ExpertSubmissionSlot:
    """Cold-bound expert operands, wait route, and one mutable result event."""

    __slots__ = (
        "dispatch_wait",
        "expert_counts",
        "expert_input",
        "expert_stream",
        "handle",
        "input_ready_event",
        "layout_ranges",
        "output",
        "ready_event",
        "source_info",
    )

    def __init__(
        self,
        *,
        handle: DispatchHandle,
        input_ready_event: Any | None,
        communication_stream: Any,
        expert_stream: Any,
    ) -> None:
        buffers = handle._buffers
        self.handle = handle
        self.expert_input = buffers.expert_input
        self.expert_counts = buffers.expert_counts
        self.source_info = buffers.source_info
        self.layout_ranges = buffers.layout_ranges
        self.output = buffers.combine_stage
        self.ready_event: Any | None = None
        self.input_ready_event = input_ready_event
        self.expert_stream = expert_stream
        self.dispatch_wait = (
            None if expert_stream is communication_stream else expert_stream.wait_event
        )


class _FastRollingLink:
    """Cold-linked one-combine/one-dispatch steady-state action."""

    __slots__ = (
        "combine_frame",
        "combine_slot",
        "dispatch_frame",
        "dispatch_slot",
        "next_link",
    )

    def __init__(
        self,
        *,
        combine_frame: Any,
        combine_slot: _ExpertSubmissionSlot,
        dispatch_frame: Any,
        dispatch_slot: _ExpertSubmissionSlot,
        next_link: "_FastRollingLink | None",
    ) -> None:
        self.combine_frame = combine_frame
        self.combine_slot = combine_slot
        self.dispatch_frame = dispatch_frame
        self.dispatch_slot = dispatch_slot
        self.next_link = next_link


class _FastRollingPlan:
    """Cold prologue, linked steady state, and epilogue for one submission."""

    __slots__ = (
        "final_frame",
        "final_slot",
        "first_frame",
        "first_slot",
        "penultimate_frame",
        "penultimate_slot",
        "second_frame",
        "second_slot",
        "steady_head",
    )

    def __init__(
        self,
        *,
        first_frame: Any,
        first_slot: _ExpertSubmissionSlot,
        second_frame: Any,
        second_slot: _ExpertSubmissionSlot,
        steady_head: _FastRollingLink | None,
        penultimate_frame: Any,
        penultimate_slot: _ExpertSubmissionSlot,
        final_frame: Any,
        final_slot: _ExpertSubmissionSlot,
    ) -> None:
        self.first_frame = first_frame
        self.first_slot = first_slot
        self.second_frame = second_frame
        self.second_slot = second_slot
        self.steady_head = steady_head
        self.penultimate_frame = penultimate_frame
        self.penultimate_slot = penultimate_slot
        self.final_frame = final_frame
        self.final_slot = final_slot


@dataclass(frozen=True, slots=True)
class LayerInputs:
    """Live device tensors, caller-owned output, and router-ready event."""

    activations: Any
    topk_indices: Any
    topk_weights: Any
    output: Any
    input_ready_event: Any | None = None


def _contiguous_byte_span(tensor: Any, name: str) -> tuple[int, int] | None:
    """Return an exact logical byte span using synchronization-free metadata."""

    methods = {
        method: getattr(tensor, method, None)
        for method in ("data_ptr", "numel", "element_size", "is_contiguous")
    }
    missing = [method for method, value in methods.items() if not callable(value)]
    if missing:
        raise TypeError(
            f"{name} must expose synchronization-free tensor storage metadata: "
            + ", ".join(f"{method}()" for method in missing)
        )
    if not bool(methods["is_contiguous"]()):
        raise ValueError(f"{name} must be contiguous")
    elements = methods["numel"]()
    width = methods["element_size"]()
    if (
        isinstance(elements, bool)
        or not isinstance(elements, int)
        or elements < 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
    ):
        raise ValueError(f"{name} returned invalid storage metadata")
    size = elements * width
    if size == 0:
        return None
    begin = methods["data_ptr"]()
    if isinstance(begin, bool) or not isinstance(begin, int) or begin <= 0:
        raise ValueError(f"{name} returned an invalid data pointer")
    return begin, begin + size


def _spans_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool:
    return (
        left is not None
        and right is not None
        and left[0] < right[1]
        and right[0] < left[1]
    )


@dataclass(frozen=True, slots=True)
class PreparedTwoBankWindow:
    """Alias-safe two-operation plan whose storage proof is computed once.

    Construct and then pass plans through :func:`bind_two_bank_window` before a
    measured loop. Reuse the resulting bound schedule while tensor storage
    remains immutable. This keeps pointer/size introspection out of command
    submission while proving, before dispatch zero, that neither concurrently
    written output overlaps the other operation or any router input consumed by
    either bank.
    """

    first: LayerInputs
    second: LayerInputs
    _storage_spans: tuple[tuple[int, int] | None, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.first, LayerInputs) or not isinstance(
            self.second, LayerInputs
        ):
            raise TypeError("a two-bank window requires exactly two LayerInputs")
        tensors = {
            "first.activations": self.first.activations,
            "first.topk_indices": self.first.topk_indices,
            "first.topk_weights": self.first.topk_weights,
            "first.output": self.first.output,
            "second.activations": self.second.activations,
            "second.topk_indices": self.second.topk_indices,
            "second.topk_weights": self.second.topk_weights,
            "second.output": self.second.output,
        }
        spans = {
            name: _contiguous_byte_span(tensor, name)
            for name, tensor in tensors.items()
        }
        object.__setattr__(self, "_storage_spans", tuple(spans.values()))
        read_names = tuple(name for name in tensors if not name.endswith(".output"))
        output_names = ("first.output", "second.output")
        checked: set[tuple[str, str]] = set()
        for output_name in output_names:
            for other_name in read_names + output_names:
                if output_name == other_name:
                    continue
                pair = tuple(sorted((output_name, other_name)))
                if pair in checked:
                    continue
                checked.add(pair)
                if _spans_overlap(spans[output_name], spans[other_name]):
                    raise ValueError(
                        f"{output_name} and {other_name} storage must not overlap"
                    )


def prepare_two_bank_window(
    first: LayerInputs, second: LayerInputs
) -> PreparedTwoBankWindow:
    """Preflight cross-operation storage once, before any device work posts."""

    return PreparedTwoBankWindow(first, second)


@dataclass(frozen=True, slots=True)
class PreparedRollingSchedule:
    """Preflighted finite cycle for the steady-state two-bank scheduler.

    ``windows`` contains the unique tensor bindings in one even-length cycle;
    ``repeat_count`` reuses that cycle without materializing O(operation_count)
    Python objects. Outputs may intentionally alias only at the same bank
    parity, after the prior combine on that bank. Every output remains disjoint
    from every router input in the cycle, and opposite-bank outputs are
    disjoint. Reusing a same-parity output additionally means it has no
    asynchronous downstream consumer, or that the application has ordered that
    consumer's completion before the later combine overwrites it; otherwise use
    unique caller outputs. Each window seals tensor storage metadata; the cold
    :func:`bind_rolling_schedule` step revalidates it once, after which
    :func:`enqueue_rolling_moe_layers` performs no tensor metadata query.
    """

    windows: tuple[PreparedTwoBankWindow, ...]
    repeat_count: int = 1
    _inputs: tuple[LayerInputs, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("a rolling schedule requires at least one window")
        if any(
            not isinstance(window, PreparedTwoBankWindow) for window in self.windows
        ):
            raise TypeError("rolling windows must be PreparedTwoBankWindow objects")
        if (
            isinstance(self.repeat_count, bool)
            or not isinstance(self.repeat_count, int)
            or self.repeat_count <= 0
        ):
            raise ValueError("repeat_count must be a positive integer")
        inputs = tuple(
            inputs
            for window in self.windows
            for inputs in (window.first, window.second)
        )
        object.__setattr__(self, "_inputs", inputs)
        reads: list[tuple[int, int, str]] = []
        outputs: tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]] = (
            [],
            [],
        )
        # PreparedTwoBankWindow already proved and sealed these exact spans.
        # Reusing its proof avoids a second round of tensor metadata queries.
        for window_index, window in enumerate(self.windows):
            for side in range(2):
                operation = 2 * window_index + side
                offset = 4 * side
                for field_index, name in enumerate(
                    ("activations", "topk_indices", "topk_weights")
                ):
                    span = window._storage_spans[offset + field_index]
                    if span is not None:
                        reads.append(
                            (span[0], span[1], f"operation {operation} {name}")
                        )
                output_span = window._storage_spans[offset + 3]
                if output_span is not None:
                    outputs[operation & 1].append(
                        (
                            output_span[0],
                            output_span[1],
                            f"operation {operation} output",
                        )
                    )

        def first_cross_overlap(
            left: Sequence[tuple[int, int, str]],
            right: Sequence[tuple[int, int, str]],
        ) -> tuple[str, str] | None:
            left_sorted = sorted(left)
            right_sorted = sorted(right)
            left_index = right_index = 0
            while left_index < len(left_sorted) and right_index < len(right_sorted):
                left_begin, left_end, left_name = left_sorted[left_index]
                right_begin, right_end, right_name = right_sorted[right_index]
                if left_begin < right_end and right_begin < left_end:
                    return left_name, right_name
                if left_end <= right_end:
                    left_index += 1
                else:
                    right_index += 1
            return None

        output_spans = outputs[0] + outputs[1]
        overlap = first_cross_overlap(output_spans, reads)
        if overlap is not None:
            raise ValueError(f"{overlap[0]} overlaps {overlap[1]}")
        overlap = first_cross_overlap(outputs[0], outputs[1])
        if overlap is not None:
            raise ValueError(f"opposite-bank {overlap[0]} and {overlap[1]} overlap")

    @property
    def operation_count(self) -> int:
        return len(self._inputs) * self.repeat_count

    @property
    def unique_inputs(self) -> tuple[LayerInputs, ...]:
        """Return the immutable tensor bindings in one finite cycle."""

        return self._inputs

    def _input_for_operation(self, operation: int) -> LayerInputs:
        return self._inputs[operation % len(self._inputs)]


@dataclass(frozen=True, slots=True)
class BoundRollingSchedule:
    """Finite schedule whose exact operands are bound through one pipeline."""

    storage_schedule: PreparedRollingSchedule
    operations: tuple[PreparedOperation, ...]
    _pipeline_owner: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.storage_schedule, PreparedRollingSchedule):
            raise TypeError("storage_schedule must be a PreparedRollingSchedule")
        if len(self.operations) != len(self.storage_schedule.unique_inputs):
            raise ValueError("one prepared operation is required per unique input")
        if any(not isinstance(item, PreparedOperation) for item in self.operations):
            raise TypeError("operations must contain PreparedOperation objects")
        owner = self.operations[0]._owner
        for inputs, operation in zip(
            self.storage_schedule.unique_inputs, self.operations
        ):
            if operation._owner is not owner:
                raise ValueError("every operation must belong to one pipeline")
            if (
                operation._activations is not inputs.activations
                or operation._topk_indices is not inputs.topk_indices
                or operation._topk_weights is not inputs.topk_weights
                or operation._output is not inputs.output
            ):
                raise ValueError("prepared operation does not match schedule storage")
        object.__setattr__(self, "_pipeline_owner", owner)

    @property
    def operation_count(self) -> int:
        return self.storage_schedule.operation_count

    def _input_for_operation(self, operation: int) -> LayerInputs:
        return self.storage_schedule._input_for_operation(operation)

    def _binding_for_operation(self, operation: int) -> PreparedOperation:
        return self.operations[operation % len(self.operations)]


class RollingSubmissionPreflight:
    """Opaque one-shot allocation-free rolling submission proof.

    Create this object with :func:`preflight_rolling_moe_layers`. The cold plan
    owns one immutable epoch, one unique dispatch handle, and one private expert
    result slot per admitted operation, plus exactly two terminal completion
    leases selected by physical bank. The plan is capped at 65,536 operations;
    longer services submit multiple finite plans so a publicly returned handle
    is never mutated or reused.

    The backend capture observation is the final action in construction. Until
    :func:`enqueue_rolling_moe_layers` consumes this proof, the caller must keep
    exclusive single-host-submitter ownership and issue no capture or CUDA
    operation on any admitted stream.
    """

    __slots__ = (
        "_executor",
        "_expert_slots",
        "_fast_enqueue",
        "_fast_plan",
        "_operation_count",
        "_operation_events",
        "_pending_experts",
        "_pipeline_owner",
        "_pipeline_preflight",
        "_schedule",
        "_terminal_submissions",
    )

    def __init__(
        self,
        *,
        pipeline_owner: object,
        pipeline_preflight: SubmissionPreflight,
        schedule: BoundRollingSchedule,
        executor: ExpertExecutor,
        operation_count: int,
        operation_events: Sequence[tuple[Any, Any]] | None,
        expert_slots: tuple[_ExpertSubmissionSlot, ...],
        fast_enqueue: Any | None,
        fast_plan: _FastRollingPlan,
        terminal_submissions: tuple[LayerSubmission, LayerSubmission],
    ) -> None:
        self._pipeline_owner = pipeline_owner
        self._pipeline_preflight = pipeline_preflight
        self._schedule = schedule
        self._executor = executor
        self._operation_count = operation_count
        self._operation_events = operation_events
        self._expert_slots = expert_slots
        self._fast_enqueue = fast_enqueue
        self._fast_plan = fast_plan
        self._terminal_submissions = terminal_submissions
        self._pending_experts: list[ExpertSubmission | None] = [None, None]

    def __repr__(self) -> str:
        return "<RollingSubmissionPreflight opaque>"


def prepare_rolling_schedule(
    windows: Sequence[PreparedTwoBankWindow],
    *,
    repeat_count: int = 1,
) -> PreparedRollingSchedule:
    """Seal one finite storage cycle outside the hot path."""

    return PreparedRollingSchedule(tuple(windows), repeat_count=repeat_count)


def bind_rolling_schedule(
    pipeline: MoEPipeline,
    schedule: PreparedRollingSchedule,
) -> BoundRollingSchedule:
    """Cold-revalidate storage and bind every unique operand descriptor."""

    if not isinstance(pipeline, MoEPipeline):
        raise TypeError("pipeline must be a MoEPipeline")
    if not isinstance(schedule, PreparedRollingSchedule):
        raise TypeError("schedule must be a PreparedRollingSchedule")
    if pipeline.outstanding_handles:
        raise ValueError("schedule binding requires a complete two-bank boundary")
    # A prepared schedule's cross-operation alias proof is valid only for its
    # exact sealed spans. PyTorch ``set_``/``resize_`` can change those spans
    # without changing tensor object identity, so repeat the cold metadata
    # proof before creating even the first backend descriptor/token.
    for window_index, window in enumerate(schedule.windows):
        tensors = (
            ("first.activations", window.first.activations),
            ("first.topk_indices", window.first.topk_indices),
            ("first.topk_weights", window.first.topk_weights),
            ("first.output", window.first.output),
            ("second.activations", window.second.activations),
            ("second.topk_indices", window.second.topk_indices),
            ("second.topk_weights", window.second.topk_weights),
            ("second.output", window.second.output),
        )
        current_spans = tuple(
            _contiguous_byte_span(tensor, f"windows[{window_index}].{name}")
            for name, tensor in tensors
        )
        if current_spans != window._storage_spans:
            raise ValueError(
                "prepared rolling schedule storage changed; rebuild the schedule "
                "before binding"
            )
    return BoundRollingSchedule(
        storage_schedule=schedule,
        operations=tuple(
            pipeline.prepare_operation(
                inputs.activations,
                inputs.topk_indices,
                inputs.topk_weights,
                inputs.output,
            )
            for inputs in schedule.unique_inputs
        ),
    )


def bind_two_bank_window(
    pipeline: MoEPipeline,
    window: PreparedTwoBankWindow,
) -> BoundRollingSchedule:
    """Cold-bind one alias-checked two-operation window for submission.

    The explicit migration path from raw inputs is::

        planned = prepare_two_bank_window(first, second)
        bound = bind_two_bank_window(pipeline, planned)
        enqueue_two_bank_window(pipeline, executor, bound, ...)
    """

    if not isinstance(window, PreparedTwoBankWindow):
        raise TypeError("window must be a PreparedTwoBankWindow")
    return bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule((window,)),
    )


def _resolve_rolling_count(
    schedule: BoundRollingSchedule,
    operation_count: int | None,
) -> int:
    count = schedule.operation_count if operation_count is None else operation_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("operation_count must be a positive integer")
    if count % 2:
        raise ValueError("operation_count must end on a complete two-bank window")
    if count > schedule.operation_count:
        raise ValueError("operation_count exceeds the prepared schedule")
    if count > MAX_PREPARED_SUBMISSION_OPERATIONS:
        raise ValueError(
            "rolling operation_count exceeds the cold-plan capacity "
            f"{MAX_PREPARED_SUBMISSION_OPERATIONS}"
        )
    return count


def _validate_operation_events(
    operation_events: Sequence[tuple[Any, Any]] | None,
    count: int,
) -> None:
    if operation_events is not None and len(operation_events) < count:
        raise ValueError("operation_events does not cover every submitted operation")
    if operation_events is not None:
        for operation in range(count):
            pair = operation_events[operation]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    f"operation_events[{operation}] must be a two-event tuple"
                )
            if any(not callable(getattr(event, "record", None)) for event in pair):
                raise TypeError(
                    f"operation_events[{operation}] events must expose record(stream)"
                )


def _build_fast_rolling_plan(
    frames: Sequence[Any],
    slots: Sequence[_ExpertSubmissionSlot],
) -> _FastRollingPlan:
    """Cold-link the steady traversal so posting performs no cursor math."""

    count = len(frames)
    steady_head: _FastRollingLink | None = None
    for dispatch_index in range(count - 1, 1, -1):
        combine_index = dispatch_index - 2
        steady_head = _FastRollingLink(
            combine_frame=frames[combine_index],
            combine_slot=slots[combine_index],
            dispatch_frame=frames[dispatch_index],
            dispatch_slot=slots[dispatch_index],
            next_link=steady_head,
        )
    return _FastRollingPlan(
        first_frame=frames[0],
        first_slot=slots[0],
        second_frame=frames[1],
        second_slot=slots[1],
        steady_head=steady_head,
        penultimate_frame=frames[-2],
        penultimate_slot=slots[-2],
        final_frame=frames[-1],
        final_slot=slots[-1],
    )


def preflight_rolling_moe_layers(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    schedule: BoundRollingSchedule,
    *,
    operation_count: int | None = None,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
    operation_events: Sequence[tuple[Any, Any]] | None = None,
) -> RollingSubmissionPreflight:
    """Cold-materialize one exact allocation-free rolling submission.

    Allocation is deliberately O(operation_count) and bounded at 65,536
    operations. Every operation receives a unique immutable epoch and handle;
    its backend scalar/stream/event launch record, expert operands, and
    cross-stream wait callable are bound here too. The final two
    :class:`LayerSubmission` leases are also constructed here. No separate
    backend submission-context preparation is required. The mapped backend
    capture/range query is the final action before return, so callers must
    consume the proof without touching an admitted stream.

    An executor may implement ``enqueue_prepared(**kwargs) -> ready_event`` to
    select the allocation-free adapter path. The library then uses private cold
    result slots and constructs no :class:`ExpertSubmission` while posting.
    Allocations performed inside an arbitrary application executor remain
    outside this library's control and must be audited by that adapter.
    """

    if not isinstance(pipeline, MoEPipeline):
        raise TypeError("pipeline must be a MoEPipeline")
    if not isinstance(schedule, BoundRollingSchedule):
        raise TypeError(
            "schedule must be a BoundRollingSchedule; call bind_rolling_schedule() "
            "during cold preparation"
        )
    if schedule._pipeline_owner is not pipeline._owner:
        raise ValueError("bound schedule belongs to another pipeline")
    count = _resolve_rolling_count(schedule, operation_count)
    if len(communication_streams) != 2 or len(expert_streams) != 2:
        raise ValueError("rolling submission requires exactly two streams of each kind")
    all_streams = communication_streams + expert_streams
    if any(stream is None for stream in all_streams):
        raise ValueError("every communication and expert stream must be explicit")
    if any(not callable(getattr(stream, "wait_event", None)) for stream in all_streams):
        raise TypeError("every rolling stream must expose wait_event(event)")
    if not callable(getattr(expert_executor, "enqueue", None)):
        raise TypeError("expert_executor must provide enqueue()")
    fast_enqueue = getattr(expert_executor, "enqueue_prepared", None)
    if fast_enqueue is not None and not callable(fast_enqueue):
        raise TypeError("expert_executor.enqueue_prepared must be callable")
    if pipeline.outstanding_handles:
        raise ValueError(
            "rolling submission must start at a complete two-bank boundary"
        )
    _validate_operation_events(operation_events, count)

    first_step = pipeline.next_operation_step
    prepared_operations = tuple(
        schedule._binding_for_operation(operation) for operation in range(count)
    )
    pipeline._validate_submission_boundary(
        streams=all_streams,
        first_step=first_step,
        step_count=count,
    )
    pipeline._require_local_active()
    backend_context = pipeline.backend.prepare_submission_context(
        streams=all_streams,
        first_step=first_step,
        step_count=count,
    )
    if backend_context is None:
        raise TypeError(
            "backend prepare_submission_context must return an opaque context"
        )
    frames = pipeline._materialize_bound_submission_frames(
        streams=all_streams,
        first_step=first_step,
        prepared_operations=prepared_operations,
        dispatch_streams=communication_streams,
        expert_streams=expert_streams,
        backend_context=backend_context,
    )
    expert_slots = tuple(
        _ExpertSubmissionSlot(
            handle=frame.handle,
            input_ready_event=schedule._input_for_operation(
                operation
            ).input_ready_event,
            communication_stream=communication_streams[frame.bank],
            expert_stream=expert_streams[frame.bank],
        )
        for operation, frame in enumerate(frames)
    )
    terminal_by_bank: list[LayerSubmission | None] = [None, None]
    for frame in frames[-2:]:
        bank = frame.bank
        terminal_by_bank[bank] = LayerSubmission(
            output=frame.prepared_operation._output,
            handle=frame.handle,
        )
    first_terminal = terminal_by_bank[0]
    second_terminal = terminal_by_bank[1]
    if first_terminal is None or second_terminal is None:
        raise AssertionError("terminal plan did not cover both physical banks")
    terminal_submissions = (first_terminal, second_terminal)
    fast_plan = _build_fast_rolling_plan(frames, expert_slots)
    pipeline_preflight = pipeline._build_bound_submission_preflight(
        streams=all_streams,
        first_step=first_step,
        frames=frames,
        binding_owner=schedule,
        backend_context=backend_context,
    )
    result = RollingSubmissionPreflight(
        pipeline_owner=pipeline._owner,
        pipeline_preflight=pipeline_preflight,
        schedule=schedule,
        executor=expert_executor,
        operation_count=count,
        operation_events=operation_events,
        expert_slots=expert_slots,
        fast_enqueue=fast_enqueue,
        fast_plan=fast_plan,
        terminal_submissions=terminal_submissions,
    )
    # This point-in-time CUDA capture/range observation is intentionally the
    # final cold action. Only returning the already-built proof follows it.
    pipeline._admit_bound_submission_preflight(pipeline_preflight)
    return result


def _fail_stop_preserving_original(pipeline: MoEPipeline, error: BaseException) -> None:
    """Quarantine an in-flight transaction without replacing its first error."""

    state = pipeline.state
    if state is PipelineState.FAILED or state is PipelineState.CLOSED:
        return
    try:
        # Enter the fixed-storage quarantine directly. Calling the public
        # guard here would repeat validation and, on some Python versions,
        # construct a state-membership tuple while handling MemoryError.
        pipeline._enter_failed(error)
    except BaseException:  # noqa: B036 - preserve the original post-enqueue error.
        # ``error`` is the authoritative failure.  Backend fail-stop is already
        # best-effort and the pipeline retains uncertain device allocations.
        pass


def _enqueue_expert(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    handle: DispatchHandle,
    *,
    communication_stream: Any,
    expert_stream: Any,
) -> ExpertSubmission:
    try:
        output_target = pipeline.get_combine_buffer(handle)
        if expert_stream is not communication_stream:
            handle.enqueue_dispatch_wait(expert_stream)
        expert = expert_executor.enqueue(
            expert_input=handle.expert_input,
            expert_counts=handle.expert_counts,
            source_info=handle.source_info,
            layout_ranges=handle.layout_ranges,
            output=output_target,
            handle=handle,
            stream=expert_stream,
        )
        if not isinstance(expert, ExpertSubmission):
            raise TypeError("expert executor must return an ExpertSubmission")
        if expert.output is not output_target:
            raise ValueError(
                "production expert executor must write the registered output target"
            )
        if expert_stream is not communication_stream and expert.ready_event is None:
            raise ValueError("cross-stream expert execution must return a device event")
        return expert
    except BaseException as error:
        _fail_stop_preserving_original(pipeline, error)
        raise


def _enqueue_preallocated_expert(
    pipeline: MoEPipeline,
    fast_enqueue: Any,
    slot: _ExpertSubmissionSlot,
) -> None:
    """Fill one private cold slot through an allocation-audited adapter."""

    try:
        handle = slot.handle
        dispatch_wait = slot.dispatch_wait
        if dispatch_wait is not None:
            if not handle._dispatch_event_valid:
                raise PipelineError("prepared dispatch event lease is unavailable")
            dispatch_wait(handle._dispatch_ready_event)
        ready_event = fast_enqueue(
            expert_input=slot.expert_input,
            expert_counts=slot.expert_counts,
            source_info=slot.source_info,
            layout_ranges=slot.layout_ranges,
            output=slot.output,
            handle=handle,
            stream=slot.expert_stream,
        )
        if dispatch_wait is not None and ready_event is None:
            raise ValueError(
                "cross-stream prepared expert execution must return a device event"
            )
        slot.ready_event = ready_event
    except BaseException as error:
        _fail_stop_preserving_original(pipeline, error)
        raise


def _enqueue_combine(
    pipeline: MoEPipeline,
    handle: DispatchHandle,
    expert: ExpertSubmission,
    topk_weights: Any,
    output: Any,
    *,
    communication_stream: Any,
    prepared_operation: PreparedOperation | None = None,
) -> LayerSubmission:
    if prepared_operation is None:
        output = pipeline.combine(
            expert.output,
            topk_weights,
            handle,
            output=output,
            stream=communication_stream,
            expert_ready_event=expert.ready_event,
        )
    else:
        output = pipeline.combine_prepared(
            expert.output,
            handle,
            prepared_operation,
            stream=communication_stream,
            expert_ready_event=expert.ready_event,
        )
    return LayerSubmission(
        output=output,
        handle=handle,
    )


def _consume_schedule_preflight(
    pipeline: MoEPipeline,
    *,
    streams: Sequence[Any],
    step_count: int,
    submission_preflight: SubmissionPreflight | None,
) -> None:
    """Fully validate or identity-consume one proof before the first post."""

    first_step = pipeline.next_operation_step
    preflight = submission_preflight
    if preflight is None:
        preflight = pipeline.validate_submission_context(
            streams=streams,
            first_step=first_step,
            step_count=step_count,
        )
    pipeline._consume_submission_preflight(
        preflight,
        streams=streams,
        first_step=first_step,
        step_count=step_count,
    )


def _consume_rolling_preflight(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    schedule: BoundRollingSchedule,
    preflight: RollingSubmissionPreflight,
    *,
    operation_count: int,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
    operation_events: Sequence[tuple[Any, Any]] | None,
) -> None:
    """Identity-consume a fully materialized proof without cold allocation."""

    if not isinstance(preflight, RollingSubmissionPreflight):
        raise TypeError(
            "submission_preflight must be returned by " "preflight_rolling_moe_layers()"
        )
    if preflight._pipeline_owner is not pipeline._owner:
        raise PipelineError("rolling preflight belongs to another pipeline")
    if preflight._schedule is not schedule:
        raise PipelineError("rolling preflight belongs to another schedule")
    if preflight._executor is not expert_executor:
        raise PipelineError("rolling preflight belongs to another expert executor")
    if preflight._operation_count != operation_count:
        raise PipelineError("rolling preflight operation count does not match")
    if preflight._operation_events is not operation_events:
        raise PipelineError("rolling preflight timing events do not match")
    admitted = preflight._pipeline_preflight._streams
    if (
        len(admitted) != 4
        or admitted[0] is not communication_streams[0]
        or admitted[1] is not communication_streams[1]
        or admitted[2] is not expert_streams[0]
        or admitted[3] is not expert_streams[1]
    ):
        raise PipelineError("rolling preflight stream sequence does not match")
    pipeline._consume_submission_preflight(
        preflight._pipeline_preflight,
        streams=admitted,
        first_step=preflight._pipeline_preflight._first_step,
        step_count=operation_count,
        binding_owner=schedule,
    )


def enqueue_moe_layer(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    operation: PreparedOperation,
    *,
    communication_stream: Any,
    expert_stream: Any | None = None,
    input_ready_event: Any | None = None,
    submission_preflight: SubmissionPreflight | None = None,
) -> LayerSubmission:
    """Enqueue one full layer with same-stream or event-linked dependencies.

    All count and layout arguments passed to ``expert_executor`` are device
    tensors.  Implementations should use a grouped-GEMM scheduler that consumes
    them directly; calling ``item()``, ``tolist()``, or ``synchronize()`` here
    would put CPU latency back on the critical path. ``output`` must be
    preallocated by the caller; the pipeline never allocates in this hot path
    and never aliases the result with reusable communication-bank storage.

    A repeated call does *not* pipeline consecutive layers: combine's event wait
    reaches the communication stream before the next call can dispatch. Use
    :func:`enqueue_two_bank_window` for explicit communication/compute overlap.
    ``operation`` must be created by :meth:`MoEPipeline.prepare_operation` in a
    cold phase. That explicit bind validates all four operands and makes the
    backend's strongly retained descriptor set finite and intentional::

        operation = pipeline.prepare_operation(x, indices, weights, output)
        enqueue_moe_layer(pipeline, executor, operation, ...)

    A supplied ``submission_preflight`` is a historical capture observation,
    not a stream lock. It is valid only under exclusive single-host-submitter
    ownership, with no capture or CUDA operation on either admitted stream
    between token issue and this helper's immediate consumption.
    """

    if not callable(getattr(expert_executor, "enqueue", None)):
        raise TypeError("expert_executor must provide enqueue()")
    pipeline._validate_prepared_operation(operation)
    compute_stream = communication_stream if expert_stream is None else expert_stream
    if not callable(getattr(communication_stream, "wait_event", None)) or not callable(
        getattr(compute_stream, "wait_event", None)
    ):
        raise TypeError(
            "communication_stream and resolved expert_stream must expose "
            "wait_event(event)"
        )
    _consume_schedule_preflight(
        pipeline,
        streams=(communication_stream, compute_stream),
        step_count=1,
        submission_preflight=submission_preflight,
    )
    posted = False
    try:
        # The wrapper may be interrupted immediately after the CUDA-facing call
        # has posted work but before Python regains control.  Cross that
        # uncertainty boundary first so every such failure is fail-stop.
        posted = True
        handle = pipeline.dispatch_prepared(
            operation,
            stream=communication_stream,
            input_ready_event=input_ready_event,
        )
        expert = _enqueue_expert(
            pipeline,
            expert_executor,
            handle,
            communication_stream=communication_stream,
            expert_stream=compute_stream,
        )
        return _enqueue_combine(
            pipeline,
            handle,
            expert,
            operation._topk_weights,
            operation._output,
            communication_stream=communication_stream,
            prepared_operation=operation,
        )
    except BaseException as error:
        if posted:
            _fail_stop_preserving_original(pipeline, error)
        raise


def enqueue_two_bank_window(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    window: BoundRollingSchedule,
    *,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
    submission_preflight: RollingSubmissionPreflight | None = None,
) -> tuple[LayerSubmission, LayerSubmission]:
    """Submit ``D0,E0,D1,E1,C0,C1`` with device-only dependency edges.

    ``E0`` waits for dispatch zero on ``expert_streams[0]``. The host then
    submits ``D1`` before either combine wait reaches a communication stream, so
    dispatch one can overlap expert zero. Both combine calls wait on their
    expert events on-device. The returned bank-reuse events protect the next
    window; no event is queried by this function.

    Create ``window`` with :func:`bind_two_bank_window` during cold setup. That
    bind validates both operations, their cross-bank alias contract, and all
    backend descriptors before this function can reach D0. The seal is valid
    only while every tensor keeps the same storage, shape, and data pointer;
    resizing or rebinding a sealed tensor is a caller contract violation.

    The two operations must be independent microbatches: ``second`` input
    readiness cannot depend on ``first.output``. Use sequential single-layer
    submission for such a dependency; otherwise D1 waits for an event produced
    after C0 while the communication order makes C0 wait for D1.

    A supplied preflight has the same exclusive-submitter,
    no-intervening-stream-operation contract as :func:`enqueue_moe_layer`.
    """

    if not isinstance(window, BoundRollingSchedule):
        raise TypeError(
            "window must be returned by bind_two_bank_window() during cold "
            "preparation"
        )
    storage = window.storage_schedule
    if len(storage.windows) != 1 or storage.repeat_count != 1:
        raise ValueError("two-bank submission requires exactly one bound window")
    return enqueue_rolling_moe_layers(
        pipeline,
        expert_executor,
        window,
        communication_streams=communication_streams,
        expert_streams=expert_streams,
        submission_preflight=submission_preflight,
    )


def _enqueue_preflighted_fast_rolling(
    pipeline: MoEPipeline,
    preflight: RollingSubmissionPreflight,
    *,
    submission_ready_event: Any | None,
) -> tuple[LayerSubmission, LayerSubmission]:
    """Post the uninstrumented prepared-adapter loop with no per-op mode branch."""

    plan = preflight._fast_plan
    fast_enqueue = preflight._fast_enqueue
    if fast_enqueue is None:
        raise AssertionError("fast rolling path requires a prepared expert adapter")
    admitted = preflight._pipeline_preflight._streams
    posted = False
    try:
        if submission_ready_event is not None:
            posted = True
            admitted[0].wait_event(submission_ready_event)
            admitted[1].wait_event(submission_ready_event)

        # Prologue: fill both physical banks before either combine dependency
        # reaches a communication stream.
        first_frame = plan.first_frame
        first_slot = plan.first_slot
        # A Python asynchronous exception may land after this CUDA-facing call
        # returns but before the next bytecode.  Mark the pipeline uncertain
        # before entering that boundary so it cannot remain reusable.
        posted = True
        pipeline._dispatch_preallocated(
            first_frame,
            input_ready_event=first_slot.input_ready_event,
        )
        _enqueue_preallocated_expert(
            pipeline,
            fast_enqueue,
            first_slot,
        )
        second_frame = plan.second_frame
        second_slot = plan.second_slot
        pipeline._dispatch_preallocated(
            second_frame,
            input_ready_event=second_slot.input_ready_event,
        )
        _enqueue_preallocated_expert(
            pipeline,
            fast_enqueue,
            second_slot,
        )

        # Steady state: one combine frees the bank used by the dispatch two
        # positions ahead. There is one loop condition and no per-op mode or
        # instrumentation branch.
        link = plan.steady_head
        while link is not None:
            pipeline._combine_preallocated(
                link.combine_frame,
                expert_output=link.combine_slot.output,
                expert_ready_event=link.combine_slot.ready_event,
            )
            pipeline._dispatch_preallocated(
                link.dispatch_frame,
                input_ready_event=link.dispatch_slot.input_ready_event,
            )
            _enqueue_preallocated_expert(
                pipeline,
                fast_enqueue,
                link.dispatch_slot,
            )
            link = link.next_link

        # Epilogue: drain the final complete physical-bank pair.
        pipeline._combine_preallocated(
            plan.penultimate_frame,
            expert_output=plan.penultimate_slot.output,
            expert_ready_event=plan.penultimate_slot.ready_event,
        )
        pipeline._combine_preallocated(
            plan.final_frame,
            expert_output=plan.final_slot.output,
            expert_ready_event=plan.final_slot.ready_event,
        )
        # Publishing the external leases is part of the guarded transaction:
        # an asynchronous exception at return must not leave an apparently
        # reusable pipeline whose completed leases were never delivered.
        return preflight._terminal_submissions
    except BaseException as error:
        if posted:
            _fail_stop_preserving_original(pipeline, error)
        raise


def enqueue_rolling_moe_layers(
    pipeline: MoEPipeline,
    expert_executor: ExpertExecutor,
    schedule: BoundRollingSchedule,
    *,
    operation_count: int | None = None,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
    operation_events: Sequence[tuple[Any, Any]] | None = None,
    submission_preflight: RollingSubmissionPreflight | None = None,
    submission_ready_event: Any | None = None,
) -> tuple[LayerSubmission, LayerSubmission]:
    """Submit a steady-state ``D0,D1,C0,D2,C1,D3,...`` schedule.

    Only peer-polling communication kernels are totally ordered; expert work
    remains on two independent streams. Thus dispatch ``i+1`` can overlap
    expert ``i``, and combine ``i`` can be followed by dispatch ``i+2`` before
    combine ``i+1`` waits for its expert. Every active rank must call this
    collective helper with the same operation count and ordering, exactly like
    an NCCL collective sequence. Divergent rank-local call order is invalid.

    ``schedule`` seals storage before submission. Its tensors must not be
    resized, rebound with ``set_()``, or otherwise given a new data pointer
    until the returned bank-reuse milestones have completed. The rolling order
    is intended for independent microbatches: input ``i+2`` must not depend on
    output ``i+1``. Sequential layers with that dependency must use complete
    two-bank windows, because globally ordering D(i+2) before C(i+1) would form
    a device dependency cycle.

    Production submission requires a :class:`BoundRollingSchedule`; call
    :func:`bind_rolling_schedule` in a cold phase. The bound object holds only
    one backend operand token per unique tensor binding, regardless of repeat
    count. For an allocation-free measured post loop, also call
    :func:`preflight_rolling_moe_layers` before timing and supply its one-shot
    proof here. Omitting it is a convenience path that materializes the same
    finite plan immediately before submission.
    ``submission_ready_event``, when supplied, is waited by both communication
    streams after preflight consumption and before D0/D1. A historical supplied
    preflight is valid only with exclusive single-host-submitter ownership and
    no capture or intervening CUDA operation on its admitted streams.
    """

    if not isinstance(schedule, BoundRollingSchedule):
        raise TypeError(
            "schedule must be a BoundRollingSchedule; call bind_rolling_schedule() "
            "during cold preparation"
        )
    count = _resolve_rolling_count(schedule, operation_count)
    preflight = submission_preflight
    if preflight is None:
        preflight = preflight_rolling_moe_layers(
            pipeline,
            expert_executor,
            schedule,
            operation_count=count,
            communication_streams=communication_streams,
            expert_streams=expert_streams,
            operation_events=operation_events,
        )
    _consume_rolling_preflight(
        pipeline,
        expert_executor,
        schedule,
        preflight,
        operation_count=count,
        communication_streams=communication_streams,
        expert_streams=expert_streams,
        operation_events=operation_events,
    )

    frames = preflight._pipeline_preflight._frames
    expert_slots = preflight._expert_slots
    pending_experts = preflight._pending_experts
    fast_enqueue = preflight._fast_enqueue
    posted = False
    try:
        if fast_enqueue is not None and operation_events is None:
            # The nested helper protects every device-facing operation and its
            # own return.  This outer guard is still required: an asynchronous
            # exception can land in this caller after CALL completes but before
            # its RETURN_VALUE publishes the terminal leases.  Conservatively
            # cross the uncertainty boundary before handing off.
            posted = True
            return _enqueue_preflighted_fast_rolling(
                pipeline,
                preflight,
                submission_ready_event=submission_ready_event,
            )
        if submission_ready_event is not None:
            # The first wait may have reached CUDA even if its Python wrapper
            # raises, so cross this uncertainty boundary before calling it.
            posted = True
            for stream in communication_streams:
                stream.wait_event(submission_ready_event)
        # Two extra cursors flush the final two physical banks. Frame epochs,
        # handles, retention tuples, private expert slots, and terminal leases
        # were all constructed before capture admission.
        for cursor in range(count + 2):
            completed_operation = cursor - 2
            if completed_operation >= 0:
                frame = frames[completed_operation]
                bank = frame.handle.bank
                if fast_enqueue is None:
                    expert = pending_experts[bank]
                    if expert is None:
                        raise AssertionError(
                            "rolling scheduler lost a pending expert submission"
                        )
                    expert_output = expert.output
                    expert_ready_event = expert.ready_event
                    pending_experts[bank] = None
                else:
                    slot = expert_slots[completed_operation]
                    expert_output = slot.output
                    expert_ready_event = slot.ready_event
                pipeline._combine_preallocated(
                    frame,
                    expert_output=expert_output,
                    expert_ready_event=expert_ready_event,
                )
                if operation_events is not None:
                    operation_events[completed_operation][1].record(
                        communication_streams[bank]
                    )

            if cursor < count:
                frame = frames[cursor]
                bank = frame.handle.bank
                slot = expert_slots[cursor]
                if operation_events is not None:
                    posted = True
                    operation_events[cursor][0].record(communication_streams[bank])
                # Cross the CUDA-post uncertainty boundary before dispatch.  In
                # the uninstrumented case this is the first device-facing call.
                posted = True
                handle = pipeline._dispatch_preallocated(
                    frame,
                    input_ready_event=slot.input_ready_event,
                )
                if fast_enqueue is None:
                    pending_experts[bank] = _enqueue_expert(
                        pipeline,
                        expert_executor,
                        handle,
                        communication_stream=communication_streams[bank],
                        expert_stream=expert_streams[bank],
                    )
                else:
                    _enqueue_preallocated_expert(
                        pipeline,
                        fast_enqueue,
                        slot,
                    )
        # Keep terminal lease publication under the same fail-stop guard as the
        # CUDA posts.  Losing this return to an asynchronous exception would
        # otherwise strand leases while leaving the pipeline apparently usable.
        return preflight._terminal_submissions
    except BaseException as error:
        if posted:
            _fail_stop_preserving_original(pipeline, error)
        raise


def next_topology(
    current: StableSparseTopology,
    active_ranks: Iterable[int],
    *,
    restarted_ranks: Iterable[int] = (),
) -> StableSparseTopology:
    """Build the next stable sparse membership and its incarnation table.

    A rank that merely waits as a live standby keeps its incarnation.  A rank
    backed by a newly started process must be listed in ``restarted_ranks`` and
    receives a larger incarnation.  Stable rank and expert IDs never move.
    """

    if not isinstance(current, StableSparseTopology):
        raise TypeError("current must be a StableSparseTopology")
    active = tuple(active_ranks)
    restarted = tuple(restarted_ranks)
    for name, values in (("active_ranks", active), ("restarted_ranks", restarted)):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicates")
        for rank in values:
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"{name} must contain integer rank IDs")
            if not 0 <= rank < current.max_ranks:
                raise ValueError(
                    f"{name} rank {rank} is outside [0, {current.max_ranks})"
                )
    if not set(restarted) <= set(active):
        raise ValueError("every restarted rank must be active in the next generation")

    incarnations = list(current.rank_incarnations)
    for rank in restarted:
        incarnations[rank] += 1
    for rank in active:
        if incarnations[rank] == 0:
            raise ValueError(
                f"new rank {rank} has no process incarnation; list it in "
                "restarted_ranks or install its incarnation from the control plane"
            )
    return current.transition(
        active,
        rank_incarnations=tuple(incarnations),
    )


@dataclass(frozen=True, slots=True)
class GracefulChange:
    """Control-plane object staged while the old generation may still run."""

    staged: StagedGeneration

    @classmethod
    def stage(
        cls,
        pipeline: MoEPipeline,
        active_ranks: Iterable[int],
        *,
        restarted_ranks: Iterable[int] = (),
    ) -> "GracefulChange":
        topology = next_topology(
            pipeline.topology,
            active_ranks,
            restarted_ranks=restarted_ranks,
        )
        try:
            staged = pipeline.stage_generation(topology)
            return cls(staged)
        except BaseException as error:
            # The pipeline/backend now own staged resources. If wrapper
            # publication fails, no caller can otherwise reach drain/commit.
            _fail_stop_preserving_original(pipeline, error)
            raise

    def drain_and_commit(self, pipeline: MoEPipeline, *, stream: Any) -> None:
        """Leave the hot loop, drain old WQEs, globally converge, then swap."""

        try:
            drain = pipeline.begin_generation_drain(self.staged, stream=stream)
            pipeline.commit_generation(self.staged, drain)
        except BaseException as error:
            # Once drain is published, retrying this convenience transaction is
            # ambiguous. This also covers an async exception immediately after
            # a successful commit by inspecting the staged object's state.
            if (
                pipeline.state == PipelineState.DRAINING
                or self.staged.state != GenerationChangeState.STAGED
            ):
                _fail_stop_preserving_original(pipeline, error)
            raise


def parse_membership(value: str, *, max_ranks: int) -> tuple[tuple[int, ...], ...]:
    """Parse ``"0,1;0,1,2;0,2"`` without renumbering sparse rank slots."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("membership must be a non-empty string")
    if isinstance(max_ranks, bool) or not isinstance(max_ranks, int) or max_ranks <= 0:
        raise ValueError("max_ranks must be a positive integer")
    result: list[tuple[int, ...]] = []
    for phase_index, phase_text in enumerate(value.split(";")):
        fields = tuple(field.strip() for field in phase_text.split(","))
        if not fields or any(not field for field in fields):
            raise ValueError(f"membership phase {phase_index} is empty")
        try:
            ranks = tuple(int(field, 10) for field in fields)
        except ValueError as error:
            raise ValueError(
                f"membership phase {phase_index} contains a non-integer rank"
            ) from error
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"membership phase {phase_index} contains duplicates")
        if any(rank < 0 or rank >= max_ranks for rank in ranks):
            raise ValueError(f"membership phase {phase_index} exceeds [0, {max_ranks})")
        result.append(tuple(sorted(ranks)))
    return tuple(result)


def topology_plan(
    membership: Sequence[Sequence[int]],
    *,
    max_ranks: int,
    experts_per_rank: int,
) -> tuple[StableSparseTopology, ...]:
    """Materialize a live-standby plan with stable process incarnations."""

    if not membership:
        raise ValueError("membership must contain at least one generation")
    # This example assumes all fixed-capacity workers were started together and
    # wait as live standbys.  Process replacement would increment one entry.
    incarnations = (1,) * max_ranks
    topologies = []
    for generation, active in enumerate(membership):
        topologies.append(
            StableSparseTopology(
                max_ranks=max_ranks,
                experts_per_rank=experts_per_rank,
                active_ranks=tuple(active),
                membership_generation=generation,
                rank_incarnations=incarnations,
            )
        )
    return tuple(topologies)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-ranks", type=int, default=4)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument(
        "--membership",
        default="0,1;0,1,2,3;0,1,3;0,1,2,3",
        help="semicolon-separated stable sparse membership generations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    membership = parse_membership(args.membership, max_ranks=args.max_ranks)
    topologies = topology_plan(
        membership,
        max_ranks=args.max_ranks,
        experts_per_rank=args.experts_per_rank,
    )
    print("Validated split-phase elastic MoE plan (live standby processes):")
    for topology in topologies:
        print(
            f"  generation={topology.membership_generation} "
            f"active={topology.active_ranks} mask={topology.nixl_mask} "
            f"experts={topology.active_experts}"
        )
    print(
        "\nSteady state: dispatch(x, topk_idx) -> grouped expert -> "
        "combine(expert_output, weights, output=out); no host tensor read or sync."
    )
    print(
        "Generation boundary: stage views -> finish every handle -> enqueue "
        "cumulative drain -> global commit. Abrupt peer loss is fail-stop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundRollingSchedule",
    "ExpertExecutor",
    "ExpertSubmission",
    "GracefulChange",
    "LayerInputs",
    "LayerSubmission",
    "PreparedExpertExecutor",
    "PreparedRollingSchedule",
    "PreparedTwoBankWindow",
    "RollingSubmissionPreflight",
    "bind_rolling_schedule",
    "bind_two_bank_window",
    "enqueue_moe_layer",
    "enqueue_rolling_moe_layers",
    "enqueue_two_bank_window",
    "main",
    "next_topology",
    "parse_membership",
    "preflight_rolling_moe_layers",
    "prepare_rolling_schedule",
    "prepare_two_bank_window",
    "topology_plan",
]
