# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from examples.python.cute import elastic_moe_pipeline as elastic_module
from examples.python.cute.elastic_moe_pipeline import (
    BoundRollingSchedule,
    ExpertSubmission,
    GracefulChange,
    LayerInputs,
    PreparedRollingSchedule,
    PreparedTwoBankWindow,
    RollingSubmissionPreflight,
    bind_rolling_schedule,
    bind_two_bank_window,
    enqueue_moe_layer,
    enqueue_rolling_moe_layers,
    enqueue_two_bank_window,
    next_topology,
    parse_membership,
    preflight_rolling_moe_layers,
    prepare_rolling_schedule,
    prepare_two_bank_window,
    topology_plan,
)
from examples.python.cute.moe import pipeline as pipeline_module
from examples.python.cute.moe.ll_protocol import (
    BankPhase,
    Direction,
    PipelineLLArenaLayout,
    StableSparseTopology,
)
from examples.python.cute.moe.pipeline import (
    GENERATION_REBASE_PLANES,
    MAX_PREPARED_SUBMISSION_OPERATIONS,
    AbruptPeerLoss,
    CombineEvents,
    GenerationChangeState,
    GenerationCommit,
    GenerationDrain,
    HandleState,
    MoEPipeline,
    PipelineBankBuffers,
    PipelineError,
    PipelineFailed,
    PipelineState,
    SubmissionPreflight,
)


class FakeTensor:
    """Shape-only tensor whose value-reading APIs fail the test immediately."""

    _next_pointer = 0x100000

    def __init__(self, shape: tuple[int, ...], device: str = "cuda:0") -> None:
        self.shape = shape
        self.device = device
        self._pointer = FakeTensor._next_pointer
        FakeTensor._next_pointer += max(self.numel() * self.element_size(), 4096)

    def data_ptr(self) -> int:
        return self._pointer

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self) -> int:
        return 4

    def is_contiguous(self) -> bool:
        return True

    def item(self) -> Any:
        raise AssertionError("hot path called Tensor.item()")

    def tolist(self) -> Any:
        raise AssertionError("hot path called Tensor.tolist()")

    def synchronize(self) -> Any:
        raise AssertionError("hot path synchronized a tensor")


@dataclass(frozen=True)
class FakeEvent:
    name: str


class FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waits: list[FakeEvent] = []

    def wait_event(self, event: FakeEvent) -> None:
        self.waits.append(event)

    def __repr__(self) -> str:
        return f"FakeStream({self.name!r})"


class EqualityMaskingStream(FakeStream):
    """Opaque wrappers may compare equal without naming the same CUDA stream."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EqualityMaskingStream)


@dataclass(frozen=True)
class FakeSubmissionContext:
    streams: tuple[Any, ...]
    first_step: int
    step_count: int


@dataclass(frozen=True)
class FakeOperationLaunch:
    context: FakeSubmissionContext
    operation: Any
    source_incarnation: int
    buffers: PipelineBankBuffers
    prepared_operation: Any
    communication_stream: Any
    expert_stream: Any


class FakeBackend:
    def __init__(self, layout: PipelineLLArenaLayout, device: str = "cuda:0") -> None:
        self.layout = layout
        self.device = device
        self.events: list[tuple[Any, ...]] = []
        self.prepared_tokens: list[Any | None] = []
        self.submission_validations: list[tuple[Any, ...]] = []
        self.submission_validation_error: BaseException | None = None
        self.submission_context: FakeSubmissionContext | None = None
        self.submission_preparations: list[FakeSubmissionContext] = []
        self.fail_dispatch: BaseException | None = None
        self.fail_combine: BaseException | None = None
        self.bad_drain = False
        self.missing_rebase_plane: str | None = None
        self.last_commit: GenerationCommit | None = None

    def _bank(self, bank: int) -> PipelineBankBuffers:
        shape = self.layout
        rows = shape.max_ranks * shape.num_tokens
        return PipelineBankBuffers(
            bank=bank,
            expert_input=FakeTensor(
                (shape.experts_per_rank, rows, shape.hidden_size), self.device
            ),
            expert_counts=FakeTensor((shape.experts_per_rank,), self.device),
            source_info=FakeTensor((shape.experts_per_rank, rows, 4), self.device),
            layout_ranges=FakeTensor(
                (shape.experts_per_rank, shape.max_ranks), self.device
            ),
            combine_stage=FakeTensor(
                (shape.experts_per_rank, rows, shape.hidden_size), self.device
            ),
        )

    def preallocate(self, *, rank, layout, topology):
        assert layout is self.layout
        self.events.append(("preallocate", rank, topology.membership_generation))
        return (self._bank(0), self._bank(1))

    def prepare_operation(
        self, activations, topk_indices, topk_weights, combine_output
    ):
        token = object()
        self.events.append(
            (
                "prepare_operation",
                activations,
                topk_indices,
                topk_weights,
                combine_output,
                token,
            )
        )
        return token

    def prepare_submission_context(self, *, streams, first_step, step_count):
        stream_tuple = tuple(streams)
        current = self.submission_context
        if (
            current is not None
            and current.first_step == first_step
            and current.step_count == step_count
            and len(current.streams) == len(stream_tuple)
            and all(
                actual is expected
                for actual, expected in zip(stream_tuple, current.streams)
            )
        ):
            return current
        current = FakeSubmissionContext(stream_tuple, first_step, step_count)
        self.submission_context = current
        self.submission_preparations.append(current)
        return current

    def prepare_operation_launch(self, **arguments):
        context = arguments["submission_context"]
        if context is not self.submission_context:
            raise PipelineError("stale fake submission context")
        return FakeOperationLaunch(
            context=context,
            operation=arguments["operation"],
            source_incarnation=arguments["source_incarnation"],
            buffers=arguments["buffers"],
            prepared_operation=arguments["prepared_operation"],
            communication_stream=arguments["communication_stream"],
            expert_stream=arguments["expert_stream"],
        )

    def validate_prepared_submission_context(self, context):
        if context is not self.submission_context:
            raise PipelineError("fake submission context was replaced")

    def validate_submission_context(self, *, streams, first_step, step_count):
        self.submission_validations.append((tuple(streams), first_step, step_count))
        if self.submission_validation_error is not None:
            raise self.submission_validation_error

    def enqueue_dispatch(self, **arguments):
        if self.fail_dispatch is not None:
            raise self.fail_dispatch
        self.prepared_tokens.append(arguments.get("prepared_operation"))
        self.events.append(
            (
                "dispatch",
                arguments["operation"],
                arguments["source_incarnation"],
                arguments["buffers"].bank,
                arguments["activations"],
                arguments["topk_indices"],
                arguments["stream"],
                arguments["communication_predecessor_event"],
                arguments["input_ready_event"],
                arguments["bank_reuse_event"],
            )
        )
        return FakeEvent(f"dispatch-{arguments['operation'].wire_value}")

    def enqueue_prepared_dispatch(
        self,
        launch,
        *,
        input_ready_event,
        communication_predecessor_event,
    ):
        return self.enqueue_dispatch(
            operation=launch.operation,
            source_incarnation=launch.source_incarnation,
            buffers=launch.buffers,
            activations=launch.prepared_operation,
            topk_indices=launch.prepared_operation,
            stream=launch.communication_stream,
            communication_predecessor_event=communication_predecessor_event,
            input_ready_event=input_ready_event,
            bank_reuse_event=None,
            prepared_operation=launch.prepared_operation,
        )

    def enqueue_combine(self, **arguments):
        if self.fail_combine is not None:
            raise self.fail_combine
        self.prepared_tokens.append(arguments.get("prepared_operation"))
        self.events.append(
            (
                "combine",
                arguments["operation"],
                arguments["buffers"].bank,
                arguments["expert_output"],
                arguments["topk_weights"],
                arguments["combine_output"],
                arguments["stream"],
                arguments["communication_predecessor_event"],
                arguments["expert_ready_event"],
                arguments["zero_copy"],
            )
        )
        operation = arguments["operation"].wire_value
        return CombineEvents(
            FakeEvent(f"output-{operation}"), FakeEvent(f"reuse-{operation}")
        )

    def enqueue_prepared_combine(
        self,
        launch,
        *,
        expert_ready_event,
        communication_predecessor_event,
    ):
        return self.enqueue_combine(
            operation=launch.operation,
            buffers=launch.buffers,
            expert_output=launch.buffers.combine_stage,
            topk_weights=launch.prepared_operation,
            combine_output=launch.prepared_operation,
            stream=launch.communication_stream,
            communication_predecessor_event=communication_predecessor_event,
            expert_ready_event=expert_ready_event,
            zero_copy=True,
            prepared_operation=launch.prepared_operation,
        )

    def stage_generation(self, *, current, next_topology):
        token = object()
        self.events.append(
            (
                "stage",
                current.membership_generation,
                next_topology.membership_generation,
                token,
            )
        )
        return token

    def enqueue_generation_drain(
        self, *, current, staged_token, completion_events, stream
    ):
        self.events.append(
            (
                "drain",
                current.membership_generation,
                staged_token,
                tuple(completion_events),
                stream,
            )
        )
        if self.bad_drain:
            return object()
        return GenerationDrain(object())

    def commit_generation(
        self, *, current, next_topology, staged_token, drain
    ) -> GenerationCommit:
        self.events.append(
            (
                "commit",
                current.membership_generation,
                next_topology.membership_generation,
                staged_token,
                drain,
            )
        )
        planes = GENERATION_REBASE_PLANES
        if self.missing_rebase_plane is not None:
            planes = planes - {self.missing_rebase_plane}
        self.last_commit = GenerationCommit(next_topology.membership_generation, planes)
        return self.last_commit

    def fail_stop(self, error):
        self.events.append(("fail_stop", error))

    def close(self):
        self.events.append(("close",))


class FastFakeExpert:
    """Model the prepared adapter without constructing ExpertSubmission."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def enqueue(self, **_arguments):
        raise AssertionError("prepared rolling path called generic expert enqueue")

    def enqueue_prepared(self, **arguments):
        step = arguments["handle"].operation.step
        self.steps.append(step)
        return FakeEvent(f"expert-{step}")


def _layout() -> PipelineLLArenaLayout:
    return PipelineLLArenaLayout(
        max_ranks=3,
        experts_per_rank=2,
        num_tokens=4,
        top_k=2,
        hidden_size=8,
        element_size=2,
    )


def _topology(
    generation: int = 3,
    active: tuple[int, ...] = (0, 1),
    incarnations: tuple[int, ...] = (5, 7, 0),
) -> StableSparseTopology:
    return StableSparseTopology(
        max_ranks=3,
        experts_per_rank=2,
        active_ranks=active,
        membership_generation=generation,
        rank_incarnations=incarnations,
    )


def _pipeline(
    topology: StableSparseTopology | None = None,
) -> tuple[MoEPipeline, FakeBackend]:
    layout = _layout()
    backend = FakeBackend(layout)
    return (
        MoEPipeline(
            rank=0,
            topology=_topology() if topology is None else topology,
            layout=layout,
            backend=backend,
        ),
        backend,
    )


def _inputs(num_tokens: int = 4):
    return (
        FakeTensor((num_tokens, 8)),
        FakeTensor((num_tokens, 2)),
        FakeTensor((num_tokens, 2)),
    )


def _expert_output() -> FakeTensor:
    return FakeTensor((2, 12, 8))


def _output(num_tokens: int = 4) -> FakeTensor:
    return FakeTensor((num_tokens, 8))


def _bound_two_bank(
    pipeline: MoEPipeline,
    first: LayerInputs,
    second: LayerInputs,
) -> BoundRollingSchedule:
    return bind_two_bank_window(
        pipeline,
        prepare_two_bank_window(first, second),
    )


def test_preallocation_is_exactly_two_typed_banks_and_happens_once():
    pipeline, backend = _pipeline()

    assert backend.events == [("preallocate", 0, 3)]
    handle = pipeline.dispatch(*_inputs()[:2], stream="stream-0")
    assert handle.expert_input.shape == (2, 12, 8)
    assert handle.expert_counts.shape == (2,)
    assert handle.source_info.shape == (2, 12, 4)
    assert handle.layout_ranges.shape == (2, 3)
    assert handle.combine_buffer.shape == (2, 12, 8)
    consumer = FakeStream("expert")
    handle.enqueue_dispatch_wait(consumer)
    assert consumer.waits == [FakeEvent(f"dispatch-{handle.operation.wire_value}")]
    assert [event[0] for event in backend.events].count("preallocate") == 1

    bad_backend = FakeBackend(_layout())
    bad_backend.preallocate = lambda **_: (bad_backend._bank(0),)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="exactly 2 banks"):
        MoEPipeline(
            rank=0, topology=_topology(), layout=bad_backend.layout, backend=bad_backend
        )

    wrong_shape = FakeBackend(_layout())
    original = wrong_shape._bank

    def bad_bank(bank):
        value = original(bank)
        if bank == 1:
            return PipelineBankBuffers(
                bank=bank,
                expert_input=FakeTensor((2, 11, 8)),
                expert_counts=value.expert_counts,
                source_info=value.source_info,
                layout_ranges=value.layout_ranges,
                combine_stage=value.combine_stage,
            )
        return value

    wrong_shape._bank = bad_bank  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="expert_input must have shape"):
        MoEPipeline(
            rank=0,
            topology=_topology(),
            layout=wrong_shape.layout,
            backend=wrong_shape,
        )


def test_dispatch_combine_is_split_phase_two_bank_and_one_shot():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs(2)

    first = pipeline.dispatch(x0, route0, stream="stream-0")
    second = pipeline.dispatch(x1, route1, stream="stream-0")
    assert (first.bank, second.bank) == (0, 1)
    assert first.operation.step == 0
    assert second.operation.step == 1
    assert first.source_incarnation == 5
    assert first.state == HandleState.DISPATCHED
    assert pipeline.outstanding_handles == (first, second)

    with pytest.raises(PipelineError, match="bank 0 still has uncombined"):
        pipeline.dispatch(x0, route0, stream="stream-0")

    destination0 = _output()
    output = pipeline.combine(
        _expert_output(),
        weight0,
        first,
        output=destination0,
        stream="stream-0",
    )
    assert output is destination0
    assert output.shape == (4, 8)
    assert first.state == HandleState.COMBINE_ENQUEUED
    output_consumer = FakeStream("output")
    bank_observer = FakeStream("bank")
    first.enqueue_output_wait(output_consumer)
    first.enqueue_bank_reuse_wait(bank_observer)
    assert output_consumer.waits == [FakeEvent(f"output-{first.operation.wire_value}")]
    assert bank_observer.waits == [FakeEvent(f"reuse-{first.operation.wire_value}")]
    with pytest.raises(PipelineError, match="dispatch-ready event lease expired"):
        first.enqueue_dispatch_wait(FakeStream("late-expert"))
    assert pipeline.outstanding_handles == (second,)
    with pytest.raises(PipelineError, match="already consumed"):
        pipeline.combine(
            _expert_output(),
            weight0,
            first,
            output=_output(),
            stream="stream-0",
        )

    third = pipeline.dispatch(x0, route0, stream="stream-0")
    assert third.bank == 0
    assert third.operation.step == 2
    with pytest.raises(PipelineError, match="lease expired"):
        first.enqueue_output_wait(output_consumer)
    with pytest.raises(PipelineError, match="lease expired"):
        first.enqueue_bank_reuse_wait(bank_observer)
    pipeline.combine(
        _expert_output(), weight1, second, output=_output(2), stream="stream-0"
    )
    pipeline.combine(
        _expert_output(), weight0, third, output=_output(), stream="stream-0"
    )

    assert [event[0] for event in backend.events] == [
        "preallocate",
        "dispatch",
        "dispatch",
        "combine",
        "dispatch",
        "combine",
        "combine",
    ]
    assert backend.events[1][4] is x0
    assert backend.events[1][5] is route0


def test_zero_token_active_rank_still_submits_collective_step_and_empty_output():
    pipeline, backend = _pipeline()
    activations, routes, weights = _inputs(0)
    destination = _output(0)

    handle = pipeline.dispatch(activations, routes, stream="stream-0")
    result = pipeline.combine(
        _expert_output(),
        weights,
        handle,
        output=destination,
        stream="stream-0",
    )

    assert handle.num_tokens == 0
    assert result is destination
    assert result.shape == (0, 8)
    assert [event[0] for event in backend.events] == [
        "preallocate",
        "dispatch",
        "combine",
    ]


def test_hot_path_rejects_bad_shape_device_stream_and_foreign_handle():
    pipeline, _backend = _pipeline()
    x, route, weights = _inputs()

    with pytest.raises(ValueError, match="hidden size"):
        pipeline.dispatch(FakeTensor((4, 7)), route, stream="stream-0")
    with pytest.raises(ValueError, match="topk_indices must have shape"):
        pipeline.dispatch(x, FakeTensor((4, 1)), stream="stream-0")
    with pytest.raises(ValueError, match="share one device"):
        pipeline.dispatch(x, FakeTensor((4, 2), "cuda:1"), stream="stream-0")
    handle = pipeline.dispatch(x, route, stream="stream-0")
    with pytest.raises(PipelineError, match="cross-stream combine"):
        pipeline.combine(
            _expert_output(), weights, handle, output=_output(), stream="stream-1"
        )
    with pytest.raises(ValueError, match="expert_output must have shape"):
        pipeline.combine(
            FakeTensor((2, 11, 8)),
            weights,
            handle,
            output=_output(),
            stream="stream-0",
        )
    with pytest.raises(ValueError, match="output must have shape"):
        pipeline.combine(
            _expert_output(),
            weights,
            handle,
            output=FakeTensor((3, 8)),
            stream="stream-0",
        )
    with pytest.raises(ValueError, match="combine tensors must share"):
        pipeline.combine(
            _expert_output(),
            FakeTensor((4, 2), "cuda:1"),
            handle,
            output=_output(),
            stream="stream-0",
        )

    other, _ = _pipeline()
    with pytest.raises(PipelineError, match="another pipeline"):
        other.combine(
            _expert_output(), weights, handle, output=_output(), stream="stream-0"
        )


def test_hot_methods_have_no_tensor_materialization_or_host_sync_calls():
    source_path = Path(inspect.getsourcefile(MoEPipeline) or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"dispatch", "combine"}
    }
    forbidden = {"item", "tolist", "synchronize", "cpu", "numpy"}

    assert set(methods) == {"dispatch", "combine"}
    for name, method in methods.items():
        calls = {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not calls & forbidden, f"{name} contains host-visible call {calls}"

    pipeline, _ = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline.dispatch(x, routes, stream="stream-0")
    pipeline.combine(
        _expert_output(), weights, handle, output=_output(), stream="stream-0"
    )


def test_two_banks_encode_cross_stream_overlap_with_device_events_only():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    producer0 = FakeEvent("producer-0")
    producer1 = FakeEvent("producer-1")

    first = pipeline.dispatch(
        x0,
        route0,
        stream="dispatch-stream-0",
        input_ready_event=producer0,
    )
    second = pipeline.dispatch(
        x1,
        route1,
        stream="dispatch-stream-1",
        input_ready_event=producer1,
    )
    # An expert scheduler submits waits on these events to independent compute
    # streams. No CPU query of either event is part of the API.
    expert0 = FakeEvent("expert-0")
    expert1 = FakeEvent("expert-1")
    pipeline.combine(
        pipeline.get_combine_buffer(first),
        weight0,
        first,
        output=_output(),
        stream="combine-stream-0",
        expert_ready_event=expert0,
    )
    pipeline.combine(
        pipeline.get_combine_buffer(second),
        weight1,
        second,
        output=_output(),
        stream="combine-stream-1",
        expert_ready_event=expert1,
    )

    dispatch_events = [event for event in backend.events if event[0] == "dispatch"]
    combine_events = [event for event in backend.events if event[0] == "combine"]
    assert dispatch_events[0][-2:] == (producer0, None)
    assert dispatch_events[1][-2:] == (producer1, None)
    assert combine_events[0][-2:] == (expert0, True)
    assert combine_events[1][-2:] == (expert1, True)

    # The global communication tail transitively covers bank-zero's returned
    # peer credit, so a second wait on the older same-bank event is omitted.
    third = pipeline.dispatch(
        x0,
        route0,
        stream="dispatch-stream-0",
        input_ready_event=FakeEvent("producer-2"),
    )
    assert third.bank == 0
    assert [event for event in backend.events if event[0] == "dispatch"][-1][-1] is None

    pipeline.combine(
        pipeline.get_combine_buffer(third),
        weight0,
        third,
        output=_output(),
        stream="combine-stream-0",
        expert_ready_event=FakeEvent("expert-2"),
    )
    staged = pipeline.stage_generation(
        _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    )
    pipeline.begin_generation_drain(staged, stream="drain-stream")
    drain_event = [event for event in backend.events if event[0] == "drain"][-1]
    assert drain_event[3] == (
        FakeEvent(f"reuse-{third.operation.wire_value}"),
        FakeEvent(f"reuse-{second.operation.wire_value}"),
    )


def test_graceful_generation_is_stage_drain_global_commit_then_new_epoch():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline.dispatch(x, routes, stream="stream-0")
    shrunk = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))

    staged = pipeline.stage_generation(shrunk)
    assert staged.state == GenerationChangeState.STAGED
    # Preparation does not activate the new mask.
    assert pipeline.topology.membership_generation == 3
    with pytest.raises(PipelineError, match="every dispatch handle"):
        pipeline.begin_generation_drain(staged, stream="stream-0")

    pipeline.combine(
        _expert_output(), weights, handle, output=_output(), stream="stream-0"
    )
    drain = pipeline.begin_generation_drain(staged, stream="stream-0")
    assert pipeline.state == PipelineState.DRAINING
    assert staged.state == GenerationChangeState.DRAIN_ENQUEUED
    with pytest.raises(PipelineError, match="does not belong"):
        pipeline.commit_generation(staged, GenerationDrain(object()))

    pipeline.commit_generation(staged, drain)
    assert pipeline.state == PipelineState.ACTIVE
    assert pipeline.topology == shrunk
    assert staged.state == GenerationChangeState.COMMITTED
    assert backend.last_commit == GenerationCommit(4, GENERATION_REBASE_PLANES)
    next_handle = pipeline.dispatch(x, routes, stream="stream-1")
    assert next_handle.operation.membership_generation == 4
    assert next_handle.operation.step == 0
    event_names = [event[0] for event in backend.events]
    assert event_names[2:6] == [
        "stage",
        "combine",
        "drain",
        "commit",
    ]
    assert event_names[-1] == "dispatch"


def test_standby_rejoin_preserves_or_advances_incarnation_and_fixed_capacity():
    current = _topology(generation=8, active=(0,), incarnations=(5, 7, 0))
    pipeline, _ = _pipeline(current)

    same_process_rejoin = _topology(generation=9, active=(0, 1), incarnations=(5, 7, 0))
    staged = pipeline.stage_generation(same_process_rejoin)
    assert staged.topology.incarnation(1) == 7

    pipeline2, _ = _pipeline(current)
    valid_rejoin = _topology(generation=9, active=(0, 1), incarnations=(5, 8, 0))
    assert pipeline2.stage_generation(valid_rejoin).topology == valid_rejoin

    resized = StableSparseTopology(
        max_ranks=4,
        experts_per_rank=2,
        active_ranks=(0,),
        membership_generation=10,
        rank_incarnations=(5, 8, 0, 0),
    )
    other, _ = _pipeline(current)
    with pytest.raises(ValueError, match="fixed rank capacity"):
        other.stage_generation(resized)


def test_abrupt_active_peer_loss_is_irreversible_fail_stop():
    pipeline, backend = _pipeline()

    with pytest.raises(AbruptPeerLoss) as captured:
        pipeline.report_abrupt_peer_loss(1, "heartbeat deadline expired")
    error = captured.value
    assert error.membership_generation == 3
    assert error.peer_incarnation == 7
    assert "cannot be cancelled or flushed safely" in str(error)
    assert pipeline.state == PipelineState.FAILED
    assert pipeline.failure is error
    assert backend.events[-1] == ("fail_stop", error)

    with pytest.raises(PipelineFailed, match="fail-stopped"):
        pipeline.dispatch(*_inputs()[:2], stream="stream-0")
    with pytest.raises(PipelineFailed, match="fail-stopped"):
        pipeline.stage_generation(
            _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
        )
    with pytest.raises(PipelineFailed, match="quarantined until process teardown"):
        pipeline.close()
    assert pipeline.state == PipelineState.FAILED
    assert not any(event[0] == "close" for event in backend.events)


def test_uncertain_enqueue_failure_fail_stops_instead_of_reusing_bank():
    pipeline, backend = _pipeline()
    dispatch_inputs = _inputs()[:2]
    backend.fail_dispatch = RuntimeError("post state is unknown")

    with pytest.raises(RuntimeError, match="unknown"):
        pipeline.dispatch(*dispatch_inputs, stream="stream-0")
    assert pipeline.state == PipelineState.FAILED
    assert backend.events[-1][0] == "fail_stop"
    assert pipeline._retained[0][-2:] == dispatch_inputs

    pipeline2, backend2 = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline2.dispatch(x, routes, stream="stream-0")
    backend2.fail_combine = RuntimeError("publication state is unknown")
    expert_output = _expert_output()
    output = _output()
    with pytest.raises(RuntimeError, match="unknown"):
        pipeline2.combine(
            expert_output, weights, handle, output=output, stream="stream-0"
        )
    assert pipeline2.state == PipelineState.FAILED
    assert handle.state == HandleState.DISPATCHED
    assert pipeline2._retained[handle.bank][-3:] == (
        expert_output,
        weights,
        output,
    )
    with pytest.raises(PipelineFailed, match="quarantined until process teardown"):
        pipeline2.close()
    assert not any(event[0] == "close" for event in backend2.events)

    pipeline3, backend3 = _pipeline()
    expert_error = RuntimeError("grouped GEMM launch failed")
    pipeline3.fail_stop(expert_error)
    assert pipeline3.state == PipelineState.FAILED
    assert backend3.events[-1] == ("fail_stop", expert_error)


def test_dispatch_handle_allocation_precedes_the_device_enqueue(monkeypatch):
    pipeline, backend = _pipeline()
    events_before_dispatch = tuple(backend.events)

    class ExhaustedHandle:
        def __init__(self, **_arguments):
            raise MemoryError("handle allocation failed")

    monkeypatch.setattr(pipeline_module, "DispatchHandle", ExhaustedHandle)
    with pytest.raises(MemoryError, match="handle allocation failed"):
        pipeline.dispatch(*_inputs()[:2], stream="stream-0")

    assert tuple(backend.events) == events_before_dispatch
    assert pipeline.state == PipelineState.ACTIVE
    assert pipeline.next_operation_step == 0
    assert pipeline.outstanding_handles == ()


def test_dispatch_publication_failure_after_enqueue_is_fail_stop():
    pipeline, backend = _pipeline()

    class FailedPublication(list):
        def __setitem__(self, index, value):
            del index, value
            raise MemoryError("dispatch publication failed")

    pipeline._handles = FailedPublication(pipeline._handles)
    dispatch_inputs = _inputs()[:2]
    with pytest.raises(MemoryError, match="dispatch publication failed"):
        pipeline.dispatch(*dispatch_inputs, stream="stream-0")

    assert [event[0] for event in backend.events][-2:] == ["dispatch", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED
    assert pipeline.next_operation_step == 0
    assert pipeline._retained[0] == dispatch_inputs


def test_combine_publication_failure_after_enqueue_is_fail_stop():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline.dispatch(x, routes, stream="stream-0")

    class FailedPublication(list):
        def __setitem__(self, index, value):
            del index, value
            raise MemoryError("combine publication failed")

    pipeline._completed_handles = FailedPublication(pipeline._completed_handles)
    expert_output = _expert_output()
    output = _output()
    with pytest.raises(MemoryError, match="combine publication failed"):
        pipeline.combine(
            expert_output, weights, handle, output=output, stream="stream-0"
        )

    assert [event[0] for event in backend.events][-2:] == ["combine", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED
    assert pipeline._retained[handle.bank] == (
        x,
        routes,
        expert_output,
        weights,
        output,
    )
    with pytest.raises(PipelineError, match="dispatch-ready event lease expired"):
        handle.enqueue_dispatch_wait(FakeStream("late-expert"))
    with pytest.raises(PipelineError, match="output-ready event is unavailable"):
        handle.enqueue_output_wait(FakeStream("late-output"))
    with pytest.raises(PipelineError, match="bank-reuse event is unavailable"):
        handle.enqueue_bank_reuse_wait(FakeStream("late-observer"))


def test_close_quarantines_before_teardown_and_drops_healthy_references():
    pipeline, backend = _pipeline()
    observed_states = []

    def healthy_close():
        observed_states.append(pipeline.state)
        backend.events.append(("close",))

    backend.close = healthy_close  # type: ignore[method-assign]
    pipeline.close()
    assert observed_states == [PipelineState.CLOSING]
    assert pipeline.state == PipelineState.CLOSED
    assert pipeline._buffers == ()
    assert pipeline._retained == [(), ()]
    assert pipeline._completion_events == [None, None]

    failed, failed_backend = _pipeline()

    def broken_close():
        assert failed.state == PipelineState.CLOSING
        raise RuntimeError("teardown state is uncertain")

    failed_backend.close = broken_close  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="teardown state is uncertain"):
        failed.close()
    assert failed.state == PipelineState.FAILED
    assert failed_backend.events[-1][0] == "fail_stop"

    staged_pipeline, staged_backend = _pipeline()
    staged = staged_pipeline.stage_generation(
        _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    )
    staged_pipeline.close()
    assert staged.state == GenerationChangeState.CANCELLED
    assert staged_pipeline._staged is None
    assert staged_pipeline.state == PipelineState.CLOSED
    assert [event[0] for event in staged_backend.events][-2:] == ["stage", "close"]


def test_close_revokes_external_completion_leases_before_backend_teardown():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline.dispatch(x, routes, stream="stream-0")
    pipeline.combine(
        _expert_output(), weights, handle, output=_output(), stream="stream-0"
    )

    def broken_close():
        raise MemoryError("backend teardown failed")

    backend.close = broken_close  # type: ignore[method-assign]
    with pytest.raises(MemoryError, match="backend teardown failed"):
        pipeline.close()

    assert pipeline.state == PipelineState.FAILED
    with pytest.raises(PipelineError, match="output-ready event is unavailable"):
        handle.enqueue_output_wait(FakeStream("late-output"))
    with pytest.raises(PipelineError, match="bank-reuse event is unavailable"):
        handle.enqueue_bank_reuse_wait(FakeStream("late-observer"))


def test_checked_credit_models_remain_distinct_from_async_submission():
    pipeline, _ = _pipeline()
    x, routes, weights = _inputs()
    handle = pipeline.dispatch(x, routes, stream="stream-0")

    # Enqueue is not lied about as device completion: the optional CPU model is
    # untouched until a test or backend observer feeds it actual protocol events.
    assert pipeline.dispatch_credit_model.snapshot(0).phase == BankPhase.FREE
    peers = pipeline.topology.active_ranks
    model = pipeline.dispatch_credit_model
    model.begin(handle.operation, peers)
    for peer in peers:
        model.mark_published(handle.operation, peer)
    model.finish_publishing(handle.operation)
    for peer in peers:
        model.return_credit(handle.operation, peer)
    assert model.quiescent

    pipeline.combine(
        _expert_output(), weights, handle, output=_output(), stream="stream-0"
    )
    assert pipeline.combine_credit_model.direction == Direction.COMBINE


def test_bad_drain_response_is_fail_stop():
    pipeline, backend = _pipeline()
    next_topology = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    staged = pipeline.stage_generation(next_topology)
    backend.bad_drain = True

    with pytest.raises(TypeError, match="invalid GenerationDrain"):
        pipeline.begin_generation_drain(staged, stream="stream-0")
    assert pipeline.state == PipelineState.FAILED
    assert backend.events[-1][0] == "fail_stop"


def test_generation_commit_requires_every_sequence_credit_and_state_rebase():
    pipeline, backend = _pipeline()
    topology = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    staged = pipeline.stage_generation(topology)
    drain = pipeline.begin_generation_drain(staged, stream="drain-stream")
    backend.missing_rebase_plane = "dispatch_credit"

    with pytest.raises(PipelineError, match="dispatch_credit"):
        pipeline.commit_generation(staged, drain)
    assert pipeline.state == PipelineState.FAILED
    assert pipeline.topology.membership_generation == 3
    assert backend.events[-1][0] == "fail_stop"


def test_generation_publication_failure_after_backend_commit_is_fail_stop():
    pipeline, backend = _pipeline()
    topology = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    staged = pipeline.stage_generation(topology)
    drain = pipeline.begin_generation_drain(staged, stream="drain-stream")

    class FailedPublication(list):
        def __setitem__(self, index, value):
            del index, value
            raise MemoryError("generation publication failed")

    pipeline._retained = FailedPublication(pipeline._retained)
    with pytest.raises(MemoryError, match="generation publication failed"):
        pipeline.commit_generation(staged, drain)

    assert [event[0] for event in backend.events][-2:] == ["commit", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED
    assert pipeline.topology is topology


def test_generation_stage_async_exception_is_fail_stop():
    pipeline, backend = _pipeline()
    topology = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))

    def interrupted_stage(*, current, next_topology):
        backend.events.append(
            (
                "stage-posted",
                current.membership_generation,
                next_topology.membership_generation,
            )
        )
        raise KeyboardInterrupt("interrupted after stage publication")

    backend.stage_generation = interrupted_stage  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt, match="stage publication"):
        pipeline.stage_generation(topology)

    assert [event[0] for event in backend.events][-2:] == [
        "stage-posted",
        "fail_stop",
    ]
    assert pipeline.state == PipelineState.FAILED


def test_graceful_change_wrapper_allocation_failure_is_fail_stop():
    pipeline, backend = _pipeline()

    class ExhaustedGracefulChange(GracefulChange):
        def __new__(cls, *_arguments, **_keywords):
            del cls
            raise MemoryError("graceful wrapper allocation failed")

    with pytest.raises(MemoryError, match="graceful wrapper allocation failed"):
        ExhaustedGracefulChange.stage(pipeline, (0,))

    assert [event[0] for event in backend.events][-2:] == ["stage", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED


def test_graceful_change_interruption_after_drain_is_fail_stop():
    pipeline, backend = _pipeline()
    change = GracefulChange.stage(pipeline, (0,))

    def interrupted_commit(staged, drain):
        del staged, drain
        assert pipeline.state == PipelineState.DRAINING
        raise KeyboardInterrupt("interrupted before generation commit")

    pipeline.commit_generation = interrupted_commit  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt, match="before generation commit"):
        change.drain_and_commit(pipeline, stream="drain-stream")

    assert [event[0] for event in backend.events][-2:] == ["drain", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED


def test_full_layer_handoff_passes_device_metadata_to_external_expert():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()

    class Expert:
        def __init__(self):
            self.arguments = None

        def enqueue(self, **arguments):
            self.arguments = arguments
            return ExpertSubmission(arguments["output"], None)

    expert = Expert()
    output = _output()
    operation = pipeline.prepare_operation(x, routes, weights, output)
    communication_stream = FakeStream("stream-0")
    submission = enqueue_moe_layer(
        pipeline,
        expert,
        operation,
        communication_stream=communication_stream,
    )

    assert submission.output.shape == (4, 8)
    assert expert.arguments is not None
    assert expert.arguments["expert_input"].shape == (2, 12, 8)
    assert expert.arguments["expert_counts"].shape == (2,)
    assert expert.arguments["layout_ranges"].shape == (2, 3)
    assert expert.arguments["source_info"].shape == (2, 12, 4)
    assert expert.arguments["output"] is submission.handle.combine_buffer
    assert "input_ready_event" not in expert.arguments
    assert expert.arguments["handle"].state == HandleState.COMBINE_ENQUEUED
    assert backend.submission_validations == [
        ((communication_stream, communication_stream), 0, 1)
    ]
    output_consumer = FakeStream("output")
    bank_observer = FakeStream("bank")
    submission.enqueue_output_wait(output_consumer)
    submission.enqueue_bank_reuse_wait(bank_observer)
    assert output_consumer.waits == [
        FakeEvent(f"output-{submission.handle.operation.wire_value}")
    ]
    assert bank_observer.waits == [
        FakeEvent(f"reuse-{submission.handle.operation.wire_value}")
    ]
    assert [event[0] for event in backend.events] == [
        "preallocate",
        "prepare_operation",
        "dispatch",
        "combine",
    ]


@pytest.mark.parametrize(
    ("weights", "output", "message"),
    [
        (FakeTensor((4, 1)), _output(), "topk_weights must have shape"),
        (FakeTensor((4, 2)), FakeTensor((4, 7)), "output must have shape"),
    ],
)
def test_single_layer_invalid_weight_or_output_fails_in_cold_prepare_before_post(
    weights, output, message
):
    pipeline, backend = _pipeline()
    x, routes, _ = _inputs()

    with pytest.raises(ValueError, match=message):
        pipeline.prepare_operation(x, routes, weights, output)

    assert backend.submission_validations == []
    assert backend.events == [("preallocate", 0, 3)]
    assert pipeline.outstanding_handles == ()


def test_single_layer_rejects_invalid_executor_before_preflight_or_post():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    before = tuple(backend.events)

    with pytest.raises(TypeError, match="expert_executor must provide enqueue"):
        enqueue_moe_layer(
            pipeline,
            object(),  # type: ignore[arg-type]
            operation,
            communication_stream=FakeStream("comm"),
        )

    assert backend.submission_validations == []
    assert tuple(backend.events) == before
    assert pipeline.outstanding_handles == ()


@pytest.mark.parametrize(
    ("communication_stream", "expert_stream"),
    [
        (object(), FakeStream("expert")),
        (FakeStream("comm"), object()),
    ],
)
def test_single_layer_rejects_invalid_stream_before_preflight_or_post(
    communication_stream, expert_stream
):
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    operation = pipeline.prepare_operation(x, routes, weights, _output())

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], None)

    before = tuple(backend.events)
    with pytest.raises(TypeError, match="wait_event"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            operation,
            communication_stream=communication_stream,
            expert_stream=expert_stream,
        )

    assert backend.submission_validations == []
    assert tuple(backend.events) == before
    assert pipeline.outstanding_handles == ()


def test_single_layer_rejects_unprepared_operation_before_preflight_or_post():
    pipeline, backend = _pipeline()
    x, _routes, _weights = _inputs()

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], None)

    with pytest.raises(TypeError, match="prepared must be a PreparedOperation"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            x,  # type: ignore[arg-type]
            communication_stream=FakeStream("comm"),
        )

    assert backend.submission_validations == []
    assert backend.events == [("preallocate", 0, 3)]


def test_full_layer_cross_stream_edges_and_expert_failure_are_fail_safe():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()

    class CrossStreamExpert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("expert-done"))

    comm = FakeStream("comm")
    compute = FakeStream("compute")
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    submission = enqueue_moe_layer(
        pipeline,
        CrossStreamExpert(),
        operation,
        communication_stream=comm,
        expert_stream=compute,
        input_ready_event=FakeEvent("router-done"),
    )
    combine_event = [event for event in backend.events if event[0] == "combine"][-1]
    assert combine_event[-2:] == (FakeEvent("expert-done"), True)
    assert compute.waits == [
        FakeEvent(f"dispatch-{submission.handle.operation.wire_value}")
    ]
    output_consumer = FakeStream("output")
    submission.enqueue_output_wait(output_consumer)
    assert len(output_consumer.waits) == 1

    failed, failed_backend = _pipeline()

    class BrokenExpert:
        def enqueue(self, **_arguments):
            raise RuntimeError("expert launch is uncertain")

    failed_operation = failed.prepare_operation(x, routes, weights, _output())
    with pytest.raises(RuntimeError, match="uncertain"):
        enqueue_moe_layer(
            failed,
            BrokenExpert(),
            failed_operation,
            communication_stream=FakeStream("comm"),
        )
    assert failed.state == PipelineState.FAILED
    assert failed_backend.events[-1][0] == "fail_stop"


def test_distinct_equal_streams_preserve_both_cross_stream_dependencies():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    communication_stream = EqualityMaskingStream("comm")
    expert_stream = EqualityMaskingStream("expert")
    assert communication_stream == expert_stream
    assert communication_stream is not expert_stream
    ready_event = FakeEvent("expert-done")

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], ready_event)

    operation = pipeline.prepare_operation(x, routes, weights, _output())
    submission = enqueue_moe_layer(
        pipeline,
        Expert(),
        operation,
        communication_stream=communication_stream,
        expert_stream=expert_stream,
    )

    assert expert_stream.waits == [
        FakeEvent(f"dispatch-{submission.handle.operation.wire_value}")
    ]
    combine_event = [event for event in backend.events if event[0] == "combine"][-1]
    assert combine_event[6] is communication_stream
    assert combine_event[8] is ready_event


def test_distinct_equal_dispatch_and_combine_streams_still_require_event():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    dispatch_stream = EqualityMaskingStream("dispatch")
    combine_stream = EqualityMaskingStream("combine")
    assert dispatch_stream == combine_stream
    assert dispatch_stream is not combine_stream

    operation = pipeline.prepare_operation(x, routes, weights, _output())
    handle = pipeline.dispatch_prepared(operation, stream=dispatch_stream)
    before = tuple(backend.events)

    with pytest.raises(PipelineError, match="expert_ready_event"):
        pipeline.combine_prepared(
            handle.combine_buffer,
            handle,
            operation,
            stream=combine_stream,
        )

    assert tuple(backend.events) == before
    ready_event = FakeEvent("expert-done")
    pipeline.combine_prepared(
        handle.combine_buffer,
        handle,
        operation,
        stream=combine_stream,
        expert_ready_event=ready_event,
    )
    combine_event = backend.events[-1]
    assert combine_event[0] == "combine"
    assert combine_event[6] is combine_stream
    assert combine_event[8] is ready_event


def test_full_layer_wait_failure_after_dispatch_is_irreversible_fail_stop():
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()

    class BrokenWaitStream(FakeStream):
        def wait_event(self, event):
            del event
            raise RuntimeError("dispatch wait submission is uncertain")

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("expert-done"))

    operation = pipeline.prepare_operation(x, routes, weights, _output())
    with pytest.raises(RuntimeError, match="wait submission"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            operation,
            communication_stream=FakeStream("comm"),
            expert_stream=BrokenWaitStream("expert"),
        )

    assert pipeline.state == PipelineState.FAILED
    assert pipeline.failure is backend.events[-1][1]
    assert backend.events[-1][0] == "fail_stop"


def test_full_layer_interruption_immediately_after_dispatch_is_fail_stop(monkeypatch):
    pipeline, backend = _pipeline()
    x, routes, weights = _inputs()
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    original_dispatch = MoEPipeline.dispatch_prepared

    def dispatch_then_interrupt(self, *arguments, **keywords):
        original_dispatch(self, *arguments, **keywords)
        raise KeyboardInterrupt("interrupted after dispatch publication")

    monkeypatch.setattr(MoEPipeline, "dispatch_prepared", dispatch_then_interrupt)

    with pytest.raises(KeyboardInterrupt, match="after dispatch publication"):
        enqueue_moe_layer(
            pipeline,
            FastFakeExpert(),
            operation,
            communication_stream=FakeStream("comm"),
        )

    assert [event[0] for event in backend.events][-2:] == ["dispatch", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED


def test_fast_rolling_interruption_immediately_after_first_dispatch_is_fail_stop(
    monkeypatch,
):
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    original_dispatch = MoEPipeline._dispatch_preallocated

    def dispatch_then_interrupt(self, *arguments, **keywords):
        original_dispatch(self, *arguments, **keywords)
        raise KeyboardInterrupt("interrupted after rolling dispatch publication")

    monkeypatch.setattr(
        MoEPipeline,
        "_dispatch_preallocated",
        dispatch_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="rolling dispatch publication"):
        enqueue_rolling_moe_layers(
            pipeline,
            executor,
            schedule,
            communication_streams=communication,
            expert_streams=experts,
            submission_preflight=preflight,
        )

    assert [event[0] for event in backend.events][-2:] == ["dispatch", "fail_stop"]
    assert pipeline.state == PipelineState.FAILED


@pytest.mark.parametrize("prepared_adapter", [False, True])
def test_rolling_terminal_lease_publication_failure_is_fail_stop(
    monkeypatch, prepared_adapter
):
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )

    class GenericExpert:
        def enqueue(self, **arguments):
            step = arguments["handle"].operation.step
            return ExpertSubmission(arguments["output"], FakeEvent(f"expert-{step}"))

    executor = FastFakeExpert() if prepared_adapter else GenericExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )

    class InterruptedTerminalPublication:
        def __get__(self, _instance, _owner):
            raise KeyboardInterrupt("interrupted while publishing terminal leases")

    monkeypatch.setattr(
        RollingSubmissionPreflight,
        "_terminal_submissions",
        InterruptedTerminalPublication(),
    )

    with pytest.raises(KeyboardInterrupt, match="publishing terminal leases"):
        enqueue_rolling_moe_layers(
            pipeline,
            executor,
            schedule,
            communication_streams=communication,
            expert_streams=experts,
            submission_preflight=preflight,
        )

    assert backend.events[-1][0] == "fail_stop"
    assert pipeline.state == PipelineState.FAILED


def test_fast_rolling_outer_call_completion_interrupt_is_fail_stop(monkeypatch):
    """Protect the caller after the nested helper has posted both combines."""

    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )

    original_fast_submit = elastic_module._enqueue_preflighted_fast_rolling

    def submit_then_interrupt(*arguments, **keywords):
        original_fast_submit(*arguments, **keywords)
        raise KeyboardInterrupt("interrupted after nested terminal completion")

    monkeypatch.setattr(
        elastic_module,
        "_enqueue_preflighted_fast_rolling",
        submit_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="nested terminal completion"):
        enqueue_rolling_moe_layers(
            pipeline,
            executor,
            schedule,
            communication_streams=communication,
            expert_streams=experts,
            submission_preflight=preflight,
        )

    assert backend.events[-1][0] == "fail_stop"
    assert pipeline.state == PipelineState.FAILED


def test_post_enqueue_fail_stop_path_constructs_no_state_container():
    helper_source = textwrap.dedent(
        inspect.getsource(elastic_module._fail_stop_preserving_original)
    )
    public_source = textwrap.dedent(inspect.getsource(MoEPipeline.fail_stop))
    for source in (helper_source, public_source):
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set))
            for node in ast.walk(tree)
        )
    assert "pipeline._enter_failed(error)" in helper_source
    assert "pipeline.fail_stop(error)" not in helper_source


def test_two_bank_invalid_second_operation_fails_during_bind_before_d0():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, _route1, weight1 = _inputs()

    with pytest.raises(ValueError, match="topk_indices must have shape"):
        _bound_two_bank(
            pipeline,
            LayerInputs(x0, route0, weight0, _output()),
            LayerInputs(x1, FakeTensor((4, 1)), weight1, _output()),
        )

    assert [event[0] for event in backend.events] == [
        "preallocate",
        "prepare_operation",
    ]
    assert pipeline.state == PipelineState.ACTIVE
    assert pipeline.outstanding_handles == ()


@pytest.mark.parametrize(
    ("second_weights", "second_output", "message"),
    [
        (FakeTensor((4, 1)), _output(), "topk_weights must have shape"),
        (FakeTensor((4, 2)), FakeTensor((4, 7)), "output must have shape"),
    ],
)
def test_two_bank_invalid_later_weight_or_output_never_posts_device_work(
    second_weights, second_output, message
):
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, _ = _inputs()

    with pytest.raises(ValueError, match=message):
        _bound_two_bank(
            pipeline,
            LayerInputs(x0, route0, weight0, _output()),
            LayerInputs(x1, route1, second_weights, second_output),
        )

    assert not any(event[0] in ("dispatch", "combine") for event in backend.events)
    assert pipeline.state == PipelineState.ACTIVE
    assert pipeline.outstanding_handles == ()


def test_two_bank_alias_preflight_runs_before_dispatch_and_read_sharing_is_safe():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    shared_output = _output()

    with pytest.raises(ValueError, match="first.output and second.output"):
        prepare_two_bank_window(
            LayerInputs(x0, route0, weight0, shared_output),
            LayerInputs(x1, route1, weight1, shared_output),
        )

    with pytest.raises(ValueError, match="first.output and second.activations"):
        _bound_two_bank(
            pipeline,
            LayerInputs(x0, route0, weight0, x1),
            LayerInputs(x1, route1, weight1, _output()),
        )
    assert backend.events == [("preallocate", 0, 3)]
    assert pipeline.state == PipelineState.ACTIVE

    # Read/read sharing is legal: only concurrently written outputs need to be
    # disjoint from both operations' immutable inputs.
    prepared = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x0, route0, weight0, _output()),
    )
    assert isinstance(prepared, PreparedTwoBankWindow)


def test_bound_two_bank_window_keeps_pointer_queries_out_of_hot_submission():
    pipeline, _backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    prepared = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    bound = bind_two_bank_window(pipeline, prepared)

    for inputs in (prepared.first, prepared.second):
        for tensor in (
            inputs.activations,
            inputs.topk_indices,
            inputs.topk_weights,
            inputs.output,
        ):
            tensor.data_ptr = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
                AssertionError("hot path repeated pointer introspection")
            )

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("expert-done"))

    submissions = enqueue_two_bank_window(
        pipeline,
        Expert(),
        bound,
        communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
        expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
    )
    assert len(submissions) == 2


def test_two_bank_submission_rejects_unbound_window_before_preflight_or_post():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )

    with pytest.raises(TypeError, match="bind_two_bank_window"):
        enqueue_two_bank_window(
            pipeline,
            type("Expert", (), {"enqueue": lambda self, **_: None})(),
            window,  # type: ignore[arg-type]
            communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
            expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
        )

    assert backend.submission_validations == []
    assert backend.events == [("preallocate", 0, 3)]


def test_two_bank_window_rejects_bad_stream_or_partial_boundary_before_posting():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("ready"))

    good = FakeStream("good")
    before = tuple(backend.events)
    with pytest.raises(TypeError, match="wait_event"):
        enqueue_two_bank_window(
            pipeline,
            Expert(),
            window,
            communication_streams=(good, good),
            expert_streams=(good, object()),
        )
    assert tuple(backend.events) == before

    pipeline.dispatch(x0, route0, stream=good)
    after_dispatch = tuple(backend.events)
    with pytest.raises(ValueError, match="complete two-bank boundary"):
        enqueue_two_bank_window(
            pipeline,
            Expert(),
            window,
            communication_streams=(good, good),
            expert_streams=(good, good),
        )
    assert tuple(backend.events) == after_dispatch


def test_rolling_scheduler_preserves_overlap_without_communication_abba():
    pipeline, backend = _pipeline()

    class TracedExpert:
        def enqueue(self, **arguments):
            handle = arguments["handle"]
            backend.events.append(("expert", handle.operation, arguments["stream"]))
            return ExpertSubmission(
                arguments["output"], FakeEvent(f"expert-{handle.operation.step}")
            )

    layers = []
    for _operation in range(4):
        x, routes, weights = _inputs()
        layers.append(LayerInputs(x, routes, weights, _output()))
    prepared_schedule = prepare_rolling_schedule(
        (
            prepare_two_bank_window(layers[0], layers[1]),
            prepare_two_bank_window(layers[2], layers[3]),
        )
    )
    assert isinstance(prepared_schedule, PreparedRollingSchedule)
    schedule = bind_rolling_schedule(pipeline, prepared_schedule)
    assert isinstance(schedule, BoundRollingSchedule)
    comm = (FakeStream("comm-0"), FakeStream("comm-1"))
    expert = (FakeStream("expert-0"), FakeStream("expert-1"))
    submissions = enqueue_rolling_moe_layers(
        pipeline,
        TracedExpert(),
        schedule,
        communication_streams=comm,
        expert_streams=expert,
    )
    assert backend.submission_validations == [(comm + expert, 0, 4)]
    assert [
        (context.streams, context.first_step, context.step_count)
        for context in backend.submission_preparations
    ] == [(comm + expert, 0, 4)]

    assert [event[0] for event in backend.events] == [
        "preallocate",
        "prepare_operation",
        "prepare_operation",
        "prepare_operation",
        "prepare_operation",
        "dispatch",
        "expert",
        "dispatch",
        "expert",
        "combine",
        "dispatch",
        "expert",
        "combine",
        "dispatch",
        "expert",
        "combine",
        "combine",
    ]
    communication = [
        event for event in backend.events if event[0] in ("dispatch", "combine")
    ]
    assert [(event[0], event[1].step) for event in communication] == [
        ("dispatch", 0),
        ("dispatch", 1),
        ("combine", 0),
        ("dispatch", 2),
        ("combine", 1),
        ("dispatch", 3),
        ("combine", 2),
        ("combine", 3),
    ]
    assert communication[0][7] is None
    assert communication[1][7] == FakeEvent(
        f"dispatch-{communication[0][1].wire_value}"
    )
    assert communication[2][7] == FakeEvent(
        f"dispatch-{communication[1][1].wire_value}"
    )
    assert communication[3][7] is None  # C0 -> D2 is same-stream FIFO.
    assert communication[4][7] == FakeEvent(
        f"dispatch-{communication[3][1].wire_value}"
    )
    assert communication[5][7] is None  # C1 -> D3 is same-stream FIFO.
    assert communication[6][7] == FakeEvent(
        f"dispatch-{communication[5][1].wire_value}"
    )
    assert communication[7][7] == FakeEvent(f"reuse-{communication[6][1].wire_value}")
    assert [submission.handle.operation.step for submission in submissions] == [2, 3]


def test_prelinked_fast_rolling_preserves_d0_d1_c0_d2_order():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    schedule = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule((window,), repeat_count=3),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )

    enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=preflight,
    )

    communication_events = [
        event for event in backend.events if event[0] in ("dispatch", "combine")
    ]
    assert [(event[0], event[1].step) for event in communication_events] == [
        ("dispatch", 0),
        ("dispatch", 1),
        ("combine", 0),
        ("dispatch", 2),
        ("combine", 1),
        ("dispatch", 3),
        ("combine", 2),
        ("dispatch", 4),
        ("combine", 3),
        ("dispatch", 5),
        ("combine", 4),
        ("combine", 5),
    ]
    assert executor.steps == [0, 1, 2, 3, 4, 5]


def test_fast_rolling_cold_binds_expert_wait_and_operands(monkeypatch):
    class GuardedWaitLookupStream(FakeStream):
        def __init__(self, name: str) -> None:
            object.__setattr__(self, "reject_wait_lookup", False)
            object.__setattr__(self, "wait_lookup_count", 0)
            super().__init__(name)

        def __getattribute__(self, name: str):
            if name == "wait_event":
                count = object.__getattribute__(self, "wait_lookup_count")
                object.__setattr__(self, "wait_lookup_count", count + 1)
                if object.__getattribute__(self, "reject_wait_lookup"):
                    raise AssertionError("hot path looked up stream.wait_event")
            return object.__getattribute__(self, name)

    class ColdOperandExpert:
        def __init__(self) -> None:
            self.calls = []

        def enqueue(self, **_arguments):
            raise AssertionError("prepared rolling path called generic expert enqueue")

        def enqueue_prepared(self, **arguments):
            self.calls.append(arguments)
            return FakeEvent("expert-ready")

    pipeline, _backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule(
            (
                prepare_two_bank_window(
                    LayerInputs(x0, route0, weight0, _output()),
                    LayerInputs(x1, route1, weight1, _output()),
                ),
            ),
            repeat_count=2,
        ),
    )
    executor = ColdOperandExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (
        GuardedWaitLookupStream("expert-0"),
        GuardedWaitLookupStream("expert-1"),
    )
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    slots = preflight._expert_slots
    lookup_counts = tuple(stream.wait_lookup_count for stream in experts)
    for stream in experts:
        object.__setattr__(stream, "reject_wait_lookup", True)

    def reject_property(_self):
        raise AssertionError("hot path recovered an expert operand through a property")

    for name in (
        "expert_input",
        "expert_counts",
        "source_info",
        "layout_ranges",
        "combine_buffer",
    ):
        monkeypatch.setattr(
            pipeline_module.DispatchHandle, name, property(reject_property)
        )

    enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=preflight,
    )

    assert tuple(stream.wait_lookup_count for stream in experts) == lookup_counts
    assert len(executor.calls) == len(slots)
    for call, slot in zip(executor.calls, slots):
        assert call["expert_input"] is slot.expert_input
        assert call["expert_counts"] is slot.expert_counts
        assert call["source_info"] is slot.source_info
        assert call["layout_ranges"] is slot.layout_ranges
        assert call["output"] is slot.output
        assert call["handle"] is slot.handle
        assert call["stream"] is slot.expert_stream
    frames = preflight._pipeline_preflight._frames
    assert experts[0].waits == [
        frames[index].handle._dispatch_ready_event for index in (0, 2)
    ]
    assert experts[1].waits == [
        frames[index].handle._dispatch_ready_event for index in (1, 3)
    ]


def test_rolling_submission_rejects_unbound_schedule_before_preflight_or_post():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = prepare_rolling_schedule(
        (
            prepare_two_bank_window(
                LayerInputs(x0, route0, weight0, _output()),
                LayerInputs(x1, route1, weight1, _output()),
            ),
        )
    )
    before = tuple(backend.events)

    with pytest.raises(TypeError, match="BoundRollingSchedule"):
        enqueue_rolling_moe_layers(
            pipeline,
            type("Expert", (), {"enqueue": lambda self, **_: None})(),
            schedule,  # type: ignore[arg-type]
            communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
            expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
        )

    assert backend.submission_validations == []
    assert tuple(backend.events) == before


def test_schedule_context_rejection_precedes_every_device_post():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule(
            (
                prepare_two_bank_window(
                    LayerInputs(x0, route0, weight0, _output()),
                    LayerInputs(x1, route1, weight1, _output()),
                ),
            )
        ),
    )
    streams = (
        FakeStream("comm-0"),
        FakeStream("comm-1"),
        FakeStream("expert-0"),
        FakeStream("expert-1"),
    )
    backend.submission_validation_error = PipelineError(
        "CUDA Graph capture is unsupported"
    )
    before = tuple(backend.events)

    with pytest.raises(PipelineError, match="capture is unsupported"):
        enqueue_rolling_moe_layers(
            pipeline,
            type("Expert", (), {"enqueue": lambda self, **_: None})(),
            schedule,
            communication_streams=streams[:2],
            expert_streams=streams[2:],
        )

    assert backend.submission_validations == [(streams, 0, 2)]
    assert tuple(backend.events) == before
    assert pipeline.state == PipelineState.ACTIVE


def test_submission_preflight_token_is_one_shot_and_skips_repeat_query():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    streams = (
        FakeStream("comm-0"),
        FakeStream("comm-1"),
        FakeStream("expert-0"),
        FakeStream("expert-1"),
    )

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("ready"))

    executor = Expert()
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        window,
        communication_streams=streams[:2],
        expert_streams=streams[2:],
    )
    assert isinstance(preflight, RollingSubmissionPreflight)
    assert repr(preflight) == "<RollingSubmissionPreflight opaque>"
    assert backend.submission_validations == [(streams, 0, 2)]

    enqueue_two_bank_window(
        pipeline,
        executor,
        window,
        communication_streams=streams[:2],
        expert_streams=streams[2:],
        submission_preflight=preflight,
    )
    assert backend.submission_validations == [(streams, 0, 2)]
    before_reuse = tuple(backend.events)
    with pytest.raises(PipelineError, match="already consumed"):
        enqueue_two_bank_window(
            pipeline,
            executor,
            window,
            communication_streams=streams[:2],
            expert_streams=streams[2:],
            submission_preflight=preflight,
        )
    assert tuple(backend.events) == before_reuse


def test_submission_preflight_is_consumed_before_any_admitted_stream_wait():
    pipeline, _backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    order = []

    class OrderedStream(FakeStream):
        def wait_event(self, event):
            order.append(("wait", self.name, event))
            super().wait_event(event)

    communication = (OrderedStream("comm-0"), OrderedStream("comm-1"))
    expert = (FakeStream("expert-0"), FakeStream("expert-1"))

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("expert-ready"))

    executor = Expert()
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=expert,
    )
    original_consume = pipeline._consume_submission_preflight

    def traced_consume(*args, **kwargs):
        order.append(("consume",))
        return original_consume(*args, **kwargs)

    pipeline._consume_submission_preflight = traced_consume  # type: ignore[method-assign]

    ready = FakeEvent("schedule-ready")
    enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=expert,
        submission_preflight=preflight,
        submission_ready_event=ready,
    )

    assert order[:3] == [
        ("consume",),
        ("wait", "comm-0", ready),
        ("wait", "comm-1", ready),
    ]


def test_pipeline_rejects_noncurrent_preflight_range_before_backend_query():
    pipeline, backend = _pipeline()

    with pytest.raises(PipelineError, match="pipeline's next operation step"):
        pipeline.validate_submission_context(
            streams=(FakeStream("comm"),), first_step=1, step_count=1
        )

    assert backend.submission_validations == []
    assert backend.events == [("preallocate", 0, 3)]


def test_preflight_contract_explicitly_disclaims_stream_locking():
    token_contract = SubmissionPreflight.__doc__ or ""
    pipeline_contract = MoEPipeline.validate_submission_context.__doc__ or ""
    combined = token_contract + pipeline_contract

    assert "cannot lock a CUDA stream" in token_contract
    assert "exclusive single-host-submitter ownership" in combined
    assert "no capture or intervening CUDA operation" in combined


def test_submission_preflight_rejects_wrong_range_stream_pipeline_and_generation():
    x, routes, weights = _inputs()

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("ready"))

    # A two-operation proof cannot attest a one-operation helper.
    pipeline, backend = _pipeline()
    comm = FakeStream("comm")
    expert = FakeStream("expert")
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    wrong_range = pipeline.validate_submission_context(
        streams=(comm, expert), first_step=0, step_count=2
    )
    before = tuple(backend.events)
    with pytest.raises(PipelineError, match="operation range does not match"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            operation,
            communication_stream=comm,
            expert_stream=expert,
            submission_preflight=wrong_range,
        )
    assert tuple(backend.events) == before

    # Stream objects and order are identity-bound.
    pipeline, backend = _pipeline()
    exact_streams = (FakeStream("comm"), FakeStream("expert"))
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    wrong_stream = pipeline.validate_submission_context(
        streams=exact_streams, first_step=0, step_count=1
    )
    before = tuple(backend.events)
    with pytest.raises(PipelineError, match="stream sequence does not match"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            operation,
            communication_stream=exact_streams[0],
            expert_stream=FakeStream("expert"),
            submission_preflight=wrong_stream,
        )
    assert tuple(backend.events) == before

    # Ownership is not transferable between otherwise identical pipelines.
    owner, _ = _pipeline()
    other, other_backend = _pipeline()
    other_operation = other.prepare_operation(x, routes, weights, _output())
    cross_pipeline = owner.validate_submission_context(
        streams=exact_streams, first_step=0, step_count=1
    )
    before = tuple(other_backend.events)
    with pytest.raises(PipelineError, match="belongs to another pipeline"):
        enqueue_moe_layer(
            other,
            Expert(),
            other_operation,
            communication_stream=exact_streams[0],
            expert_stream=exact_streams[1],
            submission_preflight=cross_pipeline,
        )
    assert tuple(other_backend.events) == before

    # A generation commit resets step zero but cannot revive an old proof.
    pipeline, backend = _pipeline()
    operation = pipeline.prepare_operation(x, routes, weights, _output())
    stale = pipeline.validate_submission_context(
        streams=exact_streams, first_step=0, step_count=1
    )
    staged = pipeline.stage_generation(
        _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    )
    drain = pipeline.begin_generation_drain(staged, stream="drain")
    pipeline.commit_generation(staged, drain)
    before = tuple(backend.events)
    with pytest.raises(PipelineError, match="stale membership generation"):
        enqueue_moe_layer(
            pipeline,
            Expert(),
            operation,
            communication_stream=exact_streams[0],
            expert_stream=exact_streams[1],
            submission_preflight=stale,
        )
    assert tuple(backend.events) == before


def test_rolling_schedule_reuses_a_finite_storage_cycle_without_reinspection():
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    for inputs in (window.first, window.second):
        for tensor in (
            inputs.activations,
            inputs.topk_indices,
            inputs.topk_weights,
            inputs.output,
        ):
            tensor.data_ptr = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
                AssertionError("cycle preparation repeated pointer introspection")
            )

    schedule = prepare_rolling_schedule((window,), repeat_count=1_000_000)
    assert schedule.operation_count == 2_000_000
    assert len(schedule._inputs) == 2
    assert schedule._input_for_operation(999_998) is window.first
    assert schedule._input_for_operation(999_999) is window.second


def test_bind_rejects_storage_drift_before_creating_any_backend_token():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    first_output = _output()
    second_output = _output()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, first_output),
        LayerInputs(x1, route1, weight1, second_output),
    )
    schedule = prepare_rolling_schedule((window,))
    # Model Tensor.set_() introducing an opposite-bank output alias without
    # changing the Python tensor object that the prepared window owns.
    second_output._pointer = first_output._pointer
    before = tuple(backend.events)

    with pytest.raises(ValueError, match="storage changed; rebuild"):
        bind_rolling_schedule(pipeline, schedule)

    assert tuple(backend.events) == before
    assert all(event[0] != "prepare_operation" for event in backend.events)


def test_bound_rolling_schedule_uses_opaque_tokens_without_tensor_metadata():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    bound = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule((window,), repeat_count=2),
    )
    assert isinstance(bound, BoundRollingSchedule)
    tokens = tuple(
        event[-1] for event in backend.events if event[0] == "prepare_operation"
    )
    assert len(tokens) == 2

    class PoisonShape:
        def __iter__(self):
            raise AssertionError("prepared hot path inspected tensor.shape")

    for inputs in (window.first, window.second):
        for tensor in (
            inputs.activations,
            inputs.topk_indices,
            inputs.topk_weights,
            inputs.output,
        ):
            tensor.shape = PoisonShape()

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("ready"))

    enqueue_rolling_moe_layers(
        pipeline,
        Expert(),
        bound,
        communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
        expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
    )
    assert backend.prepared_tokens == [
        tokens[0],
        tokens[1],
        tokens[0],
        tokens[0],
        tokens[1],
        tokens[1],
        tokens[0],
        tokens[1],
    ]


def test_rolling_scheduler_preflights_boundary_and_all_timing_events():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule(
            (
                prepare_two_bank_window(
                    LayerInputs(x0, route0, weight0, _output()),
                    LayerInputs(x1, route1, weight1, _output()),
                ),
            )
        ),
    )
    comm = (FakeStream("comm-0"), FakeStream("comm-1"))
    expert = (FakeStream("expert-0"), FakeStream("expert-1"))

    class Expert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], FakeEvent("ready"))

    class RecordEvent:
        def record(self, _stream):
            return None

    before = tuple(backend.events)
    with pytest.raises(TypeError, match=r"operation_events\[1\]"):
        enqueue_rolling_moe_layers(
            pipeline,
            Expert(),
            schedule,
            communication_streams=comm,
            expert_streams=expert,
            operation_events=((RecordEvent(), RecordEvent()), (object(), object())),
        )
    assert tuple(backend.events) == before

    pipeline.dispatch(x0, route0, stream=comm[0])
    after_dispatch = tuple(backend.events)
    with pytest.raises(ValueError, match="complete two-bank boundary"):
        enqueue_rolling_moe_layers(
            pipeline,
            Expert(),
            schedule,
            communication_streams=comm,
            expert_streams=expert,
        )
    assert tuple(backend.events) == after_dispatch


def test_two_bank_window_submits_next_dispatch_before_first_combine_wait():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()

    class TracedExpert:
        def enqueue(self, **arguments):
            handle = arguments["handle"]
            ready = FakeEvent(f"expert-{handle.operation.step}")
            backend.events.append(
                (
                    "expert",
                    handle.operation,
                    arguments["stream"],
                    ready,
                )
            )
            return ExpertSubmission(arguments["output"], ready)

    comm_streams = (FakeStream("comm-0"), FakeStream("comm-1"))
    expert_streams = (FakeStream("expert-stream-0"), FakeStream("expert-stream-1"))
    bound = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output(), FakeEvent("router-0")),
        LayerInputs(x1, route1, weight1, _output(), FakeEvent("router-1")),
    )
    submissions = enqueue_two_bank_window(
        pipeline,
        TracedExpert(),
        bound,
        communication_streams=comm_streams,
        expert_streams=expert_streams,
    )

    runtime_events = [
        event
        for event in backend.events
        if event[0] in ("dispatch", "expert", "combine")
    ]
    assert [event[0] for event in runtime_events] == [
        "dispatch",
        "expert",
        "dispatch",
        "expert",
        "combine",
        "combine",
    ]
    assert [submission.handle.bank for submission in submissions] == [0, 1]
    assert expert_streams[0].waits == [
        FakeEvent(f"dispatch-{submissions[0].handle.operation.wire_value}")
    ]
    assert expert_streams[1].waits == [
        FakeEvent(f"dispatch-{submissions[1].handle.operation.wire_value}")
    ]
    assert runtime_events[4][-2:] == (FakeEvent("expert-0"), True)
    assert runtime_events[5][-2:] == (FakeEvent("expert-1"), True)
    # Every peer-polling cooperative communication kernel has one global
    # cross-stream predecessor. This forbids rank-divergent D0/D1 or C0/C1
    # residency while retaining D1/expert-0 and C0/expert-1 overlap.
    assert runtime_events[0][7] is None
    assert runtime_events[2][7] == FakeEvent(
        f"dispatch-{submissions[0].handle.operation.wire_value}"
    )
    assert runtime_events[4][7] == FakeEvent(
        f"dispatch-{submissions[1].handle.operation.wire_value}"
    )
    assert runtime_events[5][7] == FakeEvent(
        f"reuse-{submissions[0].handle.operation.wire_value}"
    )

    # The next window starts behind C1 even when it selects fresh stream
    # objects, so the total order extends across helper invocations.
    next_bound = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    enqueue_two_bank_window(
        pipeline,
        TracedExpert(),
        next_bound,
        communication_streams=(FakeStream("next-comm-0"), FakeStream("next-comm-1")),
        expert_streams=(FakeStream("next-expert-0"), FakeStream("next-expert-1")),
    )
    next_dispatch = [event for event in backend.events if event[0] == "dispatch"][-2]
    assert next_dispatch[7] == FakeEvent(
        f"reuse-{submissions[1].handle.operation.wire_value}"
    )


def test_elastic_helpers_distinguish_live_standby_from_restarted_process():
    current = _topology(generation=4, active=(0,), incarnations=(5, 7, 0))
    live_rejoin = next_topology(current, (0, 1))
    assert live_rejoin.membership_generation == 5
    assert live_rejoin.incarnation(1) == 7

    restarted = next_topology(current, (0, 1), restarted_ranks=(1,))
    assert restarted.incarnation(1) == 8
    started_for_first_time = next_topology(current, (0, 2), restarted_ranks=(2,))
    assert started_for_first_time.incarnation(2) == 1
    with pytest.raises(ValueError, match="must be active"):
        next_topology(current, (0,), restarted_ranks=(1,))
    with pytest.raises(ValueError, match="no process incarnation"):
        next_topology(current, (0, 2))


def test_control_plane_plan_keeps_sparse_ids_and_masks():
    membership = parse_membership("0,1;3,0,2;0,3;0,1,2,3", max_ranks=4)
    assert membership == ((0, 1), (0, 2, 3), (0, 3), (0, 1, 2, 3))
    plan = topology_plan(membership, max_ranks=4, experts_per_rank=2)
    assert [topology.membership_generation for topology in plan] == [0, 1, 2, 3]
    assert plan[2].nixl_mask == (0, 1, 1, 0)
    assert plan[2].active_experts == (0, 1, 6, 7)
    assert all(topology.rank_incarnations == (1, 1, 1, 1) for topology in plan)

    with pytest.raises(ValueError, match="duplicates"):
        parse_membership("0,0", max_ranks=2)
    with pytest.raises(ValueError, match="exceeds"):
        parse_membership("0,2", max_ranks=2)


def test_graceful_change_helper_only_commits_after_layer_handle_closes():
    pipeline, _ = _pipeline()
    change = GracefulChange.stage(pipeline, (0,))
    x, routes, _ = _inputs()
    pipeline.dispatch(x, routes, stream="stream-0")
    with pytest.raises(PipelineError, match="every dispatch handle"):
        change.drain_and_commit(pipeline, stream="stream-0")


def test_preflighted_fast_loop_constructs_no_semantic_object_after_first_post(
    monkeypatch,
):
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )

    original_dispatch = backend.enqueue_dispatch
    armed = True

    def poison(*_args, **_kwargs):
        raise AssertionError("semantic constructor ran after the first device post")

    def dispatch_then_poison(**arguments):
        nonlocal armed
        result = original_dispatch(**arguments)
        if armed:
            armed = False
            for cls in (
                pipeline_module.OperationEpoch,
                pipeline_module.DispatchHandle,
                pipeline_module.PreparedOperation,
                pipeline_module._PreparedSubmissionFrame,
                elastic_module.ExpertSubmission,
                elastic_module.LayerSubmission,
                elastic_module._ExpertSubmissionSlot,
                elastic_module.RollingSubmissionPreflight,
            ):
                monkeypatch.setattr(cls, "__init__", poison)
        return result

    backend.enqueue_dispatch = dispatch_then_poison  # type: ignore[method-assign]
    submissions = enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=preflight,
    )

    assert armed is False
    assert submissions is preflight._terminal_submissions
    assert executor.steps == [0, 1]


def test_each_fast_plan_returns_unique_terminal_leases_that_never_revive():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))

    first_plan = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    first = enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=first_plan,
    )
    second_plan = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    second = enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=second_plan,
    )

    assert [
        (context.first_step, context.step_count)
        for context in backend.submission_preparations
    ] == [(0, 2), (2, 2)]
    assert first[0] is not second[0]
    assert first[1] is not second[1]
    assert first[0].handle is not second[0].handle
    assert first[1].handle is not second[1].handle
    assert [submission.handle.operation.step for submission in first] == [0, 1]
    assert [submission.handle.operation.step for submission in second] == [2, 3]
    for stale in first:
        with pytest.raises(PipelineError, match="lease expired"):
            stale.enqueue_output_wait(FakeStream("stale-output"))
        with pytest.raises(PipelineError, match="lease expired"):
            stale.enqueue_bank_reuse_wait(FakeStream("stale-bank"))
    for current in second:
        current.enqueue_output_wait(FakeStream("current-output"))
        current.enqueue_bank_reuse_wait(FakeStream("current-bank"))


def test_rolling_cold_plan_cap_rejects_before_expansion_or_backend_admission():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    window = prepare_two_bank_window(
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    schedule = bind_rolling_schedule(
        pipeline,
        prepare_rolling_schedule(
            (window,),
            repeat_count=(MAX_PREPARED_SUBMISSION_OPERATIONS + 2) // 2,
        ),
    )
    before = tuple(backend.events)

    with pytest.raises(ValueError, match="cold-plan capacity"):
        preflight_rolling_moe_layers(
            pipeline,
            FastFakeExpert(),
            schedule,
            communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
            expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
        )

    assert tuple(backend.events) == before
    assert backend.submission_validations == []
    assert pipeline.state == PipelineState.ACTIVE


def test_materialization_failure_precedes_capture_admission_and_device_post(
    monkeypatch,
):
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    before = tuple(backend.events)

    def fail_handle_allocation(self, **_kwargs):
        del self
        raise MemoryError("injected cold handle allocation failure")

    monkeypatch.setattr(
        pipeline_module.DispatchHandle,
        "__init__",
        fail_handle_allocation,
    )
    with pytest.raises(MemoryError, match="cold handle"):
        preflight_rolling_moe_layers(
            pipeline,
            FastFakeExpert(),
            schedule,
            communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
            expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
        )

    assert tuple(backend.events) == before
    assert backend.submission_validations == []
    assert pipeline.state == PipelineState.ACTIVE


def test_fast_preflight_rejects_backend_without_cold_context_before_post():
    pipeline, backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    before = tuple(backend.events)
    backend.prepare_submission_context = lambda **_kwargs: None  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="must return an opaque context"):
        preflight_rolling_moe_layers(
            pipeline,
            FastFakeExpert(),
            schedule,
            communication_streams=(FakeStream("comm-0"), FakeStream("comm-1")),
            expert_streams=(FakeStream("expert-0"), FakeStream("expert-1")),
        )

    assert backend.submission_validations == []
    assert tuple(backend.events) == before
    assert pipeline.state == PipelineState.ACTIVE


def test_preflight_epochs_rebase_exactly_across_graceful_generation_commit():
    pipeline, _backend = _pipeline()
    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    generation_three = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    assert [
        (frame.handle.operation.membership_generation, frame.handle.operation.step)
        for frame in generation_three._pipeline_preflight._frames
    ] == [(3, 0), (3, 1)]

    staged = pipeline.stage_generation(_topology(generation=4))
    enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=generation_three,
    )
    drain = pipeline.begin_generation_drain(staged, stream=FakeStream("drain"))
    pipeline.commit_generation(staged, drain)

    generation_four = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    assert [
        (frame.handle.operation.membership_generation, frame.handle.operation.step)
        for frame in generation_four._pipeline_preflight._frames
    ] == [(4, 0), (4, 1)]


def test_rolling_frames_select_physical_banks_from_odd_absolute_first_step():
    pipeline, backend = _pipeline()
    seed_x, seed_route, seed_weight = _inputs()
    seed = pipeline.prepare_operation(
        seed_x,
        seed_route,
        seed_weight,
        _output(),
    )

    class SeedExpert:
        def enqueue(self, **arguments):
            return ExpertSubmission(arguments["output"], None)

    seed_stream = FakeStream("seed")
    enqueue_moe_layer(
        pipeline,
        SeedExpert(),
        seed,
        communication_stream=seed_stream,
        expert_stream=seed_stream,
    )
    assert pipeline.next_operation_step == 1

    x0, route0, weight0 = _inputs()
    x1, route1, weight1 = _inputs()
    schedule = _bound_two_bank(
        pipeline,
        LayerInputs(x0, route0, weight0, _output()),
        LayerInputs(x1, route1, weight1, _output()),
    )
    executor = FastFakeExpert()
    communication = (FakeStream("comm-0"), FakeStream("comm-1"))
    experts = (FakeStream("expert-0"), FakeStream("expert-1"))
    preflight = preflight_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
    )
    frames = preflight._pipeline_preflight._frames
    assert [frame.handle.operation.step for frame in frames] == [1, 2]
    assert [frame.handle.bank for frame in frames] == [1, 0]
    assert [frame.handle._dispatch_stream for frame in frames] == [
        communication[1],
        communication[0],
    ]

    submissions = enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication,
        expert_streams=experts,
        submission_preflight=preflight,
    )
    assert [submission.handle.bank for submission in submissions] == [0, 1]
    assert [submission.handle.operation.step for submission in submissions] == [2, 1]
    dispatches = [event for event in backend.events if event[0] == "dispatch"][-2:]
    assert [(event[1].step, event[6]) for event in dispatches] == [
        (1, communication[1]),
        (2, communication[0]),
    ]


def test_fast_rolling_source_contains_no_hot_semantic_constructor():
    hot_functions = (
        elastic_module._enqueue_preflighted_fast_rolling,
        elastic_module._enqueue_preallocated_expert,
        pipeline_module.MoEPipeline._dispatch_preallocated,
        pipeline_module.MoEPipeline._combine_preallocated,
        pipeline_module.MoEPipeline._validate_preallocated_handle,
    )
    sources = "\n".join(inspect.getsource(function) for function in hot_functions)
    for constructor in (
        "OperationEpoch(",
        "DispatchHandle(",
        "PreparedOperation(",
        "ExpertSubmission(",
        "LayerSubmission(",
        "_PreparedSubmissionFrame(",
        "_ExpertSubmissionSlot(",
    ):
        assert constructor not in sources
    for function in hot_functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        definition = tree.body[0]
        definition.returns = None
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        ):
            argument.annotation = None
        assert not any(
            isinstance(
                node,
                (
                    ast.List,
                    ast.ListComp,
                    ast.Dict,
                    ast.DictComp,
                    ast.Set,
                    ast.SetComp,
                    ast.Tuple,
                ),
            )
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
            for node in ast.walk(tree)
        )
        forbidden_calls = {"getattr", "id", "range", "list", "dict", "tuple"}
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not calls & forbidden_calls
    assert "operation.bank" not in inspect.getsource(
        pipeline_module.MoEPipeline._dispatch_preallocated
    )
    assert "handle.bank" not in inspect.getsource(
        pipeline_module.MoEPipeline._combine_preallocated
    )
    expert_source = inspect.getsource(elastic_module._enqueue_preallocated_expert)
    for property_name in (
        "handle.expert_input",
        "handle.expert_counts",
        "handle.source_info",
        "handle.layout_ranges",
        "handle.combine_buffer",
    ):
        assert property_name not in expert_source
    contract = preflight_rolling_moe_layers.__doc__ or ""
    assert "outside this library's control" in contract


def test_layer_helper_has_no_hidden_cpu_tensor_read_or_sync():
    source = inspect.getsource(enqueue_moe_layer)
    tree = ast.parse(source)
    forbidden = {"item", "tolist", "synchronize", "cpu", "numpy"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & forbidden
