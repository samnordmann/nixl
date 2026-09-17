#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the split NIXL/CuTe dispatch -> expert -> combine MoE pipeline.

This example starts one long-lived process per stable GPU slot.  It exercises
unequal live-token counts (including zero), skewed and empty experts, dropped
``-1`` routes, two physical banks, and graceful shrink/rejoin without changing
process incarnations.  The expert boundary uses the deterministic CuTe kernel
``output = input + 64 * (global_expert + 1)``.  It is intentionally a boundary
adapter, not a performance proxy for grouped GEMM: production applications
replace only :class:`StandinExpertExecutor`.

All tensors, CUDA streams, CUDA events, registered memory, views, and live-N
kernel specializations are prepared before measurement.  The measured loop
contains no device-to-host value read, CPU synchronization, NIXL host progress,
device allocation, or per-operation timing event. A separate instrumented
diagnostic pass completes before the throughput barrier. Correctness and event
timing are read only after a device drain. Results are emitted as JSONL records prefixed by
``NIXL_CUTE_MOE_PIPELINE_RESULT``; rank zero can additionally append the same
records to ``--jsonl``.

Only same-node, all-mapped execution is supported.  Every fixed-capacity worker
must stay alive while masked.  Abrupt process loss/restart is fail-stop and must
be contained by an external Slurm timeout.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping, Sequence


# A torch ``spawn`` target defined by a directly executed ``__main__`` module
# is re-created from this path as ``__mp_main__`` in every child.  Importing and
# calling the package entry point here is insufficient: the parent still has no
# module spec, so children first construct a duplicate package-less module
# graph.  Re-execute the parent through a tiny canonical package bootstrap
# before Torch or CuTe is imported.  The bootstrap puts this source tree ahead
# of the caller's working directory, and ``run_module`` installs the same named
# module spec that ``-m`` would.  Spawned workers therefore bootstrap the
# canonical graph rather than re-running this path as package-less code.
if __name__ == "__main__" and not __package__:
    source_root = str(Path(__file__).resolve().parents[3])
    canonical_module = "examples.python.cute.elastic_moe_pipeline_run"
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "").split(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root]
        + [
            entry
            for entry in inherited_pythonpath
            if entry and os.path.abspath(entry) != source_root
        ]
    )
    bootstrap = (
        "import runpy,sys;"
        f"sys.path.insert(0,{source_root!r});"
        f"runpy.run_module({canonical_module!r},run_name='__main__',alter_sys=True)"
    )
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-c",
            bootstrap,
            *sys.argv[1:],
        ],
        environment,
    )

if __package__:
    from .elastic_moe_pipeline import (
        BoundRollingSchedule,
        ExpertSubmission,
        LayerInputs,
        PreparedRollingSchedule,
        RollingSubmissionPreflight,
        bind_rolling_schedule,
        enqueue_rolling_moe_layers,
        parse_membership,
        preflight_rolling_moe_layers,
        prepare_rolling_schedule,
        prepare_two_bank_window,
        topology_plan,
    )
    from .moe.ll_protocol import (
        STANDIN_EXPERT_BIAS_SCALE,
        PipelineLLArenaLayout,
        StableSparseTopology,
    )
    from .moe.pipeline import MoEPipeline
else:  # Dependency-free top-level import for source-tree API inspection.
    from elastic_moe_pipeline import (  # type: ignore[no-redef]
        BoundRollingSchedule,
        ExpertSubmission,
        LayerInputs,
        PreparedRollingSchedule,
        RollingSubmissionPreflight,
        bind_rolling_schedule,
        enqueue_rolling_moe_layers,
        parse_membership,
        preflight_rolling_moe_layers,
        prepare_rolling_schedule,
        prepare_two_bank_window,
        topology_plan,
    )
    from moe.ll_protocol import (  # type: ignore[no-redef]
        STANDIN_EXPERT_BIAS_SCALE,
        PipelineLLArenaLayout,
        StableSparseTopology,
    )
    from moe.pipeline import MoEPipeline  # type: ignore[no-redef]

if TYPE_CHECKING:
    from ._runtime import FileControlPlane
    from .moe.mapped_pipeline_backend import MappedPipelineBackend
else:
    # Keep ``typing.get_type_hints`` usable without importing the accelerator
    # stack before a spawned rank installs its private CuTe environment.  The
    # concrete classes are imported locally by ``_worker`` below.
    FileControlPlane = Any
    MappedPipelineBackend = Any


RESULT_PREFIX = "NIXL_CUTE_MOE_PIPELINE_RESULT "
OUTPUT_POISON = -512.0
_CORRECTNESS_SIGNATURE_BYTES = 4
_CORRECTNESS_SIGNATURE_FEATURE_OFFSETS = (4, 5, 6, 7)
_MIN_CORRECTNESS_SIGNATURE_SCALE = 2048


@dataclass(frozen=True, slots=True)
class PipelineRunCase:
    """Immutable allocation, workload, and measurement geometry."""

    max_ranks: int
    experts_per_rank: int
    token_capacity: int
    top_k: int
    hidden_size: int
    worker_ctas: int
    warmup: int
    iterations: int
    membership: tuple[tuple[int, ...], ...]
    phase_live_tokens: tuple[tuple[int, ...], ...]
    event_pool_depth: int = 64
    correctness_iterations: int = 132
    latency_iterations: int = 32
    throughput_submission_mode: str = "preflight"

    def __post_init__(self) -> None:
        for name in (
            "max_ranks",
            "experts_per_rank",
            "token_capacity",
            "top_k",
            "hidden_size",
            "worker_ctas",
            "event_pool_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "warmup",
            "iterations",
            "correctness_iterations",
            "latency_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if value % 2:
                raise ValueError(f"{name} must be even for a complete two-bank window")
        if self.iterations == 0:
            raise ValueError("iterations must be positive")
        if self.latency_iterations == 0:
            raise ValueError("latency_iterations must be positive")
        if self.correctness_iterations <= 2 * self.event_pool_depth:
            raise ValueError(
                "correctness_iterations must cross the first per-bank event-ring "
                "reuse (greater than 2 * event_pool_depth)"
            )
        if self.throughput_submission_mode not in ("preflight", "convenience"):
            raise ValueError(
                "throughput_submission_mode must be 'preflight' or 'convenience'"
            )
        if self.hidden_size % 8:
            raise ValueError("hidden_size must be divisible by eight BF16 values")
        if self.max_ranks > 32 or self.top_k > 32:
            raise ValueError("mapped rank and route dimensions cannot exceed one warp")
        if self.top_k > self.max_ranks * self.experts_per_rank:
            raise ValueError("top_k exceeds the fixed expert namespace")
        if len(self.phase_live_tokens) != len(self.membership):
            raise ValueError("phase_live_tokens must cover every membership phase")
        if not self.membership:
            raise ValueError("membership must contain at least one phase")
        for generation, (active, live) in enumerate(
            zip(self.membership, self.phase_live_tokens)
        ):
            if not active:
                raise ValueError(f"membership phase {generation} is empty")
            if tuple(sorted(set(active))) != active:
                raise ValueError(f"membership phase {generation} is not canonical")
            if any(rank < 0 or rank >= self.max_ranks for rank in active):
                raise ValueError(f"membership phase {generation} exceeds capacity")
            if len(live) != self.max_ranks:
                raise ValueError(
                    f"phase {generation} must specify one live count per rank"
                )
            for rank, count in enumerate(live):
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or not 0 <= count <= self.token_capacity
                ):
                    raise ValueError(
                        f"phase {generation} rank {rank} live count is outside "
                        f"[0, {self.token_capacity}]"
                    )
                if rank not in active and count != 0:
                    raise ValueError(
                        f"standby rank {rank} must have zero live tokens in phase "
                        f"{generation}"
                    )

    @property
    def layout(self) -> PipelineLLArenaLayout:
        return PipelineLLArenaLayout(
            max_ranks=self.max_ranks,
            experts_per_rank=self.experts_per_rank,
            num_tokens=self.token_capacity,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            element_size=2,
        )

    @property
    def specializations(self) -> tuple[int, ...]:
        return tuple(
            sorted({count for phase in self.phase_live_tokens for count in phase})
        )


@dataclass(slots=True)
class _RankPhaseTensors:
    inputs: tuple[LayerInputs, LayerInputs]
    performance_schedule: PreparedRollingSchedule | BoundRollingSchedule
    correctness_schedule: PreparedRollingSchedule | BoundRollingSchedule
    correctness_inputs: tuple[LayerInputs, ...]
    correctness_activations: Any
    correctness_outputs: Any
    correctness_input_signature: dict[str, Any]
    route_pattern: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PhaseMeasurement:
    """All timing objects for one phase, allocated before phase zero."""

    diagnostic_operation_events: tuple[tuple[Any, Any], ...]
    diagnostic_end: Any
    correctness_end: Any
    phase_start: Any
    phase_end: Any


@dataclass(frozen=True, slots=True)
class _PreparedWorker:
    """GPU resources whose construction must precede every measured phase."""

    pipeline: MoEPipeline
    executor: "StandinExpertExecutor"
    communication_streams: tuple[Any, Any]
    expert_streams: tuple[Any, Any]
    measurement_stream: Any
    environment: dict[str, Any]
    phase_tensors: tuple[_RankPhaseTensors, ...]
    phase_measurements: tuple[_PhaseMeasurement, ...]
    fused_oracle: dict[str, Any]


class StandinExpertExecutor:
    """Replaceable allocation-audited adapter for communication QA.

    ``enqueue_prepared`` is the measured rolling interface: the library owns a
    private cold result slot and this method returns an event already drawn from
    the backend's bounded pool. ``enqueue`` remains the ordinary compatibility
    interface and necessarily constructs its public ``ExpertSubmission``.
    """

    def __init__(self, backend: MappedPipelineBackend) -> None:
        self.backend = backend

    def enqueue(
        self,
        *,
        expert_input: Any,
        expert_counts: Any,
        source_info: Any,
        layout_ranges: Any,
        output: Any,
        handle: Any,
        stream: Any,
    ) -> ExpertSubmission:
        del expert_input, expert_counts, source_info, layout_ranges
        event = self.backend.enqueue_standin_expert(
            handle=handle,
            output=output,
            stream=stream,
            input_ready_event=None,
        )
        return ExpertSubmission(output=output, ready_event=event)

    def enqueue_prepared(
        self,
        *,
        expert_input: Any,
        expert_counts: Any,
        source_info: Any,
        layout_ranges: Any,
        output: Any,
        handle: Any,
        stream: Any,
    ) -> Any:
        """Post into a private cold slot without constructing a result object."""

        del expert_input, expert_counts, source_info, layout_ranges
        return self.backend.enqueue_standin_expert(
            handle=handle,
            output=output,
            stream=stream,
            input_ready_event=None,
        )


def parse_phase_live_tokens(
    value: str | None,
    *,
    membership: Sequence[Sequence[int]],
    max_ranks: int,
    capacity: int,
) -> tuple[tuple[int, ...], ...]:
    """Parse ``"128,64;96,0"`` or build a deterministic unequal default."""

    if value is None:
        result = []
        for generation, active_values in enumerate(membership):
            active = set(active_values)
            phase = []
            for rank in range(max_ranks):
                if rank not in active:
                    phase.append(0)
                elif (rank + generation) % 3 == 2:
                    phase.append(0)
                elif (rank + generation) % 2:
                    phase.append(max(1, capacity // 2))
                else:
                    phase.append(capacity)
            result.append(tuple(phase))
        return tuple(result)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("phase-live-tokens must be non-empty")
    phases = value.split(";")
    if len(phases) != len(membership):
        raise ValueError("phase-live-tokens must have one row per membership phase")
    result = []
    for generation, phase in enumerate(phases):
        try:
            counts = tuple(int(field.strip(), 10) for field in phase.split(","))
        except ValueError as error:
            raise ValueError(
                f"phase-live-tokens phase {generation} contains a non-integer"
            ) from error
        if len(counts) != max_ranks:
            raise ValueError(
                f"phase-live-tokens phase {generation} must have {max_ranks} values"
            )
        result.append(counts)
    return tuple(result)


def _route_plan(
    *,
    rank: int,
    live_tokens: int,
    topology: StableSparseTopology,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
    """Build distinct valid routes with skew, drops, and one empty expert."""

    experts = list(topology.active_experts)
    # Leave the final active expert empty when possible, as a publication and
    # count-zero adversary.  One-expert generations still need useful traffic.
    route_pool = experts[:-1] if len(experts) > 1 else experts
    indices: list[list[int]] = []
    weights: list[list[float]] = []
    dropped = 0
    remote = 0
    routed = 0
    expert_route_counts = [
        0 for _ in range(topology.max_ranks * topology.experts_per_rank)
    ]
    for token in range(live_tokens):
        choices: list[int] = []
        for route_slot in range(top_k):
            if (rank + token + route_slot) % 11 == 0:
                choices.append(-1)
                dropped += 1
                continue
            candidates = [expert for expert in route_pool if expert not in choices]
            if not candidates:
                choices.append(-1)
                dropped += 1
                continue
            # Slot zero is deliberately skewed; later slots rotate to exercise
            # local and remote owners without duplicate nonnegative experts.
            candidate_index = 0 if route_slot == 0 else rank + token + route_slot
            expert = candidates[candidate_index % len(candidates)]
            choices.append(expert)
            expert_route_counts[expert] += 1
            routed += 1
            if topology.owner(expert) != rank:
                remote += 1
        # Distinct slot-dependent coefficients make a route-slot/source-info
        # swap observable.  The odd normalizer additionally prevents any
        # integer-coefficient subset from contributing exactly one half.  That
        # avoids an FP32 FMA-versus-separate-mul/add delta deciding opposite
        # sides of a final BF16 midpoint in the CPU/device correctness oracle.
        coefficients, normalizer = _route_weight_coefficients(choices)
        indices.append(choices)
        weights.append(
            [
                coefficient / normalizer if normalizer else 0.0
                for coefficient in coefficients
            ]
        )
    return (
        indices,
        weights,
        {
            "routed_records": routed,
            "remote_records": remote,
            "dropped_routes": dropped,
            "intentionally_empty_experts": int(len(experts) > 1),
            # Index is the stable global expert ID.  Inactive and deliberately
            # empty experts remain explicit zeroes so JSONL evidence can prove
            # both skew and zero-count bucket coverage without reading a GPU
            # counter in the measured path.
            "expert_route_counts": expert_route_counts,
            "max_expert_route_count": max(expert_route_counts, default=0),
            "gate_weight_policy": (
                "normalized_positive_distinct_route_slot_plus_one_odd_normalizer"
            ),
        },
    )


def _route_weight_coefficients(choices: Sequence[int]) -> tuple[list[int], int]:
    """Return distinct positive slot weights with an odd nonzero sum.

    Incrementing only the highest valid slot preserves coefficient uniqueness:
    every earlier valid slot remains at most that slot index plus one, while
    the adjusted coefficient is one larger than its already-largest value.
    An odd total cannot be twice an integer subset sum, so no subset of valid
    routes can land at an exact half-weight BF16 rounding boundary.
    """

    coefficients = [
        route_slot + 1 if expert >= 0 else 0
        for route_slot, expert in enumerate(choices)
    ]
    normalizer = sum(coefficients)
    if normalizer and normalizer % 2 == 0:
        last_valid_slot = max(
            route_slot for route_slot, expert in enumerate(choices) if expert >= 0
        )
        coefficients[last_valid_slot] += 1
        normalizer += 1
    return coefficients, normalizer


def _correctness_signature_scale(topology: StableSparseTopology) -> float:
    """Return a power-of-two scale that keeps signature bytes BF16-exact.

    A byte has at most eight significant bits, so ``byte * 2**n`` is exactly
    representable in BF16.  Keeping the maximum stand-in expert bias below one
    quarter of the scale also prevents its BF16 rounding from collapsing two
    adjacent operation codes.
    """

    maximum_expert_bias = (
        topology.max_ranks * topology.experts_per_rank * STANDIN_EXPERT_BIAS_SCALE
    )
    lower_bound = max(
        _MIN_CORRECTNESS_SIGNATURE_SCALE,
        4 * maximum_expert_bias,
    )
    return float(1 << (lower_bound - 1).bit_length())


def _correctness_operation_signature(operation: int, scale: float) -> tuple[float, ...]:
    """Encode one Uint32 operation ID as four scaled little-endian bytes."""

    if isinstance(operation, bool) or not isinstance(operation, int):
        raise TypeError("correctness operation must be an integer")
    if not 0 <= operation < 1 << (8 * _CORRECTNESS_SIGNATURE_BYTES):
        raise ValueError("correctness operation must fit in Uint32")
    return tuple(
        float((operation >> (8 * byte)) & 0xFF) * scale
        for byte in range(_CORRECTNESS_SIGNATURE_BYTES)
    )


def _correctness_signature_attestation(
    *,
    case: PipelineRunCase,
    topology: StableSparseTopology,
    live_tokens: int,
    routed_records: int,
) -> dict[str, Any]:
    """Describe and self-check the operation-discriminating input pattern."""

    scale = _correctness_signature_scale(topology)
    same_parity_operations = (0, 2)
    ring_wrap_operations = (0, 2 * case.event_pool_depth)

    def probe(operations: tuple[int, int]) -> dict[str, Any]:
        signatures = tuple(
            _correctness_operation_signature(operation, scale)
            for operation in operations
        )
        return {
            "operations": list(operations),
            "signatures": [list(signature) for signature in signatures],
            "distinct": signatures[0] != signatures[1],
        }

    return {
        "policy": "uint32_le_bfloat16_activation_channels_v1",
        "feature_offsets_mod_8": list(_CORRECTNESS_SIGNATURE_FEATURE_OFFSETS),
        "scale": scale,
        "exact_bfloat16": True,
        "operation_count": case.correctness_iterations,
        "materialized_rows_per_operation": live_tokens,
        "payload_observable": bool(live_tokens and routed_records),
        "same_parity_probe": probe(same_parity_operations),
        "event_ring_wrap_probe": {
            "event_pool_depth": case.event_pool_depth,
            **probe(ring_wrap_operations),
        },
    }


def _make_phase_tensors(
    torch: Any,
    *,
    case: PipelineRunCase,
    topology: StableSparseTopology,
    generation: int,
    rank: int,
    device: int,
    preparation_stream: Any,
) -> _RankPhaseTensors:
    live_tokens = case.phase_live_tokens[generation][rank]
    indices, weights, pattern = _route_plan(
        rank=rank,
        live_tokens=live_tokens,
        topology=topology,
        top_k=case.top_k,
    )
    bank_inputs = []
    with torch.cuda.device(device), torch.cuda.stream(preparation_stream):
        # Every correctness operation owns one output slice, so an early bad
        # result cannot be overwritten by a later reuse of the same physical
        # communication bank.  This allocation and poison fill are deliberately
        # outside both the correctness submission loop and measured path.
        correctness_outputs = torch.full(
            (case.correctness_iterations, live_tokens, case.hidden_size),
            OUTPUT_POISON,
            dtype=torch.bfloat16,
            device=device,
        )
        for bank in range(2):
            # Operation-independent inputs keep the measured path identical on
            # every iteration. Its two outputs are checked in addition to the
            # separate per-operation correctness pass below.
            feature_index = torch.arange(
                case.hidden_size,
                dtype=torch.int64,
                device=device,
            ).reshape(1, case.hidden_size)
            token_index = torch.arange(
                live_tokens,
                dtype=torch.float32,
                device=device,
            ).reshape(live_tokens, 1)
            channel = feature_index.remainder(3)
            feature_signature = feature_index.remainder(16).to(torch.float32) * 16.0
            token_signature = (token_index + 1.0) * 512.0
            rank_signature = torch.full_like(token_index, float(rank + 1) * 512.0)
            bank_signature = torch.full_like(token_index, float(bank + 1) * 512.0)
            activations = (
                torch.where(
                    channel == 0,
                    token_signature,
                    torch.where(channel == 1, rank_signature, bank_signature),
                )
                + feature_signature
            ).to(torch.bfloat16)
            topk_indices = torch.tensor(indices, dtype=torch.int32, device=device)
            topk_indices = topk_indices.reshape(live_tokens, case.top_k)
            topk_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            topk_weights = topk_weights.reshape(live_tokens, case.top_k)
            output = torch.full(
                (live_tokens, case.hidden_size),
                OUTPUT_POISON,
                dtype=torch.bfloat16,
                device=device,
            )
            bank_inputs.append(
                LayerInputs(
                    activations=activations,
                    topk_indices=topk_indices,
                    topk_weights=topk_weights,
                    output=output,
                    input_ready_event=None,
                )
            )
        if len(bank_inputs) != 2:
            raise AssertionError("two physical banks were not materialized")
        typed_inputs = (bank_inputs[0], bank_inputs[1])

        # Correctness uses unique logical inputs as well as unique outputs.
        # Four channels per eight-feature vector carry the little-endian bytes
        # of the Uint32 operation ID.  Each byte is multiplied by a power of
        # two, hence is exactly representable in BF16.  In particular, steps
        # 0, 2, and 128 differ despite sharing a physical bank, and 0/128 also
        # share the first event-ring slot at the default depth of 64.
        correctness_activations = torch.empty(
            (case.correctness_iterations, live_tokens, case.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        correctness_activations[0::2].copy_(typed_inputs[0].activations)
        correctness_activations[1::2].copy_(typed_inputs[1].activations)
        signature_scale = _correctness_signature_scale(topology)
        operation_signatures = torch.tensor(
            tuple(
                _correctness_operation_signature(operation, signature_scale)
                for operation in range(case.correctness_iterations)
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        for byte, feature_offset in enumerate(_CORRECTNESS_SIGNATURE_FEATURE_OFFSETS):
            correctness_activations[:, :, feature_offset::8] = operation_signatures[
                :, byte
            ].reshape(case.correctness_iterations, 1, 1)
    performance_window = prepare_two_bank_window(*typed_inputs)
    correctness_inputs = tuple(
        LayerInputs(
            activations=correctness_activations[operation],
            topk_indices=typed_inputs[operation & 1].topk_indices,
            topk_weights=typed_inputs[operation & 1].topk_weights,
            output=correctness_outputs[operation],
            input_ready_event=typed_inputs[operation & 1].input_ready_event,
        )
        for operation in range(case.correctness_iterations)
    )
    correctness_windows = tuple(
        prepare_two_bank_window(
            correctness_inputs[operation], correctness_inputs[operation + 1]
        )
        for operation in range(0, case.correctness_iterations, 2)
    )
    performance_schedule = prepare_rolling_schedule(
        (performance_window,),
        repeat_count=(max(case.warmup, case.iterations, case.latency_iterations) // 2),
    )
    correctness_schedule = prepare_rolling_schedule(correctness_windows)
    return _RankPhaseTensors(
        inputs=typed_inputs,
        performance_schedule=performance_schedule,
        correctness_schedule=correctness_schedule,
        correctness_inputs=correctness_inputs,
        correctness_activations=correctness_activations,
        correctness_outputs=correctness_outputs,
        correctness_input_signature=_correctness_signature_attestation(
            case=case,
            topology=topology,
            live_tokens=live_tokens,
            routed_records=int(pattern["routed_records"]),
        ),
        route_pattern=pattern,
    )


def _submit_windows(
    *,
    pipeline: MoEPipeline,
    executor: StandinExpertExecutor,
    tensors: _RankPhaseTensors,
    count: int,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
    timing_events: Sequence[tuple[Any, Any]] | None,
    submission_preflight: RollingSubmissionPreflight | None = None,
    submission_ready_event: Any | None = None,
) -> tuple[Any, Any] | None:
    if count == 0:
        return None
    schedule = tensors.performance_schedule
    if not isinstance(schedule, BoundRollingSchedule):
        raise AssertionError("performance schedule was not cold-bound")
    return enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        operation_count=count,
        communication_streams=communication_streams,
        expert_streams=expert_streams,
        operation_events=timing_events,
        submission_preflight=submission_preflight,
        submission_ready_event=submission_ready_event,
    )


def _bind_phase_schedules(
    pipeline: MoEPipeline,
    phase_tensors: Sequence[_RankPhaseTensors],
) -> None:
    """Seal each finite cycle into pipeline/backend operand tokens."""

    for tensors in phase_tensors:
        if not isinstance(tensors.performance_schedule, PreparedRollingSchedule):
            raise AssertionError("performance schedule was already bound")
        if not isinstance(tensors.correctness_schedule, PreparedRollingSchedule):
            raise AssertionError("correctness schedule was already bound")
        tensors.performance_schedule = bind_rolling_schedule(
            pipeline, tensors.performance_schedule
        )
        tensors.correctness_schedule = bind_rolling_schedule(
            pipeline, tensors.correctness_schedule
        )


def _submit_correctness_windows(
    *,
    pipeline: MoEPipeline,
    executor: StandinExpertExecutor,
    tensors: _RankPhaseTensors,
    communication_streams: tuple[Any, Any],
    expert_streams: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Submit sealed unique-output windows without timing instrumentation."""

    schedule = tensors.correctness_schedule
    if not isinstance(schedule, BoundRollingSchedule):
        raise AssertionError("correctness schedule was not cold-bound")
    return enqueue_rolling_moe_layers(
        pipeline,
        executor,
        schedule,
        communication_streams=communication_streams,
        expert_streams=expert_streams,
    )


def _record_transitive_phase_end(
    *,
    measurement_stream: Any,
    latest_submissions: tuple[Any, Any],
    phase_end: Any,
) -> None:
    """Record a fence that transitively covers both expert and comm streams.

    Each submission's bank-reuse milestone is recorded on its communication
    stream after combine. Combine first waits for ``expert_ready`` from the
    corresponding expert stream, whose stand-in/GEMM first waits for
    dispatch-ready. Enqueueing both guarded waits here therefore covers both
    communication streams and both expert streams without exposing or reusing
    a raw pooled CUDA event.
    """

    if len(latest_submissions) != 2:
        raise ValueError("phase end requires the final complete two-bank window")
    for submission in latest_submissions:
        submission.enqueue_bank_reuse_wait(measurement_stream)
    phase_end.record(measurement_stream)


def _prime_phase_measurements(
    resources: Sequence[_PhaseMeasurement], preparation_stream: Any
) -> None:
    """Materialize every lazy CUDA timing event, then cold-drain setup."""

    for measurement in resources:
        for start, end in measurement.diagnostic_operation_events:
            start.record(preparation_stream)
            end.record(preparation_stream)
        measurement.diagnostic_end.record(preparation_stream)
        measurement.correctness_end.record(preparation_stream)
        measurement.phase_start.record(preparation_stream)
        measurement.phase_end.record(preparation_stream)
    # All phase inputs were enqueued on this same stream before this helper.
    # This one pre-phase CPU drain both materializes timing events and proves
    # input construction complete. Because every synthetic operand was built
    # on this stream before the drain, the benchmark intentionally omits
    # already-satisfied per-dispatch input-ready waits. Real router adapters
    # retain those GPU event dependencies.
    preparation_stream.synchronize()


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "min_ms": min(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p90_ms": _percentile(values, 0.90),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values) if values else 0.0,
        "mean_ms": statistics.fmean(values) if values else 0.0,
    }


def _expected_output_from_host(
    torch: Any, activations: Any, topk_indices: Any, topk_weights: Any
) -> Any:
    """Build a CPU golden from host tensors with both device BF16 roundings."""

    x = activations.float()
    indices = topk_indices
    weights = topk_weights
    expected = torch.zeros_like(x, dtype=torch.float32)
    for route_slot in range(indices.shape[1]):
        route = indices[:, route_slot]
        valid = route >= 0
        if bool(valid.any()):
            # Mirror the device stand-in's BF16 expert output before the
            # combine kernel accumulates weighted rows in FP32.  The 64x route
            # code makes an adjacent-expert substitution contribute at least
            # 2.0 at K=8, safely above the strict 0.5 absolute tolerance after
            # the final BF16 store.
            expert_row = (
                x[valid]
                + (route[valid, None].float() + 1.0) * STANDIN_EXPERT_BIAS_SCALE
            ).to(torch.bfloat16)
            expected[valid] += expert_row.float() * weights[valid, route_slot, None]
    return expected.to(torch.bfloat16).float()


def _expected_output(torch: Any, inputs: LayerInputs) -> Any:
    """Copy one operand set to the host and build its deterministic golden."""

    return _expected_output_from_host(
        torch,
        inputs.activations.cpu(),
        inputs.topk_indices.cpu(),
        inputs.topk_weights.cpu(),
    )


def _compare_output(torch: Any, observed: Any, expected: Any) -> tuple[bool, float]:
    observed_float = observed.float()
    maximum_error = (
        float((observed_float - expected).abs().max().item())
        if observed_float.numel()
        else 0.0
    )
    return (
        bool(torch.allclose(observed_float, expected, rtol=0.0, atol=5.0e-1)),
        maximum_error,
    )


def _validate_outputs(
    torch: Any, inputs_sequence: Sequence[LayerInputs]
) -> tuple[bool, float]:
    """Read selected outputs only after the caller has drained GPU work."""

    correct = True
    maximum_error = 0.0
    for inputs in inputs_sequence:
        expected = _expected_output(torch, inputs)
        observed = inputs.output.cpu()
        item_correct, error = _compare_output(torch, observed, expected)
        correct = correct and item_correct
        maximum_error = max(maximum_error, error)
    return correct, maximum_error


def _validate_correctness_outputs(
    torch: Any, tensors: _RankPhaseTensors
) -> tuple[bool, float]:
    """Validate every unique output after one terminal correctness drain."""

    # Bulk transfers preserve every operation while avoiding a D2H copy and
    # synchronization per slice. Routes and weights still alternate by bank;
    # activations carry a unique BF16-exact Uint32 operation signature.
    observed = tensors.correctness_outputs.cpu()
    activations = tensors.correctness_activations.cpu()
    indices = tuple(inputs.topk_indices.cpu() for inputs in tensors.inputs)
    weights = tuple(inputs.topk_weights.cpu() for inputs in tensors.inputs)
    correct = True
    maximum_error = 0.0
    for operation in range(observed.shape[0]):
        bank = operation & 1
        expected = _expected_output_from_host(
            torch,
            activations[operation],
            indices[bank],
            weights[bank],
        )
        item_correct, error = _compare_output(torch, observed[operation], expected)
        correct = correct and item_correct
        maximum_error = max(maximum_error, error)
    return correct, maximum_error


def _environment(
    torch: Any, backend: MappedPipelineBackend, device: int
) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        nixl_version = importlib.metadata.version("nixl")
    except importlib.metadata.PackageNotFoundError:
        nixl_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git_revision": revision,
        "container_image": os.environ.get("SLURM_CONTAINER_IMAGE"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "device": device,
        "nixl": nixl_version,
        "nixl_cute_abi": backend._deps.nixl_cute.NIXL_CUTE_ABI_VERSION,
        "native_peer_atomics": backend.native_atomic_evidence,
        "mapped_pointer_preflight": backend.mapped_pointer_evidence,
    }


def _encode(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_rank_records(
    values: Mapping[int, bytes], max_ranks: int
) -> tuple[dict[str, Any], ...]:
    if set(values) != set(range(max_ranks)):
        raise RuntimeError("rank result exchange has incomplete stable-slot coverage")
    return tuple(json.loads(values[rank].decode("utf-8")) for rank in range(max_ranks))


def _prepare_worker(
    torch: Any,
    *,
    rank: int,
    device: int,
    case: PipelineRunCase,
    control: FileControlPlane,
    backend: MappedPipelineBackend,
    topologies: tuple[StableSparseTopology, ...],
    fused_oracle_json: str | None,
) -> _PreparedWorker:
    """Construct and cold-prime all worker resources."""

    pipeline = MoEPipeline(
        rank=rank,
        topology=topologies[0],
        layout=case.layout,
        backend=backend,
    )
    executor = StandinExpertExecutor(backend)
    communication_streams = (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    )
    expert_streams = (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    )
    preparation_stream = torch.cuda.Stream(device=device)
    measurement_stream = torch.cuda.Stream(device=device)
    environment = _environment(torch, backend, device)

    # Allocate every phase operand and every event before generation zero can
    # enter a measured loop. Input construction is isolated on one preparation
    # stream and cold-drained below; synthetic inputs therefore need no
    # already-satisfied wait node in the benchmark hot path. A real router
    # adapter still supplies its explicit GPU readiness event.
    phase_tensors = tuple(
        _make_phase_tensors(
            torch,
            case=case,
            topology=topology,
            generation=generation,
            rank=rank,
            device=device,
            preparation_stream=preparation_stream,
        )
        for generation, topology in enumerate(topologies)
    )
    _bind_phase_schedules(pipeline, phase_tensors)
    phase_measurements = tuple(
        _PhaseMeasurement(
            diagnostic_operation_events=tuple(
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(case.latency_iterations)
            ),
            diagnostic_end=torch.cuda.Event(enable_timing=False),
            correctness_end=torch.cuda.Event(enable_timing=False),
            phase_start=torch.cuda.Event(enable_timing=True),
            phase_end=torch.cuda.Event(enable_timing=True),
        )
        for _ in topologies
    )
    _prime_phase_measurements(phase_measurements, preparation_stream)
    fused_oracle = _load_fused_oracle(fused_oracle_json)
    control.barrier("runner-all-phase-resources-enqueued")
    return _PreparedWorker(
        pipeline=pipeline,
        executor=executor,
        communication_streams=communication_streams,
        expert_streams=expert_streams,
        measurement_stream=measurement_stream,
        environment=environment,
        phase_tensors=phase_tensors,
        phase_measurements=phase_measurements,
        fused_oracle=fused_oracle,
    )


def _configure_codegen_dump(root: str | None, rank: int) -> str | None:
    """Create one cold, rank-private CuTe dump directory before CuTe import."""

    if root is None:
        inherited_keep = tuple(
            name
            for name in (
                "CUTE_DSL_KEEP",
                "CUTE_DSL_KEEP_IR",
                "CUTE_DSL_KEEP_PTX",
                "CUTE_DSL_KEEP_CUBIN",
            )
            if os.environ.get(name, "").strip().lower()
            not in ("", "0", "false", "no", "off")
        )
        if inherited_keep:
            raise ValueError(
                "CuTe artifact retention requires --codegen-dump-root so each "
                f"worker has a private directory; inherited {inherited_keep!r}"
            )
        return None
    parent = Path(root).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("codegen dump root must be an existing directory")
    rank_dump = parent / f"rank-{rank}"
    rank_dump.mkdir(mode=0o700, exist_ok=False)
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(rank_dump)
    return str(rank_dump)


def _worker(
    rank: int,
    devices: tuple[int, ...],
    directory: str,
    case: PipelineRunCase,
    timeout_s: float,
    jsonl: str | None,
    fused_oracle_json: str | None,
    codegen_dump_root: str | None,
) -> None:
    # CuTe DSL snapshots its dump directory when imported.  Configure a fresh,
    # rank-private evidence directory before Torch or the lazy device backend
    # can import the DSL.  This is compile/admission-only work.
    _configure_codegen_dump(codegen_dump_root, rank)
    if __package__:
        from ._runtime import FileControlPlane
        from .moe.mapped_pipeline_backend import MappedPipelineBackend
    else:  # Direct ``python examples/python/cute/...`` execution.
        from _runtime import FileControlPlane  # type: ignore[no-redef]
        from moe.mapped_pipeline_backend import (  # type: ignore[no-redef]
            MappedPipelineBackend,
        )

    import torch  # pylint: disable=import-outside-toplevel

    device = devices[rank]
    torch.cuda.set_device(device)
    control = FileControlPlane(directory, rank, case.max_ranks, timeout_s)
    topologies = topology_plan(
        case.membership,
        max_ranks=case.max_ranks,
        experts_per_rank=case.experts_per_rank,
    )
    backend = MappedPipelineBackend(
        control=control,
        devices=devices,
        live_token_specializations=case.specializations,
        worker_ctas=case.worker_ctas,
        run_id=Path(directory).name,
        event_pool_depth=case.event_pool_depth,
        timeout_s=timeout_s,
    )
    try:
        prepared = _prepare_worker(
            torch,
            rank=rank,
            device=device,
            case=case,
            control=control,
            backend=backend,
            topologies=topologies,
            fused_oracle_json=fused_oracle_json,
        )
    except BaseException as error:
        if backend.state.value not in ("failed", "closed"):
            backend.fail_stop(error)
        raise
    pipeline = prepared.pipeline
    executor = prepared.executor
    communication_streams = prepared.communication_streams
    expert_streams = prepared.expert_streams
    measurement_stream = prepared.measurement_stream
    environment = prepared.environment
    phase_tensors = prepared.phase_tensors
    phase_measurements = prepared.phase_measurements
    fused_oracle = prepared.fused_oracle

    try:
        for generation, topology in enumerate(topologies):
            tensors = phase_tensors[generation]
            measurement = phase_measurements[generation]
            staged = None
            stage_wall_ms = 0.0
            next_generation = (
                topologies[generation + 1] if generation + 1 < len(topologies) else None
            )
            if next_generation is not None:
                stage_start_ns = time.perf_counter_ns()
                staged = pipeline.stage_generation(next_generation)
                stage_wall_ms = (time.perf_counter_ns() - stage_start_ns) / 1.0e6
                # All fixed-capacity processes have now prepared hidden next-
                # generation views, but the old generation remains committed.
                control.barrier(f"runner-g{generation + 1}-staged")
            # poison_data_planes uses the backend's dedicated control stream
            # and performs a cold synchronous drain. Consumers are submitted
            # only afterward, so neither a CPU file barrier nor default-stream
            # behavior is being used as a GPU dependency.
            backend.poison_data_planes()
            control.barrier(f"runner-g{generation}-prepared")
            active = topology.is_active(rank)
            timings: list[float] = []
            wall_ms = 0.0
            host_enqueue_total_ms = 0.0
            host_measurement_start_ns = 0
            host_measurement_terminal_ns = 0
            correct = True
            maximum_error = 0.0
            if active:
                correctness_latest = _submit_correctness_windows(
                    pipeline=pipeline,
                    executor=executor,
                    tensors=tensors,
                    communication_streams=communication_streams,
                    expert_streams=expert_streams,
                )
                _record_transitive_phase_end(
                    measurement_stream=measurement_stream,
                    latest_submissions=correctness_latest,
                    phase_end=measurement.correctness_end,
                )
                # The only correctness-pass drain occurs after every unique
                # output has been submitted. It is outside warmup/measurement.
                measurement.correctness_end.synchronize()
                correct, maximum_error = _validate_correctness_outputs(torch, tensors)
            control.barrier(f"runner-g{generation}-correctness")

            if active:
                _submit_windows(
                    pipeline=pipeline,
                    executor=executor,
                    tensors=tensors,
                    count=case.warmup,
                    communication_streams=communication_streams,
                    expert_streams=expert_streams,
                    timing_events=None,
                )
                diagnostic_latest = _submit_windows(
                    pipeline=pipeline,
                    executor=executor,
                    tensors=tensors,
                    count=case.latency_iterations,
                    communication_streams=communication_streams,
                    expert_streams=expert_streams,
                    timing_events=measurement.diagnostic_operation_events,
                )
                if diagnostic_latest is None:
                    raise AssertionError("positive diagnostic count produced no work")
                _record_transitive_phase_end(
                    measurement_stream=measurement_stream,
                    latest_submissions=diagnostic_latest,
                    phase_end=measurement.diagnostic_end,
                )
                # One setup/diagnostic drain covers every warmup and latency
                # operation. The throughput pass below is therefore neither
                # instrumented nor contaminated by unfinished setup work.
                measurement.diagnostic_end.synchronize()
                timings = [
                    float(start.elapsed_time(end))
                    for start, end in measurement.diagnostic_operation_events
                ]
            throughput_preflight = None
            if active and case.throughput_submission_mode == "preflight":
                # Capture queries complete before the all-rank timing barrier.
                # This worker is the exclusive submitter, and until the rolling
                # helper consumes the token it posts only to measurement_stream,
                # which is deliberately outside the admitted stream set.
                throughput_schedule = tensors.performance_schedule
                if not isinstance(throughput_schedule, BoundRollingSchedule):
                    raise AssertionError("performance schedule was not cold-bound")
                throughput_preflight = preflight_rolling_moe_layers(
                    pipeline,
                    executor,
                    throughput_schedule,
                    operation_count=case.iterations,
                    communication_streams=communication_streams,
                    expert_streams=expert_streams,
                )
            # ``convenience`` deliberately leaves the token empty. The same
            # helper, kernels, tensors, streams, and fast expert adapter run,
            # but finite-plan construction/admission then falls inside the
            # uninterrupted host bracket as a matched submission-path control.
            control.barrier(f"runner-g{generation}-warm")

            if active:
                # CLOCK_MONOTONIC is a single same-node clock domain. These
                # timestamps bound rank launch skew and terminal completion;
                # CUDA events below measure the exact local device interval.
                host_measurement_start_ns = time.monotonic_ns()
                measurement.phase_start.record(measurement_stream)
                host_enqueue_start_ns = time.perf_counter_ns()
                latest = _submit_windows(
                    pipeline=pipeline,
                    executor=executor,
                    tensors=tensors,
                    count=case.iterations,
                    communication_streams=communication_streams,
                    expert_streams=expert_streams,
                    timing_events=None,
                    submission_preflight=throughput_preflight,
                    submission_ready_event=measurement.phase_start,
                )
                host_enqueue_end_ns = time.perf_counter_ns()
                host_enqueue_total_ms = (
                    host_enqueue_end_ns - host_enqueue_start_ns
                ) / 1.0e6
                if latest is None:
                    raise AssertionError("positive iteration count produced no work")
                _record_transitive_phase_end(
                    measurement_stream=measurement_stream,
                    latest_submissions=latest,
                    phase_end=measurement.phase_end,
                )

            # Capture old-generation evidence before make-before-activate swaps
            # the backend's current mapped view. The transition wall interval
            # includes the cumulative GPU drain, device rebase, global commit
            # barriers, and old-view release. Candidate pointer resolution was
            # already hidden in the earlier staging interval.
            phase_mapping_evidence = backend.mapped_pointer_evidence
            phase_occupancy_evidence = backend.occupancy_evidence
            drain_commit_wall_ms = 0.0
            if staged is not None:
                transition_start_ns = time.perf_counter_ns()
                drain = pipeline.begin_generation_drain(
                    staged, stream=measurement_stream
                )
                pipeline.commit_generation(staged, drain)
                drain_commit_wall_ms = (
                    time.perf_counter_ns() - transition_start_ns
                ) / 1.0e6
            elif active:
                # The terminal generation has no commit whose drain can cover
                # the measured tail, so it needs this one explicit end wait.
                measurement.phase_end.synchronize()

            if active:
                host_measurement_terminal_ns = time.monotonic_ns()
                # A transition commit above synchronized its cumulative drain;
                # the terminal case synchronized phase_end directly. Timing
                # reads and D2H validation therefore happen only after either
                # terminal path, never before a measured elastic cutover.
                wall_ms = float(
                    measurement.phase_start.elapsed_time(measurement.phase_end)
                )
                performance_correct, performance_error = _validate_outputs(
                    torch, tensors.inputs
                )
                correct = correct and performance_correct
                maximum_error = max(maximum_error, performance_error)

            live_tokens = case.phase_live_tokens[generation][rank]
            # Count only non-dropped routes.  Each routed BF16 row moves once
            # during dispatch and once during combine.  Reporting the dense
            # N*K envelope would credit `-1` routes that issue no payload copy.
            payload_bytes = (
                int(tensors.route_pattern["routed_records"])
                * case.hidden_size
                * 2
                * 2
                * case.iterations
            )
            local_result = {
                "schema_version": 1,
                "record_type": "rank_phase",
                "rank": rank,
                "generation": generation,
                "active_ranks": list(topology.active_ranks),
                "rank_incarnations": list(topology.rank_incarnations),
                "active": active,
                "live_tokens": live_tokens,
                "shape": {
                    "max_ranks": case.max_ranks,
                    "experts_per_rank": case.experts_per_rank,
                    "token_capacity": case.token_capacity,
                    "top_k": case.top_k,
                    "hidden_size": case.hidden_size,
                },
                "warmup": case.warmup,
                "iterations": case.iterations,
                "throughput_submission_mode": case.throughput_submission_mode,
                "latency_iterations": case.latency_iterations,
                "correctness_iterations": case.correctness_iterations,
                "correctness_crosses_event_ring_reuse": (
                    case.correctness_iterations > 2 * case.event_pool_depth
                ),
                "correctness_input_signature": (tensors.correctness_input_signature),
                "route_pattern": tensors.route_pattern,
                "correctness": "PASS" if correct else "FAIL",
                "maximum_absolute_error": maximum_error,
                "diagnostic_event_latency": _distribution(timings),
                "rolling_device_wall_ms": wall_ms,
                "host_measurement_envelope": {
                    "clock": "CLOCK_MONOTONIC_same_node",
                    "start_ns": host_measurement_start_ns,
                    "terminal_ns": host_measurement_terminal_ns,
                    "includes_elastic_cutover": staged is not None,
                },
                "host_enqueue_total_ms": host_enqueue_total_ms,
                "host_enqueue_us_per_operation": (
                    host_enqueue_total_ms * 1000.0 / case.iterations if active else 0.0
                ),
                "routed_logical_bidirectional_payload_bytes": payload_bytes,
                "local_tokens_per_second": (
                    live_tokens * case.iterations * 1000.0 / wall_ms
                    if wall_ms > 0
                    else 0.0
                ),
                "local_routed_logical_payload_gbps": (
                    payload_bytes * 8.0 / (wall_ms * 1.0e6) if wall_ms > 0 else 0.0
                ),
                "occupancy": phase_occupancy_evidence,
                "environment": environment,
                "mapped_pointer_preflight": phase_mapping_evidence,
                "elastic_transition": {
                    "status": (
                        "COMMITTED" if next_generation is not None else "TERMINAL"
                    ),
                    "next_generation": (
                        next_generation.membership_generation
                        if next_generation is not None
                        else None
                    ),
                    "stage_wall_ms": stage_wall_ms,
                    "old_generation_operations_while_staged": (
                        case.correctness_iterations
                        + case.warmup
                        + case.latency_iterations
                        + case.iterations
                        if active and next_generation is not None
                        else 0
                    ),
                    "drain_commit_wall_ms": drain_commit_wall_ms,
                },
                "performance_scope": (
                    "split communication plus deterministic stand-in expert; "
                    "uninstrumented rolling throughput pass; not grouped-GEMM "
                    "application throughput"
                ),
                "throughput_submission_scope": (
                    "preflight: finite host plan built before timing"
                    if case.throughput_submission_mode == "preflight"
                    else "convenience: identical finite host plan built inside timing"
                ),
            }
            gathered = control.exchange(
                f"runner-g{generation}-results", _encode(local_result)
            )
            records = _decode_rank_records(gathered, case.max_ranks)
            phase_correct = all(record["correctness"] == "PASS" for record in records)
            if rank == 0:
                active_records = [record for record in records if record["active"]]
                maximum_rank_device_duration_ms = max(
                    (
                        float(record["rolling_device_wall_ms"])
                        for record in active_records
                    ),
                    default=0.0,
                )
                envelope_start_ns = min(
                    int(record["host_measurement_envelope"]["start_ns"])
                    for record in active_records
                )
                envelope_terminal_ns = max(
                    int(record["host_measurement_envelope"]["terminal_ns"])
                    for record in active_records
                )
                service_envelope_ms = (envelope_terminal_ns - envelope_start_ns) / 1.0e6
                total_tokens = sum(
                    int(record["live_tokens"]) * case.iterations
                    for record in active_records
                )
                total_payload = sum(
                    int(record["routed_logical_bidirectional_payload_bytes"])
                    for record in active_records
                )
                aggregate = {
                    "schema_version": 1,
                    "record_type": "aggregate_phase",
                    "generation": generation,
                    "throughput_submission_mode": case.throughput_submission_mode,
                    "active_ranks": list(topology.active_ranks),
                    "correctness": "PASS" if phase_correct else "FAIL",
                    "rank_results": records,
                    "maximum_rank_device_duration_ms": (
                        maximum_rank_device_duration_ms
                    ),
                    "distributed_service_envelope_ms": service_envelope_ms,
                    "distributed_service_envelope_scope": (
                        "same-node earliest active-rank throughput submission to "
                        "latest terminal return; elastic phases conservatively "
                        "include their drain/commit cutover"
                    ),
                    "aggregate_tokens_per_second": (
                        total_tokens * 1000.0 / service_envelope_ms
                        if service_envelope_ms > 0
                        else 0.0
                    ),
                    "aggregate_routed_logical_payload_gbps": (
                        total_payload * 8.0 / (service_envelope_ms * 1.0e6)
                        if service_envelope_ms > 0
                        else 0.0
                    ),
                    "elastic_transition": {
                        "status": (
                            "COMMITTED" if next_generation is not None else "TERMINAL"
                        ),
                        "next_generation": (
                            next_generation.membership_generation
                            if next_generation is not None
                            else None
                        ),
                        "maximum_stage_wall_ms": max(
                            float(record["elastic_transition"]["stage_wall_ms"])
                            for record in records
                        ),
                        "maximum_drain_commit_wall_ms": max(
                            float(record["elastic_transition"]["drain_commit_wall_ms"])
                            for record in records
                        ),
                        "active_operations_while_staged": sum(
                            int(
                                record["elastic_transition"][
                                    "old_generation_operations_while_staged"
                                ]
                            )
                            for record in records
                        ),
                    },
                    "fused_oracle": fused_oracle,
                    "throughput_submission_scope": (
                        "matched kernels/storage/shape/order; mode changes only "
                        "whether finite host-plan construction is outside or "
                        "inside the throughput bracket"
                    ),
                }
                line = RESULT_PREFIX + json.dumps(aggregate, sort_keys=True)
                print(line, flush=True)
                if jsonl is not None:
                    with Path(jsonl).open("a", encoding="utf-8") as output_file:
                        output_file.write(json.dumps(aggregate, sort_keys=True) + "\n")
            control.barrier(f"runner-g{generation}-reported")
            if not phase_correct:
                raise RuntimeError(
                    f"generation {generation} failed end-to-end correctness"
                )
        pipeline.close()
    except BaseException as error:
        # Pipeline/backend errors already select fail-stop.  A runner error
        # after registration must do the same rather than unwinding owners.
        if backend.state.value not in ("failed", "closed"):
            backend.fail_stop(error)
        raise


def _load_fused_oracle(path: str | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "comparison": "not run; no fused-oracle evidence was supplied",
        }
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "provided": True,
        "source": str(source.resolve()),
        "payload": payload,
        "comparison": "external evidence attached verbatim; no metric fabricated",
    }


def run(
    *,
    devices: tuple[int, ...],
    case: PipelineRunCase,
    timeout_s: float,
    jsonl: str | None = None,
    fused_oracle_json: str | None = None,
    codegen_dump_root: str | None = None,
) -> None:
    """Spawn every fixed-capacity rank and wait for terminal completion."""

    import torch  # pylint: disable=import-outside-toplevel
    import torch.multiprocessing as mp  # pylint: disable=import-outside-toplevel

    if len(devices) != case.max_ranks or len(set(devices)) != len(devices):
        raise ValueError("devices must contain one distinct GPU per stable rank")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a positive finite number")
    if not torch.cuda.is_available() or torch.cuda.device_count() < case.max_ranks:
        raise RuntimeError(f"this run requires {case.max_ranks} visible CUDA devices")
    for device in devices:
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    if fused_oracle_json is not None:
        _load_fused_oracle(fused_oracle_json)
    with TemporaryDirectory(prefix=f"nixl_cute_pipeline_{os.getpid()}_") as directory:
        mp.spawn(
            _worker,
            args=(
                devices,
                directory,
                case,
                float(timeout_s),
                jsonl,
                fused_oracle_json,
                codegen_dump_root,
            ),
            nprocs=case.max_ranks,
            join=True,
        )


def _default_phase_tokens(
    membership: tuple[tuple[int, ...], ...], max_ranks: int, capacity: int
) -> tuple[tuple[int, ...], ...]:
    return parse_phase_live_tokens(
        None,
        membership=membership,
        max_ranks=max_ranks,
        capacity=capacity,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--token-capacity", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--worker-ctas", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--throughput-submission-mode",
        choices=("preflight", "convenience"),
        default="preflight",
        help=(
            "preflight materializes the finite rolling host plan before timing; "
            "convenience builds the identical plan inside the host interval as "
            "a matched submission-overhead control"
        ),
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=32,
        help=(
            "even per-operation event-timed diagnostic operations, completed "
            "before the uninstrumented throughput pass"
        ),
    )
    parser.add_argument(
        "--correctness-iterations",
        type=int,
        default=132,
        help=(
            "even unique-output operations before timing; must exceed twice "
            "the per-bank event-pool depth to prove event reuse"
        ),
    )
    parser.add_argument(
        "--membership",
        default="0,1;0;0,1",
        help="stable sparse phases; every process remains alive as a standby",
    )
    parser.add_argument(
        "--phase-live-tokens",
        help=(
            "semicolon rows with one count per stable rank; omitted selects "
            "unequal counts and includes logical N=0"
        ),
    )
    parser.add_argument("--event-pool-depth", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--jsonl", help="rank-zero aggregate JSONL output")
    parser.add_argument(
        "--fused-oracle-json",
        help="optional immutable fused-oracle JSON attached without reinterpretation",
    )
    parser.add_argument(
        "--codegen-dump-root",
        help=(
            "existing fresh parent for rank-private retained PTX/CUBIN evidence; "
            "compile/admission only"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    devices = tuple(args.devices)
    membership = parse_membership(args.membership, max_ranks=len(devices))
    phase_tokens = parse_phase_live_tokens(
        args.phase_live_tokens,
        membership=membership,
        max_ranks=len(devices),
        capacity=args.token_capacity,
    )
    case = PipelineRunCase(
        max_ranks=len(devices),
        experts_per_rank=args.experts_per_rank,
        token_capacity=args.token_capacity,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        worker_ctas=args.worker_ctas,
        warmup=args.warmup,
        iterations=args.iterations,
        membership=membership,
        phase_live_tokens=phase_tokens,
        event_pool_depth=args.event_pool_depth,
        correctness_iterations=args.correctness_iterations,
        latency_iterations=args.latency_iterations,
        throughput_submission_mode=args.throughput_submission_mode,
    )
    run(
        devices=devices,
        case=case,
        timeout_s=args.timeout,
        jsonl=args.jsonl,
        fused_oracle_json=args.fused_oracle_json,
        codegen_dump_root=args.codegen_dump_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PipelineRunCase",
    "RESULT_PREFIX",
    "StandinExpertExecutor",
    "main",
    "parse_phase_live_tokens",
    "run",
]
