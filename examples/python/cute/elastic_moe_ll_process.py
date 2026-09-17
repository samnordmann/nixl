#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graceful process-elastic orchestration for the persistent NIXL/CuTe MoE.

Unlike :mod:`elastic_moe_ll`, inactive stable slots have no live worker.  A
local coordinator starts a process when a slot joins, keeps it alive while the
slot remains active, and joins it after a graceful removal.  Rejoining the
same slot creates a new process, NIXL agent, CUDA context, memory registration,
and monotonically larger incarnation.

Every membership generation follows this bounded protocol::

    candidate registration/connect/preflight
      -> CANDIDATE_PREPARED
      -> COMMIT / all-rank COMMIT_ACKNOWLEDGED
      -> GO / one existing persistent kernel launch
      -> OLD_PHASE_STREAM_DRAINED
      -> OLD_VIEW_RELEASED
      -> OBSOLETE_IDENTITIES_UNLOADED
      -> retiring owner deregistered
      -> retiring process exited

    candidate/setup failure -> ABORT
      -> all-rank SAFE_TO_SHUTDOWN
      -> SHUTDOWN / owner deregistration and process exit

The coordinator does not start a replacement on a stable GPU slot until the
old owner has deregistered and its process has exited.  All lifecycle work is
outside the persistent kernel's measured loop; there is no added kernel
instruction, host synchronization, or launch in that loop.

This is deliberately a same-node, graceful-transition harness. Actual process
or coordinator loss, failure of its filesystem control channel, ambiguous GO
observation, or cleanup that cannot prove quiescence is catastrophic fail-stop.
It does not claim SIGKILL recovery, continued service during a transition, or
cross-node process management: direct CUDA-IPC access can fault when its owning
process disappears. Joining workers are not pre-provisioned, so this is safety
evidence rather than a production transition-latency reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import tempfile
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence, cast

import torch

try:
    from . import elastic_moe_ll as _ll
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
        wait_until,
    )
    from .elastic_moe_ll import (
        OUTPUT_POISON,
        UCX_NUM_WORKERS,
        UCX_POST_THREADS,
        WORKER_PLAN_ELEMENT_NBYTES,
        WORKER_PLAN_FIELDS,
        ElasticLLCase,
        PhaseSpec,
        PinnedPhaseInputs,
        PinnedPhaseResults,
        _compile_kernels,
        _rank_codegen_dump_directory,
        _rank_phase_result,
        _region,
        _timestamp_buffer_elements,
        _validate_outputs,
        _validate_statuses,
        build_phase_specs,
        parse_membership_plan,
        phase_incarnation_tables,
        summarize_phase_results,
    )
except ImportError:
    import elastic_moe_ll as _ll  # type: ignore[no-redef]
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
        wait_until,
    )
    from elastic_moe_ll import (  # type: ignore[no-redef]
        OUTPUT_POISON,
        UCX_NUM_WORKERS,
        UCX_POST_THREADS,
        WORKER_PLAN_ELEMENT_NBYTES,
        WORKER_PLAN_FIELDS,
        ElasticLLCase,
        PhaseSpec,
        PinnedPhaseInputs,
        PinnedPhaseResults,
        _compile_kernels,
        _rank_codegen_dump_directory,
        _rank_phase_result,
        _region,
        _timestamp_buffer_elements,
        _validate_outputs,
        _validate_statuses,
        build_phase_specs,
        parse_membership_plan,
        phase_incarnation_tables,
        summarize_phase_results,
    )


_PHASE_CONTROL_REGIONS = _ll._PHASE_CONTROL_REGIONS
_COMMAND_SCHEMA = 1
_RESULT_PREFIX = "NIXL_CUTE_PROCESS_ELASTIC_MOE_LL_RESULT "
_LIFECYCLE_PREFIX = "NIXL_CUTE_PROCESS_ELASTIC_LIFECYCLE_RESULT "
_SAFE_TO_SHUTDOWN = "SAFE_TO_SHUTDOWN"
_CLEANUP_FAILED = "CLEANUP_FAILED"

if _ll._CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    import nixl.device.cute as nixl_cute


class LifecycleEvent(str, Enum):
    """Externally observable states in one worker's membership generation."""

    CANDIDATE_PREPARED = "CANDIDATE_PREPARED"
    COMMIT_ACKNOWLEDGED = "COMMIT_ACKNOWLEDGED"
    GO_OBSERVED = "GO_OBSERVED"
    OLD_PHASE_STREAM_DRAINED = "OLD_PHASE_STREAM_DRAINED"
    OLD_VIEW_RELEASED = "OLD_VIEW_RELEASED"
    OBSOLETE_IDENTITIES_UNLOADED = "OBSOLETE_IDENTITIES_UNLOADED"
    OWNER_DEREGISTERED = "OWNER_DEREGISTERED"
    PROCESS_EXITED = "PROCESS_EXITED"


class _WorkerReportedFailure(RuntimeError):
    """A live worker reported an error while retaining its registered owner."""


class _CatastrophicLifecycleFailure(RuntimeError):
    """Global safety cannot be established by the readable control plane."""


class _CandidateAborted(RuntimeError):
    """The coordinator rejected a generation before GO."""


_COMMON_LIFECYCLE = (
    LifecycleEvent.CANDIDATE_PREPARED,
    LifecycleEvent.COMMIT_ACKNOWLEDGED,
    LifecycleEvent.GO_OBSERVED,
    LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
    LifecycleEvent.OLD_VIEW_RELEASED,
    LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED,
)


def validate_lifecycle_trace(
    events: Sequence[LifecycleEvent | str],
    *,
    retiring: bool,
    complete: bool = True,
) -> tuple[LifecycleEvent, ...]:
    """Validate the safety order for a continuing or retiring worker.

    ``complete=False`` validates a runtime prefix. A retiring process may not
    deregister its owner allocation until every old remote view is released and
    every remote identity that will retire has been unloaded. Connections among
    unchanged identities may remain live. ``PROCESS_EXITED`` is observed only
    by the coordinator after a successful ``join``.
    """

    try:
        normalized = tuple(LifecycleEvent(event) for event in events)
    except ValueError as error:
        raise ValueError(f"unknown lifecycle event: {error}") from None
    expected: tuple[LifecycleEvent, ...] = _COMMON_LIFECYCLE
    if retiring:
        expected += (
            LifecycleEvent.OWNER_DEREGISTERED,
            LifecycleEvent.PROCESS_EXITED,
        )
    if len(normalized) > len(expected) or normalized != expected[: len(normalized)]:
        next_event = (
            expected[len(normalized)] if len(normalized) < len(expected) else None
        )
        raise ValueError(
            f"unsafe lifecycle trace {tuple(event.value for event in normalized)!r}; "
            f"expected prefix {tuple(event.value for event in expected)!r}, "
            f"next={getattr(next_event, 'value', None)!r}"
        )
    if complete and normalized != expected:
        raise ValueError(
            f"incomplete lifecycle trace; expected "
            f"{tuple(event.value for event in expected)!r}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Stable slot plus process incarnation and globally unique agent name."""

    slot: int
    incarnation: int
    agent_name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or self.slot < 0
        ):
            raise ValueError("slot must be a non-negative integer")
        if (
            isinstance(self.incarnation, bool)
            or not isinstance(self.incarnation, int)
            or self.incarnation <= 0
        ):
            raise ValueError("incarnation must be a positive integer")
        if not isinstance(self.agent_name, str) or not self.agent_name:
            raise ValueError("agent_name must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ProcessTransition:
    """Coordinator actions and identities for one membership generation."""

    generation: int
    active_ranks: tuple[int, ...]
    joining_ranks: tuple[int, ...]
    continuing_ranks: tuple[int, ...]
    retiring_ranks: tuple[int, ...]
    identities: tuple[ProcessIdentity, ...]


@dataclass(frozen=True, slots=True)
class PeerConnectionDelta:
    """Per-generation connection changes for one persistent worker."""

    new_slots: tuple[int, ...]
    retained_slots: tuple[int, ...]
    remove_after_phase: tuple[int, ...]


def _agent_name(run_id: str, slot: int, incarnation: int) -> str:
    return f"cute_llp_{run_id}_s{slot}_i{incarnation}"


def build_process_transitions(
    membership_plan: Sequence[Sequence[int]],
    max_ranks: int,
    *,
    run_id: str = "plan",
) -> tuple[ProcessTransition, ...]:
    """Build exact start/continue/retire actions for a sparse membership plan."""

    plan = tuple(tuple(phase) for phase in membership_plan)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if isinstance(max_ranks, bool) or not isinstance(max_ranks, int) or max_ranks <= 0:
        raise ValueError("max_ranks must be a positive integer")
    for generation, phase in enumerate(plan):
        if tuple(sorted(set(phase))) != phase:
            raise ValueError(
                f"membership phase {generation} must be sorted and duplicate-free"
            )
    incarnations = phase_incarnation_tables(plan, max_ranks)
    transitions: list[ProcessTransition] = []
    previous: set[int] = set()
    for generation, (phase, table) in enumerate(zip(plan, incarnations)):
        active = set(phase)
        following = set(plan[generation + 1]) if generation + 1 < len(plan) else set()
        identities = tuple(
            ProcessIdentity(rank, table[rank], _agent_name(run_id, rank, table[rank]))
            for rank in phase
        )
        transitions.append(
            ProcessTransition(
                generation=generation,
                active_ranks=phase,
                joining_ranks=tuple(sorted(active - previous)),
                continuing_ranks=tuple(sorted(active & previous)),
                retiring_ranks=tuple(sorted(active - following)),
                identities=identities,
            )
        )
        previous = active
    return tuple(transitions)


def plan_peer_connection_delta(
    *,
    rank: int,
    generation: int,
    phases: Sequence[PhaseSpec],
    peers: Mapping[int, PeerCoordinates],
    loaded_peers: Mapping[int, PeerCoordinates],
) -> PeerConnectionDelta:
    """Plan identity-safe reuse without touching the NIXL control plane."""

    if not 0 <= generation < len(phases):
        raise ValueError("generation is outside the phase plan")
    phase = phases[generation]
    if rank not in phase.active_ranks:
        raise ValueError("rank must be active in the selected generation")
    if set(peers) != set(phase.active_ranks):
        raise ValueError("peer coordinates must cover the active membership exactly")

    expected_remote_slots = set(phase.active_ranks) - {rank}
    stale_slots = set(loaded_peers) - expected_remote_slots
    if stale_slots:
        raise RuntimeError(
            "stale NIXL peer identities survived the prior phase: "
            f"{sorted(stale_slots)}"
        )

    retained: list[int] = []
    new: list[int] = []
    for slot in sorted(expected_remote_slots):
        loaded = loaded_peers.get(slot)
        if loaded is None:
            new.append(slot)
        elif loaded != peers[slot]:
            raise RuntimeError(
                f"stable slot {slot} changed identity or registered coordinates "
                "while its NIXL connection was retained"
            )
        else:
            retained.append(slot)

    keep_after_phase: set[int] = set()
    if generation + 1 < len(phases):
        following = phases[generation + 1]
        if rank in following.active_ranks:
            for slot in expected_remote_slots & set(following.active_ranks):
                if phase.rank_incarnations[slot] != following.rank_incarnations[slot]:
                    raise RuntimeError(
                        f"stable slot {slot} changed incarnation without leaving "
                        "the active membership"
                    )
                keep_after_phase.add(slot)

    return PeerConnectionDelta(
        new_slots=tuple(new),
        retained_slots=tuple(retained),
        remove_after_phase=tuple(sorted(expected_remote_slots - keep_after_phase)),
    )


@dataclass(frozen=True, slots=True)
class PhaseCommand:
    """Strict, non-executable COMMIT/GO payload checked by every worker."""

    run_id: str
    generation: int
    active_ranks: tuple[int, ...]
    rank_incarnations: tuple[int, ...]

    @classmethod
    def from_phase(cls, run_id: str, phase: PhaseSpec) -> PhaseCommand:
        return cls(
            run_id,
            phase.generation,
            phase.active_ranks,
            phase.rank_incarnations,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if not self.active_ranks or tuple(sorted(set(self.active_ranks))) != tuple(
            self.active_ranks
        ):
            raise ValueError("active_ranks must be sorted, unique, and non-empty")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in self.active_ranks
        ):
            raise ValueError("active_ranks must contain non-negative integers")
        if not self.rank_incarnations:
            raise ValueError("rank_incarnations must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.rank_incarnations
        ):
            raise ValueError("rank_incarnations must contain non-negative integers")
        if self.active_ranks[-1] >= len(self.rank_incarnations):
            raise ValueError("active rank exceeds incarnation table")
        if any(self.rank_incarnations[rank] <= 0 for rank in self.active_ranks):
            raise ValueError("every active rank must have a positive incarnation")

    def to_bytes(self) -> bytes:
        document = {
            "schema_version": _COMMAND_SCHEMA,
            "run_id": self.run_id,
            "generation": self.generation,
            "active_ranks": list(self.active_ranks),
            "rank_incarnations": list(self.rank_incarnations),
        }
        return json.dumps(document, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> PhaseCommand:
        if not isinstance(payload, bytes):
            raise TypeError("phase command payload must be bytes")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid phase command payload") from error
        expected_keys = {
            "schema_version",
            "run_id",
            "generation",
            "active_ranks",
            "rank_incarnations",
        }
        if not isinstance(document, dict) or set(document) != expected_keys:
            raise ValueError("unsupported phase command schema")
        if document["schema_version"] != _COMMAND_SCHEMA or isinstance(
            document["schema_version"], bool
        ):
            raise ValueError("unsupported phase command version")
        if not isinstance(document["active_ranks"], list) or not isinstance(
            document["rank_incarnations"], list
        ):
            raise ValueError("phase command rank fields must be arrays")
        return cls(
            document["run_id"],
            document["generation"],
            tuple(document["active_ranks"]),
            tuple(document["rank_incarnations"]),
        )


def _tag(generation: int, event: LifecycleEvent | str) -> str:
    value = event.value if isinstance(event, LifecycleEvent) else event
    return f"g{generation}-{value}"


def _command_path(
    directory: str | os.PathLike[str], generation: int, verb: str
) -> Path:
    return Path(directory) / f"g{generation}-{verb}.command"


def _atomic_write(path: Path, payload: bytes) -> None:
    """Create one command/result atomically; lifecycle files are single-assignment."""

    if path.exists():
        raise RuntimeError(f"refusing to replace single-assignment file {path.name}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_command(
    directory: str | os.PathLike[str], verb: str, command: PhaseCommand
) -> None:
    _atomic_write(
        _command_path(directory, command.generation, verb), command.to_bytes()
    )


def _await_command(
    directory: str | os.PathLike[str],
    verb: str,
    expected: PhaseCommand,
    timeout_s: float,
) -> None:
    path = _command_path(directory, expected.generation, verb)

    def read_command() -> PhaseCommand | None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        return PhaseCommand.from_bytes(payload)

    observed = wait_until(
        read_command,
        timeout_s=timeout_s,
        description=f"generation {expected.generation} {verb}",
        poll_interval_s=0.01,
    )
    if observed != expected:
        raise RuntimeError(
            f"generation {expected.generation} {verb} command does not match "
            "the prepared candidate membership"
        )


def _await_command_choice(
    directory: str | os.PathLike[str],
    verbs: Sequence[str],
    expected: PhaseCommand,
    timeout_s: float,
) -> str:
    """Wait for exactly one coordinator decision and validate its payload."""

    choices = tuple(verbs)
    if not choices or len(set(choices)) != len(choices):
        raise ValueError("command choices must be non-empty and unique")
    paths = {
        verb: _command_path(directory, expected.generation, verb) for verb in choices
    }

    def read_choice() -> tuple[str, PhaseCommand] | None:
        observed: list[tuple[str, PhaseCommand]] = []
        for verb, path in paths.items():
            try:
                payload = path.read_bytes()
            except FileNotFoundError:
                continue
            observed.append((verb, PhaseCommand.from_bytes(payload)))
        if len(observed) > 1:
            if set(choices) == {"COMMIT", "ABORT"} and {
                verb for verb, _ in observed
            } == {"COMMIT", "ABORT"}:
                if any(command != expected for _, command in observed):
                    raise RuntimeError(
                        f"generation {expected.generation} coordinator command "
                        "does not match the candidate"
                    )
                # ABORT is allowed to supersede an already-durable COMMIT until
                # GO is issued. A delayed worker must not park merely because it
                # observes both single-assignment files at once.
                return next(item for item in observed if item[0] == "ABORT")
            raise RuntimeError(
                f"generation {expected.generation} received conflicting coordinator "
                f"commands {[verb for verb, _ in observed]}"
            )
        return observed[0] if observed else None

    verb, command = cast(
        tuple[str, PhaseCommand],
        wait_until(
            read_choice,
            timeout_s=timeout_s,
            description=(
                f"generation {expected.generation} coordinator decision "
                f"{'|'.join(choices)}"
            ),
            poll_interval_s=0.01,
        ),
    )
    if command != expected:
        raise RuntimeError(
            f"generation {expected.generation} {verb} command does not match "
            "the prepared candidate membership"
        )
    return verb


def _raise_if_candidate_aborted(
    directory: str | os.PathLike[str], expected: PhaseCommand
) -> None:
    """Cheaply leave candidate setup after the current bounded operation."""

    path = _command_path(directory, expected.generation, "ABORT")
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return
    if PhaseCommand.from_bytes(payload) != expected:
        raise RuntimeError(
            f"generation {expected.generation} ABORT command does not match candidate"
        )
    raise _CandidateAborted(
        f"coordinator aborted generation {expected.generation} candidate"
    )


def _phase_remote_coordinates(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    local_region: DeviceRegion,
    peers: Mapping[int, PeerCoordinates],
) -> list[tuple[int, int, int, str | None]]:
    """Create fixed slots with registered local-loopback padding."""

    active = set(phase.active_ranks)
    if set(peers) != active:
        raise ValueError("peer coordinate keys must exactly match active ranks")
    coordinates: list[tuple[int, int, int, str | None]] = []
    for slot in range(case.max_ranks):
        if slot in active and slot != rank:
            coordinate = peers[slot]
            if len(coordinate.regions) != 1:
                raise ValueError("every active rank must publish exactly one arena")
            region = coordinate.regions[0]
            name = coordinate.agent_name
        else:
            # Current UCX device lists cannot safely address a NULL_AGENT gap.
            # Reuse this worker's registered loopback extent for inactive and
            # self slots; rank_mask and the dedicated self path exclude them
            # from remote communication without adding hot-loop work.
            region = local_region
            name = peers[rank].agent_name
        coordinates.append((region.address, region.length, region.device_id, name))
    return coordinates


@dataclass(slots=True)
class _WorkerTensors:
    arena: torch.Tensor
    rank_mask: torch.Tensor
    advertised_bases: torch.Tensor
    incarnations: torch.Tensor
    generation_state: torch.Tensor
    bucket_counts: torch.Tensor
    bucket_offsets: torch.Tensor
    worker_plan: torch.Tensor
    route_gates: torch.Tensor
    outputs: torch.Tensor
    statuses: torch.Tensor
    timestamps: torch.Tensor

    def compile_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.arena,
            self.rank_mask,
            self.advertised_bases,
            self.incarnations,
            self.generation_state,
            self.bucket_counts,
            self.bucket_offsets,
            self.worker_plan,
            self.route_gates,
            self.outputs,
            self.statuses,
            self.timestamps,
        )


def _allocate_worker_tensors(
    case: ElasticLLCase, device: int, stream: torch.cuda.Stream
) -> _WorkerTensors:
    layout = case.layout
    with torch.cuda.stream(stream):
        arena = torch.empty(layout.arena_nbytes, dtype=torch.uint8, device=device)
        rank_mask = torch.ones(case.max_ranks, dtype=torch.int64, device=device)
        advertised_bases = torch.zeros(
            case.max_ranks, dtype=torch.uint64, device=device
        )
        incarnations = torch.zeros(case.max_ranks, dtype=torch.int64, device=device)
        generation_state = torch.zeros(1, dtype=torch.int64, device=device)
        bucket_counts = torch.zeros(
            case.max_ranks * case.max_experts, dtype=torch.int64, device=device
        )
        bucket_offsets = torch.zeros_like(bucket_counts)
        worker_plan = torch.zeros(
            case.max_ranks * case.workers_per_peer * WORKER_PLAN_FIELDS,
            dtype=torch.int32,
            device=device,
        )
        route_gates = torch.zeros(
            case.route_capacity, dtype=torch.float32, device=device
        )
        outputs = torch.full(
            (case.output_elements,),
            OUTPUT_POISON,
            dtype=torch.bfloat16,
            device=device,
        )
        statuses = torch.zeros(case.max_ranks, dtype=torch.int32, device=device)
        timestamps = torch.zeros(
            _timestamp_buffer_elements(case),
            # NIXL retags CuTe's signless extern result as the public unsigned
            # timer type with a scalar bitcast that LLVM/NVVM erases.
            dtype=torch.uint64,
            device=device,
        )
    return _WorkerTensors(
        arena,
        rank_mask,
        advertised_bases,
        incarnations,
        generation_state,
        bucket_counts,
        bucket_offsets,
        worker_plan,
        route_gates,
        outputs,
        statuses,
        timestamps,
    )


def _stage_phase(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    tensors: _WorkerTensors,
    host_inputs: PinnedPhaseInputs,
    advertised_bases: Sequence[int],
    stream: torch.cuda.Stream,
) -> None:
    """Enqueue immutable routes and small controls before candidate commit."""

    layout = case.layout
    stage_region = layout.region("dispatch_stage")
    host_inputs.prepare(case, phase, rank, advertised_bases)
    with torch.cuda.stream(stream):
        tensors.arena[stage_region.offset : stage_region.end].copy_(
            host_inputs.dispatch_stage, non_blocking=True
        )
        for region_name in _PHASE_CONTROL_REGIONS:
            region = layout.region(region_name)
            tensors.arena[region.offset : region.end].zero_()
        tensors.rank_mask.copy_(host_inputs.rank_mask, non_blocking=True)
        tensors.advertised_bases.copy_(host_inputs.advertised_bases, non_blocking=True)
        tensors.incarnations.copy_(host_inputs.incarnations, non_blocking=True)
        tensors.generation_state.copy_(host_inputs.generation_state, non_blocking=True)
        tensors.bucket_counts.copy_(host_inputs.bucket_counts, non_blocking=True)
        tensors.bucket_offsets.copy_(host_inputs.bucket_offsets, non_blocking=True)
        tensors.worker_plan.copy_(host_inputs.worker_plan, non_blocking=True)
        tensors.route_gates.copy_(host_inputs.route_gates, non_blocking=True)
        tensors.outputs.fill_(OUTPUT_POISON)
        if case.timing_mode != "none":
            tensors.timestamps.zero_()


def _inactive_rank_result(case: ElasticLLCase, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "active": False,
        "process_present": False,
        "rank_round_starts_ns": [],
        "rank_round_ends_ns": [],
        "rank_observer_span_ns": [],
        "peer_round_ns": [],
        "remote_peer_round_ns": [],
        "per_peer_round_ns": {},
        "operation_steps": 0,
        "timing_mode": case.timing_mode,
        "correctness": "PASS",
    }


def _publish_event(
    control: FileControlPlane,
    phase: PhaseSpec,
    trace: list[LifecycleEvent],
    event: LifecycleEvent,
    *,
    retiring: bool,
    barrier: bool = False,
) -> None:
    trace.append(event)
    validate_lifecycle_trace(trace, retiring=retiring, complete=False)
    if barrier:
        control.barrier(_tag(phase.generation, event), ranks=phase.active_ranks)
    else:
        control.publish(_tag(phase.generation, event))


def _merge_phase_error(
    current: Exception | None, operation: str, error: Exception
) -> Exception:
    """Preserve the first failure while retaining later cleanup context."""

    if current is None:
        return RuntimeError(f"{operation} failed: {error}")
    return RuntimeError(f"{current}; {operation} also failed: {error}")


def _release_phase_views(
    phase_views: ExitStack, remote_view: object | None
) -> tuple[bool, Exception | None]:
    """Release a drained view, retrying the handle if native release failed.

    ``nixl_device_view_handle.release`` deliberately leaves a failed handle
    valid. ExitStack has already popped its callback when that happens, so an
    explicit retry is required before any peer owner can be retired.
    """

    first_error: Exception | None = None
    try:
        phase_views.close()
    except Exception as error:
        first_error = error
    if remote_view is None or not getattr(remote_view, "valid", False):
        return True, first_error
    try:
        getattr(remote_view, "release")()
    except Exception as retry_error:
        return False, _merge_phase_error(
            first_error,
            "explicit NIXL device-view release retry",
            retry_error,
        )
    return not getattr(remote_view, "valid", False), first_error


def _report_worker_failure(
    error_path: Path,
    *,
    rank: int,
    incarnation: int,
    generation: int,
    stage: str,
    error: Exception,
) -> None:
    """Publish the first recoverable worker failure without exiting its owner."""

    if error_path.exists():
        return
    document = {
        "rank": rank,
        "incarnation": incarnation,
        "pid": os.getpid(),
        "generation": generation,
        "stage": stage,
        "error": repr(error),
    }
    _atomic_write(error_path, json.dumps(document, sort_keys=True).encode("utf-8"))


def _park_registered_owner(timeout_s: float) -> None:
    """Retain a mapped owner until the coordinator performs whole-job fail-stop."""

    while True:
        time.sleep(min(60.0, max(1.0, timeout_s)))


def _report_or_park(
    error_path: Path,
    *,
    rank: int,
    incarnation: int,
    generation: int,
    stage: str,
    error: Exception,
    timeout_s: float,
) -> None:
    try:
        _report_worker_failure(
            error_path,
            rank=rank,
            incarnation=incarnation,
            generation=generation,
            stage=stage,
            error=error,
        )
    except Exception:
        _park_registered_owner(timeout_s)


def _await_shutdown(
    directory: str | os.PathLike[str], command: PhaseCommand, timeout_s: float
) -> None:
    """Wait for global safety; a local timeout must never destroy an owner."""

    while True:
        try:
            _await_command(directory, "SHUTDOWN", command, timeout_s)
            return
        except Exception:
            # A worker that already advertised its allocation may not infer
            # global safety from a local filesystem timeout or malformed file.
            time.sleep(min(1.0, timeout_s))


def _await_worker_decision(
    *,
    directory: str | os.PathLike[str],
    verbs: Sequence[str],
    command: PhaseCommand,
    timeout_s: float,
    error_path: Path,
    rank: int,
    incarnation: int,
    stage: str,
) -> str:
    """Wait without allowing a rank-local command timeout to drop its owner."""

    while True:
        try:
            return _await_command_choice(directory, verbs, command, timeout_s)
        except TimeoutError:
            # The coordinator owns the bounded decision timeout and will write
            # ABORT before GO. A worker-local timeout is not itself a failure;
            # continuing to wait retains the owner without creating a race in
            # which GO arrives immediately after a failure report.
            continue
        except Exception as error:
            try:
                _report_worker_failure(
                    error_path,
                    rank=rank,
                    incarnation=incarnation,
                    generation=command.generation,
                    stage=stage,
                    error=error,
                )
            finally:
                # Conflicting or malformed coordinator decisions cannot be
                # resolved locally once peers may hold this owner's address.
                _park_registered_owner(timeout_s)


def _shutdown_failed_worker(
    *,
    agent: Any,
    registration: Any,
    control: FileControlPlane,
    command: PhaseCommand,
    phase_views: ExitStack,
    remote_view: object | None,
    loaded_peers: dict[int, PeerCoordinates],
    stream: torch.cuda.Stream,
    stream_drained: bool,
    error_path: Path,
    rank: int,
    incarnation: int,
    failure: Exception,
    stage: str,
    timeout_s: float,
) -> None:
    """Quiesce one failed worker and retain its owner through global shutdown."""

    try:
        _report_worker_failure(
            error_path,
            rank=rank,
            incarnation=incarnation,
            generation=command.generation,
            stage=stage,
            error=failure,
        )
    except Exception:
        # The failure channel itself is part of the explicit catastrophic
        # boundary. Keep the registered CUDA owner alive for job-level abort.
        _park_registered_owner(timeout_s)

    cleanup_error: Exception | None = None
    if not stream_drained:
        try:
            stream.synchronize()
        except Exception as error:
            cleanup_error = _merge_phase_error(
                cleanup_error, "failure-path CUDA stream drain", error
            )

    released, release_error = _release_phase_views(phase_views, remote_view)
    if release_error is not None:
        cleanup_error = _merge_phase_error(
            cleanup_error, "failure-path NIXL device-view release", release_error
        )

    if released:
        # A failed run will not retain any connection into another owner. Keep
        # trying every known identity even if one removal fails.
        for peer_rank, peer in tuple(sorted(loaded_peers.items())):
            try:
                agent.remove_remote_agent(peer.agent_name)
                del loaded_peers[peer_rank]
            except Exception as error:
                cleanup_error = _merge_phase_error(
                    cleanup_error,
                    f"failure-path NIXL identity removal for slot {peer_rank}",
                    error,
                )

    if not released or loaded_peers:
        if cleanup_error is None:
            cleanup_error = RuntimeError("failure cleanup did not reach a safe state")
        try:
            control.publish(
                _tag(command.generation, _CLEANUP_FAILED),
                _ll._phase_status_payload(rank, cleanup_error),
            )
        finally:
            _park_registered_owner(timeout_s)

    try:
        control.publish(
            _tag(command.generation, _SAFE_TO_SHUTDOWN),
            _ll._phase_status_payload(rank, cleanup_error),
        )
    except Exception:
        _park_registered_owner(timeout_s)

    _await_shutdown(directory=control.directory, command=command, timeout_s=timeout_s)
    agent.deregister_memory(registration, backends=["UCX"])


def _run_worker(
    rank: int,
    incarnation: int,
    start_generation: int,
    devices: tuple[int, ...],
    directory: str,
    run_id: str,
    case: ElasticLLCase,
    timeout_s: float,
) -> None:
    if not _ll._CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for this example")

    from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    phases = build_phase_specs(case)
    if rank not in phases[start_generation].active_ranks:
        raise RuntimeError("worker start generation does not contain its stable slot")
    expected_incarnation = phases[start_generation].rank_incarnations[rank]
    if incarnation != expected_incarnation:
        raise RuntimeError("worker incarnation does not match the membership plan")

    codegen_dump_dir = _rank_codegen_dump_directory(
        str(Path(directory) / "codegen"), rank, incarnation=incarnation
    )
    device = devices[rank]
    torch.cuda.set_device(device)
    native_atomic_preflight = nixl_cute.require_peer_native_atomics(
        devices, accessing_devices=(device,)
    )
    stream = torch.cuda.Stream(device=device)
    layout = case.layout
    if layout.arena_nbytes > torch.cuda.get_device_properties(device).total_memory:
        raise ValueError("fixed-capacity LL arena exceeds total GPU memory")
    tensors = _allocate_worker_tensors(case, device, stream)
    host_inputs = PinnedPhaseInputs.allocate(case)
    host_results = PinnedPhaseResults.allocate(case)

    control = FileControlPlane(directory, rank, case.max_ranks, timeout_s)
    identity = ProcessIdentity(
        rank, incarnation, _agent_name(run_id, rank, incarnation)
    )
    agent = nixl_agent(
        identity.agent_name,
        nixl_agent_config(
            # The committed LL phase is mapped-only. Synchronous setup calls
            # progress UCX themselves; a zero-delay background progress thread
            # would otherwise busy-poll beside the persistent GPU kernel.
            enable_prog_thread=False,
            num_threads=UCX_POST_THREADS,
            backends=[],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_DEFAULT,
        ),
    )
    agent.create_backend(
        "UCX",
        {
            "num_threads": str(UCX_POST_THREADS),
            "num_workers": str(UCX_NUM_WORKERS),
            "ucx_num_device_channels": str(max(1, case.experts_per_rank)),
        },
    )
    registration = agent.register_memory([tensors.arena], backends=["UCX"])
    owner_registered = True
    local_coordinates = PeerCoordinates(identity.agent_name, (_region(tensors.arena),))
    compiled_preflight, compiled_main, occupancy = _compile_kernels(
        case, rank, tensors.compile_args(), stream, codegen_dump_dir
    )
    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "device": device,
        "native_peer_atomics": native_atomic_preflight,
        "pid": os.getpid(),
        "stable_slot": rank,
        "incarnation": incarnation,
        "agent_name": identity.agent_name,
        "nixl_cute_abi": nixl_cute.NIXL_CUTE_ABI_VERSION,
        "cooperative_occupancy": occupancy.as_dict(),
        "codegen_dump_dir": str(codegen_dump_dir),
    }
    # Key remote state by stable slot, but validate the full agent name,
    # incarnation, and registered coordinates before every reuse. Continuing
    # peers keep their NIXL metadata and UCX connection across generations.
    loaded_peers: dict[int, PeerCoordinates] = {}
    error_path = Path(directory) / f"worker-error-s{rank}-i{incarnation}"

    generation = start_generation
    try:
        while generation < len(phases) and rank in phases[generation].active_ranks:
            phase = phases[generation]
            if phase.rank_incarnations[rank] != incarnation:
                raise RuntimeError("a live worker cannot silently change incarnation")
            following = (
                set(phases[generation + 1].active_ranks)
                if generation + 1 < len(phases)
                else set()
            )
            retiring = rank not in following
            trace: list[LifecycleEvent] = []
            command = PhaseCommand.from_phase(run_id, phase)

            expected_identities = {
                peer_rank: _agent_name(
                    run_id, peer_rank, phase.rank_incarnations[peer_rank]
                )
                for peer_rank in phase.active_ranks
            }
            expected_remote_slots = set(phase.active_ranks) - {rank}
            control_plane_exchange_skipped = set(
                loaded_peers
            ) == expected_remote_slots and all(
                loaded_peers[slot].agent_name == expected_identities[slot]
                for slot in expected_remote_slots
            )
            phase_views = ExitStack()
            mapped_preflight: dict[str, object] | None = None
            preflight_error: Exception | None = None
            remote_view = None
            connection_delta: PeerConnectionDelta | None = None
            try:
                _raise_if_candidate_aborted(directory, command)
                if control_plane_exchange_skipped:
                    # Registration coordinates and full identities are immutable
                    # for a persistent worker. Reuse them on unchanged or
                    # removal-only generations instead of two filesystem exchanges.
                    metadata: dict[int, bytes] = {}
                    peers = {rank: local_coordinates, **loaded_peers}
                else:
                    metadata = control.exchange(
                        _tag(generation, "agent-metadata"),
                        agent.get_agent_metadata(),
                        ranks=phase.active_ranks,
                    )
                    _raise_if_candidate_aborted(directory, command)
                    serialized_coordinates = control.exchange(
                        _tag(generation, "arena-coordinates"),
                        local_coordinates.to_bytes(),
                        ranks=phase.active_ranks,
                    )
                    _raise_if_candidate_aborted(directory, command)
                    peers = {
                        peer_rank: PeerCoordinates.from_bytes(payload)
                        for peer_rank, payload in serialized_coordinates.items()
                    }
                for peer_rank, peer in peers.items():
                    if peer.agent_name != expected_identities[peer_rank]:
                        raise RuntimeError(
                            f"stable slot {peer_rank} published unexpected agent "
                            f"{peer.agent_name!r}"
                        )

                connection_delta = plan_peer_connection_delta(
                    rank=rank,
                    generation=generation,
                    phases=phases,
                    peers=peers,
                    loaded_peers=loaded_peers,
                )
                for peer_rank in connection_delta.new_slots:
                    expected_name = expected_identities[peer_rank]
                    loaded_name = normalize_agent_name(
                        agent.add_remote_agent(metadata[peer_rank])
                    )
                    if loaded_name != expected_name:
                        # Retain the actually loaded identity so ABORT cleanup
                        # can invalidate it even though candidate validation fails.
                        loaded_peers[peer_rank] = PeerCoordinates(
                            loaded_name, peers[peer_rank].regions
                        )
                        raise RuntimeError(
                            f"loaded unexpected NIXL agent {loaded_name!r}"
                        )
                    loaded_peers[peer_rank] = peers[peer_rank]

                if set(loaded_peers) != expected_remote_slots:
                    raise AssertionError(
                        "loaded NIXL peers do not match active membership"
                    )
                new_peer_names = [
                    loaded_peers[peer_rank].agent_name
                    for peer_rank in connection_delta.new_slots
                ]

                if new_peer_names:
                    # One rendezvous ensures every joining identity is loaded
                    # before endpoint creation. Any setup failure, including a
                    # rendezvous timeout, is reported to the coordinator while
                    # this owner remains registered for ABORT cleanup.
                    control.barrier(
                        _tag(generation, "metadata-loaded"), ranks=phase.active_ranks
                    )
                    _raise_if_candidate_aborted(directory, command)
                    for peer_name in new_peer_names:
                        agent.make_connection(peer_name, backends=["UCX"])
                    complete_ucx_setup_handshake(
                        agent,
                        control,
                        expected_identities,
                        generation=generation,
                        nonce=run_id,
                        notification_peer_ranks=connection_delta.new_slots,
                        timeout_s=timeout_s,
                        poll_callback=lambda: _raise_if_candidate_aborted(
                            directory, command
                        ),
                    )
                    _raise_if_candidate_aborted(directory, command)

                advertised_bases = tuple(
                    peers[slot].regions[0].address if slot in peers else 0
                    for slot in range(case.max_ranks)
                )
                _stage_phase(
                    case,
                    phase,
                    rank,
                    tensors,
                    host_inputs,
                    advertised_bases,
                    stream,
                )
                # Candidate staging and mapped preflight share this stream.
                # The one post-preflight drain covers both operations.
                composite = _phase_remote_coordinates(
                    case,
                    phase,
                    rank,
                    local_coordinates.regions[0],
                    peers,
                )
                remote_view = phase_views.enter_context(
                    agent.prepare_device_view(
                        composite,
                        mem_type="VRAM",
                        backend="UCX",
                        connection_timeout_ms=max(1, int(timeout_s * 1000)),
                    )
                )
                compiled_preflight(
                    remote_view,
                    from_dlpack(tensors.rank_mask).mark_layout_dynamic(),
                    from_dlpack(tensors.advertised_bases).mark_layout_dynamic(),
                    from_dlpack(tensors.statuses).mark_layout_dynamic(),
                    cuda.CUstream(stream.cuda_stream),
                )
                host_results.enqueue_preflight(tensors.statuses, stream)
            except Exception as error:
                preflight_error = error
            # CUDA stream synchronization drains queued candidate work even
            # when it reports an earlier asynchronous operation failure. It is
            # also required after a partial staging/launch exception.
            try:
                stream.synchronize()
            except Exception as drain_error:
                preflight_error = _merge_phase_error(
                    preflight_error, "candidate CUDA stream drain", drain_error
                )
            if preflight_error is None:
                try:
                    mapped_preflight = _ll._classify_mapped_preflight(
                        host_results.statuses,
                        phase,
                        rank,
                        case.allow_unverified_mapped,
                    )
                except Exception as error:
                    preflight_error = error
            if preflight_error is None and (
                mapped_preflight is None
                or remote_view is None
                or connection_delta is None
            ):
                preflight_error = AssertionError(
                    "successful preflight has incomplete candidate state"
                )

            if preflight_error is None:
                try:
                    _publish_event(
                        control,
                        phase,
                        trace,
                        LifecycleEvent.CANDIDATE_PREPARED,
                        retiring=retiring,
                    )
                except Exception as error:
                    preflight_error = _merge_phase_error(
                        preflight_error,
                        "candidate-prepared evidence publication",
                        error,
                    )
            if preflight_error is not None:
                _report_or_park(
                    error_path,
                    rank=rank,
                    incarnation=incarnation,
                    generation=generation,
                    stage="candidate-prepare",
                    error=preflight_error,
                    timeout_s=timeout_s,
                )

            decision = _await_worker_decision(
                directory=directory,
                verbs=("COMMIT", "ABORT"),
                command=command,
                timeout_s=timeout_s,
                error_path=error_path,
                rank=rank,
                incarnation=incarnation,
                stage="candidate-decision",
            )
            if preflight_error is not None and decision == "COMMIT":
                # PREPARED publication can succeed immediately before a later
                # local failure is reported. Never acknowledge that candidate.
                decision = _await_worker_decision(
                    directory=directory,
                    verbs=("ABORT",),
                    command=command,
                    timeout_s=timeout_s,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    stage="candidate-abort-after-commit-race",
                )
            if decision == "ABORT":
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=True,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=(
                        preflight_error
                        if preflight_error is not None
                        else RuntimeError("coordinator aborted candidate preparation")
                    ),
                    stage="candidate-abort",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return

            assert mapped_preflight is not None
            assert remote_view is not None
            assert connection_delta is not None
            with torch.cuda.stream(stream):
                tensors.statuses.zero_()

            commit_error: Exception | None = None
            try:
                _publish_event(
                    control,
                    phase,
                    trace,
                    LifecycleEvent.COMMIT_ACKNOWLEDGED,
                    retiring=retiring,
                )
            except Exception as error:
                commit_error = error
                _report_or_park(
                    error_path,
                    rank=rank,
                    incarnation=incarnation,
                    generation=generation,
                    stage="commit-acknowledgement",
                    error=error,
                    timeout_s=timeout_s,
                )
            decision = _await_worker_decision(
                directory=directory,
                verbs=("GO", "ABORT"),
                command=command,
                timeout_s=timeout_s,
                error_path=error_path,
                rank=rank,
                incarnation=incarnation,
                stage="go-decision",
            )
            if decision == "ABORT":
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=False,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=(
                        commit_error
                        if commit_error is not None
                        else RuntimeError("coordinator aborted committed candidate")
                    ),
                    stage="commit-abort",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return

            # Publish durable evidence that this worker observed GO, then wait
            # on the same per-rank markers before the sole measured launch.
            # The barrier adds no second write: it proves every worker's marker
            # fsync finished before any persistent GPU interval can begin.
            phase_error: Exception | None = commit_error
            try:
                _publish_event(
                    control,
                    phase,
                    trace,
                    LifecycleEvent.GO_OBSERVED,
                    retiring=retiring,
                    barrier=True,
                )
            except Exception as error:
                phase_error = _merge_phase_error(
                    phase_error, "GO_OBSERVED evidence publication", error
                )
            try:
                compiled_main(
                    remote_view,
                    from_dlpack(tensors.arena).mark_layout_dynamic(),
                    from_dlpack(tensors.rank_mask).mark_layout_dynamic(),
                    from_dlpack(tensors.incarnations).mark_layout_dynamic(),
                    from_dlpack(tensors.generation_state).mark_layout_dynamic(),
                    from_dlpack(tensors.bucket_counts).mark_layout_dynamic(),
                    from_dlpack(tensors.bucket_offsets).mark_layout_dynamic(),
                    from_dlpack(tensors.worker_plan).mark_layout_dynamic(),
                    from_dlpack(tensors.route_gates).mark_layout_dynamic(),
                    from_dlpack(tensors.outputs).mark_layout_dynamic(),
                    from_dlpack(tensors.statuses).mark_layout_dynamic(),
                    from_dlpack(tensors.timestamps).mark_layout_dynamic(),
                    cuda.CUstream(stream.cuda_stream),
                )
                host_results.enqueue_main(
                    tensors.statuses,
                    tensors.outputs,
                    tensors.timestamps,
                    stream,
                    copy_timestamps=case.timing_mode != "none",
                )
            except Exception as error:
                phase_error = _merge_phase_error(
                    phase_error, "main launch or result enqueue", error
                )
            try:
                stream.synchronize()
            except Exception as error:
                # cudaStreamSynchronize drains prior work before returning an
                # asynchronous error. A kernel that never makes progress still
                # relies on the external scheduler timeout.
                phase_error = _merge_phase_error(
                    phase_error, "main CUDA stream drain", error
                )
            try:
                _publish_event(
                    control,
                    phase,
                    trace,
                    LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
                    retiring=retiring,
                )
            except Exception as error:
                phase_error = _merge_phase_error(
                    phase_error, "stream-drained evidence publication", error
                )

            main_stream_drained = True
            released, release_error = _release_phase_views(phase_views, remote_view)
            if release_error is not None:
                phase_error = _merge_phase_error(
                    phase_error, "NIXL device-view release", release_error
                )
            if not released:
                failure = phase_error or RuntimeError(
                    "NIXL device view remained valid after explicit release retry"
                )
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=main_stream_drained,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=failure,
                    stage="post-go-view-release",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return
            else:
                # Publish only after native release succeeded. CPU validation and
                # serialization happen later and cannot extend mapping lifetime.
                try:
                    _publish_event(
                        control,
                        phase,
                        trace,
                        LifecycleEvent.OLD_VIEW_RELEASED,
                        retiring=retiring,
                    )
                except Exception as error:
                    phase_error = _merge_phase_error(
                        phase_error, "view-released evidence publication", error
                    )

            # Remove only identities that will not continue unchanged into the
            # next generation. Program order is local stream drain -> view
            # release -> identity removal; the one unload barrier
            # below makes all three globally complete before owner retirement.
            removal_error: Exception | None = None
            for peer_rank in connection_delta.remove_after_phase:
                peer_name = loaded_peers[peer_rank].agent_name
                try:
                    agent.remove_remote_agent(peer_name)
                    del loaded_peers[peer_rank]
                except Exception as error:
                    removal_error = _merge_phase_error(
                        removal_error,
                        f"obsolete NIXL identity removal for slot {peer_rank}",
                        error,
                    )
            needs_unload_convergence = retiring or bool(
                connection_delta.remove_after_phase
            )
            if needs_unload_convergence:
                try:
                    unload_statuses = control.exchange(
                        _tag(
                            generation,
                            LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED,
                        ),
                        _ll._phase_status_payload(rank, removal_error),
                        ranks=phase.active_ranks,
                    )
                    unload_failures = _ll._decode_phase_failures(
                        unload_statuses, phase.active_ranks
                    )
                    if unload_failures:
                        phase_error = _merge_phase_error(
                            phase_error,
                            "obsolete NIXL identity removal",
                            RuntimeError(str(unload_failures)),
                        )
                    else:
                        trace.append(LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED)
                        validate_lifecycle_trace(
                            trace, retiring=retiring, complete=False
                        )
                except Exception as error:
                    phase_error = _merge_phase_error(
                        phase_error, "remote-unload convergence", error
                    )
            else:
                try:
                    _publish_event(
                        control,
                        phase,
                        trace,
                        LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED,
                        retiring=retiring,
                    )
                except Exception as error:
                    phase_error = _merge_phase_error(
                        phase_error,
                        "obsolete-identities-unloaded evidence publication",
                        error,
                    )

            if phase_error is not None:
                _report_or_park(
                    error_path,
                    rank=rank,
                    incarnation=incarnation,
                    generation=generation,
                    stage="post-go-cleanup",
                    error=phase_error,
                    timeout_s=timeout_s,
                )

            # A terminal-operation error need not be imported by a peer that
            # already passed its final device wait. Converge success/failure
            # only after every old view has been released.
            validation_error = phase_error
            rank_result: dict[str, object] | None = None
            if validation_error is None:
                try:
                    _validate_statuses(host_results.statuses, phase.active_ranks)
                    _validate_outputs(case, phase, rank, host_results.outputs)
                    built_result = _rank_phase_result(
                        case, phase, rank, host_results.timestamps
                    )
                    built_result["environment"] = {
                        **environment,
                        "mapped_preflight": mapped_preflight,
                        "connection_delta": {
                            "control_plane_exchange_skipped": (
                                control_plane_exchange_skipped
                            ),
                            "new_peer_slots": list(connection_delta.new_slots),
                            "retained_peer_slots": list(
                                connection_delta.retained_slots
                            ),
                            "removed_peer_slots": list(
                                connection_delta.remove_after_phase
                            ),
                        },
                    }
                    built_result["process_present"] = True
                    rank_result = built_result
                except Exception as error:
                    validation_error = error
            try:
                gathered = control.exchange(
                    _tag(generation, "rank-result"),
                    _ll._phase_result_payload(rank, validation_error, rank_result),
                    ranks=phase.active_ranks,
                )
            except Exception as error:
                failure = _merge_phase_error(
                    validation_error, "combined rank-result exchange", error
                )
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=main_stream_drained,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=failure,
                    stage="rank-result-exchange",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return
            try:
                validation_failures, decoded = _ll._decode_phase_result_exchange(
                    gathered, phase.active_ranks
                )
            except Exception as error:
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=main_stream_drained,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=error,
                    stage="rank-result-decode",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return
            if validation_failures:
                failure = RuntimeError(
                    f"generation {generation} failed after the active-rank "
                    f"kernel drain: {validation_failures}"
                )
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=main_stream_drained,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=failure,
                    stage="post-go-phase-result",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return
            if rank == min(phase.active_ranks):
                try:
                    all_slots = [
                        decoded.get(slot, _inactive_rank_result(case, slot))
                        for slot in range(case.max_ranks)
                    ]
                    result = summarize_phase_results(case, phase, all_slots)
                    result["case"] = "nixl_cute_process_elastic_moe_ll"
                    result["process_elasticity"] = {
                        "mode": "same-node graceful OS-process membership",
                        "candidate_prepare_before_commit": True,
                        "commit_ack_before_go": True,
                        "continuing_nixl_connections_retained": True,
                        "inactive_slots_have_live_process": False,
                        "stable_slot_incarnations": list(phase.rank_incarnations),
                        "agent_names": [
                            expected_identities[peer] for peer in phase.active_ranks
                        ],
                        "post_go_process_loss": "fails the complete run",
                        "continued_service_during_transition": False,
                        "abrupt_failure_recovery": False,
                        "cross_node_orchestration": False,
                    }
                    result["elasticity"] = {
                        "stable_sparse_descriptor_slots": True,
                        "view_swap_between_drained_phases": True,
                        "graceful_process_exit_and_rejoin": True,
                        "all_fixed_capacity_processes_alive": False,
                        "abrupt_failure_recovery": False,
                        "abrupt_mapped_owner_loss": (
                            "unsupported; direct CUDA-IPC access may fault"
                        ),
                    }
                    result["arena"] = {
                        "bytes_per_live_rank": layout.arena_nbytes,
                        "layout": "CompactLLArenaLayout",
                        "fixed_rank_capacity": case.max_ranks,
                        "phase_copy_bytes": layout.region("dispatch_stage").nbytes,
                        "worker_plan_fields": WORKER_PLAN_FIELDS,
                        "worker_plan_layout": (
                            "int32[task,outgoing_begin,outgoing_count,"
                            "incoming_begin,incoming_count]"
                        ),
                        "worker_plan_bytes_per_phase": (
                            case.max_ranks
                            * case.workers_per_peer
                            * WORKER_PLAN_FIELDS
                            * WORKER_PLAN_ELEMENT_NBYTES
                        ),
                        "worker_plan_staging": (
                            "pinned asynchronous H2D on the existing phase stream; "
                            "no added synchronization or launch"
                        ),
                        "phase_zero_bytes": sum(
                            layout.region(name).nbytes
                            for name in _PHASE_CONTROL_REGIONS
                        ),
                    }
                    result["backend_parameters"] = agent.get_backend_params("UCX")
                    print(
                        _RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True
                    )
                except Exception as error:
                    _shutdown_failed_worker(
                        agent=agent,
                        registration=registration,
                        control=control,
                        command=command,
                        phase_views=phase_views,
                        remote_view=remote_view,
                        loaded_peers=loaded_peers,
                        stream=stream,
                        stream_drained=main_stream_drained,
                        error_path=error_path,
                        rank=rank,
                        incarnation=incarnation,
                        failure=error,
                        stage="phase-result-reporting",
                        timeout_s=timeout_s,
                    )
                    owner_registered = False
                    return

            if retiring:
                # The coordinator emits RETIRE only after observing that every
                # worker released its old view and unloaded retiring identities.
                _await_command(directory, "RETIRE", command, timeout_s)
                agent.deregister_memory(registration, backends=["UCX"])
                owner_registered = False
                _publish_event(
                    control,
                    phase,
                    trace,
                    LifecycleEvent.OWNER_DEREGISTERED,
                    retiring=True,
                )
                validate_lifecycle_trace(trace, retiring=True, complete=False)
                return

            try:
                validate_lifecycle_trace(trace, retiring=False)
            except Exception as error:
                _shutdown_failed_worker(
                    agent=agent,
                    registration=registration,
                    control=control,
                    command=command,
                    phase_views=phase_views,
                    remote_view=remote_view,
                    loaded_peers=loaded_peers,
                    stream=stream,
                    stream_drained=main_stream_drained,
                    error_path=error_path,
                    rank=rank,
                    incarnation=incarnation,
                    failure=error,
                    stage="lifecycle-validation",
                    timeout_s=timeout_s,
                )
                owner_registered = False
                return
            generation += 1
    except BaseException as error:
        if owner_registered:
            unexpected = (
                error if isinstance(error, Exception) else RuntimeError(repr(error))
            )
            _report_or_park(
                error_path,
                rank=rank,
                incarnation=incarnation,
                generation=generation,
                stage="unexpected-registered-owner-failure",
                error=unexpected,
                timeout_s=timeout_s,
            )
            # The structured paths above normally quiesce the phase. An
            # unexpected escape cannot prove peer-view release, so retain the
            # registration for explicit whole-job fail-stop.
            _park_registered_owner(timeout_s)
        raise
    finally:
        # Normal retirement and coordinated failure shutdown are explicit.
        if owner_registered:
            pass


def _worker_entry(*args) -> None:
    rank = int(args[0])
    incarnation = int(args[1])
    directory = str(args[4])
    error_path = Path(directory) / f"worker-error-s{rank}-i{incarnation}"
    try:
        _run_worker(*args)
    except BaseException as error:
        document = {
            "rank": rank,
            "incarnation": incarnation,
            "pid": os.getpid(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_write(
                error_path,
                json.dumps(document, sort_keys=True).encode("utf-8"),
            )
        finally:
            raise


@dataclass(slots=True)
class _WorkerHandle:
    identity: ProcessIdentity
    process: Any
    error_path: Path


def _wait_for_markers(
    directory: str | os.PathLike[str],
    generation: int,
    event: LifecycleEvent,
    ranks: Sequence[int],
    handles: Mapping[int, _WorkerHandle],
    timeout_s: float,
    *,
    go_issued: bool,
) -> None:
    pending = set(ranks)
    deadline = time.monotonic() + timeout_s
    while pending:
        for rank in ranks:
            handle = handles[rank]
            if handle.error_path.exists():
                detail = handle.error_path.read_text(encoding="utf-8")
                if getattr(handle.process, "exitcode", None) is not None:
                    raise _CatastrophicLifecycleFailure(
                        f"stable slot {rank} exited after reporting {detail}"
                    )
                raise _WorkerReportedFailure(
                    f"live stable slot {rank} reported failure: {detail}"
                )
        for rank in tuple(pending):
            marker = Path(directory) / f"{_tag(generation, event)}.{rank}"
            if marker.exists():
                pending.remove(rank)
        for rank in tuple(pending):
            handle = handles[rank]
            exitcode = getattr(handle.process, "exitcode", None)
            if exitcode is not None:
                qualifier = "post-GO process loss" if go_issued else "candidate failure"
                raise _CatastrophicLifecycleFailure(
                    f"{qualifier}: stable slot {rank} exited with code {exitcode} "
                    f"before {event.value}"
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            qualifier = " after GO" if go_issued else ""
            raise TimeoutError(
                f"timed out{qualifier} waiting for generation {generation} "
                f"{event.value} from ranks {sorted(pending)}"
            )
        time.sleep(min(0.02, remaining))


def _wait_for_shutdown_safety(
    directory: str | os.PathLike[str],
    generation: int,
    ranks: Sequence[int],
    handles: Mapping[int, _WorkerHandle],
    timeout_s: float,
) -> None:
    """Wait until every live owner is locally drained, unmapped, and unloaded."""

    pending = set(ranks)
    deadline = time.monotonic() + timeout_s
    while pending:
        for rank in tuple(pending):
            cleanup_failure = (
                Path(directory) / f"{_tag(generation, _CLEANUP_FAILED)}.{rank}"
            )
            if cleanup_failure.exists():
                detail = cleanup_failure.read_text(encoding="utf-8")
                raise _CatastrophicLifecycleFailure(
                    f"stable slot {rank} could not reach shutdown safety: {detail}"
                )
            handle = handles[rank]
            if getattr(handle.process, "exitcode", None) is not None:
                raise _CatastrophicLifecycleFailure(
                    f"stable slot {rank} exited before global SHUTDOWN"
                )
            safe = Path(directory) / f"{_tag(generation, _SAFE_TO_SHUTDOWN)}.{rank}"
            if safe.exists():
                pending.remove(rank)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _CatastrophicLifecycleFailure(
                f"timed out waiting for generation {generation} shutdown safety "
                f"from ranks {sorted(pending)}"
            )
        time.sleep(min(0.02, remaining))


def _wait_for_rank_result_payloads(
    directory: str | os.PathLike[str],
    generation: int,
    ranks: Sequence[int],
    handles: Mapping[int, _WorkerHandle],
    timeout_s: float,
) -> dict[int, bytes]:
    """Observe the existing worker result exchange before advancing membership."""

    pending = set(ranks)
    payloads: dict[int, bytes] = {}
    deadline = time.monotonic() + timeout_s
    while pending:
        for rank in ranks:
            handle = handles[rank]
            if handle.error_path.exists():
                detail = handle.error_path.read_text(encoding="utf-8")
                if getattr(handle.process, "exitcode", None) is not None:
                    raise _CatastrophicLifecycleFailure(
                        f"stable slot {rank} exited after reporting {detail}"
                    )
                raise _WorkerReportedFailure(
                    f"live stable slot {rank} reported failure: {detail}"
                )
        for rank in tuple(pending):
            path = Path(directory) / f"{_tag(generation, 'rank-result')}.{rank}"
            try:
                payloads[rank] = path.read_bytes()
            except FileNotFoundError:
                continue
            pending.remove(rank)
        for rank in tuple(pending):
            if getattr(handles[rank].process, "exitcode", None) is not None:
                raise _CatastrophicLifecycleFailure(
                    f"stable slot {rank} exited before publishing its rank result"
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _CatastrophicLifecycleFailure(
                f"timed out waiting for generation {generation} rank results "
                f"from ranks {sorted(pending)}"
            )
        time.sleep(min(0.02, remaining))
    return payloads


def _join_retired(
    handles: dict[int, _WorkerHandle], ranks: Sequence[int], timeout_s: float
) -> list[dict[str, object]]:
    exited: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout_s
    for rank in ranks:
        handle = handles[rank]
        remaining = deadline - time.monotonic()
        handle.process.join(max(0.0, remaining))
        if handle.process.is_alive():
            raise TimeoutError(f"stable slot {rank} did not exit after deregistration")
        if handle.process.exitcode != 0:
            detail = (
                handle.error_path.read_text(encoding="utf-8")
                if handle.error_path.exists()
                else "no worker traceback was published"
            )
            raise RuntimeError(
                f"stable slot {rank} failed while exiting with code "
                f"{handle.process.exitcode}: {detail}"
            )
        exited.append(
            {
                "slot": rank,
                "incarnation": handle.identity.incarnation,
                "agent_name": handle.identity.agent_name,
                "pid": handle.process.pid,
                "last_event": LifecycleEvent.PROCESS_EXITED.value,
            }
        )
        del handles[rank]
    return exited


def _join_shutdown_workers(
    handles: dict[int, _WorkerHandle], ranks: Sequence[int], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    for rank in ranks:
        handle = handles[rank]
        handle.process.join(max(0.0, deadline - time.monotonic()))
        if handle.process.is_alive():
            raise _CatastrophicLifecycleFailure(
                f"stable slot {rank} did not exit after global SHUTDOWN"
            )
        if handle.process.exitcode != 0:
            raise _CatastrophicLifecycleFailure(
                f"stable slot {rank} failed while executing global SHUTDOWN "
                f"with code {handle.process.exitcode}"
            )
        del handles[rank]


def _coordinate_failed_phase_shutdown(
    *,
    directory: str,
    command: PhaseCommand,
    ranks: Sequence[int],
    handles: dict[int, _WorkerHandle],
    timeout_s: float,
    before_go: bool,
) -> None:
    """Abort a candidate, then release every owner only after the SAFE quorum."""

    if before_go:
        _write_command(directory, "ABORT", command)
    _wait_for_shutdown_safety(directory, command.generation, ranks, handles, timeout_s)
    _write_command(directory, "SHUTDOWN", command)
    _join_shutdown_workers(handles, ranks, timeout_s)


def _wait_for_progress_or_shutdown(
    *,
    directory: str,
    phase: PhaseSpec,
    event: LifecycleEvent,
    handles: dict[int, _WorkerHandle],
    timeout_s: float,
    command: PhaseCommand,
    before_go: bool,
) -> None:
    try:
        _wait_for_markers(
            directory,
            phase.generation,
            event,
            phase.active_ranks,
            handles,
            timeout_s,
            go_issued=not before_go,
        )
    except _WorkerReportedFailure:
        _coordinate_failed_phase_shutdown(
            directory=directory,
            command=command,
            ranks=phase.active_ranks,
            handles=handles,
            timeout_s=timeout_s,
            before_go=before_go,
        )
        raise
    except TimeoutError:
        if not before_go:
            raise
        # No candidate may have observed GO. ABORT converts a missing PREPARED
        # or ACK into the same safe failure-only teardown.
        _coordinate_failed_phase_shutdown(
            directory=directory,
            command=command,
            ranks=phase.active_ranks,
            handles=handles,
            timeout_s=timeout_s,
            before_go=True,
        )
        raise


def _abort_workers(handles: Mapping[int, _WorkerHandle]) -> None:
    workers = tuple(handles.values())
    for handle in workers:
        if handle.process.is_alive():
            handle.process.terminate()
    terminate_deadline = time.monotonic() + 10.0
    for handle in workers:
        handle.process.join(max(0.0, terminate_deadline - time.monotonic()))
    survivors = tuple(handle for handle in workers if handle.process.is_alive())
    for handle in survivors:
        # These are only child processes created by this coordinator. A hard
        # kill is the final cleanup fallback after the shared SIGTERM budget;
        # the run is already failed and must not leave a CUDA owner alive.
        handle.process.kill()
    kill_deadline = time.monotonic() + 10.0
    for handle in survivors:
        handle.process.join(max(0.0, kill_deadline - time.monotonic()))
        if handle.process.is_alive():
            raise RuntimeError(
                f"failed to stop worker process {handle.process.pid} within 20s"
            )


def _validate_run_inputs(
    devices: tuple[int, ...], case: ElasticLLCase, timeout_s: float
) -> None:
    if len(devices) != case.max_ranks or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain one distinct device per stable slot")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a positive finite number")
    _ll._require_diagnostic_timeout_margin(case, float(timeout_s))
    if not torch.cuda.is_available() or torch.cuda.device_count() < case.max_ranks:
        raise RuntimeError(f"this run requires {case.max_ranks} visible CUDA devices")
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device indices must be integers")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")


def run(
    *, devices: tuple[int, ...], case: ElasticLLCase, timeout_s: float
) -> dict[str, object]:
    """Run a bounded, same-node process-elastic membership plan."""

    _validate_run_inputs(devices, case, timeout_s)
    case = _ll._resolve_case_geometry(case, devices)
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    phases = build_phase_specs(case)
    with TemporaryDirectory(
        prefix=f"nixl_cute_elastic_ll_process_{os.getpid()}_"
    ) as directory:
        (Path(directory) / "codegen").mkdir(mode=0o700)
        run_id = hashlib.sha256(directory.encode("utf-8")).hexdigest()[:12]
        transitions = build_process_transitions(
            case.membership_plan, case.max_ranks, run_id=run_id
        )
        handles: dict[int, _WorkerHandle] = {}
        phase_records: list[dict[str, object]] = []
        previous_phase_stream_drained_ns: int | None = None
        try:
            for phase, transition in zip(phases, transitions):
                identity_by_rank = {
                    identity.slot: identity for identity in transition.identities
                }
                for rank in transition.joining_ranks:
                    identity = identity_by_rank[rank]
                    process = context.Process(
                        target=_worker_entry,
                        args=(
                            rank,
                            identity.incarnation,
                            phase.generation,
                            devices,
                            directory,
                            run_id,
                            case,
                            float(timeout_s),
                        ),
                        name=f"nixl-cute-slot-{rank}-i{identity.incarnation}",
                        daemon=False,
                    )
                    process.start()
                    handles[rank] = _WorkerHandle(
                        identity,
                        process,
                        Path(directory)
                        / f"worker-error-s{rank}-i{identity.incarnation}",
                    )

                if set(handles) != set(phase.active_ranks):
                    raise RuntimeError(
                        "live worker slots do not match phase membership"
                    )
                command = PhaseCommand.from_phase(run_id, phase)
                _wait_for_progress_or_shutdown(
                    directory=directory,
                    phase=phase,
                    event=LifecycleEvent.CANDIDATE_PREPARED,
                    handles=handles,
                    timeout_s=timeout_s,
                    command=command,
                    before_go=True,
                )
                candidate_prepared_ns = time.monotonic_ns()
                _write_command(directory, "COMMIT", command)
                commit_issued_ns = time.monotonic_ns()
                _wait_for_progress_or_shutdown(
                    directory=directory,
                    phase=phase,
                    event=LifecycleEvent.COMMIT_ACKNOWLEDGED,
                    handles=handles,
                    timeout_s=timeout_s,
                    command=command,
                    before_go=True,
                )
                commit_acknowledged_ns = time.monotonic_ns()
                _write_command(directory, "GO", command)
                go_issued_ns = time.monotonic_ns()
                _wait_for_progress_or_shutdown(
                    directory=directory,
                    phase=phase,
                    event=LifecycleEvent.GO_OBSERVED,
                    handles=handles,
                    timeout_s=timeout_s,
                    command=command,
                    before_go=False,
                )
                go_observed_ns = time.monotonic_ns()
                event_observed_ns: dict[LifecycleEvent, int] = {}
                for event in (
                    LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
                    LifecycleEvent.OLD_VIEW_RELEASED,
                    LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED,
                ):
                    _wait_for_progress_or_shutdown(
                        directory=directory,
                        phase=phase,
                        event=event,
                        handles=handles,
                        timeout_s=timeout_s,
                        command=command,
                        before_go=False,
                    )
                    event_observed_ns[event] = time.monotonic_ns()
                stream_and_snapshot_drained_ns = event_observed_ns[
                    LifecycleEvent.OLD_PHASE_STREAM_DRAINED
                ]

                try:
                    coordinator_results = _wait_for_rank_result_payloads(
                        directory,
                        phase.generation,
                        phase.active_ranks,
                        handles,
                        timeout_s,
                    )
                    coordinator_failures = _ll._decode_phase_failures(
                        coordinator_results, phase.active_ranks
                    )
                    if coordinator_failures:
                        raise _WorkerReportedFailure(
                            f"generation {phase.generation} worker results reported "
                            f"{coordinator_failures}"
                        )
                except _WorkerReportedFailure:
                    _coordinate_failed_phase_shutdown(
                        directory=directory,
                        command=command,
                        ranks=phase.active_ranks,
                        handles=handles,
                        timeout_s=timeout_s,
                        before_go=False,
                    )
                    raise

                active_processes = [
                    {
                        "slot": rank,
                        "incarnation": identity_by_rank[rank].incarnation,
                        "agent_name": identity_by_rank[rank].agent_name,
                        "pid": handles[rank].process.pid,
                    }
                    for rank in phase.active_ranks
                ]
                retired: list[dict[str, object]] = []
                if transition.retiring_ranks:
                    _write_command(directory, "RETIRE", command)
                    _wait_for_markers(
                        directory,
                        phase.generation,
                        LifecycleEvent.OWNER_DEREGISTERED,
                        transition.retiring_ranks,
                        handles,
                        timeout_s,
                        go_issued=True,
                    )
                    retired = _join_retired(
                        handles, transition.retiring_ranks, timeout_s
                    )
                    validate_lifecycle_trace(
                        (
                            *_COMMON_LIFECYCLE,
                            LifecycleEvent.OWNER_DEREGISTERED,
                            LifecycleEvent.PROCESS_EXITED,
                        ),
                        retiring=True,
                    )

                phase_records.append(
                    {
                        "generation": phase.generation,
                        "active_ranks": list(phase.active_ranks),
                        "joining_ranks": list(transition.joining_ranks),
                        "continuing_ranks": list(transition.continuing_ranks),
                        "retiring_ranks": list(transition.retiring_ranks),
                        "active_processes": active_processes,
                        "retired_processes": retired,
                        "protocol": [event.value for event in _COMMON_LIFECYCLE],
                        "control_plane_timing": {
                            "clock": "coordinator time.monotonic_ns",
                            "configured_marker_poll_interval_ns": 20_000_000,
                            "candidate_prepared_ns": candidate_prepared_ns,
                            "commit_issued_ns": commit_issued_ns,
                            "commit_acknowledged_ns": commit_acknowledged_ns,
                            "go_issued_ns": go_issued_ns,
                            "go_observed_ns": go_observed_ns,
                            "stream_and_snapshot_drained_ns": (
                                stream_and_snapshot_drained_ns
                            ),
                            "view_released_ns": event_observed_ns[
                                LifecycleEvent.OLD_VIEW_RELEASED
                            ],
                            "obsolete_identities_unloaded_ns": event_observed_ns[
                                LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED
                            ],
                            "coordinator_go_issue_gap_after_observed_stream_drain_ns": (
                                None
                                if previous_phase_stream_drained_ns is None
                                else go_issued_ns - previous_phase_stream_drained_ns
                            ),
                        },
                    }
                )
                previous_phase_stream_drained_ns = stream_and_snapshot_drained_ns
            if handles:
                raise RuntimeError("final generation left live workers behind")
        except BaseException:
            _abort_workers(handles)
            raise

    result = {
        "schema_version": 1,
        "case": "nixl_cute_process_elastic_lifecycle",
        "membership_plan": [list(phase) for phase in case.membership_plan],
        "phases": phase_records,
        "same_node_only": True,
        "graceful_transitions_only": True,
        "elasticity_mode": "break-before-make drained generation swap",
        "online_serving_during_transition": False,
        "post_go_process_loss": "FAIL",
        "kernel_hot_path_modified": False,
        "correctness": "PASS",
    }
    print(_LIFECYCLE_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--workers-per-peer",
        type=int,
        default=0,
        help=(
            "fixed logical worker warps per stable peer slot; inactive peer "
            "slots remain local combine workers (0 bounds the total by expert "
            "tasks, tokens, and a 128-warp large-case tuning ceiling)"
        ),
    )
    parser.add_argument(
        "--warps-per-cta",
        type=int,
        default=0,
        help="logical worker warps packed into each cooperative CTA",
    )
    parser.add_argument(
        "--num-banks",
        type=int,
        choices=(1, 2),
        default=1,
        help="communication banks; two is available for performance A/B",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--membership",
        default="0,1;0;0,1",
        help=(
            "semicolon-delimited sparse phases; inactive slots have no worker "
            "and rejoin creates a new process incarnation"
        ),
    )
    parser.add_argument("--no-empty-last-expert", action="store_true")
    parser.add_argument(
        "--allow-unverified-mapped",
        action="store_true",
        help=(
            "deprecated compatibility no-op; mapped execution is qualified by "
            "non-null process-local pointers and directed native peer atomics"
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("none", "envelope", "cadence", "peer"),
        default="none",
        help=(
            "compile-time timing policy: none is uninstrumented; envelope emits "
            "two phase markers; cadence emits round starts; peer adds diagnostics"
        ),
    )
    parser.add_argument(
        "--instrument-per-peer",
        action="store_true",
        help="compatibility alias for --timing-mode peer",
    )
    parser.add_argument(
        "--validate-every-iteration",
        action="store_true",
        help=(
            "qualification mode with operation-varying source payloads and "
            "per-iteration output preservation"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-lifecycle-step timeout; also use an external Slurm timeout",
    )
    parser.add_argument(
        "--device-timeout-ms",
        type=int,
        default=0,
        help=(
            "opt-in diagnostic per-wait bound; zero uses the timer-free "
            "production peer-abort path"
        ),
    )
    args = parser.parse_args()
    devices = tuple(args.devices)
    plan = parse_membership_plan(args.membership, len(devices))
    case = ElasticLLCase(
        max_ranks=len(devices),
        experts_per_rank=args.experts_per_rank,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        warmup=args.warmup,
        iterations=args.iterations,
        membership_plan=plan,
        empty_last_expert=not args.no_empty_last_expert,
        device_timeout_ns=args.device_timeout_ms * 1_000_000,
        allow_unverified_mapped=args.allow_unverified_mapped,
        workers_per_peer=args.workers_per_peer,
        warps_per_cta=args.warps_per_cta,
        instrument_per_peer=args.instrument_per_peer,
        validate_every_iteration=args.validate_every_iteration,
        num_banks=args.num_banks,
        timing_mode=args.timing_mode,
        target_sm=(_ll._detect_common_sm(devices) if args.warps_per_cta == 0 else None),
    )
    run(devices=devices, case=case, timeout_s=args.timeout)


if __name__ == "__main__":
    main()


__all__ = [
    "LifecycleEvent",
    "PhaseCommand",
    "ProcessIdentity",
    "ProcessTransition",
    "build_process_transitions",
    "main",
    "run",
    "validate_lifecycle_trace",
]
