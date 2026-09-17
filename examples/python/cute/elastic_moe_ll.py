#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent, mapped-memory NIXL/CuTe low-latency MoE communication example.

This is the performance-oriented companion to ``elastic_moe.py``.  Every
membership phase runs one persistent CuTe kernel per rank.  Its inner loop has
no CPU synchronization or kernel launch and implements the complete
communication lifecycle:

``dispatch -> stand-in expert -> reverse combine -> next-round handshakes``.

The registered arena has stable rank slots and one bank by default; a two-bank
specialization remains available for performance comparison. A descriptor keeps
the same index across membership changes; inactive and self slots use the local
registered loopback region because current UCX device-list gaps are unsafe to
address. Separate release-published stamps carry membership
generation, operation step, and source-membership incarnation once per dispatch
bucket and once per aggregate combine peer.  Dispatch ready counters publish
``record_count + 1`` even for empty expert buckets; combine joins all local
experts and publishes their aggregate count once per peer.  A bank is not
reused until the reverse ready handshake proves that every prior consumer has
finished; the closed round protocol needs no separate remote credit.

The hot path intentionally requires a locally mapped peer pointer.  Dispatch
uses ``nixl_cute.mapped_copy_warp_ptr_readonly`` after resolving each stable
peer once. Host preflight first requires native peer atomics for every directed
pair of participating GPUs. Device preflight then proves that every active
descriptor returns a non-null process-local pointer and records whether CUDA
happened to map it at the owner's numeric address. Numeric equality across
process address spaces is neither expected nor required: all accesses use the
pointer returned by ``nixlGetPtr`` plus offsets within the registered arena.
``--allow-unverified-mapped`` remains a deprecated compatibility no-op.
The deterministic BF16 expert
writes its result directly into the origin's mapped combine bank. GPU-scope
expert joins feed one system-release publication per peer; one peer-leader
system-acquire plus a cooperative grid barrier makes all combine payloads
visible to the reducers.  A correct requestless network fallback would require
the available device release fence plus a separate ordered deferred-PUT,
publish, and credit path.  That transport path is not implemented here, so this
example fails preflight instead of silently mixing protocols for a non-mapped
peer.

Expansion, contraction, and standby rejoin are graceful: all fixed-capacity
processes stay alive and registered while the host stages a new sparse view,
drains the preceding persistent kernel, swaps the view, and advances the
generation/incarnation table. Abrupt process loss and process replacement are
not supported. Production waits are timer-free and inspect local and peer abort
epochs only after repeated ready misses; a positive diagnostic timeout is an
explicit opt-in. Device-detected protocol failures cascade through those abort
epochs, but a kernel that never launches cannot announce itself from device
code and remains an external-job-fatal condition. A direct CUDA-IPC load or
store may fault if its mapped owner exits. Host view release is retried through
the handle's idempotent API; if a live view or cross-rank cleanup state cannot
be disproved, workers deliberately fail-stop with registrations intact until
the complete external allocation is stopped. Production timing is also a
compile-time policy: the default ``none`` specialization emits no timer read,
timestamp store, or timing-only grid barrier; envelope, cadence, and per-peer
specializations add progressively more explicit observation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shlex
import socket
import struct
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import torch

try:
    from ._cooperative import bind_and_validate_cooperative_launch
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from .moe.ll_protocol import (
        DISPATCH_METADATA_NBYTES,
        MESSAGE_STAMP_NBYTES,
        CompactLLArenaLayout,
        StableSparseTopology,
    )
except ImportError:
    from _cooperative import (  # type: ignore[no-redef]
        bind_and_validate_cooperative_launch,
    )
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from moe.ll_protocol import (  # type: ignore[no-redef]
        DISPATCH_METADATA_NBYTES,
        MESSAGE_STAMP_NBYTES,
        CompactLLArenaLayout,
        StableSparseTopology,
    )


ELEMENT_NBYTES = 2
WARP_SIZE = 32
BF16_PER_VECTOR = 8
PAYLOAD_VECTOR_NBYTES = BF16_PER_VECTOR * ELEMENT_NBYTES
EXPERT_VECTOR_UNROLL = 4
COMBINE_VECTOR_UNROLL = 2
MAX_RANKS = 32
TARGET_FIXED_WORKER_WARPS = 128
TARGET_WARPS_PER_CTA = 8
# The exact two-rank production specialization below has a measured SM100
# geometry policy: p8 was a strict loser and p2 was the unique directionally
# faster candidate inside the 1%-margin proven-noninferior set. Other shapes
# and architectures retain the portable exact-divisor search.
TUNED_TARGET_WARPS_PER_CTA = 2
TUNED_TARGET_SM = 100
MAX_FIXED_WORKER_WARPS = 1024
MAX_WARPS_PER_CTA = 32
WORKER_PLAN_FIELDS = 5
WORKER_PLAN_ELEMENT_NBYTES = 4
UCX_NUM_WORKERS = 2
UCX_POST_THREADS = 0
OUTPUT_POISON = -512.0
DRAIN_TIMEOUT_MARGIN_S = 10.0
_UINT32_MAX = (1 << 32) - 1
_INT64_MAX = (1 << 63) - 1
_MAX_DEVICE_TIMEOUT_NS = _INT64_MAX
_INT32_MAX = (1 << 31) - 1
_STATUS_MISMATCH = -5
_STATUS_NOT_SUPPORTED = -9
_PHASE_CONTROL_REGIONS = (
    "dispatch_ready",
    "combine_consumed",
    "abort_state",
    "combine_ready",
)

try:
    _CUTE_AVAILABLE = importlib.util.find_spec("cutlass.cute") is not None
except ModuleNotFoundError:
    _CUTE_AVAILABLE = False

if _CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    import nixl.device.cute as nixl_cute


def _require_diagnostic_timeout_margin(case: "ElasticLLCase", timeout_s: float) -> None:
    """Check a best-effort lifecycle margin for diagnostic timed waits.

    Production mode is timer-free and relies on graceful, all-rank draining.
    A positive device timeout is diagnostic only: this margin gives peers time
    to observe that timeout and rendezvous, but cannot make abrupt mapped-owner
    loss recoverable.
    """

    if case.device_timeout_ns == 0:
        return
    minimum = case.device_timeout_ns / 1_000_000_000 + DRAIN_TIMEOUT_MARGIN_S
    if timeout_s <= minimum:
        raise ValueError(
            "diagnostic control-plane timeout must exceed the device wait timeout "
            f"by more than {DRAIN_TIMEOUT_MARGIN_S:g}s; this is a best-effort "
            "drain margin, not abrupt mapped-owner recovery"
        )


def parse_membership_plan(value: str, max_ranks: int) -> tuple[tuple[int, ...], ...]:
    """Parse ``"0,1;0;0,1"`` into stable, sparse membership phases."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("membership plan must be a non-empty string")
    if isinstance(max_ranks, bool) or not isinstance(max_ranks, int):
        raise TypeError("max_ranks must be an integer")
    if not 1 <= max_ranks <= MAX_RANKS:
        raise ValueError(f"max_ranks must be in [1, {MAX_RANKS}]")
    phases: list[tuple[int, ...]] = []
    for phase_text in value.split(";"):
        fields = [field.strip() for field in phase_text.split(",")]
        if any(not field for field in fields):
            raise ValueError("every membership phase must contain at least one rank")
        try:
            ranks = tuple(int(field, 10) for field in fields)
        except ValueError as error:
            raise ValueError(
                f"invalid rank in membership phase {phase_text!r}"
            ) from error
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"membership phase {phase_text!r} contains duplicates")
        if any(rank < 0 or rank >= max_ranks for rank in ranks):
            raise ValueError(
                f"membership phase {phase_text!r} exceeds fixed rank capacity "
                f"[0, {max_ranks})"
            )
        phases.append(tuple(sorted(ranks)))
    return tuple(phases)


def phase_incarnation_tables(
    plan: Sequence[Sequence[int]], max_ranks: int
) -> tuple[tuple[int, ...], ...]:
    """Assign a larger incarnation whenever a standby rank rejoins.

    First activation uses incarnation one.  Remaining active across a phase
    boundary preserves it; inactive-to-active transitions increment it.
    """

    normalized = tuple(tuple(phase) for phase in plan)
    if not normalized:
        raise ValueError("membership plan must contain at least one phase")
    incarnations = [0] * max_ranks
    previous: set[int] = set()
    tables: list[tuple[int, ...]] = []
    for phase_index, phase in enumerate(normalized):
        current = set(phase)
        if not current:
            raise ValueError(f"membership phase {phase_index} is empty")
        if len(current) != len(phase):
            raise ValueError(f"membership phase {phase_index} contains duplicates")
        for rank in current:
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError("membership ranks must be integers")
            if not 0 <= rank < max_ranks:
                raise ValueError(f"rank {rank} exceeds fixed rank capacity")
            if rank not in previous:
                incarnations[rank] += 1
        tables.append(tuple(incarnations))
        previous = current
    return tuple(tables)


def _detect_common_sm(devices: Sequence[int]) -> int | None:
    """Return one common CUDA SM target without synchronizing a device stream."""

    if not torch.cuda.is_available():
        return None
    capabilities = {
        tuple(int(part) for part in torch.cuda.get_device_capability(device))
        for device in devices
    }
    if len(capabilities) != 1:
        return None
    major, minor = capabilities.pop()
    return major * 10 + minor


@dataclass(frozen=True, slots=True)
class ElasticLLCase:
    """Compile-time shape and persistent-loop configuration."""

    max_ranks: int
    experts_per_rank: int
    num_tokens: int
    top_k: int
    hidden_size: int
    warmup: int
    iterations: int
    membership_plan: tuple[tuple[int, ...], ...]
    empty_last_expert: bool = True
    device_timeout_ns: int = 0
    allow_unverified_mapped: bool = False
    workers_per_peer: int = 0
    warps_per_cta: int = 0
    instrument_per_peer: bool = False
    validate_every_iteration: bool = False
    num_banks: int = 1
    timing_mode: str = "none"
    target_sm: int | None = None
    _warps_per_cta_is_auto: bool = field(
        init=False, repr=False, compare=False, default=False
    )

    def __post_init__(self) -> None:
        for name in (
            "max_ranks",
            "experts_per_rank",
            "num_tokens",
            "top_k",
            "hidden_size",
            "iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_ranks > MAX_RANKS:
            raise ValueError(f"max_ranks must not exceed {MAX_RANKS}")
        if (
            isinstance(self.workers_per_peer, bool)
            or not isinstance(self.workers_per_peer, int)
            or self.workers_per_peer < 0
        ):
            raise ValueError("workers_per_peer must be a non-negative integer")
        if self.workers_per_peer == 0:
            # Do not launch idle cooperative CTAs merely to occupy a fixed
            # device-wide target. Every active (peer, expert) needs a leader,
            # while output work exposes at most one useful CTA per token. Cap
            # the automatic shape at the measured large-case tuning target;
            # explicit overrides remain available for architecture sweeps.
            useful_worker_warps = max(
                self.max_ranks * self.experts_per_rank,
                min(self.num_tokens, TARGET_FIXED_WORKER_WARPS),
            )
            object.__setattr__(
                self,
                "workers_per_peer",
                max(
                    self.experts_per_rank,
                    (useful_worker_warps + self.max_ranks - 1) // self.max_ranks,
                ),
            )
        if self.workers_per_peer < self.experts_per_rank:
            raise ValueError("workers_per_peer must cover every fixed local expert")
        if self.max_ranks * self.workers_per_peer > MAX_FIXED_WORKER_WARPS:
            raise ValueError(
                "max_ranks * workers_per_peer must not exceed "
                f"{MAX_FIXED_WORKER_WARPS} logical worker warps"
            )
        fixed_worker_count = self.max_ranks * self.workers_per_peer
        if self.target_sm is not None and (
            isinstance(self.target_sm, bool)
            or not isinstance(self.target_sm, int)
            or self.target_sm <= 0
        ):
            raise ValueError("target_sm must be a positive integer or None")
        if (
            isinstance(self.warps_per_cta, bool)
            or not isinstance(self.warps_per_cta, int)
            or self.warps_per_cta < 0
        ):
            raise ValueError("warps_per_cta must be a non-negative integer")
        object.__setattr__(self, "_warps_per_cta_is_auto", self.warps_per_cta == 0)
        if self.warps_per_cta == 0:
            # Keep each active task's adjacent shard warps together without
            # concentrating one warp on every SM. The exact divisor guarantees
            # a branch-free launch with no partial CTA. Only the exact measured
            # SM100 production signature uses its qualified p2 tie-break;
            # eight warps (256 threads) is the portable starting point.
            portable_candidate = min(TARGET_WARPS_PER_CTA, fixed_worker_count)
            use_tuned_geometry = (
                self.target_sm == TUNED_TARGET_SM
                and self.max_ranks == 2
                and self.experts_per_rank == 4
                and self.num_tokens == 128
                and self.top_k == 8
                and self.hidden_size == 7168
                and self.workers_per_peer == 64
                and self.membership_plan == ((0, 1),)
                and not self.empty_last_expert
                and self.device_timeout_ns == 0
                and not self.instrument_per_peer
                and not self.validate_every_iteration
                and self.num_banks == 1
                and self.timing_mode == "none"
            )
            candidate = (
                TUNED_TARGET_WARPS_PER_CTA if use_tuned_geometry else portable_candidate
            )
            while fixed_worker_count % candidate:
                candidate -= 1
            object.__setattr__(self, "warps_per_cta", candidate)
        if self.warps_per_cta > MAX_WARPS_PER_CTA:
            raise ValueError(f"warps_per_cta must not exceed {MAX_WARPS_PER_CTA}")
        if fixed_worker_count % self.warps_per_cta:
            raise ValueError(
                "max_ranks * workers_per_peer must be divisible by warps_per_cta"
            )
        if self.num_tokens > _UINT32_MAX - fixed_worker_count + 1:
            raise ValueError(
                "num_tokens is too large for wrap-safe uint32 worker stepping"
            )
        if self.top_k > WARP_SIZE:
            raise ValueError(f"top_k must not exceed warp size {WARP_SIZE}")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int):
            raise ValueError("warmup must be a non-negative integer")
        if self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if not isinstance(self.empty_last_expert, bool):
            raise TypeError("empty_last_expert must be bool")
        if not isinstance(self.allow_unverified_mapped, bool):
            raise TypeError("allow_unverified_mapped must be bool")
        if not isinstance(self.instrument_per_peer, bool):
            raise TypeError("instrument_per_peer must be bool")
        if self.timing_mode not in {"none", "envelope", "cadence", "peer"}:
            raise ValueError(
                "timing_mode must be 'none', 'envelope', 'cadence', or 'peer'"
            )
        # Preserve the original diagnostic boolean as a compatibility alias,
        # while keeping one canonical compile-time timing mode.
        if self.instrument_per_peer:
            if self.timing_mode not in {"none", "peer"}:
                raise ValueError("instrument_per_peer conflicts with timing_mode")
            object.__setattr__(self, "timing_mode", "peer")
        elif self.timing_mode == "peer":
            object.__setattr__(self, "instrument_per_peer", True)
        if not isinstance(self.validate_every_iteration, bool):
            raise TypeError("validate_every_iteration must be bool")
        if (
            isinstance(self.num_banks, bool)
            or not isinstance(self.num_banks, int)
            or self.num_banks not in (1, 2)
        ):
            raise ValueError("num_banks must be one or two")
        if (
            isinstance(self.device_timeout_ns, bool)
            or not isinstance(self.device_timeout_ns, int)
            or self.device_timeout_ns < 0
            or self.device_timeout_ns > _MAX_DEVICE_TIMEOUT_NS
        ):
            raise ValueError("device_timeout_ns must fit CuTe's signed literal range")
        if self.hidden_size > _UINT32_MAX:
            raise ValueError("hidden_size exceeds uint32")
        if self.hidden_size % BF16_PER_VECTOR:
            raise ValueError(
                "hidden_size must be divisible by 8 for 128-bit BF16 payload IO"
            )
        if self.num_tokens > _UINT32_MAX - 1:
            raise ValueError("num_tokens + 1 must fit in uint32 publication field")
        if self.route_capacity > _UINT32_MAX - 1:
            raise ValueError(
                "uint32 route capacity must include the aggregate publication "
                "sentinel (num_tokens * top_k + 1)"
            )
        if self.num_tokens > _INT32_MAX:
            raise ValueError(
                "num_tokens exceeds the signed Int32 range required by CuTe "
                "dynamic record loops"
            )
        if self.total_iterations > _UINT32_MAX:
            raise ValueError("warmup + iterations exceeds uint32")
        if self.validate_every_iteration and self.total_iterations > 64:
            raise ValueError(
                "validate_every_iteration supports at most 64 total iterations"
            )

        plan = tuple(tuple(phase) for phase in self.membership_plan)
        if not plan:
            raise ValueError("membership_plan must contain at least one phase")
        for index, phase in enumerate(plan):
            if not phase:
                raise ValueError(f"membership phase {index} is empty")
            if tuple(sorted(set(phase))) != phase:
                raise ValueError(
                    f"membership phase {index} must be sorted and duplicate-free"
                )
            if any(rank < 0 or rank >= self.max_ranks for rank in phase):
                raise ValueError(f"membership phase {index} exceeds fixed capacity")
            active_experts = len(phase) * self.experts_per_rank
            if self.top_k > active_experts:
                raise ValueError(
                    f"top_k {self.top_k} exceeds the {active_experts} active "
                    f"experts in membership phase {index}"
                )
        object.__setattr__(self, "membership_plan", plan)

        # Constructing the checked protocol layout is also the authoritative
        # overflow/alignment validation for every byte offset used by the JIT.
        _ = self.layout

    @property
    def total_iterations(self) -> int:
        return self.warmup + self.iterations

    @property
    def max_tokens_per_rank(self) -> int:
        # The stage is a flat array of route records. Origin token indices
        # remain below num_tokens; each of their top-k routes needs a record.
        return self.route_capacity

    @property
    def route_capacity(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    @property
    def output_copies(self) -> int:
        return self.iterations if self.validate_every_iteration else 1

    @property
    def output_elements(self) -> int:
        return self.output_copies * self.num_tokens * self.hidden_size

    @property
    def layout(self) -> CompactLLArenaLayout:
        return CompactLLArenaLayout(
            max_ranks=self.max_ranks,
            experts_per_rank=self.experts_per_rank,
            num_tokens=self.num_tokens,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            element_size=ELEMENT_NBYTES,
            workers_per_peer=self.workers_per_peer,
            dispatch_stage_copies=(
                self.total_iterations if self.validate_every_iteration else 1
            ),
            num_banks=self.num_banks,
        )


def _resolve_case_geometry(
    case: ElasticLLCase, devices: Sequence[int]
) -> ElasticLLCase:
    """Bind an automatic launch shape to the selected devices' common SM."""

    if not isinstance(case, ElasticLLCase):
        raise TypeError("case must be an ElasticLLCase")
    if not case._warps_per_cta_is_auto:
        return case
    # Construction resolves zero to a portable provisional launch shape so a
    # caller can inspect a complete case without touching CUDA. Capability
    # queries do not synchronize a CUDA stream. A mixed or unavailable target
    # resolves to ``None`` and retains portable geometry. Explicit geometry
    # never enters this path.
    return replace(
        case,
        target_sm=_detect_common_sm(devices),
        warps_per_cta=0,
    )


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    generation: int
    active_ranks: tuple[int, ...]
    rank_incarnations: tuple[int, ...]

    @property
    def rank_mask(self) -> tuple[int, ...]:
        active = set(self.active_ranks)
        return tuple(
            0 if rank in active else 1 for rank in range(len(self.rank_incarnations))
        )


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    """One host-staged top-k route encoded in a dispatch record."""

    origin_token: int
    route_slot: int
    global_expert: int
    gate: float


@dataclass(frozen=True, slots=True)
class PhaseRouteState:
    """Dense GPU inputs plus bucket-packed routes for one source rank."""

    bucket_counts: tuple[int, ...]
    bucket_offsets: tuple[int, ...]
    route_experts: tuple[int, ...]
    route_gates: tuple[float, ...]
    packed_routes: tuple[RouteAssignment, ...]
    worker_plan: tuple[int, ...]


@dataclass(slots=True)
class PinnedPhaseInputs:
    """Persistent page-locked host buffers for asynchronous phase staging."""

    dispatch_stage: torch.Tensor
    rank_mask: torch.Tensor
    advertised_bases: torch.Tensor
    incarnations: torch.Tensor
    generation_state: torch.Tensor
    bucket_counts: torch.Tensor
    bucket_offsets: torch.Tensor
    worker_plan: torch.Tensor
    route_gates: torch.Tensor

    @classmethod
    def allocate(cls, case: ElasticLLCase) -> "PinnedPhaseInputs":
        layout = case.layout

        def pinned(size: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(size, dtype=dtype, pin_memory=True)

        return cls(
            pinned(layout.region("dispatch_stage").nbytes, torch.uint8),
            pinned(case.max_ranks, torch.int64),
            pinned(case.max_ranks, torch.uint64),
            pinned(case.max_ranks, torch.int64),
            pinned(1, torch.int64),
            pinned(case.max_ranks * case.max_experts, torch.int64),
            pinned(case.max_ranks * case.max_experts, torch.int64),
            pinned(
                case.max_ranks * case.workers_per_peer * WORKER_PLAN_FIELDS,
                torch.int32,
            ),
            pinned(case.route_capacity, torch.float32),
        )

    def prepare(
        self,
        case: ElasticLLCase,
        phase: PhaseSpec,
        rank: int,
        advertised_bases: Sequence[int],
    ) -> None:
        """Fill the reusable buffers without registering pageable H2D sources."""

        route_state = build_phase_route_state(case, phase, rank)
        make_phase_dispatch_stage(
            case,
            phase,
            rank,
            route_state=route_state,
            out=self.dispatch_stage,
        )
        self.rank_mask.copy_(torch.as_tensor(phase.rank_mask, dtype=torch.int64))
        self.advertised_bases.copy_(
            torch.as_tensor(tuple(advertised_bases), dtype=torch.uint64)
        )
        self.incarnations.copy_(
            torch.as_tensor(phase.rank_incarnations, dtype=torch.int64)
        )
        self.generation_state[0] = phase.generation
        self.bucket_counts.copy_(
            torch.as_tensor(route_state.bucket_counts, dtype=torch.int64)
        )
        self.bucket_offsets.copy_(
            torch.as_tensor(route_state.bucket_offsets, dtype=torch.int64)
        )
        self.worker_plan.copy_(
            torch.as_tensor(route_state.worker_plan, dtype=torch.int32)
        )
        self.route_gates.copy_(
            torch.as_tensor(route_state.route_gates, dtype=torch.float32)
        )


@dataclass(slots=True)
class PinnedPhaseResults:
    """Reusable page-locked destinations for one-drain GPU result capture."""

    statuses: torch.Tensor
    outputs: torch.Tensor
    timestamps: torch.Tensor

    @classmethod
    def allocate(cls, case: ElasticLLCase) -> "PinnedPhaseResults":
        return cls(
            torch.empty(case.max_ranks, dtype=torch.int32, pin_memory=True),
            torch.empty(case.output_elements, dtype=torch.bfloat16, pin_memory=True),
            # CuTe still needs a non-null kernel argument in the uninstrumented
            # specialization, but no timing storage may scale with iteration
            # count when every timestamp instruction is compiled out.
            torch.zeros(
                _timestamp_buffer_elements(case),
                dtype=torch.uint64,
                pin_memory=True,
            ),
        )

    def enqueue_preflight(
        self, statuses: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        with torch.cuda.stream(stream):
            self.statuses.copy_(statuses, non_blocking=True)

    def enqueue_main(
        self,
        statuses: torch.Tensor,
        outputs: torch.Tensor,
        timestamps: torch.Tensor,
        stream: torch.cuda.Stream,
        *,
        copy_timestamps: bool,
    ) -> None:
        with torch.cuda.stream(stream):
            self.statuses.copy_(statuses, non_blocking=True)
            self.outputs.copy_(outputs, non_blocking=True)
            if copy_timestamps:
                self.timestamps.copy_(timestamps, non_blocking=True)


def _timestamp_buffer_elements(case: ElasticLLCase) -> int:
    """Return the physical timing-buffer extent for one specialization."""

    if case.timing_mode == "none":
        return 1
    return case.iterations * (case.max_ranks + 1) * 2


def build_phase_specs(case: ElasticLLCase) -> tuple[PhaseSpec, ...]:
    """Build checked generation/incarnation state for each view swap."""

    tables = phase_incarnation_tables(case.membership_plan, case.max_ranks)
    specs: list[PhaseSpec] = []
    for generation, (active, incarnations) in enumerate(
        zip(case.membership_plan, tables)
    ):
        topology = StableSparseTopology(
            max_ranks=case.max_ranks,
            experts_per_rank=case.experts_per_rank,
            active_ranks=active,
            membership_generation=generation,
            rank_incarnations=incarnations,
        )
        specs.append(
            PhaseSpec(
                topology.membership_generation,
                topology.active_ranks,
                topology.rank_incarnations,
            )
        )
    return tuple(specs)


def _eligible_experts(case: ElasticLLCase, phase: PhaseSpec) -> tuple[int, ...]:
    experts = tuple(
        rank * case.experts_per_rank + local_expert
        for rank in phase.active_ranks
        for local_expert in range(case.experts_per_rank)
    )
    # Reserve one active expert as a genuine zero-record bucket when capacity
    # allows it. Every active bucket is still published by the GPU.
    if case.empty_last_expert and len(experts) > case.top_k:
        experts = experts[:-1]
    if len(experts) < case.top_k:
        raise ValueError("not enough eligible experts for top_k")
    return experts


def _gate(route_slot: int, top_k: int) -> float:
    """Return positive, exactly binary, normalized gate weights."""

    if route_slot == top_k - 1:
        return 0.5 + 2.0 ** (-top_k)
    return 2.0 ** (-(route_slot + 2))


def build_routes(
    case: ElasticLLCase, phase: PhaseSpec, rank: int
) -> tuple[RouteAssignment, ...]:
    """Build deterministic token-major, duplicate-free top-k routing."""

    if not 0 <= rank < case.max_ranks:
        raise IndexError("rank is outside fixed capacity")
    if rank not in phase.active_ranks:
        return ()
    eligible = _eligible_experts(case, phase)
    routes: list[RouteAssignment] = []
    for token in range(case.num_tokens):
        first = (rank * case.num_tokens + token) % len(eligible)
        for route_slot in range(case.top_k):
            routes.append(
                RouteAssignment(
                    origin_token=token,
                    route_slot=route_slot,
                    global_expert=eligible[(first + route_slot) % len(eligible)],
                    gate=_gate(route_slot, case.top_k),
                )
            )
    if len(routes) != case.route_capacity:
        raise AssertionError("router produced the wrong number of route records")
    for token in range(case.num_tokens):
        token_experts = {
            route.global_expert for route in routes if route.origin_token == token
        }
        if len(token_experts) != case.top_k:
            raise AssertionError("one token contains duplicate expert routes")
    return tuple(routes)


def build_phase_route_state(
    case: ElasticLLCase, phase: PhaseSpec, rank: int
) -> PhaseRouteState:
    """Create all-source bucket metadata and this rank's route tables."""

    # The device reconstructs the active-rank ordinal by scanning ``rank_mask``
    # in ascending slot order. Runtime phases come from ``build_phase_specs``,
    # but this helper is public and also accepts a directly constructed
    # ``PhaseSpec``. Fail closed here so a non-canonical host ordinal can never
    # select a different peer than the device worker plan.
    if any(
        isinstance(rank_id, bool) or not isinstance(rank_id, int)
        for rank_id in phase.active_ranks
    ):
        raise TypeError("phase active ranks must be integers")
    active_rank_set = set(phase.active_ranks)
    if not phase.active_ranks or tuple(sorted(active_rank_set)) != phase.active_ranks:
        raise ValueError("phase active ranks must be sorted and duplicate-free")
    if any(rank_id < 0 or rank_id >= case.max_ranks for rank_id in phase.active_ranks):
        raise ValueError("phase active ranks exceed fixed rank capacity")
    if isinstance(phase.generation, bool) or not isinstance(phase.generation, int):
        raise TypeError("phase generation must be an integer")
    if not 0 <= phase.generation <= _UINT32_MAX:
        raise ValueError("phase generation must fit the uint32 wire epoch")
    if len(phase.rank_incarnations) != case.max_ranks:
        raise ValueError("phase incarnation table does not match fixed rank capacity")
    for rank_id, incarnation in enumerate(phase.rank_incarnations):
        if isinstance(incarnation, bool) or not isinstance(incarnation, int):
            raise TypeError("phase incarnations must be integers")
        if not 0 <= incarnation <= _INT64_MAX:
            raise ValueError("phase incarnations must fit signed Int64 staging")
        if rank_id in active_rank_set and incarnation == 0:
            raise ValueError("every active rank must have a positive incarnation")

    all_counts: list[int] = []
    all_offsets: list[int] = []
    local_packed: tuple[RouteAssignment, ...] = ()
    local_token_routes = build_routes(case, phase, rank)
    for source_rank in range(case.max_ranks):
        source_routes = build_routes(case, phase, source_rank)
        counts = [0] * case.max_experts
        for route in source_routes:
            counts[route.global_expert] += 1
        if any(count > case.num_tokens for count in counts):
            raise AssertionError(
                "one expert bucket exceeds compact dispatch receive capacity"
            )
        offsets: list[int] = []
        cursor = 0
        for count in counts:
            offsets.append(cursor)
            cursor += count
        if cursor not in (0, case.route_capacity):
            raise AssertionError("bucket prefix sum does not span route capacity")
        all_counts.extend(counts)
        all_offsets.extend(offsets)
        if source_rank == rank:
            local_packed = tuple(
                sorted(
                    source_routes,
                    key=lambda route: (
                        route.global_expert,
                        route.origin_token,
                        route.route_slot,
                    ),
                )
            )

    route_experts = [0] * case.route_capacity
    route_gates = [0.0] * case.route_capacity
    for route in local_token_routes:
        item = route.origin_token * case.top_k + route.route_slot
        route_experts[item] = route.global_expert
        route_gates[item] = route.gate

    # Membership changes only at a drained phase boundary, where route metadata
    # is already rebuilt and copied asynchronously.  Compute each logical
    # worker's contiguous record ranges there as well.  Every field is signed
    # Int32: task is below the fixed-worker bound and each begin/count is at
    # most num_tokens, both enforced by ElasticLLCase validation.  Storing
    # (begin, count), rather than (begin, end), also avoids device subtraction.
    # The plan removes four dynamic Int64 divisions from every device launch;
    # CUDA otherwise lowers them to out-of-line helpers that can contribute to
    # stack and spill pressure in register-heavy shapes even though they sit
    # before the persistent operation loop.
    fixed_worker_count = case.max_ranks * case.workers_per_peer
    active_task_count = len(phase.active_ranks) * case.experts_per_rank
    worker_plan: list[int] = []
    for worker in range(fixed_worker_count):
        task = worker % active_task_count
        task_shard = worker // active_task_count
        task_shards = (
            fixed_worker_count - task + active_task_count - 1
        ) // active_task_count
        peer = phase.active_ranks[task // case.experts_per_rank]
        expert = task % case.experts_per_rank
        outgoing_bucket = (
            rank * case.max_experts + peer * case.experts_per_rank + expert
        )
        incoming_bucket = (
            peer * case.max_experts + rank * case.experts_per_rank + expert
        )
        outgoing_count = all_counts[outgoing_bucket]
        incoming_count = all_counts[incoming_bucket]
        outgoing_begin = outgoing_count * task_shard // task_shards
        outgoing_end = outgoing_count * (task_shard + 1) // task_shards
        incoming_begin = incoming_count * task_shard // task_shards
        incoming_end = incoming_count * (task_shard + 1) // task_shards
        worker_plan.extend(
            (
                task,
                outgoing_begin,
                outgoing_end - outgoing_begin,
                incoming_begin,
                incoming_end - incoming_begin,
            )
        )
    return PhaseRouteState(
        tuple(all_counts),
        tuple(all_offsets),
        tuple(route_experts),
        tuple(route_gates),
        local_packed,
        tuple(worker_plan),
    )


def bank_cycle_schedule(
    iterations: int, num_banks: int = 1
) -> tuple[tuple[int, int, int], ...]:
    """Return the monotonically versioned ``(step, bank, cycle)`` sequence."""

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 0
    ):
        raise ValueError("iterations must be a non-negative integer")
    if (
        isinstance(num_banks, bool)
        or not isinstance(num_banks, int)
        or num_banks not in (1, 2)
    ):
        raise ValueError("num_banks must be one or two")
    return tuple(
        (step, step % num_banks, step // num_banks + 1) for step in range(iterations)
    )


def cumulative_publication_value(record_count: int, bank_cycle: int) -> int:
    """Pack a monotonic bank cycle and ``record_count + 1`` publication."""

    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
    ):
        raise ValueError("record_count must be a non-negative integer")
    if (
        isinstance(bank_cycle, bool)
        or not isinstance(bank_cycle, int)
        or bank_cycle <= 0
    ):
        raise ValueError("bank_cycle must be a positive integer")
    if record_count > _UINT32_MAX - 1:
        raise ValueError("record_count + 1 must fit in uint32")
    return (bank_cycle << 32) | (record_count + 1)


def _input_values(
    rank: int, token: int, hidden_size: int, operation: int = 0
) -> torch.Tensor:
    """Return deterministic BF16 token data exactly reproducible on every rank."""

    values = torch.arange(hidden_size, dtype=torch.float32)
    values = values * 0.00390625 + rank * 8.0 + token * 0.125 + operation * 32.0
    return values.to(torch.bfloat16)


def make_phase_dispatch_stage(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    *,
    route_state: PhaseRouteState | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create only the immutable dispatch template copied at a phase swap."""

    if rank not in range(case.max_ranks):
        raise IndexError("rank is outside fixed capacity")
    layout = case.layout
    stage_region = layout.region("dispatch_stage")
    if out is None:
        stage = torch.zeros(stage_region.nbytes, dtype=torch.uint8)
    else:
        if out.device.type != "cpu" or out.dtype != torch.uint8:
            raise ValueError("dispatch stage output must be a CPU uint8 tensor")
        if out.numel() != stage_region.nbytes:
            raise ValueError("dispatch stage output has the wrong size")
        stage = out
        stage.zero_()
    if route_state is None:
        route_state = build_phase_route_state(case, phase, rank)
    # The direct-copy body is immutable and shared by both receive banks.
    # Operation/incarnation identity is stamped once per released destination
    # bucket, so it is neither copied nor overwritten in every route record.
    for stage_copy in range(layout.dispatch_stage_copies):
        for route_index, route in enumerate(route_state.packed_routes):
            span = layout.dispatch_stage_record(route_index, stage_copy)
            relative_offset = span.offset - stage_region.offset
            metadata = struct.pack(
                "<IIfI",
                route.origin_token,
                route.route_slot,
                route.gate,
                0,
            )
            stage[relative_offset : relative_offset + len(metadata)] = torch.tensor(
                list(metadata), dtype=torch.uint8
            )
            payload = _input_values(
                rank, route.origin_token, case.hidden_size, stage_copy
            )
            payload_bytes = payload.view(torch.uint8)
            begin = relative_offset + DISPATCH_METADATA_NBYTES
            stage[begin : begin + payload_bytes.numel()] = payload_bytes
    return stage


def make_phase_arena(case: ElasticLLCase, phase: PhaseSpec, rank: int) -> torch.Tensor:
    """Build a fully zeroed arena for protocol inspection and unit tests.

    Production phase swaps use :func:`make_phase_dispatch_stage` and clear only
    publication/control words. Receive payloads are protected by those words and
    need not be copied or zeroed between drained generations.
    """

    layout = case.layout
    arena = torch.zeros(layout.arena_nbytes, dtype=torch.uint8)
    stage_region = layout.region("dispatch_stage")
    arena[stage_region.offset : stage_region.end] = make_phase_dispatch_stage(
        case, phase, rank
    )
    return arena


def expected_outputs(
    case: ElasticLLCase, phase: PhaseSpec, rank: int, operation: int = 0
) -> dict[int, torch.Tensor]:
    """Return exact FP32 weighted reduction followed by one BF16 cast."""

    if rank not in phase.active_ranks:
        return {}
    expected: dict[int, torch.Tensor] = {}
    routes = build_routes(case, phase, rank)
    for token in range(case.num_tokens):
        source = _input_values(rank, token, case.hidden_size, operation)
        accumulator = torch.zeros(case.hidden_size, dtype=torch.float32)
        token_routes = [route for route in routes if route.origin_token == token]
        if len(token_routes) != case.top_k:
            raise AssertionError("router did not produce top_k records for one token")
        for route in token_routes:
            expert_output = (
                source + torch.tensor(route.global_expert + 1, dtype=torch.bfloat16)
            ).to(torch.bfloat16)
            accumulator += expert_output.float() * route.gate
        expected[token] = accumulator.to(torch.bfloat16)
    return expected


def _summarize_native_peer_atomics(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Validate one directional row per worker and return one canonical matrix.

    Each worker qualifies only the row whose accessor is its local GPU. This
    keeps every exchanged rank payload linear in ``max_ranks`` while preserving
    complete directed coverage across the workers that can execute the phase.
    """

    expected_capability = "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED"
    expected_query_scope = (
        "selected accessing devices to every distinct participating owner device"
    )
    expected_execution_scope = "host preflight only; no steady-state device-path cost"
    active_slots = set(phase.active_ranks)
    canonical_shared: dict[str, object] | None = None
    checked_slots: list[int] = []
    accessing_devices: list[int] = []
    combined_pairs: list[dict[str, object]] = []
    for slot, result in enumerate(rank_results):
        environment = result.get("environment")
        if environment is None:
            if slot in active_slots:
                raise ValueError(
                    f"active rank {slot} did not report native-atomic environment"
                )
            continue
        if not isinstance(environment, dict):
            raise ValueError(f"rank {slot} reported a malformed environment")
        evidence = environment.get("native_peer_atomics")
        if evidence is None:
            if slot in active_slots:
                raise ValueError(
                    f"active rank {slot} did not report native peer-atomic evidence"
                )
            continue
        if not isinstance(evidence, dict):
            raise ValueError(
                f"rank {slot} reported malformed native peer-atomic evidence"
            )
        if (
            type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 1
        ):
            raise ValueError(
                f"rank {slot} reported the wrong native-atomic evidence schema"
            )
        if evidence.get("capability") != expected_capability:
            raise ValueError(f"rank {slot} reported the wrong native-atomic capability")
        if evidence.get("query_scope") != expected_query_scope:
            raise ValueError(
                f"rank {slot} reported the wrong native-atomic query scope"
            )
        if evidence.get("execution_scope") != expected_execution_scope:
            raise ValueError(
                f"rank {slot} reported the wrong native-atomic execution scope"
            )
        visible_devices = evidence.get("cuda_visible_devices")
        if visible_devices is not None and not isinstance(visible_devices, str):
            raise ValueError(
                f"rank {slot} reported malformed CUDA_VISIBLE_DEVICES evidence"
            )
        driver_module = evidence.get("driver_module")
        if not isinstance(driver_module, str) or not driver_module:
            raise ValueError(
                f"rank {slot} reported malformed CUDA driver-module evidence"
            )

        devices = evidence.get("devices")
        if (
            not isinstance(devices, list)
            or len(devices) != case.max_ranks
            or any(
                isinstance(device, bool) or not isinstance(device, int) or device < 0
                for device in devices
            )
            or len(set(devices)) != case.max_ranks
        ):
            raise ValueError(f"rank {slot} native-atomic device coverage is invalid")
        local_device = environment.get("device")
        if (
            isinstance(local_device, bool)
            or not isinstance(local_device, int)
            or local_device != devices[slot]
        ):
            raise ValueError(
                f"rank {slot} native-atomic evidence names the wrong local device"
            )
        evidence_accessors = evidence.get("accessing_devices")
        if evidence_accessors != [local_device]:
            raise ValueError(
                f"rank {slot} native-atomic accessor-row coverage is invalid"
            )

        identities = evidence.get("device_identities")
        if not isinstance(identities, list) or len(identities) != case.max_ranks:
            raise ValueError(
                f"rank {slot} native-atomic physical-device evidence is invalid"
            )
        bus_ids: list[str] = []
        for index, identity in enumerate(identities):
            if (
                not isinstance(identity, dict)
                or isinstance(identity.get("device_ordinal"), bool)
                or not isinstance(identity.get("device_ordinal"), int)
                or identity.get("device_ordinal") != devices[index]
                or not isinstance(identity.get("pci_bus_id"), str)
                or not identity["pci_bus_id"]
            ):
                raise ValueError(
                    f"rank {slot} has malformed physical identity for device "
                    f"index {index}"
                )
            bus_ids.append(identity["pci_bus_id"])
        if len(set(bus_ids)) != case.max_ranks:
            raise ValueError(
                f"rank {slot} native-atomic evidence aliases physical devices"
            )

        expected_pairs = [
            (local_device, owner) for owner in devices if local_device != owner
        ]
        ordered_pairs = evidence.get("ordered_pairs")
        if not isinstance(ordered_pairs, list) or len(ordered_pairs) != len(
            expected_pairs
        ):
            raise ValueError(
                f"rank {slot} native-atomic directed-pair coverage is invalid"
            )
        observed_pairs: list[tuple[int, int]] = []
        for pair in ordered_pairs:
            if not isinstance(pair, dict):
                raise ValueError(f"rank {slot} reported a malformed native-atomic pair")
            accessing = pair.get("accessing_device")
            owner = pair.get("owner_device")
            if (
                isinstance(accessing, bool)
                or not isinstance(accessing, int)
                or isinstance(owner, bool)
                or not isinstance(owner, int)
            ):
                raise ValueError(
                    f"rank {slot} reported malformed directed-pair ordinals"
                )
            observed_pairs.append((accessing, owner))
            if pair.get("native_atomics_supported") is not True:
                raise ValueError(
                    f"rank {slot} did not qualify native atomics for accessing "
                    f"device {accessing} to owner device {owner}"
                )
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"rank {slot} native-atomic directed-pair coverage is invalid"
            )
        if evidence.get("all_supported") is not True:
            raise ValueError(
                f"rank {slot} native-atomic aggregate result is not supported"
            )
        shared_keys = (
            "schema_version",
            "capability",
            "devices",
            "device_identities",
            "cuda_visible_devices",
            "query_scope",
            "execution_scope",
            "driver_module",
        )
        if any(key not in evidence for key in shared_keys):
            raise ValueError(f"rank {slot} native-atomic shared evidence is incomplete")
        shared = {key: evidence[key] for key in shared_keys}
        if canonical_shared is None:
            canonical_shared = shared
        elif shared != canonical_shared:
            raise ValueError(
                f"rank {slot} native-atomic evidence disagrees across workers"
            )
        checked_slots.append(slot)
        accessing_devices.append(local_device)
        combined_pairs.extend(ordered_pairs)

    missing_active = active_slots - set(checked_slots)
    if missing_active:
        raise ValueError(
            "native peer-atomic evidence is missing for active ranks "
            f"{sorted(missing_active)}"
        )
    if canonical_shared is None:
        raise ValueError("no native peer-atomic evidence was reported")
    return {
        **canonical_shared,
        "accessing_devices": accessing_devices,
        "ordered_pairs": combined_pairs,
        "all_supported": True,
        "checked_by_worker_slots": checked_slots,
        "policy": "required for every directed accessor-to-owner pair",
        "worker_evidence_shape": (
            "one local-accessor row per worker; canonical rows combined once"
        ),
    }


def summarize_phase_results(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build one exact, machine-readable cross-rank performance record."""

    if len(rank_results) != case.max_ranks:
        raise ValueError("one result is required for every stable rank slot")
    active_rank_starts: list[tuple[int, ...]] = []
    active_rank_ends: list[tuple[int, ...]] = []
    active_result_count = 0
    active_slots = set(phase.active_ranks)
    for expected_rank, result in enumerate(rank_results):
        if result.get("correctness") != "PASS":
            raise RuntimeError("a rank failed correctness validation")
        if result.get("rank") != expected_rank:
            raise ValueError("rank results are not in stable-slot order")
        if result.get("active") is not (expected_rank in active_slots):
            raise ValueError("a rank reported the wrong phase membership")
        if result.get("timing_mode") != case.timing_mode:
            raise ValueError("a rank reported the wrong timing specialization")
        if result.get("active"):
            active_result_count += 1
            raw_starts = result.get("rank_round_starts_ns")
            raw_ends = result.get("rank_round_ends_ns")
            if not isinstance(raw_starts, list) or not isinstance(raw_ends, list):
                raise ValueError("an active rank has malformed round samples")
            starts = tuple(int(value) for value in raw_starts)
            ends = tuple(int(value) for value in raw_ends)
            if case.timing_mode == "none":
                if starts or ends:
                    raise ValueError("a timing-disabled rank emitted timing samples")
                continue
            if len(starts) != case.iterations or len(ends) != case.iterations:
                raise ValueError("an active rank has the wrong round sample count")
            for iteration, (start, end) in enumerate(zip(starts, ends)):
                start_is_required = case.timing_mode in {"cadence", "peer"} or (
                    case.timing_mode == "envelope" and iteration == 0
                )
                end_is_required = case.timing_mode == "peer" or (
                    case.timing_mode in {"envelope", "cadence"}
                    and iteration == case.iterations - 1
                )
                if start_is_required and start == 0:
                    raise ValueError("an active rank has an invalid start marker")
                if not start_is_required and start != 0:
                    raise ValueError(
                        "an active rank emitted an unnecessary start marker"
                    )
                if end_is_required:
                    comparison_start = start if start_is_required else starts[0]
                    if end == 0 or end < comparison_start:
                        raise ValueError("an active rank has an invalid end marker")
                elif end != 0:
                    raise ValueError("an active rank emitted an unnecessary end marker")
            active_rank_starts.append(starts)
            active_rank_ends.append(ends)
    if not active_result_count:
        raise ValueError("phase has no active ranks")
    native_peer_atomics = _summarize_native_peer_atomics(case, phase, rank_results)
    expected_mapping_policy = "require_non_null_process_local_pointer"
    mapped_by_rank: dict[str, dict[str, object]] = {}
    for expected_rank, result in enumerate(rank_results):
        if not result.get("active"):
            continue
        rank = expected_rank
        environment = result.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"active rank {rank} did not report its environment")
        mapped = environment.get("mapped_preflight")
        if not isinstance(mapped, dict):
            raise ValueError(
                f"active rank {rank} did not report exact-allocation mapped "
                "preflight evidence"
            )
        if mapped.get("rank_active") is not True:
            raise ValueError(f"active rank {rank} reported an inactive preflight")
        if mapped.get("policy") != expected_mapping_policy:
            raise ValueError(f"active rank {rank} reported the wrong mapping policy")
        expected_peers = {str(peer) for peer in phase.active_ranks if peer != rank}
        remote_peers = mapped.get("remote_peers")
        if not isinstance(remote_peers, dict) or set(remote_peers) != expected_peers:
            raise ValueError(
                f"active rank {rank} mapped-preflight peer coverage is invalid"
            )
        classifications: set[str] = set()
        for peer, evidence in remote_peers.items():
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"rank {rank}, peer {peer} has malformed mapping evidence"
                )
            classification = evidence.get("classification")
            status = evidence.get("status")
            if (classification, status) not in {
                ("owner_va_coincident", 0),
                ("translated_process_local_va", _STATUS_MISMATCH),
            }:
                raise ValueError(
                    f"rank {rank}, peer {peer} did not report a usable mapping"
                )
            if not isinstance(classification, str):
                raise ValueError(
                    f"rank {rank}, peer {peer} has a non-string classification"
                )
            classifications.add(classification)
        rank_all_mapped = classifications <= {
            "owner_va_coincident",
            "translated_process_local_va",
        }
        rank_has_translation = "translated_process_local_va" in classifications
        if mapped.get("all_active_remote_peers_mapped") is not rank_all_mapped:
            raise ValueError(f"active rank {rank} mapping summary is inconsistent")
        if mapped.get("translated_va_observed") is not rank_has_translation:
            raise ValueError(
                f"active rank {rank} mapping-translation summary is inconsistent"
            )
        if mapped.get("legacy_allow_unverified_mapped_ignored") is not (
            case.allow_unverified_mapped
        ):
            raise ValueError(
                f"active rank {rank} legacy-option summary is inconsistent"
            )
        mapped_by_rank[str(rank)] = mapped
    translated_link_observed = any(
        bool(mapped["translated_va_observed"]) for mapped in mapped_by_rank.values()
    )
    all_active_remote_links_mapped = all(
        bool(mapped["all_active_remote_peers_mapped"])
        for mapped in mapped_by_rank.values()
    )
    observer_span_samples = (
        [
            max(
                rank_ends[iteration] - rank_starts[iteration]
                for rank_starts, rank_ends in zip(active_rank_starts, active_rank_ends)
            )
            for iteration in range(case.iterations)
        ]
        if case.timing_mode == "peer"
        else []
    )
    if case.timing_mode in {"cadence", "peer"}:
        all_cadence_samples = [
            max(
                rank_starts[iteration] - rank_starts[iteration - 1]
                for rank_starts in active_rank_starts
            )
            for iteration in range(1, case.iterations)
        ]
    else:
        all_cadence_samples = []
    if case.timing_mode == "none":
        output_ready_intervals: list[int] = []
    else:
        output_ready_intervals = [
            rank_ends[-1] - rank_starts[0]
            for rank_starts, rank_ends in zip(active_rank_starts, active_rank_ends)
        ]
    if case.timing_mode == "none":
        steady_intervals: list[int] = []
        steady_rounds = 0
        steady_sample_kind = "disabled in deployment specialization"
        cadence_samples: list[int] = []
    elif case.timing_mode == "envelope":
        steady_intervals = output_ready_intervals
        steady_rounds = case.iterations
        steady_sample_kind = "whole-phase first-start to final-output-ready envelope"
        cadence_samples = []
    elif case.iterations >= 3:
        steady_intervals = [
            rank_starts[-1] - rank_starts[1] for rank_starts in active_rank_starts
        ]
        steady_rounds = case.iterations - 2
        steady_sample_kind = "post-fence start-to-start persistent-loop cadence"
        cadence_samples = all_cadence_samples[1:]
    elif case.iterations == 2:
        steady_intervals = [
            rank_starts[1] - rank_starts[0] for rank_starts in active_rank_starts
        ]
        steady_rounds = 1
        steady_sample_kind = "two-round cadence fallback including timing fence"
        cadence_samples = all_cadence_samples
    else:
        steady_intervals = output_ready_intervals
        steady_rounds = 1
        steady_sample_kind = "single-round fenced output-ready fallback"
        cadence_samples = []

    def percentile(values: Sequence[int], q: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(int(len(ordered) * q), len(ordered) - 1)]

    remote_routes = sum(
        1
        for source_rank in phase.active_ranks
        for route in build_routes(case, phase, source_rank)
        if route.global_expert // case.experts_per_rank != source_rank
    )
    remote_payload = remote_routes * case.hidden_size * ELEMENT_NBYTES * 2
    remote_record_stores = remote_routes * (
        case.layout.dispatch_record_stride + case.layout.combine_record_nbytes
    )
    ordered_remote_pairs = len(phase.active_ranks) * (len(phase.active_ranks) - 1)
    # Per ordered rank pair and local expert: one dispatch stamp+ready.  Combine
    # joins every expert locally and therefore needs only one aggregate
    # stamp+ready per ordered rank pair. The closed dispatch/combine handshake
    # proves both receive buffers have been consumed before their next write, so
    # no separate remote credit store is necessary.
    remote_control_stores = ordered_remote_pairs * (
        case.experts_per_rank * (MESSAGE_STAMP_NBYTES + 8) + MESSAGE_STAMP_NBYTES + 8
    )
    steady_interval = max(steady_intervals) if steady_intervals else None
    steady_logical_bytes = remote_payload * steady_rounds
    output_ready_interval = (
        max(output_ready_intervals) if output_ready_intervals else None
    )
    output_ready_logical_bytes = (
        remote_payload * case.iterations if output_ready_interval is not None else 0
    )

    def throughput(logical_bytes: int, interval_ns: int | None) -> float | None:
        if interval_ns is None:
            return None
        return logical_bytes / interval_ns if interval_ns else math.inf

    marker_counts = {
        "none": (0, 0, 0),
        "envelope": (1, 1, 1),
        "cadence": (case.iterations, 1, 1),
        "peer": (case.iterations, case.iterations, 1),
    }
    start_reads, end_reads, timing_grid_fences = marker_counts[case.timing_mode]
    if case.num_banks == 1:
        end_of_round_grid_barriers = case.total_iterations
    elif case.timing_mode == "none":
        end_of_round_grid_barriers = 0
    elif case.timing_mode == "envelope":
        end_of_round_grid_barriers = 1
    else:
        end_of_round_grid_barriers = case.iterations
    total_protocol_grid_barriers = case.total_iterations + end_of_round_grid_barriers
    return {
        "schema_version": 7,
        "case": "nixl_cute_elastic_moe_ll",
        "generation": phase.generation,
        "active_ranks": list(phase.active_ranks),
        "rank_incarnations": list(phase.rank_incarnations),
        "warmup": case.warmup,
        "iterations": case.iterations,
        "experts_per_rank": case.experts_per_rank,
        "num_tokens_per_rank": case.num_tokens,
        "top_k": case.top_k,
        "workers_per_peer": case.workers_per_peer,
        "fixed_worker_warps": case.max_ranks * case.workers_per_peer,
        "warps_per_cta": case.warps_per_cta,
        "cooperative_ctas": (
            case.max_ranks * case.workers_per_peer // case.warps_per_cta
        ),
        "validate_every_iteration": case.validate_every_iteration,
        "allow_unverified_mapped": case.allow_unverified_mapped,
        "native_peer_atomics": native_peer_atomics,
        "mapped_preflight": {
            "policy": (expected_mapping_policy),
            "evidence_scope": "exact timed-phase device view and arena allocation",
            "all_active_remote_links_mapped": all_active_remote_links_mapped,
            "translated_va_observed": translated_link_observed,
            "per_rank": mapped_by_rank,
        },
        "instrument_per_peer": case.instrument_per_peer,
        "timing_mode": case.timing_mode,
        "empty_bucket_publication": case.empty_last_expert,
        "hidden_size": case.hidden_size,
        "logical_remote_payload_bytes_per_phase_round": remote_payload,
        "remote_record_store_bytes_per_phase_round": remote_record_stores,
        "remote_control_store_bytes_per_phase_round": remote_control_stores,
        "remote_mapped_store_bytes_per_phase_round": (
            remote_record_stores + remote_control_stores
        ),
        "steady_state_run": {
            "rank_intervals_ns": steady_intervals,
            "measured_interval_ns": steady_interval,
            "measured_rounds": steady_rounds,
            "sample_kind": steady_sample_kind,
            "logical_remote_payload_bytes": steady_logical_bytes,
            "aggregate_logical_remote_GBps_cadence_estimate": throughput(
                steady_logical_bytes, steady_interval
            ),
            "definition": (
                "device timing is compiled out; use same-stream CUDA events around "
                "the entire launch for non-perturbing deployment throughput"
                if case.timing_mode == "none"
                else (
                    "one first-start/final-output-ready device envelope divided by "
                    "all phase rounds; this is an amortized phase rate, not a cadence "
                    "or latency percentile"
                    if case.timing_mode == "envelope"
                    else (
                        (
                            "post-fence sustainable start-to-start cadence"
                            if case.iterations >= 3
                            else "bounded fallback; too few rounds to remove the "
                            "one-time timing-fence interval"
                        )
                        + "; includes the complete prior-round output-read barrier "
                        "before the next round; logical payload counts dispatch and "
                        "combine once and is not physical wire traffic; aggregate "
                        "bytes divided by the slowest independently measured per-rank "
                        "interval is a cadence estimate, not synchronized global "
                        "wall-clock throughput"
                    )
                )
            ),
        },
        "output_ready_run": {
            "rank_intervals_ns": output_ready_intervals,
            "measured_interval_ns": output_ready_interval,
            "logical_remote_payload_bytes": output_ready_logical_bytes,
            "aggregate_logical_remote_GBps_interval_estimate": throughput(
                output_ready_logical_bytes, output_ready_interval
            ),
            "definition": (
                "disabled because the deployment specialization emits no device "
                "timestamps or timing-only grid fence"
                if case.timing_mode == "none"
                else "fenced first measured start through final measured local output "
                "readiness; includes one timing-only grid fence and ends after the "
                "final all-output-read barrier; aggregate bytes use the slowest "
                "independent rank interval and are not synchronized global wall-clock "
                "or physical-wire throughput"
            ),
        },
        "rank_observer_span": {
            "enabled": case.timing_mode == "peer",
            "samples_ns": observer_span_samples,
            "p50_ns": percentile(observer_span_samples, 0.5),
            "p90_ns": percentile(observer_span_samples, 0.9),
            "p99_ns": percentile(observer_span_samples, 0.99),
            "max_ns": max(observer_span_samples) if observer_span_samples else None,
            "definition": (
                "opt-in diagnostic maximum across ranks from logical worker zero's "
                "per-round start marker through all local output slices; disabled "
                "in production because intermediate end markers perturb every round; "
                "only the first measured marker is fenced before every worker's "
                "timed work, so samples are not claimed as per-round critical latency"
            ),
        },
        "round_cadence": {
            "enabled": case.timing_mode in {"cadence", "peer"},
            "samples_ns": cadence_samples,
            "all_samples_including_timing_fence_ns": all_cadence_samples,
            "p50_ns": percentile(cadence_samples, 0.5),
            "p90_ns": percentile(cadence_samples, 0.9),
            "definition": (
                "logical-worker-zero start-to-start cadence per rank"
                if case.timing_mode in {"cadence", "peer"}
                else "disabled by the selected timing specialization"
            ),
        },
        "timing": {
            "enabled": case.timing_mode != "none",
            "mode": case.timing_mode,
            "clock": "CUDA %globaltimer" if case.timing_mode != "none" else None,
            "scope": (
                "compiled out, including the timing-only cooperative grid fence"
                if case.timing_mode == "none"
                else (
                    "one worker-zero first-start and one final all-output-ready end "
                    "marker per rank"
                    if case.timing_mode == "envelope"
                    else (
                        "worker-zero periodic start markers and one final "
                        "all-output-ready end marker per rank"
                        if case.timing_mode == "cadence"
                        else "worker-zero periodic starts and ends plus per-peer "
                        "diagnostic spans"
                    )
                    + "; the first measured start is ordered before all measured "
                    "work by one timing-only cooperative grid fence"
                )
            ),
            "cross_device_clock_rule": (
                "raw clock epochs are never subtracted across GPUs; only per-rank "
                "durations are compared"
            ),
            "jit_compilation_excluded": True,
            "kernel_launch_excluded": True,
            "cpu_synchronization_inside_measured_loop": False,
            "warmup_inside_same_persistent_kernel": True,
            "remote_wait_mode": (
                "abort_only_timer_free"
                if case.device_timeout_ns == 0
                else "timeout_or_abort_diagnostic"
            ),
            "device_operation_timeout_ns": case.device_timeout_ns or None,
            "device_timeout_role": (
                "disabled in the production timer-free path; graceful membership "
                "transitions drain every mapped owner before view release"
                if case.device_timeout_ns == 0
                else "opt-in remote-wait diagnostic and best-effort drain aid; "
                "abrupt mapped-owner loss is not recovered"
            ),
            "explicit_start_marker_reads_per_rank_phase": start_reads,
            "explicit_end_marker_reads_per_rank_phase": end_reads,
            "timing_grid_fences_per_phase": timing_grid_fences,
            "pre_reduction_grid_barriers_per_round": 1,
            "end_of_round_grid_barriers_per_rank_phase": (end_of_round_grid_barriers),
            "total_protocol_grid_barriers_per_rank_phase": (
                total_protocol_grid_barriers
            ),
            "per_peer_timing_enabled": case.instrument_per_peer,
            "local_wait_timer_reads": 0,
            "remote_wait_ready_path_timer_reads": 0,
            "remote_wait_first_miss_timer_reads": (
                0 if case.device_timeout_ns == 0 else 1
            ),
            "timeout_clock_check_period_failed_loads": (
                None if case.device_timeout_ns == 0 else 256
            ),
            "abort_check_period_failed_loads": 4096,
        },
        "protocol": {
            "transport": "NIXL locally mapped peer pointer",
            "mapped_address_validation": (
                "host preflight requires native peer atomics for every directed "
                "GPU pair; exact-allocation device preflight requires a non-null "
                "process-local nixlGetPtr result, while owner and importer numeric "
                "VAs are recorded only as diagnostics"
            ),
            "dispatch": (
                "mapped_copy_warp_ptr_readonly payload, one separate bucket stamp, "
                "then release-ready publication"
            ),
            "expert": "BF16 x + (stable_global_expert + 1)",
            "combine": (
                "unique route-slot result, GPU-scope expert joins, one aggregate "
                "system release/acquire per peer, token-sharded logical worker "
                "warps, 128-bit loads/stores, FP32 gated reduction, single BF16 cast"
            ),
            "combine_barrier": (
                "one pre-reduction cooperative grid barrier; two-bank reuse is "
                "joined by the following round's retained barrier before N+2 can "
                "reuse bank N"
                if case.num_banks == 2 and case.timing_mode == "none"
                else (
                    "one pre-reduction barrier per round plus one terminal "
                    "all-output-ready barrier for the phase envelope"
                    if case.num_banks == 2 and case.timing_mode == "envelope"
                    else (
                        "one pre-reduction barrier per round plus an end barrier "
                        "on every measured round; warmup end barriers are elided"
                        if case.num_banks == 2
                        else "two cooperative grid barriers per round: all remote "
                        "peer publications before reduction, then all output slices "
                        "before the next round"
                    )
                )
            ),
            "routing": "deterministic host-staged routing and packing outside timing",
            "banks": case.num_banks,
            "bank_reuse_proof": (
                "next combine-ready proves prior dispatch consumption; the N+1 "
                "pre-reduction grid barrier joins every N reader before N+2 can "
                "reuse the same bank"
                if case.num_banks == 2
                else "next combine-ready proves prior dispatch consumption; next "
                "dispatch-ready follows the prior all-combine-read grid barrier"
            ),
            "immutable_dispatch_template_banks": 1,
            "dispatch_stage_copies": case.layout.dispatch_stage_copies,
            "source_working_set_mode": (
                "operation-varying immutable stage copies"
                if case.validate_every_iteration
                else "single immutable hot stage copy"
            ),
            "source_reuse": (
                "combine-ready proves dispatch consumption; the next dispatch-"
                "ready proves prior combine consumption; no separate credit"
            ),
            "dispatch_ready_encoding": (
                "(bank cycle << 32) | (expert-bucket record_count + 1)"
            ),
            "combine_ready_encoding": (
                "(bank cycle << 32) | (peer-aggregate record_count + 1)"
            ),
            "network_fallback": (
                "unsupported in this example; the fence ABI exists, but the "
                "ordered transport publish/credit path is not implemented"
            ),
        },
        "elasticity": {
            "stable_sparse_descriptor_slots": True,
            "view_swap_between_drained_phases": True,
            "graceful_standby_rejoin": True,
            "abrupt_failure_recovery": False,
            "abrupt_mapped_owner_loss": (
                "unsupported; direct CUDA-IPC loads or stores may fault before a "
                "diagnostic wait can time out"
            ),
        },
        "rank_results": list(rank_results),
        "validation": (
            "every measured iteration with operation-varying dispatch payloads"
            if case.validate_every_iteration
            else "final output state with operation-invariant dispatch payloads"
        ),
        "correctness": "PASS",
    }


def _decode_rank_results(
    gathered: Mapping[int, bytes], max_ranks: int
) -> tuple[dict[str, object], ...]:
    """Decode one result per stable slot without depending on dict iteration."""

    expected = set(range(max_ranks))
    actual = set(gathered)
    if actual != expected:
        raise RuntimeError(
            "rank-result exchange did not cover every stable slot: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return tuple(
        json.loads(gathered[rank].decode("utf-8")) for rank in range(max_ranks)
    )


def _phase_status_payload(rank: int, error: Exception | None) -> bytes:
    """Encode a bounded all-rank lifecycle outcome before any rank raises."""

    payload: dict[str, object] = {"rank": rank, "ok": error is None}
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["message"] = str(error)[:4096]
    return json.dumps(payload, sort_keys=True).encode("utf-8")


class _UnreleasedDeviceViewError(RuntimeError):
    """Native release failed and the mapped handle is still live."""


class _RecoveredDeviceViewReleaseError(RuntimeError):
    """Native release failed once but its explicit idempotent retry succeeded."""


class _DeviceViewExitStack:
    """ExitStack that records and retries a drained NIXL view release.

    ``nixl_device_view_handle.release`` guarantees that a failed release leaves
    the handle valid. ``ExitStack`` has already popped that callback when it
    raises, so retry the handle directly and expose whether owner teardown is
    proven safe. This wrapper never hides an exception from the managed body.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._remote_view: object | None = None
        self._closed = False
        self.release_error: Exception | None = None

    def enter_context(self, context_manager: Any) -> Any:
        view = self._stack.enter_context(context_manager)
        self._remote_view = view
        return view

    @staticmethod
    def _is_valid(view: object | None) -> bool:
        return view is not None and bool(getattr(view, "valid", False))

    def close(self) -> Exception | None:
        if self._closed:
            return self.release_error
        self._closed = True
        try:
            self._stack.close()
        except Exception as first_error:
            view = self._remote_view
            if self._is_valid(view):
                release = getattr(view, "release", None)
                try:
                    if not callable(release):
                        raise TypeError("live NIXL device view has no release method")
                    release()
                except Exception as retry_error:
                    if self._is_valid(view):
                        self.release_error = _UnreleasedDeviceViewError(
                            f"native device-view release failed: {first_error}; "
                            f"explicit retry also failed: {retry_error}"
                        )
                        return self.release_error
            if self._is_valid(view):
                self.release_error = _UnreleasedDeviceViewError(
                    f"native device-view release failed and the handle remains "
                    f"valid: {first_error}"
                )
            else:
                self.release_error = _RecoveredDeviceViewReleaseError(
                    f"native device-view release required an explicit retry: "
                    f"{first_error}"
                )
        return self.release_error

    def __enter__(self) -> "_DeviceViewExitStack":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _contains_unreleased_view(
    failures: Sequence[dict[str, object]],
) -> bool:
    return any(
        failure.get("error_type") == _UnreleasedDeviceViewError.__name__
        for failure in failures
    )


def _fail_stop_registered_owner(reason: str) -> None:
    """Retain direct-mapping owners until the external allocation is stopped."""

    print(f"NIXL CuTe fail-stop while retaining registered owner: {reason}", flush=True)
    while True:
        time.sleep(60.0)


def _phase_result_payload(
    rank: int, error: Exception | None, result: dict[str, object] | None
) -> bytes:
    """Encode validation and an optional rank result in one exchange payload."""

    if error is None and (not isinstance(result, dict) or result.get("rank") != rank):
        error = AssertionError("successful phase has no matching rank result")
    if error is not None:
        return _phase_status_payload(rank, error)
    assert isinstance(result, dict)
    payload = dict(result)
    payload["ok"] = True
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _decode_phase_failures(
    gathered: Mapping[int, bytes], participants: Sequence[int]
) -> tuple[dict[str, object], ...]:
    """Return failures after verifying one well-formed envelope per participant."""

    participant_tuple = tuple(participants)
    expected = set(participant_tuple)
    if len(expected) != len(participant_tuple):
        raise ValueError("phase-status participants must be unique")
    if set(gathered) != expected:
        raise RuntimeError("phase-status exchange did not cover every participant")
    failures: list[dict[str, object]] = []
    for rank in sorted(expected):
        payload = json.loads(gathered[rank].decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("rank") != rank:
            raise RuntimeError(f"rank {rank} published a malformed phase status")
        ok = payload.get("ok")
        if not isinstance(ok, bool):
            raise RuntimeError(f"rank {rank} published a non-boolean phase status")
        if not ok:
            if not isinstance(payload.get("error_type"), str) or not isinstance(
                payload.get("message"), str
            ):
                raise RuntimeError(f"rank {rank} published a malformed failure")
            failures.append(payload)
    return tuple(failures)


def _decode_phase_result_exchange(
    gathered: Mapping[int, bytes], participants: Sequence[int]
) -> tuple[tuple[dict[str, object], ...], dict[int, dict[str, object]]]:
    """Decode one combined validation/result rendezvous in stable-rank order."""

    failures = _decode_phase_failures(gathered, participants)
    if failures:
        return failures, {}
    results: dict[int, dict[str, object]] = {}
    for rank in sorted(set(participants)):
        payload = json.loads(gathered[rank].decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("rank") != rank:
            raise RuntimeError(f"rank {rank} published a malformed phase result")
        result = dict(payload)
        del result["ok"]
        results[rank] = result
    return (), results


if _CUTE_AVAILABLE:

    @cute.jit
    def _remote_wait(
        address,
        expected,
        timeout_ns,
        status_address,
        peer,
        lane,
        abort_address,
        abort_epoch,
        peer_abort_address,
    ):
        """Wait timer-free in production, with an opt-in diagnostic timeout."""

        observed = cutlass.Uint64(0)
        if cutlass.const_expr(timeout_ns == 0):
            observed = nixl_cute.wait_acquire_system_u64_or_aborts(
                address,
                expected,
                abort_address,
                abort_epoch,
                peer_abort_address,
                abort_epoch,
                scope=nixl_cute.Scope.WARP,
            )
        else:
            observed = nixl_cute.wait_acquire_system_u64_for_or_abort(
                address,
                expected,
                abort_address,
                abort_epoch,
                timeout_ns,
                scope=nixl_cute.Scope.WARP,
            )
            if observed < expected:
                if lane == 0:
                    abort_observed = nixl_cute.load_acquire_gpu_u64(abort_address)
                    if abort_observed < abort_epoch:
                        nixl_cute.compare_exchange_status_gpu_i32(
                            status_address + cutlass.Uint64(peer) * 4,
                            int(nixl_cute.NIXL_ERR_REMOTE_DISCONNECT),
                        )
        if observed < expected:
            if lane == 0:
                nixl_cute.atomic_max_release_system_u64(abort_address, abort_epoch)
        return observed

    @cute.jit
    def _remote_wait_thread(
        address,
        expected,
        timeout_ns,
        status_address,
        peer,
        abort_address,
        abort_epoch,
        peer_abort_address,
    ):
        """Lane-zero remote wait used when a later grid join shares visibility."""

        observed = cutlass.Uint64(0)
        if cutlass.const_expr(timeout_ns == 0):
            observed = nixl_cute.wait_acquire_system_u64_or_aborts(
                address,
                expected,
                abort_address,
                abort_epoch,
                peer_abort_address,
                abort_epoch,
                scope=nixl_cute.Scope.THREAD,
            )
        else:
            observed = nixl_cute.wait_acquire_system_u64_for_or_abort(
                address,
                expected,
                abort_address,
                abort_epoch,
                timeout_ns,
                scope=nixl_cute.Scope.THREAD,
            )
            if observed < expected:
                abort_observed = nixl_cute.load_acquire_gpu_u64(abort_address)
                if abort_observed < abort_epoch:
                    nixl_cute.compare_exchange_status_gpu_i32(
                        status_address + cutlass.Uint64(peer) * 4,
                        int(nixl_cute.NIXL_ERR_REMOTE_DISCONNECT),
                    )
        if observed < expected:
            nixl_cute.atomic_max_release_system_u64(abort_address, abort_epoch)
        return observed

    @cute.jit
    def _wait_gpu(address, expected, abort_address, abort_epoch):
        """Warp wait when every lane consumes data covered by the acquire."""

        return nixl_cute.wait_acquire_gpu_u64_or_abort(
            address,
            expected,
            abort_address,
            abort_epoch,
            scope=nixl_cute.Scope.WARP,
        )

    @cute.jit
    def _wait_gpu_thread(address, expected, abort_address, abort_epoch):
        """Lane-zero local join wait; its caller broadcasts once after the group."""

        return nixl_cute.wait_acquire_gpu_u64_or_abort(
            address,
            expected,
            abort_address,
            abort_epoch,
            scope=nixl_cute.Scope.THREAD,
        )

    @cute.kernel
    def _mapped_preflight_kernel(
        remote: nixl_cute.MemoryView,
        rank_mask: cute.Tensor,
        advertised_bases: cute.Tensor,
        statuses: cute.Tensor,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
    ):
        """Classify the exact active non-local descriptors used by this phase."""

        tidx, _, _ = cute.arch.thread_idx()
        if tidx < max_ranks:
            statuses[tidx] = cutlass.Int32(0)
            if rank_mask[rank] == 0:
                if rank_mask[tidx] == 0:
                    if tidx != rank:
                        peer = cutlass.Uint64(
                            cute.make_ptr(
                                cutlass.Uint8,
                                nixl_cute.get_ptr(remote, tidx),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ).toint()
                        )
                        if peer == 0:
                            statuses[tidx] = cutlass.Int32(
                                int(nixl_cute.NIXL_ERR_NOT_SUPPORTED)
                            )
                        if peer != 0:
                            if peer != advertised_bases[tidx]:
                                statuses[tidx] = cutlass.Int32(
                                    int(nixl_cute.NIXL_ERR_MISMATCH)
                                )

    @cute.jit
    def _launch_mapped_preflight(
        remote: nixl_cute.MemoryView,
        rank_mask: cute.Tensor,
        advertised_bases: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
    ):
        _mapped_preflight_kernel(
            remote,
            rank_mask,
            advertised_bases,
            statuses,
            rank,
            max_ranks,
        ).launch(grid=[1, 1, 1], block=[max_ranks, 1, 1], stream=stream)

    @cute.kernel
    def _elastic_moe_ll_kernel(
        remote: nixl_cute.MemoryView,
        arena: cute.Tensor,
        rank_mask: cute.Tensor,
        incarnations: cute.Tensor,
        generation_state: cute.Tensor,
        bucket_counts: cute.Tensor,
        bucket_offsets: cute.Tensor,
        worker_plan: cute.Tensor,
        route_gates: cute.Tensor,
        outputs: cute.Tensor,
        statuses: cute.Tensor,
        timestamps: cute.Tensor,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        num_tokens: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        workers_per_peer: cutlass.Constexpr[int],
        warps_per_cta: cutlass.Constexpr[int],
        warmup: cutlass.Constexpr[int],
        iterations: cutlass.Constexpr[int],
        num_banks: cutlass.Constexpr[int],
        dispatch_stage_base: cutlass.Constexpr[int],
        dispatch_recv_base: cutlass.Constexpr[int],
        dispatch_stamp_base: cutlass.Constexpr[int],
        dispatch_ready_base: cutlass.Constexpr[int],
        combine_consumed_base: cutlass.Constexpr[int],
        abort_state_base: cutlass.Constexpr[int],
        combine_recv_base: cutlass.Constexpr[int],
        combine_stamp_base: cutlass.Constexpr[int],
        combine_ready_base: cutlass.Constexpr[int],
        dispatch_stride: cutlass.Constexpr[int],
        combine_stride: cutlass.Constexpr[int],
        device_timeout_ns: cutlass.Constexpr[int],
        record_timing: cutlass.Constexpr[bool],
        record_cadence: cutlass.Constexpr[bool],
        instrument_per_peer: cutlass.Constexpr[bool],
        validate_every_iteration: cutlass.Constexpr[bool],
    ):
        """Execute all warmup and measured MoE rounds without host pacing."""

        lane = cute.arch.lane_idx()
        warp_in_cta = cute.arch.warp_idx()
        cta, _, _ = cute.arch.block_idx()
        arena_address = arena.iterator.toint()
        output_address = outputs.iterator.toint()
        status_address = statuses.iterator.toint()
        max_experts = max_ranks * experts_per_rank
        fixed_worker_count = max_ranks * workers_per_peer
        worker = cta * warps_per_cta + warp_in_cta
        fixed_worker_ordinal = worker
        # ``copy_atom_call`` interprets mode zero as the atom's V-profile.
        # Keep the rank-one profile explicit: ``make_layout(4)`` has a scalar
        # shape, which CuTe 4.5.1 cannot index as mode zero, whereas ``(4,)``
        # represents the same four contiguous words with the required mode.
        payload_vector_layout = cute.make_layout((4,))
        expert_words_layout = cute.make_layout((4, EXPERT_VECTOR_UNROLL), stride=(1, 4))
        combine_words_layout = cute.make_layout(
            (4, COMBINE_VECTOR_UNROLL), stride=(1, 4)
        )
        combine_values_layout = cute.make_layout(
            (BF16_PER_VECTOR, COMBINE_VECTOR_UNROLL),
            stride=(1, BF16_PER_VECTOR),
        )
        payload_load = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Uint32,
            num_bits_per_copy=128,
            memory_order=cute.nvgpu.MemoryOrder.WEAK,
            memory_scope=cute.nvgpu.MemoryScope.CTA,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.SIZE_256B,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
        )
        payload_store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            cutlass.Uint32,
            num_bits_per_copy=128,
            memory_order=cute.nvgpu.MemoryOrder.WEAK,
            memory_scope=cute.nvgpu.MemoryScope.CTA,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
        )
        expert_words = cute.make_rmem_tensor(expert_words_layout, cutlass.Uint32)
        combine_words = cute.make_rmem_tensor(combine_words_layout, cutlass.Uint32)
        accumulator_values = cute.make_rmem_tensor(
            combine_values_layout, cutlass.Float32
        )

        # Map every fixed CTA onto one active (peer, expert, record-shard) task.
        # Because the validated launch shape has at least one CTA per fixed
        # expert slot, shrink redistributes spare CTAs over the remaining expert
        # buckets instead of retiring their communication capacity.
        rank_is_masked = cutlass.Uint64(1)
        active_peer_count = cutlass.Int32(0)
        if lane == 0:
            rank_is_masked = cutlass.Uint64(rank_mask[rank])
            for candidate in cutlass.range(max_ranks, unroll=1):
                if rank_mask[candidate] == 0:
                    active_peer_count += cutlass.Int32(1)
        rank_is_masked = cute.arch.shuffle_sync(rank_is_masked, 0)
        active_peer_count = cute.arch.shuffle_sync(active_peer_count, 0)
        active_task_count = active_peer_count * experts_per_rank
        task = cutlass.Int32(0)
        if lane == 0:
            task = cutlass.Int32(worker_plan[worker * WORKER_PLAN_FIELDS])
        task = cute.arch.shuffle_sync(task, 0)
        task_peer_ordinal = task // experts_per_rank
        expert = task % experts_per_rank
        task_leader = task
        peer_leader = task_peer_ordinal * experts_per_rank
        has_sibling = task_leader + active_task_count < fixed_worker_count

        peer = cutlass.Int32(0)
        if lane == 0:
            ordinal = cutlass.Int32(0)
            for candidate in cutlass.range(max_ranks, unroll=1):
                if rank_mask[candidate] == 0:
                    if ordinal == task_peer_ordinal:
                        peer = candidate
                    ordinal += cutlass.Int32(1)
        peer = cute.arch.shuffle_sync(peer, 0)
        peer_address = cutlass.Uint64(0)
        if rank_is_masked == 0:
            # This constexpr wrapper keeps the fixed-grid body one indentation
            # level below the inactive-rank guard.
            if cutlass.const_expr(True):
                # Dynamic IR needs a value on every path before later use.
                # The local arena address is a valid, low-cost fallback and is
                # overwritten with the remote peer pointer when needed.
                peer_address = cutlass.Uint64(arena_address)
                if cutlass.const_expr(True):
                    if peer == rank:
                        peer_address = cutlass.Uint64(arena_address)
                    else:
                        if lane == 0:
                            peer_address = cutlass.Uint64(
                                cute.make_ptr(
                                    cutlass.Uint8,
                                    nixl_cute.get_ptr(remote, peer),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ).toint()
                            )
                        peer_address = cute.arch.shuffle_sync(peer_address, 0)
                phase_generation = cutlass.Uint64(0)
                peer_incarnation = cutlass.Uint64(0)
                if lane == 0:
                    phase_generation = cutlass.Uint64(generation_state[0])
                    peer_incarnation = cutlass.Uint64(incarnations[peer])
                phase_generation = cute.arch.shuffle_sync(phase_generation, 0)
                # The peer incarnation is consumed only by lane zero while
                # validating message stamps. Do not broadcast it into every
                # lane of this long-lived persistent loop. The local phase
                # input is immutable and is reloaded at its two publication
                # sites to keep it out of the persistent-loop live set.

                # Routing is phase-stable in this persistent benchmark. Hoist
                # bucket metadata out of the operation loop; no count/offset
                # arithmetic is paid per round. Membership-specific contiguous
                # shard bounds were computed on the host at this drained phase
                # boundary, avoiding CUDA's out-of-line Int64 divide helpers.
                outgoing_global_expert = peer * experts_per_rank + expert
                outgoing_bucket = rank * max_experts + outgoing_global_expert
                outgoing_record_count = cutlass.Int64(0)
                outgoing_packed_offset = cutlass.Int64(0)
                incoming_global_expert = rank * experts_per_rank + expert
                incoming_bucket = peer * max_experts + incoming_global_expert
                incoming_record_count = cutlass.Int64(0)
                if lane == 0:
                    outgoing_record_count = bucket_counts[outgoing_bucket]
                    outgoing_packed_offset = bucket_offsets[outgoing_bucket]
                    incoming_record_count = bucket_counts[incoming_bucket]
                outgoing_record_count = cute.arch.shuffle_sync(outgoing_record_count, 0)
                outgoing_packed_offset = cute.arch.shuffle_sync(
                    outgoing_packed_offset, 0
                )
                incoming_record_count = cute.arch.shuffle_sync(incoming_record_count, 0)
                # Preserve the count component of the ready-word mismatch check
                # after combine publication is coarsened from expert to peer.
                # These routing-table loads execute once before the persistent
                # loop and are excluded from every measured round.
                outgoing_peer_record_count = cutlass.Int64(0)
                incoming_peer_record_count = cutlass.Int64(0)
                if lane == 0:
                    for peer_expert in cutlass.range(experts_per_rank, unroll=1):
                        outgoing_peer_record_count += bucket_counts[
                            rank * max_experts + peer * experts_per_rank + peer_expert
                        ]
                        incoming_peer_record_count += bucket_counts[
                            peer * max_experts + rank * experts_per_rank + peer_expert
                        ]
                outgoing_peer_record_count = cute.arch.shuffle_sync(
                    outgoing_peer_record_count, 0
                )
                incoming_peer_record_count = cute.arch.shuffle_sync(
                    incoming_peer_record_count, 0
                )
                outgoing_shard_begin = cutlass.Int32(0)
                outgoing_shard_count = cutlass.Int32(0)
                incoming_shard_begin = cutlass.Int32(0)
                incoming_shard_count = cutlass.Int32(0)
                if lane == 0:
                    worker_plan_base = worker * WORKER_PLAN_FIELDS
                    outgoing_shard_begin = worker_plan[worker_plan_base + 1]
                    outgoing_shard_count = worker_plan[worker_plan_base + 2]
                    incoming_shard_begin = worker_plan[worker_plan_base + 3]
                    incoming_shard_count = worker_plan[worker_plan_base + 4]
                outgoing_shard_begin = cute.arch.shuffle_sync(outgoing_shard_begin, 0)
                outgoing_shard_count = cute.arch.shuffle_sync(outgoing_shard_count, 0)
                incoming_shard_begin = cute.arch.shuffle_sync(incoming_shard_begin, 0)
                incoming_shard_count = cute.arch.shuffle_sync(incoming_shard_count, 0)
                expert_bias = cutlass.BFloat16(incoming_global_expert + 1)

                # Keep failure warp-uniform and suppress later mapped access
                # from this peer task. Timer-free failures cascade through
                # system-visible owner-local abort epochs.
                peer_ok = cutlass.Int32(1)
                for operation in cutlass.range(warmup + iterations, unroll=1):
                    bank = operation % num_banks
                    bank_cycle = operation // num_banks + 1
                    operation_epoch = phase_generation * cutlass.Uint64(
                        1 << 32
                    ) + cutlass.Uint64(operation)
                    # One phase-wide threshold is essential: ranks need not enter
                    # the same operation simultaneously, so an operation-relative
                    # threshold could miss an earlier peer's terminal abort.
                    abort_epoch = phase_generation + cutlass.Uint64(1)
                    abort_address = arena_address + abort_state_base
                    worker_state_offset = (
                        combine_consumed_base + (bank * fixed_worker_count + worker) * 8
                    )
                    task_leader_offset = (
                        combine_consumed_base
                        + (bank * fixed_worker_count + task_leader) * 8
                    )
                    state_base = cutlass.Uint64(bank_cycle - 1) * cutlass.Uint64(4)
                    dispatch_copy_done_state = state_base + cutlass.Uint64(1)
                    incoming_ready_state = state_base + cutlass.Uint64(2)
                    expert_done_state = state_base + cutlass.Uint64(3)
                    expert_complete_state = state_base + cutlass.Uint64(4)
                    start = cutlass.Uint64(0)

                    # Timing is a compile-time observer policy. The deployment
                    # specialization removes this whole block. Envelope mode
                    # emits only first-start/final-end markers; cadence and peer
                    # modes retain round starts for performance analysis.
                    if cutlass.const_expr(record_timing):
                        if operation >= warmup:
                            if worker == 0:
                                if lane == 0:
                                    sample = (operation - warmup) * (max_ranks + 1) * 2
                                    if cutlass.const_expr(record_cadence):
                                        timestamps[sample] = nixl_cute.globaltimer_ns()
                                    else:
                                        if operation == warmup:
                                            timestamps[sample] = (
                                                nixl_cute.globaltimer_ns()
                                            )
                            if operation == warmup:
                                # Order the first marker before every CTA's
                                # measured work. This timing-only barrier is
                                # absent from the deployment specialization.
                                nixl_cute.sync_grid()

                    # Peer leaders timestamp diagnostic service; sustainable
                    # throughput is derived separately from successive starts.
                    if cutlass.const_expr(instrument_per_peer):
                        if peer_ok != 0:
                            if worker == peer_leader:
                                if operation >= warmup:
                                    if lane == 0:
                                        start = nixl_cute.globaltimer_ns()
                                    start = cute.arch.shuffle_sync(start, 0)

                    # Dispatch one contiguous shard of this CTA's destination
                    # expert bucket. The source stage is immutable for the
                    # kernel lifetime, satisfying the readonly mapped-copy ABI.
                    if cutlass.const_expr(True):
                        source_stage_base = cutlass.Uint64(dispatch_stage_base)
                        if cutlass.const_expr(validate_every_iteration):
                            source_stage_base += cutlass.Uint64(
                                operation
                            ) * cutlass.Uint64(num_tokens * top_k * dispatch_stride)
                        source_offset = source_stage_base + (
                            cutlass.Uint64(outgoing_packed_offset)
                            + cutlass.Uint64(outgoing_shard_begin)
                        ) * cutlass.Uint64(dispatch_stride)

                        # Self dispatch consumes the immutable stage in place.
                        # Only remote tasks copy and participate in the local
                        # shard-to-publication handshake.
                        if peer != rank:
                            receive_item = (
                                (
                                    cutlass.Uint64(bank)
                                    * cutlass.Uint64(experts_per_rank)
                                    + cutlass.Uint64(expert)
                                )
                                * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(rank)
                            ) * cutlass.Uint64(num_tokens)
                            destination_offset = cutlass.Uint64(dispatch_recv_base) + (
                                receive_item + cutlass.Uint64(outgoing_shard_begin)
                            ) * cutlass.Uint64(dispatch_stride)
                            if peer_ok != 0:
                                if outgoing_shard_count > 0:
                                    copy_status = (
                                        nixl_cute.mapped_copy_warp_ptr_readonly(
                                            arena_address + source_offset,
                                            peer_address + destination_offset,
                                            cutlass.Uint64(outgoing_shard_count)
                                            * cutlass.Uint64(dispatch_stride),
                                        )
                                    )
                                    if copy_status != int(nixl_cute.NIXL_SUCCESS):
                                        peer_ok = cutlass.Int32(0)
                                        if lane == 0:
                                            nixl_cute.compare_exchange_status_gpu_i32(
                                                status_address
                                                + cutlass.Uint64(peer) * 4,
                                                copy_status,
                                            )
                                            nixl_cute.atomic_max_release_system_u64(
                                                abort_address, abort_epoch
                                            )
                            if peer_ok != 0:
                                # A leader with no sibling shard can publish the
                                # remote ready word immediately; avoid an otherwise
                                # unconsumed local release store in the full-rank
                                # one-worker-per-task configuration.
                                if has_sibling:
                                    if lane == 0:
                                        nixl_cute.store_release_gpu_u64(
                                            arena_address + worker_state_offset,
                                            dispatch_copy_done_state,
                                        )

                    # The task leader joins only its record shards, then performs
                    # the sole remote publication for this destination expert.
                    if peer != rank:
                        if worker == task_leader:
                            if lane == 0:
                                sibling = task_leader + active_task_count
                                while sibling < fixed_worker_count:
                                    if peer_ok != 0:
                                        sibling_offset = (
                                            combine_consumed_base
                                            + (bank * fixed_worker_count + sibling) * 8
                                        )
                                        observed = _wait_gpu_thread(
                                            arena_address + sibling_offset,
                                            dispatch_copy_done_state,
                                            abort_address,
                                            abort_epoch,
                                        )
                                        if observed < dispatch_copy_done_state:
                                            peer_ok = cutlass.Int32(0)
                                    sibling += active_task_count
                            # Only lane zero needs each acquire for the cumulative
                            # system release. Re-converge once after the join.
                            peer_ok = cute.arch.shuffle_sync(peer_ok, 0)
                            if peer_ok != 0:
                                if lane == 0:
                                    stamp_offset = (
                                        dispatch_stamp_base
                                        + (
                                            (bank * experts_per_rank + expert)
                                            * max_ranks
                                            + rank
                                        )
                                        * MESSAGE_STAMP_NBYTES
                                    )
                                    bucket_stamp = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Uint64,
                                            peer_address + stamp_offset,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout(2),
                                    )
                                    bucket_stamp[0] = operation_epoch
                                    bucket_stamp[1] = cutlass.Uint64(incarnations[rank])
                                    ready_offset = (
                                        dispatch_ready_base
                                        + (
                                            (bank * experts_per_rank + expert)
                                            * max_ranks
                                            + rank
                                        )
                                        * 8
                                    )
                                    publication = (
                                        cutlass.Uint64(bank_cycle) << 32
                                    ) + cutlass.Uint64(outgoing_record_count + 1)
                                    nixl_cute.store_release_system_u64(
                                        peer_address + ready_offset, publication
                                    )

                    # Consume dispatch sent by ``peer``, execute the
                    # deterministic expert, and write results directly into
                    # the origin's route-slot-major combine bank.
                    if cutlass.const_expr(True):
                        shard_record_count = incoming_shard_count
                        # For self, incoming/outgoing bucket identities and shard
                        # intervals are equal. Read the immutable source stage
                        # directly and omit every dispatch-ready protocol word.
                        incoming_offset = source_offset
                        if peer != rank:
                            incoming_item = (
                                (
                                    cutlass.Uint64(bank)
                                    * cutlass.Uint64(experts_per_rank)
                                    + cutlass.Uint64(expert)
                                )
                                * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(peer)
                            ) * cutlass.Uint64(num_tokens)
                            incoming_offset = cutlass.Uint64(dispatch_recv_base) + (
                                incoming_item + cutlass.Uint64(incoming_shard_begin)
                            ) * cutlass.Uint64(dispatch_stride)
                            incoming_ready_offset = (
                                dispatch_ready_base
                                + (
                                    (bank * experts_per_rank + expert) * max_ranks
                                    + peer
                                )
                                * 8
                            )
                            expected_ready = (
                                cutlass.Uint64(bank_cycle) << 32
                            ) + cutlass.Uint64(incoming_record_count + 1)
                            minimum_ready = (
                                cutlass.Uint64(bank_cycle) << 32
                            ) + cutlass.Uint64(1)
                            if worker == task_leader:
                                if peer_ok != 0:
                                    observed = _remote_wait(
                                        arena_address + incoming_ready_offset,
                                        minimum_ready,
                                        device_timeout_ns,
                                        status_address,
                                        peer,
                                        lane,
                                        abort_address,
                                        abort_epoch,
                                        peer_address + abort_state_base,
                                    )
                                    if observed != expected_ready:
                                        peer_ok = cutlass.Int32(0)
                                        if lane == 0:
                                            # A value below minimum_ready is an
                                            # already-published abort or diagnostic
                                            # timeout. Any current/future-cycle value
                                            # with the wrong count is a protocol
                                            # mismatch, including under-publication.
                                            if observed >= minimum_ready:
                                                nixl_cute.compare_exchange_status_gpu_i32(
                                                    status_address
                                                    + cutlass.Uint64(peer) * 4,
                                                    int(nixl_cute.NIXL_ERR_MISMATCH),
                                                )
                                            nixl_cute.atomic_max_release_system_u64(
                                                abort_address, abort_epoch
                                            )
                                    else:
                                        if lane == 0:
                                            stamp_offset = (
                                                dispatch_stamp_base
                                                + (
                                                    (bank * experts_per_rank + expert)
                                                    * max_ranks
                                                    + peer
                                                )
                                                * MESSAGE_STAMP_NBYTES
                                            )
                                            bucket_stamp = cute.make_tensor(
                                                cute.make_ptr(
                                                    cutlass.Uint64,
                                                    arena_address + stamp_offset,
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=16,
                                                ),
                                                cute.make_layout(2),
                                            )
                                            if bucket_stamp[0] != operation_epoch:
                                                peer_ok = cutlass.Int32(0)
                                                nixl_cute.compare_exchange_status_gpu_i32(
                                                    status_address
                                                    + cutlass.Uint64(peer) * 4,
                                                    int(nixl_cute.NIXL_ERR_MISMATCH),
                                                )
                                            if bucket_stamp[1] != peer_incarnation:
                                                peer_ok = cutlass.Int32(0)
                                                nixl_cute.compare_exchange_status_gpu_i32(
                                                    status_address
                                                    + cutlass.Uint64(peer) * 4,
                                                    int(nixl_cute.NIXL_ERR_MISMATCH),
                                                )
                                            if peer_ok == 0:
                                                nixl_cute.atomic_max_release_system_u64(
                                                    abort_address, abort_epoch
                                                )
                                    peer_ok = cute.arch.shuffle_sync(peer_ok, 0)
                                if peer_ok != 0:
                                    if has_sibling:
                                        if lane == 0:
                                            nixl_cute.store_release_gpu_u64(
                                                arena_address + task_leader_offset,
                                                incoming_ready_state,
                                            )
                            else:
                                if peer_ok != 0:
                                    observed = _wait_gpu(
                                        arena_address + task_leader_offset,
                                        incoming_ready_state,
                                        abort_address,
                                        abort_epoch,
                                    )
                                    if observed < incoming_ready_state:
                                        peer_ok = cutlass.Int32(0)
                        if peer_ok == 0:
                            # Prevent stale record reads after an abort without
                            # requiring an unsupported dynamic loop break.
                            shard_record_count = cutlass.Int32(0)

                        if shard_record_count > 0:
                            for slot in cutlass.range(shard_record_count, unroll=1):
                                # A prior invalid record cannot break a dynamic
                                # CuTe loop. Keep iterating without touching any
                                # later record or peer mapping.
                                if peer_ok != 0:
                                    incoming_record = (
                                        incoming_offset + slot * dispatch_stride
                                    )
                                    origin_token = cutlass.Uint32(0)
                                    route_slot = cutlass.Uint32(0)
                                    metadata_ok = cutlass.Int32(1)
                                    if lane == 0:
                                        metadata = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                arena_address + incoming_record,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            cute.make_layout(4),
                                        )
                                        # CuTe 4.5.1 represents scalar tensor
                                        # loads as signed SSA integers unless
                                        # the destination width/signedness is
                                        # explicit. Preserve the Uint32
                                        # region-carried types used after this
                                        # dynamic lane branch.
                                        origin_token = cutlass.Uint32(metadata[0])
                                        route_slot = cutlass.Uint32(metadata[1])
                                        if origin_token >= num_tokens:
                                            metadata_ok = cutlass.Int32(0)
                                        if route_slot >= top_k:
                                            metadata_ok = cutlass.Int32(0)
                                        if metadata_ok == 0:
                                            nixl_cute.compare_exchange_status_gpu_i32(
                                                status_address
                                                + cutlass.Uint64(peer) * 4,
                                                int(nixl_cute.NIXL_ERR_MISMATCH),
                                            )
                                    # Metadata validity is warp-uniform before
                                    # deriving any destination address.
                                    origin_token = cute.arch.shuffle_sync(
                                        origin_token, 0
                                    )
                                    route_slot = cute.arch.shuffle_sync(route_slot, 0)
                                    metadata_ok = cute.arch.shuffle_sync(metadata_ok, 0)
                                    if metadata_ok == 0:
                                        peer_ok = cutlass.Int32(0)
                                        if lane == 0:
                                            nixl_cute.atomic_max_release_system_u64(
                                                abort_address, abort_epoch
                                            )
                                    peer_ok = cute.arch.shuffle_sync(peer_ok, 0)

                                    if peer_ok != 0:
                                        combine_item = (
                                            cutlass.Uint64(bank) * cutlass.Uint64(top_k)
                                            + cutlass.Uint64(route_slot)
                                        ) * cutlass.Uint64(num_tokens) + cutlass.Uint64(
                                            origin_token
                                        )
                                        # Keep the vector expert body in a
                                        # compile-time region; this emits no
                                        # runtime branch.
                                        if cutlass.const_expr(True):
                                            combine_offset = (
                                                combine_recv_base
                                                + combine_item * combine_stride
                                            )
                                            source_address = (
                                                arena_address
                                                + incoming_record
                                                + DISPATCH_METADATA_NBYTES
                                            )
                                            remote_address = (
                                                peer_address + combine_offset
                                            )
                                            total_vectors = (
                                                hidden_size // BF16_PER_VECTOR
                                            )
                                            expert_warp_chunk = (
                                                WARP_SIZE * EXPERT_VECTOR_UNROLL
                                            )
                                            full_vectors = (
                                                total_vectors // expert_warp_chunk
                                            ) * expert_warp_chunk
                                            vector_base = cutlass.Uint32(lane)
                                            while vector_base < full_vectors:
                                                # Expose four independent
                                                # LDG.128 operations before
                                                # consuming any result.
                                                for (
                                                    vector_unroll
                                                ) in cutlass.range_constexpr(
                                                    EXPERT_VECTOR_UNROLL
                                                ):
                                                    vector = (
                                                        vector_base
                                                        + cutlass.Uint32(
                                                            vector_unroll * WARP_SIZE
                                                        )
                                                    )
                                                    byte_offset = cutlass.Uint64(
                                                        vector
                                                    ) * cutlass.Uint64(
                                                        PAYLOAD_VECTOR_NBYTES
                                                    )
                                                    source_vector = cute.make_tensor(
                                                        cute.make_ptr(
                                                            cutlass.Uint32,
                                                            source_address
                                                            + byte_offset,
                                                            cute.AddressSpace.gmem,
                                                            assumed_align=16,
                                                        ),
                                                        payload_vector_layout,
                                                    )
                                                    cute.copy_atom_call(
                                                        payload_load,
                                                        source_vector,
                                                        expert_words[
                                                            (None, vector_unroll)
                                                        ],
                                                    )
                                                for (
                                                    vector_unroll
                                                ) in cutlass.range_constexpr(
                                                    EXPERT_VECTOR_UNROLL
                                                ):
                                                    fragment = expert_words[
                                                        (None, vector_unroll)
                                                    ]
                                                    fragment_bf16 = cute.recast_tensor(
                                                        fragment, cutlass.BFloat16
                                                    )
                                                    fragment_bf16.store(
                                                        fragment_bf16.load()
                                                        + expert_bias
                                                    )
                                                for (
                                                    vector_unroll
                                                ) in cutlass.range_constexpr(
                                                    EXPERT_VECTOR_UNROLL
                                                ):
                                                    vector = (
                                                        vector_base
                                                        + cutlass.Uint32(
                                                            vector_unroll * WARP_SIZE
                                                        )
                                                    )
                                                    byte_offset = cutlass.Uint64(
                                                        vector
                                                    ) * cutlass.Uint64(
                                                        PAYLOAD_VECTOR_NBYTES
                                                    )
                                                    remote_vector = cute.make_tensor(
                                                        cute.make_ptr(
                                                            cutlass.Uint32,
                                                            remote_address
                                                            + byte_offset,
                                                            cute.AddressSpace.gmem,
                                                            assumed_align=16,
                                                        ),
                                                        payload_vector_layout,
                                                    )
                                                    cute.copy_atom_call(
                                                        payload_store,
                                                        expert_words[
                                                            (None, vector_unroll)
                                                        ],
                                                        remote_vector,
                                                    )
                                                vector_base += cutlass.Uint32(
                                                    expert_warp_chunk
                                                )

                                            # At most four warp-strided tail
                                            # rounds, still entirely 128-bit.
                                            # Remove this whole path from the
                                            # common H7168 specialization.
                                            if cutlass.const_expr(
                                                full_vectors != total_vectors
                                            ):
                                                vector = cutlass.Uint32(
                                                    full_vectors
                                                ) + cutlass.Uint32(lane)
                                                while vector < total_vectors:
                                                    byte_offset = cutlass.Uint64(
                                                        vector
                                                    ) * cutlass.Uint64(
                                                        PAYLOAD_VECTOR_NBYTES
                                                    )
                                                    source_vector = cute.make_tensor(
                                                        cute.make_ptr(
                                                            cutlass.Uint32,
                                                            source_address
                                                            + byte_offset,
                                                            cute.AddressSpace.gmem,
                                                            assumed_align=16,
                                                        ),
                                                        payload_vector_layout,
                                                    )
                                                    remote_vector = cute.make_tensor(
                                                        cute.make_ptr(
                                                            cutlass.Uint32,
                                                            remote_address
                                                            + byte_offset,
                                                            cute.AddressSpace.gmem,
                                                            assumed_align=16,
                                                        ),
                                                        payload_vector_layout,
                                                    )
                                                    fragment = expert_words[(None, 0)]
                                                    cute.copy_atom_call(
                                                        payload_load,
                                                        source_vector,
                                                        fragment,
                                                    )
                                                    fragment_bf16 = cute.recast_tensor(
                                                        fragment, cutlass.BFloat16
                                                    )
                                                    fragment_bf16.store(
                                                        fragment_bf16.load()
                                                        + expert_bias
                                                    )
                                                    cute.copy_atom_call(
                                                        payload_store,
                                                        fragment,
                                                        remote_vector,
                                                    )
                                                    vector += cutlass.Uint32(WARP_SIZE)
                        # Remote output needs a cumulative lane-zero publication.
                        # Self output is already local, and the common grid join
                        # below orders every writer before any reducer.
                        if peer != rank:
                            if peer_ok != 0:
                                if shard_record_count > 0:
                                    cute.arch.sync_warp()
                            if peer_ok != 0:
                                if has_sibling:
                                    if lane == 0:
                                        nixl_cute.store_release_gpu_u64(
                                            arena_address + worker_state_offset,
                                            expert_done_state,
                                        )
                            if worker == task_leader:
                                if lane == 0:
                                    sibling = task_leader + active_task_count
                                    while sibling < fixed_worker_count:
                                        if peer_ok != 0:
                                            sibling_offset = (
                                                combine_consumed_base
                                                + (bank * fixed_worker_count + sibling)
                                                * 8
                                            )
                                            observed = _wait_gpu_thread(
                                                arena_address + sibling_offset,
                                                expert_done_state,
                                                abort_address,
                                                abort_epoch,
                                            )
                                            if observed < expert_done_state:
                                                peer_ok = cutlass.Int32(0)
                                        sibling += active_task_count
                                peer_ok = cute.arch.shuffle_sync(peer_ok, 0)
                                if peer_ok != 0:
                                    if lane == 0:
                                        # Expert zero is the peer leader itself, so
                                        # its completion is program ordered and
                                        # needs no self-publication or self-wait.
                                        if expert != 0:
                                            nixl_cute.store_release_gpu_u64(
                                                arena_address + task_leader_offset,
                                                expert_complete_state,
                                            )

                    # One peer leader acquires every expert completion.  Its
                    # single system-scope release then cumulatively publishes
                    # all combine payloads to the origin rank.  Coarsening this
                    # join from expert to peer removes E-1 remote ready stores
                    # and E-1 remote polls without delaying reduction: all CTAs
                    # already meet at the following cooperative grid barrier.
                    if peer != rank:
                        if worker == peer_leader:
                            if lane == 0:
                                peer_expert = cutlass.Int32(0)
                                while peer_expert < experts_per_rank:
                                    expert_leader = peer_leader + peer_expert
                                    if peer_ok != 0:
                                        if expert_leader != worker:
                                            expert_leader_offset = (
                                                combine_consumed_base
                                                + (
                                                    bank * fixed_worker_count
                                                    + expert_leader
                                                )
                                                * 8
                                            )
                                            observed = _wait_gpu_thread(
                                                arena_address + expert_leader_offset,
                                                expert_complete_state,
                                                abort_address,
                                                abort_epoch,
                                            )
                                            if observed < expert_complete_state:
                                                peer_ok = cutlass.Int32(0)
                                    peer_expert += cutlass.Int32(1)
                            peer_ok = cute.arch.shuffle_sync(peer_ok, 0)
                            if peer_ok != 0:
                                if lane == 0:
                                    stamp_offset = (
                                        combine_stamp_base
                                        + (bank * max_ranks + rank)
                                        * MESSAGE_STAMP_NBYTES
                                    )
                                    peer_stamp = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Uint64,
                                            peer_address + stamp_offset,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout(2),
                                    )
                                    peer_stamp[0] = operation_epoch
                                    peer_stamp[1] = cutlass.Uint64(incarnations[rank])
                                    combine_ready_offset = (
                                        combine_ready_base
                                        + (bank * max_ranks + rank) * 8
                                    )
                                    nixl_cute.store_release_system_u64(
                                        peer_address + combine_ready_offset,
                                        (cutlass.Uint64(bank_cycle) << 32)
                                        + cutlass.Uint64(
                                            incoming_peer_record_count + 1
                                        ),
                                    )

                    # One peer leader acquires the reverse aggregate publication
                    # for every outgoing expert bucket targeting that peer.  The
                    # cooperative grid barrier then distributes visibility to
                    # all reducers before any combine payload is consumed.
                    if peer != rank:
                        if worker == peer_leader:
                            if lane == 0:
                                combine_ready_offset = (
                                    combine_ready_base + (bank * max_ranks + peer) * 8
                                )
                                expected_ready = (
                                    cutlass.Uint64(bank_cycle) << 32
                                ) + cutlass.Uint64(outgoing_peer_record_count + 1)
                                minimum_ready = (
                                    cutlass.Uint64(bank_cycle) << 32
                                ) + cutlass.Uint64(1)
                                if peer_ok != 0:
                                    observed = _remote_wait_thread(
                                        arena_address + combine_ready_offset,
                                        minimum_ready,
                                        device_timeout_ns,
                                        status_address,
                                        peer,
                                        abort_address,
                                        abort_epoch,
                                        peer_address + abort_state_base,
                                    )
                                    if observed != expected_ready:
                                        peer_ok = cutlass.Int32(0)
                                        if observed >= minimum_ready:
                                            nixl_cute.compare_exchange_status_gpu_i32(
                                                status_address
                                                + cutlass.Uint64(peer) * 4,
                                                int(nixl_cute.NIXL_ERR_MISMATCH),
                                            )
                                        nixl_cute.atomic_max_release_system_u64(
                                            abort_address, abort_epoch
                                        )
                                    else:
                                        stamp_offset = (
                                            combine_stamp_base
                                            + (bank * max_ranks + peer)
                                            * MESSAGE_STAMP_NBYTES
                                        )
                                        bucket_stamp = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint64,
                                                arena_address + stamp_offset,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            cute.make_layout(2),
                                        )
                                        if bucket_stamp[0] != operation_epoch:
                                            peer_ok = cutlass.Int32(0)
                                            nixl_cute.compare_exchange_status_gpu_i32(
                                                status_address
                                                + cutlass.Uint64(peer) * 4,
                                                int(nixl_cute.NIXL_ERR_MISMATCH),
                                            )
                                        if bucket_stamp[1] != peer_incarnation:
                                            peer_ok = cutlass.Int32(0)
                                            nixl_cute.compare_exchange_status_gpu_i32(
                                                status_address
                                                + cutlass.Uint64(peer) * 4,
                                                int(nixl_cute.NIXL_ERR_MISMATCH),
                                            )
                                        if peer_ok == 0:
                                            nixl_cute.atomic_max_release_system_u64(
                                                abort_address, abort_epoch
                                            )
                            # The following grid barrier distributes lane zero's
                            # system acquire to every local reducer.
                            peer_ok = cute.arch.shuffle_sync(peer_ok, 0)
                    # A cooperative grid barrier replaces the O(active tasks)
                    # software coordinator. Every CTA reaches it even after a
                    # diagnostic failure; the shared abort epoch then makes the
                    # failure state uniform before any combine payload is read.
                    nixl_cute.sync_grid()
                    abort_observed = cutlass.Uint64(0)
                    if lane == 0:
                        abort_observed = nixl_cute.load_acquire_gpu_u64(abort_address)
                    abort_observed = cute.arch.shuffle_sync(abort_observed, 0)
                    if abort_observed >= abort_epoch:
                        peer_ok = cutlass.Int32(0)

                    # The fixed cooperative grid reduces a stable token
                    # partition. Membership shrink therefore preserves the same
                    # local memory-level parallelism. Uint32 manual stepping is
                    # intentionally used: CuTe 4.5.1 lowers it without div/rem or
                    # the signed range helper's abs/select prologue.
                    token = cutlass.Uint32(fixed_worker_ordinal)
                    while token < num_tokens:
                        lane_route_gate = cutlass.Float32(0.0)
                        if peer_ok != 0:
                            if lane < top_k:
                                route_item = token * top_k + lane
                                lane_route_gate = route_gates[route_item]

                        if peer_ok != 0:
                            route_gates_rmem = cute.make_rmem_tensor(
                                cute.make_layout(top_k), cutlass.Float32
                            )
                            for route_slot in cutlass.range_constexpr(top_k):
                                route_gates_rmem[route_slot] = cute.arch.shuffle_sync(
                                    lane_route_gate, route_slot
                                )
                            output_element_base = cutlass.Uint64(0)
                            if cutlass.const_expr(validate_every_iteration):
                                if operation >= warmup:
                                    output_element_base = cutlass.Uint64(
                                        operation - warmup
                                    ) * cutlass.Uint64(num_tokens * hidden_size)
                            output_token_address = output_address + (
                                output_element_base
                                + cutlass.Uint64(token) * cutlass.Uint64(hidden_size)
                            ) * cutlass.Uint64(ELEMENT_NBYTES)
                            result_token_base = (
                                arena_address
                                + combine_recv_base
                                + (
                                    cutlass.Uint64(bank)
                                    * cutlass.Uint64(top_k * num_tokens)
                                    + cutlass.Uint64(token)
                                )
                                * combine_stride
                            )
                            total_vectors = hidden_size // BF16_PER_VECTOR
                            combine_warp_chunk = WARP_SIZE * COMBINE_VECTOR_UNROLL
                            full_vectors = (
                                total_vectors // combine_warp_chunk
                            ) * combine_warp_chunk
                            vector_base = cutlass.Uint32(lane)
                            while vector_base < full_vectors:
                                accumulator_values.fill(0.0)
                                for route_slot in cutlass.range_constexpr(top_k):
                                    route_base = result_token_base + cutlass.Uint64(
                                        route_slot * num_tokens * combine_stride
                                    )
                                    # Issue two independent LDG.128 operations
                                    # before the dependent BF16-to-FP32 work.
                                    for vector_unroll in cutlass.range_constexpr(
                                        COMBINE_VECTOR_UNROLL
                                    ):
                                        vector = vector_base + cutlass.Uint32(
                                            vector_unroll * WARP_SIZE
                                        )
                                        byte_offset = cutlass.Uint64(
                                            vector
                                        ) * cutlass.Uint64(PAYLOAD_VECTOR_NBYTES)
                                        result_vector = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                route_base + byte_offset,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            payload_vector_layout,
                                        )
                                        cute.copy_atom_call(
                                            payload_load,
                                            result_vector,
                                            combine_words[(None, vector_unroll)],
                                        )
                                    for vector_unroll in cutlass.range_constexpr(
                                        COMBINE_VECTOR_UNROLL
                                    ):
                                        accumulator = accumulator_values[
                                            (None, vector_unroll)
                                        ]
                                        combine_bf16 = cute.recast_tensor(
                                            combine_words[(None, vector_unroll)],
                                            cutlass.BFloat16,
                                        )
                                        accumulator.store(
                                            accumulator.load()
                                            + combine_bf16.load().to(cutlass.Float32)
                                            * route_gates_rmem[route_slot]
                                        )
                                for vector_unroll in cutlass.range_constexpr(
                                    COMBINE_VECTOR_UNROLL
                                ):
                                    vector = vector_base + cutlass.Uint32(
                                        vector_unroll * WARP_SIZE
                                    )
                                    byte_offset = cutlass.Uint64(
                                        vector
                                    ) * cutlass.Uint64(PAYLOAD_VECTOR_NBYTES)
                                    output_vector = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Uint32,
                                            output_token_address + byte_offset,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        payload_vector_layout,
                                    )
                                    fragment = combine_words[(None, vector_unroll)]
                                    fragment_bf16 = cute.recast_tensor(
                                        fragment, cutlass.BFloat16
                                    )
                                    fragment_bf16.store(
                                        accumulator_values[(None, vector_unroll)]
                                        .load()
                                        .to(cutlass.BFloat16)
                                    )
                                    cute.copy_atom_call(
                                        payload_store,
                                        fragment,
                                        output_vector,
                                    )
                                vector_base += cutlass.Uint32(combine_warp_chunk)

                            # Generic U1 tail keeps the H%8 contract and avoids
                            # a scalar fallback for less common hidden sizes.
                            # H7168 has no tail, so specialize the path away.
                            if cutlass.const_expr(full_vectors != total_vectors):
                                vector = cutlass.Uint32(full_vectors) + cutlass.Uint32(
                                    lane
                                )
                                while vector < total_vectors:
                                    byte_offset = cutlass.Uint64(
                                        vector
                                    ) * cutlass.Uint64(PAYLOAD_VECTOR_NBYTES)
                                    accumulator = accumulator_values[(None, 0)]
                                    accumulator.fill(0.0)
                                    for route_slot in cutlass.range_constexpr(top_k):
                                        result_vector = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                result_token_base
                                                + cutlass.Uint64(
                                                    route_slot
                                                    * num_tokens
                                                    * combine_stride
                                                )
                                                + byte_offset,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            payload_vector_layout,
                                        )
                                        fragment = combine_words[(None, 0)]
                                        cute.copy_atom_call(
                                            payload_load,
                                            result_vector,
                                            fragment,
                                        )
                                        fragment_bf16 = cute.recast_tensor(
                                            fragment, cutlass.BFloat16
                                        )
                                        accumulator.store(
                                            accumulator.load()
                                            + fragment_bf16.load().to(cutlass.Float32)
                                            * route_gates_rmem[route_slot]
                                        )
                                    output_vector = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Uint32,
                                            output_token_address + byte_offset,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        payload_vector_layout,
                                    )
                                    fragment = combine_words[(None, 0)]
                                    fragment_bf16 = cute.recast_tensor(
                                        fragment, cutlass.BFloat16
                                    )
                                    fragment_bf16.store(
                                        accumulator.load().to(cutlass.BFloat16)
                                    )
                                    cute.copy_atom_call(
                                        payload_store,
                                        fragment,
                                        output_vector,
                                    )
                                    vector += cutlass.Uint32(WARP_SIZE)
                        token += cutlass.Uint32(fixed_worker_count)

                    # One bank needs an immediate all-reader join before the peer
                    # may overwrite it in the next operation. With two banks, the
                    # retained pre-reduction grid barrier in operation N+1 joins
                    # every operation-N reader before any CTA can publish
                    # dispatch-ready for N+2, when bank N is reused. Remove this
                    # barrier from the uninstrumented two-bank specialization.
                    # Envelope timing needs it only at the terminal endpoint;
                    # cadence/peer timing needs it on measured rounds so the next
                    # start follows every prior output reader. The first measured
                    # timing barrier absorbs warmup skew.
                    if cutlass.const_expr(num_banks == 1):
                        nixl_cute.sync_grid()
                    else:
                        if cutlass.const_expr(record_timing):
                            if cutlass.const_expr(record_cadence):
                                if operation >= warmup:
                                    nixl_cute.sync_grid()
                            else:
                                if operation == warmup + iterations - 1:
                                    nixl_cute.sync_grid()

                    if cutlass.const_expr(record_timing):
                        if operation >= warmup:
                            if worker == 0:
                                if lane == 0:
                                    if cutlass.const_expr(instrument_per_peer):
                                        timestamps[
                                            (operation - warmup) * (max_ranks + 1) * 2
                                            + 1
                                        ] = nixl_cute.globaltimer_ns()
                                    else:
                                        if operation == warmup + iterations - 1:
                                            timestamps[
                                                (operation - warmup)
                                                * (max_ranks + 1)
                                                * 2
                                                + 1
                                            ] = nixl_cute.globaltimer_ns()

                    if cutlass.const_expr(instrument_per_peer):
                        if peer_ok != 0:
                            if worker == peer_leader:
                                if operation >= warmup:
                                    if lane == 0:
                                        peer_sample = (
                                            (operation - warmup) * (max_ranks + 1)
                                            + peer
                                            + 1
                                        ) * 2
                                        timestamps[peer_sample] = start
                                        timestamps[peer_sample + 1] = (
                                            nixl_cute.globaltimer_ns()
                                        )

    @cute.jit
    def _launch_elastic_moe_ll(
        remote: nixl_cute.MemoryView,
        arena: cute.Tensor,
        rank_mask: cute.Tensor,
        incarnations: cute.Tensor,
        generation_state: cute.Tensor,
        bucket_counts: cute.Tensor,
        bucket_offsets: cute.Tensor,
        worker_plan: cute.Tensor,
        route_gates: cute.Tensor,
        outputs: cute.Tensor,
        statuses: cute.Tensor,
        timestamps: cute.Tensor,
        stream: cuda.CUstream,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        num_tokens: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        workers_per_peer: cutlass.Constexpr[int],
        warps_per_cta: cutlass.Constexpr[int],
        warmup: cutlass.Constexpr[int],
        iterations: cutlass.Constexpr[int],
        num_banks: cutlass.Constexpr[int],
        dispatch_stage_base: cutlass.Constexpr[int],
        dispatch_recv_base: cutlass.Constexpr[int],
        dispatch_stamp_base: cutlass.Constexpr[int],
        dispatch_ready_base: cutlass.Constexpr[int],
        combine_consumed_base: cutlass.Constexpr[int],
        abort_state_base: cutlass.Constexpr[int],
        combine_recv_base: cutlass.Constexpr[int],
        combine_stamp_base: cutlass.Constexpr[int],
        combine_ready_base: cutlass.Constexpr[int],
        dispatch_stride: cutlass.Constexpr[int],
        combine_stride: cutlass.Constexpr[int],
        device_timeout_ns: cutlass.Constexpr[int],
        record_timing: cutlass.Constexpr[bool],
        record_cadence: cutlass.Constexpr[bool],
        instrument_per_peer: cutlass.Constexpr[bool],
        validate_every_iteration: cutlass.Constexpr[bool],
    ):
        _elastic_moe_ll_kernel(
            remote,
            arena,
            rank_mask,
            incarnations,
            generation_state,
            bucket_counts,
            bucket_offsets,
            worker_plan,
            route_gates,
            outputs,
            statuses,
            timestamps,
            rank,
            max_ranks,
            experts_per_rank,
            num_tokens,
            top_k,
            hidden_size,
            workers_per_peer,
            warps_per_cta,
            warmup,
            iterations,
            num_banks,
            dispatch_stage_base,
            dispatch_recv_base,
            dispatch_stamp_base,
            dispatch_ready_base,
            combine_consumed_base,
            abort_state_base,
            combine_recv_base,
            combine_stamp_base,
            combine_ready_base,
            dispatch_stride,
            combine_stride,
            device_timeout_ns,
            record_timing,
            record_cadence,
            instrument_per_peer,
            validate_every_iteration,
        ).launch(
            grid=[max_ranks * workers_per_peer // warps_per_cta, 1, 1],
            block=[WARP_SIZE * warps_per_cta, 1, 1],
            stream=stream,
            cooperative=True,
            # The qualified H7168 grid has only 16 CTAs on 152 SMs. Giving
            # ptxas the true one-CTA residency requirement keeps the mapped
            # copy's eight-vector load-ahead window in registers, without
            # reducing useful grid occupancy.
            min_blocks_per_mp=1,
        )


def _region(tensor: torch.Tensor) -> DeviceRegion:
    return DeviceRegion(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.get_device()
    )


def _rank_codegen_dump_directory(
    root: str, rank: int, *, incarnation: int | None = None
) -> Path:
    """Create one fresh rank-private directory for every CuTe compile dump."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("codegen rank must be a non-negative integer")
    if incarnation is not None and (
        isinstance(incarnation, bool)
        or not isinstance(incarnation, int)
        or incarnation <= 0
    ):
        raise ValueError("codegen incarnation must be a positive integer")
    parent = Path(root).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("codegen dump root must be an existing directory")
    suffix = f"rank-{rank}"
    if incarnation is not None:
        suffix += f"-incarnation-{incarnation}"
    rank_dump = parent / suffix
    if rank_dump.exists() or rank_dump.is_symlink():
        raise RuntimeError(f"refusing to reuse CuTe codegen directory: {rank_dump}")
    rank_dump.mkdir(mode=0o700, exist_ok=False)
    return rank_dump


def _compile_dump_options(codegen_dump_dir: Path, *, keep_cubin: bool) -> str:
    """Return pinned CuTe 4.5.1 options with a shell-safe explicit dump path."""

    prefix = "--keep-cubin " if keep_cubin else ""
    return prefix + f"--dump-dir={shlex.quote(str(codegen_dump_dir))}"


def _compile_kernels(
    case: ElasticLLCase,
    rank: int,
    tensors: tuple[torch.Tensor, ...],
    stream: torch.cuda.Stream,
    codegen_dump_dir: Path,
):
    if not _CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for this example")
    (
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
    ) = tensors
    layout = case.layout
    fake_remote = nixl_cute.make_fake_memory_view(
        "remote", (layout.arena_nbytes,) * case.max_ranks
    )
    preflight = nixl_cute.compile(
        _launch_mapped_preflight,
        fake_remote,
        from_dlpack(rank_mask).mark_layout_dynamic(),
        from_dlpack(advertised_bases).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        rank,
        case.max_ranks,
        options=_compile_dump_options(codegen_dump_dir, keep_cubin=False),
    )
    main = nixl_cute.compile(
        _launch_elastic_moe_ll,
        fake_remote,
        from_dlpack(arena).mark_layout_dynamic(),
        from_dlpack(rank_mask).mark_layout_dynamic(),
        from_dlpack(incarnations).mark_layout_dynamic(),
        from_dlpack(generation_state).mark_layout_dynamic(),
        from_dlpack(bucket_counts).mark_layout_dynamic(),
        from_dlpack(bucket_offsets).mark_layout_dynamic(),
        from_dlpack(worker_plan).mark_layout_dynamic(),
        from_dlpack(route_gates).mark_layout_dynamic(),
        from_dlpack(outputs).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        from_dlpack(timestamps).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        rank,
        case.max_ranks,
        case.experts_per_rank,
        case.num_tokens,
        case.top_k,
        case.hidden_size,
        case.workers_per_peer,
        case.warps_per_cta,
        case.warmup,
        case.iterations,
        case.num_banks,
        layout.region("dispatch_stage").offset,
        layout.region("dispatch_recv").offset,
        layout.region("dispatch_stamp").offset,
        layout.region("dispatch_ready").offset,
        layout.region("combine_consumed").offset,
        layout.region("abort_state").offset,
        layout.region("combine_recv").offset,
        layout.region("combine_stamp").offset,
        layout.region("combine_ready").offset,
        layout.dispatch_record_stride,
        layout.combine_record_stride,
        case.device_timeout_ns,
        case.timing_mode != "none",
        case.timing_mode in {"cadence", "peer"},
        case.instrument_per_peer,
        case.validate_every_iteration,
        options=_compile_dump_options(codegen_dump_dir, keep_cubin=True),
    )
    main, occupancy = bind_and_validate_cooperative_launch(
        main,
        device_ordinal=arena.get_device(),
        block_threads=WARP_SIZE * case.warps_per_cta,
        planned_ctas=(case.max_ranks * case.workers_per_peer // case.warps_per_cta),
    )
    return preflight, main, occupancy


def _classify_mapped_preflight(
    statuses: torch.Tensor,
    phase: PhaseSpec,
    rank: int,
    allow_unverified_mapped: bool,
) -> dict[str, object]:
    """Validate and describe process-local mappings outside the hot path.

    ``nixlGetPtr`` returns an address in the importing process. CUDA IPC does
    not promise that this number equals the allocation owner's address in a
    different process, and no equality is needed because the kernel computes
    every remote address from the returned base. The worker separately requires
    native peer atomics for every directed GPU pair before this allocation is
    prepared. ``allow_unverified_mapped`` is therefore a deprecated
    compatibility input with no policy effect.
    """

    if not isinstance(allow_unverified_mapped, bool):
        raise TypeError("allow_unverified_mapped must be bool")
    if not 0 <= rank < len(phase.rank_incarnations):
        raise ValueError("rank is outside the fixed-capacity membership table")
    values = [int(value) for value in statuses.cpu().tolist()]
    if len(values) != len(phase.rank_incarnations):
        raise ValueError("mapped-preflight status tensor has an unexpected shape")
    rank_active = rank in phase.active_ranks
    remote_peers: dict[str, object] = {}
    if rank_active:
        for peer in phase.active_ranks:
            if peer == rank:
                continue
            status = values[peer]
            if status == 0:
                classification = "owner_va_coincident"
            elif status == _STATUS_MISMATCH:
                classification = "translated_process_local_va"
            elif status == _STATUS_NOT_SUPPORTED:
                classification = "unmapped"
            else:
                raise RuntimeError(
                    f"mapped preflight failed for rank {rank}, peer {peer}: "
                    f"unexpected status {status}"
                )
            remote_peers[str(peer)] = {
                "classification": classification,
                "status": status,
            }
            if classification == "unmapped":
                raise RuntimeError(
                    f"active peer {peer} is not locally mapped for rank {rank}; "
                    "elastic_moe_ll deliberately has no generated-output "
                    "network fallback"
                )
    classifications = {
        str(peer["classification"])
        for peer in remote_peers.values()
        if isinstance(peer, dict)
    }
    return {
        "policy": "require_non_null_process_local_pointer",
        "evidence_scope": "exact timed-phase device view and arena allocation",
        "rank_active": rank_active,
        "remote_peers": remote_peers,
        "all_active_remote_peers_mapped": classifications
        <= {"owner_va_coincident", "translated_process_local_va"},
        "translated_va_observed": "translated_process_local_va" in classifications,
        "legacy_allow_unverified_mapped_ignored": allow_unverified_mapped,
    }


def _validate_statuses(statuses: torch.Tensor, active_ranks: Sequence[int]) -> None:
    values = [int(value) for value in statuses.cpu().tolist()]
    failures = [(rank, values[rank]) for rank in active_ranks if values[rank] != 0]
    if failures:
        if any(status == _STATUS_NOT_SUPPORTED for _, status in failures):
            raise RuntimeError(
                "active peer is not locally mapped; elastic_moe_ll deliberately "
                "does not use the unsafe generated-output network fallback: "
                f"{failures}"
            )
        raise RuntimeError(f"NIXL/CuTe device protocol failed: {failures}")


def _validate_outputs(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    outputs: torch.Tensor,
) -> None:
    actual = outputs.cpu().reshape(
        case.output_copies, case.num_tokens, case.hidden_size
    )
    poison = torch.tensor(OUTPUT_POISON, dtype=torch.bfloat16)
    for output_copy in range(case.output_copies):
        operation = case.warmup + output_copy if case.validate_every_iteration else 0
        expected = expected_outputs(case, phase, rank, operation)
        for token in range(case.num_tokens):
            if token in expected:
                if not torch.equal(actual[output_copy, token], expected[token]):
                    delta = (
                        (actual[output_copy, token].float() - expected[token].float())
                        .abs()
                        .max()
                        .item()
                    )
                    raise RuntimeError(
                        f"rank {rank} generation {phase.generation} operation "
                        f"{operation} token {token} failed exact BF16 combine "
                        f"validation (max abs {delta})"
                    )
            elif not bool(torch.all(actual[output_copy, token] == poison).item()):
                raise RuntimeError(
                    f"rank {rank} generation {phase.generation} operation "
                    f"{operation} wrote unrouted token slot {token}"
                )


def _rank_phase_result(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    timestamps: torch.Tensor,
) -> dict[str, object]:
    active = set(phase.active_ranks)
    rank_is_active = rank in active
    round_starts: list[int] = []
    round_ends: list[int] = []
    samples: list[int] = []
    remote_samples: list[int] = []
    per_peer: dict[str, list[int]] = {
        str(peer): [] for peer in phase.active_ranks if rank_is_active
    }
    if case.timing_mode == "none":
        # Do not even inspect the ABI placeholder. This keeps the production
        # specialization free of an implicit device-to-host synchronization if
        # a caller passes its device tensor directly.
        return {
            "rank": rank,
            "active": rank_is_active,
            "rank_round_starts_ns": round_starts,
            "rank_round_ends_ns": round_ends,
            "rank_observer_span_ns": [],
            "peer_round_ns": samples,
            "remote_peer_round_ns": remote_samples,
            "per_peer_round_ns": per_peer,
            "operation_steps": case.iterations,
            "timing_mode": case.timing_mode,
            "correctness": "PASS",
        }

    raw = [int(value) for value in timestamps.cpu().tolist()]
    slots_per_iteration = case.max_ranks + 1
    expected_values = case.iterations * slots_per_iteration * 2
    if len(raw) != expected_values:
        raise RuntimeError("timestamp tensor has an unexpected shape")
    first_rank_start = raw[0] if raw else 0
    for iteration in range(case.iterations):
        iteration_base = iteration * slots_per_iteration * 2
        round_start, round_end = raw[iteration_base : iteration_base + 2]
        if rank_is_active:
            start_is_required = case.timing_mode in {"cadence", "peer"} or (
                case.timing_mode == "envelope" and iteration == 0
            )
            end_is_required = case.timing_mode == "peer" or (
                case.timing_mode in {"envelope", "cadence"}
                and iteration == case.iterations - 1
            )
            if start_is_required and round_start == 0:
                raise RuntimeError(
                    f"invalid round-start GPU timestamp for rank {rank}, "
                    f"iteration {iteration}: {round_start}"
                )
            if not start_is_required and round_start != 0:
                raise RuntimeError(
                    f"unnecessary round-start GPU timestamp for rank {rank}, "
                    f"iteration {iteration}: {round_start}"
                )
            comparison_start = round_start if start_is_required else first_rank_start
            if end_is_required and (round_end == 0 or round_end < comparison_start):
                raise RuntimeError(
                    f"invalid round-end GPU timestamp for rank {rank}, "
                    f"iteration {iteration}: {(round_start, round_end)}"
                )
            if not end_is_required and round_end != 0:
                raise RuntimeError(
                    f"rank {rank} emitted an unnecessary end marker "
                    f"for iteration {iteration}"
                )
            round_starts.append(round_start)
            round_ends.append(round_end)
        elif round_start != 0 or round_end != 0:
            raise RuntimeError("inactive rank unexpectedly produced a timing sample")

        for peer in range(case.max_ranks):
            base = iteration_base + (peer + 1) * 2
            start, end = raw[base], raw[base + 1]
            should_exist = (
                case.instrument_per_peer and rank_is_active and peer in active
            )
            if should_exist:
                if start == 0 or end < start:
                    raise RuntimeError(
                        f"invalid GPU timestamps for rank {rank}, peer {peer}, "
                        f"iteration {iteration}: {(start, end)}"
                    )
                elapsed = end - start
                samples.append(elapsed)
                per_peer[str(peer)].append(elapsed)
                if peer != rank:
                    remote_samples.append(elapsed)
            elif start != 0 or end != 0:
                raise RuntimeError(
                    "disabled or inactive peer unexpectedly produced a timing sample"
                )
    return {
        "rank": rank,
        "active": rank_is_active,
        "rank_round_starts_ns": round_starts,
        "rank_round_ends_ns": round_ends,
        "rank_observer_span_ns": (
            [end - start for start, end in zip(round_starts, round_ends)]
            if case.timing_mode == "peer"
            else []
        ),
        "peer_round_ns": samples,
        "remote_peer_round_ns": remote_samples,
        "per_peer_round_ns": per_peer,
        "operation_steps": case.iterations,
        "timing_mode": case.timing_mode,
        "correctness": "PASS",
    }


def _phase_remote_coordinates(
    case: ElasticLLCase,
    phase: PhaseSpec,
    rank: int,
    peers: Sequence[PeerCoordinates],
) -> list[tuple[int, int, int, str | None]]:
    """Build a stable-index view with safe local-loopback padding."""

    coordinates: list[tuple[int, int, int, str | None]] = []
    active = set(phase.active_ranks)
    local = peers[rank]
    for slot in range(case.max_ranks):
        if slot in active and slot != rank:
            peer = peers[slot]
        else:
            # UCX v1.23.x cannot create an all-gap device list and its raw
            # transport entry points do not accept a gap index. Point every
            # inactive/self slot at this rank's registered loopback region;
            # rank_mask and the dedicated self path ensure it is never used as
            # a remote destination. This costs nothing in the persistent loop.
            peer = local
        region = peer.regions[0]
        coordinates.append(
            (region.address, region.length, region.device_id, peer.agent_name)
        )
    return coordinates


def _deserialize_rank_coordinates(
    serialized_coordinates: dict[int, bytes], max_ranks: int
) -> tuple[PeerCoordinates, ...]:
    """Decode an exchange result in stable-rank order, independent of dict order."""

    expected_ranks = tuple(range(max_ranks))
    expected_rank_set = set(expected_ranks)
    if set(serialized_coordinates) != expected_rank_set:
        missing = [
            rank for rank in expected_ranks if rank not in serialized_coordinates
        ]
        unexpected = [
            rank for rank in serialized_coordinates if rank not in expected_rank_set
        ]
        raise RuntimeError(
            "arena coordinate exchange returned invalid rank keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(
        PeerCoordinates.from_bytes(serialized_coordinates[peer_rank])
        for peer_rank in expected_ranks
    )


def _worker(
    rank: int,
    devices: tuple[int, ...],
    directory: str,
    case: ElasticLLCase,
    timeout_s: float,
    codegen_dump_root: str,
) -> None:
    if not _CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for this example")

    from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    codegen_dump_dir = _rank_codegen_dump_directory(codegen_dump_root, rank)
    device = devices[rank]
    torch.cuda.set_device(device)
    native_atomic_preflight = nixl_cute.require_peer_native_atomics(
        devices, accessing_devices=(device,)
    )
    stream = torch.cuda.Stream(device=device)
    layout = case.layout
    if layout.arena_nbytes > torch.cuda.get_device_properties(device).total_memory:
        raise ValueError("fixed-capacity LL arena exceeds total GPU memory")

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
    host_inputs = PinnedPhaseInputs.allocate(case)
    host_results = PinnedPhaseResults.allocate(case)

    control = FileControlPlane(directory, rank, case.max_ranks, timeout_s)
    run_id = Path(directory).name
    name = f"cute_elastic_ll_{run_id}_{rank}"
    agent = nixl_agent(
        name,
        nixl_agent_config(
            # Every timed payload and publication is a direct mapped GPU
            # access. Host-side setup calls make synchronous progress as
            # needed, so a zero-delay UCX progress thread would only busy-poll
            # a CPU throughout the persistent kernel.
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

    with ExitStack() as resources:
        registration = agent.register_memory([arena], backends=["UCX"])
        resources.callback(agent.deregister_memory, registration, backends=["UCX"])
        coordinates = PeerCoordinates(name, (_region(arena),))
        metadata = control.exchange("agent-metadata", agent.get_agent_metadata())
        serialized_coordinates = control.exchange(
            "arena-coordinates", coordinates.to_bytes()
        )
        all_coordinates = _deserialize_rank_coordinates(
            serialized_coordinates, case.max_ranks
        )
        if any(len(peer.regions) != 1 for peer in all_coordinates):
            raise RuntimeError("each rank must publish exactly one registered arena")
        participant_names = {
            peer_rank: f"cute_elastic_ll_{run_id}_{peer_rank}"
            for peer_rank in range(case.max_ranks)
        }
        peer_names: list[str] = []
        for peer_rank, peer in enumerate(all_coordinates):
            expected_name = participant_names[peer_rank]
            if peer.agent_name != expected_name:
                raise RuntimeError(
                    f"stable rank {peer_rank} published unexpected agent "
                    f"{peer.agent_name!r}"
                )
            if peer_rank == rank:
                continue
            loaded_name = normalize_agent_name(
                agent.add_remote_agent(metadata[peer_rank])
            )
            if loaded_name != expected_name:
                raise RuntimeError(f"loaded unexpected NIXL agent {loaded_name!r}")
            peer_names.append(expected_name)
            resources.callback(agent.remove_remote_agent, expected_name)

        control.barrier("metadata-loaded")
        for peer_name in peer_names:
            agent.make_connection(peer_name, backends=["UCX"])
        complete_ucx_setup_handshake(
            agent,
            control,
            participant_names,
            generation=0,
            nonce=run_id,
            timeout_s=timeout_s,
        )

        compiled_preflight, compiled_main, occupancy = _compile_kernels(
            case,
            rank,
            (
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
            ),
            stream,
            codegen_dump_dir,
        )
        control.barrier("compiled")

        phase_specs = build_phase_specs(case)
        environment = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "device": device,
            "native_peer_atomics": native_atomic_preflight,
            "nixl_cute_abi": nixl_cute.NIXL_CUTE_ABI_VERSION,
            "cooperative_occupancy": occupancy.as_dict(),
            "codegen_dump_dir": str(codegen_dump_dir),
        }
        for phase in phase_specs:
            with _DeviceViewExitStack() as phase_views:
                mapped_preflight: dict[str, object] | None = None
                preflight_error: Exception | None = None
                remote_view = None
                try:
                    # The prior phase was synchronized and exchanged before its
                    # view was released. Copy only immutable dispatch templates
                    # and clear small publication/control regions. Receive
                    # payloads are generation-gated, so copying/zeroing the full
                    # arena would waste almost a GiB at R32/E8/T128/H7168.
                    stage_region = layout.region("dispatch_stage")
                    host_inputs.prepare(
                        case,
                        phase,
                        rank,
                        [peer.regions[0].address for peer in all_coordinates],
                    )
                    with torch.cuda.stream(stream):
                        arena[stage_region.offset : stage_region.end].copy_(
                            host_inputs.dispatch_stage, non_blocking=True
                        )
                        for region_name in _PHASE_CONTROL_REGIONS:
                            region = layout.region(region_name)
                            arena[region.offset : region.end].zero_()
                        rank_mask.copy_(host_inputs.rank_mask, non_blocking=True)
                        advertised_bases.copy_(
                            host_inputs.advertised_bases, non_blocking=True
                        )
                        incarnations.copy_(host_inputs.incarnations, non_blocking=True)
                        generation_state.copy_(
                            host_inputs.generation_state, non_blocking=True
                        )
                        bucket_counts.copy_(
                            host_inputs.bucket_counts, non_blocking=True
                        )
                        bucket_offsets.copy_(
                            host_inputs.bucket_offsets, non_blocking=True
                        )
                        worker_plan.copy_(host_inputs.worker_plan, non_blocking=True)
                        route_gates.copy_(host_inputs.route_gates, non_blocking=True)
                        outputs.fill_(OUTPUT_POISON)
                        if case.timing_mode != "none":
                            timestamps.zero_()

                    # Staging and preflight share this stream. The one
                    # post-preflight drain covers both without an extra CPU wait.
                    composite = _phase_remote_coordinates(
                        case, phase, rank, all_coordinates
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
                        from_dlpack(rank_mask).mark_layout_dynamic(),
                        from_dlpack(advertised_bases).mark_layout_dynamic(),
                        from_dlpack(statuses).mark_layout_dynamic(),
                        cuda.CUstream(stream.cuda_stream),
                    )
                    host_results.enqueue_preflight(statuses, stream)
                except Exception as error:
                    preflight_error = error
                # A failed staging or launch call may still have enqueued work.
                # Preserve the single normal CPU drain while making it an
                # unconditional prerequisite for graceful candidate rollback.
                # CUDA stream synchronization waits for queued work even when
                # it reports an error from an earlier asynchronous operation.
                try:
                    stream.synchronize()
                except Exception as drain_error:
                    if preflight_error is None:
                        preflight_error = drain_error
                    else:
                        preflight_error = RuntimeError(
                            f"{preflight_error}; candidate CUDA stream drain also "
                            f"failed: {drain_error}"
                        )
                if preflight_error is None:
                    try:
                        mapped_preflight = _classify_mapped_preflight(
                            host_results.statuses,
                            phase,
                            rank,
                            case.allow_unverified_mapped,
                        )
                    except Exception as error:
                        preflight_error = error
                # Treat an impossible successful setup as a normal candidate
                # failure before rendezvous. Every rank then releases its view
                # through the same converged rejection path.
                if preflight_error is None:
                    if mapped_preflight is None or remote_view is None:
                        preflight_error = AssertionError(
                            "successful preflight has no classification or remote view"
                        )
                try:
                    preflight_statuses = control.exchange(
                        f"phase-{phase.generation}-preflight-status",
                        _phase_status_payload(rank, preflight_error),
                    )
                except Exception as exchange_error:
                    local_release_error = phase_views.close()
                    _fail_stop_registered_owner(
                        f"generation {phase.generation} candidate outcome "
                        f"exchange failed: {exchange_error}; local candidate "
                        f"release: {local_release_error}"
                    )
                    raise AssertionError("registered-owner fail-stop returned")
                try:
                    preflight_failures = _decode_phase_failures(
                        preflight_statuses, tuple(range(case.max_ranks))
                    )
                except Exception as decode_error:
                    local_release_error = phase_views.close()
                    _fail_stop_registered_owner(
                        f"generation {phase.generation} candidate outcome was "
                        f"malformed: {decode_error}; local candidate release: "
                        f"{local_release_error}"
                    )
                    raise AssertionError("registered-owner fail-stop returned")
                if preflight_failures:
                    # Successful candidates may hold mappings into a rejecting
                    # rank. Release every locally-created view, then converge
                    # before any outer registration/agent teardown can begin.
                    release_error = phase_views.close()
                    try:
                        release_statuses = control.exchange(
                            f"phase-{phase.generation}-candidate-views-released",
                            _phase_status_payload(rank, release_error),
                        )
                    except Exception as exchange_error:
                        _fail_stop_registered_owner(
                            f"generation {phase.generation} candidate release "
                            f"exchange failed: {exchange_error}"
                        )
                        raise AssertionError("registered-owner fail-stop returned")
                    try:
                        release_failures = _decode_phase_failures(
                            release_statuses, tuple(range(case.max_ranks))
                        )
                    except Exception as decode_error:
                        _fail_stop_registered_owner(
                            f"generation {phase.generation} candidate release "
                            f"outcome was malformed after local close/retry: "
                            f"{decode_error}; local candidate release: "
                            f"{release_error}"
                        )
                        raise AssertionError("registered-owner fail-stop returned")
                    if _contains_unreleased_view(release_failures):
                        _fail_stop_registered_owner(
                            f"generation {phase.generation} retained an unreleased "
                            f"candidate view: {release_failures}"
                        )
                    raise RuntimeError(
                        f"generation {phase.generation} candidate staging or mapped "
                        f"preflight failed across ranks: {preflight_failures}; "
                        f"release failures: {release_failures}"
                    ) from preflight_error
                # The status exchange above already rendezvoused every rank.
                # This reset and the main launch are ordered on the same stream;
                # peer protocol waits tolerate launch skew, so another host/file
                # barrier would add transition latency without an ordering edge.
                with torch.cuda.stream(stream):
                    statuses.zero_()

                kernel_error: Exception | None = None
                try:
                    compiled_main(
                        remote_view,
                        from_dlpack(arena).mark_layout_dynamic(),
                        from_dlpack(rank_mask).mark_layout_dynamic(),
                        from_dlpack(incarnations).mark_layout_dynamic(),
                        from_dlpack(generation_state).mark_layout_dynamic(),
                        from_dlpack(bucket_counts).mark_layout_dynamic(),
                        from_dlpack(bucket_offsets).mark_layout_dynamic(),
                        from_dlpack(worker_plan).mark_layout_dynamic(),
                        from_dlpack(route_gates).mark_layout_dynamic(),
                        from_dlpack(outputs).mark_layout_dynamic(),
                        from_dlpack(statuses).mark_layout_dynamic(),
                        from_dlpack(timestamps).mark_layout_dynamic(),
                        cuda.CUstream(stream.cuda_stream),
                    )
                    host_results.enqueue_main(
                        statuses,
                        outputs,
                        timestamps,
                        stream,
                        copy_timestamps=case.timing_mode != "none",
                    )
                except Exception as error:
                    kernel_error = error
                try:
                    stream.synchronize()
                except Exception as error:
                    # Synchronize drains prior stream work even when reporting a
                    # previous asynchronous launch error. A launch that cannot
                    # make device progress remains bounded only by the external
                    # process or scheduler timeout.
                    if kernel_error is None:
                        kernel_error = error
            # ExitStack closes this rank's view immediately after its stream
            # drain. The validation exchange below is the single all-rank
            # convergence proving every view is gone before a failure can unwind
            # the outer registrations. CPU result work remains example overhead.

            # A peer can pass its terminal device wait before another rank
            # detects a final-operation error. Exchange one host envelope only
            # after every old view is gone, so failures converge without risking
            # owner teardown while a successful peer still holds a mapping.
            validation_error = phase_views.release_error or kernel_error
            rank_result: dict[str, object] | None = None
            if validation_error is None:
                try:
                    _validate_statuses(host_results.statuses, phase.active_ranks)
                    _validate_outputs(case, phase, rank, host_results.outputs)
                    rank_result = _rank_phase_result(
                        case, phase, rank, host_results.timestamps
                    )
                    rank_result["environment"] = {
                        **environment,
                        "mapped_preflight": mapped_preflight,
                    }
                except Exception as error:
                    validation_error = error
            try:
                gathered = control.exchange(
                    f"phase-{phase.generation}-result",
                    _phase_result_payload(rank, validation_error, rank_result),
                )
            except Exception as exchange_error:
                _fail_stop_registered_owner(
                    f"generation {phase.generation} post-release result exchange "
                    f"failed: {exchange_error}"
                )
                raise AssertionError("registered-owner fail-stop returned")
            try:
                validation_failures, decoded_by_rank = _decode_phase_result_exchange(
                    gathered, tuple(range(case.max_ranks))
                )
            except Exception as decode_error:
                # The context has already closed and retried the local view. A
                # malformed peer envelope still prevents proving global safety,
                # so retain the registered owner until external termination.
                local_release_error = phase_views.close()
                _fail_stop_registered_owner(
                    f"generation {phase.generation} post-release result was "
                    f"malformed: {decode_error}; local view release: "
                    f"{local_release_error}"
                )
                raise AssertionError("registered-owner fail-stop returned")
            if validation_failures:
                if _contains_unreleased_view(validation_failures):
                    _fail_stop_registered_owner(
                        f"generation {phase.generation} retained an unreleased "
                        f"view: {validation_failures}"
                    )
                raise RuntimeError(
                    f"generation {phase.generation} failed after the all-rank "
                    f"kernel drain: {validation_failures}"
                ) from validation_error
            if rank == 0:
                decoded = tuple(
                    decoded_by_rank[peer_rank] for peer_rank in range(case.max_ranks)
                )
                result = summarize_phase_results(case, phase, decoded)
                result["arena"] = {
                    "bytes_per_rank": layout.arena_nbytes,
                    "layout": "CompactLLArenaLayout",
                    "dispatch_record_stride": layout.dispatch_record_stride,
                    "combine_record_stride": layout.combine_record_stride,
                    "fixed_rank_capacity": case.max_ranks,
                    "dispatch_stage_records": case.route_capacity,
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
                        layout.region(name).nbytes for name in _PHASE_CONTROL_REGIONS
                    ),
                    "receive_payload_reset": "generation-gated; not copied or zeroed",
                    "dispatch_receive_slots_per_source_expert": case.num_tokens,
                    "combine_receive_records": case.route_capacity,
                }
                result["backend_parameters"] = agent.get_backend_params("UCX")
                print(
                    "NIXL_CUTE_ELASTIC_MOE_LL_RESULT "
                    + json.dumps(result, sort_keys=True),
                    flush=True,
                )
    control.barrier("cleanup-complete")


def run(
    *,
    devices: tuple[int, ...],
    case: ElasticLLCase,
    timeout_s: float,
    codegen_dump_root: str | None = None,
) -> None:
    """Spawn one persistent MoE rank for every fixed-capacity device slot."""

    if len(devices) != case.max_ranks or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain one distinct device per stable rank")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a positive finite number")
    _require_diagnostic_timeout_margin(case, float(timeout_s))
    if not torch.cuda.is_available() or torch.cuda.device_count() < case.max_ranks:
        raise RuntimeError(f"this run requires {case.max_ranks} visible CUDA devices")
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device indices must be integers")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    case = _resolve_case_geometry(case, devices)
    import torch.multiprocessing as mp

    with TemporaryDirectory(prefix=f"nixl_cute_elastic_ll_{os.getpid()}_") as directory:
        if codegen_dump_root is None:
            codegen_root = Path(directory) / "codegen"
            codegen_root.mkdir(mode=0o700)
        else:
            codegen_root = Path(codegen_dump_root).resolve(strict=True)
            if not codegen_root.is_dir():
                raise ValueError("codegen dump root must be an existing directory")
        for rank in range(case.max_ranks):
            rank_dump = codegen_root / f"rank-{rank}"
            if rank_dump.exists() or rank_dump.is_symlink():
                raise RuntimeError(
                    f"refusing to reuse CuTe codegen directory: {rank_dump}"
                )
        mp.spawn(
            _worker,
            args=(devices, directory, case, float(timeout_s), str(codegen_root)),
            nprocs=case.max_ranks,
            join=True,
        )


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
        help=(
            "logical worker warps packed into each cooperative CTA (0 selects "
            "an exact-divisor default; sweep explicitly when tuning)"
        ),
    )
    parser.add_argument(
        "--num-banks",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "communication banks (default 1; 2 is retained for architecture "
            "A/B qualification)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--membership",
        default="0,1;0;0,1",
        help="semicolon-delimited sparse phases; default exercises shrink/rejoin",
    )
    parser.add_argument(
        "--no-empty-last-expert",
        action="store_true",
        help="send records to the last expert instead of testing empty publication",
    )
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
            "compile-time timing policy: none is the uninstrumented production "
            "path; envelope emits two phase markers; cadence emits round starts; "
            "peer is the perturbing diagnostic mode"
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
            "qualification mode: use operation-varying immutable source copies "
            "and preserve every measured output; disabled in performance runs"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="control-plane timeout; also use an external scheduler timeout",
    )
    parser.add_argument(
        "--device-timeout-ms",
        type=int,
        default=0,
        help=(
            "opt-in diagnostic per-wait GPU polling bound; zero (default) uses "
            "the timer-free production path; neither mode recovers abrupt mapped-"
            "owner loss"
        ),
    )
    parser.add_argument(
        "--codegen-dump-root",
        help=(
            "optional existing parent for fresh rank-private CuTe artifacts; "
            "without it, compile artifacts stay in the run's private temporary "
            "directory"
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
        target_sm=(_detect_common_sm(devices) if args.warps_per_cta == 0 else None),
    )
    run(
        devices=devices,
        case=case,
        timeout_s=args.timeout,
        codegen_dump_root=args.codegen_dump_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ELEMENT_NBYTES",
    "ElasticLLCase",
    "PhaseRouteState",
    "PhaseSpec",
    "RouteAssignment",
    "bank_cycle_schedule",
    "build_phase_route_state",
    "build_phase_specs",
    "build_routes",
    "cumulative_publication_value",
    "expected_outputs",
    "main",
    "make_phase_arena",
    "make_phase_dispatch_stage",
    "parse_membership_plan",
    "phase_incarnation_tables",
    "run",
    "summarize_phase_results",
]
