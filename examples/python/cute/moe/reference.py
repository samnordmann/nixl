# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden model for fixed-capacity, elastic NIXL MoE communication.

This module intentionally imports neither NIXL nor CuTe.  It specifies the
wire metadata, stable expert numbering, packing order, receive layout, expert
calculation, and weighted combine that the GPU examples must reproduce.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from .arena import (
    DISPATCH_HEADER_NBYTES,
    UINT32_MAX,
    UINT64_MAX,
    PeerSlabLayout,
    RecordLocation,
)

_DISPATCH_HEADER = struct.Struct("<QIIII")
assert _DISPATCH_HEADER.size == DISPATCH_HEADER_NBYTES


def _uint(name: str, value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _positive(name: str, value: int, maximum: int) -> int:
    _uint(name, value, maximum)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _cpu_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {tensor.device}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True)
class ElasticExpertTopology:
    """Fixed rank/expert capacity with an elastic active-rank mask.

    Expert IDs are permanent: ``rank * experts_per_rank + local_expert``.
    Removing a rank leaves holes in the expert namespace, matching NIXL EP's
    rank mask and sparse ``max(active_ranks) + 1`` rank bound.
    """

    max_ranks: int
    experts_per_rank: int
    active_ranks: tuple[int, ...]
    generation: int = 0

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, UINT32_MAX + 1)
        _positive("experts_per_rank", self.experts_per_rank, UINT32_MAX)
        _uint("generation", self.generation, UINT64_MAX)
        if self.max_ranks * self.experts_per_rank > UINT32_MAX + 1:
            raise OverflowError("maximum expert ID does not fit the u32 wire field")

        ranks = tuple(self.active_ranks)
        if not ranks:
            raise ValueError("at least one rank must be active")
        for rank in ranks:
            _uint("active rank", rank, UINT32_MAX)
            if rank >= self.max_ranks:
                raise ValueError(
                    f"active rank {rank} is outside max_ranks={self.max_ranks}"
                )
        if len(set(ranks)) != len(ranks):
            raise ValueError("active_ranks contains duplicates")
        object.__setattr__(self, "active_ranks", tuple(sorted(ranks)))

    @classmethod
    def from_mask(
        cls,
        *,
        max_ranks: int,
        experts_per_rank: int,
        active_mask: Sequence[bool],
        generation: int = 0,
    ) -> "ElasticExpertTopology":
        _positive("max_ranks", max_ranks, UINT32_MAX + 1)
        _positive("experts_per_rank", experts_per_rank, UINT32_MAX)
        _uint("generation", generation, UINT64_MAX)
        if len(active_mask) != max_ranks:
            raise ValueError(
                f"active_mask has length {len(active_mask)}, expected {max_ranks}"
            )
        if any(type(value) is not bool for value in active_mask):
            raise TypeError("active_mask values must be bool")
        return cls(
            max_ranks=max_ranks,
            experts_per_rank=experts_per_rank,
            active_ranks=tuple(i for i, active in enumerate(active_mask) if active),
            generation=generation,
        )

    @property
    def active_mask(self) -> tuple[bool, ...]:
        active = set(self.active_ranks)
        return tuple(rank in active for rank in range(self.max_ranks))

    @property
    def nixl_mask(self) -> tuple[int, ...]:
        """Return NIXL EP mask encoding: zero active, nonzero inactive."""

        return tuple(0 if active else 1 for active in self.active_mask)

    @property
    def rank_bound(self) -> int:
        return self.active_ranks[-1] + 1

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    @property
    def active_experts(self) -> tuple[int, ...]:
        return tuple(
            self.expert_id(rank, local_expert)
            for rank in self.active_ranks
            for local_expert in range(self.experts_per_rank)
        )

    def is_active(self, rank: int) -> bool:
        _uint("rank", rank, UINT32_MAX)
        if rank >= self.max_ranks:
            raise ValueError(f"rank {rank} is outside max_ranks={self.max_ranks}")
        return self.active_mask[rank]

    def expert_id(self, rank: int, local_expert: int) -> int:
        _uint("rank", rank, UINT32_MAX)
        _uint("local_expert", local_expert, UINT32_MAX)
        if rank >= self.max_ranks:
            raise ValueError(f"rank {rank} is outside max_ranks={self.max_ranks}")
        if local_expert >= self.experts_per_rank:
            raise ValueError(
                f"local_expert {local_expert} is outside "
                f"experts_per_rank={self.experts_per_rank}"
            )
        return rank * self.experts_per_rank + local_expert

    def owner(self, expert: int) -> int:
        _uint("expert", expert, UINT32_MAX)
        if expert >= self.max_experts:
            raise ValueError(
                f"expert {expert} is outside max_experts={self.max_experts}"
            )
        return expert // self.experts_per_rank

    def local_expert(self, expert: int) -> int:
        self.owner(expert)
        return expert % self.experts_per_rank

    def reconfigure(
        self, active_ranks: Iterable[int], *, generation: int | None = None
    ) -> "ElasticExpertTopology":
        """Return a new generation without changing capacity or expert IDs."""

        next_generation = self.generation + 1 if generation is None else generation
        _uint("generation", next_generation, UINT64_MAX)
        if next_generation <= self.generation:
            raise ValueError(
                f"generation must advance beyond {self.generation}, got {next_generation}"
            )
        return ElasticExpertTopology(
            max_ranks=self.max_ranks,
            experts_per_rank=self.experts_per_rank,
            active_ranks=tuple(active_ranks),
            generation=next_generation,
        )


@dataclass(frozen=True)
class RoutingPlan:
    generation: int
    origin_rank: int
    expert_indices: torch.Tensor
    weights: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.expert_indices.shape[0]

    @property
    def top_k(self) -> int:
        return self.expert_indices.shape[1]


def route_topk(
    scores: torch.Tensor,
    topology: ElasticExpertTopology,
    *,
    origin_rank: int,
    top_k: int,
    normalize_weights: bool = True,
) -> RoutingPlan:
    """Route to unique active experts with deterministic expert-ID tie breaks."""

    _cpu_tensor("scores", scores, 2)
    if not scores.dtype.is_floating_point:
        raise TypeError(f"scores must have floating dtype, got {scores.dtype}")
    if scores.shape[1] != topology.max_experts:
        raise ValueError(
            f"scores has {scores.shape[1]} experts, expected fixed capacity "
            f"{topology.max_experts}"
        )
    if not torch.isfinite(scores).all().item():
        raise ValueError("scores must contain only finite values")
    _uint("origin_rank", origin_rank, UINT32_MAX)
    if not topology.is_active(origin_rank):
        raise ValueError(f"origin_rank {origin_rank} is inactive")
    _positive("top_k", top_k, UINT32_MAX)
    active_experts = topology.active_experts
    if top_k > len(active_experts):
        raise ValueError(f"top_k={top_k} exceeds {len(active_experts)} active experts")

    indices = torch.empty((scores.shape[0], top_k), dtype=torch.int64)
    for token in range(scores.shape[0]):
        # Python's sort is stable.  Since active_experts is in ascending ID
        # order, equal scores are resolved by the smaller permanent expert ID.
        ranked = sorted(
            active_experts,
            key=lambda expert: (-float(scores[token, expert].item()), expert),
        )
        indices[token] = torch.tensor(ranked[:top_k], dtype=torch.int64)

    selected = scores.to(torch.float64).gather(1, indices)
    weights = (torch.softmax(selected, dim=1) if normalize_weights else selected).to(
        torch.float32
    )
    return RoutingPlan(topology.generation, origin_rank, indices, weights)


@dataclass(frozen=True)
class DispatchMetadata:
    generation: int
    origin_rank: int
    origin_token: int
    route_slot: int
    expert: int

    def __post_init__(self) -> None:
        _uint("generation", self.generation, UINT64_MAX)
        _uint("origin_rank", self.origin_rank, UINT32_MAX)
        _uint("origin_token", self.origin_token, UINT32_MAX)
        _uint("route_slot", self.route_slot, UINT32_MAX)
        _uint("expert", self.expert, UINT32_MAX)

    def to_bytes(self, header_size: int = DISPATCH_HEADER_NBYTES) -> bytes:
        _positive("header_size", header_size, UINT32_MAX)
        if header_size < DISPATCH_HEADER_NBYTES:
            raise ValueError(
                f"header_size must be at least {DISPATCH_HEADER_NBYTES} bytes"
            )
        encoded = _DISPATCH_HEADER.pack(
            self.generation,
            self.origin_rank,
            self.origin_token,
            self.route_slot,
            self.expert,
        )
        return encoded + bytes(header_size - len(encoded))

    @classmethod
    def from_bytes(cls, data: bytes, *, require_zero_padding: bool = True):
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) < DISPATCH_HEADER_NBYTES:
            raise ValueError(
                f"header has {len(data)} bytes, expected at least "
                f"{DISPATCH_HEADER_NBYTES}"
            )
        if require_zero_padding and any(data[DISPATCH_HEADER_NBYTES:]):
            raise ValueError("dispatch header padding must be zero")
        return cls(*_DISPATCH_HEADER.unpack_from(data))


@dataclass(frozen=True)
class DispatchRecord:
    destination_rank: int
    slot: int
    location: RecordLocation
    metadata: DispatchMetadata
    payload: torch.Tensor


@dataclass(frozen=True)
class ExpertBatch:
    expert: int
    payloads: torch.Tensor
    metadata: tuple[DispatchMetadata, ...]
    # One (begin, count) entry per fixed-capacity rank, including mask holes.
    origin_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ExpertOutput:
    metadata: DispatchMetadata
    payload: torch.Tensor


def _validate_routing(routing: RoutingPlan, topology: ElasticExpertTopology) -> None:
    _cpu_tensor("routing.expert_indices", routing.expert_indices, 2)
    _cpu_tensor("routing.weights", routing.weights, 2)
    if routing.expert_indices.dtype != torch.int64:
        raise TypeError("routing.expert_indices must have dtype torch.int64")
    if not routing.weights.dtype.is_floating_point:
        raise TypeError("routing.weights must have floating dtype")
    if routing.expert_indices.shape != routing.weights.shape:
        raise ValueError("routing indices and weights must have equal shapes")
    if routing.generation != topology.generation:
        raise ValueError(
            f"stale routing generation {routing.generation}; current generation is "
            f"{topology.generation}"
        )
    if not topology.is_active(routing.origin_rank):
        raise ValueError(f"routing origin rank {routing.origin_rank} is inactive")
    if not torch.isfinite(routing.weights).all().item():
        raise ValueError("routing weights must contain only finite values")
    active_experts = set(topology.active_experts)
    for token, row in enumerate(routing.expert_indices.tolist()):
        if len(set(row)) != len(row):
            raise ValueError(f"routing token {token} contains duplicate experts")
        invalid = [expert for expert in row if expert not in active_experts]
        if invalid:
            raise ValueError(
                f"routing token {token} targets inactive or invalid experts {invalid}"
            )


def _validate_layout(
    layout: PeerSlabLayout,
    topology: ElasticExpertTopology,
    routing: RoutingPlan | None = None,
) -> None:
    if layout.max_ranks != topology.max_ranks:
        raise ValueError(
            f"layout max_ranks={layout.max_ranks} does not match topology "
            f"max_ranks={topology.max_ranks}"
        )
    if routing is not None and layout.top_k != routing.top_k:
        raise ValueError(
            f"layout top_k={layout.top_k} does not match routing top_k={routing.top_k}"
        )


def _validate_payload(
    name: str, payload: torch.Tensor, layout: PeerSlabLayout, *, vector: bool
) -> None:
    _cpu_tensor(name, payload, 1 if vector else 2)
    if not payload.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not payload.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype, got {payload.dtype}")
    if not torch.isfinite(payload).all().item():
        raise ValueError(f"{name} must contain only finite values")
    hidden = payload.shape[0] if vector else payload.shape[1]
    if hidden != layout.hidden_size:
        raise ValueError(
            f"{name} hidden size {hidden} does not match layout {layout.hidden_size}"
        )
    if payload.element_size() != layout.element_size:
        raise ValueError(
            f"{name} element size {payload.element_size()} does not match layout "
            f"{layout.element_size}"
        )


def pack_dispatch(
    tokens: torch.Tensor,
    routing: RoutingPlan,
    topology: ElasticExpertTopology,
    layout: PeerSlabLayout,
) -> tuple[DispatchRecord, ...]:
    """Pack token-major routes into per-destination, per-origin slab slots."""

    _validate_routing(routing, topology)
    _validate_layout(layout, topology, routing)
    _validate_payload("tokens", tokens, layout, vector=False)
    if tokens.shape[0] != routing.num_tokens:
        raise ValueError(
            f"tokens has {tokens.shape[0]} rows, routing has {routing.num_tokens}"
        )
    if tokens.shape[0] > layout.max_tokens_per_rank:
        raise ValueError(
            f"token count {tokens.shape[0]} exceeds max_tokens_per_rank="
            f"{layout.max_tokens_per_rank}"
        )
    if tokens.shape[0] > UINT32_MAX:
        raise OverflowError("origin token index does not fit the u32 wire field")

    next_slot = [0] * topology.max_ranks
    records: list[DispatchRecord] = []
    for origin_token in range(routing.num_tokens):
        for route_slot in range(routing.top_k):
            expert = int(routing.expert_indices[origin_token, route_slot].item())
            destination = topology.owner(expert)
            slot = next_slot[destination]
            if slot >= layout.record_capacity:
                raise OverflowError(
                    f"destination rank {destination} dispatch exceeds slab capacity "
                    f"{layout.record_capacity}"
                )
            next_slot[destination] += 1
            records.append(
                DispatchRecord(
                    destination_rank=destination,
                    slot=slot,
                    location=layout.record(routing.origin_rank, slot),
                    metadata=DispatchMetadata(
                        generation=topology.generation,
                        origin_rank=routing.origin_rank,
                        origin_token=origin_token,
                        route_slot=route_slot,
                        expert=expert,
                    ),
                    payload=tokens[origin_token].detach().clone().contiguous(),
                )
            )
    return tuple(records)


def unpack_dispatch(
    records: Sequence[DispatchRecord],
    topology: ElasticExpertTopology,
    layout: PeerSlabLayout,
    *,
    destination_rank: int,
) -> tuple[ExpertBatch, ...]:
    """Validate and group incoming records like NIXL EP's packed receive view."""

    _validate_layout(layout, topology)
    if not topology.is_active(destination_rank):
        raise ValueError(f"destination_rank {destination_rank} is inactive")

    by_expert: dict[int, list[DispatchRecord]] = {
        topology.expert_id(destination_rank, local): []
        for local in range(topology.experts_per_rank)
    }
    route_keys: set[tuple[int, int, int]] = set()
    slots_by_origin: dict[int, set[int]] = {}
    payload_dtype: torch.dtype | None = None

    for record in records:
        if not isinstance(record, DispatchRecord):
            raise TypeError("records must contain DispatchRecord instances")
        metadata = record.metadata
        if record.destination_rank != destination_rank:
            raise ValueError(
                f"record destination {record.destination_rank} does not match "
                f"receiver {destination_rank}"
            )
        if metadata.generation != topology.generation:
            raise ValueError(
                f"stale dispatch generation {metadata.generation}; current generation "
                f"is {topology.generation}"
            )
        if not topology.is_active(metadata.origin_rank):
            raise ValueError(f"record origin rank {metadata.origin_rank} is inactive")
        if metadata.origin_token >= layout.max_tokens_per_rank:
            raise ValueError(
                f"origin token {metadata.origin_token} exceeds layout capacity"
            )
        if metadata.route_slot >= layout.top_k:
            raise ValueError(f"route slot {metadata.route_slot} exceeds layout top_k")
        if topology.owner(metadata.expert) != destination_rank:
            raise ValueError(
                f"expert {metadata.expert} is not owned by rank {destination_rank}"
            )
        if record.slot >= layout.record_capacity:
            raise ValueError(f"record slot {record.slot} exceeds slab capacity")
        expected_location = layout.record(metadata.origin_rank, record.slot)
        if record.location != expected_location:
            raise ValueError("record byte location does not match origin slab and slot")
        _validate_payload("record payload", record.payload, layout, vector=True)
        if payload_dtype is None:
            payload_dtype = record.payload.dtype
        elif record.payload.dtype != payload_dtype:
            raise TypeError("all dispatch payloads must have the same dtype")

        route_key = (
            metadata.origin_rank,
            metadata.origin_token,
            metadata.route_slot,
        )
        if route_key in route_keys:
            raise ValueError(f"duplicate dispatch route {route_key}")
        route_keys.add(route_key)
        origin_slots = slots_by_origin.setdefault(metadata.origin_rank, set())
        if record.slot in origin_slots:
            raise ValueError(
                f"duplicate slab slot {record.slot} from rank {metadata.origin_rank}"
            )
        origin_slots.add(record.slot)
        by_expert[metadata.expert].append(record)

    for origin, slots in slots_by_origin.items():
        expected = set(range(len(slots)))
        if slots != expected:
            raise ValueError(
                f"origin rank {origin} slab slots must be dense from zero; got "
                f"{sorted(slots)}"
            )

    dtype = torch.float32 if payload_dtype is None else payload_dtype
    batches: list[ExpertBatch] = []
    for expert in sorted(by_expert):
        ordered = sorted(
            by_expert[expert],
            key=lambda record: (
                record.metadata.origin_rank,
                record.metadata.origin_token,
                record.metadata.route_slot,
            ),
        )
        payloads = (
            torch.stack([record.payload for record in ordered])
            if ordered
            else torch.empty((0, layout.hidden_size), dtype=dtype)
        )
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for origin in range(topology.max_ranks):
            count = sum(record.metadata.origin_rank == origin for record in ordered)
            ranges.append((cursor, count))
            cursor += count
        batches.append(
            ExpertBatch(
                expert=expert,
                payloads=payloads,
                metadata=tuple(record.metadata for record in ordered),
                origin_ranges=tuple(ranges),
            )
        )
    return tuple(batches)


def expert_transform(payloads: torch.Tensor, expert: int) -> torch.Tensor:
    """Deterministic stand-in for an expert MLP used by correctness tests."""

    _cpu_tensor("payloads", payloads, 2)
    if not payloads.dtype.is_floating_point:
        raise TypeError("expert payloads must have floating dtype")
    _uint("expert", expert, UINT32_MAX)
    # An affine transform keeps the golden model cheap while making both a
    # wrong expert and a dropped route visible in the combined output.
    scale = float(expert + 1)
    bias = float((expert % 17) - 8) / 32.0
    return payloads * scale + bias


def process_expert_batches(
    batches: Sequence[ExpertBatch],
) -> tuple[ExpertOutput, ...]:
    outputs: list[ExpertOutput] = []
    seen_experts: set[int] = set()
    for batch in batches:
        if batch.expert in seen_experts:
            raise ValueError(f"duplicate expert batch {batch.expert}")
        seen_experts.add(batch.expert)
        if batch.payloads.shape[0] != len(batch.metadata):
            raise ValueError("expert batch payload and metadata counts differ")
        transformed = expert_transform(batch.payloads, batch.expert)
        for row, metadata in zip(transformed, batch.metadata):
            if metadata.expert != batch.expert:
                raise ValueError("expert batch contains metadata for another expert")
            outputs.append(ExpertOutput(metadata, row.contiguous()))
    return tuple(outputs)


def weighted_combine(
    outputs: Sequence[ExpertOutput],
    routing: RoutingPlan,
    topology: ElasticExpertTopology,
    *,
    hidden_size: int,
    require_complete: bool = True,
) -> torch.Tensor:
    """Return outputs to origin-token order and apply route weights."""

    _validate_routing(routing, topology)
    _positive("hidden_size", hidden_size, UINT32_MAX)
    by_route: dict[tuple[int, int], torch.Tensor] = {}
    dtype: torch.dtype | None = None
    for output in outputs:
        if not isinstance(output, ExpertOutput):
            raise TypeError("outputs must contain ExpertOutput instances")
        metadata = output.metadata
        if metadata.generation != routing.generation:
            raise ValueError("expert output has a stale generation")
        if metadata.origin_rank != routing.origin_rank:
            raise ValueError("expert output belongs to a different origin rank")
        if metadata.origin_token >= routing.num_tokens:
            raise ValueError("expert output has an invalid origin token")
        if metadata.route_slot >= routing.top_k:
            raise ValueError("expert output has an invalid route slot")
        expected_expert = int(
            routing.expert_indices[metadata.origin_token, metadata.route_slot].item()
        )
        if metadata.expert != expected_expert:
            raise ValueError(
                f"expert output claims expert {metadata.expert}, expected "
                f"{expected_expert}"
            )
        _cpu_tensor("expert output payload", output.payload, 1)
        if output.payload.shape[0] != hidden_size:
            raise ValueError("expert output hidden size does not match combine output")
        if not output.payload.dtype.is_floating_point:
            raise TypeError("expert output payload must have floating dtype")
        if dtype is None:
            dtype = output.payload.dtype
        elif output.payload.dtype != dtype:
            raise TypeError("all expert outputs must have the same dtype")
        key = (metadata.origin_token, metadata.route_slot)
        if key in by_route:
            raise ValueError(f"duplicate expert output route {key}")
        by_route[key] = output.payload

    expected_routes = routing.num_tokens * routing.top_k
    if len(by_route) > expected_routes:
        raise ValueError("more expert outputs than routing entries")
    if require_complete and len(by_route) != expected_routes:
        missing: list[tuple[int, int]] = []
        for token in range(routing.num_tokens):
            for slot in range(routing.top_k):
                if (token, slot) not in by_route:
                    missing.append((token, slot))
                    if len(missing) == 8:
                        break
            if len(missing) == 8:
                break
        raise ValueError(f"missing expert outputs for routes {missing}")

    result_dtype = torch.float32 if dtype is None else dtype
    combined = torch.zeros((routing.num_tokens, hidden_size), dtype=result_dtype)
    # Iterate in canonical order so input message arrival order cannot alter
    # floating-point accumulation order.
    for token in range(routing.num_tokens):
        for slot in range(routing.top_k):
            payload = by_route.get((token, slot))
            if payload is not None:
                combined[token] += payload * routing.weights[token, slot].to(
                    result_dtype
                )
    if not torch.isfinite(combined).all().item():
        raise ValueError("combined output contains non-finite values")
    return combined


@dataclass(frozen=True)
class DistributedReferenceResult:
    routings: Mapping[int, RoutingPlan]
    dispatch_records: tuple[DispatchRecord, ...]
    expert_batches: Mapping[int, tuple[ExpertBatch, ...]]
    expert_outputs: tuple[ExpertOutput, ...]
    combined: Mapping[int, torch.Tensor]


def distributed_moe_reference(
    tokens_by_rank: Mapping[int, torch.Tensor],
    scores_by_rank: Mapping[int, torch.Tensor],
    topology: ElasticExpertTopology,
    layout: PeerSlabLayout,
    *,
    top_k: int,
    normalize_weights: bool = True,
) -> DistributedReferenceResult:
    """Execute a complete dispatch/expert/combine golden pass on CPU."""

    token_ranks = set(tokens_by_rank)
    score_ranks = set(scores_by_rank)
    active_ranks = set(topology.active_ranks)
    if token_ranks != active_ranks or score_ranks != active_ranks:
        raise ValueError(
            "tokens_by_rank and scores_by_rank must each contain exactly the "
            f"active ranks {sorted(active_ranks)}"
        )

    routings: dict[int, RoutingPlan] = {}
    records: list[DispatchRecord] = []
    for rank in topology.active_ranks:
        routing = route_topk(
            scores_by_rank[rank],
            topology,
            origin_rank=rank,
            top_k=top_k,
            normalize_weights=normalize_weights,
        )
        routings[rank] = routing
        records.extend(pack_dispatch(tokens_by_rank[rank], routing, topology, layout))

    batches_by_rank: dict[int, tuple[ExpertBatch, ...]] = {}
    all_outputs: list[ExpertOutput] = []
    for rank in topology.active_ranks:
        incoming = [record for record in records if record.destination_rank == rank]
        batches = unpack_dispatch(incoming, topology, layout, destination_rank=rank)
        batches_by_rank[rank] = batches
        all_outputs.extend(process_expert_batches(batches))

    combined: dict[int, torch.Tensor] = {}
    for rank, routing in routings.items():
        origin_outputs = [
            output for output in all_outputs if output.metadata.origin_rank == rank
        ]
        combined[rank] = weighted_combine(
            origin_outputs,
            routing,
            topology,
            hidden_size=layout.hidden_size,
        )

    return DistributedReferenceResult(
        routings=routings,
        dispatch_records=tuple(records),
        expert_batches=batches_by_rank,
        expert_outputs=tuple(all_outputs),
        combined=combined,
    )
