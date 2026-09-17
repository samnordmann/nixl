#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference full-MoE communication with elastic NIXL CuTe device PUTs.

The example follows the elastic contract used by ``examples/device/ep``:

* capacity is allocated once for the maximum rank count;
* expert IDs remain ``rank * experts_per_rank + local_expert``;
* inactive ranks are sparse mask holes, never renumbered experts;
* new peer metadata and device views are staged before activation;
* old-generation traffic is quiesced before the mask/view swap;
* every remote PUT and completion atomic is gated by the committed GPU mask; and
* removed views are released before remote metadata is invalidated.

Every active rank performs top-k routing, dispatches BF16 token copies to the
owning expert ranks, applies a deterministic stand-in expert transform, sends
the results back, and performs weighted combine.  A fixed padded slab per
source/destination makes every remote region single-writer.  PUT and its
completion atomic use the same NIXL channel.

This is a correctness and lifecycle reference, not the optimized NIXL EP data
path.  All configured processes remain alive as standby ranks.  This reference
uses the synchronous, unbounded device wait API, so abrupt process/node failure
while a kernel targets that peer is intentionally not claimed or simulated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Sequence

import torch

try:  # Package import in tests.
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_masked_thread_put_then_signal_host,
        normalize_agent_name,
        wait_until,
    )
    from .moe import (
        DispatchMetadata,
        DispatchRecord,
        ElasticExpertTopology,
        ExpertOutput,
        PeerSlabLayout,
        expert_transform,
        pack_dispatch,
        process_expert_batches,
        route_topk,
        unpack_dispatch,
        weighted_combine,
    )
except ImportError:  # Direct ``python examples/python/cute/elastic_moe.py``.
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_masked_thread_put_then_signal_host,
        normalize_agent_name,
        wait_until,
    )
    from moe import (  # type: ignore[no-redef]
        DispatchMetadata,
        DispatchRecord,
        ElasticExpertTopology,
        ExpertOutput,
        PeerSlabLayout,
        expert_transform,
        pack_dispatch,
        process_expert_batches,
        route_topk,
        unpack_dispatch,
        weighted_combine,
    )

DEFAULT_PLAN = Path(__file__).with_name("elastic_expansion_contraction.json")
DISPATCH = 0
COMBINE = 1
_DIRECTION_NAMES = ("dispatch", "combine")

# Local view descriptors.
_LOCAL_DISPATCH_SEND = 0
_LOCAL_COMBINE_SEND = 1

# Every peer-specific remote view has this exact descriptor order.
_REMOTE_DISPATCH_RECV = 0
_REMOTE_COMBINE_RECV = 1
_REMOTE_COUNTERS = 2
_NIXL_ERR_NOT_ALLOWED = -6


def parse_membership_plan(
    document: object, *, max_ranks: int | None = None
) -> tuple[tuple[int, ...], ...]:
    """Validate the same list-of-rank-lists shape used by NIXL EP plans.

    Negative entries in the mature EP tests mean a rank killed during a phase.
    They are rejected here because blocking CuTe operations cannot bound a
    device-side wait after an abrupt peer loss. Graceful shrink is expressed by
    omitting the rank from the next phase.
    """

    if not isinstance(document, list) or not document:
        raise ValueError("membership plan must be a non-empty JSON list")
    if max_ranks is not None:
        if isinstance(max_ranks, bool) or not isinstance(max_ranks, int):
            raise TypeError("max_ranks must be an integer")
        if max_ranks <= 0:
            raise ValueError("max_ranks must be positive")

    phases: list[tuple[int, ...]] = []
    for phase_index, raw_phase in enumerate(document):
        if not isinstance(raw_phase, list) or not raw_phase:
            raise ValueError(f"phase {phase_index} must be a non-empty rank list")
        ranks: list[int] = []
        for raw_rank in raw_phase:
            if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
                raise TypeError(
                    f"phase {phase_index} rank IDs must be integers, got "
                    f"{type(raw_rank).__name__}"
                )
            if raw_rank < 0:
                raise ValueError(
                    f"phase {phase_index} contains killed-rank marker {raw_rank}; "
                    "this blocking reference supports only quiescent graceful shrink"
                )
            if max_ranks is not None and raw_rank >= max_ranks:
                raise ValueError(
                    f"phase {phase_index} rank {raw_rank} is outside fixed "
                    f"capacity {max_ranks}"
                )
            ranks.append(raw_rank)
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"phase {phase_index} contains duplicate ranks")
        phases.append(tuple(sorted(ranks)))
    return tuple(phases)


def load_membership_plan(
    path: str | os.PathLike[str], *, max_ranks: int
) -> tuple[tuple[int, ...], ...]:
    """Read and validate an elastic membership plan."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read membership plan {path}: {error}") from error
    return parse_membership_plan(document, max_ranks=max_ranks)


def validate_example_configuration(
    *,
    devices: Sequence[int],
    plan: Sequence[Sequence[int]],
    experts_per_rank: int,
    top_k: int,
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
) -> None:
    """Validate shape/routing constraints independently of CUDA availability."""

    if not devices:
        raise ValueError("at least one CUDA device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA device indices")
    for name, value in (
        ("experts_per_rank", experts_per_rank),
        ("top_k", top_k),
        ("num_tokens", num_tokens),
        ("hidden_size", hidden_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("warmup", warmup), ("iterations", iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if iterations == 0:
        raise ValueError("iterations must be positive")
    for phase_index, phase in enumerate(plan):
        if top_k > len(phase) * experts_per_rank:
            raise ValueError(
                f"top_k={top_k} exceeds {len(phase) * experts_per_rank} active "
                f"experts in phase {phase_index}"
            )


def _bytes_tensor(value: bytes) -> torch.Tensor:
    # bytearray makes the buffer writable and avoids torch.frombuffer warnings.
    return torch.frombuffer(bytearray(value), dtype=torch.uint8)


def _validate_host_slab(slab: torch.Tensor, layout: PeerSlabLayout) -> None:
    if not isinstance(slab, torch.Tensor):
        raise TypeError("slab must be a torch.Tensor")
    if slab.device.type != "cpu" or slab.dtype != torch.uint8 or slab.ndim != 1:
        raise ValueError("slab must be a one-dimensional CPU uint8 tensor")
    if not slab.is_contiguous() or slab.numel() != layout.slab_nbytes:
        raise ValueError(f"slab must be contiguous with {layout.slab_nbytes} bytes")


def _validate_payload_dtype(payload_dtype: torch.dtype, layout: PeerSlabLayout) -> None:
    try:
        prototype = torch.empty((), dtype=payload_dtype)
    except (TypeError, RuntimeError) as error:
        raise TypeError("payload_dtype must be a torch dtype") from error
    if not prototype.dtype.is_floating_point:
        raise TypeError("payload_dtype must be floating point")
    if prototype.element_size() != layout.element_size:
        raise ValueError(
            f"payload dtype uses {prototype.element_size()} bytes, layout uses "
            f"{layout.element_size}"
        )


def _write_bytes(slab: torch.Tensor, offset: int, value: bytes) -> None:
    slab[offset : offset + len(value)].copy_(_bytes_tensor(value))


def _payload_bytes(payload: torch.Tensor, layout: PeerSlabLayout) -> torch.Tensor:
    if payload.device.type != "cpu" or payload.ndim != 1:
        raise ValueError("wire payloads must be one-dimensional CPU tensors")
    if payload.shape[0] != layout.hidden_size:
        raise ValueError("wire payload hidden size does not match layout")
    if payload.element_size() != layout.element_size:
        raise ValueError("wire payload element size does not match layout")
    return payload.detach().contiguous().view(torch.uint8).reshape(-1)


def encode_dispatch_slab(
    records: Sequence[DispatchRecord],
    layout: PeerSlabLayout,
    *,
    origin_rank: int,
    destination_rank: int,
    generation: int,
) -> torch.Tensor:
    """Encode one source's dense records for one destination rank."""

    ordered = sorted(records, key=lambda record: record.slot)
    if [record.slot for record in ordered] != list(range(len(ordered))):
        raise ValueError("dispatch slab slots must be dense from zero")
    if len(ordered) > layout.record_capacity:
        raise ValueError("dispatch record count exceeds slab capacity")

    layout.slab(destination_rank)
    slab = torch.zeros(layout.slab_nbytes, dtype=torch.uint8)
    _write_bytes(
        slab,
        0,
        layout.encode_preamble(
            origin_rank=origin_rank,
            destination_rank=destination_rank,
            generation=generation,
            record_count=len(ordered),
        ),
    )
    origin_slab = layout.slab(origin_rank)
    for record in ordered:
        if record.destination_rank != destination_rank:
            raise ValueError("dispatch record belongs to another destination")
        if record.metadata.origin_rank != origin_rank:
            raise ValueError("dispatch record belongs to another origin")
        if record.metadata.generation != generation:
            raise ValueError("dispatch record has a stale generation")
        expected = layout.record(origin_rank, record.slot)
        if record.location != expected:
            raise ValueError("dispatch record location does not match layout")
        header_offset = expected.header_offset - origin_slab.offset
        payload_offset = expected.payload_offset - origin_slab.offset
        _write_bytes(
            slab,
            header_offset,
            record.metadata.to_bytes(layout.header_size),
        )
        payload = _payload_bytes(record.payload, layout)
        slab[payload_offset : payload_offset + layout.payload_nbytes].copy_(payload)
    return slab


def decode_dispatch_slab(
    slab: torch.Tensor,
    layout: PeerSlabLayout,
    *,
    origin_rank: int,
    destination_rank: int,
    generation: int,
    payload_dtype: torch.dtype,
) -> tuple[DispatchRecord, ...]:
    """Decode and generation-check a received dispatch slab."""

    _validate_host_slab(slab, layout)
    _validate_payload_dtype(payload_dtype, layout)
    layout.slab(destination_rank)
    origin_slab = layout.slab(origin_rank)
    preamble = layout.decode_preamble(
        bytes(slab[: layout.preamble_size].tolist()),
        expected_origin_rank=origin_rank,
        expected_destination_rank=destination_rank,
        expected_generation=generation,
    )
    records: list[DispatchRecord] = []
    for slot in range(preamble.record_count):
        location = layout.record(origin_rank, slot)
        header_offset = location.header_offset - origin_slab.offset
        payload_offset = location.payload_offset - origin_slab.offset
        metadata = DispatchMetadata.from_bytes(
            bytes(slab[header_offset : header_offset + layout.header_size].tolist())
        )
        if metadata.generation != generation or metadata.origin_rank != origin_rank:
            raise ValueError("dispatch header does not match slab generation/origin")
        payload = (
            slab[payload_offset : payload_offset + layout.payload_nbytes]
            .contiguous()
            .view(payload_dtype)
            .clone()
        )
        records.append(
            DispatchRecord(
                destination_rank=destination_rank,
                slot=slot,
                location=location,
                metadata=metadata,
                payload=payload,
            )
        )
    return tuple(records)


def encode_combine_slab(
    outputs: Sequence[ExpertOutput],
    layout: PeerSlabLayout,
    *,
    source_rank: int,
    destination_rank: int,
    generation: int,
) -> torch.Tensor:
    """Encode expert results from one source rank back to one origin rank."""

    ordered = sorted(
        outputs,
        key=lambda output: (
            output.metadata.origin_token,
            output.metadata.route_slot,
            output.metadata.expert,
        ),
    )
    if len(ordered) > layout.record_capacity:
        raise ValueError("combine record count exceeds slab capacity")
    layout.slab(destination_rank)
    slab = torch.zeros(layout.slab_nbytes, dtype=torch.uint8)
    _write_bytes(
        slab,
        0,
        layout.encode_preamble(
            origin_rank=source_rank,
            destination_rank=destination_rank,
            generation=generation,
            record_count=len(ordered),
        ),
    )
    source_slab = layout.slab(source_rank)
    for slot, output in enumerate(ordered):
        metadata = output.metadata
        if metadata.origin_rank != destination_rank:
            raise ValueError("combine output belongs to another origin rank")
        if metadata.generation != generation:
            raise ValueError("combine output has a stale generation")
        location = layout.record(source_rank, slot)
        header_offset = location.header_offset - source_slab.offset
        payload_offset = location.payload_offset - source_slab.offset
        _write_bytes(slab, header_offset, metadata.to_bytes(layout.header_size))
        payload = _payload_bytes(output.payload, layout)
        slab[payload_offset : payload_offset + layout.payload_nbytes].copy_(payload)
    return slab


def decode_combine_slab(
    slab: torch.Tensor,
    layout: PeerSlabLayout,
    *,
    source_rank: int,
    destination_rank: int,
    generation: int,
    payload_dtype: torch.dtype,
) -> tuple[ExpertOutput, ...]:
    """Decode expert results and reject stale or misrouted records."""

    _validate_host_slab(slab, layout)
    _validate_payload_dtype(payload_dtype, layout)
    layout.slab(destination_rank)
    source_slab = layout.slab(source_rank)
    preamble = layout.decode_preamble(
        bytes(slab[: layout.preamble_size].tolist()),
        expected_origin_rank=source_rank,
        expected_destination_rank=destination_rank,
        expected_generation=generation,
    )
    outputs: list[ExpertOutput] = []
    for slot in range(preamble.record_count):
        location = layout.record(source_rank, slot)
        header_offset = location.header_offset - source_slab.offset
        payload_offset = location.payload_offset - source_slab.offset
        metadata = DispatchMetadata.from_bytes(
            bytes(slab[header_offset : header_offset + layout.header_size].tolist())
        )
        if metadata.generation != generation:
            raise ValueError("combine header has a stale generation")
        if metadata.origin_rank != destination_rank:
            raise ValueError("combine header belongs to another origin rank")
        payload = (
            slab[payload_offset : payload_offset + layout.payload_nbytes]
            .contiguous()
            .view(payload_dtype)
            .clone()
        )
        outputs.append(ExpertOutput(metadata=metadata, payload=payload))
    return tuple(outputs)


def _region(tensor: torch.Tensor) -> DeviceRegion:
    return DeviceRegion(
        address=tensor.data_ptr(),
        length=tensor.numel() * tensor.element_size(),
        device_id=tensor.get_device(),
    )


@dataclass(frozen=True)
class RoundMetrics:
    rank: int
    elapsed_s: float
    dispatch_s: float
    combine_s: float
    remote_routes: int
    logical_remote_bytes: int
    padded_remote_bytes: int
    max_abs_error: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rank": self.rank,
            "elapsed_s": self.elapsed_s,
            "dispatch_s": self.dispatch_s,
            "combine_s": self.combine_s,
            "remote_routes": self.remote_routes,
            "logical_remote_bytes": self.logical_remote_bytes,
            "padded_remote_bytes": self.padded_remote_bytes,
            "max_abs_error": self.max_abs_error,
        }


class _ElasticTransport:
    """Fixed-capacity buffers and staged peer-specific NIXL device views."""

    def __init__(
        self,
        *,
        rank: int,
        device: int,
        max_ranks: int,
        layout: PeerSlabLayout,
        control: FileControlPlane,
        timeout_s: float,
        run_id: str,
        disconnect_grace_s: float,
    ) -> None:
        from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

        self.rank = rank
        self.device = device
        self.max_ranks = max_ranks
        self.layout = layout
        self.control = control
        self.timeout_s = timeout_s
        self.disconnect_grace_s = disconnect_grace_s
        self.stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(self.stream):
            self.dispatch_send = torch.zeros(
                layout.arena_nbytes, dtype=torch.uint8, device=f"cuda:{device}"
            )
            self.combine_send = torch.zeros_like(self.dispatch_send)
            self.dispatch_recv = torch.zeros_like(self.dispatch_send)
            self.combine_recv = torch.zeros_like(self.dispatch_send)
            self.counters = torch.zeros(
                (2, max_ranks), dtype=torch.int64, device=f"cuda:{device}"
            )
            self.rank_mask = torch.ones(
                max_ranks, dtype=torch.int32, device=f"cuda:{device}"
            )
            self.statuses = torch.empty(2, dtype=torch.int32, device=f"cuda:{device}")
        self.stream.synchronize()

        self.name = f"cute_elastic_{run_id}_{rank}"
        self.agent = nixl_agent(
            self.name,
            nixl_agent_config(
                enable_prog_thread=True,
                backends=["UCX"],
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
            ),
        )
        self._registered_tensors = [
            self.dispatch_send,
            self.combine_send,
            self.dispatch_recv,
            self.combine_recv,
            self.counters,
        ]
        self._registration = self.agent.register_memory(
            self._registered_tensors, backends=["UCX"]
        )
        self._local_view = self.agent.prepare_device_view(
            [self.dispatch_send, self.combine_send], backend="UCX"
        )
        self._local_coordinates = PeerCoordinates(
            self.name, tuple(_region(tensor) for tensor in self._registered_tensors)
        )
        self._loaded: set[int] = set()
        self._active_views: dict[int, Any] = {}
        self._staged_views: dict[int, Any] = {}
        self._seen = [[0] * max_ranks for _ in range(2)]
        self._remote_views_released = False
        self._metadata_invalidated = False
        self._local_resources_released = False
        self._closed = False

    def _wait_for_stage_notifications(
        self, peers: Iterable[int], message: bytes
    ) -> None:
        pending = set(peers)

        def complete() -> bool:
            notifications = self.agent.get_new_notifs()
            for peer in tuple(pending):
                peer_name = self._peer_name(peer)
                if message in notifications.get(peer_name, ()):
                    pending.remove(peer)
            return not pending

        wait_until(
            complete,
            timeout_s=self.timeout_s,
            description=(
                f"generation notification {message!r} from ranks {sorted(pending)}"
            ),
            poll_interval_s=0.02,
        )

    def _peer_name(self, rank: int) -> str:
        return f"cute_elastic_{Path(self.control.directory).name}_{rank}"

    def stage(self, topology: ElasticExpertTopology) -> tuple[float, int]:
        """Prepare a complete next-generation view without activating it."""

        if self._staged_views:
            raise RuntimeError("a staged generation is already pending")
        start = time.perf_counter()
        prepared = 0
        if topology.is_active(self.rank):
            participants = topology.active_ranks
            metadata = self.control.exchange(
                f"g{topology.generation}.metadata",
                self.agent.get_agent_metadata(),
                participants,
            )
            coordinate_payloads = self.control.exchange(
                f"g{topology.generation}.coordinates",
                self._local_coordinates.to_bytes(),
                participants,
            )
            coordinates = {
                peer: PeerCoordinates.from_bytes(payload)
                for peer, payload in coordinate_payloads.items()
            }
            desired = set(participants) - {self.rank}
            newly_loaded: list[int] = []
            for peer in sorted(desired):
                expected_name = coordinates[peer].agent_name
                if expected_name != self._peer_name(peer):
                    raise RuntimeError(
                        f"rank {self.rank} expected peer {self._peer_name(peer)!r}, "
                        f"got {expected_name!r}"
                    )
                if peer not in self._loaded:
                    loaded_name = normalize_agent_name(
                        self.agent.add_remote_agent(metadata[peer])
                    )
                    if loaded_name != expected_name:
                        raise RuntimeError(
                            f"metadata loaded {loaded_name!r}, expected "
                            f"{expected_name!r}"
                        )
                    self._loaded.add(peer)
                    newly_loaded.append(peer)

            self.control.barrier(
                f"g{topology.generation}.metadata-loaded", participants
            )
            message = f"stage:{topology.generation}".encode("ascii")
            for peer in newly_loaded:
                peer_name = self._peer_name(peer)
                self.agent.make_connection(peer_name, backends=["UCX"])
                self.agent.send_notif(peer_name, message, backend="UCX")
            if newly_loaded:
                self._wait_for_stage_notifications(newly_loaded, message)
            self.control.barrier(
                f"g{topology.generation}.connections-ready", participants
            )

            try:
                for peer in sorted(desired):
                    peer_regions = coordinates[peer].regions
                    if len(peer_regions) != 5:
                        raise RuntimeError(
                            f"rank {peer} published {len(peer_regions)} regions; "
                            "expected five fixed-capacity regions"
                        )
                    remote_regions = (
                        peer_regions[2],
                        peer_regions[3],
                        peer_regions[4],
                    )
                    self._staged_views[peer] = self.agent.prepare_device_view(
                        [region.descriptor for region in remote_regions],
                        remote_agent=self._peer_name(peer),
                        mem_type="VRAM",
                        backend="UCX",
                        connection_timeout_ms=max(1, int(self.timeout_s * 1000)),
                    )
                    prepared += 1
            except Exception:
                self.stream.synchronize()
                for view in self._staged_views.values():
                    view.release()
                self._staged_views.clear()
                raise

        # Standby and retiring ranks participate so old traffic cannot race a
        # commit merely because they are absent from the next active set.
        self.control.barrier(f"g{topology.generation}.stage-ready")
        return time.perf_counter() - start, prepared

    def commit(self, topology: ElasticExpertTopology) -> float:
        """Switch views and the stable GPU mask, then retire old capabilities."""

        start = time.perf_counter()
        self.stream.synchronize()
        old_views = self._active_views
        self._active_views = self._staged_views
        self._staged_views = {}

        mask = torch.tensor(topology.nixl_mask, dtype=torch.int32)
        with torch.cuda.stream(self.stream):
            self.rank_mask.copy_(mask)
        self.stream.synchronize()

        for view in old_views.values():
            view.release()
        desired = (
            set(topology.active_ranks) - {self.rank}
            if topology.is_active(self.rank)
            else set()
        )
        removed = sorted(self._loaded - desired)
        for peer in removed:
            self.agent.remove_remote_agent(self._peer_name(peer))
            self._loaded.remove(peer)
        retire_flags = self.control.exchange(
            f"g{topology.generation}.retire-flags",
            b"1" if removed else b"0",
        )
        if any(flag == b"1" for flag in retire_flags.values()):
            # The in-tree NIXL EP elastic test uses the same grace interval to
            # prevent metadata invalidation racing re-addition of an identical
            # rank/agent name. The host API exposes no stronger disconnect
            # completion primitive to this host wrapper.
            time.sleep(self.disconnect_grace_s)
        self.control.barrier(f"g{topology.generation}.retired")
        return time.perf_counter() - start

    def verify_staged_joins_are_masked(
        self,
        current: ElasticExpertTopology | None,
        staged: ElasticExpertTopology,
    ) -> int:
        """Prove that staged join views remain blocked by the committed GPU mask."""

        if (
            current is None
            or not current.is_active(self.rank)
            or not staged.is_active(self.rank)
        ):
            return 0
        joining = sorted(set(staged.active_ranks) - set(current.active_ranks))
        for peer in joining:
            try:
                remote_view = self._staged_views[peer]
            except KeyError as error:
                raise RuntimeError(
                    f"joining rank {peer} has no staged generation view"
                ) from error
            local_slab = self.layout.slab(peer)
            remote_slab = self.layout.slab(self.rank)
            launch_masked_thread_put_then_signal_host(
                self._local_view,
                remote_view,
                self.rank_mask,
                self.statuses,
                self.stream,
                destination_rank=peer,
                size=self.layout.slab_nbytes,
                local_index=_LOCAL_DISPATCH_SEND,
                local_offset=local_slab.offset,
                remote_index=_REMOTE_DISPATCH_RECV,
                remote_offset=remote_slab.offset,
                signal_index=_REMOTE_COUNTERS,
                signal_offset=(DISPATCH * self.max_ranks + self.rank) * 8,
                signal_value=1,
                channel=0,
            )
            self.stream.synchronize()
            observed = tuple(int(value) for value in self.statuses.tolist())
            expected = (_NIXL_ERR_NOT_ALLOWED, _NIXL_ERR_NOT_ALLOWED)
            if observed != expected:
                raise RuntimeError(
                    f"staged rank {peer} bypassed the committed device mask: "
                    f"expected statuses {expected}, got {observed}"
                )
        return len(joining)

    def load_send_slab(
        self, direction: int, destination_rank: int, host_slab: torch.Tensor
    ) -> None:
        _validate_host_slab(host_slab, self.layout)
        slab = self.layout.slab(destination_rank)
        target = self.dispatch_send if direction == DISPATCH else self.combine_send
        with torch.cuda.stream(self.stream):
            target[slab.offset : slab.end].copy_(host_slab)

    def finish_loading(self) -> None:
        self.stream.synchronize()

    def publish(self, direction: int, ranks: Sequence[int]) -> float:
        """PUT one full slab to every peer and signal each on channel zero."""

        if direction not in (DISPATCH, COMBINE):
            raise ValueError("invalid communication direction")
        send = self.dispatch_send if direction == DISPATCH else self.combine_send
        receive = self.dispatch_recv if direction == DISPATCH else self.combine_recv
        local_index = (
            _LOCAL_DISPATCH_SEND if direction == DISPATCH else _LOCAL_COMBINE_SEND
        )
        remote_index = (
            _REMOTE_DISPATCH_RECV if direction == DISPATCH else _REMOTE_COMBINE_RECV
        )
        start = time.perf_counter()
        for peer in ranks:
            local_slab = self.layout.slab(peer)
            remote_slab = self.layout.slab(self.rank)
            if peer == self.rank:
                with torch.cuda.stream(self.stream):
                    receive[remote_slab.offset : remote_slab.end].copy_(
                        send[local_slab.offset : local_slab.end]
                    )
                    self.counters[direction, self.rank].add_(1)
                self.stream.synchronize()
                continue
            try:
                remote_view = self._active_views[peer]
            except KeyError as error:
                raise RuntimeError(
                    f"rank {peer} has no active generation view"
                ) from error
            launch_masked_thread_put_then_signal_host(
                self._local_view,
                remote_view,
                self.rank_mask,
                self.statuses,
                self.stream,
                destination_rank=peer,
                size=self.layout.slab_nbytes,
                local_index=local_index,
                local_offset=local_slab.offset,
                remote_index=remote_index,
                remote_offset=remote_slab.offset,
                signal_index=_REMOTE_COUNTERS,
                signal_offset=(direction * self.max_ranks + self.rank) * 8,
                signal_value=1,
                channel=0,
            )
            self.stream.synchronize()
            put_status, signal_status = (int(value) for value in self.statuses.tolist())
            check_status(
                put_status, f"{_DIRECTION_NAMES[direction]} PUT to rank {peer}"
            )
            check_status(
                signal_status,
                f"{_DIRECTION_NAMES[direction]} completion signal to rank {peer}",
            )
        return time.perf_counter() - start

    def receive_slabs(
        self, direction: int, ranks: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        """Wait for exactly one new signal per source, then copy slabs to CPU."""

        if direction not in (DISPATCH, COMBINE):
            raise ValueError("invalid communication direction")
        receive = self.dispatch_recv if direction == DISPATCH else self.combine_recv
        result: dict[int, torch.Tensor] = {}
        for source in ranks:
            expected = self._seen[direction][source] + 1

            def arrived(source: int = source, expected: int = expected) -> bool:
                return int(self.counters[direction, source].item()) == expected

            wait_until(
                arrived,
                timeout_s=self.timeout_s,
                description=(
                    f"{_DIRECTION_NAMES[direction]} signal {expected} from "
                    f"rank {source}"
                ),
                poll_interval_s=0.005,
            )
            self._seen[direction][source] = expected
            slab = self.layout.slab(source)
            result[source] = receive[slab.offset : slab.end].detach().cpu().contiguous()
        return result

    def release_remote_views(self) -> None:
        """Release outbound capabilities after all GPU work is quiescent."""

        if self._remote_views_released:
            return
        self.stream.synchronize()
        for view in self._staged_views.values():
            view.release()
        self._staged_views.clear()
        for view in self._active_views.values():
            view.release()
        self._active_views.clear()
        self._remote_views_released = True

    def invalidate_remote_metadata(self) -> None:
        """Invalidate peers only after a world fence on released views."""

        if self._metadata_invalidated:
            return
        if not self._remote_views_released:
            raise RuntimeError("release remote views before invalidating metadata")
        for peer in sorted(self._loaded):
            self.agent.remove_remote_agent(self._peer_name(peer))
        self._loaded.clear()
        self._metadata_invalidated = True

    def release_local_resources(self) -> None:
        """Deregister local buffers only after peers released remote views."""

        if self._local_resources_released:
            return
        if not self._metadata_invalidated:
            raise RuntimeError("invalidate metadata before releasing local resources")
        self._local_view.release()
        self.agent.deregister_memory(self._registration, backends=["UCX"])
        self._local_resources_released = True
        self._closed = True

    def close(self) -> None:
        """Best-effort exceptional cleanup; normal teardown uses world fences."""

        if self._closed:
            return
        self.release_remote_views()
        self.invalidate_remote_metadata()
        self.release_local_resources()


def _make_tokens(
    rank: int, generation: int, iteration: int, rows: int, hidden: int
) -> torch.Tensor:
    values = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)
    values = ((values + rank * 17 + generation * 31 + iteration * 7) % 251) / 32
    return values.to(torch.bfloat16)


def _make_scores(
    rank: int,
    generation: int,
    iteration: int,
    rows: int,
    topology: ElasticExpertTopology,
    top_k: int,
) -> torch.Tensor:
    scores = torch.full((rows, topology.max_experts), -1000.0)
    active = topology.active_experts
    for token in range(rows):
        start = (rank * rows + generation * 3 + iteration + token * top_k) % len(active)
        for slot in range(top_k):
            expert = active[(start + slot) % len(active)]
            scores[token, expert] = 4.0 - slot * 0.5
    return scores


def _golden_origin(
    tokens: torch.Tensor,
    routing: Any,
    topology: ElasticExpertTopology,
    hidden_size: int,
) -> torch.Tensor:
    outputs: list[ExpertOutput] = []
    for token in range(routing.num_tokens):
        for slot in range(routing.top_k):
            expert = int(routing.expert_indices[token, slot].item())
            payload = expert_transform(tokens[token].reshape(1, -1), expert)[0]
            outputs.append(
                ExpertOutput(
                    DispatchMetadata(
                        topology.generation,
                        routing.origin_rank,
                        token,
                        slot,
                        expert,
                    ),
                    payload.contiguous(),
                )
            )
    return weighted_combine(outputs, routing, topology, hidden_size=hidden_size)


def _execute_round(
    transport: _ElasticTransport,
    topology: ElasticExpertTopology,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    iteration: int,
    label: str,
) -> RoundMetrics | None:
    if not topology.is_active(transport.rank):
        return None
    active = topology.active_ranks
    tag = f"g{topology.generation}.{label}"
    transport.control.barrier(f"{tag}.start", active)
    round_start = time.perf_counter()

    tokens = _make_tokens(
        transport.rank, topology.generation, iteration, num_tokens, hidden_size
    )
    scores = _make_scores(
        transport.rank,
        topology.generation,
        iteration,
        num_tokens,
        topology,
        top_k,
    )
    routing = route_topk(
        scores,
        topology,
        origin_rank=transport.rank,
        top_k=top_k,
    )
    records = pack_dispatch(tokens, routing, topology, transport.layout)
    by_destination = {
        destination: [
            record for record in records if record.destination_rank == destination
        ]
        for destination in active
    }
    for destination in active:
        transport.load_send_slab(
            DISPATCH,
            destination,
            encode_dispatch_slab(
                by_destination[destination],
                transport.layout,
                origin_rank=transport.rank,
                destination_rank=destination,
                generation=topology.generation,
            ),
        )
    transport.finish_loading()

    dispatch_start = time.perf_counter()
    transport.publish(DISPATCH, active)
    dispatch_slabs = transport.receive_slabs(DISPATCH, active)
    dispatch_s = time.perf_counter() - dispatch_start

    incoming: list[DispatchRecord] = []
    for source in active:
        incoming.extend(
            decode_dispatch_slab(
                dispatch_slabs[source],
                transport.layout,
                origin_rank=source,
                destination_rank=transport.rank,
                generation=topology.generation,
                payload_dtype=tokens.dtype,
            )
        )
    batches = unpack_dispatch(
        incoming,
        topology,
        transport.layout,
        destination_rank=transport.rank,
    )
    local_outputs = process_expert_batches(batches)
    by_origin = {
        origin: [
            output for output in local_outputs if output.metadata.origin_rank == origin
        ]
        for origin in active
    }
    for origin in active:
        transport.load_send_slab(
            COMBINE,
            origin,
            encode_combine_slab(
                by_origin[origin],
                transport.layout,
                source_rank=transport.rank,
                destination_rank=origin,
                generation=topology.generation,
            ),
        )
    transport.finish_loading()

    combine_start = time.perf_counter()
    transport.publish(COMBINE, active)
    combine_slabs = transport.receive_slabs(COMBINE, active)
    combine_s = time.perf_counter() - combine_start

    returned: list[ExpertOutput] = []
    for source in active:
        returned.extend(
            decode_combine_slab(
                combine_slabs[source],
                transport.layout,
                source_rank=source,
                destination_rank=transport.rank,
                generation=topology.generation,
                payload_dtype=tokens.dtype,
            )
        )
    combined = weighted_combine(returned, routing, topology, hidden_size=hidden_size)
    expected = _golden_origin(tokens, routing, topology, hidden_size)
    max_abs_error = float((combined.float() - expected.float()).abs().max().item())
    if not torch.allclose(combined.float(), expected.float(), rtol=2e-2, atol=2e-2):
        raise RuntimeError(
            f"rank {transport.rank} generation {topology.generation} MoE output "
            f"mismatch (max_abs_error={max_abs_error})"
        )

    elapsed_s = time.perf_counter() - round_start
    transport.control.barrier(f"{tag}.validated", active)
    remote_routes = sum(record.destination_rank != transport.rank for record in records)
    logical_one_way = remote_routes * (
        transport.layout.header_size + transport.layout.payload_nbytes
    )
    return RoundMetrics(
        rank=transport.rank,
        elapsed_s=elapsed_s,
        dispatch_s=dispatch_s,
        combine_s=combine_s,
        remote_routes=remote_routes,
        logical_remote_bytes=2 * logical_one_way,
        padded_remote_bytes=(2 * (len(active) - 1) * transport.layout.slab_nbytes),
        max_abs_error=max_abs_error,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_phase(
    *,
    topology: ElasticExpertTopology,
    rank_samples: Mapping[int, Sequence[RoundMetrics]],
    num_tokens: int,
    stage_s_by_rank: Mapping[int, float],
    commit_s_by_rank: Mapping[int, float],
) -> dict[str, object]:
    """Build the machine-readable phase record printed by the example."""

    if set(rank_samples) != set(topology.active_ranks):
        raise ValueError("rank_samples must contain exactly the active ranks")
    sample_counts = {len(samples) for samples in rank_samples.values()}
    if len(sample_counts) != 1 or not sample_counts or next(iter(sample_counts)) == 0:
        raise ValueError("every active rank must provide an equal non-empty sample")
    iterations = next(iter(sample_counts))
    critical_path = [
        max(rank_samples[rank][i].elapsed_s for rank in topology.active_ranks)
        for i in range(iterations)
    ]
    logical_bytes = [
        sum(
            rank_samples[rank][i].logical_remote_bytes for rank in topology.active_ranks
        )
        for i in range(iterations)
    ]
    padded_bytes = [
        sum(rank_samples[rank][i].padded_remote_bytes for rank in topology.active_ranks)
        for i in range(iterations)
    ]
    return {
        "case": "nixl_cute_elastic_full_moe",
        "generation": topology.generation,
        "active_ranks": list(topology.active_ranks),
        "rank_bound": topology.rank_bound,
        "nixl_mask": list(topology.nixl_mask),
        "iterations": iterations,
        "critical_path_samples_s": critical_path,
        "p50_us": _percentile(critical_path, 0.50) * 1e6,
        "p90_us": _percentile(critical_path, 0.90) * 1e6,
        "p99_us": _percentile(critical_path, 0.99) * 1e6,
        "max_us": max(critical_path) * 1e6,
        "tokens_per_s": (
            len(topology.active_ranks) * num_tokens / _percentile(critical_path, 0.5)
        ),
        "logical_remote_GBps": _percentile(
            [
                nbytes / seconds / 1e9
                for nbytes, seconds in zip(logical_bytes, critical_path)
            ],
            0.5,
        ),
        "padded_transfer_GBps": _percentile(
            [
                nbytes / seconds / 1e9
                for nbytes, seconds in zip(padded_bytes, critical_path)
            ],
            0.5,
        ),
        "stage_max_ms": max(stage_s_by_rank.values()) * 1e3,
        "commit_max_ms": max(commit_s_by_rank.values()) * 1e3,
        "rank_samples": {
            str(rank): [sample.to_dict() for sample in samples]
            for rank, samples in rank_samples.items()
        },
    }


def _worker(
    rank: int,
    devices: tuple[int, ...],
    directory: str,
    phases: tuple[tuple[int, ...], ...],
    experts_per_rank: int,
    top_k: int,
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    staged_traffic: int,
    timeout_s: float,
    disconnect_grace_s: float,
) -> None:
    torch.cuda.set_device(devices[rank])
    max_ranks = len(devices)
    control = FileControlPlane(directory, rank, max_ranks, timeout_s)
    layout = PeerSlabLayout(
        max_ranks=max_ranks,
        max_tokens_per_rank=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        element_size=torch.empty((), dtype=torch.bfloat16).element_size(),
    )
    transport = _ElasticTransport(
        rank=rank,
        device=devices[rank],
        max_ranks=max_ranks,
        layout=layout,
        control=control,
        timeout_s=timeout_s,
        run_id=Path(directory).name,
        disconnect_grace_s=disconnect_grace_s,
    )
    current: ElasticExpertTopology | None = None
    normal_cleanup = False
    try:
        for generation, active in enumerate(phases):
            topology = ElasticExpertTopology(
                max_ranks=max_ranks,
                experts_per_rank=experts_per_rank,
                active_ranks=active,
                generation=generation,
            )
            stage_s, prepared = transport.stage(topology)
            masked_stage_probes = transport.verify_staged_joins_are_masked(
                current, topology
            )

            # Exercise the important activate=False behavior: the next views
            # exist, but routing still uses the old generation until a global
            # quiescent point. This is intentionally bounded graceful traffic.
            if current is not None:
                for index in range(staged_traffic):
                    _execute_round(
                        transport,
                        current,
                        num_tokens=num_tokens,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        iteration=10_000 + generation * 100 + index,
                        label=f"precommit-to-g{generation}-{index}",
                    )
            control.barrier(f"g{generation}.old-generation-quiesced")
            commit_s = transport.commit(topology)
            transition_payload = json.dumps(
                {
                    "rank": rank,
                    "stage_s": stage_s,
                    "commit_s": commit_s,
                    "prepared_views": prepared,
                    "masked_stage_probes": masked_stage_probes,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            transitions = control.exchange(
                f"g{generation}.transition", transition_payload
            )

            local_samples: list[RoundMetrics] = []
            if topology.is_active(rank):
                for index in range(warmup):
                    _execute_round(
                        transport,
                        topology,
                        num_tokens=num_tokens,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        iteration=index,
                        label=f"warmup-{index}",
                    )
                for index in range(iterations):
                    metrics = _execute_round(
                        transport,
                        topology,
                        num_tokens=num_tokens,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        iteration=warmup + index,
                        label=f"sample-{index}",
                    )
                    if metrics is None:
                        raise AssertionError("active rank produced no metrics")
                    local_samples.append(metrics)
                samples = control.exchange(
                    f"g{generation}.metrics",
                    json.dumps(
                        [sample.to_dict() for sample in local_samples],
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    topology.active_ranks,
                )
                if rank == min(topology.active_ranks):
                    rank_samples = {
                        peer: tuple(
                            RoundMetrics(**entry) for entry in json.loads(payload)
                        )
                        for peer, payload in samples.items()
                    }
                    transition_values = {
                        peer: json.loads(payload)
                        for peer, payload in transitions.items()
                    }
                    result = summarize_phase(
                        topology=topology,
                        rank_samples=rank_samples,
                        num_tokens=num_tokens,
                        stage_s_by_rank={
                            peer: value["stage_s"]
                            for peer, value in transition_values.items()
                        },
                        commit_s_by_rank={
                            peer: value["commit_s"]
                            for peer, value in transition_values.items()
                        },
                    )
                    result["layout"] = {
                        "slab_nbytes": layout.slab_nbytes,
                        "arena_nbytes": layout.arena_nbytes,
                        "record_capacity": layout.record_capacity,
                    }
                    result["configuration"] = {
                        "max_ranks": max_ranks,
                        "experts_per_rank": experts_per_rank,
                        "top_k": top_k,
                        "num_tokens_per_rank": num_tokens,
                        "hidden_size": hidden_size,
                        "dtype": "bfloat16",
                        "scope": "thread",
                        "channel": 0,
                        "cuda_graph": False,
                    }
                    result["masked_stage_probes"] = sum(
                        value["masked_stage_probes"]
                        for value in transition_values.values()
                    )
                    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)

            control.barrier(f"g{generation}.phase-complete")
            current = topology

        control.barrier("final-quiesced")
        transport.release_remote_views()
        control.barrier("final-remote-views-released")
        transport.invalidate_remote_metadata()
        control.barrier("final-metadata-invalidated")
        transport.release_local_resources()
        normal_cleanup = True
    finally:
        if not normal_cleanup:
            transport.close()
    control.barrier("cleanup-complete")


def run(
    *,
    devices: tuple[int, ...],
    plan_path: str | os.PathLike[str] = DEFAULT_PLAN,
    experts_per_rank: int = 2,
    top_k: int = 2,
    num_tokens: int = 8,
    hidden_size: int = 256,
    warmup: int = 1,
    iterations: int = 3,
    staged_traffic: int = 1,
    timeout_s: float = 60.0,
    disconnect_grace_s: float = 5.0,
) -> None:
    """Spawn one persistent process per fixed-capacity local GPU rank."""

    phases = load_membership_plan(plan_path, max_ranks=len(devices))
    validate_example_configuration(
        devices=devices,
        plan=phases,
        experts_per_rank=experts_per_rank,
        top_k=top_k,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        warmup=warmup,
        iterations=iterations,
    )
    if (
        isinstance(staged_traffic, bool)
        or not isinstance(staged_traffic, int)
        or staged_traffic < 0
    ):
        raise ValueError("staged_traffic must be a non-negative integer")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    if (
        isinstance(disconnect_grace_s, bool)
        or not isinstance(disconnect_grace_s, (int, float))
        or not math.isfinite(disconnect_grace_s)
        or disconnect_grace_s < 0
    ):
        raise ValueError("disconnect grace must be a non-negative finite number")
    if not torch.cuda.is_available() or torch.cuda.device_count() < len(devices):
        raise RuntimeError(
            f"this example requires {len(devices)} visible CUDA GPUs; found "
            f"{torch.cuda.device_count()}"
        )
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device indices must be integers")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")

    import torch.multiprocessing as mp

    with TemporaryDirectory(prefix=f"nixl_cute_elastic_{os.getpid()}_") as directory:
        mp.spawn(
            _worker,
            args=(
                devices,
                directory,
                phases,
                experts_per_rank,
                top_k,
                num_tokens,
                hidden_size,
                warmup,
                iterations,
                staged_traffic,
                timeout_s,
                disconnect_grace_s,
            ),
            nprocs=len(devices),
            join=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3),
        help="distinct local CUDA device indices; count fixes rank capacity",
    )
    parser.add_argument(
        "--plan",
        default=str(DEFAULT_PLAN),
        help="JSON list of active rank lists (same shape as NIXL EP plans)",
    )
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--staged-traffic",
        type=int,
        default=1,
        help="old-generation rounds run after next views are staged (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds allowed for each bounded host/control-plane wait",
    )
    parser.add_argument(
        "--disconnect-grace",
        type=float,
        default=5.0,
        help=(
            "seconds after metadata invalidation before same-rank re-add "
            "(matches the in-tree EP elastic test default: 5)"
        ),
    )
    args = parser.parse_args()
    run(
        devices=tuple(args.devices),
        plan_path=args.plan,
        experts_per_rank=args.experts_per_rank,
        top_k=args.top_k,
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        warmup=args.warmup,
        iterations=args.iterations,
        staged_traffic=args.staged_traffic,
        timeout_s=args.timeout,
        disconnect_grace_s=args.disconnect_grace,
    )


if __name__ == "__main__":
    main()
