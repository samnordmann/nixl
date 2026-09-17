# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

# Load the file directly: this proves the protocol contract does not need the
# CuTe, NIXL, CUDA, or Torch runtime (and avoids executing moe/__init__.py).
_PROTOCOL_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "python"
    / "cute"
    / "moe"
    / "ll_protocol.py"
)
_SPEC = importlib.util.spec_from_file_location("_cute_moe_ll_protocol", _PROTOCOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROTOCOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROTOCOL
_SPEC.loader.exec_module(_PROTOCOL)

BankPhase = _PROTOCOL.BankPhase
BucketKey = _PROTOCOL.BucketKey
BucketSchedule = _PROTOCOL.BucketSchedule
ByteSpan = _PROTOCOL.ByteSpan
CompactLLArenaLayout = _PROTOCOL.CompactLLArenaLayout
DataPost = _PROTOCOL.DataPost
Direction = _PROTOCOL.Direction
DispatchHeader = _PROTOCOL.DispatchHeader
LLArenaLayout = _PROTOCOL.LLArenaLayout
MessageStamp = _PROTOCOL.MessageStamp
OperationEpoch = _PROTOCOL.OperationEpoch
PipelineBucketStamp = _PROTOCOL.PipelineBucketStamp
PipelineLLArenaLayout = _PROTOCOL.PipelineLLArenaLayout
PipelineSourceInfo = _PROTOCOL.PipelineSourceInfo
PostFlag = _PROTOCOL.PostFlag
ProtocolError = _PROTOCOL.ProtocolError
PublicationPost = _PROTOCOL.PublicationPost
ReceiveCreditLifecycle = _PROTOCOL.ReceiveCreditLifecycle
ReceivePhase = _PROTOCOL.ReceivePhase
StableSparseTopology = _PROTOCOL.StableSparseTopology
TwoBankCreditLifecycle = _PROTOCOL.TwoBankCreditLifecycle
UINT32_MAX = _PROTOCOL.UINT32_MAX
UINT64_MAX = _PROTOCOL.UINT64_MAX
make_bucket_schedule = _PROTOCOL.make_bucket_schedule
gda_bucket_channel = _PROTOCOL.gda_bucket_channel
normalize_gda_channel_count = _PROTOCOL.normalize_gda_channel_count
pack_layout_range = _PROTOCOL.pack_layout_range
unpack_layout_range = _PROTOCOL.unpack_layout_range


def _topology(generation: int = 5) -> Any:
    return StableSparseTopology(
        max_ranks=4,
        experts_per_rank=2,
        active_ranks=(2, 0),
        membership_generation=generation,
        rank_incarnations=(11, 0, 7, 0),
    )


def _layout() -> Any:
    # A ten-byte payload intentionally exercises padding between expert rows.
    return LLArenaLayout(
        max_ranks=4,
        experts_per_rank=2,
        max_tokens_per_rank=3,
        top_k=2,
        hidden_size=5,
        element_size=2,
    )


def _compact_layout() -> Any:
    return CompactLLArenaLayout(
        max_ranks=4,
        experts_per_rank=2,
        num_tokens=3,
        top_k=2,
        hidden_size=5,
        element_size=2,
        num_banks=2,
    )


def _pipeline_layout() -> Any:
    return PipelineLLArenaLayout(
        max_ranks=4,
        experts_per_rank=2,
        num_tokens=3,
        top_k=2,
        hidden_size=8,
        element_size=2,
    )


def _assert_nonoverlapping(spans: list[Any]) -> None:
    ordered = sorted(spans, key=lambda span: span.offset)
    assert len({span.offset for span in ordered}) == len(ordered)
    for left, right in pairwise(ordered):
        assert left.end <= right.offset, f"{left.name} overlaps {right.name}"


def test_module_has_no_accelerator_or_tensor_runtime_imports():
    tree = ast.parse(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_roots <= {
        "__future__",
        "dataclasses",
        "enum",
        "itertools",
        "struct",
        "typing",
    }


def test_two_bank_arena_is_checked_aligned_and_nonoverlapping():
    layout = _layout()

    assert layout.payload_nbytes == 10
    assert layout.payload_stride == 16
    assert layout.dispatch_record_nbytes == 42
    assert layout.dispatch_record_stride == 48
    assert layout.combine_record_nbytes == 26
    assert layout.combine_record_stride == 32
    assert layout.arena_nbytes % layout.arena_alignment == 0
    _assert_nonoverlapping(list(layout.regions))

    banked_regions = (
        "dispatch_stage",
        "dispatch_recv",
        "dispatch_ready",
        "dispatch_credit",
        "expert_input",
        "dispatch_src_info",
        "dispatch_layout_range",
        "combine_stage",
        "combine_recv",
        "combine_ready",
        "combine_credit",
    )
    for name in banked_regions:
        whole = layout.region(name)
        bank_zero = layout.bank_region(name, 0)
        bank_one = layout.bank_region(name, 1)
        assert bank_zero.offset == whole.offset
        assert bank_zero.nbytes == bank_one.nbytes
        assert bank_zero.end == bank_one.offset
        assert bank_one.end == whole.end
        assert not bank_zero.overlaps(bank_one)

    dispatch_records = [
        layout.dispatch_receive_record(bank, expert, source, slot)
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for source in range(layout.max_ranks)
        for slot in range(layout.max_tokens_per_rank)
    ]
    _assert_nonoverlapping(dispatch_records)
    assert all(span.offset % layout.record_alignment == 0 for span in dispatch_records)
    assert all(
        layout.region("dispatch_recv").offset
        <= span.offset
        < span.end
        <= layout.region("dispatch_recv").end
        for span in dispatch_records
    )

    expert_rows = [
        layout.expert_input_record(bank, expert, source, slot)
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for source in range(layout.max_ranks)
        for slot in range(layout.max_tokens_per_rank)
    ]
    _assert_nonoverlapping(expert_rows)
    assert all(span.nbytes == layout.payload_stride for span in expert_rows)
    assert all(span.offset % layout.record_alignment == 0 for span in expert_rows)


def test_arena_offsets_are_stable_and_bounds_fail_closed():
    layout = _layout()

    assert layout.dispatch_stage_record(0, 2) == layout.dispatch_stage_record(0, 2)
    assert layout.dispatch_receive_record(1, 1, 2, 0) != (
        layout.dispatch_receive_record(1, 1, 0, 2)
    )
    assert not layout.dispatch_credit_word(0, 3).overlaps(
        layout.dispatch_credit_word(1, 0)
    )
    assert not layout.combine_stage_record(0, 1, 3, 2).overlaps(
        layout.combine_stage_record(1, 0, 0, 0)
    )
    assert not layout.combine_receive_record(0, 7, 2).overlaps(
        layout.combine_receive_record(1, 0, 0)
    )

    with pytest.raises(IndexError, match="bank"):
        layout.dispatch_stage_record(2, 0)
    with pytest.raises(IndexError, match="local_expert"):
        layout.dispatch_receive_record(0, 2, 0, 0)
    with pytest.raises(IndexError, match="source_rank"):
        layout.dispatch_receive_record(0, 0, 4, 0)
    with pytest.raises(IndexError, match="origin_token"):
        layout.combine_receive_record(0, 0, 3)
    with pytest.raises(KeyError, match="unknown"):
        layout.region("missing")
    with pytest.raises(ValueError, match="not banked"):
        layout.bank_region("control", 0)
    with pytest.raises(ValueError, match="16-byte PUT ABI"):
        LLArenaLayout(2, 1, 1, 1, 8, 2, record_alignment=8)
    with pytest.raises(ValueError, match="expert capacity"):
        LLArenaLayout(2, 1, 1, 3, 8, 2)
    with pytest.raises(OverflowError, match="address limit"):
        LLArenaLayout(4, 2, 3, 2, 5, 2, address_limit=256)
    with pytest.raises(OverflowError, match="span"):
        ByteSpan("overflow", UINT64_MAX, 1, 1)


def test_compact_arena_has_exact_shapes_bounds_and_no_overlap():
    layout = _compact_layout()
    expected_nbytes = {
        "dispatch_stage": 3 * 2 * layout.dispatch_record_stride,
        "dispatch_recv": 2 * 2 * 4 * 3 * layout.dispatch_record_stride,
        "dispatch_stamp": 2 * 2 * 4 * 16,
        "dispatch_ready": 2 * 2 * 4 * 8,
        "combine_consumed": 2 * 4 * 8,
        "abort_state": 8,
        "combine_recv": 2 * 2 * 3 * layout.combine_record_stride,
        "combine_stamp": 2 * 4 * 16,
        "combine_ready": 2 * 4 * 8,
    }

    assert layout.route_capacity == 6
    assert layout.dispatch_record_nbytes == 26
    assert layout.dispatch_record_stride == 32
    assert layout.combine_record_nbytes == 10
    assert layout.combine_record_stride == 16
    assert tuple(region.name for region in layout.regions) == tuple(expected_nbytes)
    assert {region.name: region.nbytes for region in layout.regions} == expected_nbytes
    assert layout.arena_nbytes == 4096
    assert layout.arena_nbytes % layout.arena_alignment == 0
    _assert_nonoverlapping(list(layout.regions))
    for name in (
        "dispatch_recv",
        "dispatch_stamp",
        "dispatch_ready",
        "combine_consumed",
        "combine_recv",
        "combine_stamp",
        "combine_ready",
    ):
        bank_zero = layout.bank_region(name, 0)
        bank_one = layout.bank_region(name, 1)
        assert bank_zero.end == bank_one.offset
        assert bank_one.end == layout.region(name).end
    with pytest.raises(ValueError, match="not banked"):
        layout.bank_region("dispatch_stage", 0)
    with pytest.raises(ValueError, match="not banked"):
        layout.bank_region("abort_state", 0)

    items = [
        layout.dispatch_stage_record(route) for route in range(layout.route_capacity)
    ]
    items += [
        layout.dispatch_receive_record(bank, expert, source, token)
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for source in range(layout.max_ranks)
        for token in range(layout.num_tokens)
    ]
    items += [
        layout.dispatch_stamp(bank, expert, source)
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for source in range(layout.max_ranks)
    ]
    items += [
        layout.dispatch_ready_word(bank, expert, source)
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for source in range(layout.max_ranks)
    ]
    items += [
        accessor(bank, rank)
        for accessor in (layout.combine_consumed_word,)
        for bank in range(2)
        for rank in range(layout.max_ranks)
    ]
    items += [
        layout.combine_receive_record(bank, route_slot, token)
        for bank in range(2)
        for route_slot in range(layout.top_k)
        for token in range(layout.num_tokens)
    ]
    items += [
        layout.combine_stamp(bank, rank)
        for bank in range(2)
        for rank in range(layout.max_ranks)
    ]
    items += [
        layout.combine_ready_word(bank, rank)
        for bank in range(2)
        for rank in range(layout.max_ranks)
    ]
    items.append(layout.abort_state_word())
    _assert_nonoverlapping(items)
    assert all(0 <= span.offset < span.end <= layout.arena_nbytes for span in items)

    with pytest.raises(IndexError, match="slot"):
        layout.dispatch_receive_record(0, 0, 0, layout.num_tokens)
    with pytest.raises(IndexError, match="route"):
        layout.dispatch_stage_record(layout.route_capacity)
    with pytest.raises(IndexError, match="route_slot"):
        layout.combine_receive_record(0, layout.top_k, 0)
    with pytest.raises(IndexError, match="origin_token"):
        layout.combine_receive_record(0, 0, layout.num_tokens)
    with pytest.raises(OverflowError, match="address limit"):
        CompactLLArenaLayout(4, 2, 3, 2, 5, 2, address_limit=256)
    with pytest.raises(ValueError, match="num_banks"):
        CompactLLArenaLayout(4, 2, 3, 2, 5, 2, num_banks=3)


def test_pipeline_arena_has_two_banks_external_expert_planes_and_credits():
    layout = _pipeline_layout()
    expected_nbytes = {
        "control": 4 * 8,
        "rank_mask": 4 * 8,
        "rank_incarnation": 4 * 8,
        "dispatch_stage": 2 * 3 * 16,
        "dispatch_stage_src_info": 2 * 6 * 16,
        "dispatch_recv": 2 * 2 * 4 * 3 * 16,
        "dispatch_recv_src_info": 2 * 2 * 4 * 3 * 16,
        "expert_input": 2 * 2 * 4 * 3 * 16,
        "dispatch_src_info": 2 * 2 * 4 * 3 * 16,
        "dispatch_stamp": 2 * 2 * 4 * 32,
        "dispatch_ready_seq": 2 * 2 * 4 * 8,
        "dispatch_credit": 2 * 4 * 8,
        "dispatch_layout_range": 2 * 2 * 4 * 8,
        "combine_stage": 2 * 2 * 4 * 3 * 16,
        "combine_recv": 2 * 6 * 16,
        "combine_stamp": 2 * 4 * 32,
        "combine_ready_seq": 2 * 4 * 8,
        "combine_credit": 2 * 4 * 8,
        "bank_state": 2 * 8,
        "abort_state": 8,
    }

    assert layout.route_capacity == 6
    assert layout.payload_nbytes == 16
    assert layout.payload_stride == 16
    assert tuple(region.name for region in layout.regions) == tuple(expected_nbytes)
    assert {region.name: region.nbytes for region in layout.regions} == expected_nbytes
    assert layout.arena_nbytes % layout.arena_alignment == 0
    _assert_nonoverlapping(list(layout.regions))

    for name in expected_nbytes.keys() - {
        "control",
        "rank_mask",
        "rank_incarnation",
        "abort_state",
    }:
        bank_zero = layout.bank_region(name, 0)
        bank_one = layout.bank_region(name, 1)
        assert bank_zero.end == bank_one.offset
        assert bank_one.end == layout.region(name).end
        assert not bank_zero.overlaps(bank_one)

    items = [layout.control_word(index) for index in range(4)]
    items += [layout.rank_mask_word(rank) for rank in range(layout.max_ranks)]
    items += [layout.rank_incarnation_word(rank) for rank in range(layout.max_ranks)]
    items += [
        layout.dispatch_stage_record(bank, token)
        for bank in range(2)
        for token in range(layout.num_tokens)
    ]
    items += [
        layout.dispatch_stage_source_info(bank, token, route_slot)
        for bank in range(2)
        for token in range(layout.num_tokens)
        for route_slot in range(layout.top_k)
    ]
    items += [
        layout.combine_receive_record(bank, route_slot, token)
        for bank in range(2)
        for route_slot in range(layout.top_k)
        for token in range(layout.num_tokens)
    ]
    items += [
        accessor(bank, expert, rank, slot)
        for accessor in (
            layout.dispatch_receive_record,
            layout.dispatch_receive_source_info,
        )
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for rank in range(layout.max_ranks)
        for slot in range(layout.num_tokens)
    ]
    items += [
        accessor(bank, expert, packed_slot)
        for accessor in (
            layout.expert_input_record,
            layout.dispatch_source_info,
            layout.combine_stage_record,
        )
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for packed_slot in range(layout.max_ranks * layout.num_tokens)
    ]
    items += [
        accessor(bank, expert, rank)
        for accessor in (
            layout.dispatch_stamp,
            layout.dispatch_ready_sequence_word,
            layout.dispatch_layout_word,
        )
        for bank in range(2)
        for expert in range(layout.experts_per_rank)
        for rank in range(layout.max_ranks)
    ]
    items += [
        accessor(bank, rank)
        for accessor in (
            layout.dispatch_credit_word,
            layout.combine_stamp,
            layout.combine_ready_sequence_word,
            layout.combine_credit_word,
        )
        for bank in range(2)
        for rank in range(layout.max_ranks)
    ]
    items += [layout.bank_state_word(bank) for bank in range(2)]
    items.append(layout.abort_state_word())
    _assert_nonoverlapping(items)
    assert all(0 <= span.offset < span.end <= layout.arena_nbytes for span in items)

    # The expert boundary sees ordinary padded payload records. Route metadata
    # is in a disjoint plane and cannot silently change the GEMM row stride.
    raw_payload = layout.dispatch_receive_record(0, 0, 0, 0)
    raw_source_info = layout.dispatch_receive_source_info(0, 0, 0, 0)
    payload = layout.expert_input_record(0, 0, 0)
    source_info = layout.dispatch_source_info(0, 0, 0)
    combine = layout.combine_stage_record(0, 0, 0)
    assert raw_payload.nbytes == layout.payload_stride
    assert raw_source_info.nbytes == _PROTOCOL.SOURCE_INFO_NBYTES
    assert not raw_payload.overlaps(raw_source_info)
    assert payload.nbytes == layout.payload_stride
    assert source_info.nbytes == _PROTOCOL.SOURCE_INFO_NBYTES
    assert not payload.overlaps(source_info)
    assert not payload.overlaps(combine)
    assert (
        layout.combine_receive_record(0, 0, 0).end
        == layout.combine_receive_record(0, 0, 1).offset
    )
    assert (
        layout.combine_receive_record(0, 0, layout.num_tokens - 1).end
        == layout.combine_receive_record(0, 1, 0).offset
    )

    with pytest.raises(IndexError, match="bank"):
        layout.dispatch_stage_record(2, 0)
    with pytest.raises(IndexError, match="route_slot"):
        layout.combine_receive_record(0, layout.top_k, 0)
    with pytest.raises(IndexError, match="source_slot"):
        layout.dispatch_receive_record(0, 0, 0, layout.num_tokens)
    with pytest.raises(IndexError, match="packed_slot"):
        layout.expert_input_record(0, 0, layout.max_ranks * layout.num_tokens)
    with pytest.raises(ValueError, match="not banked"):
        layout.bank_region("control", 0)
    with pytest.raises(ValueError, match="16-byte PUT ABI"):
        PipelineLLArenaLayout(2, 1, 1, 1, 8, 2, record_alignment=8)
    with pytest.raises(ValueError, match="expert capacity"):
        PipelineLLArenaLayout(2, 1, 1, 3, 8, 2)
    with pytest.raises(ValueError, match="multiple of record_alignment"):
        PipelineLLArenaLayout(4, 2, 3, 2, 5, 2)
    with pytest.raises(OverflowError, match="address limit"):
        PipelineLLArenaLayout(4, 2, 3, 2, 8, 2, address_limit=256)


def test_pipeline_layout_avoids_legacy_full_expert_namespace_combine_plane():
    shape = dict(
        max_ranks=32,
        experts_per_rank=8,
        top_k=8,
        hidden_size=7168,
        element_size=2,
    )
    pipeline = PipelineLLArenaLayout(num_tokens=128, **shape)
    legacy = LLArenaLayout(max_tokens_per_rank=128 * 8, **shape)

    assert pipeline.region("combine_recv").nbytes == (
        2 * 128 * 8 * pipeline.payload_stride
    )
    assert pipeline.region("expert_input").nbytes == (
        2 * 8 * 32 * 128 * pipeline.payload_stride
    )
    assert pipeline.arena_nbytes * 8 < legacy.arena_nbytes


def test_compact_arena_materially_reduces_production_moe_storage():
    shape = dict(
        max_ranks=32,
        experts_per_rank=8,
        top_k=8,
        hidden_size=7168,
        element_size=2,
    )
    compact = CompactLLArenaLayout(num_tokens=128, **shape)
    two_bank = CompactLLArenaLayout(num_tokens=128, num_banks=2, **shape)
    legacy = LLArenaLayout(max_tokens_per_rank=128 * 8, **shape)

    assert compact.arena_nbytes == 499_671_040
    assert compact.arena_nbytes / (1 << 20) == pytest.approx(476.5234375)
    assert two_bank.arena_nbytes == 984_645_632
    assert compact.arena_nbytes * 1.9 < two_bank.arena_nbytes
    assert legacy.arena_nbytes == 30_136_156_160
    assert compact.arena_nbytes * 60 < legacy.arena_nbytes


def test_sparse_topology_preserves_rank_slots_experts_and_incarnations():
    topology = _topology()

    assert topology.active_ranks == (0, 2)
    assert topology.active_mask == (True, False, True, False)
    assert topology.nixl_mask == (0, 1, 0, 1)
    assert topology.rank_bound == 3
    assert topology.active_experts == (0, 1, 4, 5)
    assert topology.expert_id(2, 1) == 5
    assert topology.owner(5) == 2
    assert topology.local_expert(5) == 1

    shrunk = topology.transition((0,), membership_generation=6)
    assert shrunk.active_mask == (True, False, False, False)
    assert shrunk.incarnation(2) == 7
    rejoined = shrunk.transition(
        (0, 2),
        membership_generation=7,
        rank_incarnations=(11, 0, 8, 0),
    )
    expanded = rejoined.transition(
        (0, 2, 3),
        membership_generation=8,
        rank_incarnations=(11, 0, 8, 1),
    )
    assert rejoined.expert_id(2, 1) == topology.expert_id(2, 1)
    assert rejoined.incarnation(2) == 8
    assert expanded.active_experts == (0, 1, 4, 5, 6, 7)


def test_sparse_topology_rejects_ambiguous_or_backward_membership():
    topology = _topology()

    with pytest.raises(ValueError, match="at least one"):
        StableSparseTopology(2, 1, (), 0, (0, 0))
    with pytest.raises(ValueError, match="duplicates"):
        StableSparseTopology(2, 1, (0, 0), 0, (1, 0))
    with pytest.raises(IndexError, match="active rank"):
        StableSparseTopology(2, 1, (2,), 0, (0, 0))
    with pytest.raises(ValueError, match="no process incarnation"):
        StableSparseTopology(2, 1, (1,), 0, (1, 0))
    with pytest.raises(ValueError, match="exactly one"):
        StableSparseTopology(2, 1, (0,), 0, (1,))
    with pytest.raises(ValueError, match="advance"):
        topology.transition((0,), membership_generation=5)
    with pytest.raises(ValueError, match="moved backwards"):
        topology.transition(
            (0,),
            membership_generation=6,
            rank_incarnations=(10, 0, 7, 0),
        )
    with pytest.raises(OverflowError, match="expert namespace"):
        StableSparseTopology(UINT32_MAX, 2, (0,), 0, (1,) + (0,) * 3)


def test_operation_and_incarnation_stamp_round_trip_and_validation():
    topology = _topology()
    operation = OperationEpoch(topology.membership_generation, 3)
    stamp = MessageStamp(operation, topology.incarnation(2))

    assert operation.bank == 1
    assert OperationEpoch.from_wire(operation.wire_value) == operation
    assert MessageStamp.from_bytes(stamp.to_bytes()) == stamp
    stamp.validate(topology, source_rank=2, expected_operation=operation)

    header = DispatchHeader(stamp, origin_token=UINT32_MAX)
    assert DispatchHeader.from_bytes(header.to_bytes()) == header
    bad_padding = bytearray(header.to_bytes())
    bad_padding[-1] = 1
    with pytest.raises(ValueError, match="reserved"):
        DispatchHeader.from_bytes(bytes(bad_padding))

    with pytest.raises(ProtocolError, match="stale operation"):
        MessageStamp(OperationEpoch(5, 1), 7).validate(
            topology,
            source_rank=2,
            expected_operation=operation,
        )
    with pytest.raises(ProtocolError, match="committed membership"):
        stamp.validate(
            topology,
            source_rank=2,
            expected_operation=OperationEpoch(4, 3),
        )
    with pytest.raises(ProtocolError, match="masked"):
        MessageStamp(operation, 1).validate(
            topology,
            source_rank=1,
            expected_operation=operation,
        )
    with pytest.raises(ProtocolError, match="incarnation"):
        MessageStamp(operation, 6).validate(
            topology,
            source_rank=2,
            expected_operation=operation,
        )


def test_pipeline_bucket_stamp_separates_monotonic_publication_from_count():
    topology = _topology()
    operation = OperationEpoch(topology.membership_generation, 7)
    value = PipelineBucketStamp(
        MessageStamp(operation, topology.incarnation(2)),
        record_count=3,
    )

    encoded = value.to_bytes()
    assert len(encoded) == _PROTOCOL.PIPELINE_BUCKET_STAMP_NBYTES
    assert PipelineBucketStamp.from_bytes(encoded) == value
    value.validate(
        topology,
        source_rank=2,
        expected_operation=operation,
        capacity=3,
    )

    # Empty buckets are real publications: the sequence advances while the
    # ordered stamp carries a legitimate zero count.
    empty = PipelineBucketStamp(value.stamp, record_count=0)
    empty.validate(
        topology,
        source_rank=2,
        expected_operation=operation,
        capacity=3,
    )
    with pytest.raises(ProtocolError, match="exceeds capacity"):
        value.validate(
            topology,
            source_rank=2,
            expected_operation=operation,
            capacity=2,
        )
    failed = PipelineBucketStamp(value.stamp, record_count=3, failed=True)
    with pytest.raises(ProtocolError, match="transport failure"):
        failed.validate(
            topology,
            source_rank=2,
            expected_operation=operation,
            capacity=3,
        )
    invalid_failure_word = encoded[:-8] + (2).to_bytes(8, "little")
    with pytest.raises(ValueError, match="zero or one"):
        PipelineBucketStamp.from_bytes(invalid_failure_word)
    with pytest.raises(ValueError, match="expected"):
        PipelineBucketStamp.from_bytes(encoded[:-1])


def test_pipeline_bank_sequences_and_gda_channels_are_stable_and_bounded():
    assert [OperationEpoch(3, step).bank for step in range(6)] == [0, 1, 0, 1, 0, 1]
    assert [OperationEpoch(3, step).bank_sequence for step in range(6)] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [
        normalize_gda_channel_count(value) for value in (1, 2, 3, 4, 5, 255, 256)
    ] == [
        1,
        2,
        4,
        4,
        8,
        256,
        256,
    ]
    channel = gda_bucket_channel(peer_rank=7, bucket_ordinal=13, channel_count=8)
    assert channel == 4
    # Every post in one bucket calls the same pure mapping; transport code must
    # not choose a different channel for the ordered publication tail.
    assert {
        gda_bucket_channel(peer_rank=7, bucket_ordinal=13, channel_count=8)
        for _ in range(32)
    } == {channel}
    with pytest.raises(ValueError, match="positive"):
        normalize_gda_channel_count(0)
    with pytest.raises(ValueError, match=r"\[0, 256\]"):
        normalize_gda_channel_count(257)
    with pytest.raises(ValueError, match="power of two"):
        gda_bucket_channel(peer_rank=0, bucket_ordinal=0, channel_count=3)
    with pytest.raises(ValueError, match="must not exceed"):
        gda_bucket_channel(peer_rank=0, bucket_ordinal=0, channel_count=512)


def test_pipeline_source_info_and_layout_range_have_exact_wire_formats():
    info = PipelineSourceInfo(origin_rank=7, origin_token=19, route_slot=3)
    encoded = info.to_bytes()
    assert len(encoded) == _PROTOCOL.SOURCE_INFO_NBYTES
    assert PipelineSourceInfo.from_bytes(encoded) == info
    with pytest.raises(ValueError, match="reserved"):
        PipelineSourceInfo.from_bytes(encoded[:-4] + b"\x01\x00\x00\x00")
    with pytest.raises(ValueError, match="expected"):
        PipelineSourceInfo.from_bytes(encoded[:-1])

    word = pack_layout_range(0x12345678, 0x10203040)
    assert word == 0x1234567810203040
    assert unpack_layout_range(word) == (0x12345678, 0x10203040)
    with pytest.raises(OverflowError, match="end"):
        pack_layout_range(UINT32_MAX, 1)
    with pytest.raises(ValueError, match=r"\[0"):
        unpack_layout_range(UINT64_MAX + 1)


def test_each_bucket_has_exact_three_defer_one_none_cadence():
    dispatch = BucketKey(Direction.DISPATCH, bank=0, peer_rank=2, channel=5)
    combine = BucketKey(Direction.COMBINE, bank=0, peer_rank=2, channel=6)

    schedule = make_bucket_schedule(dispatch, 10)
    assert [post.slot for post in schedule.data_posts] == list(range(10))
    assert [post.flag for post in schedule.data_posts] == [
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.NONE,
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.NONE,
        PostFlag.DEFER,
        PostFlag.DEFER,
    ]
    assert schedule.publication.flag == PostFlag.NONE
    assert schedule.publication.value == 11

    # Cadence is local to the bucket, not a global counter shared by channels.
    assert [post.flag for post in make_bucket_schedule(combine, 4).data_posts] == [
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.DEFER,
        PostFlag.NONE,
    ]
    assert make_bucket_schedule(combine, 4).publication.value == 5


def test_empty_bucket_still_publishes_count_plus_one_non_deferred():
    key = BucketKey(Direction.DISPATCH, bank=1, peer_rank=0, channel=0)

    schedule = make_bucket_schedule(key, 0)

    assert schedule.record_count == 0
    assert schedule.data_posts == ()
    assert schedule.publication.value == 1
    assert schedule.publication.flag == PostFlag.NONE
    with pytest.raises(ValueError, match="record_count"):
        make_bucket_schedule(key, -1)


def test_direct_bucket_objects_cannot_bypass_wire_cadence():
    key = BucketKey(Direction.DISPATCH, bank=0, peer_rank=1, channel=0)

    with pytest.raises(ValueError, match="requires DEFER"):
        DataPost(0, PostFlag.NONE)
    with pytest.raises(TypeError, match="PostFlag"):
        DataPost(0, 1)
    with pytest.raises(ValueError, match="non-deferred"):
        PublicationPost(1, PostFlag.DEFER)
    with pytest.raises(ValueError, match="publication value"):
        PublicationPost(UINT32_MAX + 2)
    with pytest.raises(ValueError, match="contiguous"):
        BucketSchedule(
            key,
            (DataPost(1, PostFlag.DEFER),),
            PublicationPost(2),
        )
    with pytest.raises(ValueError, match=r"record_count \+ 1"):
        BucketSchedule(key, (), PublicationPost(2))


def test_sender_bank_reuse_waits_for_every_post_consume_credit():
    lifecycle = TwoBankCreditLifecycle(Direction.DISPATCH, max_ranks=4)
    operation_zero = OperationEpoch(5, 0)
    operation_one = OperationEpoch(5, 1)
    operation_two = OperationEpoch(5, 2)

    assert lifecycle.begin(operation_zero, (0, 2)) == 0
    assert lifecycle.begin(operation_one, (2,)) == 1
    lifecycle.mark_published(operation_zero, 0)
    lifecycle.return_credit(operation_zero, 0)  # Legal early credit.
    with pytest.raises(ProtocolError, match="duplicate credit"):
        lifecycle.return_credit(operation_zero, 0)
    with pytest.raises(ProtocolError, match="missing peers"):
        lifecycle.finish_publishing(operation_zero)

    lifecycle.mark_published(operation_zero, 2)
    lifecycle.finish_publishing(operation_zero)
    snapshot = lifecycle.snapshot(0)
    assert snapshot.phase == BankPhase.WAITING_CREDITS
    assert snapshot.credited_peers == frozenset({0})
    with pytest.raises(ProtocolError, match="still owned"):
        lifecycle.begin(operation_two, (0, 2))

    lifecycle.return_credit(operation_zero, 2)
    assert lifecycle.snapshot(0).phase == BankPhase.FREE
    assert lifecycle.last_completed(0) == operation_zero

    lifecycle.mark_published(operation_one, 2)
    lifecycle.finish_publishing(operation_one)
    lifecycle.return_credit(operation_one, 2)
    assert lifecycle.quiescent
    lifecycle.assert_quiescent()

    assert lifecycle.begin(operation_two, ()) == 0
    lifecycle.finish_publishing(operation_two)
    assert lifecycle.snapshot(0).phase == BankPhase.FREE
    assert lifecycle.last_completed(0) == operation_two
    with pytest.raises(ProtocolError, match="does not own bank"):
        lifecycle.return_credit(operation_zero, 2)


def test_sender_lifecycle_rejects_invalid_peer_and_operation_ordering():
    lifecycle = TwoBankCreditLifecycle(Direction.COMBINE, max_ranks=3)
    operation = OperationEpoch(2, 0)

    with pytest.raises(ValueError, match="duplicates"):
        lifecycle.begin(operation, (1, 1))
    lifecycle.begin(operation, (1,))
    with pytest.raises(ProtocolError, match="not expected"):
        lifecycle.mark_published(operation, 2)
    with pytest.raises(ProtocolError, match="before its publication"):
        lifecycle.return_credit(operation, 1)
    lifecycle.mark_published(operation, 1)
    with pytest.raises(ProtocolError, match="more than once"):
        lifecycle.mark_published(operation, 1)
    with pytest.raises(ProtocolError, match="outstanding credits"):
        lifecycle.assert_quiescent()
    lifecycle.finish_publishing(operation)
    lifecycle.return_credit(operation, 1)
    with pytest.raises(ProtocolError, match="does not advance"):
        lifecycle.begin(operation, (1,))


def test_membership_generation_change_requires_both_banks_quiescent():
    lifecycle = TwoBankCreditLifecycle(Direction.DISPATCH, max_ranks=2)
    old_zero = OperationEpoch(3, 0)
    old_one = OperationEpoch(3, 1)

    lifecycle.begin(old_zero, (1,))
    lifecycle.begin(old_one, (1,))
    lifecycle.mark_published(old_zero, 1)
    lifecycle.finish_publishing(old_zero)
    lifecycle.return_credit(old_zero, 1)
    assert lifecycle.snapshot(0).phase == BankPhase.FREE
    assert lifecycle.snapshot(1).phase == BankPhase.PUBLISHING

    with pytest.raises(ProtocolError, match="quiescent bank boundary"):
        lifecycle.begin(OperationEpoch(4, 0), (1,))

    lifecycle.mark_published(old_one, 1)
    lifecycle.finish_publishing(old_one)
    lifecycle.return_credit(old_one, 1)
    assert lifecycle.begin(OperationEpoch(4, 0), ()) == 0


def test_receiver_credit_follows_acquire_consume_and_ready_reset():
    operation = OperationEpoch(9, 4)
    receiver = ReceiveCreditLifecycle(
        Direction.DISPATCH,
        operation,
        source_rank=2,
        channels=(0, 1),
    )

    assert receiver.observe_publication(0, 1) == 0  # Empty channel publication.
    assert receiver.snapshot().phase == ReceivePhase.WAITING_PUBLICATIONS
    with pytest.raises(ProtocolError, match="not every"):
        _ = receiver.total_records
    with pytest.raises(ProtocolError, match="all channel publications"):
        receiver.mark_consumed()

    assert receiver.observe_publication(1, 4) == 3
    assert receiver.total_records == 3
    assert receiver.snapshot().phase == ReceivePhase.READY_TO_CONSUME
    with pytest.raises(ProtocolError, match="duplicate publication"):
        receiver.observe_publication(1, 4)

    receiver.mark_consumed()
    with pytest.raises(ProtocolError, match="ready-word resets"):
        receiver.issue_credit()
    receiver.mark_ready_reset(0)
    with pytest.raises(ProtocolError, match="reset twice"):
        receiver.mark_ready_reset(0)
    receiver.mark_ready_reset(1)
    receiver.issue_credit()
    snapshot = receiver.snapshot()
    assert snapshot.phase == ReceivePhase.CREDIT_ISSUED
    assert snapshot.observed_counts == {0: 0, 1: 3}
    assert snapshot.reset_channels == frozenset({0, 1})
    with pytest.raises(ProtocolError, match="only after consumption"):
        receiver.issue_credit()


def test_receiver_lifecycle_rejects_missing_or_unexpected_channels():
    operation = OperationEpoch(1, 0)
    with pytest.raises(ValueError, match="at least one channel"):
        ReceiveCreditLifecycle(Direction.COMBINE, operation, 0, ())
    with pytest.raises(ValueError, match="duplicates"):
        ReceiveCreditLifecycle(Direction.COMBINE, operation, 0, (3, 3))

    receiver = ReceiveCreditLifecycle(Direction.COMBINE, operation, 0, (3,))
    with pytest.raises(ProtocolError, match="unexpected channel"):
        receiver.observe_publication(4, 1)
    with pytest.raises(ValueError, match="publication value"):
        receiver.observe_publication(3, 0)
    with pytest.raises(ValueError, match="publication value"):
        receiver.observe_publication(3, UINT32_MAX + 2)
