# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

_CUTE_EXAMPLES = Path(__file__).parents[2] / "examples" / "python" / "cute"
sys.path.insert(0, str(_CUTE_EXAMPLES))

from moe import (  # noqa: E402
    DISPATCH_HEADER_NBYTES,
    SLAB_PREAMBLE_WIRE_NBYTES,
    DispatchMetadata,
    ElasticExpertTopology,
    PeerSlabLayout,
    SlabPreamble,
    align_up,
    distributed_moe_reference,
    pack_dispatch,
    process_expert_batches,
    route_topk,
    unpack_dispatch,
    weighted_combine,
)


def _topology(generation: int = 7) -> ElasticExpertTopology:
    return ElasticExpertTopology(
        max_ranks=4,
        experts_per_rank=2,
        active_ranks=(0, 2),
        generation=generation,
    )


def _layout(*, top_k: int = 2) -> PeerSlabLayout:
    return PeerSlabLayout(
        max_ranks=4,
        max_tokens_per_rank=3,
        top_k=top_k,
        hidden_size=5,
        element_size=4,
    )


def test_elastic_topology_preserves_sparse_rank_and_expert_ids():
    topology = _topology()

    assert topology.active_mask == (True, False, True, False)
    assert topology.nixl_mask == (0, 1, 0, 1)
    assert topology.rank_bound == 3
    assert topology.active_experts == (0, 1, 4, 5)
    assert topology.expert_id(2, 1) == 5
    assert topology.owner(5) == 2

    changed = topology.reconfigure((1, 2))
    assert changed.generation == 8
    assert changed.active_experts == (2, 3, 4, 5)
    assert changed.expert_id(2, 1) == topology.expert_id(2, 1) == 5


def test_topology_from_mask_and_validation():
    topology = ElasticExpertTopology.from_mask(
        max_ranks=4,
        experts_per_rank=1,
        active_mask=(True, False, True, False),
    )
    assert topology.active_ranks == (0, 2)

    with pytest.raises(ValueError, match="at least one"):
        ElasticExpertTopology(2, 1, ())
    with pytest.raises(ValueError, match="duplicates"):
        ElasticExpertTopology(2, 1, (0, 0))
    with pytest.raises(ValueError, match="outside"):
        ElasticExpertTopology(2, 1, (2,))
    with pytest.raises(TypeError, match="bool"):
        ElasticExpertTopology.from_mask(
            max_ranks=2, experts_per_rank=1, active_mask=(1, False)
        )
    with pytest.raises(ValueError, match="advance"):
        topology.reconfigure((0,), generation=0)
    with pytest.raises(OverflowError, match="expert ID"):
        ElasticExpertTopology(1 << 31, 3, (0,))


def test_route_topk_ignores_holes_and_breaks_ties_by_stable_expert_id():
    topology = _topology()
    scores = torch.zeros((2, topology.max_experts), dtype=torch.float32)
    scores[0, 2] = 1_000  # rank 1 is inactive
    scores[0, 4] = 2
    scores[1, 6] = 1_000  # rank 3 is inactive
    scores[1, 5] = 3

    route = route_topk(scores, topology, origin_rank=0, top_k=3)

    assert route.expert_indices.tolist() == [[4, 0, 1], [5, 0, 1]]
    assert all(len(set(row)) == 3 for row in route.expert_indices.tolist())
    torch.testing.assert_close(route.weights.sum(dim=1), torch.ones(2))


def test_route_topk_rejects_unsafe_inputs():
    topology = _topology()
    scores = torch.zeros((1, topology.max_experts))

    with pytest.raises(ValueError, match="inactive"):
        route_topk(scores, topology, origin_rank=1, top_k=1)
    with pytest.raises(ValueError, match="exceeds"):
        route_topk(scores, topology, origin_rank=0, top_k=5)
    with pytest.raises(ValueError, match="fixed capacity"):
        route_topk(scores[:, :-1], topology, origin_rank=0, top_k=1)
    scores[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        route_topk(scores, topology, origin_rank=0, top_k=1)


def test_peer_slab_layout_alignment_and_nonoverlap():
    layout = _layout()

    assert layout.record_capacity == 6
    assert layout.header_stride == 32
    assert layout.payload_stride == 128
    assert layout.arena_nbytes == layout.max_ranks * layout.slab_nbytes
    assert layout.slab_nbytes % layout.slab_alignment == 0

    previous_end = 0
    for rank in range(layout.max_ranks):
        slab = layout.slab(rank)
        assert slab.offset == previous_end
        assert slab.offset % layout.slab_alignment == 0
        assert slab.preamble_offset == slab.offset
        assert slab.preamble_nbytes == layout.preamble_size
        assert slab.headers_offset >= slab.preamble_offset + slab.preamble_nbytes
        assert slab.payloads_offset % layout.payload_alignment == 0
        for slot in range(layout.record_capacity):
            record = layout.record(rank, slot)
            assert record.header_offset % layout.header_alignment == 0
            assert record.payload_offset % layout.payload_alignment == 0
            assert record.header_end <= slab.payloads_offset
            assert record.payload_end <= slab.end
        previous_end = slab.end
    assert previous_end == layout.arena_nbytes


def test_layout_bounds_and_overflow_fail_closed():
    layout = _layout()
    layout.validate_span(layout.arena_nbytes - 1, 1)
    with pytest.raises(ValueError, match="exceeds"):
        layout.validate_span(layout.arena_nbytes, 1)
    with pytest.raises(IndexError, match="peer_rank"):
        layout.slab(layout.max_ranks)
    with pytest.raises(IndexError, match="slot"):
        layout.record(0, layout.record_capacity)
    with pytest.raises(ValueError, match="power of two"):
        align_up(3, 3)
    with pytest.raises(OverflowError, match="layout limit"):
        PeerSlabLayout(
            max_ranks=2,
            max_tokens_per_rank=8,
            top_k=4,
            hidden_size=32,
            element_size=4,
            slab_alignment=128,
            address_limit=1024,
        )


def test_dispatch_header_round_trip_and_padding_validation():
    metadata = DispatchMetadata(
        generation=2**40,
        origin_rank=3,
        origin_token=17,
        route_slot=2,
        expert=99,
    )
    encoded = metadata.to_bytes(32)

    assert len(encoded) == 32
    assert DispatchMetadata.from_bytes(encoded) == metadata
    assert encoded[DISPATCH_HEADER_NBYTES:] == bytes(8)
    with pytest.raises(ValueError, match="padding"):
        DispatchMetadata.from_bytes(encoded[:-1] + b"\x01")
    with pytest.raises(ValueError, match="generation"):
        DispatchMetadata(-1, 0, 0, 0, 0)


def test_slab_preamble_makes_empty_and_all_zero_headers_unambiguous():
    layout = _layout()
    encoded = layout.encode_preamble(
        origin_rank=0, destination_rank=2, generation=0, record_count=1
    )

    assert len(encoded) == layout.preamble_size
    assert len(encoded) == SLAB_PREAMBLE_WIRE_NBYTES
    assert layout.decode_preamble(
        encoded,
        expected_origin_rank=0,
        expected_destination_rank=2,
        expected_generation=0,
    ) == SlabPreamble(
        generation=0,
        record_count=1,
        origin_rank=0,
        destination_rank=2,
        element_size=layout.element_size,
    )

    empty = layout.encode_preamble(
        origin_rank=0, destination_rank=2, generation=0, record_count=0
    )
    assert layout.decode_preamble(empty).record_count == 0
    with pytest.raises(ValueError, match="magic"):
        layout.decode_preamble(bytes(layout.preamble_size))
    with pytest.raises(ValueError, match="capacity"):
        layout.encode_preamble(
            origin_rank=0,
            destination_rank=2,
            generation=0,
            record_count=layout.record_capacity + 1,
        )
    with pytest.raises(ValueError, match="expected generation"):
        layout.decode_preamble(encoded, expected_generation=1)
    with pytest.raises(ValueError, match="expected destination"):
        layout.decode_preamble(encoded, expected_destination_rank=1)

    float_layout = PeerSlabLayout(
        max_ranks=3,
        max_tokens_per_rank=2,
        top_k=2,
        hidden_size=5,
        element_size=2,
    )
    with pytest.raises(ValueError, match="element size"):
        float_layout.decode_preamble(encoded)


def test_dispatch_pack_uses_origin_slab_and_dense_destination_slots():
    topology = _topology()
    layout = _layout()
    tokens = torch.arange(15, dtype=torch.float32).view(3, 5)
    scores = torch.zeros((3, topology.max_experts))
    scores[:, 4] = 4
    scores[:, 0] = 3
    routing = route_topk(scores, topology, origin_rank=2, top_k=2)

    records = pack_dispatch(tokens, routing, topology, layout)

    assert len(records) == 6
    assert [record.destination_rank for record in records] == [2, 0] * 3
    assert [record.slot for record in records if record.destination_rank == 2] == [
        0,
        1,
        2,
    ]
    assert [record.slot for record in records if record.destination_rank == 0] == [
        0,
        1,
        2,
    ]
    for record in records:
        assert record.location == layout.record(2, record.slot)
        assert record.metadata.generation == topology.generation
        assert record.metadata.origin_rank == 2
        assert (
            record.metadata.expert
            == routing.expert_indices[
                record.metadata.origin_token, record.metadata.route_slot
            ]
        )


def test_unpack_orders_by_origin_and_exposes_sparse_origin_ranges():
    topology = _topology()
    layout = _layout(top_k=1)
    incoming = []
    for origin in topology.active_ranks:
        tokens = torch.full((2, 5), float(origin + 1))
        scores = torch.zeros((2, topology.max_experts))
        scores[:, 4] = 10
        routing = route_topk(scores, topology, origin_rank=origin, top_k=1)
        incoming.extend(pack_dispatch(tokens, routing, topology, layout))

    batches = unpack_dispatch(
        list(reversed(incoming)), topology, layout, destination_rank=2
    )
    expert_four = next(batch for batch in batches if batch.expert == 4)

    assert [metadata.origin_rank for metadata in expert_four.metadata] == [0, 0, 2, 2]
    assert expert_four.origin_ranges == ((0, 2), (2, 0), (2, 2), (4, 0))
    torch.testing.assert_close(
        expert_four.payloads,
        torch.tensor([[1.0] * 5, [1.0] * 5, [3.0] * 5, [3.0] * 5]),
    )


def test_unpack_rejects_stale_duplicate_and_wrong_location_records():
    topology = _topology()
    layout = _layout(top_k=1)
    tokens = torch.ones((1, 5))
    scores = torch.zeros((1, topology.max_experts))
    scores[0, 4] = 1
    routing = route_topk(scores, topology, origin_rank=0, top_k=1)
    record = pack_dispatch(tokens, routing, topology, layout)[0]

    stale = replace(
        record,
        metadata=replace(record.metadata, generation=topology.generation - 1),
    )
    with pytest.raises(ValueError, match="stale"):
        unpack_dispatch([stale], topology, layout, destination_rank=2)
    with pytest.raises(ValueError, match="duplicate dispatch"):
        unpack_dispatch([record, record], topology, layout, destination_rank=2)
    wrong_location = replace(record, location=layout.record(2, record.slot))
    with pytest.raises(ValueError, match="byte location"):
        unpack_dispatch([wrong_location], topology, layout, destination_rank=2)


def test_full_distributed_reference_matches_independent_weighted_formula():
    topology = ElasticExpertTopology(
        max_ranks=3,
        experts_per_rank=2,
        active_ranks=(0, 2),
        generation=11,
    )
    layout = PeerSlabLayout(
        max_ranks=3,
        max_tokens_per_rank=2,
        top_k=2,
        hidden_size=3,
        element_size=4,
    )
    tokens = {
        0: torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        2: torch.tensor([[3.0, 1.0, 2.0], [5.0, 2.0, 1.0]]),
    }
    scores = {
        0: torch.tensor([[4.0, 1.0, 99.0, 98.0, 3.0, 2.0]] * 2),
        2: torch.tensor([[1.0, 2.0, 99.0, 98.0, 3.0, 4.0]] * 2),
    }

    result = distributed_moe_reference(tokens, scores, topology, layout, top_k=2)

    assert set(result.combined) == {0, 2}
    assert all(
        record.destination_rank in topology.active_ranks
        for record in result.dispatch_records
    )
    for origin, routing in result.routings.items():
        expected = torch.zeros_like(tokens[origin])
        for token in range(routing.num_tokens):
            for slot in range(routing.top_k):
                expert = int(routing.expert_indices[token, slot])
                scale = float(expert + 1)
                bias = float((expert % 17) - 8) / 32.0
                transformed = tokens[origin][token] * scale + bias
                expected[token] += transformed * routing.weights[token, slot]
        torch.testing.assert_close(result.combined[origin], expected)


def test_combine_is_arrival_order_independent_and_detects_missing_routes():
    topology = _topology()
    layout = _layout()
    tokens = torch.arange(10, dtype=torch.float32).view(2, 5)
    scores = torch.zeros((2, topology.max_experts))
    scores[:, 0] = 2
    scores[:, 4] = 1
    routing = route_topk(scores, topology, origin_rank=0, top_k=2)
    records = pack_dispatch(tokens, routing, topology, layout)
    outputs = []
    for destination in topology.active_ranks:
        batches = unpack_dispatch(
            [record for record in records if record.destination_rank == destination],
            topology,
            layout,
            destination_rank=destination,
        )
        outputs.extend(process_expert_batches(batches))

    forward = weighted_combine(outputs, routing, topology, hidden_size=5)
    reverse = weighted_combine(
        list(reversed(outputs)), routing, topology, hidden_size=5
    )
    torch.testing.assert_close(forward, reverse, rtol=0, atol=0)

    with pytest.raises(ValueError, match="missing"):
        weighted_combine(outputs[:-1], routing, topology, hidden_size=5)
    partial = weighted_combine(
        outputs[:-1],
        routing,
        topology,
        hidden_size=5,
        require_complete=False,
    )
    assert torch.isfinite(partial).all()


def test_pack_rejects_stale_routing_and_capacity_mismatch():
    topology = _topology()
    layout = _layout(top_k=1)
    tokens = torch.ones((1, 5))
    scores = torch.zeros((1, topology.max_experts))
    routing = route_topk(scores, topology, origin_rank=0, top_k=1)

    with pytest.raises(ValueError, match="stale"):
        pack_dispatch(tokens, routing, topology.reconfigure((0, 2)), layout)
    with pytest.raises(ValueError, match="routing has"):
        pack_dispatch(torch.ones((2, 5)), routing, topology, layout)
    with pytest.raises(ValueError, match="element size"):
        pack_dispatch(tokens.to(torch.float64), routing, topology, layout)
    with pytest.raises(TypeError, match="floating dtype"):
        pack_dispatch(tokens.to(torch.int32), routing, topology, layout)
    with pytest.raises(ValueError, match="contiguous"):
        pack_dispatch(torch.ones((1, 10))[:, ::2], routing, topology, layout)
