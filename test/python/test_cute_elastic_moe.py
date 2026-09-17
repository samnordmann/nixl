# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import examples.python.cute.elastic_moe as elastic_moe
from examples.python.cute.elastic_moe import (
    DISPATCH,
    RoundMetrics,
    _ElasticTransport,
    decode_combine_slab,
    decode_dispatch_slab,
    encode_combine_slab,
    encode_dispatch_slab,
    parse_membership_plan,
    summarize_phase,
    validate_example_configuration,
)
from examples.python.cute.moe import (
    ElasticExpertTopology,
    ExpertOutput,
    PeerSlabLayout,
    pack_dispatch,
    route_topk,
)


def _layout() -> PeerSlabLayout:
    return PeerSlabLayout(
        max_ranks=2,
        max_tokens_per_rank=2,
        top_k=1,
        hidden_size=4,
        element_size=2,
    )


def _routing_and_records():
    topology = ElasticExpertTopology(2, 1, (0, 1), generation=0)
    tokens = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=torch.bfloat16,
    )
    scores = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    routing = route_topk(scores, topology, origin_rank=0, top_k=1)
    records = pack_dispatch(tokens, routing, topology, _layout())
    return topology, tokens, routing, records


def test_membership_plan_matches_sparse_nixl_ep_semantics():
    phases = parse_membership_plan(
        [[0, 1], [3, 0, 2], [0, 3], [0, 1, 2, 3]], max_ranks=4
    )

    assert phases == ((0, 1), (0, 2, 3), (0, 3), (0, 1, 2, 3))
    with pytest.raises(ValueError, match="killed-rank marker"):
        parse_membership_plan([[0, -1]], max_ranks=2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_membership_plan([[0, 0]], max_ranks=2)
    with pytest.raises(ValueError, match="fixed capacity"):
        parse_membership_plan([[0, 2]], max_ranks=2)


def test_configuration_rejects_invalid_route_capacity():
    validate_example_configuration(
        devices=(0, 1),
        plan=((0,), (0, 1)),
        experts_per_rank=2,
        top_k=2,
        num_tokens=4,
        hidden_size=8,
        warmup=0,
        iterations=1,
    )

    with pytest.raises(ValueError, match="active experts"):
        validate_example_configuration(
            devices=(0, 1),
            plan=((0,),),
            experts_per_rank=1,
            top_k=2,
            num_tokens=4,
            hidden_size=8,
            warmup=0,
            iterations=1,
        )


def test_dispatch_wire_round_trip_including_all_zero_first_header():
    topology, tokens, _routing, records = _routing_and_records()
    rank_zero_records = tuple(
        record for record in records if record.destination_rank == 0
    )

    slab = encode_dispatch_slab(
        rank_zero_records,
        _layout(),
        origin_rank=0,
        destination_rank=0,
        generation=0,
    )
    decoded = decode_dispatch_slab(
        slab,
        _layout(),
        origin_rank=0,
        destination_rank=0,
        generation=0,
        payload_dtype=torch.bfloat16,
    )

    assert len(decoded) == 1
    assert decoded[0].metadata == rank_zero_records[0].metadata
    assert torch.equal(decoded[0].payload, tokens[0])
    with pytest.raises(ValueError, match="generation"):
        decode_dispatch_slab(
            slab,
            _layout(),
            origin_rank=0,
            destination_rank=0,
            generation=1,
            payload_dtype=torch.bfloat16,
        )


def test_empty_and_combine_slabs_are_unambiguous():
    layout = _layout()
    empty = encode_dispatch_slab(
        (), layout, origin_rank=1, destination_rank=0, generation=7
    )
    assert (
        decode_dispatch_slab(
            empty,
            layout,
            origin_rank=1,
            destination_rank=0,
            generation=7,
            payload_dtype=torch.bfloat16,
        )
        == ()
    )
    with pytest.raises(ValueError, match="expected destination"):
        decode_dispatch_slab(
            empty,
            layout,
            origin_rank=1,
            destination_rank=1,
            generation=7,
            payload_dtype=torch.bfloat16,
        )

    topology, _tokens, _routing, records = _routing_and_records()
    output = ExpertOutput(records[1].metadata, records[1].payload + 2)
    slab = encode_combine_slab(
        (output,),
        layout,
        source_rank=1,
        destination_rank=0,
        generation=topology.generation,
    )
    restored = decode_combine_slab(
        slab,
        layout,
        source_rank=1,
        destination_rank=0,
        generation=topology.generation,
        payload_dtype=torch.bfloat16,
    )
    assert restored[0].metadata == output.metadata
    assert torch.equal(restored[0].payload, output.payload)

    with pytest.raises(IndexError, match="peer_rank"):
        encode_dispatch_slab(
            (), layout, origin_rank=0, destination_rank=2, generation=0
        )
    with pytest.raises(ValueError, match="payload dtype"):
        decode_dispatch_slab(
            empty,
            layout,
            origin_rank=1,
            destination_rank=0,
            generation=7,
            payload_dtype=torch.float32,
        )


def test_phase_summary_uses_cross_rank_critical_path():
    topology = ElasticExpertTopology(2, 1, (0, 1), generation=3)
    rank_samples = {
        0: (
            RoundMetrics(0, 0.002, 0.0005, 0.0005, 2, 100, 200, 0.0),
            RoundMetrics(0, 0.004, 0.001, 0.001, 2, 100, 200, 0.0),
        ),
        1: (
            RoundMetrics(1, 0.003, 0.0005, 0.0005, 2, 100, 200, 0.0),
            RoundMetrics(1, 0.001, 0.0005, 0.0005, 2, 100, 200, 0.0),
        ),
    }

    result = summarize_phase(
        topology=topology,
        rank_samples=rank_samples,
        num_tokens=8,
        stage_s_by_rank={0: 0.010, 1: 0.020},
        commit_s_by_rank={0: 0.001, 1: 0.002},
    )

    assert result["critical_path_samples_s"] == [0.003, 0.004]
    assert result["stage_max_ms"] == pytest.approx(20)
    assert result["commit_max_ms"] == pytest.approx(2)
    assert result["logical_remote_GBps"] > 0


def test_transport_teardown_api_enforces_remote_then_metadata_then_local():
    events = []

    class View:
        def __init__(self, name):
            self.name = name

        def release(self):
            events.append(("release", self.name))

    class Stream:
        def synchronize(self):
            events.append(("synchronize",))

    class Agent:
        def remove_remote_agent(self, name):
            events.append(("remove", name))

        def deregister_memory(self, registration, *, backends):
            events.append(("deregister", registration, tuple(backends)))

    transport = object.__new__(_ElasticTransport)
    transport.stream = Stream()
    transport._staged_views = {3: View("staged")}
    transport._active_views = {1: View("active")}
    transport._loaded = {1, 3}
    transport.agent = Agent()
    transport.control = SimpleNamespace(directory=Path("/tmp/elastic-run"))
    transport._local_view = View("local")
    transport._registration = "registration"
    transport._remote_views_released = False
    transport._metadata_invalidated = False
    transport._local_resources_released = False
    transport._closed = False

    with pytest.raises(RuntimeError, match="release remote views"):
        transport.invalidate_remote_metadata()
    transport.release_remote_views()
    transport.invalidate_remote_metadata()
    transport.release_local_resources()

    assert events == [
        ("synchronize",),
        ("release", "staged"),
        ("release", "active"),
        ("remove", "cute_elastic_elastic-run_1"),
        ("remove", "cute_elastic_elastic-run_3"),
        ("release", "local"),
        ("deregister", "registration", ("UCX",)),
    ]


def test_remote_publish_uses_committed_device_rank_mask(monkeypatch):
    calls = []

    class Stream:
        def synchronize(self):
            calls.append(("synchronize",))

    def launch(local, remote, rank_mask, statuses, stream, **kwargs):
        calls.append(("launch", local, remote, rank_mask, kwargs))
        statuses[:] = 0

    monkeypatch.setattr(
        elastic_moe, "launch_masked_thread_put_then_signal_host", launch
    )
    transport = object.__new__(_ElasticTransport)
    transport.rank = 0
    transport.max_ranks = 2
    transport.layout = _layout()
    transport.dispatch_send = object()
    transport.combine_send = object()
    transport.dispatch_recv = object()
    transport.combine_recv = object()
    transport._local_view = object()
    transport._active_views = {1: object()}
    transport.rank_mask = object()
    transport.statuses = torch.zeros(2, dtype=torch.int32)
    transport.stream = Stream()

    transport.publish(DISPATCH, (1,))

    launch_call = calls[0]
    assert launch_call[1] is transport._local_view
    assert launch_call[2] is transport._active_views[1]
    assert launch_call[3] is transport.rank_mask
    assert launch_call[4]["destination_rank"] == 1
    assert calls[1] == ("synchronize",)
