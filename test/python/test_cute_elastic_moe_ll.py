# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import shlex
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from examples.python.cute._runtime import DeviceRegion, PeerCoordinates
from examples.python.cute.elastic_moe_ll import (
    OUTPUT_POISON,
    WORKER_PLAN_ELEMENT_NBYTES,
    WORKER_PLAN_FIELDS,
    ElasticLLCase,
    PhaseSpec,
    _classify_mapped_preflight,
    _contains_unreleased_view,
    _decode_phase_failures,
    _decode_phase_result_exchange,
    _decode_rank_results,
    _DeviceViewExitStack,
    _phase_result_payload,
    _phase_status_payload,
    _rank_phase_result,
    _RecoveredDeviceViewReleaseError,
    _require_diagnostic_timeout_margin,
    _timestamp_buffer_elements,
    _UnreleasedDeviceViewError,
    _validate_outputs,
    bank_cycle_schedule,
    build_phase_route_state,
    build_phase_specs,
    build_routes,
    cumulative_publication_value,
    expected_outputs,
    make_phase_arena,
    make_phase_dispatch_stage,
    parse_membership_plan,
    phase_incarnation_tables,
    summarize_phase_results,
)
from examples.python.cute.moe.ll_protocol import CompactLLArenaLayout, LLArenaLayout


def _case() -> ElasticLLCase:
    plan = parse_membership_plan("0,2;0;0,2;0,1,2", 3)
    return ElasticLLCase(
        max_ranks=3,
        experts_per_rank=2,
        num_tokens=4,
        top_k=2,
        hidden_size=8,
        warmup=2,
        iterations=3,
        membership_plan=plan,
        empty_last_expert=True,
        timing_mode="cadence",
    )


def _native_atomic_evidence(
    devices: tuple[int, ...], accessing_devices: tuple[int, ...] | None = None
) -> dict[str, object]:
    accessors = devices if accessing_devices is None else accessing_devices
    return {
        "schema_version": 1,
        "capability": "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED",
        "devices": list(devices),
        "device_identities": [
            {"device_ordinal": device, "pci_bus_id": f"0000:{device:02x}:00.0"}
            for device in devices
        ],
        "cuda_visible_devices": ",".join(str(device) for device in devices),
        "accessing_devices": list(accessors),
        "ordered_pairs": [
            {
                "accessing_device": accessing,
                "owner_device": owner,
                "native_atomics_supported": True,
            }
            for accessing in accessors
            for owner in devices
            if accessing != owner
        ],
        "all_supported": True,
        "query_scope": (
            "selected accessing devices to every distinct participating owner device"
        ),
        "execution_scope": "host preflight only; no steady-state device-path cost",
        "driver_module": "/cuda/bindings/driver.py",
    }


def _valid_summary_rows(
    case: ElasticLLCase, phase: PhaseSpec
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(case.max_ranks):
        active = rank in phase.active_ranks
        rows.append(
            {
                "rank": rank,
                "active": active,
                "rank_round_starts_ns": (
                    [1000, 2000, 3000]
                    if rank == 0
                    else ([5000, 6000, 7000] if active else [])
                ),
                "rank_round_ends_ns": (
                    [0, 0, 3400] if rank == 0 else ([0, 0, 7410] if active else [])
                ),
                "rank_observer_span_ns": [],
                "peer_round_ns": [],
                "remote_peer_round_ns": [],
                "per_peer_round_ns": {},
                "operation_steps": case.iterations,
                "timing_mode": case.timing_mode,
                "environment": (
                    {
                        "device": rank,
                        "native_peer_atomics": _native_atomic_evidence(
                            tuple(range(case.max_ranks)), (rank,)
                        ),
                        "mapped_preflight": {
                            "policy": "require_non_null_process_local_pointer",
                            "evidence_scope": (
                                "exact timed-phase device view and arena allocation"
                            ),
                            "rank_active": True,
                            "remote_peers": {
                                str(peer): {
                                    "classification": "owner_va_coincident",
                                    "status": 0,
                                }
                                for peer in phase.active_ranks
                                if peer != rank
                            },
                            "all_active_remote_peers_mapped": True,
                            "translated_va_observed": False,
                            "legacy_allow_unverified_mapped_ignored": False,
                        },
                    }
                    if active
                    else {}
                ),
                "correctness": "PASS",
            }
        )
    return rows


def test_membership_parser_preserves_sparse_stable_slots():
    assert parse_membership_plan("2,0;0;2,0", 3) == ((0, 2), (0,), (0, 2))
    with pytest.raises(ValueError, match="duplicates"):
        parse_membership_plan("0,0", 2)
    with pytest.raises(ValueError, match="fixed rank capacity"):
        parse_membership_plan("0,2", 2)
    with pytest.raises(ValueError, match="at least one"):
        parse_membership_plan("0;", 2)


def test_incarnation_advances_only_on_rejoin():
    plan = ((0, 2), (0,), (0, 2), (0, 1, 2), (0, 1, 2))
    assert phase_incarnation_tables(plan, 3) == (
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 2),
        (1, 1, 2),
    )
    specs = build_phase_specs(_case())
    assert [spec.generation for spec in specs] == [0, 1, 2, 3]
    assert specs[2].rank_incarnations == (1, 0, 2)
    assert specs[3].rank_mask == (0, 0, 0)


def test_case_builds_checked_fixed_capacity_single_bank_layout_by_default():
    case = _case()
    layout = case.layout
    assert isinstance(layout, CompactLLArenaLayout)
    assert case.route_capacity == 8
    assert case.max_tokens_per_rank == 8
    assert layout.route_capacity == case.route_capacity
    assert tuple(region.name for region in layout.regions) == (
        "dispatch_stage",
        "dispatch_recv",
        "dispatch_stamp",
        "dispatch_ready",
        "combine_consumed",
        "abort_state",
        "combine_recv",
        "combine_stamp",
        "combine_ready",
    )
    assert layout.num_banks == 1
    with pytest.raises(IndexError, match="bank"):
        layout.dispatch_receive_record(1, 0, 0, 0)
    assert layout.abort_state_word().nbytes == 8

    legacy = LLArenaLayout(
        max_ranks=case.max_ranks,
        experts_per_rank=case.experts_per_rank,
        max_tokens_per_rank=case.route_capacity,
        top_k=case.top_k,
        hidden_size=case.hidden_size,
        element_size=2,
    )
    assert layout.arena_nbytes < legacy.arena_nbytes

    with pytest.raises(ValueError, match="positive integer"):
        ElasticLLCase(2, 1, 4, 0, 8, 0, 1, ((0, 1),), True)
    with pytest.raises(ValueError, match="active experts"):
        ElasticLLCase(2, 2, 4, 3, 8, 0, 1, ((0,),), True)
    with pytest.raises(ValueError, match="warp size"):
        ElasticLLCase(2, 17, 1, 33, 8, 0, 1, ((0, 1),), False)
    with pytest.raises(ValueError, match="sorted"):
        ElasticLLCase(3, 2, 4, 2, 8, 0, 1, ((2, 0),), True)
    with pytest.raises(ValueError, match="device_timeout_ns"):
        ElasticLLCase(2, 2, 4, 2, 8, 0, 1, ((0, 1),), True, -1)
    assert ElasticLLCase(2, 2, 4, 2, 8, 0, 1, ((0, 1),), True, 0).device_timeout_ns == 0
    with pytest.raises(ValueError, match="device_timeout_ns"):
        ElasticLLCase(
            2,
            2,
            4,
            2,
            8,
            0,
            1,
            ((0, 1),),
            device_timeout_ns=1 << 63,
        )
    for invalid_target_sm in (True, 0, -1, 100.0):
        with pytest.raises(ValueError, match="target_sm"):
            ElasticLLCase(
                2,
                2,
                4,
                2,
                8,
                0,
                1,
                ((0, 1),),
                target_sm=invalid_target_sm,
            )
    with pytest.raises(ValueError, match="timing_mode"):
        ElasticLLCase(2, 2, 4, 2, 8, 0, 1, ((0, 1),), timing_mode="sampled")
    assert (
        ElasticLLCase(
            2, 2, 4, 2, 8, 0, 1, ((0, 1),), instrument_per_peer=True
        ).timing_mode
        == "peer"
    )
    assert (
        ElasticLLCase(
            2, 2, 4, 2, 8, 0, 1, ((0, 1),), timing_mode="peer"
        ).instrument_per_peer
        is True
    )
    with pytest.raises(ValueError, match="conflicts"):
        ElasticLLCase(
            2,
            2,
            4,
            2,
            8,
            0,
            1,
            ((0, 1),),
            instrument_per_peer=True,
            timing_mode="cadence",
        )
    with pytest.raises(ValueError, match="num_banks"):
        ElasticLLCase(2, 2, 4, 2, 8, 0, 1, ((0, 1),), num_banks=3)
    with pytest.raises(ValueError, match="divisible by 8"):
        ElasticLLCase(2, 2, 4, 2, 10, 0, 1, ((0, 1),), True)
    with pytest.raises(ValueError, match="uint32 route capacity"):
        ElasticLLCase(2, 2, 1 << 31, 2, 8, 0, 1, ((0, 1),), True)
    with pytest.raises(ValueError, match="Int32 range"):
        ElasticLLCase(2, 2, 1 << 31, 1, 8, 0, 1, ((0, 1),), True)
    with pytest.raises(ValueError, match="wrap-safe uint32"):
        ElasticLLCase(
            1,
            1,
            (1 << 32) - 63,
            1,
            8,
            0,
            1,
            ((0,),),
            False,
            workers_per_peer=128,
        )


def test_auto_worker_grid_preserves_parallelism_and_expert_coverage():
    case = _case()
    assert case.workers_per_peer == 2
    assert case.max_ranks * case.workers_per_peer == 6
    assert case.warps_per_cta == 6
    large = ElasticLLCase(8, 4, 128, 8, 7168, 2, 3, (tuple(range(8)),))
    assert large.workers_per_peer == 16
    assert large.max_ranks * large.workers_per_peer == 128
    assert large.warps_per_cta == 8
    with pytest.raises(ValueError, match="cover every fixed local expert"):
        ElasticLLCase(2, 4, 4, 2, 8, 0, 1, ((0, 1),), workers_per_peer=3)
    with pytest.raises(ValueError, match="divisible by warps_per_cta"):
        ElasticLLCase(
            2,
            2,
            4,
            2,
            8,
            0,
            1,
            ((0, 1),),
            workers_per_peer=3,
            warps_per_cta=4,
        )


@pytest.mark.parametrize(
    ("fixed_worker_count", "expected_warps_per_cta"),
    ((1, 1), (6, 6), (7, 7), (30, 6), (96, 8), (128, 8)),
)
def test_auto_warps_per_cta_keeps_portable_policy_without_qualified_sm(
    fixed_worker_count, expected_warps_per_cta
):
    case = ElasticLLCase(
        1,
        1,
        1,
        1,
        8,
        0,
        1,
        ((0,),),
        workers_per_peer=fixed_worker_count,
    )
    assert case.warps_per_cta == expected_warps_per_cta
    assert fixed_worker_count % case.warps_per_cta == 0


def test_auto_warps_per_cta_uses_only_exact_qualified_sm100_signature():
    qualified = ElasticLLCase(
        2,
        4,
        128,
        8,
        7168,
        2,
        3,
        ((0, 1),),
        empty_last_expert=False,
        workers_per_peer=64,
        target_sm=100,
    )
    assert qualified.warps_per_cta == 2

    assert replace(qualified, target_sm=90, warps_per_cta=0).warps_per_cta == 8
    assert replace(qualified, hidden_size=128, warps_per_cta=0).warps_per_cta == 8
    assert (
        replace(
            qualified, membership_plan=((0, 1), (0, 1)), warps_per_cta=0
        ).warps_per_cta
        == 8
    )
    assert (
        replace(qualified, empty_last_expert=True, warps_per_cta=0).warps_per_cta == 8
    )
    assert replace(qualified, num_banks=2, warps_per_cta=0).warps_per_cta == 8
    assert replace(qualified, timing_mode="cadence", warps_per_cta=0).warps_per_cta == 8


def test_explicit_warps_per_cta_overrides_measured_shape_policy():
    case = ElasticLLCase(
        2,
        4,
        128,
        8,
        7168,
        0,
        1,
        ((0, 1),),
        workers_per_peer=64,
        warps_per_cta=4,
        target_sm=100,
    )
    assert case.warps_per_cta == 4


@pytest.mark.parametrize(
    ("detected_sm", "expected_warps"), ((100, 2), (90, 8), (None, 8))
)
def test_run_reconciles_auto_geometry_with_selected_devices(
    monkeypatch, detected_sm, expected_warps
):
    import examples.python.cute.elastic_moe_ll as example

    case = ElasticLLCase(
        2,
        4,
        128,
        8,
        7168,
        0,
        1,
        ((0, 1),),
        empty_last_expert=False,
        workers_per_peer=64,
        target_sm=90,
    )
    assert case.warps_per_cta == 8

    captured = {}

    def fake_spawn(worker, *, args, nprocs, join):
        codegen_root = Path(args[4])
        assert codegen_root.name == "codegen"
        assert codegen_root.is_dir()
        assert all(
            not (codegen_root / f"rank-{rank}").exists() for rank in range(nprocs)
        )
        captured.update(worker=worker, args=args, nprocs=nprocs, join=join)

    monkeypatch.setattr(example.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(example.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(example, "_detect_common_sm", lambda devices: detected_sm)
    monkeypatch.setattr(example.torch.multiprocessing, "spawn", fake_spawn)

    example.run(devices=(0, 1), case=case, timeout_s=30.0)

    resolved = captured["args"][2]
    assert resolved.target_sm == detected_sm
    assert resolved.warps_per_cta == expected_warps
    assert resolved._warps_per_cta_is_auto is True
    assert captured["nprocs"] == 2
    assert captured["join"] is True


def test_run_preserves_explicit_geometry_without_querying_capability(monkeypatch):
    import examples.python.cute.elastic_moe_ll as example

    case = ElasticLLCase(
        2,
        4,
        128,
        8,
        7168,
        0,
        1,
        ((0, 1),),
        empty_last_expert=False,
        workers_per_peer=64,
        warps_per_cta=4,
        target_sm=100,
    )
    captured = {}

    def fail_if_queried(devices):
        raise AssertionError("explicit geometry must not query CUDA capability")

    def fake_spawn(worker, *, args, nprocs, join):
        captured["case"] = args[2]

    monkeypatch.setattr(example.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(example.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(example, "_detect_common_sm", fail_if_queried)
    monkeypatch.setattr(example.torch.multiprocessing, "spawn", fake_spawn)

    example.run(devices=(0, 1), case=case, timeout_s=30.0)

    assert captured["case"] is case
    assert captured["case"].warps_per_cta == 4
    assert captured["case"]._warps_per_cta_is_auto is False


def test_rank_private_codegen_dump_is_fresh_and_compile_options_are_quoted(tmp_path):
    import examples.python.cute.elastic_moe_ll as example

    dump = example._rank_codegen_dump_directory(str(tmp_path), 1)
    assert dump == tmp_path.resolve() / "rank-1"
    assert dump.is_dir()
    assert dump.stat().st_mode & 0o777 == 0o700
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        example._rank_codegen_dump_directory(str(tmp_path), 1)
    incarnation_dump = example._rank_codegen_dump_directory(
        str(tmp_path), 1, incarnation=2
    )
    assert incarnation_dump.name == "rank-1-incarnation-2"
    with pytest.raises(ValueError, match="rank"):
        example._rank_codegen_dump_directory(str(tmp_path), -1)
    with pytest.raises(ValueError, match="incarnation"):
        example._rank_codegen_dump_directory(str(tmp_path), 0, incarnation=0)

    spaced = tmp_path / "a codegen path"
    assert shlex.split(example._compile_dump_options(spaced, keep_cubin=False)) == [
        f"--dump-dir={spaced}"
    ]
    assert shlex.split(example._compile_dump_options(spaced, keep_cubin=True)) == [
        "--keep-cubin",
        f"--dump-dir={spaced}",
    ]


def test_run_passes_one_fresh_explicit_codegen_parent_to_spawn(monkeypatch, tmp_path):
    import examples.python.cute.elastic_moe_ll as example

    case = ElasticLLCase(2, 2, 4, 2, 8, 0, 1, ((0, 1),))
    captured = {}

    def fake_spawn(worker, *, args, nprocs, join):
        codegen_root = Path(args[4])
        assert codegen_root == tmp_path.resolve()
        assert all(
            not (codegen_root / f"rank-{rank}").exists() for rank in range(nprocs)
        )
        captured.update(worker=worker, args=args, nprocs=nprocs, join=join)

    monkeypatch.setattr(example.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(example.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        example.torch.cuda, "get_device_capability", lambda _device: (10, 0)
    )
    monkeypatch.setattr(example.torch.multiprocessing, "spawn", fake_spawn)

    example.run(
        devices=(0, 1),
        case=case,
        timeout_s=30.0,
        codegen_dump_root=str(tmp_path),
    )
    assert captured["worker"] is example._worker
    assert captured["nprocs"] == 2
    assert captured["join"] is True

    (tmp_path / "rank-0").mkdir()
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        example.run(
            devices=(0, 1),
            case=case,
            timeout_s=30.0,
            codegen_dump_root=str(tmp_path),
        )


def test_every_iteration_qualification_varies_sources_and_catches_history_error():
    case = ElasticLLCase(
        2,
        2,
        4,
        2,
        8,
        2,
        3,
        ((0, 1),),
        validate_every_iteration=True,
    )
    phase = build_phase_specs(case)[0]
    layout = case.layout
    assert layout.dispatch_stage_copies == case.total_iterations == 5
    assert case.output_copies == case.iterations == 3
    stage = make_phase_dispatch_stage(case, phase, 0)
    first = layout.dispatch_stage_record(0, 0)
    second = layout.dispatch_stage_record(0, 1)
    stage_base = layout.region("dispatch_stage").offset
    payload0 = stage[first.offset - stage_base + 16 : first.end - stage_base]
    payload1 = stage[second.offset - stage_base + 16 : second.end - stage_base]
    assert not torch.equal(payload0, payload1)

    history = torch.stack(
        [
            torch.stack(list(expected_outputs(case, phase, 0, operation).values()))
            for operation in range(case.warmup, case.total_iterations)
        ]
    )
    _validate_outputs(case, phase, 0, history.reshape(-1))
    history[0, 0, 0] += torch.tensor(1, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="operation 2 token 0"):
        _validate_outputs(case, phase, 0, history.reshape(-1))


def test_diagnostic_timeout_has_best_effort_drain_margin():
    case = _case()
    _require_diagnostic_timeout_margin(case, 0.001)
    diagnostic = ElasticLLCase(
        case.max_ranks,
        case.experts_per_rank,
        case.num_tokens,
        case.top_k,
        case.hidden_size,
        case.warmup,
        case.iterations,
        case.membership_plan,
        case.empty_last_expert,
        1_000_000_000,
    )
    minimum = diagnostic.device_timeout_ns / 1_000_000_000 + 10.0
    with pytest.raises(ValueError, match="best-effort drain margin"):
        _require_diagnostic_timeout_margin(diagnostic, minimum)
    _require_diagnostic_timeout_margin(diagnostic, minimum + 0.001)


def test_topk_router_is_duplicate_free_normalized_and_bucket_packed():
    case = _case()
    phase = build_phase_specs(case)[0]
    routes = build_routes(case, phase, rank=0)
    assert len(routes) == case.route_capacity
    for token in range(case.num_tokens):
        token_routes = [route for route in routes if route.origin_token == token]
        assert [route.route_slot for route in token_routes] == [0, 1]
        assert len({route.global_expert for route in token_routes}) == case.top_k
        assert sum(route.gate for route in token_routes) == 1.0

    state = build_phase_route_state(case, phase, rank=0)
    assert len(state.bucket_counts) == case.max_ranks * case.max_experts
    assert len(state.bucket_offsets) == len(state.bucket_counts)
    assert len(state.route_experts) == case.route_capacity
    assert len(state.route_gates) == case.route_capacity
    assert max(state.bucket_counts) <= case.num_tokens
    for source_rank in range(case.max_ranks):
        begin = source_rank * case.max_experts
        total = sum(state.bucket_counts[begin : begin + case.max_experts])
        assert total == (case.route_capacity if source_rank in {0, 2} else 0)
    # Global expert five is deliberately empty but must still be published.
    assert state.bucket_counts[5] == 0
    assert state.bucket_counts[2 * case.max_experts + 5] == 0
    assert tuple(
        (route.global_expert, route.origin_token, route.route_slot)
        for route in state.packed_routes
    ) == tuple(
        sorted(
            (
                route.global_expert,
                route.origin_token,
                route.route_slot,
            )
            for route in routes
        )
    )


def test_phase_worker_plan_partitions_every_dynamic_task_without_device_division():
    case = _case()
    assert (
        WORKER_PLAN_ELEMENT_NBYTES == torch.empty((), dtype=torch.int32).element_size()
    )
    fixed_worker_count = case.max_ranks * case.workers_per_peer
    for phase in build_phase_specs(case):
        active_task_count = len(phase.active_ranks) * case.experts_per_rank
        for rank in range(case.max_ranks):
            state = build_phase_route_state(case, phase, rank)
            assert len(state.worker_plan) == fixed_worker_count * WORKER_PLAN_FIELDS
            ranges: dict[int, list[tuple[int, int, int, int]]] = {
                task: [] for task in range(active_task_count)
            }
            for worker in range(fixed_worker_count):
                base = worker * WORKER_PLAN_FIELDS
                task, out_begin, out_count, in_begin, in_count = state.worker_plan[
                    base : base + WORKER_PLAN_FIELDS
                ]
                assert task == worker % active_task_count
                assert 0 <= out_begin <= case.num_tokens
                assert 0 <= out_count <= case.num_tokens - out_begin
                assert 0 <= in_begin <= case.num_tokens
                assert 0 <= in_count <= case.num_tokens - in_begin
                ranges[task].append(
                    (out_begin, out_begin + out_count, in_begin, in_begin + in_count)
                )

            for task, task_ranges in ranges.items():
                peer = phase.active_ranks[task // case.experts_per_rank]
                expert = task % case.experts_per_rank
                outgoing_bucket = (
                    rank * case.max_experts + peer * case.experts_per_rank + expert
                )
                incoming_bucket = (
                    peer * case.max_experts + rank * case.experts_per_rank + expert
                )
                outgoing_cursor = 0
                incoming_cursor = 0
                for out_begin, out_end, in_begin, in_end in task_ranges:
                    assert out_begin == outgoing_cursor
                    assert in_begin == incoming_cursor
                    outgoing_cursor = out_end
                    incoming_cursor = in_end
                assert outgoing_cursor == state.bucket_counts[outgoing_bucket]
                assert incoming_cursor == state.bucket_counts[incoming_bucket]


@pytest.mark.parametrize(
    ("active_ranks", "incarnations", "message"),
    (
        ((2, 0), (1, 0, 1), "sorted and duplicate-free"),
        ((0, 0), (1, 0, 0), "sorted and duplicate-free"),
        ((0, 3), (1, 0, 0), "fixed rank capacity"),
        ((0, 2), (1, 0), "incarnation table"),
    ),
)
def test_phase_worker_plan_rejects_noncanonical_direct_phase_specs(
    active_ranks, incarnations, message
):
    case = _case()
    phase = PhaseSpec(
        generation=0,
        active_ranks=active_ranks,
        rank_incarnations=incarnations,
    )
    with pytest.raises(ValueError, match=message):
        build_phase_route_state(case, phase, rank=0)


def test_phase_worker_plan_rejects_boolean_direct_phase_rank():
    case = _case()
    phase = PhaseSpec(
        generation=0,
        active_ranks=(True, 2),
        rank_incarnations=(1, 0, 1),
    )
    with pytest.raises(TypeError, match="must be integers"):
        build_phase_route_state(case, phase, rank=0)


@pytest.mark.parametrize(
    ("generation", "incarnations", "exception", "message"),
    (
        (True, (1, 0, 1), TypeError, "generation must be an integer"),
        (-1, (1, 0, 1), ValueError, "generation must fit"),
        (1 << 32, (1, 0, 1), ValueError, "generation must fit"),
        (0, (True, 0, 1), TypeError, "incarnations must be integers"),
        (0, (1, -1, 1), ValueError, "signed Int64 staging"),
        (0, (1, 0, 1 << 63), ValueError, "signed Int64 staging"),
        (0, (0, 0, 1), ValueError, "positive incarnation"),
    ),
)
def test_phase_worker_plan_rejects_unsafe_direct_phase_epochs(
    generation, incarnations, exception, message
):
    case = _case()
    phase = PhaseSpec(
        generation=generation,
        active_ranks=(0, 2),
        rank_incarnations=incarnations,
    )
    with pytest.raises(exception, match=message):
        build_phase_route_state(case, phase, rank=0)


def test_phase_worker_plan_accepts_epoch_staging_boundaries():
    case = _case()
    phase = PhaseSpec(
        generation=(1 << 32) - 1,
        active_ranks=(0, 2),
        rank_incarnations=((1 << 63) - 1, 0, 1),
    )
    state = build_phase_route_state(case, phase, rank=0)
    assert len(state.worker_plan) == (
        case.max_ranks * case.workers_per_peer * WORKER_PLAN_FIELDS
    )


def test_route_state_fails_closed_if_bucket_exceeds_compact_capacity(monkeypatch):
    import examples.python.cute.elastic_moe_ll as example

    case = ElasticLLCase(1, 2, 2, 2, 8, 0, 1, ((0,),), False)
    phase = build_phase_specs(case)[0]
    overloaded = tuple(
        example.RouteAssignment(token, slot, 0, 0.5)
        for token in range(case.num_tokens)
        for slot in range(case.top_k)
    )
    monkeypatch.setattr(example, "build_routes", lambda *_args: overloaded)

    with pytest.raises(AssertionError, match="compact dispatch receive capacity"):
        example.build_phase_route_state(case, phase, rank=0)


def test_bank_reuse_and_empty_publication_schedule():
    assert bank_cycle_schedule(3) == (
        (0, 0, 1),
        (1, 0, 2),
        (2, 0, 3),
    )
    assert bank_cycle_schedule(6, num_banks=2) == (
        (0, 0, 1),
        (1, 1, 1),
        (2, 0, 2),
        (3, 1, 2),
        (4, 0, 3),
        (5, 1, 3),
    )
    assert cumulative_publication_value(4, 3) == (3 << 32) | 5
    assert cumulative_publication_value(0, 3) == (3 << 32) | 1
    with pytest.raises(ValueError, match="record_count"):
        cumulative_publication_value(-1, 1)


def test_phase_arena_has_exact_wire_metadata_and_read_only_payload_template():
    case = _case()
    phase = build_phase_specs(case)[2]
    arena = make_phase_arena(case, phase, rank=2)
    layout = case.layout
    packed = build_phase_route_state(case, phase, rank=2).packed_routes
    route_index = 3
    route = packed[route_index]

    span = layout.dispatch_stage_record(route_index)
    origin_token, route_slot, gate, reserved = struct.unpack(
        "<IIfI", bytes(arena[span.offset : span.offset + 16].tolist())
    )
    assert origin_token == route.origin_token
    assert route_slot == route.route_slot
    assert gate == route.gate
    assert reserved == 0
    payload = arena[span.offset + 16 : span.offset + 16 + case.hidden_size * 2].view(
        torch.bfloat16
    )
    expected = (
        torch.arange(case.hidden_size, dtype=torch.float32) * 0.00390625
        + 2 * 8.0
        + route.origin_token * 0.125
    ).to(torch.bfloat16)
    assert torch.equal(payload, expected)

    # Receive state and counters start at zero for a drained generation swap.
    ready = layout.region("dispatch_ready")
    assert torch.count_nonzero(arena[ready.offset : ready.end]).item() == 0
    dispatch_stamp = layout.region("dispatch_stamp")
    combine_stamp = layout.region("combine_stamp")
    assert (
        torch.count_nonzero(arena[dispatch_stamp.offset : dispatch_stamp.end]).item()
        == 0
    )
    assert (
        torch.count_nonzero(arena[combine_stamp.offset : combine_stamp.end]).item() == 0
    )


def test_production_phase_stage_excludes_large_receive_regions():
    case = _case()
    phase = build_phase_specs(case)[2]
    layout = case.layout
    stage_region = layout.region("dispatch_stage")

    stage = make_phase_dispatch_stage(case, phase, rank=2)
    arena = make_phase_arena(case, phase, rank=2)

    assert stage.numel() == stage_region.nbytes
    assert stage.numel() < arena.numel()
    assert torch.equal(arena[stage_region.offset : stage_region.end], stage)


def test_expected_outputs_do_fp32_topk_reduction_and_one_bf16_cast():
    case = _case()
    phase = build_phase_specs(case)[0]
    expected = expected_outputs(case, phase, rank=0)
    assert set(expected) == set(range(case.num_tokens))

    token = 0
    source = (
        torch.arange(case.hidden_size, dtype=torch.float32) * 0.00390625 + token * 0.125
    ).to(torch.bfloat16)
    accumulator = (source + torch.tensor(1, dtype=torch.bfloat16)).to(
        torch.bfloat16
    ).float() * 0.25 + (source + torch.tensor(2, dtype=torch.bfloat16)).to(
        torch.bfloat16
    ).float() * 0.75
    assert torch.equal(
        expected[token],
        accumulator.to(torch.bfloat16),
    )
    assert expected_outputs(case, phase, rank=1) == {}


def test_topk_one_remains_supported_through_shrink():
    case = ElasticLLCase(
        max_ranks=2,
        experts_per_rank=2,
        num_tokens=3,
        top_k=1,
        hidden_size=8,
        warmup=0,
        iterations=1,
        membership_plan=((0, 1), (0,)),
    )
    phase = build_phase_specs(case)[1]
    routes = build_routes(case, phase, rank=0)
    assert len(routes) == case.num_tokens
    assert all(route.route_slot == 0 and route.gate == 1.0 for route in routes)
    assert set(expected_outputs(case, phase, rank=0)) == {0, 1, 2}


def test_phase_summary_keeps_raw_gpu_samples_and_protocol_claims_exact():
    case = _case()
    phase = build_phase_specs(case)[0]
    rows = _valid_summary_rows(case, phase)
    result = summarize_phase_results(case, phase, rows)
    assert result["rank_observer_span"]["enabled"] is False
    assert result["rank_observer_span"]["samples_ns"] == []
    assert result["rank_observer_span"]["p50_ns"] is None
    assert (
        "not claimed as per-round critical latency"
        in result["rank_observer_span"]["definition"]
    )
    assert result["num_tokens_per_rank"] == case.num_tokens
    assert result["top_k"] == 2
    assert result["schema_version"] == 7
    assert result["fixed_worker_warps"] == 6
    assert result["warps_per_cta"] == 6
    assert result["cooperative_ctas"] == 1
    assert result["allow_unverified_mapped"] is False
    native = result["native_peer_atomics"]
    assert native["all_supported"] is True
    assert native["checked_by_worker_slots"] == [0, 2]
    assert native["accessing_devices"] == [0, 2]
    assert len(native["ordered_pairs"]) == 4
    assert native["worker_evidence_shape"] == (
        "one local-accessor row per worker; canonical rows combined once"
    )
    assert native["policy"] == "required for every directed accessor-to-owner pair"
    assert result["mapped_preflight"]["all_active_remote_links_mapped"] is True
    assert result["mapped_preflight"]["translated_va_observed"] is False
    assert set(result["mapped_preflight"]["per_rank"]) == {"0", "2"}
    assert "native peer atomics" in result["protocol"]["mapped_address_validation"]
    assert "non-null process-local" in result["protocol"]["mapped_address_validation"]
    assert result["timing"]["cpu_synchronization_inside_measured_loop"] is False
    assert result["timing"]["local_wait_timer_reads"] == 0
    assert result["timing"]["explicit_start_marker_reads_per_rank_phase"] == 3
    assert result["timing"]["explicit_end_marker_reads_per_rank_phase"] == 1
    assert result["timing"]["remote_wait_mode"] == "abort_only_timer_free"
    assert result["timing"]["device_operation_timeout_ns"] is None
    assert result["timing"]["remote_wait_first_miss_timer_reads"] == 0
    assert result["timing"]["timeout_clock_check_period_failed_loads"] is None
    assert result["timing"]["timing_grid_fences_per_phase"] == 1
    assert result["timing"]["pre_reduction_grid_barriers_per_round"] == 1
    assert result["timing"]["end_of_round_grid_barriers_per_rank_phase"] == 5
    assert result["timing"]["total_protocol_grid_barriers_per_rank_phase"] == 10
    assert "production timer-free path" in result["timing"]["device_timeout_role"]
    assert result["steady_state_run"]["measured_interval_ns"] == 1000
    assert result["steady_state_run"]["measured_rounds"] == 1
    assert result["output_ready_run"]["measured_interval_ns"] == 2410
    assert result["round_cadence"]["samples_ns"] == [1000]
    assert result["round_cadence"]["all_samples_including_timing_fence_ns"] == [
        1000,
        1000,
    ]
    assert "remote_wire_bytes_per_phase_round" not in result
    assert result["remote_record_store_bytes_per_phase_round"] > 0
    # Two ordered remote rank pairs; each stores two dispatch stamp/ready
    # pairs and one aggregate combine stamp/ready pair; the closed handshake
    # needs no separate credit.
    assert result["remote_control_store_bytes_per_phase_round"] == 144
    assert result["remote_mapped_store_bytes_per_phase_round"] == (
        result["remote_record_store_bytes_per_phase_round"]
        + result["remote_control_store_bytes_per_phase_round"]
    )
    assert (
        result["steady_state_run"]["aggregate_logical_remote_GBps_cadence_estimate"] > 0
    )
    assert (
        "not synchronized global wall-clock" in result["steady_state_run"]["definition"]
    )
    assert result["protocol"]["banks"] == 1
    assert result["protocol"]["immutable_dispatch_template_banks"] == 1
    assert result["protocol"]["source_working_set_mode"] == (
        "single immutable hot stage copy"
    )
    assert "peer-aggregate" in result["protocol"]["combine_ready_encoding"]
    assert "fence ABI exists" in result["protocol"]["network_fallback"]
    assert "not implemented" in result["protocol"]["network_fallback"]
    assert result["elasticity"]["abrupt_failure_recovery"] is False
    assert "may fault" in result["elasticity"]["abrupt_mapped_owner_loss"]
    assert result["correctness"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "did not report native peer-atomic evidence"),
        ("malformed", "malformed native peer-atomic evidence"),
        ("schema_float", "wrong native-atomic evidence schema"),
        ("unsupported", "did not qualify native atomics"),
        ("wrong_device", "names the wrong local device"),
        ("wrong_accessor", "accessor-row coverage is invalid"),
        ("negative_device", "device coverage is invalid"),
        ("incomplete", "directed-pair coverage is invalid"),
        ("inconsistent", "evidence disagrees across workers"),
        ("physical_alias", "aliases physical devices"),
    ],
)
def test_phase_summary_rejects_invalid_native_atomic_evidence(mutation, message):
    case = _case()
    phase = build_phase_specs(case)[0]
    rows = _valid_summary_rows(case, phase)
    environment = rows[0]["environment"]
    assert isinstance(environment, dict)
    evidence = environment["native_peer_atomics"]
    assert isinstance(evidence, dict)

    if mutation == "missing":
        del environment["native_peer_atomics"]
    elif mutation == "malformed":
        environment["native_peer_atomics"] = []
    elif mutation == "schema_float":
        evidence["schema_version"] = 1.0
    elif mutation == "unsupported":
        ordered_pairs = evidence["ordered_pairs"]
        assert isinstance(ordered_pairs, list)
        ordered_pairs[0]["native_atomics_supported"] = False
    elif mutation == "wrong_device":
        environment["device"] = 2
    elif mutation == "wrong_accessor":
        evidence["accessing_devices"] = [2]
    elif mutation == "negative_device":
        evidence["devices"][0] = -1
    elif mutation == "incomplete":
        ordered_pairs = evidence["ordered_pairs"]
        assert isinstance(ordered_pairs, list)
        ordered_pairs.pop()
    elif mutation == "inconsistent":
        other_environment = rows[2]["environment"]
        assert isinstance(other_environment, dict)
        other_evidence = json.loads(
            json.dumps(other_environment["native_peer_atomics"])
        )
        other_evidence["cuda_visible_devices"] = "different"
        other_environment["native_peer_atomics"] = other_evidence
    elif mutation == "physical_alias":
        identities = evidence["device_identities"]
        assert isinstance(identities, list)
        identities[1]["pci_bus_id"] = identities[0]["pci_bus_id"]
    else:  # pragma: no cover - keeps future table edits fail-closed
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        summarize_phase_results(case, phase, rows)


def test_rank_result_uses_start_cadence_and_only_one_production_end_marker():
    case = _case()
    phase = build_phase_specs(case)[0]
    timestamps = torch.zeros(
        case.iterations * (case.max_ranks + 1) * 2, dtype=torch.uint64
    )
    stride = (case.max_ranks + 1) * 2
    timestamps[0] = 1000
    timestamps[stride] = 2000
    timestamps[2 * stride] = 3000
    timestamps[2 * stride + 1] = 3400
    result = _rank_phase_result(case, phase, rank=0, timestamps=timestamps)
    assert result["rank_round_starts_ns"] == [1000, 2000, 3000]
    assert result["rank_round_ends_ns"] == [0, 0, 3400]
    assert result["rank_observer_span_ns"] == []

    timestamps[1] = 1300
    with pytest.raises(RuntimeError, match="unnecessary end marker"):
        _rank_phase_result(case, phase, rank=0, timestamps=timestamps)


def test_rank_result_timing_specializations_are_exact_and_fail_closed():
    base = _case()
    phase = build_phase_specs(base)[0]
    stride = (base.max_ranks + 1) * 2

    disabled = ElasticLLCase(
        base.max_ranks,
        base.experts_per_rank,
        base.num_tokens,
        base.top_k,
        base.hidden_size,
        base.warmup,
        base.iterations,
        base.membership_plan,
        timing_mode="none",
    )
    timestamps = torch.ones(1, dtype=torch.uint64)
    result = _rank_phase_result(disabled, phase, rank=0, timestamps=timestamps)
    assert result["rank_round_starts_ns"] == []
    assert result["rank_round_ends_ns"] == []
    assert result["timing_mode"] == "none"

    envelope = ElasticLLCase(
        base.max_ranks,
        base.experts_per_rank,
        base.num_tokens,
        base.top_k,
        base.hidden_size,
        base.warmup,
        base.iterations,
        base.membership_plan,
        timing_mode="envelope",
    )
    timestamps = torch.zeros(envelope.iterations * stride, dtype=torch.uint64)
    timestamps[0] = 1000
    timestamps[2 * stride + 1] = 3400
    result = _rank_phase_result(envelope, phase, rank=0, timestamps=timestamps)
    assert result["rank_round_starts_ns"] == [1000, 0, 0]
    assert result["rank_round_ends_ns"] == [0, 0, 3400]


def test_uninstrumented_timestamp_storage_is_constant_in_iteration_count():
    case = ElasticLLCase(
        1,
        1,
        1,
        1,
        8,
        0,
        1_000_000,
        ((0,),),
        timing_mode="none",
    )
    assert _timestamp_buffer_elements(case) == 1

    diagnostic = ElasticLLCase(
        1,
        1,
        1,
        1,
        8,
        0,
        17,
        ((0,),),
        timing_mode="envelope",
    )
    assert _timestamp_buffer_elements(diagnostic) == 17 * (1 + 1) * 2


class _FakeDeviceView:
    def __init__(self, failures: int) -> None:
        self.valid = True
        self.failures = failures
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        if self.release_calls <= self.failures:
            raise RuntimeError(f"release failure {self.release_calls}")
        self.valid = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


@pytest.mark.parametrize(
    "failures,error_type,is_valid,release_calls",
    [
        (0, None, False, 1),
        (1, _RecoveredDeviceViewReleaseError, False, 2),
        (2, _UnreleasedDeviceViewError, True, 2),
    ],
)
def test_device_view_exit_stack_retries_and_reports_exact_safety_state(
    failures, error_type, is_valid, release_calls
):
    view = _FakeDeviceView(failures)
    scope = _DeviceViewExitStack()
    assert scope.enter_context(view) is view

    error = scope.close()

    if error_type is None:
        assert error is None
    else:
        assert isinstance(error, error_type)
    assert view.valid is is_valid
    assert view.release_calls == release_calls
    assert scope.close() is error


def test_unreleased_view_failure_detection_is_fail_closed():
    assert _contains_unreleased_view(
        [
            {
                "rank": 1,
                "ok": False,
                "error_type": _UnreleasedDeviceViewError.__name__,
            }
        ]
    )
    assert not _contains_unreleased_view(
        [
            {
                "rank": 1,
                "ok": False,
                "error_type": _RecoveredDeviceViewReleaseError.__name__,
            }
        ]
    )


def test_uninstrumented_summary_has_no_device_timing_claims():
    case = ElasticLLCase(1, 1, 1, 1, 8, 0, 2, ((0,),), num_banks=2, timing_mode="none")
    phase = build_phase_specs(case)[0]
    timestamps = torch.zeros(
        case.iterations * (case.max_ranks + 1) * 2, dtype=torch.uint64
    )
    row = _rank_phase_result(case, phase, rank=0, timestamps=timestamps)
    row["environment"] = {
        "device": 0,
        "native_peer_atomics": _native_atomic_evidence((0,)),
        "mapped_preflight": {
            "policy": "require_non_null_process_local_pointer",
            "evidence_scope": "exact timed-phase device view and arena allocation",
            "rank_active": True,
            "remote_peers": {},
            "all_active_remote_peers_mapped": True,
            "translated_va_observed": False,
            "legacy_allow_unverified_mapped_ignored": False,
        },
    }

    result = summarize_phase_results(case, phase, [row])
    assert result["timing_mode"] == "none"
    assert result["timing"]["enabled"] is False
    assert result["timing"]["clock"] is None
    assert result["timing"]["explicit_start_marker_reads_per_rank_phase"] == 0
    assert result["timing"]["explicit_end_marker_reads_per_rank_phase"] == 0
    assert result["timing"]["timing_grid_fences_per_phase"] == 0
    assert result["timing"]["pre_reduction_grid_barriers_per_round"] == 1
    assert result["timing"]["end_of_round_grid_barriers_per_rank_phase"] == 0
    assert result["timing"]["total_protocol_grid_barriers_per_rank_phase"] == 2
    assert result["steady_state_run"]["measured_interval_ns"] is None
    assert (
        result["steady_state_run"]["aggregate_logical_remote_GBps_cadence_estimate"]
        is None
    )
    assert result["output_ready_run"]["measured_interval_ns"] is None


@pytest.mark.parametrize(
    "timing_mode,start_reads,end_reads,end_barriers",
    [
        ("none", 0, 0, 0),
        ("envelope", 1, 1, 1),
        ("cadence", 3, 1, 3),
        ("peer", 3, 3, 3),
    ],
)
def test_two_bank_timing_modes_report_exact_marker_and_barrier_costs(
    timing_mode, start_reads, end_reads, end_barriers
):
    case = ElasticLLCase(
        1,
        1,
        1,
        1,
        8,
        2,
        3,
        ((0,),),
        num_banks=2,
        timing_mode=timing_mode,
    )
    phase = build_phase_specs(case)[0]
    stride = (case.max_ranks + 1) * 2
    timestamps = torch.zeros(case.iterations * stride, dtype=torch.uint64)
    if timing_mode != "none":
        timestamps[0] = 100
        timestamps[(case.iterations - 1) * stride + 1] = 450
    if timing_mode in {"cadence", "peer"}:
        for iteration in range(case.iterations):
            timestamps[iteration * stride] = 100 + 100 * iteration
    if timing_mode == "peer":
        for iteration in range(case.iterations):
            base = iteration * stride
            timestamps[base + 1] = 175 + 100 * iteration
            timestamps[base + 2] = 110 + 100 * iteration
            timestamps[base + 3] = 160 + 100 * iteration
    row = _rank_phase_result(case, phase, rank=0, timestamps=timestamps)
    row["environment"] = {
        "device": 0,
        "native_peer_atomics": _native_atomic_evidence((0,)),
        "mapped_preflight": {
            "policy": "require_non_null_process_local_pointer",
            "evidence_scope": "exact timed-phase device view and arena allocation",
            "rank_active": True,
            "remote_peers": {},
            "all_active_remote_peers_mapped": True,
            "translated_va_observed": False,
            "legacy_allow_unverified_mapped_ignored": False,
        },
    }

    result = summarize_phase_results(case, phase, [row])
    timing = result["timing"]
    assert timing["explicit_start_marker_reads_per_rank_phase"] == start_reads
    assert timing["explicit_end_marker_reads_per_rank_phase"] == end_reads
    assert timing["end_of_round_grid_barriers_per_rank_phase"] == end_barriers
    assert timing["total_protocol_grid_barriers_per_rank_phase"] == (
        case.total_iterations + end_barriers
    )


def test_mapped_preflight_classifies_the_exact_phase_allocation():
    case = _case()
    phase = build_phase_specs(case)[0]  # active {0, 2}
    exact = _classify_mapped_preflight(
        torch.tensor([0, 0, 0], dtype=torch.int32),
        phase,
        rank=0,
        allow_unverified_mapped=False,
    )
    assert exact == {
        "policy": "require_non_null_process_local_pointer",
        "evidence_scope": "exact timed-phase device view and arena allocation",
        "rank_active": True,
        "remote_peers": {"2": {"classification": "owner_va_coincident", "status": 0}},
        "all_active_remote_peers_mapped": True,
        "translated_va_observed": False,
        "legacy_allow_unverified_mapped_ignored": False,
    }

    inactive = _classify_mapped_preflight(
        torch.tensor([0, 0, 0], dtype=torch.int32),
        phase,
        rank=1,
        allow_unverified_mapped=False,
    )
    assert inactive["rank_active"] is False
    assert inactive["remote_peers"] == {}
    assert inactive["all_active_remote_peers_mapped"] is True


def test_mapped_preflight_accepts_translated_process_local_va_by_default():
    case = _case()
    phase = build_phase_specs(case)[0]
    statuses = torch.tensor([0, 0, -5], dtype=torch.int32)
    translated = _classify_mapped_preflight(
        statuses,
        phase,
        rank=0,
        allow_unverified_mapped=False,
    )
    legacy = _classify_mapped_preflight(
        statuses,
        phase,
        rank=0,
        allow_unverified_mapped=True,
    )
    assert translated["policy"] == "require_non_null_process_local_pointer"
    assert (
        translated["remote_peers"]["2"]["classification"]
        == "translated_process_local_va"
    )
    assert translated["all_active_remote_peers_mapped"] is True
    assert translated["translated_va_observed"] is True
    assert translated["legacy_allow_unverified_mapped_ignored"] is False
    assert legacy["legacy_allow_unverified_mapped_ignored"] is True


def test_mapped_preflight_never_permits_an_unmapped_active_peer():
    phase = build_phase_specs(_case())[0]
    with pytest.raises(RuntimeError, match="not locally mapped"):
        _classify_mapped_preflight(
            torch.tensor([0, 0, -9], dtype=torch.int32),
            phase,
            rank=0,
            allow_unverified_mapped=True,
        )


def test_stable_composite_view_uses_local_padding_for_self_and_inactive_slots():
    # Import the internal constructor deliberately: descriptor-index stability
    # is a safety property, not an incidental CLI detail.
    from examples.python.cute.elastic_moe_ll import _phase_remote_coordinates

    case = _case()
    phase = build_phase_specs(case)[0]  # active {0, 2}
    peers = tuple(
        PeerCoordinates(
            f"peer-{rank}",
            (DeviceRegion(0x1000 + rank * 0x1000, 4096, rank),),
        )
        for rank in range(3)
    )
    coordinates = _phase_remote_coordinates(case, phase, rank=0, peers=peers)
    assert len(coordinates) == 3
    assert coordinates[0] == (0x1000, 4096, 0, "peer-0")
    assert coordinates[1] == (0x1000, 4096, 0, "peer-0")
    assert coordinates[2][3] == "peer-2"


def test_coordinate_exchange_is_deserialized_in_stable_rank_order():
    from examples.python.cute.elastic_moe_ll import _deserialize_rank_coordinates

    serialized = {
        rank: PeerCoordinates(
            f"peer-{rank}",
            (DeviceRegion(0x1000 + rank * 0x1000, 4096, rank),),
        ).to_bytes()
        for rank in reversed(range(3))
    }

    peers = _deserialize_rank_coordinates(serialized, 3)
    assert tuple(peer.agent_name for peer in peers) == ("peer-0", "peer-1", "peer-2")
    assert tuple(peer.regions[0].device_id for peer in peers) == (0, 1, 2)

    with pytest.raises(RuntimeError, match="missing=\\[1\\]"):
        _deserialize_rank_coordinates({0: serialized[0], 2: serialized[2]}, 3)


def test_rank_results_are_decoded_by_stable_slot_not_dict_iteration_order():
    gathered = {
        rank: json.dumps({"rank": rank}).encode("utf-8") for rank in reversed(range(3))
    }
    assert [row["rank"] for row in _decode_rank_results(gathered, 3)] == [0, 1, 2]

    with pytest.raises(RuntimeError, match="missing=\\[1\\]"):
        _decode_rank_results({0: gathered[0], 2: gathered[2]}, 3)


def test_phase_status_exchange_converges_success_and_failure_before_raise():
    success = {rank: _phase_status_payload(rank, None) for rank in reversed(range(3))}
    assert _decode_phase_failures(success, (0, 1, 2)) == ()

    failed = dict(success)
    failed[2] = _phase_status_payload(2, RuntimeError("terminal mismatch"))
    failures = _decode_phase_failures(failed, (0, 1, 2))
    assert failures == (
        {
            "rank": 2,
            "ok": False,
            "error_type": "RuntimeError",
            "message": "terminal mismatch",
        },
    )
    with pytest.raises(RuntimeError, match="cover every participant"):
        _decode_phase_failures({0: success[0], 2: success[2]}, (0, 1, 2))


def test_phase_result_payload_combines_validation_and_result_exchange():
    gathered = {
        rank: _phase_result_payload(rank, None, {"rank": rank, "value": rank + 1})
        for rank in reversed(range(3))
    }
    failures, results = _decode_phase_result_exchange(gathered, (0, 1, 2))
    assert failures == ()
    assert results == {rank: {"rank": rank, "value": rank + 1} for rank in range(3)}

    failed = dict(gathered)
    failed[1] = _phase_result_payload(1, RuntimeError("bad output"), None)
    failures, results = _decode_phase_result_exchange(failed, (0, 1, 2))
    assert len(failures) == 1
    assert failures[0]["rank"] == 1
    assert results == {}


def test_hot_loop_source_is_persistent_mapped_and_has_no_host_pacing():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    source = source_path.read_text()
    assert source.count("enable_prog_thread=False") == 1
    assert "enable_prog_thread=True" not in source
    assert source.count("require_peer_native_atomics(") == 1
    assert "devices, accessing_devices=(device,)" in source
    assert '"native_peer_atomics": native_atomic_preflight' in source
    set_device = source.index("torch.cuda.set_device(device)")
    native_preflight = source.index(
        "nixl_cute.require_peer_native_atomics(", set_device
    )
    stream_create = source.index("torch.cuda.Stream(device=device)", native_preflight)
    registration = source.index("agent.register_memory([arena]", stream_create)
    setup_handshake = source.index("complete_ucx_setup_handshake(", registration)
    measured_launch = source.index("compiled_main(", registration)
    assert (
        set_device
        < native_preflight
        < stream_create
        < registration
        < setup_handshake
        < measured_launch
    )
    assert 'control.barrier("connected")' not in source
    assert 'control.barrier("connections-made")' not in source
    kernel = source.split("def _elastic_moe_ll_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_elastic_moe_ll", 1
    )[0]
    assert "nixl_cute.mapped_copy_warp" in kernel
    assert "nixl_cute.mapped_copy_warp_ptr_readonly" in kernel
    assert "_publish_peer_u64(" not in source
    assert "nixl_cute.store_release_system_u64" in source
    assert "_remote_wait(" in source
    assert "nixl_cute.wait_acquire_system_u64_for" in source
    assert "nixl_cute.wait_acquire_system_u64_or_aborts" in source
    assert "nixl_cute.wait_acquire_system_u64_until" not in source
    assert "if peer != advertised_bases[tidx]" in source
    assert kernel.count("nixl_cute.get_ptr(") == 1
    assert kernel.index("nixl_cute.get_ptr(") < kernel.index(
        "for operation in cutlass.range(warmup + iterations"
    )
    assert "shuffle_sync(local_incarnation" not in kernel
    assert "shuffle_sync(peer_incarnation" not in kernel
    assert "put_post(" not in kernel
    assert ".synchronize(" not in kernel
    assert ".launch(" not in kernel
    assert "torch." not in kernel
    assert "for operation in cutlass.range(warmup + iterations" in kernel
    # Start marker, two-bank end-barrier policy, and end marker are three
    # separate compile-time regions.  All disappear in timing_mode="none".
    assert kernel.count("if cutlass.const_expr(record_timing):") == 3
    assert "if cutlass.const_expr(record_cadence):" in kernel
    assert "peer_ok = cutlass.Int32(1)" in kernel
    assert "_wait_peer(" not in source
    assert kernel.count("observed = _remote_wait(") == 1
    assert kernel.count("observed = _remote_wait_thread(") == 1
    assert kernel.count("observed = _wait_gpu(") == 1
    assert kernel.count("observed = _wait_gpu_thread(") == 3
    assert "nixl_cute.wait_acquire_gpu_u64_or_abort" in source
    assert "nixl_cute.atomic_max_release_system_u64" in kernel
    assert kernel.count("if observed <") == 4
    assert "record_count + 1" in kernel
    assert kernel.count("minimum_ready = (") == 2
    assert kernel.count("if observed >= minimum_ready:") == 2
    assert "route_slot" in kernel
    assert "accumulator_values = cute.make_rmem_tensor" in kernel
    assert "cute.nvgpu.CopyG2ROp()" in kernel
    assert "cute.nvgpu.CopyR2GOp()" in kernel
    assert "payload_vector_layout = cute.make_layout((4,))" in kernel
    assert "payload_vector_layout = cute.make_layout(4)" not in kernel
    assert kernel.count("num_bits_per_copy=128") == 2
    assert "L2PrefetchSize.SIZE_256B" in kernel
    assert kernel.count("CacheEvictionPriority.NO_ALLOCATE") == 2
    assert "combine_consumed_base" in kernel
    assert "task = cutlass.Int32(worker_plan[worker * WORKER_PLAN_FIELDS])" in kernel
    assert "worker % active_task_count" not in kernel
    assert "worker // active_task_count" not in kernel
    assert "// cutlass.Int64(task_shards)" not in kernel
    assert (
        "has_sibling = task_leader + active_task_count < fixed_worker_count" in kernel
    )
    assert "incoming_shard_count = cutlass.Int32(0)" in kernel
    assert "outgoing_shard_end" not in kernel
    assert "incoming_shard_end" not in kernel
    assert "shard_record_count = cutlass.Int64(0)" not in kernel
    assert "origin_token = cutlass.Uint32(metadata[0])" in kernel
    assert "route_slot = cutlass.Uint32(metadata[1])" in kernel
    # One timing-start branch, one mandatory pre-reduction join, one one-bank
    # reuse join, and two mutually-exclusive two-bank timing endpoint branches.
    assert kernel.count("nixl_cute.sync_grid()") == 5
    assert "if cutlass.const_expr(num_banks == 1):" in kernel
    assert "if cutlass.const_expr(record_cadence):" in kernel
    assert "global_coordinator" not in kernel
    assert kernel.count("nixl_cute.store_release_gpu_u64(") >= 3
    assert kernel.count("generation_state[0]") == 1
    assert kernel.count("incarnations[rank]") == 2
    assert kernel.count("incarnations[peer]") == 1
    assert kernel.count("while vector_base < full_vectors") == 2
    assert kernel.count("while vector < total_vectors") == 2
    assert "EXPERT_VECTOR_UNROLL" in kernel
    assert "COMBINE_VECTOR_UNROLL" in kernel
    assert "cute.recast_tensor(" in kernel
    assert ".bitcast(" not in kernel
    assert "cutlass.Uint16" not in kernel
    assert "cute.arch.load(" not in kernel
    assert "bank * route_capacity" not in kernel
    launcher = source.split("def _launch_elastic_moe_ll", 1)[1].split("def _region", 1)[
        0
    ]
    assert "cooperative=True" in launcher
    assert "min_blocks_per_mp=1" in launcher
    assert "block=[WARP_SIZE * warps_per_cta, 1, 1]" in launcher
    assert "grid=[max_ranks * workers_per_peer // warps_per_cta, 1, 1]" in launcher
    compiler = source.split("def _compile_kernels", 1)[1].split(
        "def _validate_statuses", 1
    )[0]
    assert (
        "options=_compile_dump_options(codegen_dump_dir, keep_cubin=False)" in compiler
    )
    assert (
        "options=_compile_dump_options(codegen_dump_dir, keep_cubin=True)" in compiler
    )
    assert "bind_and_validate_cooperative_launch(" in compiler
    assert 'case.timing_mode != "none"' in compiler
    assert 'case.timing_mode in {"cadence", "peer"}' in compiler
    assert "records_per_expert" not in source
    worker = source.split("def _worker", 1)[1].split("def run", 1)[0]
    assert "PinnedPhaseInputs.allocate(case)" in worker
    assert "host_inputs.prepare(" in worker
    assert "host_arena = make_phase_arena" not in worker
    assert "arena.copy_(host_arena" not in worker


def test_leader_only_joins_poll_on_lane_zero_and_broadcast_once():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    kernel = functions["_elastic_moe_ll_kernel"]

    assert "Scope.THREAD" in ast.unparse(functions["_wait_gpu_thread"])
    assert "Scope.WARP" not in ast.unparse(functions["_wait_gpu_thread"])
    assert "Scope.THREAD" in ast.unparse(functions["_remote_wait_thread"])
    assert "Scope.WARP" not in ast.unparse(functions["_remote_wait_thread"])
    assert "Scope.WARP" in ast.unparse(functions["_wait_gpu"])
    assert "Scope.WARP" in ast.unparse(functions["_remote_wait"])

    parents = {
        child: parent
        for parent in ast.walk(kernel)
        for child in ast.iter_child_nodes(parent)
    }

    def calls_named(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name
        ]

    def enclosing_guard(node: ast.AST, expression: str) -> ast.If | None:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.If) and ast.unparse(current.test) == expression:
                return current
        return None

    thread_waits = calls_named("_wait_gpu_thread") + calls_named("_remote_wait_thread")
    assert len(calls_named("_wait_gpu_thread")) == 3
    assert len(calls_named("_remote_wait_thread")) == 1
    for wait in thread_waits:
        lane_guard = enclosing_guard(wait, "lane == 0")
        assert lane_guard is not None
        assert enclosing_guard(wait, "peer != rank") is not None
        container = parents[lane_guard]
        assert isinstance(container, ast.If)
        lane_index = container.body.index(lane_guard)
        assert ast.unparse(container.body[lane_index + 1]) == (
            "peer_ok = cute.arch.shuffle_sync(peer_ok, 0)"
        )
        assert (
            sum(
                isinstance(node, ast.Call)
                and ast.unparse(node.func)
                in {"_wait_gpu_thread", "_remote_wait_thread"}
                for node in ast.walk(lane_guard)
            )
            == 1
        )

    # These two waits directly precede dispatch payload consumption, so every
    # consuming lane retains a WARP-scope acquire.
    warp_waits = calls_named("_remote_wait") + calls_named("_wait_gpu")
    assert len(calls_named("_remote_wait")) == 1
    assert len(calls_named("_wait_gpu")) == 1
    assert all(enclosing_guard(wait, "lane == 0") is None for wait in warp_waits)


def test_incarnation_values_remain_lane_zero_local():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    kernel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_elastic_moe_ll_kernel"
    )
    parents = {
        child: parent
        for parent in ast.walk(kernel)
        for child in ast.iter_child_nodes(parent)
    }

    def is_lane_zero_guarded(node: ast.AST) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.If) and ast.unparse(current.test) == "lane == 0":
                return True
        return False

    peer_uses = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "peer_incarnation"
    ]
    assert len(peer_uses) == 2
    assert all(is_lane_zero_guarded(use) for use in peer_uses)

    local_loads = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Subscript) and ast.unparse(node) == "incarnations[rank]"
    ]
    assert len(local_loads) == 2
    assert all(is_lane_zero_guarded(load) for load in local_loads)
    assert "local_incarnation" not in ast.unparse(kernel)


def test_self_peer_reads_stage_directly_and_skips_publication_protocol():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    kernel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_elastic_moe_ll_kernel"
    )
    parents = {
        child: parent
        for parent in ast.walk(kernel)
        for child in ast.iter_child_nodes(parent)
    }

    def is_remote_guarded(node: ast.AST) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if (
                isinstance(current, ast.If)
                and ast.unparse(current.test) == "peer != rank"
            ):
                return True
        return False

    incoming_assignments = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "incoming_offset"
    ]
    assert len(incoming_assignments) == 2
    direct = next(
        node
        for node in incoming_assignments
        if ast.unparse(node.value) == "source_offset"
    )
    remote = next(node for node in incoming_assignments if node is not direct)
    assert not is_remote_guarded(direct)
    assert is_remote_guarded(remote)

    protocol_calls = {
        "nixl_cute.mapped_copy_warp_ptr_readonly": 1,
        "nixl_cute.store_release_gpu_u64": 4,
        "nixl_cute.store_release_system_u64": 2,
        "cute.arch.sync_warp": 1,
        "_wait_gpu": 1,
        "_wait_gpu_thread": 3,
        "_remote_wait": 1,
        "_remote_wait_thread": 1,
    }
    for name, expected_count in protocol_calls.items():
        calls = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name
        ]
        assert len(calls) == expected_count
        assert all(is_remote_guarded(call) for call in calls)

    kernel_source = ast.unparse(kernel)
    direct_read = kernel_source.index("incoming_offset = source_offset")
    common_join = kernel_source.index("nixl_cute.sync_grid()", direct_read)
    reduction = kernel_source.index("while token < num_tokens", common_join)
    assert direct_read < common_join < reduction


def test_self_bucket_uses_the_same_packed_stage_slice_in_both_directions():
    case = _case()
    phase = build_phase_specs(case)[0]
    max_experts = case.max_ranks * case.experts_per_rank
    for rank in phase.active_ranks:
        route_state = build_phase_route_state(case, phase, rank)
        for expert in range(case.experts_per_rank):
            outgoing_bucket = rank * max_experts + rank * case.experts_per_rank + expert
            incoming_bucket = rank * max_experts + rank * case.experts_per_rank + expert
            assert outgoing_bucket == incoming_bucket
            begin = route_state.bucket_offsets[outgoing_bucket]
            count = route_state.bucket_counts[outgoing_bucket]
            packed = route_state.packed_routes[begin : begin + count]
            assert all(
                route.global_expert == rank * case.experts_per_rank + expert
                for route in packed
            )


def test_mismatch_validation_is_warp_uniform_and_fail_closed():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source)
    kernel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_elastic_moe_ll_kernel"
    )

    slot_loop = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "slot"
    )
    assert len(slot_loop.body) == 1
    slot_guard = slot_loop.body[0]
    assert isinstance(slot_guard, ast.If)
    assert ast.unparse(slot_guard.test) == "peer_ok != 0"
    slot_source = ast.unparse(slot_guard)

    metadata_load = slot_source.index("metadata = cute.make_tensor")
    metadata_broadcast = slot_source.index(
        "metadata_ok = cute.arch.shuffle_sync(metadata_ok, 0)"
    )
    metadata_abort = slot_source.index("if metadata_ok == 0:", metadata_broadcast)
    payload_read = slot_source.index("source_vector = cute.make_tensor")
    result_write = slot_source.index("remote_vector = cute.make_tensor")
    assert "if origin_token >= num_tokens:" in slot_source
    assert "if route_slot >= top_k:" in slot_source
    assert metadata_load < metadata_broadcast < metadata_abort < payload_read
    assert payload_read < result_write
    assert "peer_ok = cutlass.Int32(0)" in slot_source[metadata_abort:payload_read]
    assert slot_source.count("NIXL_ERR_MISMATCH") == 1

    kernel_source = ast.unparse(kernel)
    assert "incoming_header_ok" not in kernel_source
    assert "route_header_ok" not in kernel_source
    assert "result_header" not in kernel_source
    assert kernel_source.count("bucket_stamp[0] != operation_epoch") == 2
    assert kernel_source.count("bucket_stamp[1] != peer_incarnation") == 2
    dispatch_stamp_check = kernel_source.index("dispatch_stamp_base", 1)
    dispatch_fanout = kernel_source.index("incoming_ready_state", dispatch_stamp_check)
    combine_stamp_check = kernel_source.rindex("bucket_stamp[0] != operation_epoch")
    combine_barrier = kernel_source.index("nixl_cute.sync_grid()", combine_stamp_check)
    assert dispatch_stamp_check < dispatch_fanout
    assert combine_stamp_check < combine_barrier
    assert kernel_source.count("NIXL_ERR_MISMATCH") == 7


def test_combine_reduction_is_token_sharded_with_cooperative_grid_barriers():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    source = source_path.read_text()
    kernel = source.split("def _elastic_moe_ll_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_elastic_moe_ll", 1
    )[0]

    mapping = kernel.index("task = cutlass.Int32(worker_plan[")
    bounds = kernel.index("outgoing_shard_begin = worker_plan[")
    operation_loop = kernel.index("for operation in cutlass.range(")
    timing_barrier = kernel.index("nixl_cute.sync_grid()")
    publication_barrier = kernel.index("nixl_cute.sync_grid()", timing_barrier + 1)
    reduction = kernel.index("while token < num_tokens")
    second_barrier = kernel.rindex("nixl_cute.sync_grid()")

    assert mapping < bounds < operation_loop < timing_barrier
    assert timing_barrier < publication_barrier < reduction
    assert reduction < second_barrier
    timing_fence = kernel.index("if operation == warmup:")
    assert operation_loop < timing_fence < timing_barrier < publication_barrier
    assert kernel.count("nixl_cute.sync_grid()") == 5
    assert "token += cutlass.Uint32(fixed_worker_count)" in kernel
    assert "worker % active_task_count" not in kernel
    assert "worker // active_task_count" not in kernel
    assert "// cutlass.Int64(task_shards)" not in kernel
    assert "candidate < fixed_worker_count" not in kernel
    assert "global_coordinator" not in kernel
    assert "peer_combine_credit" not in kernel


def test_kernel_launcher_compile_and_runtime_arities_are_statically_consistent():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    kernel = functions["_elastic_moe_ll_kernel"]
    launcher = functions["_launch_elastic_moe_ll"]
    preflight_kernel = functions["_mapped_preflight_kernel"]
    preflight_launcher = functions["_launch_mapped_preflight"]
    compiler = functions["_compile_kernels"]
    worker = functions["_worker"]

    kernel_invocations = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_elastic_moe_ll_kernel"
    ]
    assert len(kernel_invocations) == 1
    assert len(kernel.args.args) == len(kernel_invocations[0].args)

    preflight_kernel_invocations = [
        node
        for node in ast.walk(preflight_launcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_mapped_preflight_kernel"
    ]
    assert len(preflight_kernel_invocations) == 1
    assert len(preflight_kernel.args.args) == len(preflight_kernel_invocations[0].args)

    compile_invocations = [
        node
        for node in ast.walk(compiler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_launch_elastic_moe_ll"
    ]
    assert len(compile_invocations) == 1
    assert len(launcher.args.args) == len(compile_invocations[0].args) - 1

    dynamic_count = (
        next(
            index
            for index, argument in enumerate(launcher.args.args)
            if argument.arg == "stream"
        )
        + 1
    )
    runtime_invocations = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compiled_main"
    ]
    assert len(runtime_invocations) == 1
    assert len(runtime_invocations[0].args) == dynamic_count


def test_validation_exchange_converges_after_remote_view_release():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    worker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_worker"
    )
    phase_loop = next(
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "phase"
    )
    view_context = next(
        statement
        for statement in phase_loop.body
        if isinstance(statement, ast.With)
        and ast.unparse(statement.items[0].context_expr) == "_DeviceViewExitStack()"
    )
    view_index = phase_loop.body.index(view_context)
    view_source = ast.unparse(view_context)
    after_view = "\n".join(
        ast.unparse(statement) for statement in phase_loop.body[view_index + 1 :]
    )

    assert "phase_views.enter_context(agent.prepare_device_view" in view_source
    assert view_source.index("host_inputs.prepare") < view_source.index(
        "phase_views.enter_context(agent.prepare_device_view"
    )
    assert "phase_views.close()" in view_source
    assert view_source.index("phase_views.close()") < view_source.index(
        "candidate-views-released"
    )
    assert "validation-status" not in view_source
    assert "view-released" not in view_source
    assert "kernel-drained" not in view_source
    assert "_phase_result_payload" in after_view
    assert "phase-{phase.generation}-result" in after_view
    assert "phase_views.release_error" in after_view
    assert "view-released" not in after_view
    assert "all-views-released" not in ast.unparse(worker)
    assert view_source.rindex("stream.synchronize()") > view_source.index(
        "compiled_main("
    )
    assert view_source.index("host_results.enqueue_preflight") < view_source.index(
        "stream.synchronize()"
    )
    assert view_source.index("host_results.enqueue_main") < view_source.rindex(
        "stream.synchronize()"
    )
    assert "_validate_statuses(host_results.statuses" in after_view
    assert "_validate_outputs(case, phase, rank, host_results.outputs" in after_view
    assert "rank, host_results.timestamps" in after_view


def test_malformed_control_envelopes_fail_stop_without_owner_unwind():
    source_path = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    )
    tree = ast.parse(source_path.read_text())
    worker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_worker"
    )
    parents = {
        child: parent
        for parent in ast.walk(worker)
        for child in ast.iter_child_nodes(parent)
    }

    def calls_named(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name
        ]

    def enclosing_try(node: ast.AST) -> ast.Try:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.Try):
                return current
        raise AssertionError(f"{ast.unparse(node)} is not protected by try")

    phase_decoders = calls_named("_decode_phase_failures")
    assert len(phase_decoders) == 2
    preflight_try = enclosing_try(phase_decoders[0])
    release_try = enclosing_try(phase_decoders[1])
    preflight_handler = ast.unparse(preflight_try.handlers[0])
    release_handler = ast.unparse(release_try.handlers[0])
    assert "phase_views.close()" in preflight_handler
    assert "_fail_stop_registered_owner" in preflight_handler
    assert "after local close/retry" in release_handler
    assert "_fail_stop_registered_owner" in release_handler

    result_decoders = calls_named("_decode_phase_result_exchange")
    assert len(result_decoders) == 1
    result_handler = ast.unparse(enclosing_try(result_decoders[0]).handlers[0])
    assert "phase_views.close()" in result_handler
    assert "_fail_stop_registered_owner" in result_handler

    worker_source = ast.unparse(worker)
    invariant = worker_source.index(
        "if mapped_preflight is None or remote_view is None:"
    )
    first_exchange = worker_source.index("phase-{phase.generation}-preflight-status")
    assert invariant < first_exchange
    between_exchange_and_main = worker_source[
        first_exchange : worker_source.index("compiled_main(", first_exchange)
    ]
    assert "successful preflight has no classification" not in between_exchange_and_main


def test_preflight_status_exchange_is_not_followed_by_a_redundant_host_barrier():
    source = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    ).read_text()

    assert 'f"phase-{phase.generation}-preflight-status"' in source
    assert 'control.barrier(f"phase-{phase.generation}-preflight")' not in source
    assert "statuses.zero_()" in source
    worker = source.split("def _worker", 1)[1].split("def run", 1)[0]
    staging = worker.split("with _DeviceViewExitStack() as phase_views", 1)[1].split(
        "compiled_preflight(", 1
    )[0]
    assert "statuses.zero_()" not in staging
    assert 'if case.timing_mode != "none":' in staging


def test_worker_allocation_does_not_force_an_early_host_stream_drain():
    source = (
        Path(__file__).parents[2] / "examples" / "python" / "cute" / "elastic_moe_ll.py"
    ).read_text()
    worker = source.split("def _worker", 1)[1].split("def run", 1)[0]
    allocation = worker.split("host_inputs = PinnedPhaseInputs.allocate", 1)[0]

    assert "stream.synchronize()" not in allocation


def test_cli_defaults_to_topk_two_and_explicit_token_capacity(monkeypatch):
    import examples.python.cute.elastic_moe_ll as example

    captured = {}

    def fake_run(*, devices, case, timeout_s, codegen_dump_root):
        captured.update(
            devices=devices,
            case=case,
            timeout_s=timeout_s,
            codegen_dump_root=codegen_dump_root,
        )

    monkeypatch.setattr(example, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["elastic_moe_ll.py"])
    example.main()
    assert captured["case"].top_k == 2
    assert captured["case"].num_tokens == 8
    assert captured["case"].warps_per_cta == 8
    assert captured["case"].device_timeout_ns == 0
    assert captured["case"].allow_unverified_mapped is False
    assert captured["case"].num_banks == 1
    assert captured["case"].timing_mode == "none"
    assert captured["case"].membership_plan == ((0, 1), (0,), (0, 1))
    assert captured["codegen_dump_root"] is None

    monkeypatch.setattr("sys.argv", ["elastic_moe_ll.py", "--allow-unverified-mapped"])
    example.main()
    assert captured["case"].allow_unverified_mapped is True

    monkeypatch.setattr("sys.argv", ["elastic_moe_ll.py", "--timing-mode", "envelope"])
    example.main()
    assert captured["case"].timing_mode == "envelope"

    monkeypatch.setattr("sys.argv", ["elastic_moe_ll.py", "--instrument-per-peer"])
    example.main()
    assert captured["case"].timing_mode == "peer"

    monkeypatch.setattr(
        "sys.argv",
        ["elastic_moe_ll.py", "--codegen-dump-root", "/tmp/cute-codegen"],
    )
    example.main()
    assert captured["codegen_dump_root"] == "/tmp/cute-codegen"


def test_cli_applies_sm100_geometry_only_to_the_qualified_full_moe_case(monkeypatch):
    import examples.python.cute.elastic_moe_ll as example

    captured = {}

    def fake_run(*, devices, case, timeout_s, codegen_dump_root):
        captured.update(
            devices=devices,
            case=case,
            timeout_s=timeout_s,
            codegen_dump_root=codegen_dump_root,
        )

    monkeypatch.setattr(example, "run", fake_run)
    monkeypatch.setattr(example, "_detect_common_sm", lambda devices: 100)
    monkeypatch.setattr(
        "sys.argv",
        [
            "elastic_moe_ll.py",
            "--experts-per-rank",
            "4",
            "--num-tokens",
            "128",
            "--top-k",
            "8",
            "--hidden-size",
            "7168",
            "--workers-per-peer",
            "64",
            "--membership",
            "0,1",
            "--no-empty-last-expert",
        ],
    )
    example.main()
    assert captured["case"].target_sm == 100
    assert captured["case"].warps_per_cta == 2

    def fail_if_queried(devices):
        raise AssertionError("explicit geometry must not query CUDA capability")

    monkeypatch.setattr(example, "_detect_common_sm", fail_if_queried)
    monkeypatch.setattr("sys.argv", ["elastic_moe_ll.py", "--warps-per-cta", "4"])
    example.main()
    assert captured["case"].target_sm is None
    assert captured["case"].warps_per_cta == 4


def test_common_sm_detection_fails_closed_for_unavailable_or_mixed_devices(monkeypatch):
    import examples.python.cute.elastic_moe_ll as example

    monkeypatch.setattr(example.torch.cuda, "is_available", lambda: False)
    assert example._detect_common_sm((0, 1)) is None

    monkeypatch.setattr(example.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        example.torch.cuda,
        "get_device_capability",
        lambda device: {0: (10, 0), 1: (10, 0), 2: (9, 0)}[device],
    )
    assert example._detect_common_sm((0, 1)) == 100
    assert example._detect_common_sm((0, 2)) is None


def test_output_poison_is_exactly_representable_in_bfloat16():
    poison = torch.tensor(OUTPUT_POISON, dtype=torch.bfloat16)
    assert float(poison) == OUTPUT_POISON
