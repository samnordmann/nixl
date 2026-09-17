# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import pytest
import torch

from examples.python.cute import benchmark_persistent as persistent_benchmark
from examples.python.cute.benchmark_persistent import (
    EP_DOORBELL_INTERVAL,
    PersistentCase,
    defer_put_for_slot,
    nixlbench_percentile,
    source_generation,
    summarize_timestamps,
)


def test_persistent_case_derived_work_and_threads():
    case = PersistentCase(1024, 4, 3, 2, "warp", "defer", 7, 11)

    assert case.total_iterations == 18
    assert case.payload_bytes == 12 * 1024
    assert case.bytes_per_iteration == 12 * 1024
    assert case.generation_words_per_group == 4
    assert case.generation_words_total == 12
    assert case.threads == 96
    assert case.groups_per_block == 3
    assert case.blocks == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"size": 0}, "size"),
        ({"size": 7}, "8-byte multiple"),
        ({"size": 12}, "8-byte multiple"),
        ({"groups": 4097}, "groups"),
        ({"groups": 2, "channels": 3}, "channels"),
        ({"scope": "block"}, "scope"),
        ({"mode": "fire-and-forget"}, "mode"),
        ({"warmup": -1}, "warmup"),
        ({"mode": "mapped", "scope": "thread"}, "warp scope"),
        ({"mode": "mapped", "scope": "warp", "size": 17}, "multiple of 16"),
        ({"device_timeout_ns": -1}, "device_timeout_ns"),
        ({"device_timeout_ns": 1 << 63}, "signed literal range"),
        ({"warmup": (1 << 31) - 1}, "Int32 device loop"),
        ({"groups": 4096, "channels": 1, "iterations": 131072}, "timestamp indexing"),
    ],
)
def test_persistent_case_rejects_unsafe_configurations(kwargs, match):
    values = dict(
        size=64,
        batch_size=2,
        groups=2,
        channels=2,
        scope="thread",
        mode="wait",
        warmup=0,
        iterations=2,
    )
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PersistentCase(**values)


def test_mapped_case_is_warp_specialized():
    case = PersistentCase(4096, 4, 3, 2, "warp", "mapped", 7, 11)
    assert case.mode == "mapped"
    assert case.threads == 96


def test_warp_groups_scale_across_multiple_ctas_without_idle_warps():
    case = PersistentCase(4096, 1, 128, 32, "warp", "defer", 1, 2)

    assert case.groups_per_block == 8
    assert case.threads == 256
    assert case.blocks == 16


def test_persistent_launches_use_exact_bound_cubin_occupancy():
    source = (
        Path(__file__).parents[2]
        / "examples"
        / "python"
        / "cute"
        / "benchmark_persistent.py"
    ).read_text()
    launchers = source.split("@cute.jit\n    def _launch_persistent_producer", 1)[
        1
    ].split("def _region", 1)[0]
    compilers = source.split("def _compile_producer", 1)[1].split("def _worker", 1)[0]
    run_body = source.split("def run", 1)[1].split("def main", 1)[0]

    assert launchers.count("cooperative=True") == 2
    assert compilers.count('options="--keep-cubin"') == 2
    assert compilers.count("bind_and_validate_cooperative_launch(") == 2
    assert compilers.count("block_threads=case.threads") == 2
    assert compilers.count("planned_ctas=case.blocks") == 2
    assert "peer.regions" not in compilers
    assert "expected_remote_" not in launchers
    assert "allow_unverified_mapped" not in launchers
    assert "_validate_resident_grid" not in source
    assert "multi_processor_count" not in run_body


def test_occupancy_exchange_records_both_exact_kernel_roles():
    evidence = persistent_benchmark._decode_occupancy_exchange(
        {
            0: (
                b'{"role":"producer","occupancy":{"planned_ctas":16,'
                b'"cooperative_cta_capacity":296}}'
            ),
            1: (
                b'{"role":"receiver","occupancy":{"planned_ctas":16,'
                b'"cooperative_cta_capacity":148}}'
            ),
        }
    )

    assert evidence == {
        "producer": {"planned_ctas": 16, "cooperative_cta_capacity": 296},
        "receiver": {"planned_ctas": 16, "cooperative_cta_capacity": 148},
    }
    with pytest.raises(ValueError, match="producer and receiver"):
        persistent_benchmark._decode_occupancy_exchange({0: b"{}"})
    with pytest.raises(ValueError, match="receiver occupancy document"):
        persistent_benchmark._decode_occupancy_exchange(
            {
                0: b'{"role":"producer","occupancy":{}}',
                1: b'{"role":"producer","occupancy":{}}',
            }
        )


def test_legacy_mapped_flag_is_a_typed_compatibility_noop():
    default = PersistentCase(4096, 4, 1, 1, "warp", "mapped", 0, 1)
    legacy = PersistentCase(
        4096, 4, 1, 1, "warp", "mapped", 0, 1, allow_unverified_mapped=True
    )
    wait = PersistentCase(
        4096, 4, 1, 1, "warp", "wait", 0, 1, allow_unverified_mapped=True
    )

    assert default.allow_unverified_mapped is False
    assert legacy.allow_unverified_mapped is True
    assert wait.allow_unverified_mapped is True
    default_result = summarize_timestamps(
        default, [1, 2, 3, 4], source_device=0, target_device=1
    )
    legacy_result = summarize_timestamps(
        legacy, [1, 2, 3, 4], source_device=0, target_device=1
    )
    assert legacy_result["protocol"] == default_result["protocol"]
    assert legacy_result["compatibility"]["allow_unverified_mapped"] == {
        "value": True,
        "deprecated": True,
        "affects_execution": False,
    }
    with pytest.raises(TypeError, match="allow_unverified_mapped must be bool"):
        PersistentCase(
            4096,
            4,
            1,
            1,
            "warp",
            "mapped",
            0,
            1,
            allow_unverified_mapped=1,
        )


def test_source_generation_is_uint64_and_advances_after_credit():
    assert source_generation(0) == 0
    assert source_generation(255) == 255
    assert source_generation(256) == 256
    assert source_generation(2**64) == 0
    with pytest.raises(ValueError, match="iteration"):
        source_generation(-1)


def test_generation_pattern_and_validation_cover_every_batch_slot(monkeypatch):
    case = PersistentCase(16, 3, 2, 1, "thread", "wait", 0, 1)
    monkeypatch.setattr(
        persistent_benchmark,
        "_pattern",
        lambda size, _device: torch.arange(size, dtype=torch.uint8),
    )

    payload = persistent_benchmark._payload_generation_pattern(case, 7, 0)
    assert payload.view(torch.uint64)[:: case.size // 8].tolist() == [7] * 6
    persistent_benchmark._validate_generation_words("payload", payload, case, 7)

    payload.view(torch.uint64)[4 * case.size // 8] = 6
    with pytest.raises(RuntimeError, match="generation words"):
        persistent_benchmark._validate_generation_words("payload", payload, case, 7)


def test_failed_receiver_publication_synchronizes_launched_kernel():
    calls = []

    def fail_publish():
        calls.append("publish")
        raise RuntimeError("control plane failed")

    with pytest.raises(RuntimeError, match="control plane failed"):
        persistent_benchmark._launch_then_publish_and_synchronize(
            lambda: calls.append("launch"),
            fail_publish,
            lambda: calls.append("synchronize"),
        )

    assert calls == ["launch", "publish", "synchronize"]


def test_failed_receiver_launch_does_not_synchronize_unlaunched_kernel():
    calls = []

    def fail_launch():
        calls.append("launch")
        raise RuntimeError("launch failed")

    with pytest.raises(RuntimeError, match="launch failed"):
        persistent_benchmark._launch_then_publish_and_synchronize(
            fail_launch,
            lambda: calls.append("publish"),
            lambda: calls.append("synchronize"),
        )

    assert calls == ["launch"]


def test_defer_cadence_matches_elastic_ep_doorbells():
    assert EP_DOORBELL_INTERVAL == 4
    assert [defer_put_for_slot(slot) for slot in range(10)] == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="slot"):
        defer_put_for_slot(-1)


def test_percentiles_match_nixlbench_upper_order_statistic():
    samples = [40, 10, 30, 20]
    assert nixlbench_percentile(samples, 0.0) == 10
    assert nixlbench_percentile(samples, 0.5) == 30
    assert nixlbench_percentile(samples, 0.9) == 40
    assert nixlbench_percentile(samples, 0.99) == 40
    assert nixlbench_percentile(samples, 1.0) == 40


@pytest.mark.parametrize("q", [-0.1, 1.1, float("nan"), float("inf")])
def test_nixlbench_percentile_rejects_invalid_quantiles(q):
    with pytest.raises(ValueError, match="q"):
        nixlbench_percentile([1], q)


def test_nixlbench_percentile_rejects_invalid_samples():
    with pytest.raises(ValueError, match="empty"):
        nixlbench_percentile([], 0.5)
    with pytest.raises(ValueError, match="non-negative integers"):
        nixlbench_percentile([1, -1], 0.5)


def test_summary_preserves_raw_gpu_time_and_distinguishes_completion():
    case = PersistentCase(1000, 2, 2, 2, "thread", "defer", 1, 2)
    # Iteration-major, then group-major: start, API done, credit, cycle boundary.
    timestamps = [
        100,
        120,
        200,
        250,
        110,
        140,
        230,
        260,
        300,
        325,
        410,
        450,
        305,
        345,
        425,
        470,
    ]

    occupancy = {
        "producer": {"active_ctas_per_sm": 2, "planned_ctas": 1},
        "receiver": {"active_ctas_per_sm": 1, "planned_ctas": 1},
    }
    result = summarize_timestamps(
        case,
        timestamps,
        source_device=0,
        target_device=1,
        backend_parameters={"ucx_num_device_channels": "2"},
        cooperative_occupancy=occupancy,
    )

    assert result["producer_api"]["samples_ns"] == [20, 30, 25, 40]
    assert result["safe_source_reuse_round_trip"]["samples_ns"] == [100, 120, 110, 120]
    assert result["full_iteration_cycle"]["samples_ns"] == [150, 150, 150, 165]
    assert result["iteration_cohort_producer_api_span"]["samples_ns"] == [40, 45]
    assert result["iteration_cohort_safe_source_reuse_span"]["samples_ns"] == [
        130,
        125,
    ]
    assert result["iteration_cohort_full_cycle_span"]["samples_ns"] == [160, 170]
    assert result["steady_state_run"] == {
        "duration_ns": 370,
        "logical_bytes": 8000,
        "logical_GBps": pytest.approx(8000 / 370),
        "semantics": (
            "all measured group-iterations divided by the interval from the "
            "earliest first measured start to the latest final cycle boundary"
        ),
    }
    assert result["producer_api"]["p50_ns"] == 30
    assert result["safe_source_reuse_round_trip"]["p90_ns"] == 120
    assert result["timing"]["cpu_synchronization_inside_measured_loop"] is False
    assert result["timing"]["publish_atomic_excluded_from_producer_api"] is True
    assert result["timing"]["publish_signal_excluded_from_producer_api"] is True
    assert result["timing"]["device_operation_timeout_ns"] is None
    assert result["timing"]["credit_wait_mode"] == "timer-free; externally bounded"
    assert result["timing"]["timeout_clock_check_period_failed_loads"] is None
    assert result["launch"]["cooperative"] is True
    assert result["launch"]["occupancy"] == occupancy
    assert result["metric_definitions"]["raw_gpu_timestamps_ns_axes"] == (
        "[iteration][group][timestamp]"
    )
    assert result["metric_definitions"]["percentile_rule"] == (
        "sorted[min(int(sample_count * q), sample_count - 1)]"
    )
    assert result["protocol"]["defer_put_cadence"] == (
        "DEFER,DEFER,DEFER,NONE; repeat; final atomic NONE"
    )
    assert result["agent_configuration"] == {
        "enable_progress_thread": True,
        "thread_sync": "NIXL_THREAD_SYNC_DEFAULT",
        "ucx_num_workers": 2,
        "ucx_post_threads": 0,
    }
    assert result["nixlbench_alignment"]["timer_boundary_comparable"] is False
    assert result["nixlbench_alignment"]["workload_equivalent"] is False
    assert result["validation"]["correctness"] == "PASS"
    assert result["validation"]["source_generation_layout"] == (
        "first uint64 word of every batch slot"
    )
    assert result["validation"]["source_generation_words_per_group"] == 2
    assert result["validation"]["source_generation_words_total"] == 4
    assert result["validation"]["final_target_generation"] == 2
    assert result["validation"]["final_source_generation"] == 3
    assert result["backend_parameters"]["ucx_num_device_channels"] == "2"
    assert result["raw_gpu_timestamps_ns"][0][1] == [110, 140, 230, 260]


def test_summary_rejects_missing_or_non_monotonic_samples():
    case = PersistentCase(64, 1, 1, 1, "thread", "wait", 0, 1)
    with pytest.raises(ValueError, match="expected 4"):
        summarize_timestamps(case, [1, 2], source_device=0, target_device=1)
    with pytest.raises(RuntimeError, match="ordering"):
        summarize_timestamps(case, [10, 9, 11, 12], source_device=0, target_device=1)
    with pytest.raises(ValueError, match="non-negative"):
        summarize_timestamps(case, [0, 1, 2, -1], source_device=0, target_device=1)


def test_zero_duration_reports_unbounded_rate_without_dividing_by_zero():
    case = PersistentCase(64, 1, 1, 1, "thread", "wait", 0, 1)
    result = summarize_timestamps(case, [7, 7, 7, 7], source_device=0, target_device=1)
    assert math.isinf(result["producer_api"]["logical_GBps_at_p50"])
    assert math.isinf(result["steady_state_run"]["logical_GBps"])
    assert result["nixlbench_alignment"]["timer_boundary_comparable"] is True


def test_diagnostic_timeout_metadata_names_its_clock_sampling_period():
    case = PersistentCase(64, 1, 1, 1, "thread", "wait", 0, 1, device_timeout_ns=123)
    result = summarize_timestamps(case, [1, 2, 3, 4], source_device=0, target_device=1)

    assert result["timing"]["device_operation_timeout_ns"] == 123
    assert result["timing"]["credit_wait_mode"] == "diagnostic device timeout"
    assert result["timing"]["timeout_clock_check_period_failed_loads"] == 256


def test_mapped_summary_reports_completed_direct_path():
    case = PersistentCase(64, 3, 1, 1, "warp", "mapped", 0, 1)
    result = summarize_timestamps(
        case, [10, 20, 40, 50], source_device=0, target_device=1
    )

    assert result["schema_version"] == 6
    assert result["case"] == "nixl_cute_persistent_transfer"
    assert result["producer_api"]["semantics"].startswith(
        "one completed warp-vectorized direct mapped copy"
    )
    assert result["protocol"] == {
        "data_path": "direct_mapped_peer_memory",
        "mapped_load_policy": "coherent global L1::no_allocate loads",
        "mapped_pointer_resolution": (
            "once per group before the persistent measured loop"
        ),
        "mapped_batch_copy": (
            "one contiguous copy and one warp barrier per group-iteration"
        ),
        "publish": "lane-zero system-release store after mapped-copy warp barrier",
        "defer_put_cadence": "not applicable",
        "receiver": "system-scope acquire load of completion counter",
        "credit": "lane-zero system-release store through mapped peer counter",
        "source_reuse": "only after system-scope acquire observes returned credit",
        "abort": (
            "bit 62 of the ordered completion/credit counter; healthy counters "
            "are capped at signed 32-bit iterations"
        ),
        "mapping_contract": (
            "every descriptor must yield a non-null process-local nixlGetPtr "
            "base; mapped addresses use that base plus descriptor-relative "
            "offsets, numeric equality with the owner VA is not required, and "
            "every directed peer pair is qualified for native GPU atomics"
        ),
    }
    assert result["compatibility"] == {
        "allow_unverified_mapped": {
            "value": False,
            "deprecated": True,
            "affects_execution": False,
        }
    }
    assert result["channel_mapping"] == "not used by direct mapped path"
    assert "pointer resolution" in result["producer_api"]["semantics"]
    assert result["timing"]["publish_atomic_excluded_from_producer_api"] is False
    assert result["timing"]["publish_signal_excluded_from_producer_api"] is True
    assert result["nixlbench_alignment"]["timer_boundary_comparable"] is False
    assert result["validation"]["source_generation_words_per_group"] == 3
    assert result["validation"]["source_generation_words_total"] == 3
    assert result["validation"]["final_target_generation"] == 0
    assert result["validation"]["final_source_generation"] == 1


def test_persistent_hot_loop_has_required_fence_timeout_and_abort_protocol():
    source = (
        Path(__file__).parents[2]
        / "examples"
        / "python"
        / "cute"
        / "benchmark_persistent.py"
    ).read_text()
    producer = source.split("def _persistent_producer_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_persistent_producer", 1
    )[0]
    receiver = source.split("def _persistent_receiver_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_persistent_receiver", 1
    )[0]

    assert "nixl_cute.fence_release_system()" in producer
    assert "wait_acquire_system_u64_until(" not in producer
    assert "wait_acquire_system_u64(" in producer
    assert "wait_acquire_system_u64(" in receiver
    assert "wait_acquire_system_u64_for(" in producer
    assert "wait_acquire_system_u64_for(" in receiver
    assert "batch_size * size" in producer
    assert "started = cute.arch.shuffle_sync" not in producer
    assert producer.count("if iteration >= warmup:") == 4
    assert producer.count("nixl_cute.globaltimer_ns()") == 4
    assert producer.index("api_done = cutlass.Uint64(0)") < producer.index(
        "previous_sample ="
    )
    assert "cutlass.Uint64(_ABORT_COUNTER_BIT)" not in producer
    assert "cutlass.Uint64(_ABORT_COUNTER_BIT)" not in receiver
    assert producer.count("abort_counter_bit = cutlass.Uint64(1 << 62)") == 1
    assert receiver.count("abort_counter_bit = cutlass.Uint64(1 << 62)") == 1
    assert producer.count("abort_counter_bit") == 4
    assert receiver.count("abort_counter_bit") == 4
    assert "while running & (iteration" in producer
    assert "while running & (iteration" in receiver
    assert "expected_remote_" not in producer
    assert "expected_remote_" not in receiver
    assert "allow_unverified_mapped" not in producer
    assert "allow_unverified_mapped" not in receiver
    assert "NIXL_ERR_MISMATCH" not in producer
    assert "NIXL_ERR_MISMATCH" not in receiver
    assert producer.count("nixl_cute.get_ptr(remote, index=") == 2
    assert receiver.count("nixl_cute.get_ptr(remote, index=") == 1
    assert producer.count("mapped_counter_base + counter_offset") == 1
    assert receiver.count("mapped_counter_base + counter_offset") == 1
    assert "mapped_payload_base" in producer
    assert "+ group_offset" in producer
    # CuTe DSL treats a Python name reused across compile-time branches as a
    # possible loop-carried value. The generation-update loop must not reuse
    # the request/defer branch's ``slot`` induction variable: doing so makes
    # the mapped specialization fail at JIT time with a None-to-Int32 change.
    assert "for generation_slot in cutlass.range(batch_size, unroll=1):" in producer
    assert "generation_slot * (size // 8)" in producer
    assert "for request_slot in cutlass.range(batch_size, unroll=1):" in producer
    assert "request_offset = group_offset + cutlass.Uint64(" in producer
    assert "request_status = nixl_cute.put(" in producer
    assert "for deferred_slot in" not in producer
    assert "deferred_offset = group_offset + cutlass.Uint64(" in producer
    assert "deferred_status = nixl_cute.put_post(" in producer
    assert "tail_deferred_offset = group_offset + cutlass.Uint64(" in producer
    assert "tail_deferred_status = nixl_cute.put_post(" in producer
    assert "for slot in cutlass.range(batch_size, unroll=1):" not in producer


def test_mapped_mode_has_zero_hot_path_host_progress_and_native_atomic_preflight():
    source = Path(persistent_benchmark.__file__).read_text(encoding="utf-8")

    assert source.count("require_peer_native_atomics(devices)") == 1
    assert 'if case.mode == "mapped"' in source
    assert 'enable_prog_thread=case.mode != "mapped"' in source
    assert '"native_peer_atomics": native_atomic_preflight' in source
    set_device = source.index("torch.cuda.set_device(device)")
    native_preflight = source.index(
        "nixl_cute.require_peer_native_atomics(devices)", set_device
    )
    stream_create = source.index("torch.cuda.Stream(device=device)", native_preflight)
    registration = source.index("agent.register_memory(registered", stream_create)
    setup_handshake = source.index("complete_ucx_setup_handshake(", registration)
    measured_launch = source.index("compiled(", registration)
    assert (
        set_device
        < native_preflight
        < stream_create
        < registration
        < setup_handshake
        < measured_launch
    )
    assert 'control.barrier("connected")' not in source


def test_exact_view_preflight_accepts_unequal_process_local_virtual_addresses():
    evidence = persistent_benchmark._classify_mapped_preflight(
        (0, persistent_benchmark._STATUS_MISMATCH),
        rank=0,
        allow_unverified_mapped=False,
    )

    assert evidence["remote_descriptors"] == {
        "payload": {
            "index": 0,
            "classification": "owner_va_coincident",
            "status": 0,
        },
        "counter": {
            "index": 1,
            "classification": "translated_process_local_va",
            "status": persistent_benchmark._STATUS_MISMATCH,
        },
    }
    assert evidence["all_remote_descriptors_mapped"] is True
    assert evidence["translated_va_observed"] is True


@pytest.mark.parametrize(
    "statuses,match", [((-9, 0), "payload descriptor"), ((0, -9), "counter descriptor")]
)
def test_exact_view_preflight_rejects_each_null_descriptor(statuses, match):
    with pytest.raises(RuntimeError, match=match):
        persistent_benchmark._classify_mapped_preflight(
            statuses,
            rank=1,
            allow_unverified_mapped=True,
        )


def test_exact_view_preflight_rejects_unexpected_or_ill_typed_statuses():
    with pytest.raises(RuntimeError, match="unexpected mapped-preflight status"):
        persistent_benchmark._classify_mapped_preflight(
            (0, -3), rank=0, allow_unverified_mapped=False
        )
    with pytest.raises(TypeError, match="statuses must be integers"):
        persistent_benchmark._classify_mapped_preflight(
            (0, False), rank=0, allow_unverified_mapped=False
        )
    with pytest.raises(ValueError, match="two descriptors"):
        persistent_benchmark._classify_mapped_preflight(
            (0,), rank=0, allow_unverified_mapped=False
        )


def test_exact_view_preflight_envelope_combines_both_directed_proofs():
    evidence0 = persistent_benchmark._classify_mapped_preflight(
        (0, persistent_benchmark._STATUS_MISMATCH),
        rank=0,
        allow_unverified_mapped=False,
    )
    evidence1 = persistent_benchmark._classify_mapped_preflight(
        (persistent_benchmark._STATUS_MISMATCH, 0),
        rank=1,
        allow_unverified_mapped=False,
    )
    failures, combined = persistent_benchmark._decode_mapped_preflight_exchange(
        {
            0: persistent_benchmark._mapped_preflight_status_payload(
                0, None, evidence0
            ),
            1: persistent_benchmark._mapped_preflight_status_payload(
                1, None, evidence1
            ),
        }
    )

    assert failures == ()
    assert combined is not None
    assert combined["descriptor_roles"] == ["payload", "counter"]
    assert combined["all_remote_descriptors_mapped"] is True
    assert combined["translated_va_observed"] is True
    assert combined["by_rank"] == {"0": evidence0, "1": evidence1}


def test_exact_view_preflight_envelope_converges_on_any_rank_failure():
    evidence0 = persistent_benchmark._classify_mapped_preflight(
        (0, 0), rank=0, allow_unverified_mapped=False
    )
    failures, combined = persistent_benchmark._decode_mapped_preflight_exchange(
        {
            0: persistent_benchmark._mapped_preflight_status_payload(
                0, None, evidence0
            ),
            1: persistent_benchmark._mapped_preflight_status_payload(
                1, RuntimeError("counter mapping is null"), None
            ),
        }
    )

    assert combined is None
    assert len(failures) == 1
    assert failures[0]["rank"] == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert failures[0]["message"] == "counter mapping is null"


@pytest.mark.parametrize(
    "bad_rank_zero",
    [
        b'{"rank":false,"ok":false,"error_type":"RuntimeError","message":"x"}',
        b'{"rank":0,"ok":1,"evidence":{}}',
        b'{"rank":0,"ok":true,"evidence":{},"unexpected":0}',
    ],
)
def test_exact_view_preflight_envelope_fails_closed_on_adversarial_input(
    bad_rank_zero,
):
    evidence1 = persistent_benchmark._classify_mapped_preflight(
        (0, 0), rank=1, allow_unverified_mapped=False
    )
    with pytest.raises(RuntimeError, match="rank 0 published"):
        persistent_benchmark._decode_mapped_preflight_exchange(
            {
                0: bad_rank_zero,
                1: persistent_benchmark._mapped_preflight_status_payload(
                    1, None, evidence1
                ),
            }
        )


def test_exact_view_preflight_is_setup_only_and_precedes_both_persistent_grids():
    source = Path(persistent_benchmark.__file__).read_text(encoding="utf-8")
    worker = source.split("def _worker", 1)[1].split("\ndef run(", 1)[0]
    producer = source.split("def _persistent_producer_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_persistent_producer", 1
    )[0]
    receiver = source.split("def _persistent_receiver_kernel", 1)[1].split(
        "@cute.jit\n    def _launch_persistent_receiver", 1
    )[0]

    preflight_compile = worker.index("compiled_preflight = _compile_mapped_preflight")
    preflight_launch = worker.index("compiled_preflight(", preflight_compile)
    setup_drain = worker.index("stream.synchronize()", preflight_launch)
    status_exchange = worker.index('"mapped-preflight-status"', setup_drain)
    producer_compile = worker.index("_compile_producer(", status_exchange)
    receiver_compile = worker.index("_compile_receiver(", status_exchange)
    receiver_launch = worker.index(
        "_launch_then_publish_and_synchronize(", status_exchange
    )
    producer_launch = worker.index("compiled(\n", receiver_launch)

    assert preflight_launch < setup_drain < status_exchange
    assert status_exchange < producer_compile < receiver_launch
    assert status_exchange < receiver_compile < producer_launch
    assert worker[preflight_launch:status_exchange].count("stream.synchronize()") == 1
    assert "_mapped_preflight" not in producer
    assert "_mapped_preflight" not in receiver
    assert '"mapped_preflight": mapped_preflight_evidence' in worker


def test_preflight_rejection_releases_views_before_owner_teardown():
    source = Path(persistent_benchmark.__file__).read_text(encoding="utf-8")
    worker = source.split("def _worker", 1)[1].split("\ndef run(", 1)[0]

    launch_guard = worker.index("if mapped_preflight_failure is None:")
    receiver_launch = worker.index(
        "_launch_then_publish_and_synchronize(", launch_guard
    )
    views_released = worker.index('control.barrier("views-released")', receiver_launch)
    rejection_raise = worker.index("raise mapped_preflight_failure", views_released)

    assert launch_guard < receiver_launch < views_released < rejection_raise
