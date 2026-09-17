# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import pytest

from examples.python.cute._runtime import (
    DeviceRegion,
    FileControlPlane,
    PeerCoordinates,
)
from examples.python.cute.elastic_moe_ll import (
    ElasticLLCase,
    build_phase_specs,
    parse_membership_plan,
)
from examples.python.cute.elastic_moe_ll_process import (
    LifecycleEvent,
    PhaseCommand,
    ProcessIdentity,
    _await_command_choice,
    _CatastrophicLifecycleFailure,
    _coordinate_failed_phase_shutdown,
    _phase_remote_coordinates,
    _release_phase_views,
    _shutdown_failed_worker,
    _wait_for_markers,
    _WorkerHandle,
    _WorkerReportedFailure,
    _write_command,
    build_process_transitions,
    plan_peer_connection_delta,
    validate_lifecycle_trace,
)


def _case() -> ElasticLLCase:
    plan = parse_membership_plan("0,1;0;0,1", 2)
    return ElasticLLCase(
        max_ranks=2,
        experts_per_rank=2,
        num_tokens=4,
        top_k=2,
        hidden_size=8,
        warmup=1,
        iterations=2,
        membership_plan=plan,
    )


def test_process_plan_starts_stops_and_reincarnates_stable_slot():
    transitions = build_process_transitions(_case().membership_plan, 2, run_id="unit")

    assert transitions[0].joining_ranks == (0, 1)
    assert transitions[0].retiring_ranks == (1,)
    assert transitions[1].joining_ranks == ()
    assert transitions[1].continuing_ranks == (0,)
    assert transitions[2].joining_ranks == (1,)
    assert transitions[2].retiring_ranks == (0, 1)

    identities = [
        {identity.slot: identity for identity in transition.identities}
        for transition in transitions
    ]
    assert identities[0][0] == identities[1][0] == identities[2][0]
    assert identities[0][1].incarnation == 1
    assert identities[2][1].incarnation == 2
    assert identities[0][1].agent_name != identities[2][1].agent_name


def test_identity_rejects_nonpositive_incarnation():
    with pytest.raises(ValueError, match="incarnation"):
        ProcessIdentity(0, 0, "agent")


def test_phase_command_round_trips_and_rejects_stale_or_loose_schema():
    phase = build_phase_specs(_case())[2]
    command = PhaseCommand.from_phase("run-123", phase)
    assert PhaseCommand.from_bytes(command.to_bytes()) == command

    with pytest.raises(ValueError, match="unsupported phase command schema"):
        PhaseCommand.from_bytes(
            b'{"schema_version":1,"run_id":"x","generation":0,'
            b'"active_ranks":[0],"rank_incarnations":[1],"extra":0}'
        )
    with pytest.raises(ValueError, match="active_ranks"):
        PhaseCommand("x", 0, (True,), (1,))
    with pytest.raises(ValueError, match="positive incarnation"):
        PhaseCommand("x", 0, (0,), (0,))


def test_lifecycle_state_machine_requires_drain_release_unload_before_exit():
    continuing = (
        LifecycleEvent.CANDIDATE_PREPARED,
        LifecycleEvent.COMMIT_ACKNOWLEDGED,
        LifecycleEvent.GO_OBSERVED,
        LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
        LifecycleEvent.OLD_VIEW_RELEASED,
        LifecycleEvent.OBSOLETE_IDENTITIES_UNLOADED,
    )
    assert validate_lifecycle_trace(continuing, retiring=False) == continuing
    retiring = continuing + (
        LifecycleEvent.OWNER_DEREGISTERED,
        LifecycleEvent.PROCESS_EXITED,
    )
    assert validate_lifecycle_trace(retiring, retiring=True) == retiring

    with pytest.raises(ValueError, match="unsafe lifecycle"):
        validate_lifecycle_trace(
            (
                LifecycleEvent.CANDIDATE_PREPARED,
                LifecycleEvent.COMMIT_ACKNOWLEDGED,
                LifecycleEvent.GO_OBSERVED,
                LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
                LifecycleEvent.OWNER_DEREGISTERED,
            ),
            retiring=True,
        )
    with pytest.raises(ValueError, match="incomplete lifecycle"):
        validate_lifecycle_trace(continuing, retiring=True)


def test_singleton_phase_builds_a_safe_loopback_padded_remote_view():
    case = _case()
    phase = build_phase_specs(case)[1]
    local = DeviceRegion(0x1000, case.layout.arena_nbytes, 0)
    peers = {0: PeerCoordinates("rank0", (local,))}

    coordinates = _phase_remote_coordinates(case, phase, 0, local, peers)

    assert len(coordinates) == case.max_ranks
    assert all(coordinate[3] == "rank0" for coordinate in coordinates)
    assert all(coordinate[0] == local.address for coordinate in coordinates)
    assert all(coordinate[1] == case.layout.arena_nbytes for coordinate in coordinates)


def test_connection_plan_retains_only_unchanged_peer_incarnations():
    case = ElasticLLCase(
        max_ranks=2,
        experts_per_rank=1,
        num_tokens=2,
        top_k=1,
        hidden_size=8,
        warmup=0,
        iterations=1,
        membership_plan=parse_membership_plan("0,1;0,1;0", 2),
    )
    phases = build_phase_specs(case)
    peers = {
        0: PeerCoordinates("rank0-i1", (DeviceRegion(0x1000, 4096, 0),)),
        1: PeerCoordinates("rank1-i1", (DeviceRegion(0x2000, 4096, 1),)),
    }

    first = plan_peer_connection_delta(
        rank=0,
        generation=0,
        phases=phases,
        peers=peers,
        loaded_peers={},
    )
    assert first.new_slots == (1,)
    assert first.retained_slots == ()
    assert first.remove_after_phase == ()

    second = plan_peer_connection_delta(
        rank=0,
        generation=1,
        phases=phases,
        peers=peers,
        loaded_peers={1: peers[1]},
    )
    assert second.new_slots == ()
    assert second.retained_slots == (1,)
    assert second.remove_after_phase == (1,)

    changed = {
        **peers,
        1: PeerCoordinates("rank1-i2", (DeviceRegion(0x3000, 4096, 1),)),
    }
    with pytest.raises(RuntimeError, match="identity or registered coordinates"):
        plan_peer_connection_delta(
            rank=0,
            generation=1,
            phases=phases,
            peers=changed,
            loaded_peers={1: peers[1]},
        )


def test_connection_plan_removes_retirees_and_readds_rejoined_identity():
    case = _case()
    phases = build_phase_specs(case)
    generation_zero = {
        0: PeerCoordinates("rank0-i1", (DeviceRegion(0x1000, 4096, 0),)),
        1: PeerCoordinates("rank1-i1", (DeviceRegion(0x2000, 4096, 1),)),
    }
    retiring = plan_peer_connection_delta(
        rank=1,
        generation=0,
        phases=phases,
        peers=generation_zero,
        loaded_peers={},
    )
    assert retiring.new_slots == (0,)
    assert retiring.remove_after_phase == (0,)

    generation_two = {
        0: generation_zero[0],
        1: PeerCoordinates("rank1-i2", (DeviceRegion(0x3000, 4096, 1),)),
    }
    rejoined = plan_peer_connection_delta(
        rank=0,
        generation=2,
        phases=phases,
        peers=generation_two,
        loaded_peers={},
    )
    assert rejoined.new_slots == (1,)
    assert rejoined.retained_slots == ()
    assert rejoined.remove_after_phase == (1,)

    with pytest.raises(RuntimeError, match="identity or registered coordinates"):
        plan_peer_connection_delta(
            rank=0,
            generation=2,
            phases=phases,
            peers=generation_two,
            loaded_peers={1: generation_zero[1]},
        )


class _DeadProcess:
    exitcode = 9
    pid = 1234


class _ShutdownProcess:
    exitcode = None
    pid = 2345

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout):
        self.exitcode = 0


class _FlakyView:
    def __init__(self, failures, events=None):
        self.valid = True
        self.failures = failures
        self.release_calls = 0
        self.events = [] if events is None else events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def release(self):
        self.release_calls += 1
        self.events.append("release")
        if self.release_calls <= self.failures:
            raise RuntimeError("injected release failure")
        self.valid = False


class _CleanupAgent:
    def __init__(self, events):
        self.events = events

    def remove_remote_agent(self, name):
        self.events.append(f"remove:{name}")

    def deregister_memory(self, registration, backends):
        self.events.append(f"deregister:{registration}:{backends[0]}")


def test_coordinator_classifies_a_missing_post_go_worker_as_run_failure(tmp_path):
    identity = ProcessIdentity(0, 1, "agent")
    handle = _WorkerHandle(identity, _DeadProcess(), tmp_path / "no-error")

    with pytest.raises(RuntimeError, match="post-GO process loss"):
        _wait_for_markers(
            tmp_path,
            0,
            LifecycleEvent.OLD_PHASE_STREAM_DRAINED,
            (0,),
            {0: handle},
            0.1,
            go_issued=True,
        )


def test_live_worker_failure_is_reported_without_treating_owner_as_lost(tmp_path):
    error_path = tmp_path / "worker-error"
    error_path.write_text('{"stage":"candidate-prepare"}', encoding="utf-8")
    handle = _WorkerHandle(
        ProcessIdentity(0, 1, "agent"), _ShutdownProcess(), error_path
    )

    with pytest.raises(_WorkerReportedFailure, match="live stable slot 0"):
        _wait_for_markers(
            tmp_path,
            0,
            LifecycleEvent.CANDIDATE_PREPARED,
            (0,),
            {0: handle},
            0.1,
            go_issued=False,
        )

    assert handle.process.is_alive()


def test_command_choice_accepts_one_decision_and_rejects_conflicts(tmp_path):
    command = PhaseCommand("unit", 0, (0,), (1,))
    _write_command(tmp_path, "COMMIT", command)
    assert _await_command_choice(tmp_path, ("COMMIT", "ABORT"), command, 0.1) == (
        "COMMIT"
    )
    _write_command(tmp_path, "ABORT", command)
    assert _await_command_choice(tmp_path, ("COMMIT", "ABORT"), command, 0.1) == (
        "ABORT"
    )
    _write_command(tmp_path, "GO", command)
    with pytest.raises(RuntimeError, match="conflicting coordinator commands"):
        _await_command_choice(tmp_path, ("GO", "ABORT"), command, 0.1)


def test_device_view_release_retries_popped_exitstack_callback():
    phase_views = ExitStack()
    view = phase_views.enter_context(_FlakyView(failures=1))

    released, error = _release_phase_views(phase_views, view)

    assert released is True
    assert isinstance(error, RuntimeError)
    assert view.release_calls == 2
    assert view.valid is False


def test_persistent_device_view_release_failure_is_not_reported_safe():
    phase_views = ExitStack()
    view = phase_views.enter_context(_FlakyView(failures=2))

    released, error = _release_phase_views(phase_views, view)

    assert released is False
    assert "explicit NIXL device-view release retry" in str(error)
    assert view.release_calls == 2
    assert view.valid is True


def test_failed_worker_releases_unloads_and_publishes_safe_before_deregister(tmp_path):
    events = []
    phase_views = ExitStack()
    view = phase_views.enter_context(_FlakyView(failures=0, events=events))
    agent = _CleanupAgent(events)
    command = PhaseCommand("unit", 0, (0,), (1,))
    _write_command(tmp_path, "SHUTDOWN", command)
    control = FileControlPlane(tmp_path, rank=0, world_size=1, timeout_s=0.1)
    loaded_peers = {1: PeerCoordinates("peer", (DeviceRegion(0x1000, 4096, 1),))}
    error_path = tmp_path / "worker-error"

    _shutdown_failed_worker(
        agent=agent,
        registration="registration",
        control=control,
        command=command,
        phase_views=phase_views,
        remote_view=view,
        loaded_peers=loaded_peers,
        stream=cast(Any, object()),
        stream_drained=True,
        error_path=error_path,
        rank=0,
        incarnation=1,
        failure=RuntimeError("injected candidate failure"),
        stage="candidate-abort",
        timeout_s=0.1,
    )

    assert events == ["release", "remove:peer", "deregister:registration:UCX"]
    assert loaded_peers == {}
    assert error_path.exists()
    assert (tmp_path / "g0-SAFE_TO_SHUTDOWN.0").exists()


def test_coordinator_orders_abort_safe_quorum_then_shutdown(tmp_path):
    command = PhaseCommand("unit", 0, (0, 1), (1, 1))
    handles = {
        rank: _WorkerHandle(
            ProcessIdentity(rank, 1, f"agent-{rank}"),
            _ShutdownProcess(),
            tmp_path / f"no-error-{rank}",
        )
        for rank in command.active_ranks
    }
    for rank in command.active_ranks:
        (tmp_path / f"g0-SAFE_TO_SHUTDOWN.{rank}").write_bytes(b"safe")

    _coordinate_failed_phase_shutdown(
        directory=str(tmp_path),
        command=command,
        ranks=command.active_ranks,
        handles=handles,
        timeout_s=0.1,
        before_go=True,
    )

    assert (tmp_path / "g0-ABORT.command").exists()
    assert (tmp_path / "g0-SHUTDOWN.command").exists()
    assert handles == {}


def test_coordinator_never_issues_shutdown_without_every_safe_marker(tmp_path):
    command = PhaseCommand("unit", 0, (0, 1), (1, 1))
    handles = {
        rank: _WorkerHandle(
            ProcessIdentity(rank, 1, f"agent-{rank}"),
            _ShutdownProcess(),
            tmp_path / f"no-error-{rank}",
        )
        for rank in command.active_ranks
    }
    (tmp_path / "g0-SAFE_TO_SHUTDOWN.0").write_bytes(b"safe")

    with pytest.raises(_CatastrophicLifecycleFailure, match="shutdown safety"):
        _coordinate_failed_phase_shutdown(
            directory=str(tmp_path),
            command=command,
            ranks=command.active_ranks,
            handles=handles,
            timeout_s=0.01,
            before_go=True,
        )

    assert (tmp_path / "g0-ABORT.command").exists()
    assert not (tmp_path / "g0-SHUTDOWN.command").exists()


def test_process_harness_reuses_hot_kernel_and_has_one_measured_launch_site():
    source_path = (
        Path(__file__).parents[2]
        / "examples"
        / "python"
        / "cute"
        / "elastic_moe_ll_process.py"
    )
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_worker"
    )
    calls = [
        node.func.id
        for node in ast.walk(worker)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.count("_compile_kernels") == 1
    assert calls.count("_rank_codegen_dump_directory") == 1
    assert calls.count("compiled_main") == 1
    assert source.count("enable_prog_thread=False") == 1
    assert "enable_prog_thread=True" not in source
    assert source.count("require_peer_native_atomics(") == 1
    assert "devices, accessing_devices=(device,)" in source
    assert '"native_peer_atomics": native_atomic_preflight' in source
    codegen_directory = source.index("codegen_dump_dir = _rank_codegen_dump_directory")
    set_device = source.index("torch.cuda.set_device(device)", codegen_directory)
    native_preflight = source.index(
        "nixl_cute.require_peer_native_atomics(", set_device
    )
    stream_create = source.index("torch.cuda.Stream(device=device)", native_preflight)
    registration = source.index("agent.register_memory([tensors.arena]", stream_create)
    measured_launch = source.index("compiled_main(", registration)
    assert (
        codegen_directory
        < set_device
        < native_preflight
        < stream_create
        < registration
        < measured_launch
    )
    assert '"codegen_dump_dir": str(codegen_dump_dir)' in source
    go_calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_publish_event"
        and len(node.args) >= 4
        and isinstance(node.args[3], ast.Attribute)
        and node.args[3].attr == "GO_OBSERVED"
    ]
    assert len(go_calls) == 1
    barrier_keywords = [
        keyword.value for keyword in go_calls[0].keywords if keyword.arg == "barrier"
    ]
    assert len(barrier_keywords) == 1
    assert isinstance(barrier_keywords[0], ast.Constant)
    assert barrier_keywords[0].value is True
    compiled_main_call = next(
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compiled_main"
    )
    assert go_calls[0].lineno < compiled_main_call.lineno
    assert "@cute.kernel" not in source
    assert "sole measured launch" in source


def test_process_worker_allocation_does_not_force_a_host_stream_drain():
    source_path = (
        Path(__file__).parents[2]
        / "examples"
        / "python"
        / "cute"
        / "elastic_moe_ll_process.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    allocation = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_allocate_worker_tensors"
    )

    assert "stream.synchronize()" not in ast.unparse(allocation)


def test_process_candidate_failure_and_success_release_views_before_lifecycle_work():
    source_path = (
        Path(__file__).parents[2]
        / "examples"
        / "python"
        / "cute"
        / "elastic_moe_ll_process.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_worker"
    )
    worker_source = ast.unparse(worker)

    assert "phase_views = ExitStack()" in worker_source
    assert "phase_views.enter_context(agent.prepare_device_view" in worker_source
    assert worker_source.index("_stage_phase") < worker_source.index(
        "phase_views.enter_context(agent.prepare_device_view"
    )
    assert worker_source.index("agent.get_agent_metadata()") < worker_source.index(
        "stage='candidate-prepare'"
    )
    assert worker_source.index("complete_ucx_setup_handshake") < worker_source.index(
        "stage='candidate-prepare'"
    )
    assert "preflight-status" not in worker_source
    assert "candidate-views-released" not in worker_source
    assert worker_source.index("verbs=('COMMIT', 'ABORT')") < worker_source.index(
        "LifecycleEvent.COMMIT_ACKNOWLEDGED"
    )
    assert worker_source.index("verbs=('GO', 'ABORT')") < worker_source.index(
        "compiled_main(remote_view"
    )
    assert worker_source.index("LifecycleEvent.GO_OBSERVED") < worker_source.index(
        "compiled_main(remote_view"
    )
    assert worker_source.rindex(
        "released, release_error = _release_phase_views(phase_views, remote_view)"
    ) < worker_source.index("LifecycleEvent.OLD_VIEW_RELEASED")
    assert worker_source.index(
        "LifecycleEvent.OLD_VIEW_RELEASED"
    ) < worker_source.index("agent.remove_remote_agent(peer_name)")
    assert "for peer_rank in connection_delta.new_slots:" in worker_source
    assert "for peer_name in new_peer_names:" in worker_source
    assert "for peer_rank in connection_delta.remove_after_phase:" in worker_source
    assert "control_plane_exchange_skipped" in worker_source
    assert "notification_peer_ranks=connection_delta.new_slots" in worker_source
    assert "poll_callback=lambda: _raise_if_candidate_aborted" in worker_source
    assert "_tag(generation, 'connections-made')" not in worker_source
    assert (
        "LifecycleEvent.OLD_PHASE_STREAM_DRAINED, retiring=retiring, barrier=True"
        not in worker_source
    )
    assert "LifecycleEvent.OLD_VIEW_RELEASED, retiring=retiring, barrier=True" not in (
        worker_source
    )
    assert "unload_statuses = control.exchange" in worker_source
    assert "_phase_result_payload" in worker_source
    assert "validation-status" not in worker_source
    assert worker_source.index("host_results.enqueue_preflight") < worker_source.index(
        "stream.synchronize()"
    )
    assert worker_source.index("host_results.enqueue_main") < worker_source.rindex(
        "stream.synchronize()"
    )
    assert "_validate_statuses(host_results.statuses" in worker_source
    stage = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stage_phase"
    )
    stage_source = ast.unparse(stage)
    assert "tensors.statuses.zero_()" not in stage_source
    assert "if case.timing_mode != 'none':" in stage_source


def test_process_cli_defaults_to_uninstrumented_timing(monkeypatch):
    import examples.python.cute.elastic_moe_ll_process as example

    captured = {}

    def fake_run(*, devices, case, timeout_s):
        captured.update(devices=devices, case=case, timeout_s=timeout_s)

    monkeypatch.setattr(example, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["elastic_moe_ll_process.py"])
    example.main()

    assert captured["case"].timing_mode == "none"
    assert captured["case"].instrument_per_peer is False


def test_process_cli_matches_direct_sm100_geometry_and_explicit_override(monkeypatch):
    import examples.python.cute.elastic_moe_ll_process as example

    captured = {}

    def fake_run(*, devices, case, timeout_s):
        captured.update(devices=devices, case=case, timeout_s=timeout_s)

    monkeypatch.setattr(example, "run", fake_run)
    monkeypatch.setattr(example._ll, "_detect_common_sm", lambda devices: 100)
    monkeypatch.setattr(
        "sys.argv",
        [
            "elastic_moe_ll_process.py",
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

    monkeypatch.setattr(example._ll, "_detect_common_sm", fail_if_queried)
    monkeypatch.setattr(
        "sys.argv", ["elastic_moe_ll_process.py", "--warps-per-cta", "4"]
    )
    example.main()
    assert captured["case"].target_sm is None
    assert captured["case"].warps_per_cta == 4


def test_process_run_reconciles_auto_geometry_before_building_phases(monkeypatch):
    import examples.python.cute.elastic_moe_ll_process as example

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
    captured = {}

    def capture_case(resolved):
        captured["case"] = resolved
        return ()

    monkeypatch.setattr(example, "_validate_run_inputs", lambda *args: None)
    monkeypatch.setattr(example._ll, "_detect_common_sm", lambda devices: 100)
    monkeypatch.setattr(example, "build_phase_specs", capture_case)
    monkeypatch.setattr(
        example, "build_process_transitions", lambda *args, **kwargs: ()
    )

    result = example.run(devices=(0, 1), case=case, timeout_s=30.0)

    assert captured["case"].target_sm == 100
    assert captured["case"].warps_per_cta == 2
    assert result["phases"] == []
