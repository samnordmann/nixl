# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from examples.python.cute._runtime import (
    DeviceRegion,
    FileControlPlane,
    PeerCoordinates,
    _setup_handshake_payload,
    check_status,
    complete_ucx_setup_handshake,
    normalize_agent_name,
    wait_for_value,
    wait_until,
)


class _FakeAsyncNotificationNetwork:
    """Advance each sender's queued AMs only when that sender polls progress."""

    def __init__(self, polls_before_delivery):
        self._polls_before_delivery = dict(polls_before_delivery)
        self._outbound = {name: [] for name in polls_before_delivery}
        self._inbound = {name: [] for name in polls_before_delivery}
        self._lock = Lock()
        self.progress_calls = {name: 0 for name in polls_before_delivery}

    def send(self, source, destination, payload):
        with self._lock:
            remaining = self._polls_before_delivery[source]
            if remaining == 0:
                self._inbound.setdefault(destination, []).append((source, payload))
            else:
                self._outbound[source].append([remaining, destination, payload])

    def inject(self, source, destination, payload):
        with self._lock:
            self._inbound.setdefault(destination, []).append((source, payload))

    def poll(self, name):
        with self._lock:
            self.progress_calls[name] += 1
            still_pending = []
            for remaining, destination, payload in self._outbound[name]:
                remaining -= 1
                if remaining == 0:
                    self._inbound.setdefault(destination, []).append((name, payload))
                else:
                    still_pending.append([remaining, destination, payload])
            self._outbound[name] = still_pending
            messages = self._inbound.pop(name, [])
            self._inbound[name] = []
        result = {}
        for source, payload in messages:
            result.setdefault(source, []).append(payload)
        return result


class _FakeAsyncAgent:
    def __init__(self, name, network):
        self.name = name
        self._network = network
        self.notification_backends = []

    def send_notif(self, peer_name, payload, *, backend):
        assert backend == "UCX"
        self._network.send(self.name, peer_name, payload)

    def get_new_notifs(self, *, backends):
        self.notification_backends.append(tuple(backends))
        return self._network.poll(self.name)


def test_atomic_publish_and_await(tmp_path):
    producer = FileControlPlane(tmp_path, rank=0, world_size=2, timeout_s=1)
    consumer = FileControlPlane(tmp_path, rank=1, world_size=2, timeout_s=1)

    producer.publish("generation-3.metadata", b"opaque metadata")

    assert consumer.await_rank("generation-3.metadata", 0) == b"opaque metadata"
    assert not list(tmp_path.glob("*.tmp"))


def test_sparse_barrier_uses_one_bounded_deadline(tmp_path):
    rank_zero = FileControlPlane(
        tmp_path, rank=0, world_size=4, timeout_s=1, poll_interval_s=0.001
    )
    rank_three = FileControlPlane(
        tmp_path, rank=3, world_size=4, timeout_s=1, poll_interval_s=0.001
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(control.barrier, "generation-4.commit", (0, 3))
            for control in (rank_zero, rank_three)
        ]
        for future in futures:
            assert future.result(timeout=2) is None


def test_exchange_collects_ranked_payloads(tmp_path):
    controls = [
        FileControlPlane(tmp_path, rank=rank, world_size=2, timeout_s=1)
        for rank in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(control.exchange, "coordinates", f"rank-{rank}".encode())
            for rank, control in enumerate(controls)
        ]

    assert [future.result() for future in futures] == [
        {0: b"rank-0", 1: b"rank-1"},
        {0: b"rank-0", 1: b"rank-1"},
    ]


def test_barrier_rejects_ambiguous_membership(tmp_path):
    control = FileControlPlane(tmp_path, rank=2, world_size=4)

    with pytest.raises(ValueError, match="duplicates"):
        control.barrier("phase", (0, 2, 2))
    with pytest.raises(ValueError, match="calling rank 2"):
        control.barrier("phase", (0, 3))
    with pytest.raises(ValueError, match="outside world size"):
        control.barrier("phase", (2, 4))


def test_ucx_setup_receipt_keeps_sender_progress_after_early_inbound_hello(tmp_path):
    names = {0: "async-rank-0", 1: "async-rank-1"}
    controls = {
        rank: FileControlPlane(
            tmp_path, rank=rank, world_size=2, timeout_s=1, poll_interval_s=0.001
        )
        for rank in names
    }
    # Rank 1's hello arrives immediately. Rank 0's outbound hello needs two
    # rank-0 progress polls: an inbound-only wait followed by a file barrier
    # would strand rank 1 after rank 0's first poll.
    network = _FakeAsyncNotificationNetwork({"async-rank-0": 2, "async-rank-1": 0})
    agents = {rank: _FakeAsyncAgent(name, network) for rank, name in names.items()}
    callback_calls = {rank: [] for rank in names}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                complete_ucx_setup_handshake,
                agents[rank],
                controls[rank],
                names,
                generation=7,
                nonce="asymmetric-race",
                timeout_s=1,
                poll_interval_s=0.001,
                poll_callback=lambda rank=rank: callback_calls[rank].append(True),
            )
            for rank in names
        ]
        assert [future.result(timeout=2) for future in futures] == [None] * len(names)

    assert network.progress_calls["async-rank-0"] >= 2
    assert all(callback_calls.values())
    assert all(
        backends == ("UCX",)
        for agent in agents.values()
        for backends in agent.notification_backends
    )
    assert len(list(tmp_path.glob("ucx-setup-receipt-g7-*"))) == 2


def test_ucx_setup_receipt_supports_sparse_generation_participants(tmp_path):
    names = {0: "sparse-rank-0", 2: "sparse-rank-2", 4: "sparse-rank-4"}
    # Mirror an elastic expansion: ranks 0 and 2 retain their connection while
    # rank 4 is new, so only the symmetric edges incident on rank 4 need AMs.
    new_peer_ranks = {0: (4,), 2: (4,), 4: (0, 2)}
    controls = {
        rank: FileControlPlane(
            tmp_path, rank=rank, world_size=5, timeout_s=1, poll_interval_s=0.001
        )
        for rank in names
    }
    network = _FakeAsyncNotificationNetwork({name: 1 for name in names.values()})
    agents = {rank: _FakeAsyncAgent(name, network) for rank, name in names.items()}

    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        futures = [
            executor.submit(
                complete_ucx_setup_handshake,
                agents[rank],
                controls[rank],
                names,
                generation=19,
                nonce="sparse-generation",
                notification_peer_ranks=new_peer_ranks[rank],
                timeout_s=1,
                poll_interval_s=0.001,
            )
            for rank in names
        ]
        assert [future.result(timeout=2) for future in futures] == [None] * len(names)

    receipts = list(tmp_path.glob("ucx-setup-receipt-g19-*"))
    assert sorted(int(path.suffix[1:]) for path in receipts) == [0, 2, 4]
    assert all(b'"generation":19' in path.read_bytes() for path in receipts)


def test_ucx_setup_receipt_rejects_live_asymmetric_new_edge_sets(tmp_path):
    names = {0: "asymmetric-rank-0", 1: "asymmetric-rank-1"}
    controls = {
        rank: FileControlPlane(
            tmp_path, rank=rank, world_size=2, timeout_s=1, poll_interval_s=0.001
        )
        for rank in names
    }
    network = _FakeAsyncNotificationNetwork({name: 0 for name in names.values()})
    agents = {rank: _FakeAsyncAgent(name, network) for rank, name in names.items()}

    # Rank 0 claims the edge is new while rank 1 claims it is retained.  Rank 1
    # must reject rank 0's unexpected hello, while rank 0 cannot converge from
    # rank 1's asymmetric receipt.
    with ThreadPoolExecutor(max_workers=2) as executor:
        rank_0 = executor.submit(
            complete_ucx_setup_handshake,
            agents[0],
            controls[0],
            names,
            generation=20,
            nonce="asymmetric-generation",
            notification_peer_ranks=(1,),
            timeout_s=0.05,
            poll_interval_s=0.001,
        )
        rank_1 = executor.submit(
            complete_ucx_setup_handshake,
            agents[1],
            controls[1],
            names,
            generation=20,
            nonce="asymmetric-generation",
            notification_peer_ranks=(),
            timeout_s=0.05,
            poll_interval_s=0.001,
        )
        with pytest.raises(TimeoutError, match="pending hello peers"):
            rank_0.result(timeout=1)
        with pytest.raises(RuntimeError, match="unexpected agent"):
            rank_1.result(timeout=1)


def test_ucx_setup_receipt_completes_for_singleton_participant(tmp_path):
    control = FileControlPlane(
        tmp_path, rank=2, world_size=4, timeout_s=1, poll_interval_s=0.001
    )
    network = _FakeAsyncNotificationNetwork({"singleton-rank-2": 0})
    agent = _FakeAsyncAgent("singleton-rank-2", network)

    complete_ucx_setup_handshake(
        agent,
        control,
        {2: "singleton-rank-2"},
        generation=21,
        nonce="singleton-generation",
        timeout_s=0.05,
        poll_interval_s=0.001,
    )

    receipts = list(tmp_path.glob("ucx-setup-receipt-g21-*.2"))
    assert len(receipts) == 1
    assert b'"observed_hello_source_ranks":[]' in receipts[0].read_bytes()
    assert network.progress_calls == {"singleton-rank-2": 1}


def test_ucx_setup_receipt_rejects_stale_nonce_before_publication(tmp_path):
    names = {0: "current-rank-0", 1: "current-rank-1"}
    control = FileControlPlane(
        tmp_path, rank=0, world_size=2, timeout_s=1, poll_interval_s=0.001
    )
    network = _FakeAsyncNotificationNetwork({name: 0 for name in names.values()})
    agent = _FakeAsyncAgent(names[0], network)
    network.inject(
        names[1],
        names[0],
        _setup_handshake_payload(
            kind="ucx-setup-hello",
            nonce="stale-nonce",
            generation=4,
            source_rank=1,
            source_name=names[1],
            destination_rank=0,
            destination_name=names[0],
        ),
    )

    with pytest.raises(RuntimeError, match="stale or malformed notification"):
        complete_ucx_setup_handshake(
            agent,
            control,
            names,
            generation=4,
            nonce="current-nonce",
            timeout_s=1,
            poll_interval_s=0.001,
        )

    assert not list(tmp_path.glob("ucx-setup-receipt-*"))


def test_ucx_setup_receipt_rejects_forged_asymmetric_receipt(tmp_path):
    names = {0: "receipt-rank-0", 1: "receipt-rank-1"}
    control = FileControlPlane(
        tmp_path, rank=0, world_size=2, timeout_s=1, poll_interval_s=0.001
    )
    network = _FakeAsyncNotificationNetwork({name: 0 for name in names.values()})
    agent = _FakeAsyncAgent(names[0], network)
    network.inject(
        names[1],
        names[0],
        _setup_handshake_payload(
            kind="ucx-setup-hello",
            nonce="receipt-validation",
            generation=11,
            source_rank=1,
            source_name=names[1],
            destination_rank=0,
            destination_name=names[0],
        ),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            complete_ucx_setup_handshake,
            agent,
            control,
            names,
            generation=11,
            nonce="receipt-validation",
            timeout_s=1,
            poll_interval_s=0.001,
        )
        local_receipt = wait_until(
            lambda: next(iter(tmp_path.glob("ucx-setup-receipt-g11-*.0")), None),
            timeout_s=1,
            description="local setup receipt",
            poll_interval_s=0.001,
        )
        peer_control = FileControlPlane(tmp_path, rank=1, world_size=2, timeout_s=1)
        peer_control.publish(
            local_receipt.name.rsplit(".", 1)[0],
            _setup_handshake_payload(
                kind="ucx-setup-receipt",
                nonce="receipt-validation",
                generation=11,
                source_rank=1,
                source_name=names[1],
                # Rank 1 falsely claims it did not observe rank 0's hello.
                observed_hello_source_ranks=(),
            ),
        )
        with pytest.raises(RuntimeError, match="stale, asymmetric, or malformed"):
            future.result(timeout=2)


def test_ucx_setup_receipt_propagates_poll_callback_failure(tmp_path):
    class SetupAborted(RuntimeError):
        pass

    control = FileControlPlane(tmp_path, rank=0, world_size=1, timeout_s=1)
    network = _FakeAsyncNotificationNetwork({"callback-rank-0": 0})
    agent = _FakeAsyncAgent("callback-rank-0", network)

    def abort_setup():
        raise SetupAborted("candidate aborted")

    with pytest.raises(SetupAborted, match="candidate aborted"):
        complete_ucx_setup_handshake(
            agent,
            control,
            {0: "callback-rank-0"},
            generation=0,
            nonce="callback",
            timeout_s=1,
            poll_callback=abort_setup,
        )
    assert network.progress_calls["callback-rank-0"] == 0
    assert not list(tmp_path.glob("ucx-setup-receipt-*"))


def test_ucx_setup_receipt_timeout_uses_one_deadline_and_reports_pending_peers(
    tmp_path,
):
    names = {0: "timeout-rank-0", 1: "timeout-rank-1"}
    control = FileControlPlane(
        tmp_path, rank=0, world_size=2, timeout_s=1, poll_interval_s=0.001
    )
    network = _FakeAsyncNotificationNetwork({name: 1 for name in names.values()})
    agent = _FakeAsyncAgent(names[0], network)
    callback_calls = []
    started = time.monotonic()

    with pytest.raises(TimeoutError) as failure:
        complete_ucx_setup_handshake(
            agent,
            control,
            names,
            generation=23,
            nonce="bounded-timeout",
            timeout_s=0.03,
            poll_interval_s=0.001,
            poll_callback=lambda: callback_calls.append(True),
        )

    elapsed = time.monotonic() - started
    message = str(failure.value)
    assert elapsed < 0.25
    assert callback_calls
    assert "pending hello peers [(1, 'timeout-rank-1')]" in message
    assert "pending receipt peers" in message
    assert not list(tmp_path.glob("ucx-setup-receipt-*"))


def test_ucx_setup_receipt_validates_callback_and_exact_rank_name_mapping(tmp_path):
    control = FileControlPlane(tmp_path, rank=0, world_size=4, timeout_s=1)
    network = _FakeAsyncNotificationNetwork({"agent-0": 0, "agent-2": 0})
    agent = _FakeAsyncAgent("agent-0", network)

    with pytest.raises(ValueError, match="agent name does not match"):
        complete_ucx_setup_handshake(
            agent,
            control,
            {0: "wrong-local", 2: "agent-2"},
            generation=0,
            nonce="mapping",
            timeout_s=1,
        )
    with pytest.raises(ValueError, match="agent names must be distinct"):
        complete_ucx_setup_handshake(
            agent,
            control,
            {0: "agent-0", 2: "agent-0"},
            generation=0,
            nonce="mapping",
            timeout_s=1,
        )
    with pytest.raises(ValueError, match="participant ranks"):
        complete_ucx_setup_handshake(
            agent,
            control,
            {0: "agent-0", 2: "agent-2"},
            generation=0,
            nonce="mapping",
            notification_peer_ranks=(1,),
            timeout_s=1,
        )
    with pytest.raises(TypeError, match="poll_callback"):
        complete_ucx_setup_handshake(
            agent,
            control,
            {0: "agent-0"},
            generation=0,
            nonce="mapping",
            timeout_s=1,
            poll_callback="not callable",
        )


def test_wait_timeout_reports_missing_ranks(tmp_path):
    control = FileControlPlane(
        tmp_path, rank=0, world_size=3, timeout_s=0.02, poll_interval_s=0.001
    )

    with pytest.raises(TimeoutError, match=r"ranks \[2\]"):
        control.await_rank("missing", 2)


def test_coordinate_payload_round_trip_and_validation():
    coordinates = PeerCoordinates(
        "expert-7",
        (
            DeviceRegion(0x1000, 4096, 0),
            DeviceRegion(0x4000, 8, 0),
        ),
    )

    restored = PeerCoordinates.from_bytes(coordinates.to_bytes())

    assert restored == coordinates
    assert restored.regions[1].descriptor == (0x4000, 8, 0)
    with pytest.raises(ValueError, match="unsupported schema"):
        PeerCoordinates.from_bytes(b'{"version":1}')
    with pytest.raises(TypeError, match="address"):
        DeviceRegion(True, 8, 0)


def test_bounded_polling_and_status_checks():
    observations = iter((0, 0, 7))
    assert (
        wait_for_value(
            lambda: next(observations),
            7,
            timeout_s=1,
            description="counter",
            poll_interval_s=0.001,
        )
        == 7
    )
    assert wait_until(lambda: "ready", timeout_s=1, description="readiness") == "ready"
    check_status(0, "PUT")
    with pytest.raises(RuntimeError, match=r"PUT failed: BACKEND \(-3\)"):
        check_status(-3, "PUT")
    with pytest.raises(TimeoutError, match="last observed value was 0"):
        wait_for_value(
            lambda: 0,
            1,
            timeout_s=0.02,
            description="counter",
            poll_interval_s=0.001,
        )


def test_loaded_agent_name_normalizes_binding_abi_variants():
    assert normalize_agent_name("peer") == "peer"
    assert normalize_agent_name(b"peer") == "peer"
    with pytest.raises(ValueError, match="UTF-8"):
        normalize_agent_name(b"\xff")
    with pytest.raises(TypeError, match="str or bytes"):
        normalize_agent_name(7)
