# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import weakref

import pytest

import nixl._api as api
import nixl._bindings as bindings
from nixl._api import nixl_agent, nixl_device_view_handle


class _FakeNativeAgent:
    def __init__(self):
        self.calls = []
        self.next_handle = 0xCAFE

    def prepLocalMemView(self, descs, backends, worker_id):
        self.calls.append(("prepare_local", descs, backends, worker_id))
        return self.next_handle

    def prepRemoteMemView(
        self,
        remote_agent,
        descs,
        backends,
        worker_id,
        connection_timeout_ms,
    ):
        self.calls.append(
            (
                "prepare_remote",
                remote_agent,
                descs,
                backends,
                worker_id,
                connection_timeout_ms,
            )
        )
        return self.next_handle

    def prepMemView(self, descs, backends, worker_id, connection_timeout_ms):
        self.calls.append(
            (
                "prepare_composite_remote",
                descs,
                backends,
                worker_id,
                connection_timeout_ms,
            )
        )
        return self.next_handle

    def releaseMemView(self, handle):
        self.calls.append(("release", handle))

    def deregisterMem(self, descs, backends):
        self.calls.append(("deregister", descs, backends))

    def invalidateRemoteMD(self, remote_agent):
        self.calls.append(("invalidate_remote", remote_agent))

    def genNotif(self, remote_agent, message, backends=None):
        self.calls.append(("notify", remote_agent, message, backends))


@pytest.fixture
def device_agent(monkeypatch):
    monkeypatch.setattr(bindings, "HAVE_UCX_GPU_DEVICE_API", True)
    agent = object.__new__(nixl_agent)
    agent.agent = _FakeNativeAgent()
    agent.name = "local-agent"
    agent.backends = {"UCX": 17}
    agent.backend_mems = {"UCX": ["VRAM_SEG"]}
    agent.nixl_mems = {
        "DRAM": bindings.DRAM_SEG,
        "VRAM": bindings.VRAM_SEG,
    }
    agent._device_views = weakref.WeakSet()
    agent._device_view_lock = threading.RLock()
    return agent


def test_prepare_local_device_view_owns_and_releases(device_agent):
    source = [(0x1000, 64, 0)]
    view = device_agent.prepare_device_view(source, mem_type="VRAM", worker_id=2)

    assert isinstance(view, nixl_device_view_handle)
    assert view.valid
    assert view.is_valid
    assert view.handle == 0xCAFE
    assert view.count == 1
    assert view.descriptor_count == 1
    assert view.descriptor_lengths == (64,)
    assert isinstance(view.descriptor_lengths, tuple)
    assert view.kind == "local"
    assert view.remote_agents == frozenset()
    assert view.descriptor_addresses == (0x1000,)
    assert view._keepalive[0] is source
    assert view._keepalive[1] == tuple(source)
    assert device_agent.agent.calls[0][0] == "prepare_local"
    assert device_agent.agent.calls[0][2:] == ([17], 2)

    view.release()
    view.release()
    assert not view.valid
    assert device_agent.agent.calls.count(("release", 0xCAFE)) == 1
    with pytest.raises(RuntimeError, match="released"):
        _ = view.handle


def test_raw_release_invalidates_matching_owning_device_view(device_agent):
    source = [(0x1000, 64, 0)]
    view = device_agent.prepare_device_view(source, mem_type="VRAM")

    device_agent.release_mem_view(view.handle)

    assert not view.valid
    assert device_agent.agent.calls.count(("release", 0xCAFE)) == 1
    with pytest.raises(RuntimeError, match="released"):
        _ = view.handle
    view.release()
    assert device_agent.agent.calls.count(("release", 0xCAFE)) == 1


def test_raw_release_preserves_unmanaged_compatibility(device_agent):
    device_agent.release_mem_view(0xBEEF)

    assert device_agent.agent.calls[-1] == ("release", 0xBEEF)


def test_prepare_remote_device_view_forwards_timeout(device_agent):
    source = [(0x2000, 128, 1)]
    with device_agent.prepare_device_view(
        source,
        remote_agent="peer",
        mem_type="VRAM",
        worker_id=3,
        connection_timeout_ms=1234,
    ) as view:
        assert view.kind == "remote"
        assert view.remote_agents == frozenset({"peer"})
        assert view.valid
        call = device_agent.agent.calls[0]
        assert call[0] == "prepare_remote"
        assert call[1] == "peer"
        assert call[3:] == ([17], 3, 1234)

    assert not view.valid
    assert device_agent.agent.calls[-1] == ("release", 0xCAFE)
    view.close()
    assert device_agent.agent.calls.count(("release", 0xCAFE)) == 1


def test_prepare_composite_remote_device_view(device_agent):
    descs = bindings.nixlRemoteDList(
        bindings.VRAM_SEG,
        [
            (0x2000, 128, 1, "peer-a"),
            (0x3000, 256, 2, "peer-b"),
            (0x4000, 64, 1, "peer-a"),
        ],
    )

    view = device_agent.prepare_device_view(
        descs, mem_type="VRAM", worker_id=4, connection_timeout_ms=4321
    )

    assert view.kind == "remote"
    assert view.descriptor_count == 3
    assert view.descriptor_lengths == (128, 256, 64)
    assert view.descriptor_addresses == (0x2000, 0x3000, 0x4000)
    assert view.remote_agents == frozenset({"peer-a", "peer-b"})
    call = device_agent.agent.calls[0]
    assert call == (
        "prepare_composite_remote",
        descs,
        [17],
        4,
        4321,
    )

    for peer in ("peer-a", "peer-b"):
        with pytest.raises(RuntimeError, match="while a device view references it"):
            device_agent.remove_remote_agent(peer)
    view.release()
    device_agent.remove_remote_agent("peer-a")


def test_prepare_sparse_composite_remote_coordinates_fails_closed(device_agent):
    coordinates = [
        (0x2000, 128, 1, "peer-a"),
        (0x3000, 256, 2, None),
        (0x4000, 64, 3, "peer-c"),
    ]

    with pytest.raises(ValueError, match="local loopback descriptor"):
        device_agent.prepare_device_view(coordinates, mem_type="VRAM")


def test_prepare_all_null_composite_fails_closed(device_agent):
    coordinates = [
        (0x2000, 128, 1, None),
        (0x3000, 256, 2, None),
    ]

    with pytest.raises(ValueError, match="local loopback descriptor"):
        device_agent.prepare_device_view(coordinates, mem_type="VRAM")


def test_prepare_native_null_agent_composite_fails_closed(device_agent):
    descs = bindings.nixlRemoteDList(
        bindings.VRAM_SEG,
        [(0x2000, 128, 1, bindings.NIXL_NULL_AGENT)],
    )

    with pytest.raises(ValueError, match="local loopback descriptor"):
        device_agent.prepare_device_view(descs, mem_type="VRAM")


def test_composite_remote_view_rejects_redundant_agent(device_agent):
    descs = bindings.nixlRemoteDList(bindings.VRAM_SEG, [(0x2000, 128, 1, "peer-a")])
    with pytest.raises(ValueError, match="must be None"):
        device_agent.prepare_device_view(descs, remote_agent="peer-a", mem_type="VRAM")


def test_prepare_remote_device_view_has_bounded_default(device_agent):
    view = device_agent.prepare_device_view(
        [(0x2000, 128, 1)], remote_agent="peer", mem_type="VRAM"
    )

    assert device_agent.agent.calls[0][-1] == 30000
    view.release()


def test_prepare_single_remote_null_agent_fails_before_native_call(device_agent):
    with pytest.raises(ValueError, match="cannot be NULL_AGENT"):
        device_agent.prepare_device_view(
            [(0x2000, 128, 1)],
            remote_agent=bindings.NIXL_NULL_AGENT,
            mem_type="VRAM",
        )

    assert device_agent.agent.calls == []


def test_send_notif_wraps_backend_handle_as_sequence(device_agent):
    device_agent.send_notif("peer", b"ready", backend="UCX")

    assert device_agent.agent.calls[-1] == ("notify", "peer", b"ready", [17])


@pytest.mark.parametrize("worker_id", [-1, True, 1.5, "0", 2**100])
def test_prepare_device_view_rejects_invalid_worker(device_agent, worker_id):
    with pytest.raises(ValueError, match="worker_id"):
        device_agent.prepare_device_view(
            [(0x1000, 64, 0)], mem_type="VRAM", worker_id=worker_id
        )


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5, "100", 2**100])
def test_prepare_device_view_rejects_invalid_timeout(device_agent, timeout):
    with pytest.raises(ValueError, match="connection_timeout_ms"):
        device_agent.prepare_device_view(
            [(0x1000, 64, 0)],
            remote_agent="peer",
            mem_type="VRAM",
            connection_timeout_ms=timeout,
        )


def test_prepare_device_view_validation(device_agent, monkeypatch):
    with pytest.raises(ValueError, match="only backend='UCX'"):
        device_agent.prepare_device_view(
            [(0x1000, 64, 0)], mem_type="VRAM", backend="UCX_MOCK"
        )

    device_agent.backends.clear()
    with pytest.raises(RuntimeError, match="must be initialized"):
        device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    device_agent.backends["UCX"] = 17

    device_agent.backend_mems["UCX"] = []
    with pytest.raises(RuntimeError, match="does not support VRAM"):
        device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    device_agent.backend_mems["UCX"] = ["VRAM_SEG"]

    monkeypatch.setattr(bindings, "HAVE_UCX_GPU_DEVICE_API", False)
    with pytest.raises(RuntimeError, match="without the UCX GPU device API"):
        device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    monkeypatch.setattr(bindings, "HAVE_UCX_GPU_DEVICE_API", True)

    with pytest.raises(ValueError, match="mem_type='VRAM'"):
        device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="DRAM")
    with pytest.raises(ValueError, match="at least one"):
        device_agent.prepare_device_view([], mem_type="VRAM")
    with pytest.raises(ValueError, match="non-empty string"):
        device_agent.prepare_device_view(
            [(0x1000, 64, 0)], remote_agent="", mem_type="VRAM"
        )


def test_active_local_view_blocks_overlapping_deregistration(device_agent):
    view = device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    overlap = bindings.nixlRegDList(bindings.VRAM_SEG, [(0x1020, 32, 0, "")])
    disjoint = bindings.nixlRegDList(bindings.VRAM_SEG, [(0x2000, 32, 0, "")])

    with pytest.raises(RuntimeError, match="overlapping an active"):
        device_agent.deregister_memory(overlap)
    device_agent.deregister_memory(disjoint)

    view.release()
    device_agent.deregister_memory(overlap)
    assert [call[0] for call in device_agent.agent.calls].count("deregister") == 2


def test_active_single_peer_loopback_view_blocks_local_deregistration(device_agent):
    view = device_agent.prepare_device_view(
        [(0x1000, 64, 0)], remote_agent=device_agent.name, mem_type="VRAM"
    )
    overlap = bindings.nixlRegDList(bindings.VRAM_SEG, [(0x1020, 32, 0, "")])

    with pytest.raises(RuntimeError, match="overlapping an active"):
        device_agent.deregister_memory(overlap)

    view.release()
    device_agent.deregister_memory(overlap)


def test_mixed_composite_blocks_only_locally_owned_loopback_regions(device_agent):
    descs = bindings.nixlRemoteDList(
        bindings.VRAM_SEG,
        [
            (0x1000, 64, 0, device_agent.name),
            (0x2000, 64, 0, "peer"),
        ],
    )
    view = device_agent.prepare_device_view(descs, mem_type="VRAM")
    local_overlap = bindings.nixlRegDList(bindings.VRAM_SEG, [(0x1020, 32, 0, "")])
    remote_overlap = bindings.nixlRegDList(bindings.VRAM_SEG, [(0x2020, 32, 0, "")])

    with pytest.raises(RuntimeError, match="overlapping an active"):
        device_agent.deregister_memory(local_overlap)
    device_agent.deregister_memory(remote_overlap)

    view.release()
    device_agent.deregister_memory(local_overlap)


def test_active_remote_view_blocks_matching_agent_removal(device_agent):
    view = device_agent.prepare_device_view(
        [(0x2000, 64, 0)], remote_agent="peer", mem_type="VRAM"
    )

    with pytest.raises(RuntimeError, match="while a device view references it"):
        device_agent.remove_remote_agent("peer")
    device_agent.remove_remote_agent("other")

    view.release()
    device_agent.remove_remote_agent("peer")
    assert ("invalidate_remote", "peer") in device_agent.agent.calls


def test_live_view_finalizer_releases_and_warns(device_agent, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        api.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    view = device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    ref = weakref.ref(view)

    del view
    gc.collect()

    assert ref() is None
    assert device_agent.agent.calls[-1] == ("release", 0xCAFE)
    assert "CUDA work must be synchronized" in warnings[0]


def test_finalizer_still_releases_after_logging_teardown(device_agent, monkeypatch):
    view = device_agent.prepare_device_view([(0x1000, 64, 0)], mem_type="VRAM")
    ref = weakref.ref(view)
    monkeypatch.setattr(api, "logger", None)

    del view
    gc.collect()

    assert ref() is None
    assert device_agent.agent.calls[-1] == ("release", 0xCAFE)
