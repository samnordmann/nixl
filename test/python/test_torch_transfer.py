# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


transfer = pytest.importorskip(
    "torch.distributed._transfer",
    reason="requires the experimental PyTorch endpoint-transfer API",
)

import nixl.torch_transfer as torch_transfer_module  # noqa: E402
from nixl.torch_transfer import (  # noqa: E402
    NixlBackend,
    TORCH_TRANSFER_FACTORY_API_VERSION,
    _REQUEST_RETIRING,
    _backend_factory,
    _reject_overlapping_destination,
    _selected_layout,
    register_torch_backend,
)
from nixl._api import nixl_agent as _HighLevelNixlAgent  # noqa: E402


register_torch_backend()
register_torch_backend()  # Registration is intentionally idempotent.


def test_high_level_owned_handles_cover_creation_and_release_interrupts(monkeypatch):
    class FakeReleaseState:
        def __init__(self, handle):
            assert handle != 0
            self.handle = handle
            self.released = False

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.DEFAULT_COMM_PORT = 0
    fake_bindings.NIXL_THREAD_SYNC_NONE = 0
    fake_bindings.NIXL_THREAD_SYNC_STRICT = 1
    fake_bindings.NIXL_THREAD_SYNC_RW = 2
    fake_bindings.NIXL_THREAD_SYNC_DEFAULT = 3
    fake_bindings.NIXL_INIT_AGENT = "NIXL_INIT_AGENT"
    fake_bindings.nixlXferReleaseState = FakeReleaseState
    fake_bindings.nixlDlistReleaseState = FakeReleaseState
    package = sys.modules["nixl"]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)
    module_name = "nixl._release_token_test_api"
    api_path = Path(__file__).resolve().parents[2] / "src/api/python/_api.py"
    spec = importlib.util.spec_from_file_location(module_name, api_path)
    assert spec is not None and spec.loader is not None
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, api)
    spec.loader.exec_module(api)

    class FakeRawAgent:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)
            self.calls = 0
            self.native_calls = 0

        def _release_once(self, state, handle):
            assert state.handle == handle
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if outcome == "pre":
                # The binding was entered but its native ownership commit did
                # not run; retry must still call native release.
                raise KeyboardInterrupt()
            state.released = True
            self.native_calls += 1
            if outcome == "post":
                # Native release completed before the pending signal surfaced;
                # retry must be a no-op.
                raise KeyboardInterrupt()

        releaseXferReqOnce = _release_once
        releasedDlistHOnce = _release_once

    # Legacy raw handles remain supported. Their native release token makes
    # both sides of binding entry retry-safe, but raw creation itself is a
    # deliberately retained low-level compatibility API.
    for handle_type in (api.nixl_xfer_handle, api.nixl_prepped_dlist_handle):
        pre_agent = FakeRawAgent(["pre", "success"])
        pre_handle = handle_type(pre_agent, 0x1234)
        with pytest.raises(KeyboardInterrupt):
            pre_handle.release()
        assert not pre_handle._released
        pre_handle.release()
        pre_handle.release()
        assert pre_handle._released
        assert pre_agent.calls == 2
        assert pre_agent.native_calls == 1

        post_agent = FakeRawAgent(["post"])
        post_handle = handle_type(post_agent, 0x5678)
        with pytest.raises(KeyboardInterrupt):
            post_handle.release()
        assert post_handle._released
        post_handle.release()
        assert post_agent.calls == 1
        assert post_agent.native_calls == 1

    class FakeOwned:
        def __init__(self, value, outcomes=("success",)):
            self.value = value
            self.outcomes = list(outcomes)
            self.released = False
            self.calls = 0
            self.native_calls = 0

        def release(self):
            if self.released:
                return
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if outcome == "pre":
                raise KeyboardInterrupt()
            if outcome == "error":
                raise RuntimeError("native release is still active")
            self.released = True
            self.native_calls += 1
            if outcome == "post":
                raise KeyboardInterrupt()

    # The new native owner is retained by the high-level wrapper. Its release
    # operation commits in one binding frame, so pre-entry signals retry and
    # post-commit signals cannot double-release.
    for handle_type in (api.nixl_xfer_handle, api.nixl_prepped_dlist_handle):
        pre_owner = FakeOwned(0x9ABC, ("pre", "success"))
        pre_handle = handle_type(object(), pre_owner)
        with pytest.raises(KeyboardInterrupt):
            pre_handle.release()
        assert not pre_handle._released
        pre_handle.release()
        pre_handle.release()
        assert pre_handle._released
        assert pre_owner.calls == 2
        assert pre_owner.native_calls == 1

        post_owner = FakeOwned(0xDEF0, ("post",))
        post_handle = handle_type(object(), post_owner)
        with pytest.raises(KeyboardInterrupt):
            post_handle.release()
        assert post_handle._released
        post_handle.release()
        assert post_owner.calls == 1
        assert post_owner.native_calls == 1

    # The high-level agent durably retains every native owner from wrapper
    # construction until successful release. Finalization therefore remains
    # retry-safe even if BaseException interrupts the native release itself.
    for handle_type, registry_name in (
        (api.nixl_xfer_handle, "_leaked_xfer_handles"),
        (api.nixl_prepped_dlist_handle, "_leaked_dlist_handles"),
    ):
        for interrupted_outcome in ("error", "pre"):
            owner_agent = types.SimpleNamespace(
                agent=object(),
                _leaked_xfer_handles=[],
                _leaked_dlist_handles=[],
            )
            owner = FakeOwned(0xFA11, (interrupted_outcome, "success"))
            wrapper = handle_type(owner_agent, owner)
            assert getattr(owner_agent, registry_name) == [owner]
            wrapper.__del__()
            assert getattr(owner_agent, registry_name) == [owner]
            assert not owner.released
            api.nixl_agent.__del__(owner_agent)
            assert owner.released
            assert not getattr(owner_agent, registry_name)

        owner_agent = types.SimpleNamespace(
            agent=object(),
            _leaked_xfer_handles=[],
            _leaked_dlist_handles=[],
        )
        owner = FakeOwned(0xFA12, ("post",))
        wrapper = handle_type(owner_agent, owner)
        with pytest.raises(KeyboardInterrupt):
            wrapper.release()
        # The native commit happened, but durable ownership remains until a
        # retry observes the idempotent released state and removes the entry.
        assert getattr(owner_agent, registry_name) == [owner]
        wrapper.release()
        assert not getattr(owner_agent, registry_name)

    class FakeNativeAgent:
        def __init__(self):
            self.calls = []
            self.next_value = 0x1000
            self.unwound_owner = None

        def _owner(self, kind):
            self.next_value += 1
            owner = FakeOwned(self.next_value)
            self.calls.append((kind, owner))
            return owner

        def prepXferDlistOwned(self, *args):
            return self._owner("dlist")

        def makeXferReqOwned(self, *args):
            return self._owner("make")

        def createXferReqOwned(self, *args):
            return self._owner("create")

        def prepXferDlist(self, *args):
            raise AssertionError("high-level API must not use raw dlist creation")

        def makeXferReq(self, *args):
            raise AssertionError("high-level API must not use raw request creation")

        def createXferReq(self, *args):
            raise AssertionError("high-level API must not use raw request creation")

    native = FakeNativeAgent()
    high_level = object.__new__(api.nixl_agent)
    high_level.agent = native
    high_level._leaked_xfer_handles = []
    high_level._leaked_dlist_handles = []
    high_level.backends = {}
    high_level.nixl_mems = {"DRAM": "DRAM"}
    high_level.nixl_ops = {"WRITE": "WRITE"}
    strided = np.asarray([[0x1000, 8, 0, 8, 1]], dtype=np.uint64)

    local = high_level.prep_xfer_dlist("NIXL_INIT_AGENT", strided, mem_type="DRAM")
    remote = high_level.prep_xfer_dlist("remote", strided, mem_type="DRAM")
    made = high_level.make_prepped_xfer("WRITE", local, [0], remote, [0])
    combined = high_level.initialize_xfer("WRITE", object(), object(), "remote")

    assert [kind for kind, _ in native.calls] == [
        "dlist",
        "dlist",
        "make",
        "create",
    ]
    assert local._owner is native.calls[0][1]
    assert local._owner_agent is high_level
    assert remote._owner is native.calls[1][1]
    assert made._owner is native.calls[2][1]
    assert combined._owner is native.calls[3][1]

    # Model an exception during Python return conversion: native RAII owns the
    # created pointer before returning, so unwinding destroys/releases it even
    # though no high-level wrapper can be constructed.
    def interrupt_after_native_creation(*args):
        owner = native._owner("unwind")
        native.unwound_owner = owner
        try:
            raise KeyboardInterrupt()
        finally:
            owner.release()

    native.createXferReqOwned = interrupt_after_native_creation
    with pytest.raises(KeyboardInterrupt):
        high_level.initialize_xfer("WRITE", object(), object(), "remote")
    assert native.unwound_owner.released
    assert native.unwound_owner.native_calls == 1


def _native_name(endpoint_id: str, incarnation: str) -> str:
    endpoint_bytes = endpoint_id.encode()
    incarnation_bytes = incarnation.encode()
    identity = (
        len(endpoint_bytes).to_bytes(8, "big")
        + endpoint_bytes
        + len(incarnation_bytes).to_bytes(8, "big")
        + incarnation_bytes
    )
    return f"torch-transfer-{hashlib.sha256(identity).hexdigest()[:32]}"


_REMOTE_NATIVE_NAME = _native_name("remote-id", "remote-incarnation")
_LOCAL_ADDRESS = 0x10_0000_1000
_RAW_ADDRESS = 0x40_0000_4000


class _FakeDlist:
    def __init__(self) -> None:
        self.released = False
        self.release_calls = 0
        self.release_errors = []

    def release(self) -> None:
        self.release_calls += 1
        if self.release_errors:
            raise self.release_errors.pop(0)
        self.released = True


class _FakeRegDlist:
    def __init__(self, regions, memory_type) -> None:
        self.regions = list(regions)
        self.memory_type = memory_type

    def clear(self) -> None:
        self.regions.clear()


class _FakeDeregistrationReceipt:
    def __init__(self, agent, descriptor_list, backends) -> None:
        self.agent = agent
        self.descriptor_list = descriptor_list
        self.backends = backends
        self.completed = False
        self.native_attempts = 0

    def execute(self, agent) -> None:
        if agent is not self.agent:
            raise ValueError("receipt belongs to another agent")
        if self.completed:
            return
        self.native_attempts += 1
        agent.deregister_calls.append((self.descriptor_list, self.backends))
        error = agent.deregister_error
        if error is not None:
            if agent.deregister_commit_before_error:
                self.completed = True
            raise error
        self.completed = True


class _FakeXfer:
    def __init__(self, statuses: list[str], *, cancellable: bool = True) -> None:
        self._handle = id(self)
        self.statuses = statuses
        self.cancellable = cancellable
        self.released = False
        self.release_calls = 0
        self.release_errors = []
        self.post_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        if self.release_errors:
            raise self.release_errors.pop(0)
        if not self.cancellable and self.statuses[-1] == "PROC":
            raise RuntimeError("request is still active")
        self.released = True


class _FakeExecutionDispatch:
    def __init__(self) -> None:
        self._state = None
        self._retired = False
        self._closed = False
        self.failed = False
        self.failure_epoch = 0
        self.failure_message = ""
        self.fail_next = False
        self.post_calls = 0
        self.poll_calls = 0
        self.recycle_calls = 0
        self.close_calls = 0
        self.native_recycle_calls = 0
        self.notification_posts = []

    @property
    def active(self):
        return self._state is not None and not self._retired

    def start(self):
        if self._closed:
            raise RuntimeError("closed")
        self.post_calls += 1
        self._retired = False
        if self.fail_next:
            self.fail_next = False
            self.failed = True
            self.failure_epoch += 1
            self.failure_message = "injected native execution failure"
            self._state = transfer.WorkState.FAILED
        else:
            self.failed = False
            self._state = transfer.WorkState.COMPLETED

    def start_and_poll(self, *, max_polls, timeout_ns):
        assert max_polls >= 0
        assert timeout_ns is None or timeout_ns >= 0
        self.start()
        return self._state

    def start_with_notification(self, notification):
        self.notification_posts.append(notification)
        self.start()

    def start_and_poll_with_notification(self, notification, *, max_polls, timeout_ns):
        assert max_polls >= 0
        assert timeout_ns is None or timeout_ns >= 0
        self.notification_posts.append(notification)
        self.start()
        return self._state

    def poll_state(self):
        self.poll_calls += 1
        return self._state

    def poll_bounded(self, *, max_polls, timeout_ns, timeout_check_interval):
        assert max_polls > 0
        assert timeout_check_interval > 0
        self.poll_calls += 1
        return self._state

    def cancel(self):
        return False

    def recycle(self):
        self.recycle_calls += 1
        self._retired = True

    def close(self):
        self.close_calls += 1
        self._closed = True
        self._retired = True


class _FakeAgent:
    def __init__(self, remote_name=_REMOTE_NATIVE_NAME) -> None:
        self.remote_name = remote_name
        self.register_calls = []
        self.deregister_calls = []
        self.prep_calls = []
        self.make_calls = []
        self.removed_peers = []
        self.sent_notifications = []
        self.notifications = {}
        self.full_metadata_calls = 0
        self.partial_metadata_calls = []
        self.get_reg_descs_calls = []
        self.imported_metadata = []
        self.remote_names_by_metadata = {}
        self.import_error = None
        self.remove_error = None
        self.submit_status = "DONE"
        self.register_error = None
        self.deregister_error = None
        self.deregister_commit_before_error = False
        self.deregister_receipts = []
        self.send_error = None
        self.notification_error = None
        self.transfer_error = None
        self.check_error = None
        self.poll_statuses = ["DONE"]
        self.cancellable = True
        self.created_backends = []
        self.actions = []
        self.post_notifications = []

    def create_backend(self, backend, init_params):
        self.created_backends.append((backend, init_params))

    def register_memory(self, regions, *, mem_type, backends):
        descriptor_list = (
            regions
            if isinstance(regions, _FakeRegDlist)
            else _FakeRegDlist(regions, mem_type)
        )
        self.register_calls.append((descriptor_list.regions.copy(), mem_type, backends))
        if self.register_error is not None:
            raise self.register_error
        return descriptor_list

    def deregister_memory(self, descriptor_list, *, backends):
        receipt = self.prepare_deregister_memory(descriptor_list, backends=backends)
        return self.execute_deregister_memory(receipt)

    def prepare_deregister_memory(self, descriptor_list, *, backends):
        receipt = _FakeDeregistrationReceipt(self, descriptor_list, backends)
        self.deregister_receipts.append(receipt)
        return receipt

    def execute_deregister_memory(self, receipt):
        return receipt.execute(self)

    def get_agent_metadata(self):
        self.full_metadata_calls += 1
        return b"nixl-local-metadata"

    def get_reg_descs(self, regions, *, mem_type):
        descriptor_list = _FakeRegDlist(regions, mem_type)
        self.get_reg_descs_calls.append(descriptor_list)
        return descriptor_list

    def get_partial_agent_metadata(self, descriptor_list, *, inc_conn_info, backends):
        self.partial_metadata_calls.append(
            (
                tuple(descriptor_list.regions),
                descriptor_list.memory_type,
                inc_conn_info,
                tuple(backends),
            )
        )
        return b"nixl-partial-metadata"

    def add_remote_agent(self, metadata):
        self.imported_metadata.append(metadata)
        if self.import_error is not None:
            raise self.import_error
        return self.remote_names_by_metadata.get(metadata, self.remote_name)

    def inspect_remote_agent(self, metadata):
        return self.remote_names_by_metadata.get(metadata, self.remote_name)

    def remove_remote_agent(self, name):
        if self.remove_error is not None:
            raise self.remove_error
        self.removed_peers.append(name)

    def prep_xfer_dlist(self, name, descriptors, *, mem_type, backends):
        handle = _FakeDlist()
        self.prep_calls.append((name, descriptors, mem_type, backends, handle))
        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        *,
        notif_msg,
        backends,
    ):
        self.actions.append("make")
        handle = _FakeXfer(self.poll_statuses.copy(), cancellable=self.cancellable)
        self.make_calls.append(
            (
                operation,
                local_handle,
                local_indices,
                remote_handle,
                remote_indices,
                notif_msg,
                backends,
                handle,
            )
        )
        return handle

    def transfer(self, handle, notif_msg=b""):
        self.actions.append("post")
        self.post_notifications.append(notif_msg)
        handle.post_calls += 1
        if self.transfer_error is not None:
            raise self.transfer_error
        return self.submit_status

    def check_xfer_state(self, handle):
        if self.check_error is not None:
            raise self.check_error
        if len(handle.statuses) > 1:
            return handle.statuses.pop(0)
        return handle.statuses[0]

    def send_notif(self, remote_name, payload):
        self.actions.append("send")
        if self.send_error is not None:
            raise self.send_error
        self.sent_notifications.append((remote_name, payload))

    def get_new_notifs(self, *, backends):
        notifications, self.notifications = self.notifications, {}
        if self.notification_error is not None:
            raise self.notification_error
        return notifications


def _adopt_handle(operation, release):
    adopted = []

    try:
        result = operation(adopted.append)
    except BaseException as error:
        if adopted:
            try:
                release(adopted[0])
            except BaseException as cleanup_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"test cleanup remained pending: {cleanup_error!r}")
        raise
    assert result is None
    assert len(adopted) == 1
    return adopted[0]


def _register(backend, memory):
    return _adopt_handle(
        lambda adopt: backend.register(memory, adopt_handle=adopt),
        backend.deregister,
    )


def _import_peer(backend, metadata):
    return _adopt_handle(
        lambda adopt: backend.import_peer(metadata, adopt_handle=adopt),
        backend.release_peer,
    )


def _prepare(backend, local, remote, *, indexed):
    return _adopt_handle(
        lambda adopt: backend.prepare(
            local,
            remote,
            indexed=indexed,
            adopt_handle=adopt,
        ),
        backend.release_plan,
    )


@dataclass(frozen=True)
class _Fixture:
    backend: NixlBackend
    agent: _FakeAgent
    registration: object
    peer: object
    imported: object
    metadata: object


@pytest.fixture
def provider(request):
    thread_mode = getattr(request, "param", transfer.ThreadMode.SERIALIZED)
    agent = _FakeAgent()
    backend = NixlBackend(
        "local-id",
        "local",
        "local-incarnation",
        transfer.ProgressMode.MANUAL,
        thread_mode,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    memory = transfer.RegisteredMemory(
        registration_id="local-registration",
        name="local-buffer",
        owner=bytearray(256),
        address=_LOCAL_ADDRESS,
        nbytes=256,
        device="cpu",
        memory_type="DRAM",
    )
    registration = _register(backend, memory)
    named_descriptor = backend.describe_registration(registration)
    raw_descriptor_value = json.loads(named_descriptor)
    raw_descriptor_value.update(address=_RAW_ADDRESS, nbytes=64)
    raw_descriptor = json.dumps(
        raw_descriptor_value, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="remote-id",
        name="remote",
        incarnation="remote-incarnation",
        backend_payload=b"nixl-remote-metadata",
        registrations=(
            transfer.RegistrationMetadata(
                registration_id="remote-registration",
                name="remote-buffer",
                nbytes=256,
                device="cpu",
                memory_type="DRAM",
                backend_descriptor=named_descriptor,
            ),
            transfer.RegistrationMetadata(
                registration_id="remote-raw-registration",
                name="remote-raw-buffer",
                nbytes=64,
                device="cpu",
                memory_type="DRAM",
                backend_descriptor=raw_descriptor,
            ),
        ),
    )
    peer = _import_peer(backend, metadata)
    imported = transfer.ImportedRegistration(
        peer_handle=peer,
        registration_id="remote-registration",
        name="remote-buffer",
        nbytes=256,
        device="cpu",
        memory_type="DRAM",
        descriptor=named_descriptor,
    )
    result = _Fixture(backend, agent, registration, peer, imported, metadata)
    yield result
    # Tests that deliberately leave active work own their cleanup assertions.
    try:
        backend.close()
    except transfer.BusyError:
        pass


def test_factory_v2_version_signal_and_no_fallback(monkeypatch):
    assert TORCH_TRANSFER_FACTORY_API_VERSION == 2
    monkeypatch.setattr(torch_transfer_module, "_REGISTERED", False)
    monkeypatch.setattr(transfer, "BACKEND_FACTORY_API_VERSION", 1)
    with pytest.raises(RuntimeError, match="factory API >=2"):
        register_torch_backend()


def test_factory_v2_registration_lost_return_retry_converges(monkeypatch):
    real_register = transfer.register_backend
    transfer.unregister_backend("nixl")
    monkeypatch.setattr(torch_transfer_module, "_REGISTERED", False)
    interrupted = False

    def register_then_interrupt(*args, **kwargs):
        nonlocal interrupted
        real_register(*args, **kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after Core registry commit")

    monkeypatch.setattr(transfer, "register_backend", register_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after Core registry commit"):
        register_torch_backend()
    assert not torch_transfer_module._REGISTERED

    # The exact retry is accepted atomically by Core; the provider can now
    # publish its local completion bit without replacing registry ownership.
    monkeypatch.setattr(transfer, "register_backend", real_register)
    register_torch_backend()
    assert torch_transfer_module._REGISTERED


def test_registration_repairs_external_core_unregister() -> None:
    from torch.distributed._transfer._backend import get_backend_factory

    assert torch_transfer_module._REGISTERED
    transfer.unregister_backend("nixl")

    # The module-local bit is only a diagnostic. Every call must re-enter Core's
    # exact idempotent registration transaction so an external unregister heals.
    register_torch_backend()

    assert get_backend_factory("nixl") is _backend_factory
    assert torch_transfer_module._REGISTERED


def test_factory_v2_adopts_pure_backend_before_activation():
    agent = _FakeAgent()
    adopted = []

    def adopt_backend(backend):
        assert backend._agent is None
        assert not backend._activation_started
        assert not backend._activated
        assert not backend._available
        adopted.append(backend)

    result = _backend_factory(
        "factory-v2-id",
        "factory-v2",
        "factory-v2-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {"_agent": agent, "transfer_backends": ["UCX"]},
        adopt_backend=adopt_backend,
    )
    assert result is None
    assert len(adopted) == 1
    backend = adopted[0]
    assert backend._agent is agent
    assert backend._activation_started
    assert backend._activated
    assert backend._available
    backend.close()


def test_factory_v2_activation_cut_leaves_adopted_backend_closeable(monkeypatch):
    agent = _FakeAgent()
    adopted = []

    def interrupt_activation(self):
        self._activation_started = True
        self._agent = agent
        raise KeyboardInterrupt("after Core adoption")

    with monkeypatch.context() as activation_patch:
        activation_patch.setattr(NixlBackend, "_activate", interrupt_activation)
        with pytest.raises(KeyboardInterrupt, match="after Core adoption"):
            _backend_factory(
                "factory-cut-id",
                "factory-cut",
                "factory-cut-incarnation",
                transfer.ProgressMode.MANUAL,
                transfer.ThreadMode.SERIALIZED,
                {"_agent": agent, "transfer_backends": ["UCX"]},
                adopt_backend=adopted.append,
            )
    assert len(adopted) == 1
    backend = adopted[0]
    assert not backend._available
    assert not backend._closed
    backend.close()
    assert backend._closed


def test_close_cut_between_availability_and_closing_is_retryable(monkeypatch):
    backend = NixlBackend(
        "close-cut-id",
        "close-cut",
        "close-cut-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {"_agent": _FakeAgent(), "transfer_backends": ["UCX"]},
    )
    original_commit = NixlBackend._commit_closing
    interrupt = True

    def commit_with_cut(owner):
        nonlocal interrupt
        if interrupt:
            interrupt = False
            raise KeyboardInterrupt("between availability and closing")
        original_commit(owner)

    with monkeypatch.context() as close_patch:
        close_patch.setattr(NixlBackend, "_commit_closing", commit_with_cut)
        with pytest.raises(KeyboardInterrupt, match="between availability and closing"):
            backend.close()
        assert not backend._available
        assert not backend._closing
        assert not backend._closed
        with pytest.raises(transfer.ClosedError, match="not active"):
            backend.endpoint_payload([])
        backend.close()
    assert backend._closing
    assert backend._closed


def test_provider_deregister_requires_durable_receipt_before_native_mutation(provider):
    prepare = provider.agent.prepare_deregister_memory
    execute = provider.agent.execute_deregister_memory
    provider.agent.prepare_deregister_memory = None
    provider.agent.execute_deregister_memory = None
    try:
        with pytest.raises(
            transfer.BackendFailureError, match="durable deregistration receipt"
        ):
            provider.backend.deregister(provider.registration)
        assert not provider.agent.deregister_calls
        assert provider.registration.deregistration_receipt is None
        assert not provider.registration.released
    finally:
        provider.agent.prepare_deregister_memory = prepare
        provider.agent.execute_deregister_memory = execute
    provider.backend.deregister(provider.registration)


def _region(
    owner,
    *,
    remote=False,
    offset=0,
    nbytes=16,
    stride=None,
    count=None,
    device=None,
    memory_type=None,
):
    device = device or (
        "cuda:0" if getattr(owner, "memory_type", "DRAM") == "VRAM" else "cpu"
    )
    memory_type = memory_type or getattr(owner, "memory_type", "DRAM")
    return transfer.BackendRegion(
        # Core owns this lifetime anchor; providers must consume only the
        # backend descriptor below.
        owner=object(),
        offset=offset,
        nbytes=nbytes,
        device=device,
        memory_type=memory_type,
        backend_descriptor=owner,
        remote=remote,
        stride=stride,
        count=count,
    )


def _submit(backend, operation, plan, **options):
    adopted = []

    def adopt_work(work):
        assert not adopted
        adopted.append(work)

    result = backend.submit(
        operation,
        plan,
        adopt_work=adopt_work,
        **options,
    )
    assert result is None
    assert len(adopted) == 1
    return adopted[0]


def _run_two_concurrent_calls(operation):
    results = [None, None]
    errors = []

    def run(index):
        try:
            results[index] = operation()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    return results


def _single_block_selection(op, plan):
    indices = np.asarray([0], dtype=np.int32)
    backend_module = sys.modules[transfer.BackendIndexSelection.__module__]
    return transfer.BackendIndexSelection(
        op,
        plan,
        memoryview(indices).toreadonly(),
        memoryview(indices).toreadonly(),
        selected_nbytes=16,
        uniform_pair_layout=True,
        destination_catalog_nonoverlapping=True,
        destination_indices_unique=True,
        _token=backend_module._INDEX_SELECTION_TOKEN,
    )


def _send_notification(backend, peer, payload):
    adopted = []
    result = backend.send_notification(
        peer,
        payload,
        adopt_work=adopted.append,
    )
    assert result is None
    assert len(adopted) == 1
    return adopted[0]


def _create_request_slot(backend, operation, plan, **options):
    adopted = []
    result = backend.create_request_slot(
        operation,
        plan,
        adopt_slot=adopted.append,
        **options,
    )
    assert result is None
    assert len(adopted) == 1
    return adopted[0]


def test_capabilities_are_truthful(provider):
    capabilities = provider.backend.capabilities
    assert capabilities.read
    assert capabilities.write
    assert capabilities.attached_notifications
    assert capabilities.standalone_notifications
    assert capabilities.synchronous_notification_send
    assert capabilities.notifications
    assert capabilities.grouped_notification_receive
    assert capabilities.raw_spans
    assert capabilities.strided_regions
    assert capabilities.manual_progress
    assert capabilities.background_progress
    assert capabilities.persistent_request_slots
    assert capabilities.fused_request_slot_start_poll
    assert capabilities.fused_request_slot_poll
    assert not capabilities.fused_prevalidated_submit_poll
    assert not capabilities.fused_work_poll
    assert capabilities.deferred_successful_request_slot_recycle
    assert not capabilities.request_slot_notification_overrides
    assert not capabilities.cuda_completion_event
    assert capabilities.cuda_ordering_modes == frozenset(
        {
            transfer.CudaOrderingMode.CALLER_READY,
            transfer.CudaOrderingMode.HOST_WAIT,
        }
    )
    assert (
        capabilities.standalone_notification_completion
        is transfer.NotificationCompletion.LOCAL_ACCEPTED
    )
    assert (
        provider.backend.endpoint_payload([provider.registration])
        == b"nixl-local-metadata"
    )


@pytest.mark.parametrize(
    "thread_mode",
    [
        transfer.ThreadMode.SINGLE,
        transfer.ThreadMode.CALLER_SERIALIZED,
        transfer.ThreadMode.SERIALIZED,
        transfer.ThreadMode.MULTIPLE,
    ],
)
def test_raw_owned_fresh_requests_bypass_high_level_without_weakening_ownership(
    monkeypatch, provider, thread_mode
):
    success = object()
    native_read = object()
    native_write = object()
    backend_handle = 0xBACC

    class OwnedRequest:
        def __init__(self, value):
            self.value = value
            self.released = False
            self.release_calls = 0

        def release(self):
            self.release_calls += 1
            self.released = True

    class RawAgent:
        def __init__(self):
            self.make_calls = []
            self.owners = []
            self.post_calls = []

        def makeXferReqOwned(self, *args):
            self.make_calls.append(args)
            owner = OwnedRequest(0xD000 + len(self.owners))
            self.owners.append(owner)
            return owner

        def postXferReq(self, handle):
            self.post_calls.append(handle)
            return success

        def getXferStatus(self, _handle):
            return success

    class RawOwnedAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = RawAgent()
            self.backends = {"UCX": backend_handle}
            self._leaked_xfer_handles = []

        def prep_xfer_dlist(self, *args, **kwargs):
            handle = super().prep_xfer_dlist(*args, **kwargs)
            handle._handle = 0x1000 + len(self.prep_calls)
            return handle

        def make_prepped_xfer(self, *_args, **_kwargs):
            raise AssertionError("fresh requests must bypass the high-level helper")

        def transfer(self, *_args, **_kwargs):
            raise AssertionError("raw post must bypass the high-level helper")

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = RawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = native_read
    fake_bindings.NIXL_WRITE = native_write
    package_name = NixlBackend.__module__.rpartition(".")[0]
    package = sys.modules[package_name]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = RawOwnedAgent()
    backend = NixlBackend(
        "raw-owned-id",
        "raw-owned",
        "raw-owned-incarnation",
        transfer.ProgressMode.MANUAL,
        thread_mode,
        {
            "_agent": agent,
            "transfer_backends": ["UCX"],
            "use_native_request_slot_execution": False,
        },
    )
    assert backend._raw_transfer_backend_handles == (backend_handle,)
    assert backend._raw_read_operation is native_read
    assert backend._raw_write_operation is native_write
    assert backend._raw_make_xfer_req_owned.__self__ is agent.agent

    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=True,
    )
    assert plan.local_handle._handle == 0x1001
    assert plan.remote_handle._handle == 0x1002

    # The ordinary compatibility route accepts Python integer sequences; the
    # prevalidated route below is the native-int32 zero-copy path.
    ordinary_indices = [0]
    ordinary = _submit(
        backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=ordinary_indices,
        remote_indices=ordinary_indices,
        notification=b"raw-ordinary",
    )
    assert agent._leaked_xfer_handles == []
    assert ordinary._request.handle is agent.agent.owners[-1]
    assert ordinary._request.native_handle == agent.agent.owners[-1].value

    selection = _single_block_selection(transfer.TransferOp.READ, plan)
    adopted = []
    result = backend.submit_prevalidated(
        transfer.TransferOp.READ,
        plan,
        selection=selection,
        notification=b"raw-prevalidated",
        reuse_token=None,
        adopt_work=adopted.append,
    )
    assert result is None
    assert len(adopted) == 1
    prevalidated = adopted[0]
    assert agent._leaked_xfer_handles == []
    assert prevalidated._request.handle is agent.agent.owners[-1]
    assert prevalidated._request.native_handle == agent.agent.owners[-1].value

    assert len(agent.agent.make_calls) == 2
    ordinary_call, prevalidated_call = agent.agent.make_calls
    for call, expected_op, expected_notification in (
        (ordinary_call, native_write, b"raw-ordinary"),
        (prevalidated_call, native_read, b"raw-prevalidated"),
    ):
        assert len(call) == 8
        assert call[0] is expected_op
        assert call[1] == plan.local_handle._handle
        assert call[3] == plan.remote_handle._handle
        assert list(call[2]) == [0]
        assert list(call[4]) == [0]
        assert call[5] == expected_notification
        assert call[6] is backend._raw_transfer_backend_handles
        assert call[6] == (backend_handle,)
        assert call[7] is False
    assert prevalidated_call[2] is selection.local_indices
    assert prevalidated_call[4] is selection.remote_indices
    assert agent.agent.post_calls == [owner.value for owner in agent.agent.owners]

    works = (ordinary, prevalidated)
    assert all(
        work._request.handle is owner for work, owner in zip(works, agent.agent.owners)
    )
    for work, owner in zip(works, agent.agent.owners):
        work.release()
        assert owner.released
        assert owner.release_calls == 1
    assert agent._leaked_xfer_handles == []
    backend.release_plan(plan)
    backend.close()


def test_native_request_slot_reuses_backend_handle_tuple_cached_at_init(
    monkeypatch, provider
):
    success = object()
    native_read = object()
    native_write = object()
    created = []

    class RawAgent:
        def postXferReq(self, _handle):
            return success

        def getXferStatus(self, _handle):
            return success

        def createXferRequestSlotExecution(self, *args):
            created.append(args)
            return _FakeExecutionDispatch()

    class SlotAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = RawAgent()
            self.backends = {"UCX": 0xCA11}

        def prep_xfer_dlist(self, *args, **kwargs):
            handle = super().prep_xfer_dlist(*args, **kwargs)
            handle._handle = 0x2000 + len(self.prep_calls)
            return handle

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = RawAgent
    fake_bindings.nixlRequestSlotExecution = _FakeExecutionDispatch
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = native_read
    fake_bindings.NIXL_WRITE = native_write
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = SlotAgent()
    backend = NixlBackend(
        "cached-slot-id",
        "cached-slot",
        "cached-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SINGLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    cached_handles = backend._raw_transfer_backend_handles
    assert cached_handles == (0xCA11,)
    agent.backends["UCX"] = 0xBAD
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    adopted = []

    def adopt(slot, *, execution_dispatch):
        adopted.append((slot, execution_dispatch))

    backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"cached-slot",
        adopt_slot=adopt,
    )
    assert len(created) == 1
    assert created[0][0] is native_write
    assert created[0][6] is cached_handles
    assert created[0][6] == (0xCA11,)
    slot, execution = adopted[0]
    assert slot._execution_dispatch is execution
    slot.close()
    backend.release_plan(plan)
    backend.close()


def test_raw_owned_creation_falls_back_for_older_binding(monkeypatch, provider):
    success = object()

    class OlderRawAgent:
        def postXferReq(self, _handle):
            return success

        def getXferStatus(self, _handle):
            return success

    class OlderHighLevelAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = OlderRawAgent()
            self.backends = {"UCX": 0x0D1D}

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = OlderRawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = object()
    fake_bindings.NIXL_WRITE = object()
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = OlderHighLevelAgent()
    backend = NixlBackend(
        "older-binding-id",
        "older-binding",
        "older-binding-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    assert backend._raw_make_xfer_req_owned is None
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    work = _submit(
        backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"fallback",
    )
    assert len(agent.make_calls) == 1
    assert agent.make_calls[0][5] == b"fallback"
    work.release()
    backend.release_plan(plan)
    backend.close()


def test_v0_rejects_unverified_native_backends():
    with pytest.raises(transfer.UnsupportedError, match="exactly backends"):
        NixlBackend(
            "local-id",
            "local",
            "incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.SERIALIZED,
            {"_agent": _FakeAgent(), "backends": ["MOONCAKE"]},
        )


def test_selective_metadata_does_not_leak_omitted_registrations(provider):
    second = _register(
        provider.backend,
        transfer.RegisteredMemory(
            registration_id="second-registration",
            name="second-buffer",
            owner=bytearray(64),
            address=0x2000,
            nbytes=64,
            device="cpu",
            memory_type="DRAM",
        ),
    )

    assert (
        provider.backend.endpoint_payload([second, provider.registration])
        == b"nixl-local-metadata"
    )
    assert provider.agent.full_metadata_calls == 1
    assert provider.agent.partial_metadata_calls == []

    assert provider.backend.endpoint_payload([second]) == b"nixl-partial-metadata"
    assert provider.agent.partial_metadata_calls[-1] == (
        ((0x2000, 64, 0, ""),),
        "DRAM",
        True,
        ("UCX",),
    )
    assert all(
        region[0] != _LOCAL_ADDRESS
        for region in provider.agent.partial_metadata_calls[-1][0]
    )

    assert provider.backend.endpoint_payload([]) == b"nixl-partial-metadata"
    assert provider.agent.partial_metadata_calls[-1] == (
        (),
        "DRAM",
        True,
        ("UCX",),
    )


def test_mixed_selective_metadata_is_explicitly_unsupported(provider):
    cuda_registration = _register(
        provider.backend,
        transfer.RegisteredMemory(
            registration_id="cuda-registration",
            name="cuda-buffer",
            owner=object(),
            address=0x2000,
            nbytes=64,
            device="cuda:0",
            memory_type="VRAM",
        ),
    )
    omitted = _register(
        provider.backend,
        transfer.RegisteredMemory(
            registration_id="omitted-registration",
            name="omitted-buffer",
            owner=bytearray(64),
            address=0x3000,
            nbytes=64,
            device="cpu",
            memory_type="DRAM",
        ),
    )

    assert (
        provider.backend.endpoint_payload(
            [provider.registration, cuda_registration, omitted]
        )
        == b"nixl-local-metadata"
    )
    with pytest.raises(transfer.UnsupportedError, match="mixed-memory"):
        provider.backend.endpoint_payload([provider.registration, cuda_registration])


def test_background_mode_enables_nixl_native_progress_thread(monkeypatch):
    class FakeConfig:
        def __init__(self, **options):
            self.options = options

    class FakeSyncMode:
        NIXL_THREAD_SYNC_NONE = object()
        NIXL_THREAD_SYNC_STRICT = object()
        NIXL_THREAD_SYNC_RW = object()

    fake_api = types.ModuleType("nixl._api")
    fake_api.DEFAULT_COMM_PORT = 42
    fake_api.nixl_agent = object()
    fake_api.nixl_agent_config = FakeConfig
    fake_api.nixl_thread_sync_t = FakeSyncMode
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    created = []

    def factory(name, config):
        created.append((name, config))
        return _FakeAgent()

    backend = NixlBackend(
        "endpoint-id",
        "background-agent",
        "incarnation",
        transfer.ProgressMode.BACKGROUND,
        transfer.ThreadMode.SERIALIZED,
        {
            "_agent_factory": factory,
            "backends": ["UCX"],
            "backend_init_params": {"UCX": {"thread_count": "8"}},
        },
    )
    assert created[0][0] == _native_name("endpoint-id", "incarnation")
    assert created[0][0] != _native_name("endpoint-id", "replacement-incarnation")
    assert _native_name("a\0b", "c") != _native_name("a", "b\0c")
    assert created[0][0].startswith("torch-transfer-")
    assert created[0][0] != "background-agent"
    assert created[0][1].options["enable_prog_thread"] is True
    assert created[0][1].options["backends"] == []
    assert created[0][1].options["num_threads"] == 0
    assert created[0][1].options["sync_mode"] is FakeSyncMode.NIXL_THREAD_SYNC_NONE
    assert backend._require_agent().created_backends == [("UCX", {"thread_count": "8"})]
    assert backend.supported_thread_modes == frozenset(
        {
            transfer.ThreadMode.SINGLE,
            transfer.ThreadMode.CALLER_SERIALIZED,
            transfer.ThreadMode.SERIALIZED,
            transfer.ThreadMode.MULTIPLE,
        }
    )
    backend.close()


@pytest.mark.parametrize(
    ("listen_port", "num_threads"), ((0, 0), ((1 << 16) - 1, (1 << 32) - 1))
)
def test_agent_creation_option_native_boundaries_are_forwarded(
    monkeypatch, listen_port, num_threads
):
    class FakeConfig:
        def __init__(self, **options):
            self.options = options

    class FakeSyncMode:
        NIXL_THREAD_SYNC_NONE = object()
        NIXL_THREAD_SYNC_STRICT = object()
        NIXL_THREAD_SYNC_RW = object()

    fake_api = types.ModuleType("nixl._api")
    fake_api.DEFAULT_COMM_PORT = 42
    fake_api.nixl_agent = object()
    fake_api.nixl_agent_config = FakeConfig
    fake_api.nixl_thread_sync_t = FakeSyncMode
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    created = []

    def factory(name, config):
        created.append((name, config))
        return _FakeAgent()

    backend = NixlBackend(
        f"option-boundary-{listen_port}-{num_threads}",
        "option-boundary-agent",
        "option-boundary-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {
            "_agent_factory": factory,
            "backends": ["UCX"],
            "enable_nixl_progress_thread": False,
            "enable_listen_thread": True,
            "listen_port": listen_port,
            "capture_telemetry": True,
            "num_threads": num_threads,
        },
    )

    config = created[0][1]
    assert config.options["enable_prog_thread"] is False
    assert config.options["enable_listen_thread"] is True
    assert config.options["listen_port"] == listen_port
    assert config.options["capture_telemetry"] is True
    assert config.options["num_threads"] == num_threads
    backend.close()


@pytest.mark.parametrize(
    ("option", "value", "error_type"),
    (
        ("enable_nixl_progress_thread", "false", TypeError),
        ("enable_nixl_progress_thread", 0, TypeError),
        ("enable_listen_thread", "false", TypeError),
        ("enable_listen_thread", 0, TypeError),
        ("capture_telemetry", "false", TypeError),
        ("capture_telemetry", 1, TypeError),
        ("listen_port", True, TypeError),
        ("listen_port", 1.5, TypeError),
        ("listen_port", -1, ValueError),
        ("listen_port", 1 << 16, ValueError),
        ("num_threads", True, TypeError),
        ("num_threads", "4", TypeError),
        ("num_threads", -1, ValueError),
        ("num_threads", 1 << 32, ValueError),
    ),
)
def test_agent_creation_options_reject_coercion_and_native_overflow(
    monkeypatch, option, value, error_type
):
    class FakeConfig:
        def __init__(self, **options):
            self.options = options

    class FakeSyncMode:
        NIXL_THREAD_SYNC_NONE = object()
        NIXL_THREAD_SYNC_STRICT = object()
        NIXL_THREAD_SYNC_RW = object()

    fake_api = types.ModuleType("nixl._api")
    fake_api.DEFAULT_COMM_PORT = 42
    fake_api.nixl_agent = object()
    fake_api.nixl_agent_config = FakeConfig
    fake_api.nixl_thread_sync_t = FakeSyncMode
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    created = []

    def factory(name, config):
        created.append((name, config))
        return _FakeAgent()

    with pytest.raises(error_type, match=option):
        NixlBackend(
            f"invalid-option-{option}",
            "invalid-option-agent",
            "invalid-option-incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.SERIALIZED,
            {"_agent_factory": factory, "backends": ["UCX"], option: value},
        )
    assert created == []


@pytest.mark.parametrize(
    ("thread_mode", "requested", "expected"),
    (
        (transfer.ThreadMode.SINGLE, None, "none"),
        (transfer.ThreadMode.CALLER_SERIALIZED, None, "none"),
        (transfer.ThreadMode.SERIALIZED, None, "none"),
        (transfer.ThreadMode.MULTIPLE, None, "rw"),
        (transfer.ThreadMode.SERIALIZED, "none", "none"),
        (transfer.ThreadMode.SERIALIZED, "strict", "strict"),
        (transfer.ThreadMode.SERIALIZED, "rw", "rw"),
        (transfer.ThreadMode.CALLER_SERIALIZED, "strict", "strict"),
        (transfer.ThreadMode.MULTIPLE, "strict", "strict"),
        (transfer.ThreadMode.MULTIPLE, "rw", "rw"),
    ),
)
def test_agent_thread_sync_mode_selection(
    monkeypatch, thread_mode, requested, expected
):
    class FakeConfig:
        def __init__(self, **options):
            self.options = options

    class FakeSyncMode:
        NIXL_THREAD_SYNC_NONE = object()
        NIXL_THREAD_SYNC_STRICT = object()
        NIXL_THREAD_SYNC_RW = object()

    fake_api = types.ModuleType("nixl._api")
    fake_api.DEFAULT_COMM_PORT = 42
    fake_api.nixl_agent = object()
    fake_api.nixl_agent_config = FakeConfig
    fake_api.nixl_thread_sync_t = FakeSyncMode
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    created = []

    def factory(name, config):
        created.append((name, config))
        return _FakeAgent()

    options = {"_agent_factory": factory, "backends": ["UCX"]}
    if requested is not None:
        options["nixl_thread_sync_mode"] = requested
    backend = NixlBackend(
        "sync-mode-id",
        "sync-mode-agent",
        "sync-mode-incarnation",
        transfer.ProgressMode.MANUAL,
        thread_mode,
        options,
    )

    expected_value = {
        "none": FakeSyncMode.NIXL_THREAD_SYNC_NONE,
        "strict": FakeSyncMode.NIXL_THREAD_SYNC_STRICT,
        "rw": FakeSyncMode.NIXL_THREAD_SYNC_RW,
    }[expected]
    assert created[0][1].options["sync_mode"] is expected_value
    backend.close()


@pytest.mark.parametrize("requested", (1, "NONE", "invalid"))
def test_invalid_agent_thread_sync_modes_are_rejected(requested):
    expected_error = TypeError if isinstance(requested, int) else ValueError
    with pytest.raises(expected_error, match="nixl_thread_sync_mode"):
        NixlBackend(
            "invalid-sync-id",
            "invalid-sync-agent",
            "invalid-sync-incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.SERIALIZED,
            {"_agent": _FakeAgent(), "nixl_thread_sync_mode": requested},
        )


def test_none_agent_sync_is_rejected_for_multiple_thread_mode():
    with pytest.raises(ValueError, match="unsafe with ThreadMode.MULTIPLE"):
        NixlBackend(
            "unsafe-sync-id",
            "unsafe-sync-agent",
            "unsafe-sync-incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.MULTIPLE,
            {"_agent": _FakeAgent(), "nixl_thread_sync_mode": "none"},
        )


def test_request_pool_limit_accepts_zero_and_rejects_invalid_values():
    backend = NixlBackend(
        "zero-cache-id",
        "zero-cache-agent",
        "zero-cache-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {"_agent": _FakeAgent(), "max_cached_requests_per_plan": 0},
    )
    assert backend._max_cached_requests_per_plan == 0
    backend.close()

    for value, error_type in ((-1, ValueError), (True, TypeError), (1.5, TypeError)):
        with pytest.raises(error_type, match="max_cached_requests_per_plan"):
            NixlBackend(
                "invalid-cache-id",
                "invalid-cache-agent",
                "invalid-cache-incarnation",
                transfer.ProgressMode.MANUAL,
                transfer.ThreadMode.SERIALIZED,
                {"_agent": _FakeAgent(), "max_cached_requests_per_plan": value},
            )


def test_agent_factory_rolls_back_partially_published_agent_on_signal(monkeypatch):
    class FakeConfig:
        def __init__(self, **options):
            self.options = options

    class FakeSyncMode:
        NIXL_THREAD_SYNC_NONE = object()
        NIXL_THREAD_SYNC_STRICT = object()
        NIXL_THREAD_SYNC_RW = object()

    fake_api = types.ModuleType("nixl._api")
    fake_api.DEFAULT_COMM_PORT = 42
    fake_api.nixl_agent = object()
    fake_api.nixl_agent_config = FakeConfig
    fake_api.nixl_thread_sync_t = FakeSyncMode
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    created = []

    class PartiallyPublishedAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.closed = False

        def create_backend(self, backend, init_params):
            super().create_backend(backend, init_params)
            raise KeyboardInterrupt()

        def close(self):
            self.closed = True

    def factory(name, config):
        del name, config
        agent = PartiallyPublishedAgent()
        created.append(agent)
        return agent

    with pytest.raises(KeyboardInterrupt):
        NixlBackend(
            "factory-interrupt-id",
            "factory-interrupt",
            "factory-interrupt-incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.SERIALIZED,
            {
                "_agent_factory": factory,
                "backends": ["UCX"],
                "backend_init_params": {"UCX": {}},
            },
        )

    assert len(created) == 1
    assert created[0].created_backends == [("UCX", {})]
    assert created[0].closed


def test_registration_uses_address_without_staging(provider):
    assert provider.agent.register_calls == [
        ([(_LOCAL_ADDRESS, 256, 0, "")], "DRAM", ["UCX"])
    ]
    descriptor = provider.backend.describe_registration(provider.registration)
    assert f'"address":{_LOCAL_ADDRESS}'.encode() in descriptor
    provider.backend.deregister(provider.registration)
    assert len(provider.agent.deregister_calls) == 1


def test_high_level_deregister_selects_ucx_receipt_or_legacy_route():
    class Receipt:
        completed = False

    class RawAgent:
        def __init__(self):
            self.prepare_calls = []
            self.execute_calls = []
            self.legacy_calls = []

        def prepareDeregisterMem(self, descriptors, handles):
            receipt = Receipt()
            self.prepare_calls.append((descriptors, tuple(handles), receipt))
            return receipt

        def executeDeregisterMem(self, receipt):
            self.execute_calls.append(receipt)
            receipt.completed = True
            return "SUCCESS"

        def deregisterMem(self, descriptors, handles):
            self.legacy_calls.append((descriptors, tuple(handles)))
            return "SUCCESS"

    def high_level(backends):
        result = object.__new__(_HighLevelNixlAgent)
        result.backends = dict(backends)
        result.agent = RawAgent()
        return result

    descriptors = object()
    sole_ucx = high_level((("UCX", 0x11),))
    assert sole_ucx.deregister_memory(descriptors) is None
    assert sole_ucx.agent.prepare_calls[0][1] == (0x11,)
    assert len(sole_ucx.agent.execute_calls) == 1
    assert not sole_ucx.agent.legacy_calls

    mixed = high_level((("UCX", 0x22), ("POSIX", 0x33)))
    assert mixed.deregister_memory(descriptors) is None
    assert mixed.agent.legacy_calls == [(descriptors, ())]
    assert not mixed.agent.prepare_calls

    assert mixed.deregister_memory(descriptors, backends=["POSIX"]) is None
    assert mixed.agent.legacy_calls[-1] == (descriptors, (0x33,))
    with pytest.raises(ValueError, match="exactly one configured or explicit UCX"):
        mixed.prepare_deregister_memory(descriptors, backends=["POSIX"])

    receipt = mixed.prepare_deregister_memory(descriptors, backends=["UCX"])
    assert mixed.agent.prepare_calls[-1][1] == (0x22,)
    assert not receipt.completed


def test_provider_deregister_enforces_singleton_ucx_invariant(provider):
    provider.backend._transfer_backends = ["UCX", "POSIX"]
    with pytest.raises(transfer.BackendFailureError, match="exactly one UCX backend"):
        provider.backend.deregister(provider.registration)
    assert not provider.agent.deregister_receipts
    assert not provider.registration.released

    provider.backend._transfer_backends = ["UCX"]
    provider.backend.deregister(provider.registration)


def test_deregistration_precommit_signal_retries_same_native_receipt(provider):
    provider.agent.deregister_error = KeyboardInterrupt("before native commit")

    with pytest.raises(KeyboardInterrupt, match="before native commit"):
        provider.backend.deregister(provider.registration)

    receipt = provider.registration.deregistration_receipt
    assert receipt is provider.agent.deregister_receipts[-1]
    assert not receipt.completed
    assert not provider.registration.released
    assert provider.registration in provider.backend._registrations
    provider.agent.deregister_error = None
    provider.backend.deregister(provider.registration)
    assert receipt.completed
    assert receipt.native_attempts == 2
    assert provider.registration.released


def test_deregistration_postcommit_signal_reconciles_before_reraise(provider):
    provider.agent.deregister_commit_before_error = True
    provider.agent.deregister_error = KeyboardInterrupt("after native commit")

    with pytest.raises(KeyboardInterrupt, match="after native commit"):
        provider.backend.deregister(provider.registration)

    receipt = provider.registration.deregistration_receipt
    assert receipt.completed
    assert receipt.native_attempts == 1
    assert provider.registration.released
    assert provider.registration not in provider.backend._registrations
    provider.agent.deregister_error = None
    provider.backend.deregister(provider.registration)
    assert receipt.native_attempts == 1


def test_deregistration_ordinary_failure_remains_retryable(provider):
    provider.agent.deregister_error = RuntimeError("ordinary native failure")

    with pytest.raises(transfer.BackendFailureError, match="failed to deregister"):
        provider.backend.deregister(provider.registration)

    receipt = provider.registration.deregistration_receipt
    assert not receipt.completed
    assert not provider.registration.released
    provider.agent.deregister_error = None
    provider.backend.deregister(provider.registration)
    assert receipt.completed
    assert receipt.native_attempts == 2


def test_deregistration_not_found_receipt_is_terminal_and_idempotent(provider):
    class nixlNotFoundError(RuntimeError):
        pass

    provider.agent.deregister_commit_before_error = True
    provider.agent.deregister_error = nixlNotFoundError("NIXL_ERR_NOT_FOUND")

    provider.backend.deregister(provider.registration)

    receipt = provider.registration.deregistration_receipt
    assert receipt.completed
    assert receipt.native_attempts == 1
    assert provider.registration.released
    provider.agent.deregister_error = None
    provider.backend.deregister(provider.registration)
    assert receipt.native_attempts == 1


def test_registration_adoption_precedes_native_mutation(provider):
    memory = transfer.RegisteredMemory(
        registration_id="rejected-registration",
        name="rejected-buffer",
        owner=bytearray(16),
        address=0x18_0000_1800,
        nbytes=16,
        device="cpu",
        memory_type="DRAM",
    )
    registrations_before = provider.backend._registrations.copy()
    register_calls_before = len(provider.agent.register_calls)

    def reject(handle):
        assert handle in provider.backend._registrations
        assert not handle.registration_started
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        provider.backend.register(memory, adopt_handle=reject)

    assert provider.backend._registrations == registrations_before
    assert len(provider.agent.register_calls) == register_calls_before


def test_registration_keyboard_interrupt_rolls_back_and_drops_anchor(provider):
    memory = transfer.RegisteredMemory(
        registration_id="interrupted-registration",
        name="interrupted-buffer",
        owner=bytearray(16),
        address=0x20_0000_2000,
        nbytes=16,
        device="cpu",
        memory_type="DRAM",
    )
    provider.agent.register_error = KeyboardInterrupt()
    registrations_before = provider.backend._registrations.copy()

    with pytest.raises(KeyboardInterrupt):
        _register(provider.backend, memory)

    assert provider.backend._registrations == registrations_before
    assert provider.agent.deregister_calls[-1][0].regions == [
        (memory.address, memory.nbytes, 0, "")
    ]
    provider.agent.register_error = None


def test_registration_rollback_failure_retains_allocation_until_retry(provider):
    memory = transfer.RegisteredMemory(
        registration_id="retained-registration",
        name="retained-buffer",
        owner=bytearray(16),
        address=0x30_0000_3000,
        nbytes=16,
        device="cpu",
        memory_type="DRAM",
    )
    provider.agent.register_error = KeyboardInterrupt()
    provider.agent.deregister_error = RuntimeError("rollback failed")

    with pytest.raises(KeyboardInterrupt):
        _register(provider.backend, memory)

    retained = provider.backend._registrations[-1]
    assert retained.registration is memory
    assert not retained.released
    provider.agent.register_error = None
    provider.agent.deregister_error = None
    provider.backend.deregister(retained)


def test_registration_rejects_uintptr_end_overflow_before_native_call(provider):
    maximum = (1 << (np.dtype(np.uintp).itemsize * 8)) - 1
    memory = transfer.RegisteredMemory(
        registration_id="overflow-registration",
        name="overflow-buffer",
        owner=bytearray(1),
        address=maximum,
        nbytes=1,
        device="cpu",
        memory_type="DRAM",
    )
    calls_before = len(provider.agent.register_calls)

    with pytest.raises(transfer.InvalidRegionError, match="uintptr_t"):
        _register(provider.backend, memory)

    assert len(provider.agent.register_calls) == calls_before
    provider.backend.deregister(provider.registration)
    assert len(provider.agent.deregister_calls) == 1


def test_prepared_indexed_strided_write_and_notification(provider):
    local = [
        _region(provider.registration, offset=0, nbytes=8, stride=32, count=4),
        _region(provider.registration, offset=128, nbytes=16),
    ]
    remote = [
        _region(provider.imported, remote=True, offset=64, nbytes=16),
        _region(
            provider.imported,
            remote=True,
            offset=0,
            nbytes=8,
            stride=32,
            count=4,
        ),
    ]
    plan = _prepare(provider.backend, local, remote, indexed=True)
    assert len(provider.agent.prep_calls) == 2
    assert all(
        isinstance(call[1], np.ndarray) and call[1].shape == (2, 5)
        for call in provider.agent.prep_calls
    )

    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC", "DONE"]
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=[0, 1, 2, 3],
        remote_indices=[1, 2, 3, 4],
        notification=b"after-copy",
    )
    assert work.state is transfer.WorkState.RUNNING
    assert provider.agent.make_calls[0][0] == "WRITE"
    assert provider.agent.make_calls[0][2].tolist() == [0, 1, 2, 3]
    assert provider.agent.make_calls[0][4].tolist() == [1, 2, 3, 4]
    assert provider.agent.make_calls[0][5] == b"after-copy"
    provider.backend.progress()
    assert work.state is transfer.WorkState.COMPLETED
    assert work.result == 32
    work.close()
    provider.backend.release_plan(plan)
    assert provider.agent.prep_calls[0][4].released
    assert provider.agent.prep_calls[1][4].released


def test_prepare_does_not_materialize_logical_block_indices(provider, monkeypatch):
    def reject_arange(*args, **kwargs):
        raise AssertionError("prepare expanded the logical catalog")

    monkeypatch.setattr(np, "arange", reject_arange)
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, nbytes=8, stride=16, count=8)],
        [
            _region(
                provider.imported,
                remote=True,
                nbytes=8,
                stride=16,
                count=8,
            )
        ],
        indexed=True,
    )

    assert plan.local_all_indices is None
    assert plan.remote_all_indices is None
    assert plan.local_total_nbytes == 64
    assert plan.full_layout_matches
    provider.backend.release_plan(plan)


def test_full_selection_materialization_is_lazy_cached_and_bounded(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, nbytes=8, stride=16, count=3)],
        [
            _region(
                provider.imported,
                remote=True,
                nbytes=8,
                stride=16,
                count=3,
            )
        ],
        indexed=True,
    )
    provider.backend._max_materialized_indices = 2

    with pytest.raises(
        transfer.UnsupportedError, match="explicit bounded index chunks"
    ):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )

    assert plan.local_all_indices is None
    assert plan.remote_all_indices is None
    assert not provider.agent.make_calls
    provider.backend._max_materialized_indices = 3
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert plan.local_all_indices.tolist() == [0, 1, 2]
    assert plan.remote_all_indices.tolist() == [0, 1, 2]
    work.close()
    provider.backend.release_plan(plan)


def test_prepare_keyboard_interrupt_rolls_back_pretracked_plan(provider):
    original = provider.agent.prep_xfer_dlist
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    provider.agent.prep_xfer_dlist = interrupt_second
    plans_before = provider.backend._plans.copy()

    with pytest.raises(KeyboardInterrupt):
        _prepare(
            provider.backend,
            [_region(provider.registration)],
            [_region(provider.imported, remote=True)],
            indexed=False,
        )

    assert provider.backend._plans == plans_before
    assert provider.agent.prep_calls[-1][4].released
    provider.agent.prep_xfer_dlist = original


def test_prepare_adoption_precedes_native_handle_creation(provider):
    plans_before = provider.backend._plans.copy()
    prep_calls_before = len(provider.agent.prep_calls)

    def reject(plan):
        assert plan in provider.backend._plans
        assert plan.local_handle is None
        assert plan.remote_handle is None
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        provider.backend.prepare(
            [_region(provider.registration)],
            [_region(provider.imported, remote=True)],
            indexed=False,
            adopt_handle=reject,
        )

    assert provider.backend._plans == plans_before
    assert len(provider.agent.prep_calls) == prep_calls_before


def test_indexed_selection_addresses_individual_strided_blocks(provider):
    local = [_region(provider.registration, nbytes=8, stride=32, count=4)]
    remote = [
        _region(
            provider.imported,
            remote=True,
            nbytes=8,
            stride=32,
            count=4,
        )
    ]
    plan = _prepare(provider.backend, local, remote, indexed=True)

    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=[2],
        remote_indices=[1],
        notification=None,
    )
    assert provider.agent.make_calls[-1][2].tolist() == [2]
    assert provider.agent.make_calls[-1][4].tolist() == [1]
    assert work.result == 8
    work.close()

    with pytest.raises(transfer.InvalidRegionError, match="out of range"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=[4],
            remote_indices=[0],
            notification=None,
        )
    with pytest.raises(transfer.InvalidRegionError, match="overlapping destination"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=[0, 1],
            remote_indices=[2, 2],
            notification=None,
        )
    provider.backend.release_plan(plan)


def test_plan_release_retries_only_the_failed_native_side(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    local_handle = provider.agent.prep_calls[-2][4]
    remote_handle = provider.agent.prep_calls[-1][4]
    local_handle.release_errors.append(RuntimeError("transient release failure"))

    with pytest.raises(transfer.BackendFailureError, match="release prepared"):
        provider.backend.release_plan(plan)
    assert remote_handle.release_calls == 1
    assert local_handle.release_calls == 1
    assert not plan.released

    make_call_count = len(provider.agent.make_calls)
    with pytest.raises(transfer.ClosedError, match="closing or closed"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
    assert len(provider.agent.make_calls) == make_call_count

    provider.backend.release_plan(plan)
    assert remote_handle.release_calls == 1
    assert local_handle.release_calls == 2
    assert plan.released


def test_none_indices_select_full_mixed_flattened_catalog(provider):
    local = [
        _region(provider.registration, offset=0, nbytes=8, stride=32, count=3),
        _region(provider.registration, offset=128, nbytes=16),
    ]
    remote = [
        _region(
            provider.imported,
            remote=True,
            offset=0,
            nbytes=8,
            stride=32,
            count=3,
        ),
        _region(provider.imported, remote=True, offset=128, nbytes=16),
    ]
    plan = _prepare(provider.backend, local, remote, indexed=True)

    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert provider.agent.make_calls[-1][2].tolist() == [0, 1, 2, 3]
    assert provider.agent.make_calls[-1][4].tolist() == [0, 1, 2, 3]
    assert work.result == 40
    work.close()
    provider.backend.release_plan(plan)


def test_read_and_write_are_first_class_operations(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    for operation in (transfer.TransferOp.READ, transfer.TransferOp.WRITE):
        work = _submit(
            provider.backend,
            operation,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
        assert work.state is transfer.WorkState.COMPLETED
        work.close()
    assert [call[0] for call in provider.agent.make_calls] == ["READ", "WRITE"]


def test_completed_native_request_is_reposted_for_same_token(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()

    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    handle = provider.agent.make_calls[-1][-1]
    first.close()
    assert plan.cached_request_count == 1
    assert not handle.released

    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert len(provider.agent.make_calls) == 1
    assert second._handle is handle
    assert handle.post_calls == 2
    second.close()

    provider.backend.release_plan(plan)
    assert handle.released
    assert handle.release_calls == 1


def test_persistent_request_slot_reuses_work_request_lock_and_handle(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"slot-complete",
    )

    identities = []
    for _ in range(3):
        slot.start()
        assert slot.state is transfer.WorkState.COMPLETED
        work = slot._work
        identities.append(
            (id(work), id(work._lock), id(work._request), id(work._handle))
        )
        assert plan.cached_request_count == 0
        slot.recycle()

    assert len(set(identities)) == 1
    handle = slot._work._handle
    assert len(provider.agent.make_calls) == 1
    assert provider.agent.make_calls[0][5] == b"slot-complete"
    assert handle.post_calls == 3
    assert not handle.released
    slot.close()
    slot.close()
    assert handle.released
    assert handle.release_calls == 1
    provider.backend.release_plan(plan)


def test_native_execution_dispatch_is_adopted_and_wrapper_owns_close(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    plan.local_handle._handle = 0x1000
    plan.remote_handle._handle = 0x2000
    provider.agent.backends = {"UCX": 0x3000}
    created = []

    def factory(*args):
        created.append(args)
        return _FakeExecutionDispatch()

    provider.backend._raw_request_slot_factory = factory
    provider.backend._raw_write_operation = "NATIVE_WRITE"
    adopted = []

    def adopt(slot, *, execution_dispatch=None):
        adopted.append((slot, execution_dispatch))

    provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"dispatch",
        adopt_slot=adopt,
    )
    assert len(created) == 1
    slot, execution = adopted[0]
    assert execution is slot._execution_dispatch
    assert created[0][0] == "NATIVE_WRITE"
    assert created[0][1] == 0x1000
    assert created[0][3] == 0x2000
    assert created[0][5] == b"dispatch"
    assert created[0][6] == [0x3000]
    assert created[0][10] is plan.local_handle
    assert created[0][11] is plan.remote_handle

    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None)
        is transfer.WorkState.COMPLETED
    )
    slot.recycle()
    assert execution.recycle_calls == 1
    assert execution.native_recycle_calls == 0
    execution.fail_next = True
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None) is transfer.WorkState.FAILED
    )
    error = slot.error
    assert isinstance(error, transfer.BackendFailureError)
    assert "injected native execution failure" in str(error)
    assert slot.error is error
    slot.recycle()
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None)
        is transfer.WorkState.COMPLETED
    )
    assert execution.post_calls == 3
    # Model Core's deferred-success close: the Python wrapper remains the close
    # owner even when Core cached all execution methods from the native object.
    slot.close()
    slot.close()
    assert execution.close_calls == 1
    assert id(slot) not in provider.backend._slots
    assert id(slot) not in plan.request_slots
    provider.backend.release_plan(plan)


def test_native_execution_dispatch_reposts_request_local_notifications(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    plan.local_handle._handle = 0x1000
    plan.remote_handle._handle = 0x2000
    provider.agent.backends = {"UCX": 0x3000}
    execution = _FakeExecutionDispatch()
    provider.backend._raw_request_slot_factory = lambda *args: execution
    provider.backend._raw_write_operation = "NATIVE_WRITE"
    adopted = []

    provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=lambda slot, *, execution_dispatch=None: adopted.append(slot),
    )
    slot = adopted[0]

    for notification in (b"A", b"B", b"A", None, b"\x00\xff"):
        assert (
            slot.start_and_poll_with_notification(
                notification,
                max_polls=64,
                timeout_ns=None,
            )
            is transfer.WorkState.COMPLETED
        )
        slot.recycle()
    assert execution.notification_posts == [b"A", b"B", b"A", None, b"\x00\xff"]
    assert execution.post_calls == 5
    slot.close()
    provider.backend.release_plan(plan)


def test_request_local_slot_capability_tracks_compiled_execution_type(monkeypatch):
    class RawAgent:
        def postXferReq(self, *args):
            return 0

        def getXferStatus(self, *args):
            return 0

        def createXferRequestSlotExecution(self, *args):
            return None

    class CurrentExecution:
        def start_with_notification(self, notification):
            del notification

        def start_and_poll_with_notification(
            self, notification, *, max_polls, timeout_ns
        ):
            del notification, max_polls, timeout_ns

    class OldExecution:
        pass

    package = sys.modules["nixl"]

    for execution_type, expected in (
        (CurrentExecution, True),
        (OldExecution, False),
    ):
        fake_bindings = types.ModuleType("nixl._bindings")
        fake_bindings.nixlAgent = RawAgent
        fake_bindings.nixlRequestSlotExecution = execution_type
        fake_bindings.NIXL_SUCCESS = 0
        fake_bindings.NIXL_IN_PROG = 1
        fake_bindings.NIXL_READ = 2
        fake_bindings.NIXL_WRITE = 3
        monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
        monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

        agent = _FakeAgent()
        agent.agent = RawAgent()
        backend = NixlBackend(
            "local-id",
            "local",
            "local-incarnation",
            transfer.ProgressMode.MANUAL,
            transfer.ThreadMode.SERIALIZED,
            {"_agent": agent, "transfer_backends": ["UCX"]},
        )
        assert backend.capabilities.request_slot_notification_overrides is expected
        backend.close()


def test_native_execution_dispatch_cold_close_does_not_create_request(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    plan.local_handle._handle = 0x1000
    plan.remote_handle._handle = 0x2000
    provider.agent.backends = {"UCX": 0x3000}
    execution = _FakeExecutionDispatch()
    provider.backend._raw_request_slot_factory = lambda *args: execution
    provider.backend._raw_write_operation = "NATIVE_WRITE"
    adopted = []

    def adopt(slot, *, execution_dispatch=None):
        adopted.append(slot)
        assert execution_dispatch is execution

    provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=adopt,
    )
    adopted[0].close()
    assert execution.post_calls == 0
    assert execution.close_calls == 1
    provider.backend.release_plan(plan)


def test_native_execution_dispatch_explicit_recycle_is_backend_idle(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    plan.local_handle._handle = 0x1000
    plan.remote_handle._handle = 0x2000
    provider.agent.backends = {"UCX": 0x3000}
    execution = _FakeExecutionDispatch()
    provider.backend._raw_request_slot_factory = lambda *args: execution
    provider.backend._raw_write_operation = "NATIVE_WRITE"
    adopted = []

    provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=lambda slot, *, execution_dispatch=None: adopted.append(slot),
    )
    slot = adopted[0]
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None)
        is transfer.WorkState.COMPLETED
    )
    assert execution.active
    slot.recycle()
    assert not execution.active

    # A retired completion receipt is provider-idle. Backend close consumes
    # the retained native request without requiring another Core transition.
    provider.backend.close()
    assert execution.close_calls == 1


def test_persistent_request_slot_is_adopted_before_make_and_post(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    adopted = []

    def adopt_slot(slot):
        provider.agent.actions.append("adopt-slot")
        adopted.append(slot)

    result = provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=adopt_slot,
    )
    assert result is None
    assert provider.agent.actions == ["adopt-slot"]

    slot = adopted[0]
    original_transfer = provider.agent.transfer

    def assert_work_adopted(handle):
        assert slot._work is not None
        return original_transfer(handle)

    provider.agent.transfer = assert_work_adopted
    slot.start()
    assert provider.agent.actions == ["adopt-slot", "make", "post"]
    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


def test_persistent_request_slot_adoption_interrupt_never_creates_native_work(
    provider,
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    adopted = []

    def interrupt_after_adoption(slot):
        adopted.append(slot)
        raise KeyboardInterrupt()

    result = provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=interrupt_after_adoption,
    )
    assert result is None
    assert len(adopted) == 1
    assert not provider.agent.make_calls
    assert id(adopted[0]) not in provider.backend._slots
    assert id(adopted[0]) not in plan.request_slots
    # The callback may have stored the object before interruption. It remains
    # usable even though the provider dropped its creation-transaction entry.
    adopted[0].start()
    assert adopted[0].state is transfer.WorkState.COMPLETED
    adopted[0].recycle()
    adopted[0].close()
    provider.backend.release_plan(plan)


def test_persistent_request_slot_adoption_interrupt_before_storage_rolls_back(
    provider,
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )

    def interrupt_before_storage(slot):
        del slot
        raise KeyboardInterrupt()

    result = provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=interrupt_before_storage,
    )
    assert result is None
    assert not provider.backend._slots
    assert not plan.request_slots
    assert not provider.agent.make_calls
    provider.backend.release_plan(plan)


def test_persistent_slot_registry_publication_interrupt_rolls_back(provider):
    class InterruptAfterSet(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt()

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    plan.request_slots = InterruptAfterSet()
    adopted = []

    result = provider.backend.create_request_slot(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_slot=adopted.append,
    )

    assert result is None
    assert not adopted
    assert not provider.backend._slots
    assert not plan.request_slots
    assert not provider.agent.make_calls
    provider.backend.release_plan(plan)


def test_persistent_request_slot_recycle_requires_terminal_state(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    slot.start()
    with pytest.raises(transfer.BusyError, match="active NIXL request work"):
        slot.recycle()
    with pytest.raises(transfer.BusyError, match="already active"):
        slot.start()

    slot._work._handle.statuses = ["DONE"]
    assert slot.state is transfer.WorkState.COMPLETED
    slot.recycle()
    slot.recycle()  # Retry-safe after a provider-side commit/interruption window.
    slot.close()
    provider.backend.release_plan(plan)


def test_persistent_request_slot_idle_ack_and_interrupted_prologue_are_explicit(
    provider,
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert slot.state is None

    # Model an asynchronous exception immediately after provider acknowledgement
    # but before a Work or explicit error is installed.
    slot._active = True
    assert slot.state is transfer.WorkState.FAILED
    assert isinstance(slot.error, transfer.BackendFailureError)
    slot.recycle()
    assert slot.state is None

    slot.close()
    provider.backend.release_plan(plan)


def test_persistent_request_slot_never_reposts_failed_handle(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    provider.agent.transfer_error = RuntimeError("post failed")
    slot.start()
    failed_handle = provider.agent.make_calls[-1][-1]
    assert slot.state is transfer.WorkState.FAILED
    assert failed_handle.released
    assert failed_handle.post_calls == 1
    slot.recycle()
    assert slot._work is None

    provider.agent.transfer_error = None
    slot.start()
    replacement = provider.agent.make_calls[-1][-1]
    assert replacement is not failed_handle
    assert len(provider.agent.make_calls) == 2
    assert failed_handle.post_calls == 1
    assert slot.state is transfer.WorkState.COMPLETED
    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


def test_persistent_request_slot_retains_plan_and_releases_request_first(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    local_dlist = provider.agent.prep_calls[-2][-1]
    remote_dlist = provider.agent.prep_calls[-1][-1]
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    slot.start()
    handle = slot._work._handle
    slot.recycle()

    with pytest.raises(transfer.BusyError, match="persistent request slots"):
        provider.backend.release_plan(plan)
    assert handle.release_calls == 0
    assert local_dlist.release_calls == 0
    assert remote_dlist.release_calls == 0

    slot.close()
    assert handle.release_calls == 1
    assert local_dlist.release_calls == 0
    assert remote_dlist.release_calls == 0
    provider.backend.release_plan(plan)
    assert local_dlist.release_calls == 1
    assert remote_dlist.release_calls == 1


def test_persistent_request_slot_concurrent_start_is_single_flight(provider):
    agent = _FakeAgent()
    backend = NixlBackend(
        "multiple-slot-id",
        "multiple-slot",
        "multiple-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.MULTIPLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    agent.submit_status = "PROC"
    agent.poll_statuses = ["PROC"]
    slot = _create_request_slot(
        backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    barrier = threading.Barrier(2)
    results = []

    def start():
        barrier.wait()
        try:
            slot.start()
        except transfer.BusyError:
            results.append("busy")
        else:
            results.append("started")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["busy", "started"]
    assert len(agent.make_calls) == 1
    assert slot._work._lock is not backend._lock
    assert hasattr(slot._work._lock, "acquire")
    slot._work._handle.statuses = ["DONE"]
    assert slot.state is transfer.WorkState.COMPLETED
    slot.recycle()
    slot.close()
    backend.release_plan(plan)
    backend.close()


def test_persistent_request_slots_repost_concurrently_in_multiple_mode(provider):
    barrier = threading.Barrier(2)

    class ConcurrentAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.block_reposts = False

        def transfer(self, handle):
            if self.block_reposts:
                barrier.wait(timeout=2)
            return super().transfer(handle)

    agent = ConcurrentAgent()
    backend = NixlBackend(
        "multiple-parallel-slot-id",
        "multiple-parallel-slot",
        "multiple-parallel-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.MULTIPLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slots = [
        _create_request_slot(
            backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
        for _ in range(2)
    ]
    for slot in slots:
        slot.start()
        assert slot.state is transfer.WorkState.COMPLETED

    agent.block_reposts = True
    errors = []

    def repost(slot):
        try:
            slot.start()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=repost, args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    for slot in slots:
        assert slot.state is transfer.WorkState.COMPLETED
        slot.recycle()
        slot.close()
    backend.release_plan(plan)
    backend.close()


def test_persistent_request_slots_fused_repost_concurrently_in_multiple_mode(
    monkeypatch, provider
):
    success = object()
    in_progress = object()
    fused_barrier = threading.Barrier(2)
    poll_barrier = threading.Barrier(2)

    class FakeRawAgent:
        def __init__(self):
            self.fused_handles = []
            self.poll_handles = []
            self.post_status = success

        def postXferReq(self, handle):
            return self.post_status

        def getXferStatus(self, handle):
            return success

        def postXferReqAndPoll(self, handle, max_polls, timeout_ns):
            assert max_polls == 64
            assert timeout_ns is None
            self.fused_handles.append(handle)
            fused_barrier.wait(timeout=2)
            return success

        def getXferStatusBatch(
            self, handle, max_polls, timeout_ns, timeout_check_interval
        ):
            assert max_polls == 4096
            assert timeout_ns is None
            assert timeout_check_interval == 32
            self.poll_handles.append(handle)
            poll_barrier.wait(timeout=2)
            return success

    class RawOnlyAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = FakeRawAgent()

        def make_prepped_xfer(self, *args, **kwargs):
            handle = super().make_prepped_xfer(*args, **kwargs)
            handle._handle = object()
            return handle

        def transfer(self, handle):
            raise AssertionError("high-level post must not be called")

        def check_xfer_state(self, handle):
            raise AssertionError("high-level status must not be called")

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = FakeRawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = in_progress
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = RawOnlyAgent()
    backend = NixlBackend(
        "multiple-fused-slot-id",
        "multiple-fused-slot",
        "multiple-fused-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.MULTIPLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slots = [
        _create_request_slot(
            backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
        for _ in range(2)
    ]
    for slot in slots:
        slot.start()
        assert slot.state is transfer.WorkState.COMPLETED

    results = []
    errors = []

    def fused_start(slot):
        try:
            results.append(slot.start_and_poll(max_polls=64, timeout_ns=None))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=fused_start, args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert results == [transfer.WorkState.COMPLETED] * 2
    assert {id(handle) for handle in agent.agent.fused_handles} == {
        id(slot._work._native_handle) for slot in slots
    }
    for slot in slots:
        slot.recycle()

    agent.agent.post_status = in_progress
    for slot in slots:
        slot.start()
        assert slot._work._state is transfer.WorkState.RUNNING

    results.clear()

    def fused_poll(slot):
        try:
            results.append(
                slot.poll_bounded(
                    max_polls=4096,
                    timeout_ns=None,
                    timeout_check_interval=32,
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=fused_poll, args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert results == [transfer.WorkState.COMPLETED] * 2
    assert {id(handle) for handle in agent.agent.poll_handles} == {
        id(slot._work._native_handle) for slot in slots
    }
    for slot in slots:
        slot.recycle()
        slot.close()
    backend.release_plan(plan)
    backend.close()


def test_persistent_request_slot_uses_cached_raw_post_and_status(monkeypatch, provider):
    success = object()
    in_progress = object()

    class FakeRawAgent:
        def __init__(self, owner):
            self.owner = owner
            self.posts = []
            self.fused_calls = []
            self.fused_error = None
            self.fused_status = success
            self.batch_calls = []
            self.batch_error = None
            self.batch_statuses = []
            self.status_checks = []
            self.status_results = []

        def postXferReq(self, handle):
            self.posts.append(handle)
            self.owner.make_calls[-1][-1].post_calls += 1
            return in_progress

        def getXferStatus(self, handle):
            self.status_checks.append(handle)
            if self.status_results:
                return self.status_results.pop(0)
            return success

        def postXferReqAndPoll(self, handle, max_polls, timeout_ns):
            self.fused_calls.append((handle, max_polls, timeout_ns))
            self.owner.make_calls[-1][-1].post_calls += 1
            if self.fused_error is not None:
                raise self.fused_error
            return self.fused_status

        def getXferStatusBatch(
            self, handle, max_polls, timeout_ns, timeout_check_interval
        ):
            self.batch_calls.append(
                (handle, max_polls, timeout_ns, timeout_check_interval)
            )
            if self.batch_error is not None:
                raise self.batch_error
            return self.batch_statuses.pop(0)

    class RawOnlyAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = FakeRawAgent(self)

        def make_prepped_xfer(self, *args, **kwargs):
            handle = super().make_prepped_xfer(*args, **kwargs)
            handle._handle = object()
            return handle

        def transfer(self, handle):
            raise AssertionError("high-level post must not be called")

        def check_xfer_state(self, handle):
            raise AssertionError("high-level status must not be called")

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = FakeRawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = in_progress
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = RawOnlyAgent()
    backend = NixlBackend(
        "raw-slot-id",
        "raw-slot",
        "raw-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SINGLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    slot.start()
    assert slot.state is transfer.WorkState.COMPLETED

    identities = []
    fused_budgets = ((64, None), (0, None), (2, 123), (1, 0))
    for max_polls, timeout_ns in fused_budgets:
        assert (
            slot.start_and_poll(max_polls=max_polls, timeout_ns=timeout_ns)
            is transfer.WorkState.COMPLETED
        )
        work = slot._work
        identities.append(
            (id(work), id(work._request), id(work._handle), id(work._lock))
        )
        slot.recycle()

    native_handle = slot._work._native_handle
    assert len(set(identities)) == 1
    assert agent.agent.posts == [native_handle]
    assert agent.agent.status_checks == [native_handle]
    assert agent.agent.fused_calls == [
        (native_handle, max_polls, timeout_ns)
        for max_polls, timeout_ns in fused_budgets
    ]
    assert slot._work._handle.post_calls == 5

    agent.agent.fused_status = in_progress
    agent.agent.batch_statuses = [in_progress, success]
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None) is transfer.WorkState.RUNNING
    )
    assert (
        slot.poll_bounded(
            max_polls=17,
            timeout_ns=0,
            timeout_check_interval=32,
        )
        is transfer.WorkState.RUNNING
    )
    assert (
        slot.poll_bounded(
            max_polls=4096,
            timeout_ns=100_000,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    assert agent.agent.batch_calls == [
        (native_handle, 17, 0, 32),
        (native_handle, 4096, 100_000, 32),
    ]
    assert (
        id(slot._work),
        id(slot._work._request),
        id(slot._work._handle),
        id(slot._work._lock),
    ) == identities[0]
    calls_before_recycle = (
        len(agent.agent.posts),
        len(agent.agent.fused_calls),
        len(agent.agent.batch_calls),
        len(agent.agent.status_checks),
        slot._work._handle.release_calls,
    )
    slot.recycle()
    assert (
        len(agent.agent.posts),
        len(agent.agent.fused_calls),
        len(agent.agent.batch_calls),
        len(agent.agent.status_checks),
        slot._work._handle.release_calls,
    ) == calls_before_recycle

    invalid_status = object()
    invalid_handle = slot._work._handle
    agent.agent.batch_statuses = [invalid_status]
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None) is transfer.WorkState.RUNNING
    )
    assert (
        slot.poll_bounded(
            max_polls=4096,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is transfer.WorkState.FAILED
    )
    assert invalid_handle.released
    assert slot._work._poll_bounded_xfer is None
    assert "invalid bounded-poll status" in str(slot.error)
    slot.recycle()
    assert slot._work is None

    agent.agent.fused_status = success
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None)
        is transfer.WorkState.COMPLETED
    )
    slot.recycle()

    batch_failed_handle = slot._work._handle
    batch_failed_handle.release_errors = [RuntimeError("request may still be active")]
    agent.agent.fused_status = in_progress
    agent.agent.batch_error = RuntimeError("bounded poll failed")
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None) is transfer.WorkState.RUNNING
    )
    assert (
        slot.poll_bounded(
            max_polls=4096,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is transfer.WorkState.RUNNING
    )
    assert not batch_failed_handle.released
    assert slot._work._poll_bounded_xfer is not None
    # A later probe retries retirement before any native status call. Only
    # after release proves native access stopped is the generation FAILED and
    # every cached native callable cleared.
    assert (
        slot.poll_bounded(
            max_polls=4096,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is transfer.WorkState.FAILED
    )
    assert batch_failed_handle.released
    assert slot._work._poll_bounded_xfer is None
    assert isinstance(slot.error.__cause__, RuntimeError)
    slot.recycle()
    assert slot._work is None

    agent.agent.batch_error = None
    # Simulate an older compiled binding which has raw one-at-a-time status
    # but no batch method. The provider must retain the bounded contract in
    # Python, including one observation at timeout zero.
    backend._raw_get_xfer_status_batch = None
    agent.agent.status_results = [in_progress, success]
    assert slot.start_and_poll(max_polls=1, timeout_ns=0) is transfer.WorkState.RUNNING
    assert slot._work._poll_bounded_xfer is None
    prior_batch_calls = len(agent.agent.batch_calls)
    assert (
        slot.poll_bounded(
            max_polls=1,
            timeout_ns=0,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    assert len(agent.agent.batch_calls) == prior_batch_calls
    backend._raw_get_xfer_status_batch = agent.agent.getXferStatusBatch

    failed_handle = slot._work._handle
    agent.agent.fused_error = KeyboardInterrupt()
    previous_post_calls = failed_handle.post_calls
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None) is transfer.WorkState.FAILED
    )
    assert failed_handle.released
    assert failed_handle.post_calls == previous_post_calls + 1
    slot.recycle()
    assert slot._work is None

    agent.agent.fused_error = None
    agent.agent.fused_status = success
    assert (
        slot.start_and_poll(max_polls=64, timeout_ns=None)
        is transfer.WorkState.COMPLETED
    )
    assert slot._work._handle is not failed_handle
    assert failed_handle.post_calls == previous_post_calls + 1
    slot.recycle()
    slot.close()
    backend.release_plan(plan)
    backend.close()


def test_persistent_request_slot_fused_fallback_is_bounded_and_validated(
    monkeypatch, provider
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    for invalid_max_polls in (-1, True, 1.0, 1 << 63):
        with pytest.raises((TypeError, ValueError)):
            slot.start_and_poll(
                max_polls=invalid_max_polls,
                timeout_ns=None,
            )
    for invalid_timeout_ns in (-1, True, 1.0, 1 << 63):
        with pytest.raises((TypeError, ValueError)):
            slot.start_and_poll(max_polls=1, timeout_ns=invalid_timeout_ns)
    for invalid_max_polls in (0, -1, True, 1.0, 1 << 63):
        with pytest.raises((TypeError, ValueError)):
            slot.poll_bounded(
                max_polls=invalid_max_polls,
                timeout_ns=None,
                timeout_check_interval=32,
            )
    for invalid_timeout_ns in (-1, True, 1.0, 1 << 63):
        with pytest.raises((TypeError, ValueError)):
            slot.poll_bounded(
                max_polls=1,
                timeout_ns=invalid_timeout_ns,
                timeout_check_interval=32,
            )
    for invalid_timeout_check_interval in (0, -1, True, 1.0, 1 << 63):
        with pytest.raises((TypeError, ValueError)):
            slot.poll_bounded(
                max_polls=1,
                timeout_ns=None,
                timeout_check_interval=invalid_timeout_check_interval,
            )
    assert slot.state is None
    assert (
        slot.poll_bounded(
            max_polls=1,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is None
    )
    assert not provider.agent.make_calls

    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC", "DONE"]
    assert (
        slot.start_and_poll(max_polls=1, timeout_ns=None) is transfer.WorkState.RUNNING
    )
    assert slot._work._handle.statuses == ["DONE"]
    assert (
        slot.poll_bounded(
            max_polls=1,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    slot.recycle()

    slot._work._handle.statuses = ["PROC", "DONE"]
    assert slot.start_and_poll(max_polls=1, timeout_ns=0) is transfer.WorkState.RUNNING
    # A zero timeout still performs the first observation and stops between
    # observations, both during start fusion and later continuation chunks.
    assert slot._work._handle.statuses == ["DONE"]
    assert (
        slot.poll_bounded(
            max_polls=1,
            timeout_ns=0,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    slot.recycle()

    # The source/old-binding fallback uses the same sampled timeout schedule as
    # the compiled loop: after observation 1, then at observation 32 here.
    slot._work._handle.statuses = ["PROC"] * 33 + ["DONE"]
    slot.start()
    clock_values = iter((0, 10, 200))
    provider_module = sys.modules[NixlBackend.__module__]
    monkeypatch.setattr(
        provider_module.time, "monotonic_ns", lambda: next(clock_values)
    )
    assert (
        slot.poll_bounded(
            max_polls=33,
            timeout_ns=100,
            timeout_check_interval=32,
        )
        is transfer.WorkState.RUNNING
    )
    assert slot._work._handle.statuses == ["PROC", "DONE"]
    assert (
        slot.poll_bounded(
            max_polls=2,
            timeout_ns=None,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    slot.recycle()

    # A terminal observation at the sampling boundary wins over the expired
    # soft deadline and does not perform a redundant clock read.
    slot._work._handle.statuses = ["PROC"] * 31 + ["DONE"]
    slot.start()
    terminal_clock_values = iter((0, 10, 200))
    terminal_clock_calls = []

    def sampled_clock():
        value = next(terminal_clock_values)
        terminal_clock_calls.append(value)
        return value

    monkeypatch.setattr(provider_module.time, "monotonic_ns", sampled_clock)
    assert (
        slot.poll_bounded(
            max_polls=33,
            timeout_ns=100,
            timeout_check_interval=32,
        )
        is transfer.WorkState.COMPLETED
    )
    assert terminal_clock_calls == [0, 10]
    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "terminal_state",
    (
        transfer.WorkState.RUNNING,
        transfer.WorkState.FAILED,
        transfer.WorkState.CANCELLED,
    ),
)
def test_deferred_success_never_folds_non_success_state(provider, terminal_state):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert slot.start_and_poll(max_polls=1, timeout_ns=0) is transfer.WorkState.RUNNING
    if terminal_state is not transfer.WorkState.RUNNING:
        slot._work._state = terminal_state
        assert slot.poll_state() is terminal_state

    with pytest.raises(transfer.BusyError, match="already active"):
        slot.start()
    with pytest.raises(transfer.BusyError, match="already active"):
        slot.start_and_poll(max_polls=1, timeout_ns=0)

    if terminal_state is transfer.WorkState.RUNNING:
        # Model a status failure whose native retirement remains ambiguous.
        # Neither that state nor a retry may resurrect a completed receipt.
        slot._work._pending_error = transfer.BackendFailureError(
            "request outcome is ambiguous"
        )
        slot._work._handle.release_errors = [RuntimeError("still active")]
        assert slot.poll_state() is transfer.WorkState.RUNNING
        assert not slot._repostable
        with pytest.raises(transfer.BusyError, match="already active"):
            slot.start()
    with pytest.raises(transfer.BusyError, match="active NIXL request slot"):
        slot.close()

    if terminal_state is transfer.WorkState.RUNNING:
        assert slot.poll_state() is transfer.WorkState.FAILED
    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


def test_observational_cancel_preserves_clean_completion_receipt(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["DONE"]
    slot.start()
    assert slot._work._state is transfer.WorkState.RUNNING
    assert not slot._repostable
    assert not slot.cancel()
    # NIXL cancellation is a status observation, not a destructive native
    # operation. Completion won the race, so its handle remains repost-safe.
    assert slot._repostable
    assert slot.poll_state() is transfer.WorkState.COMPLETED
    assert slot._repostable

    # Deferred start consumes the same receipt and reposts the same request.
    handle = slot._work._handle
    slot.start()
    assert slot._work._handle is handle
    assert handle.post_calls == 2
    assert slot.poll_state() is transfer.WorkState.COMPLETED

    # An asynchronous interruption during the observational poll leaves the
    # single receipt latch invalid. A later clean COMPLETED observation may
    # safely republish it because no destructive cancel was attempted.
    slot.recycle()
    slot.start()
    provider.agent.check_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        slot.cancel()
    assert not slot._repostable
    provider.agent.check_error = None
    assert slot.poll_state() is transfer.WorkState.COMPLETED
    assert slot._repostable

    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "thread_mode",
    (
        transfer.ThreadMode.SINGLE,
        transfer.ThreadMode.CALLER_SERIALIZED,
        transfer.ThreadMode.SERIALIZED,
    ),
)
def test_persistent_request_slot_uses_no_provider_lock_when_core_serializes(
    provider, thread_mode
):
    provider_module = sys.modules[NixlBackend.__module__]
    agent = _FakeAgent()
    backend = NixlBackend(
        f"{thread_mode.value}-slot-id",
        f"{thread_mode.value}-slot",
        f"{thread_mode.value}-slot-incarnation",
        transfer.ProgressMode.MANUAL,
        thread_mode,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    plan = _prepare(
        backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    slot.start()
    assert slot._work._lock is None
    assert backend._lock is provider_module._NOOP_LOCK
    slot.recycle()
    slot.start()
    assert slot.state is transfer.WorkState.COMPLETED
    slot.recycle()
    slot.close()
    backend.release_plan(plan)
    backend.close()


def test_concurrent_same_token_requests_never_share_native_handle(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]

    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert len(provider.agent.make_calls) == 2
    assert first._handle is not second._handle

    first._handle.statuses = ["DONE"]
    second._handle.statuses = ["DONE"]
    assert first.state is transfer.WorkState.COMPLETED
    assert second.state is transfer.WorkState.COMPLETED
    first.close()
    second.close()
    assert plan.cached_request_count == 2
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_submissions_overlap_legacy_make_and_post(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    make_barrier = threading.Barrier(2)
    post_barrier = threading.Barrier(2)
    original_make = provider.agent.make_prepped_xfer
    original_post = provider.agent.transfer

    def concurrent_make(*args, **kwargs):
        handle = original_make(*args, **kwargs)
        make_barrier.wait(3)
        return handle

    def concurrent_post(*args, **kwargs):
        post_barrier.wait(3)
        return original_post(*args, **kwargs)

    provider.agent.make_prepped_xfer = concurrent_make
    provider.agent.transfer = concurrent_post
    works = _run_two_concurrent_calls(
        lambda: _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
    )

    assert len(provider.agent.make_calls) == 2
    assert len({id(work._handle) for work in works}) == 2
    assert provider.backend._submission_reservations == {}
    for work in works:
        assert work.state is transfer.WorkState.COMPLETED
        work.close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
@pytest.mark.parametrize("max_polls", [0, 64])
def test_multiple_prevalidated_submissions_overlap_exact_and_fused_make_and_post(
    provider, max_polls
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=True,
    )
    selection = _single_block_selection(transfer.TransferOp.WRITE, plan)
    make_barrier = threading.Barrier(2)
    post_barrier = threading.Barrier(2)
    original_make = provider.agent.make_prepped_xfer
    raw_calls = []
    fused_calls = []

    def concurrent_make(*args, **kwargs):
        handle = original_make(*args, **kwargs)
        make_barrier.wait(3)
        return handle

    def raw_post(native_handle):
        raw_calls.append(native_handle)
        post_barrier.wait(3)
        return "DONE"

    def fused_post(native_handle, fused_max_polls, timeout_ns):
        fused_calls.append((native_handle, fused_max_polls, timeout_ns))
        post_barrier.wait(3)
        return "DONE"

    provider.agent.make_prepped_xfer = concurrent_make
    provider.backend._raw_post_xfer = raw_post
    provider.backend._raw_post_xfer_and_poll = fused_post

    def submit_prevalidated():
        adopted = []
        state = provider.backend.submit_prevalidated_and_poll(
            transfer.TransferOp.WRITE,
            plan,
            selection=selection,
            notification=None,
            max_polls=max_polls,
            timeout_ns=None,
            adopt_work=adopted.append,
        )
        assert state is transfer.WorkState.COMPLETED
        assert len(adopted) == 1
        return adopted[0]

    works = _run_two_concurrent_calls(submit_prevalidated)
    assert len(provider.agent.make_calls) == 2
    if max_polls == 0:
        assert len(raw_calls) == 2
        assert fused_calls == []
    else:
        assert raw_calls == []
        assert len(fused_calls) == 2
        assert all(call[1:] == (max_polls, None) for call in fused_calls)
    assert len({id(work._handle) for work in works}) == 2
    for work in works:
        work.close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_submission_reservation_blocks_teardown_during_make_and_post(
    provider,
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    make_entered = threading.Event()
    allow_make = threading.Event()
    post_entered = threading.Event()
    allow_post = threading.Event()
    original_make = provider.agent.make_prepped_xfer
    original_post = provider.agent.transfer
    works = []
    errors = []

    def blocking_make(*args, **kwargs):
        handle = original_make(*args, **kwargs)
        make_entered.set()
        if not allow_make.wait(5):
            raise TimeoutError("test did not release native make")
        return handle

    def blocking_post(*args, **kwargs):
        post_entered.set()
        if not allow_post.wait(5):
            raise TimeoutError("test did not release native post")
        return original_post(*args, **kwargs)

    def submit_one():
        try:
            works.append(
                _submit(
                    provider.backend,
                    transfer.TransferOp.WRITE,
                    plan,
                    local_indices=None,
                    remote_indices=None,
                    notification=None,
                )
            )
        except BaseException as error:
            errors.append(error)

    def assert_busy_without_blocking(operation):
        outcomes = []

        def run():
            try:
                operation()
            except BaseException as error:
                outcomes.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()
        assert len(outcomes) == 1
        assert isinstance(outcomes[0], transfer.BusyError)

    provider.agent.make_prepped_xfer = blocking_make
    provider.agent.transfer = blocking_post
    submit_thread = threading.Thread(target=submit_one)
    submit_thread.start()
    try:
        assert make_entered.wait(3)
        assert_busy_without_blocking(lambda: provider.backend.release_plan(plan))
        assert_busy_without_blocking(provider.backend.close)
        assert all(
            call[-1].release_calls == 0 for call in provider.agent.prep_calls[-2:]
        )
        assert not provider.backend._closed

        allow_make.set()
        assert post_entered.wait(3)
        assert_busy_without_blocking(lambda: provider.backend.release_plan(plan))
        assert_busy_without_blocking(provider.backend.close)
        assert all(
            call[-1].release_calls == 0 for call in provider.agent.prep_calls[-2:]
        )
        assert not provider.backend._closed
    finally:
        allow_make.set()
        allow_post.set()
        submit_thread.join(5)
    assert not submit_thread.is_alive()
    assert errors == []
    assert len(works) == 1
    works[0].close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_cached_request_has_one_concurrent_claim(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    cached_handle = first._handle
    first.close()
    post_barrier = threading.Barrier(2)
    original_post = provider.agent.transfer

    def concurrent_post(*args, **kwargs):
        post_barrier.wait(3)
        return original_post(*args, **kwargs)

    provider.agent.transfer = concurrent_post
    works = _run_two_concurrent_calls(
        lambda: _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            reuse_token=token,
        )
    )

    assert len(provider.agent.make_calls) == 2
    assert sum(work._handle is cached_handle for work in works) == 1
    assert len({id(work._handle) for work in works}) == 2
    for work in works:
        work.close()
    assert plan.cached_request_count == 2
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_reservation_publication_interrupt_is_reconciled(provider):
    class InterruptAfterSet(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt()

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.backend._submission_reservations = InterruptAfterSet()
    adopted = []
    with pytest.raises(KeyboardInterrupt):
        provider.backend.submit(
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            adopt_work=adopted.append,
        )
    assert adopted == []
    assert provider.agent.make_calls == []
    assert provider.backend._submission_reservations == {}
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_teardown_finishes_abandoned_prework_abort(provider):
    class InterruptAbortLock:
        def __init__(self, lock):
            self.lock = lock
            self.remaining_interrupts = 0
            self.acquired = False

        def __enter__(self):
            if self.remaining_interrupts:
                self.remaining_interrupts -= 1
                raise KeyboardInterrupt()
            self.lock.acquire()
            self.acquired = True
            return self

        def __exit__(self, *args):
            if self.acquired:
                self.acquired = False
                self.lock.release()
            return False

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    interrupting_lock = InterruptAbortLock(provider.backend._lock)
    provider.backend._lock = interrupting_lock

    def interrupt_selection(*args, **kwargs):
        del args, kwargs
        interrupting_lock.remaining_interrupts = 2
        raise KeyboardInterrupt()

    provider.backend._request_selection_locked = interrupt_selection
    with pytest.raises(KeyboardInterrupt):
        provider.backend.submit(
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            adopt_work=lambda _work: None,
        )
    reservations = provider.backend._submission_reservations
    assert reservations is not None
    assert len(reservations) == 1
    abandoned = next(iter(reservations.values()))
    assert abandoned.aborting
    assert not abandoned.done

    provider.backend.release_plan(plan)
    assert reservations == {}
    assert plan.released


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_finish_lock_interrupt_cannot_strand_reservation(provider):
    class InterruptOnceLock:
        def __init__(self, lock):
            self.lock = lock
            self.armed = False
            self.acquired = False

        def __enter__(self):
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt()
            self.lock.acquire()
            self.acquired = True
            return self

        def __exit__(self, *args):
            if self.acquired:
                self.acquired = False
                self.lock.release()
            return False

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    interrupting_lock = InterruptOnceLock(provider.backend._lock)
    provider.backend._lock = interrupting_lock
    adopted = []

    def adopt_and_arm(work):
        adopted.append(work)
        interrupting_lock.armed = True

    result = provider.backend.submit(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_work=adopt_and_arm,
    )

    assert result is None
    assert len(adopted) == 1
    assert adopted[0].state is transfer.WorkState.FAILED
    assert provider.backend._submission_reservations == {}
    adopted[0].close()
    provider.backend.release_plan(plan)


def test_request_reuse_is_separated_by_opaque_token(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    first_token = object()
    second_token = object()

    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=first_token,
    )
    first_handle = first._handle
    first.close()
    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=second_token,
    )
    second_handle = second._handle
    second.close()
    assert second_handle is not first_handle
    assert len(provider.agent.make_calls) == 2

    first_again = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=first_token,
    )
    assert first_again._handle is first_handle
    first_again.close()
    provider.backend.release_plan(plan)


def test_none_reuse_token_always_recreates_and_releases_request(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    handles = []
    for _ in range(2):
        work = _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            reuse_token=None,
        )
        handles.append(work._handle)
        work.close()

    assert len(provider.agent.make_calls) == 2
    assert handles[0] is not handles[1]
    assert all(handle.released for handle in handles)
    assert plan.cached_request_count == 0
    provider.backend.release_plan(plan)


def test_post_failure_request_is_released_and_never_reused(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    provider.agent.transfer_error = RuntimeError("post failed")
    failed = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    failed_handle = provider.agent.make_calls[-1][-1]
    assert failed.state is transfer.WorkState.FAILED
    assert failed_handle.released
    failed.close()
    assert plan.cached_request_count == 0

    provider.agent.transfer_error = None
    succeeded = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert len(provider.agent.make_calls) == 2
    assert succeeded._handle is not failed_handle
    succeeded.close()
    provider.backend.release_plan(plan)


def test_failed_status_request_is_released_and_never_reused(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    provider.agent.submit_status = "ERR"
    failed = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    failed_handle = provider.agent.make_calls[-1][-1]
    assert failed.state is transfer.WorkState.FAILED
    assert failed_handle.released
    failed.close()

    provider.agent.submit_status = "DONE"
    succeeded = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert len(provider.agent.make_calls) == 2
    succeeded.close()
    provider.backend.release_plan(plan)


def test_request_pool_cap_releases_excess_idle_handles(provider):
    provider.backend._max_cached_requests_per_plan = 1
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    first_handle = first._handle
    second_handle = second._handle
    first.close()
    second.close()

    assert plan.cached_request_count == 1
    assert not first_handle.released
    assert second_handle.released
    reused = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert reused._handle is first_handle
    assert len(provider.agent.make_calls) == 2
    reused.close()
    provider.backend.release_plan(plan)


def test_request_pool_evicts_stale_token_to_admit_new_binding(provider):
    provider.backend._max_cached_requests_per_plan = 1
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    stale_token = object()
    hot_token = object()

    stale = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=stale_token,
    )
    stale_handle = stale._handle
    stale.close()

    hot = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=hot_token,
    )
    hot_handle = hot._handle
    hot.close()
    assert stale_handle.released
    assert not hot_handle.released
    assert plan.cached_request_count == 1

    hot_again = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=hot_token,
    )
    assert hot_again._handle is hot_handle
    assert len(provider.agent.make_calls) == 2
    hot_again.close()
    provider.backend.release_plan(plan)


def test_plan_release_retries_cached_requests_before_descriptor_lists(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    handles = []
    for token in (object(), object()):
        work = _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            reuse_token=token,
        )
        handles.append(work._handle)
        work.close()
    local_handle = provider.agent.prep_calls[-2][4]
    remote_handle = provider.agent.prep_calls[-1][4]
    handles[0].release_errors.append(RuntimeError("transient request release"))

    with pytest.raises(transfer.BackendFailureError, match="cached transfer"):
        provider.backend.release_plan(plan)
    # Reconciliation idempotently retries the quarantined request before this
    # failed cleanup returns; the prior error still makes the call fail so the
    # caller observes that an interruption/failure occurred.
    assert handles[0].release_calls == 2
    assert handles[1].release_calls == 1
    assert plan.cached_request_count == 0
    assert local_handle.release_calls == 0
    assert remote_handle.release_calls == 0

    provider.backend.release_plan(plan)
    assert handles[0].release_calls == 2
    assert handles[1].release_calls == 1
    assert plan.cached_request_count == 0
    assert local_handle.release_calls == 1
    assert remote_handle.release_calls == 1


@pytest.mark.parametrize("post_release", (False, True))
def test_plan_drain_quarantines_pre_and_post_release_interrupts(provider, post_release):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=object(),
    )
    handle = work._handle
    work.close()
    (request,) = plan.requests.values()
    original_release = handle.release
    if post_release:

        def release_with_cut():
            original_release()
            raise KeyboardInterrupt("post-release drain cut")

        handle.release = release_with_cut
        message = "post-release drain cut"
    else:
        handle.release_errors.append(KeyboardInterrupt("pre-release drain cut"))
        message = "pre-release drain cut"

    with pytest.raises(KeyboardInterrupt, match=message):
        provider.backend.release_plan(plan)
    assert plan.request_pool_dirty
    assert request.state == _REQUEST_RETIRING
    assert plan.local_handle.release_calls == 0
    assert plan.remote_handle.release_calls == 0

    handle.release = original_release
    provider.backend.release_plan(plan)
    assert handle.release_calls == 2
    assert request.state != "idle"
    assert plan.cached_request_count == 0


@pytest.mark.parametrize("post_release", (False, True))
def test_lru_eviction_quarantines_pre_and_post_release_interrupts(
    provider, post_release
):
    provider.backend._max_cached_requests_per_plan = 1
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    stale_token = object()
    hot_token = object()
    stale = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=stale_token,
    )
    stale_handle = stale._handle
    stale.close()
    stale_request = next(
        request for request in plan.requests.values() if request.handle is stale_handle
    )
    hot = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=hot_token,
    )
    hot_handle = hot._handle
    original_release = stale_handle.release
    if post_release:

        def release_with_cut():
            original_release()
            raise KeyboardInterrupt("post-release eviction cut")

        stale_handle.release = release_with_cut
        message = "post-release eviction cut"
    else:
        stale_handle.release_errors.append(
            KeyboardInterrupt("pre-release eviction cut")
        )
        message = "pre-release eviction cut"

    with pytest.raises(KeyboardInterrupt, match=message):
        hot.close()
    assert plan.request_pool_dirty
    assert stale_request.state == _REQUEST_RETIRING

    stale_handle.release = original_release
    hot.close()
    assert stale_handle.release_calls == 2
    assert stale_handle.released
    assert stale_request.state != "idle"
    assert plan.cached_request_count == 1
    reused = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=hot_token,
    )
    assert reused._handle is hot_handle
    reused.close()
    provider.backend.release_plan(plan)


def test_notification_bound_requests_are_separated_by_core_token(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    first_token = object()
    second_token = object()
    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"first-notification",
        reuse_token=first_token,
    )
    first_handle = first._handle
    first.close()
    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"second-notification",
        reuse_token=second_token,
    )
    second_handle = second._handle
    second.close()

    assert first_handle is not second_handle
    assert [call[5] for call in provider.agent.make_calls[-2:]] == [
        b"first-notification",
        b"second-notification",
    ]
    first_again = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=b"first-notification",
        reuse_token=first_token,
    )
    assert first_again._handle is first_handle
    first_again.close()
    provider.backend.release_plan(plan)


def test_unhashable_reuse_token_uses_opaque_identity(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = []
    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    handle = first._handle
    first.close()
    second = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert second._handle is handle
    assert len(provider.agent.make_calls) == 1
    second.close()
    provider.backend.release_plan(plan)


def test_cache_return_interruption_reconciles_without_double_ownership(provider):
    class InterruptAfterAppend(list):
        def append(self, value):
            super().append(value)
            raise KeyboardInterrupt()

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    old_work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    handle = old_work._handle
    plan.cached_requests[id(token)] = InterruptAfterAppend()

    with pytest.raises(KeyboardInterrupt):
        old_work.close()
    assert old_work._released
    assert plan.request_pool_dirty
    assert next(iter(plan.requests.values())).state == "idle"

    new_work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert new_work._handle is handle
    old_work.close()
    assert old_work._handle is None
    assert not handle.released
    new_work.close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "provider",
    [transfer.ThreadMode.SERIALIZED, transfer.ThreadMode.MULTIPLE],
    indirect=True,
)
def test_cache_checkout_interruption_keeps_request_in_plan_master(provider):
    class InterruptAfterPop(list):
        def pop(self, *args):
            super().pop(*args)
            raise KeyboardInterrupt()

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    token = object()
    first = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    handle = first._handle
    first.close()
    plan.cached_requests[id(token)] = InterruptAfterPop(plan.cached_requests[id(token)])

    with pytest.raises(KeyboardInterrupt):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            reuse_token=token,
        )
    assert plan.request_pool_dirty
    assert next(iter(plan.requests.values())).state == "idle"

    recovered = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        reuse_token=token,
    )
    assert recovered._handle is handle
    assert len(provider.agent.make_calls) == 1
    recovered.close()
    provider.backend.release_plan(plan)


def test_submit_adopts_work_before_native_post(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    adopted = []

    def adopt_work(work):
        provider.agent.actions.append("adopt")
        adopted.append(work)

    result = provider.backend.submit(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_work=adopt_work,
    )

    assert result is None
    assert provider.agent.actions == ["make", "adopt", "post"]
    adopted[0].close()
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "provider",
    [transfer.ThreadMode.SERIALIZED, transfer.ThreadMode.MULTIPLE],
    indirect=True,
)
def test_submit_does_not_post_when_core_adoption_fails(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )

    def reject_work(work):
        del work
        raise KeyboardInterrupt()

    result = provider.backend.submit(
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
        adopt_work=reject_work,
    )

    assert result is None
    assert provider.agent.actions == ["make"]
    assert provider.agent.make_calls[-1][-1].released
    assert not provider.backend._works
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "provider",
    [transfer.ThreadMode.SINGLE, transfer.ThreadMode.SERIALIZED],
    indirect=True,
)
def test_submit_registry_publication_interrupt_retires_unadopted_work(provider):
    class InterruptAfterSet(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt()

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.backend._works = InterruptAfterSet(provider.backend._works)
    adopted = []

    with pytest.raises(KeyboardInterrupt):
        provider.backend.submit(
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            adopt_work=adopted.append,
        )

    assert not adopted
    assert provider.agent.actions == ["make"]
    assert provider.agent.make_calls[-1][-1].released
    assert not provider.backend._works
    provider.backend.release_plan(plan)


@pytest.mark.parametrize(
    "provider",
    [transfer.ThreadMode.SINGLE, transfer.ThreadMode.SERIALIZED],
    indirect=True,
)
@pytest.mark.parametrize("route", ["legacy", "prevalidated"])
def test_fresh_request_rolls_back_when_work_construction_is_interrupted(
    provider, monkeypatch, route
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=route == "prevalidated",
    )
    provider_module = sys.modules[NixlBackend.__module__]
    original_work = provider_module.NixlWork

    class InterruptOnceBeforeWork(original_work):
        armed = True

        def __init__(self, *args, **kwargs):
            if type(self).armed:
                type(self).armed = False
                raise KeyboardInterrupt("work construction interrupted")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(provider_module, "NixlWork", InterruptOnceBeforeWork)
    selection = (
        _single_block_selection(transfer.TransferOp.WRITE, plan)
        if route == "prevalidated"
        else None
    )

    def launch():
        if selection is None:
            return _submit(
                provider.backend,
                transfer.TransferOp.WRITE,
                plan,
                local_indices=None,
                remote_indices=None,
                notification=None,
            )
        adopted = []
        result = provider.backend.submit_prevalidated(
            transfer.TransferOp.WRITE,
            plan,
            selection=selection,
            notification=None,
            reuse_token=None,
            adopt_work=adopted.append,
        )
        assert result is None
        assert len(adopted) == 1
        return adopted[0]

    with pytest.raises(KeyboardInterrupt, match="work construction interrupted"):
        launch()

    abandoned_handle = provider.agent.make_calls[-1][-1]
    assert abandoned_handle.released
    assert abandoned_handle.release_calls == 1
    assert abandoned_handle.post_calls == 0
    assert not plan.requests
    assert not provider.backend._works
    assert provider.agent.actions == ["make"]

    recovered = launch()
    assert recovered is not None
    assert len(provider.agent.make_calls) == 2
    assert provider.agent.actions == ["make", "make", "post"]
    recovered.close()
    assert not plan.requests
    assert not provider.backend._works
    provider.backend.release_plan(plan)


def test_fresh_request_record_interrupt_rolls_back_before_work(provider, monkeypatch):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider_module = sys.modules[NixlBackend.__module__]
    original_record = provider_module.NixlBackend._record_fresh_request_locked

    def interrupt_after_record(backend, target_plan, request):
        original_record(backend, target_plan, request)
        raise KeyboardInterrupt("request record interrupted")

    monkeypatch.setattr(
        provider_module.NixlBackend,
        "_record_fresh_request_locked",
        interrupt_after_record,
    )
    with pytest.raises(KeyboardInterrupt, match="request record interrupted"):
        provider.backend.submit(
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            adopt_work=lambda work: pytest.fail(f"unexpected adoption: {work!r}"),
        )

    handle = provider.agent.make_calls[-1][-1]
    assert handle.released
    assert handle.release_calls == 1
    assert handle.post_calls == 0
    assert not plan.requests
    assert not provider.backend._works
    provider.backend.release_plan(plan)


def test_pre_adoption_release_interrupt_remains_plan_owned(provider):
    class InterruptBeforeSet(dict):
        def __setitem__(self, key, work):
            del key
            work._handle.release_errors.append(
                KeyboardInterrupt("native release interrupted")
            )
            raise KeyboardInterrupt("work registry interrupted")

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.backend._works = InterruptBeforeSet()
    adopted = []
    with pytest.raises(KeyboardInterrupt, match="work registry interrupted"):
        provider.backend.submit(
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
            adopt_work=adopted.append,
        )

    assert not adopted
    assert not provider.backend._works
    assert len(plan.requests) == 1
    request = next(iter(plan.requests.values()))
    assert request.handle is provider.agent.make_calls[-1][-1]
    assert request.handle.release_calls == 1
    assert not request.handle.released
    assert request.handle.post_calls == 0

    # The plan master record is the recovery anchor. Teardown retries the
    # interrupted native release before touching descriptor dependencies.
    provider.backend.release_plan(plan)
    assert request.handle.released
    assert request.handle.release_calls == 2
    assert plan.released


@pytest.mark.parametrize(
    "provider",
    [transfer.ThreadMode.SERIALIZED, transfer.ThreadMode.MULTIPLE],
    indirect=True,
)
def test_in_progress_submission_publication_interrupt_cannot_strand_work(
    provider, monkeypatch
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"

    def interrupt_after_ready(work, status):
        assert status == "PROC"
        work._submission_ready = True
        raise KeyboardInterrupt()

    provider_module = sys.modules[NixlBackend.__module__]
    monkeypatch.setattr(
        provider_module.NixlWork,
        "_submission_succeeded_locked",
        interrupt_after_ready,
    )
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert work.state is transfer.WorkState.FAILED
    assert work._submission_ready
    assert work._handle is None
    assert provider.agent.make_calls[-1][-1].released
    work.close()
    provider.backend.release_plan(plan)


def test_failure_release_post_commit_interrupt_publishes_terminal_before_cleanup(
    provider,
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )

    def interrupting_post(handle):
        provider.agent.actions.append("post")
        handle.post_calls += 1
        release = handle.release

        def interrupt_after_release():
            if handle.released:
                return
            release()
            raise KeyboardInterrupt()

        handle.release = interrupt_after_release
        raise RuntimeError("post failed")

    provider.agent.transfer = interrupting_post
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert work._handle is None
    handle = provider.agent.make_calls[-1][-1]
    assert handle.released
    assert handle.release_calls == 1
    work.close()
    provider.backend.release_plan(plan)


def test_failure_cleanup_store_interrupt_cannot_restore_running_state(provider):
    provider_module = sys.modules[NixlBackend.__module__]

    class InterruptAfterHandleClear(provider_module.NixlWork):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if (
                name == "_handle"
                and value is None
                and self.__dict__.get("_interrupt_handle_clear", False)
            ):
                self._interrupt_handle_clear = False
                raise KeyboardInterrupt()

    handle = _FakeXfer(["PROC"])
    work = InterruptAfterHandleClear(
        provider.backend,
        handle,
        state=transfer.WorkState.RUNNING,
        submission_ready=False,
    )
    provider.backend._works[id(work)] = work
    work._interrupt_handle_clear = True
    failure = transfer.BackendFailureError("submission failed")

    work._submission_failed(failure)

    assert handle.released
    assert handle.release_calls == 1
    assert work._state is transfer.WorkState.FAILED
    assert work._error is failure
    assert work._handle is None
    work.close()
    assert not provider.backend._works


def test_pending_failure_release_interrupt_is_poll_retryable(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    handle = work._handle
    handle.release_errors = [KeyboardInterrupt()]
    provider.agent.check_error = RuntimeError("status failed")

    with pytest.raises(KeyboardInterrupt):
        _ = work.state
    assert work._state is transfer.WorkState.RUNNING
    assert work._pending_error is not None
    assert work._submission_ready
    assert not handle.released

    assert work.state is transfer.WorkState.FAILED
    assert handle.released
    assert work._handle is None
    work.close()
    provider.backend.release_plan(plan)


def test_keyboard_interrupt_after_adoption_is_reported_through_work(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.transfer_error = KeyboardInterrupt()

    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert isinstance(work.error.__cause__, KeyboardInterrupt)
    work.close()
    provider.backend.release_plan(plan)


def test_nixl_rejects_different_per_block_layouts(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, nbytes=16)],
        [
            _region(provider.imported, remote=True, offset=0, nbytes=8),
            _region(provider.imported, remote=True, offset=8, nbytes=8),
        ],
        indexed=False,
    )
    with pytest.raises(transfer.UnsupportedError, match="per-block layout"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
    assert not provider.agent.make_calls


def test_empty_attached_notification_is_not_silently_dropped(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    with pytest.raises(transfer.UnsupportedError, match="empty notification"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=b"",
        )
    assert not provider.agent.make_calls


def test_standalone_notifications_and_source_identity(provider):
    work = _send_notification(provider.backend, provider.peer, b"hello")
    assert work.state is transfer.WorkState.COMPLETED
    assert provider.agent.sent_notifications == [(_REMOTE_NATIVE_NAME, b"hello")]
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"reply-1", b"reply-2"]}
    batches = provider.backend.poll_notification_batches(max_items=1)
    assert batches == [
        transfer.BackendNotificationBatch(
            payloads=(b"reply-1",),
            source_endpoint_id="remote-id",
            source_incarnation="remote-incarnation",
        )
    ]
    assert provider.backend.poll_notifications() == [
        transfer.BackendNotification(
            payload=b"reply-2",
            source_endpoint_id="remote-id",
            source_incarnation="remote-incarnation",
        )
    ]
    work.close()


def test_prepared_native_notification_sender_is_cached_and_shared(
    provider, monkeypatch
):
    success = object()

    class Sender:
        def __init__(self):
            self.payloads = []

        def send(self, payload):
            self.payloads.append(payload)
            return success

    class RawAgent:
        def __init__(self):
            self.factory_calls = []

        def postXferReq(self, handle):
            del handle
            return success

        def getXferStatus(self, handle):
            del handle
            return success

        def createNotifSender(self, remote_name, handles):
            sender = Sender()
            self.factory_calls.append((remote_name, handles, sender))
            return sender

    class Agent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = RawAgent()
            self.backends = {"UCX": 0xCA11}

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = RawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = object()
    fake_bindings.NIXL_WRITE = object()
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = Agent()
    backend = NixlBackend(
        "prepared-sender-id",
        "prepared-sender",
        "prepared-sender-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    first = _import_peer(backend, provider.metadata)
    second = _import_peer(backend, provider.metadata)
    assert len(agent.agent.factory_calls) == 1
    remote_name, handles, sender = agent.agent.factory_calls[0]
    assert remote_name == _REMOTE_NATIVE_NAME
    assert handles is backend._raw_transfer_backend_handles
    assert first.notification_sender is sender
    assert second.notification_sender is sender

    sync_payload = b"sync-exact"
    assert backend.send_notification_sync(first, sync_payload) is None
    work_payload = b"work-exact"
    work = _send_notification(backend, second, work_payload)
    assert sender.payloads[0] is sync_payload
    assert sender.payloads[1] is work_payload
    assert agent.sent_notifications == []
    work.close()

    state = backend._peer_states[("remote-id", "remote-incarnation")]
    backend.release_peer(first)
    assert first.notification_sender is None
    assert state.notification_sender is sender
    agent.remove_error = RuntimeError("release temporarily failed")
    with pytest.raises(transfer.BackendFailureError, match="failed to release"):
        backend.release_peer(second)
    assert second.notification_sender is None
    assert state.notification_sender is None
    assert state.notification_send is None
    backend.send_notification_sync(second, b"safe-high-level-fallback")
    assert agent.sent_notifications == [
        (_REMOTE_NATIVE_NAME, b"safe-high-level-fallback")
    ]
    agent.remove_error = None
    backend.release_peer(second)
    backend.close()


def test_grouped_notifications_preserve_native_sources_without_scalar_wrappers(
    provider, monkeypatch
):
    provider.agent.notifications = {
        _REMOTE_NATIVE_NAME: [b"known-1", b"known-2"],
        "unknown-native-a": [b"unknown-a"],
        "unknown-native-b": [b"unknown-b-1", b"unknown-b-2"],
    }

    def reject_scalar_wrapper(*args, **kwargs):
        del args, kwargs
        raise AssertionError("grouped receive must not create scalar wrappers")

    monkeypatch.setattr(transfer, "BackendNotification", reject_scalar_wrapper)
    batches = provider.backend.poll_notification_batches()

    assert [batch.payloads for batch in batches] == [
        (b"known-1", b"known-2"),
        (b"unknown-a",),
        (b"unknown-b-1", b"unknown-b-2"),
    ]
    assert (
        batches[0].source_endpoint_id,
        batches[0].source_incarnation,
    ) == ("remote-id", "remote-incarnation")
    assert batches[1].source is None
    assert batches[2].source is None


def test_prepared_raw_grouped_notification_route_reuses_final_payload_tuple(provider):
    payloads = (b"prepared-1", b"prepared-2")
    calls = 0

    class Receiver:
        def poll(self):
            nonlocal calls
            calls += 1
            return {_REMOTE_NATIVE_NAME: payloads}

    def reject_legacy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("prepared grouped receive must not use a legacy route")

    receiver = Receiver()
    provider.backend._raw_notification_receiver = receiver
    provider.backend._raw_poll_notification_batches = receiver.poll
    provider.backend._raw_get_notification_batches = reject_legacy
    provider.backend._raw_get_notifications = reject_legacy
    provider.agent.get_new_notifs = reject_legacy

    batches = provider.backend.poll_notification_batches()

    assert calls == 1
    assert len(batches) == 1
    assert batches[0].payloads is payloads
    assert batches[0].source == transfer.EndpointIdentity(
        endpoint_id="remote-id", incarnation="remote-incarnation"
    )
    provider.backend.close()
    assert provider.backend._raw_poll_notification_batches is None
    assert provider.backend._raw_notification_receiver is None
    assert provider.backend._agent is None
    assert provider.backend._closed


def test_prepared_bounded_notification_route_limits_native_materialization(monkeypatch):
    success = object()
    payloads = deque(
        [
            {"bounded-source": (b"one",)},
            {"bounded-source": (b"two",)},
        ]
    )

    class Receiver:
        def __init__(self):
            self.bounded_calls = []
            self.unbounded_calls = 0

        def poll(self):
            self.unbounded_calls += 1
            raise AssertionError("bounded provider route used unbounded poll")

        def poll_bounded(self, **limits):
            self.bounded_calls.append(limits)
            return payloads.popleft() if payloads else {}

    class RawAgent:
        def __init__(self):
            self.receiver = Receiver()

        def postXferReq(self, handle):
            del handle
            return success

        def getXferStatus(self, handle):
            del handle
            return success

        def createNotifReceiver(self, handles):
            assert handles == (0xCA11,)
            return self.receiver

    class Agent(_FakeAgent):
        def __init__(self):
            super().__init__(remote_name="bounded-source")
            self.agent = RawAgent()
            self.backends = {"UCX": 0xCA11}

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = RawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = object()
    fake_bindings.NIXL_WRITE = object()
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = Agent()
    backend = NixlBackend(
        "bounded-id",
        "bounded",
        "bounded-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {
            "_agent": agent,
            "transfer_backends": ["UCX"],
            "max_pending_notifications": 3,
            "max_pending_notification_bytes": 16,
        },
    )

    first = backend.poll_notification_batches(max_items=1)
    second = backend.poll_notification_batches(max_items=1)
    assert [batch.payloads for batch in first + second] == [(b"one",), (b"two",)]
    assert agent.agent.receiver.unbounded_calls == 0
    assert agent.agent.receiver.bounded_calls == [
        {
            "max_items": 1,
            "max_batch_items": 3,
            "max_batch_bytes": 16,
            "max_payload_bytes": transfer.BackendNotificationBatch.MAX_PAYLOAD_BYTES,
        },
        {
            "max_items": 1,
            "max_batch_items": 3,
            "max_batch_bytes": 16,
            "max_payload_bytes": transfer.BackendNotificationBatch.MAX_PAYLOAD_BYTES,
        },
    ]
    backend.close()


def test_one_shot_raw_grouped_notification_route_preserves_backend_handles(provider):
    payloads = (b"one-shot",)
    calls = []

    def get_grouped(handles):
        calls.append(handles)
        return {_REMOTE_NATIVE_NAME: payloads}

    def reject_legacy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("one-shot grouped receive must not retry a legacy route")

    handles = (17, 23)
    provider.backend._raw_transfer_backend_handles = handles
    provider.backend._raw_notification_receiver = None
    provider.backend._raw_poll_notification_batches = None
    provider.backend._raw_get_notification_batches = get_grouped
    provider.backend._raw_get_notifications = reject_legacy
    provider.agent.get_new_notifs = reject_legacy

    batches = provider.backend.poll_notification_batches()

    assert calls == [handles]
    assert batches[0].payloads is payloads


def test_raw_grouped_receiver_is_prepared_once_and_close_is_retryable(monkeypatch):
    success = object()
    native_read = object()
    native_write = object()
    payloads = (b"prepared-once",)

    class Receiver:
        def __init__(self):
            self.poll_calls = 0

        def poll(self):
            self.poll_calls += 1
            if self.poll_calls == 1:
                return {"prepared-native-source": payloads}
            if self.poll_calls == 2:
                return {}
            raise RuntimeError("prepared drain became ambiguous")

    class RawAgent:
        def __init__(self):
            self.factory_calls = []
            self.receiver = Receiver()
            self.lower_route_calls = 0

        def postXferReq(self, handle):
            del handle
            return success

        def getXferStatus(self, handle):
            del handle
            return success

        def createNotifReceiver(self, handles):
            self.factory_calls.append(handles)
            return self.receiver

        def getNotifsGrouped(self, handles):
            del handles
            self.lower_route_calls += 1
            raise AssertionError("prepared receive must not use one-shot fallback")

        def getNotifs(self, accumulator, handles):
            del accumulator, handles
            self.lower_route_calls += 1
            raise AssertionError("prepared receive must not use legacy fallback")

    class Agent(_FakeAgent):
        def __init__(self):
            super().__init__(remote_name="prepared-native-source")
            self.agent = RawAgent()
            self.backends = {"UCX": 0xCA11}

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = RawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = object()
    fake_bindings.NIXL_READ = native_read
    fake_bindings.NIXL_WRITE = native_write
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    agent = Agent()
    backend = NixlBackend(
        "prepared-id",
        "prepared",
        "prepared-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.SERIALIZED,
        {
            "_agent": agent,
            "transfer_backends": ["UCX"],
            "use_native_request_slot_execution": False,
        },
    )
    cached_handles = backend._raw_transfer_backend_handles
    assert cached_handles == (0xCA11,)
    assert agent.agent.factory_calls == [cached_handles]
    assert agent.agent.factory_calls[0] is cached_handles

    # Later high-level mutations cannot reprepare or redirect this endpoint.
    agent.backends["UCX"] = 0xBAD
    first = backend.poll_notification_batches()
    second = backend.poll_notification_batches()
    assert first[0].payloads is payloads
    assert second == []
    assert agent.agent.factory_calls == [cached_handles]
    assert agent.agent.receiver.poll_calls == 2
    assert agent.agent.lower_route_calls == 0

    with pytest.raises(transfer.BackendFailureError) as first_failure:
        backend.poll_notification_batches()
    poison = first_failure.value
    assert isinstance(poison.__cause__, RuntimeError)
    assert "prepared drain became ambiguous" in str(poison.__cause__)
    with pytest.raises(transfer.BackendFailureError) as repeated_failure:
        backend.poll_notification_batches()
    assert repeated_failure.value is poison
    assert agent.agent.receiver.poll_calls == 3
    assert agent.agent.lower_route_calls == 0

    original_setattr = NixlBackend.__setattr__
    interrupted = False

    def interrupt_receiver_drop(self, name, value):
        nonlocal interrupted
        original_setattr(self, name, value)
        if (
            self is backend
            and name == "_raw_notification_receiver"
            and value is None
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("during receiver teardown")

    with monkeypatch.context() as close_patch:
        close_patch.setattr(NixlBackend, "__setattr__", interrupt_receiver_drop)
        with pytest.raises(KeyboardInterrupt, match="during receiver teardown"):
            backend.close()
    assert not backend._closed
    assert not backend._available
    assert backend._closing
    assert backend._agent is agent
    assert backend._raw_notification_receiver is None

    backend.close()
    assert backend._agent is None
    assert backend._closed


def test_notification_cursor_interleaves_batch_and_scalar_without_refetch(provider):
    native_polls = 0
    original_poll = provider.agent.get_new_notifs

    def counted_poll(*, backends):
        nonlocal native_polls
        native_polls += 1
        return original_poll(backends=backends)

    provider.agent.get_new_notifs = counted_poll
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"a", b"b", b"c"]}

    assert provider.backend.poll_notification_batches(max_items=0) == []
    assert native_polls == 0
    assert [
        batch.payloads
        for batch in provider.backend.poll_notification_batches(max_items=1)
    ] == [(b"a",)]
    assert native_polls == 1
    assert provider.backend._pending_notification_available_count == 2
    assert provider.backend._pending_notification_retained_count == 3
    assert provider.backend._pending_notification_bytes == 3

    provider.agent.notifications = {"later-native-source": [b"d"]}
    provider.backend.progress()
    assert native_polls == 1
    assert [
        notification.payload
        for notification in provider.backend.poll_notifications(max_items=1)
    ] == [b"b"]
    assert native_polls == 1
    assert provider.backend._pending_notification_available_count == 1
    assert provider.backend._pending_notification_retained_count == 3
    assert provider.backend._pending_notification_bytes == 3

    assert [
        batch.payloads
        for batch in provider.backend.poll_notification_batches(max_items=8)
    ] == [(b"c",)]
    assert native_polls == 1
    assert not provider.backend._pending_notification_batches
    assert provider.backend._pending_notification_available_count == 0
    assert provider.backend._pending_notification_retained_count == 0
    assert provider.backend._pending_notification_bytes == 0

    later = provider.backend.poll_notification_batches()
    assert native_polls == 2
    assert [batch.payloads for batch in later] == [(b"d",)]
    assert later[0].source is None


def test_notification_cursor_interruption_poisons_inconsistent_egress(provider):
    class InterruptingDeque(deque):
        def popleft(self):
            result = super().popleft()
            raise KeyboardInterrupt("after dequeue commit")

    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"interrupted"]}
    provider.backend.progress()
    provider.backend._pending_notification_batches = InterruptingDeque(
        provider.backend._pending_notification_batches
    )

    with pytest.raises(KeyboardInterrupt, match="after dequeue commit"):
        provider.backend.poll_notification_batches()
    poison = provider.backend._notification_error
    assert isinstance(poison, transfer.BackendFailureError)
    with pytest.raises(transfer.BackendFailureError) as repeated:
        provider.backend.poll_notification_batches()
    assert repeated.value is poison


def test_notification_validation_interruption_is_visible_and_poisons(provider):
    class InterruptingMessage(bytearray):
        def __bytes__(self):
            raise KeyboardInterrupt("during payload snapshot")

    provider.agent.notifications = {
        _REMOTE_NATIVE_NAME: [InterruptingMessage(b"interrupted")]
    }

    with pytest.raises(KeyboardInterrupt, match="during payload snapshot"):
        provider.backend.poll_notification_batches()
    poison = provider.backend._notification_error
    assert isinstance(poison, transfer.BackendFailureError)
    with pytest.raises(transfer.BackendFailureError) as repeated:
        provider.backend.poll_notification_batches()
    assert repeated.value is poison


def test_multiple_notification_polls_deliver_each_payload_once():
    payloads = tuple(f"message-{index}".encode() for index in range(16))
    agent = _FakeAgent(remote_name="concurrent-native-source")
    agent.notifications = {"concurrent-native-source": list(payloads)}
    backend = NixlBackend(
        "concurrent-id",
        "concurrent",
        "concurrent-incarnation",
        transfer.ProgressMode.MANUAL,
        transfer.ThreadMode.MULTIPLE,
        {"_agent": agent, "transfer_backends": ["UCX"]},
    )
    barrier = threading.Barrier(len(payloads))
    received = [None] * len(payloads)
    errors = []

    def poll(index):
        try:
            barrier.wait()
            notifications = backend.poll_notifications(max_items=1)
            assert len(notifications) == 1
            received[index] = notifications[0].payload
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=poll, args=(index,)) for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(received) == sorted(payloads)
    assert len(set(received)) == len(payloads)
    backend.close()


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_synchronous_notification_sends_overlap_without_provider_lock(
    provider,
):
    entered_first = threading.Event()
    entered_both = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    entered = 0
    errors = []
    original_send = provider.agent.send_notif

    def blocking_send(remote_name, payload):
        nonlocal entered
        with count_lock:
            entered += 1
            entered_first.set()
            if entered == 2:
                entered_both.set()
        if not release.wait(2):
            raise RuntimeError("timed out waiting to release concurrent sends")
        original_send(remote_name, payload)

    provider.agent.send_notif = blocking_send

    def send(payload):
        try:
            provider.backend.send_notification_sync(provider.peer, payload)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send, args=(payload,))
        for payload in (b"first", b"second")
    ]
    try:
        for thread in threads:
            thread.start()
        assert entered_first.wait(1)
        assert entered_both.wait(1), "MULTIPLE provider calls were serialized"
    finally:
        release.set()
        for thread in threads:
            thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(provider.agent.sent_notifications) == [
        (_REMOTE_NATIVE_NAME, b"first"),
        (_REMOTE_NATIVE_NAME, b"second"),
    ]


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_asynchronous_notification_sends_overlap_and_block_close(provider):
    entered_first = threading.Event()
    entered_both = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    entered = 0
    adopted = []
    errors = []
    original_send = provider.agent.send_notif

    def blocking_send(remote_name, payload):
        nonlocal entered
        with count_lock:
            entered += 1
            entered_first.set()
            if entered == 2:
                entered_both.set()
        if not release.wait(2):
            raise RuntimeError("timed out waiting to release asynchronous sends")
        original_send(remote_name, payload)

    provider.agent.send_notif = blocking_send

    def send(payload):
        try:
            provider.backend.send_notification(
                provider.peer, payload, adopt_work=adopted.append
            )
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send, args=(payload,))
        for payload in (b"first", b"second")
    ]
    try:
        for thread in threads:
            thread.start()
        assert entered_first.wait(1)
        assert entered_both.wait(1), "MULTIPLE asynchronous sends were serialized"
        with pytest.raises(transfer.BusyError, match="active transfer"):
            provider.backend.close()
    finally:
        release.set()
        for thread in threads:
            thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(adopted) == 2
    assert all(work.state is transfer.WorkState.COMPLETED for work in adopted)
    for work in adopted:
        work.close()


@pytest.mark.parametrize("provider", [transfer.ThreadMode.MULTIPLE], indirect=True)
def test_multiple_progress_and_poll_share_consumer_lock_and_pin_close(provider):
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    calls = 0
    errors = []

    def blocking_collect(max_items=None):
        del max_items
        nonlocal calls
        with count_lock:
            calls += 1
            call = calls
            if call == 1:
                first_entered.set()
            else:
                second_entered.set()
        if call == 1 and not release.wait(2):
            raise RuntimeError("timed out waiting to release notification drain")

    provider.backend._collect_notifications_locked = blocking_collect

    def run(operation):
        try:
            operation()
        except BaseException as error:
            errors.append(error)

    progress_thread = threading.Thread(target=run, args=(provider.backend.progress,))
    poll_thread = threading.Thread(
        target=run, args=(provider.backend.poll_notification_batches,)
    )
    progress_thread.start()
    assert first_entered.wait(1)
    poll_thread.start()
    assert not second_entered.wait(0.1)
    with pytest.raises(transfer.BusyError, match="active provider calls"):
        provider.backend.close()
    assert not provider.backend._closed

    release.set()
    progress_thread.join(2)
    poll_thread.join(2)
    assert all(not thread.is_alive() for thread in (progress_thread, poll_thread))
    assert second_entered.is_set()
    assert errors == []
    (warmed_reservation,) = provider.backend._backend_call_reservation_pool
    provider.backend.progress()
    assert provider.backend._backend_call_reservation_pool == [warmed_reservation]


def test_standalone_notification_adopts_before_send_and_reports_signal(provider):
    adopted = []

    def adopt_work(work):
        provider.agent.actions.append("adopt-notification")
        adopted.append(work)

    result = provider.backend.send_notification(
        provider.peer,
        b"first",
        adopt_work=adopt_work,
    )
    assert result is None
    assert provider.agent.actions == ["adopt-notification", "send"]
    assert adopted[0].state is transfer.WorkState.COMPLETED
    adopted[0].close()

    provider.agent.actions.clear()
    provider.agent.send_error = KeyboardInterrupt()
    failed = _send_notification(provider.backend, provider.peer, b"second")
    assert provider.agent.actions == ["send"]
    assert failed.state is transfer.WorkState.FAILED
    assert isinstance(failed.error, transfer.BackendFailureError)
    assert isinstance(failed.error.__cause__, KeyboardInterrupt)
    failed.close()


@pytest.mark.parametrize(
    "provider",
    [
        transfer.ThreadMode.SINGLE,
        transfer.ThreadMode.CALLER_SERIALIZED,
        transfer.ThreadMode.SERIALIZED,
    ],
    indirect=True,
)
def test_nonmultiple_notification_hot_paths_never_enter_noop_lock(
    provider, monkeypatch
):
    provider_module = sys.modules[NixlBackend.__module__]

    def reject_noop_context(self):
        del self
        raise AssertionError("non-MULTIPLE notification path entered _NoopLock")

    monkeypatch.setattr(provider_module._NoopLock, "__enter__", reject_noop_context)
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"grouped", b"scalar"]}

    work = _send_notification(provider.backend, provider.peer, b"standalone")
    provider.backend.send_notification_sync(provider.peer, b"standalone-sync")
    assert [
        batch.payloads
        for batch in provider.backend.poll_notification_batches(max_items=1)
    ] == [(b"grouped",)]
    assert [
        notification.payload
        for notification in provider.backend.poll_notifications(max_items=1)
    ] == [b"scalar"]
    assert provider.agent.sent_notifications == [
        (_REMOTE_NATIVE_NAME, b"standalone"),
        (_REMOTE_NATIVE_NAME, b"standalone-sync"),
    ]
    work.close()


def test_standalone_notification_failure_sentinels_precede_adoption(
    provider, monkeypatch
):
    provider_module = sys.modules[NixlBackend.__module__]
    original_failure = transfer.BackendFailureError
    allocations = []

    class CountingFailure(original_failure):
        def __init__(self, message):
            allocations.append(message)
            super().__init__(message)

    monkeypatch.setattr(
        provider_module.transfer,
        "BackendFailureError",
        CountingFailure,
    )
    adopted = []

    def adopt_work(work):
        assert allocations == [
            "NIXL failed to send a notification",
            "Core failed to adopt a NIXL notification request",
        ]
        adopted.append(work)

    result = provider.backend.send_notification(
        provider.peer,
        b"preallocated",
        adopt_work=adopt_work,
    )
    assert result is None
    assert len(adopted) == 1
    assert adopted[0].state is transfer.WorkState.COMPLETED
    adopted[0].close()


def test_standalone_notification_does_not_send_if_adoption_fails(provider):
    def reject_work(work):
        del work
        raise KeyboardInterrupt()

    result = provider.backend.send_notification(
        provider.peer,
        b"never-sent",
        adopt_work=reject_work,
    )

    assert result is None
    assert provider.agent.actions == []
    assert not provider.backend._works


def test_notification_registry_publication_interrupt_retires_unadopted_work(
    provider,
):
    class InterruptAfterSet(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt()

    provider.backend._works = InterruptAfterSet(provider.backend._works)
    adopted = []

    result = provider.backend.send_notification(
        provider.peer,
        b"never-sent",
        adopt_work=adopted.append,
    )

    assert result is None
    assert not adopted
    assert provider.agent.actions == []
    assert not provider.backend._works


def test_notification_completion_publication_interrupt_becomes_failure(
    provider, monkeypatch
):
    provider_module = sys.modules[NixlBackend.__module__]
    finish = provider_module.NixlWork._finish_synchronous
    interrupted = False

    def interrupt_once(work, *, error=None, cause=None):
        nonlocal interrupted
        if error is None and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return finish(work, error=error, cause=cause)

    monkeypatch.setattr(provider_module.NixlWork, "_finish_synchronous", interrupt_once)
    work = _send_notification(provider.backend, provider.peer, b"sent-ambiguously")

    assert provider.agent.actions == ["send"]
    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert isinstance(work.error.__cause__, KeyboardInterrupt)
    work.close()


def test_manual_progress_drives_idle_agent_and_preserves_notifications(provider):
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"progress-reply"]}

    provider.backend.progress()

    assert provider.agent.notifications == {}
    assert provider.backend.poll_notifications() == [
        transfer.BackendNotification(
            payload=b"progress-reply",
            source_endpoint_id="remote-id",
            source_incarnation="remote-incarnation",
        )
    ]


def test_notification_backlog_is_bounded_and_close_remains_possible(provider):
    provider.backend._max_pending_notifications = 2
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"first", b"second"]}

    assert [
        notification.payload for notification in provider.backend.poll_notifications()
    ] == [b"first", b"second"]
    assert not provider.backend._pending_notification_batches
    assert provider.backend._pending_notification_available_count == 0
    assert provider.backend._pending_notification_retained_count == 0
    assert provider.backend._pending_notification_bytes == 0

    provider.agent.notifications = {
        _REMOTE_NATIVE_NAME: [b"third", b"fourth", b"overflow"]
    }

    with pytest.raises(transfer.BackendFailureError, match="backlog"):
        provider.backend.poll_notifications()
    assert not provider.backend._pending_notification_batches
    assert provider.backend._pending_notification_available_count == 0
    assert provider.backend._pending_notification_retained_count == 0
    assert provider.backend._pending_notification_bytes == 0

    # Progress is receive protocol work and fails closed too; close bypasses
    # that gate and can still execute the normal cleanup sequence.
    with pytest.raises(transfer.BackendFailureError):
        provider.backend.progress()
    provider.backend.release_peer(provider.peer)
    provider.backend.deregister(provider.registration)
    provider.backend.close()


@pytest.mark.parametrize("custom_raw_mapping", [False, True])
def test_compat_notification_byte_limit_precedes_mutable_snapshot(
    provider, custom_raw_mapping
):
    snapshots = 0

    class SnapshotTrackedBytearray(bytearray):
        def __bytes__(self):
            nonlocal snapshots
            snapshots += 1
            return bytes(bytearray(self))

    provider.backend._max_pending_notification_bytes = 5
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"abc"]}
    provider.backend.progress()
    assert provider.backend._pending_notification_bytes == 3

    oversized_for_remaining = SnapshotTrackedBytearray(b"def")
    if custom_raw_mapping:

        class CustomNotifications(dict):
            pass

        provider.backend._raw_poll_notification_batches = lambda: CustomNotifications(
            {_REMOTE_NATIVE_NAME: (oversized_for_remaining,)}
        )
    else:
        provider.agent.notifications = {_REMOTE_NATIVE_NAME: [oversized_for_remaining]}
    with pytest.raises(transfer.BackendFailureError, match="backlog") as failed:
        provider.backend.progress()

    assert isinstance(failed.value.__cause__, ValueError)
    assert str(failed.value.__cause__) == "notification byte limit exceeded"
    assert snapshots == 0


def test_malformed_drained_notification_poisons_new_operations(provider):
    class MalformedMessage:
        def __len__(self):
            return 1

        def __bytes__(self):
            raise ValueError("cannot convert message")

    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["DONE"]
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    provider.agent.notifications = {
        _REMOTE_NATIVE_NAME: [MalformedMessage(), b"silently-lost-if-resumed"]
    }

    with pytest.raises(transfer.BackendFailureError, match="malformed") as caught:
        provider.backend.poll_notifications()
    poison = caught.value
    assert provider.agent.notifications == {}
    assert not provider.backend._pending_notification_batches

    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"must-not-resume"]}
    with pytest.raises(transfer.BackendFailureError) as repeated:
        provider.backend.poll_notifications()
    assert repeated.value is poison
    assert provider.agent.notifications
    with pytest.raises(transfer.BackendFailureError):
        _prepare(
            provider.backend,
            [_region(provider.registration)],
            [_region(provider.imported, remote=True)],
            indexed=False,
        )
    with pytest.raises(transfer.BackendFailureError):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=None,
            remote_indices=None,
            notification=None,
        )
    with pytest.raises(transfer.BackendFailureError):
        _send_notification(provider.backend, provider.peer, b"rejected")
    with pytest.raises(transfer.BackendFailureError):
        _import_peer(provider.backend, provider.metadata)
    rejected_memory = transfer.RegisteredMemory(
        registration_id="poisoned-registration",
        name="poisoned-buffer",
        owner=bytearray(1),
        address=0x50_0000_5000,
        nbytes=1,
        device="cpu",
        memory_type="DRAM",
    )
    with pytest.raises(transfer.BackendFailureError):
        _register(provider.backend, rejected_memory)

    # Existing request progression and every release path remain usable.
    assert work.state is transfer.WorkState.COMPLETED
    work.close()
    provider.backend.release_plan(plan)
    with pytest.raises(transfer.BackendFailureError) as progress_failure:
        provider.backend.progress()
    assert progress_failure.value is poison
    provider.backend.release_peer(provider.peer)
    provider.backend.deregister(provider.registration)
    provider.backend.close()


def test_drained_notification_binding_signal_persists_poison(provider):
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"drained"]}
    provider.agent.notification_error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        provider.backend.poll_notifications()

    poison = provider.backend._notification_error
    assert isinstance(poison, transfer.BackendFailureError)
    assert isinstance(poison.__cause__, KeyboardInterrupt)
    assert provider.agent.notifications == {}

    provider.agent.notification_error = None
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"must-not-resume"]}
    with pytest.raises(transfer.BackendFailureError) as repeated:
        provider.backend.poll_notifications()
    assert repeated.value is poison
    assert provider.agent.notifications


def test_notification_second_cut_uses_prepublished_fail_stop(provider, monkeypatch):
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"destructively-drained"]}
    provider.agent.notification_error = RuntimeError("native return failed after drain")

    def interrupt_poison_publication(cause):
        assert isinstance(cause, RuntimeError)
        assert provider.backend._notification_drain_dirty
        assert provider.backend._notification_error is None
        raise KeyboardInterrupt("second cut during poison publication")

    monkeypatch.setattr(
        provider.backend,
        "_poison_notifications_locked",
        interrupt_poison_publication,
    )
    with pytest.raises(KeyboardInterrupt, match="second cut"):
        provider.backend.poll_notifications()

    assert provider.agent.notifications == {}
    assert provider.backend._notification_drain_dirty
    assert provider.backend._notification_error is None
    fallback = provider.backend._notification_drain_error

    provider.agent.notification_error = None
    provider.agent.notifications = {_REMOTE_NATIVE_NAME: [b"must-not-resume"]}
    with pytest.raises(transfer.BackendFailureError) as repeated:
        provider.backend.poll_notifications()
    assert repeated.value is fallback
    assert provider.agent.notifications
    with pytest.raises(transfer.BackendFailureError) as progress_failure:
        provider.backend.progress()
    assert progress_failure.value is fallback


def test_peer_adoption_precedes_reference_mutation(provider):
    key = ("remote-id", "remote-incarnation")
    state = provider.backend._peer_states[key]
    references_before = state.references
    peers_before = provider.backend._peers.copy()
    imports_before = provider.agent.imported_metadata.copy()

    def reject(peer):
        assert peer in provider.backend._peers
        assert state.references == references_before
        assert peer not in state.live_peers
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        provider.backend.import_peer(provider.metadata, adopt_handle=reject)

    assert provider.backend._peers == peers_before
    assert state.references == references_before
    assert provider.agent.imported_metadata == imports_before


def test_failed_peer_import_retains_cleanup_until_retry(provider):
    provider.backend.release_peer(provider.peer)
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="retry-id",
        name="retry",
        incarnation="retry-incarnation",
        backend_payload=b"retry-metadata",
        registrations=(),
    )
    retry_name = _native_name("retry-id", "retry-incarnation")
    provider.agent.remote_names_by_metadata[b"retry-metadata"] = retry_name
    provider.agent.import_error = RuntimeError("load failed after mutation")
    provider.agent.remove_error = RuntimeError("rollback temporarily failed")

    adopted = []
    with pytest.raises(transfer.BackendFailureError, match="failed to import"):
        provider.backend.import_peer(metadata, adopt_handle=adopted.append)
    assert len(adopted) == 1
    retained = adopted[0]
    assert retained in provider.backend._peers
    assert ("retry-id", "retry-incarnation") in provider.backend._peer_states
    with pytest.raises(transfer.BackendFailureError, match="failed to release"):
        provider.backend.release_peer(retained)

    provider.agent.remove_error = None
    provider.agent.import_error = None
    provider.backend.release_peer(retained)
    assert retained not in provider.backend._peers
    assert ("retry-id", "retry-incarnation") not in provider.backend._peer_states
    assert retry_name not in provider.backend._peer_keys_by_name


def test_peer_import_keyboard_interrupt_rolls_back_provisional_state(provider):
    provider.backend.release_peer(provider.peer)
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="interrupt-id",
        name="interrupt",
        incarnation="interrupt-incarnation",
        backend_payload=b"interrupt-metadata",
        registrations=(),
    )
    remote_name = _native_name("interrupt-id", "interrupt-incarnation")
    provider.agent.remote_names_by_metadata[b"interrupt-metadata"] = remote_name
    provider.agent.import_error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _import_peer(provider.backend, metadata)

    assert (
        "interrupt-id",
        "interrupt-incarnation",
    ) not in provider.backend._peer_states
    assert remote_name not in provider.backend._peer_keys_by_name
    assert remote_name in provider.agent.removed_peers
    provider.agent.import_error = None


def test_remote_raw_span_keeps_peer_identity(provider):
    raw = transfer.ImportedRawMemory(
        peer_handle=provider.peer,
        address=_RAW_ADDRESS,
        nbytes=64,
        device="cpu",
        memory_type="host",
    )
    remote = transfer.BackendRegion(
        owner=object(),
        offset=0,
        nbytes=16,
        device="cpu",
        memory_type="host",
        backend_descriptor=raw,
        remote=True,
    )
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, nbytes=16)],
        [remote],
        indexed=False,
    )
    assert provider.agent.prep_calls[1][0] == _REMOTE_NATIVE_NAME
    assert provider.agent.prep_calls[1][1] == [(_RAW_ADDRESS, 16, 0)]
    provider.backend.release_plan(plan)


def test_remote_raw_span_must_be_covered_by_imported_grant(provider):
    raw = transfer.ImportedRawMemory(
        peer_handle=provider.peer,
        address=_RAW_ADDRESS + 64,
        nbytes=1,
        device="cpu",
        memory_type="host",
    )
    remote = transfer.BackendRegion(
        owner=object(),
        offset=0,
        nbytes=1,
        device="cpu",
        memory_type="host",
        backend_descriptor=raw,
        remote=True,
    )
    with pytest.raises(transfer.InvalidRegionError, match="imported NIXL grant"):
        _prepare(
            provider.backend,
            [_region(provider.registration, nbytes=1)],
            [remote],
            indexed=False,
        )


def test_nixl_rejects_aliasing_remote_destination_registrations(provider):
    alias = transfer.ImportedRegistration(
        peer_handle=provider.peer,
        registration_id="remote-alias",
        name="remote-alias",
        nbytes=provider.imported.nbytes,
        device=provider.imported.device,
        memory_type=provider.imported.memory_type,
        descriptor=provider.imported.descriptor,
    )
    plan = _prepare(
        provider.backend,
        [
            _region(provider.registration, offset=0),
            _region(provider.registration, offset=16),
        ],
        [_region(provider.imported, remote=True), _region(alias, remote=True)],
        indexed=True,
    )

    selected = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=[0],
        remote_indices=[1],
        notification=None,
    )
    assert selected.result == 16
    selected.close()

    with pytest.raises(transfer.InvalidRegionError, match="overlapping destination"):
        _submit(
            provider.backend,
            transfer.TransferOp.WRITE,
            plan,
            local_indices=[0, 1],
            remote_indices=[0, 1],
            notification=None,
        )

    provider.backend.release_plan(plan)


def test_peer_import_is_reference_counted(provider):
    second = _import_peer(provider.backend, provider.metadata)
    assert provider.agent.imported_metadata == [b"nixl-remote-metadata"]
    provider.backend.release_peer(provider.peer)
    assert provider.agent.removed_peers == []
    provider.backend.release_peer(second)
    assert provider.agent.removed_peers == [_REMOTE_NATIVE_NAME]


def test_duplicate_peer_release_retry_discards_exact_identity_once(provider):
    class InterruptingIdentitySet(set):
        interrupt = True

        def discard(self, value):
            super().discard(value)
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt

    second = _import_peer(provider.backend, provider.metadata)
    key = (provider.peer.endpoint_id, provider.peer.incarnation)
    state = provider.backend._peer_states[key]
    state.live_peers = InterruptingIdentitySet(state.live_peers)

    with pytest.raises(KeyboardInterrupt):
        provider.backend.release_peer(provider.peer)
    assert not provider.peer.released
    assert state.references == 1
    assert second in state.live_peers

    # Retrying the same release is an idempotent identity discard: it cannot
    # decrement the surviving handle or tear down its shared native sender.
    provider.backend.release_peer(provider.peer)
    assert state.references == 1
    assert provider.agent.removed_peers == []
    provider.backend.release_peer(second)
    assert provider.agent.removed_peers == [_REMOTE_NATIVE_NAME]


def test_live_peer_grants_are_immutable(provider):
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="remote-id",
        name="remote",
        incarnation="remote-incarnation",
        backend_payload=b"changed-grant-payload",
        registrations=(),
    )

    with pytest.raises(transfer.InvalidMetadataError, match="immutable"):
        _import_peer(provider.backend, metadata)
    assert provider.agent.imported_metadata == [b"nixl-remote-metadata"]


def test_colliding_native_payload_cannot_remove_an_existing_peer(provider):
    malformed = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="different-endpoint",
        name="claimed-other-name",
        incarnation="different-incarnation",
        backend_payload=b"payload-that-resolves-to-existing-remote",
        registrations=(),
    )

    with pytest.raises(transfer.InvalidMetadataError, match="identity"):
        _import_peer(provider.backend, malformed)

    assert provider.agent.removed_peers == []
    assert (
        provider.backend._peer_states[
            (provider.peer.endpoint_id, provider.peer.incarnation)
        ].references
        == 1
    )


def test_display_names_do_not_need_to_be_globally_unique(provider):
    expected_name = _native_name("second-endpoint", "second-incarnation")
    provider.agent.remote_names_by_metadata[b"second-native-payload"] = expected_name
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="second-endpoint",
        name="remote",
        incarnation="second-incarnation",
        backend_payload=b"second-native-payload",
        registrations=(),
    )

    second = _import_peer(provider.backend, metadata)
    assert second.remote_name == expected_name
    provider.backend.release_peer(second)
    assert provider.agent.removed_peers == [expected_name]


def test_unbound_mismatched_payload_is_rejected_before_native_load(provider):
    provider.agent.remote_names_by_metadata[b"mismatch"] = "wrong-native-name"
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="new-endpoint",
        name="new-display-name",
        incarnation="new-incarnation",
        backend_payload=b"mismatch",
        registrations=(),
    )

    with pytest.raises(transfer.InvalidMetadataError, match="identity"):
        _import_peer(provider.backend, metadata)
    assert b"mismatch" not in provider.agent.imported_metadata


def test_close_refuses_active_work_and_cancel_is_unverifiable(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    provider.agent.cancellable = False
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert not work.cancel()
    with pytest.raises(transfer.BusyError, match="active transfer"):
        provider.backend.close()
    assert provider.backend._available
    assert not provider.backend._closing

    work._handle.cancellable = True
    assert not work.cancel()
    assert work.state is transfer.WorkState.RUNNING
    work._handle.statuses = ["DONE"]
    assert work.state is transfer.WorkState.COMPLETED
    work.close()
    provider.backend.release_plan(plan)
    provider.backend.close()


def test_work_bounded_fallback_zero_timeout_and_int64_validation(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"] * 64
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert work._poll_bounded_xfer is None
    assert (
        work.poll_bounded(
            max_polls=64,
            timeout_ns=0,
            timeout_check_interval=32,
        )
        is transfer.WorkState.RUNNING
    )
    assert len(work._handle.statuses) == 63

    for arguments in (
        {"max_polls": 1 << 63, "timeout_ns": None, "timeout_check_interval": 32},
        {"max_polls": 1, "timeout_ns": 1 << 63, "timeout_check_interval": 32},
        {
            "max_polls": 1,
            "timeout_ns": None,
            "timeout_check_interval": 1 << 63,
        },
    ):
        with pytest.raises(ValueError):
            work.poll_bounded(**arguments)

    work._handle.statuses = ["DONE"]
    assert work.state is transfer.WorkState.COMPLETED
    work.close()
    provider.backend.release_plan(plan)


def test_status_query_failure_is_terminal_when_release_succeeds(provider):
    local = [_region(provider.registration)]
    remote = [_region(provider.imported, remote=True)]
    plan = _prepare(provider.backend, local, remote, indexed=False)
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    provider.agent.check_error = RuntimeError("temporary status failure")

    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert isinstance(work.error.__cause__, RuntimeError)
    assert work._handle is None
    work.close()
    provider.backend.release_plan(plan)


def test_status_query_failure_retains_work_until_release_succeeds(provider):
    local = [_region(provider.registration)]
    remote = [_region(provider.imported, remote=True)]
    plan = _prepare(provider.backend, local, remote, indexed=False)
    provider.agent.submit_status = "PROC"
    provider.agent.poll_statuses = ["PROC"]
    provider.agent.cancellable = False
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    provider.agent.check_error = RuntimeError("native status failure")

    assert work.state is transfer.WorkState.RUNNING
    with pytest.raises(transfer.BusyError, match="active NIXL work"):
        work.close()
    work._handle.cancellable = True
    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    work.close()
    provider.backend.release_plan(plan)


def test_transfer_exception_completes_adopted_work_as_failed(provider):
    local = [_region(provider.registration)]
    remote = [_region(provider.imported, remote=True)]
    plan = _prepare(provider.backend, local, remote, indexed=False)
    provider.agent.transfer_error = RuntimeError("native post failure")

    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert provider.agent.make_calls[-1][-1].released
    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert isinstance(work.error.__cause__, RuntimeError)
    assert list(provider.backend._works.values()) == [work]
    work.close()
    provider.backend.release_plan(plan)


def test_transfer_exception_retains_work_until_release_succeeds(provider):
    local = [_region(provider.registration)]
    remote = [_region(provider.imported, remote=True)]
    plan = _prepare(provider.backend, local, remote, indexed=False)
    provider.agent.transfer_error = RuntimeError("native post failure")
    provider.agent.poll_statuses = ["PROC"]
    provider.agent.cancellable = False

    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )

    assert work.state is transfer.WorkState.RUNNING
    work._handle.cancellable = True
    assert work.state is transfer.WorkState.FAILED
    assert isinstance(work.error, transfer.BackendFailureError)
    assert isinstance(work.error.__cause__, RuntimeError)
    work.close()
    provider.backend.release_plan(plan)
    provider.backend.close()


def test_remote_descriptor_bounds_are_revalidated(provider):
    oversized = _region(
        provider.imported,
        remote=True,
        offset=250,
        nbytes=8,
    )
    with pytest.raises(transfer.InvalidRegionError, match="exceeds"):
        _prepare(
            provider.backend,
            [_region(provider.registration, nbytes=8)],
            [oversized],
            indexed=False,
        )


def test_remote_descriptor_rejects_uintptr_overflow_before_native_import(provider):
    maximum = (1 << (np.dtype(np.uintp).itemsize * 8)) - 1
    descriptor = json.dumps(
        {
            "address": maximum,
            "device_id": 0,
            "memory_type": "DRAM",
            "nbytes": 1,
            "version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    metadata = transfer.EndpointMetadata(
        wire_major=transfer.EndpointMetadata.WIRE_MAJOR,
        wire_minor=transfer.EndpointMetadata.WIRE_MINOR,
        required_features=(),
        backend="nixl",
        endpoint_id="overflow-peer",
        name="overflow-peer",
        incarnation="overflow-incarnation",
        backend_payload=b"overflow-metadata",
        registrations=(
            transfer.RegistrationMetadata(
                registration_id="overflow-remote-registration",
                name="overflow-remote-buffer",
                nbytes=1,
                device="cpu",
                memory_type="DRAM",
                backend_descriptor=descriptor,
            ),
        ),
    )
    imports_before = provider.agent.imported_metadata.copy()

    with pytest.raises(transfer.InvalidMetadataError, match="invalid NIXL descriptor"):
        _import_peer(provider.backend, metadata)

    assert provider.agent.imported_metadata == imports_before


def test_strided_extent_overflow_is_rejected_before_native_prepare(provider):
    maximum = (1 << (np.dtype(np.uintp).itemsize * 8)) - 1
    malformed = types.SimpleNamespace(
        remote=False,
        offset=0,
        nbytes=8,
        device="cpu",
        memory_type="DRAM",
        backend_descriptor=provider.registration,
        count=3,
        stride=maximum,
    )
    prep_count = len(provider.agent.prep_calls)

    with pytest.raises(transfer.InvalidRegionError, match="extent"):
        _prepare(
            provider.backend,
            [malformed],
            [_region(provider.imported, remote=True, nbytes=8)],
            indexed=True,
        )

    assert len(provider.agent.prep_calls) == prep_count


def test_mixed_memory_types_are_rejected(provider):
    cuda_memory = transfer.RegisteredMemory(
        registration_id="cuda-registration",
        name="cuda-buffer",
        owner=object(),
        address=0x2000,
        nbytes=64,
        device="cuda:0",
        memory_type="VRAM",
    )
    cuda_registration = _register(provider.backend, cuda_memory)
    with pytest.raises(transfer.UnsupportedError, match="mix memory types"):
        _prepare(
            provider.backend,
            [_region(provider.registration), _region(cuda_registration)],
            [_region(provider.imported, remote=True)],
            indexed=True,
        )


def test_provider_integrates_with_endpoint_contract():
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint("local", backend="nixl", options={"_agent": local_agent})
    remote = transfer.Endpoint(
        "remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region()],
        remote=[peer.region("remote-buffer")],
        indexed=True,
    )
    work = local.write(
        plan,
        local_indices=[0],
        remote_indices=[0],
        notification=b"visible-after-write",
    )
    assert work.wait().state is transfer.WorkState.COMPLETED
    assert local_agent.make_calls[0][0] == "WRITE"
    assert local_agent.make_calls[0][5] == b"visible-after-write"

    work.close()
    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


def test_prevalidated_selection_is_zero_copy_and_reuses_dynamic_notifications(
    monkeypatch,
):
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint(
        "selection-local", backend="nixl", options={"_agent": local_agent}
    )
    remote = transfer.Endpoint(
        "selection-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region(0, 8, stride=8, count=8)],
        remote=[peer.region("remote-buffer", 0, 8, stride=8, count=8)],
        indexed=True,
    )
    local_indices = np.asarray([7, 1, 4], dtype=np.int32)
    remote_indices = np.asarray([0, 5, 2], dtype=np.int32)
    selection = plan.select_indices(
        transfer.TransferOp.WRITE,
        local_indices=local_indices,
        remote_indices=remote_indices,
    )
    local_indices[:] = 0
    remote_indices[:] = 1

    def unexpected(*_args, **_kwargs):
        raise AssertionError("uniform non-overlapping selection took slow validation")

    monkeypatch.setattr("nixl.torch_transfer._selected_layout", unexpected)
    monkeypatch.setattr(
        "nixl.torch_transfer._reject_overlapping_destination", unexpected
    )

    backend = local._backend
    provider_selection = selection._backend_selection
    override_calls = []

    def raw_notification_override(handle, notification):
        override_calls.append((handle, notification))
        return "DONE"

    backend._raw_post_xfer_with_notification_override = raw_notification_override
    bound = selection.bind()
    for notification in (b"A", b"B", b"A", None, b"A", None):
        work = bound.submit(notification=notification)
        assert work.wait().state is transfer.WorkState.COMPLETED
        work.close()

    # One request and one provider-neutral token cover both A -> B -> A and
    # None -> A -> None; the tri-state raw post clears None explicitly.
    assert len(local_agent.make_calls) == 1
    assert [notification for _, notification in override_calls] == [
        b"A",
        b"B",
        b"A",
        None,
        b"A",
        None,
    ]
    assert len({handle for handle, _ in override_calls}) == 1
    assert local_agent.post_notifications == []
    assert all(call[5] == b"" for call in local_agent.make_calls)

    native_local = local_agent.make_calls[0][2]
    native_remote = local_agent.make_calls[0][4]
    assert native_local is provider_selection.local_indices
    assert native_remote is provider_selection.remote_indices
    assert native_local.tolist() == [7, 1, 4]
    assert native_remote.tolist() == [0, 5, 2]
    assert native_local.readonly
    assert native_remote.readonly
    assert native_local.c_contiguous
    assert native_remote.c_contiguous

    def direct_submit(notification, token):
        adopted = []
        backend.submit_prevalidated(
            transfer.TransferOp.WRITE,
            plan._handle,
            selection=provider_selection,
            notification=notification,
            reuse_token=token,
            adopt_work=adopted.append,
        )
        assert len(adopted) == 1
        assert adopted[0].state is transfer.WorkState.COMPLETED
        adopted[0].release()

    # A direct SPI caller that reuses one token across presence states is also
    # safe on an older binding: the provider retires the incompatible request.
    backend._raw_post_xfer_with_notification_override = None
    fallback_start = len(local_agent.make_calls)
    fallback_notifications = len(local_agent.post_notifications)
    fallback_token = object()
    for notification in (None, b"A", None):
        direct_submit(notification, fallback_token)
    fallback_calls = local_agent.make_calls[fallback_start:]
    assert len(fallback_calls) == 3
    assert len({id(call[-1]) for call in fallback_calls}) == 3
    assert local_agent.post_notifications[fallback_notifications:] == [b"", b"A", b""]

    bound.close()
    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


def test_prevalidated_one_shot_binds_final_notification_before_raw_post(monkeypatch):
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint(
        "one-shot-local", backend="nixl", options={"_agent": local_agent}
    )
    remote = transfer.Endpoint(
        "one-shot-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region(0, 8, stride=8, count=8)],
        remote=[peer.region("remote-buffer", 0, 8, stride=8, count=8)],
        indexed=True,
    )
    selection = plan.select_indices(
        transfer.TransferOp.WRITE,
        local_indices=np.asarray([7, 1, 4], dtype=np.int32),
        remote_indices=np.asarray([0, 5, 2], dtype=np.int32),
    )
    backend = local._backend
    raw_posts = []

    def raw_post(native_handle):
        raw_posts.append(native_handle)
        return "DONE"

    def unexpected_override(*_args, **_kwargs):
        raise AssertionError("one-shot selection used a notification override")

    backend._raw_post_xfer = raw_post
    backend._raw_post_xfer_with_notification_override = unexpected_override

    def unexpected_cache_path(*_args, **_kwargs):
        raise AssertionError("one-shot selection entered the reusable cache path")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "_take_cached_request_locked", unexpected_cache_path)
        patch.setattr(backend, "_reconcile_request_pool_locked", unexpected_cache_path)
        for notification in (b"final-payload", None):
            work = selection.submit(notification=notification)
            assert work.wait().state is transfer.WorkState.COMPLETED
            work.close()

    assert len(local_agent.make_calls) == 2
    assert [call[5] for call in local_agent.make_calls] == [b"final-payload", b""]
    assert raw_posts == [call[-1]._handle for call in local_agent.make_calls]
    assert local_agent.post_notifications == []
    assert plan._handle.cached_request_count == 0

    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


@pytest.mark.parametrize("commit_before_interrupt", (False, True))
def test_prevalidated_one_shot_release_interrupt_is_exactly_retryable(
    provider, commit_before_interrupt
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, stride=16, count=4)],
        [_region(provider.imported, remote=True, stride=16, count=4)],
        indexed=True,
    )
    selection = _single_block_selection(transfer.TransferOp.WRITE, plan)
    adopted = []

    def adopt_work(work):
        provider.agent.actions.append("adopt-work")
        adopted.append(work)

    result = provider.backend.submit_prevalidated(
        transfer.TransferOp.WRITE,
        plan,
        selection=selection,
        notification=None,
        reuse_token=None,
        adopt_work=adopt_work,
    )
    assert result is None
    assert provider.agent.actions == ["make", "adopt-work", "post"]
    work = adopted[0]
    handle = work._handle
    request = work._request
    assert handle is not None
    assert request is not None

    if commit_before_interrupt:
        native_release = handle.release
        armed = True

        def interrupt_after_commit():
            nonlocal armed
            if handle.released:
                return
            native_release()
            if armed:
                armed = False
                raise KeyboardInterrupt("release interrupted after native commit")

        handle.release = interrupt_after_commit
    else:
        handle.release_errors.append(
            KeyboardInterrupt("release interrupted before native commit")
        )

    with pytest.raises(KeyboardInterrupt, match="release interrupted"):
        work.close()

    assert work._handle is handle
    assert work._request is request
    assert id(work) in provider.backend._works
    assert plan.requests[id(request)] is request
    assert handle.released is commit_before_interrupt
    assert handle.release_calls == 1

    work.close()
    assert work._released
    assert work._handle is None
    assert id(work) not in provider.backend._works
    assert id(request) not in plan.requests
    assert handle.release_calls == (1 if commit_before_interrupt else 2)
    provider.backend.release_plan(plan)


@pytest.mark.parametrize("provider", tuple(transfer.ThreadMode), indirect=True)
def test_prevalidated_zero_poll_prefers_plain_post_and_preserves_fused_fallbacks(
    provider, monkeypatch
):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration, stride=16, count=4)],
        [_region(provider.imported, remote=True, stride=16, count=4)],
        indexed=True,
    )
    selection = _single_block_selection(transfer.TransferOp.WRITE, plan)

    def unexpected_frombuffer(*_args, **_kwargs):
        raise AssertionError("prevalidated selection allocated a NumPy buffer wrapper")

    provider_module = sys.modules[NixlBackend.__module__]
    monkeypatch.setattr(provider_module.np, "frombuffer", unexpected_frombuffer)
    raw_statuses = ["DONE", "PROC"]
    raw_calls = []
    fused_calls = []

    def raw_post(native_handle):
        raw_calls.append(native_handle)
        return raw_statuses.pop(0)

    def fused_post(native_handle, max_polls, timeout_ns):
        fused_calls.append((native_handle, max_polls, timeout_ns))
        return "DONE"

    def submit(max_polls, timeout_ns, notification):
        adopted = []
        state = provider.backend.submit_prevalidated_and_poll(
            transfer.TransferOp.WRITE,
            plan,
            selection=selection,
            notification=notification,
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            adopt_work=adopted.append,
        )
        assert len(adopted) == 1
        return state, adopted[0]

    provider.backend._raw_post_xfer = raw_post
    provider.backend._raw_post_xfer_and_poll = fused_post
    state, work = submit(0, None, b"post-only-immediate")
    assert state is transfer.WorkState.COMPLETED
    work.close()

    state, work = submit(0, None, b"post-only-pending")
    assert state is transfer.WorkState.RUNNING
    assert work.state is transfer.WorkState.COMPLETED
    work.close()

    assert len(raw_calls) == 2
    assert fused_calls == []

    for max_polls, timeout_ns in ((0, 0), (3, None)):
        state, work = submit(max_polls, timeout_ns, b"fused")
        assert state is transfer.WorkState.COMPLETED
        work.close()

    provider.backend._raw_post_xfer = None
    state, work = submit(0, None, b"raw-unavailable")
    assert state is transfer.WorkState.COMPLETED
    work.close()

    assert [call[1:] for call in fused_calls] == [(0, 0), (3, None), (0, None)]
    assert [call[5] for call in provider.agent.make_calls] == [
        b"post-only-immediate",
        b"post-only-pending",
        b"fused",
        b"fused",
        b"raw-unavailable",
    ]

    # A reusable/cached selection needs a request-local notification override.
    # Even the internal zero-poll entry must keep that override ahead of both
    # the exact plain-post route and the fused fallback.
    override_calls = []
    cached_token = object()
    cached_make_start = len(provider.agent.make_calls)

    def unexpected_cached_raw_post(*_args):
        raise AssertionError("cached notification override used plain raw post")

    def notification_override(native_handle, notification):
        override_calls.append((native_handle, notification))
        return "DONE"

    provider.backend._raw_post_xfer = unexpected_cached_raw_post
    provider.backend._raw_post_xfer_with_notification_override = notification_override
    for notification in (b"cached-A", None, b"cached-B"):
        adopted = []
        state = provider.backend._submit_prevalidated_impl(
            transfer.TransferOp.WRITE,
            plan,
            selection=selection,
            notification=notification,
            reuse_token=cached_token,
            adopt_work=adopted.append,
            initial_max_polls=0,
            initial_timeout_ns=None,
        )
        assert state is transfer.WorkState.COMPLETED
        assert len(adopted) == 1
        adopted[0].close()

    assert len(provider.agent.make_calls) == cached_make_start + 1
    assert len({call[0] for call in override_calls}) == 1
    assert [call[1] for call in override_calls] == [b"cached-A", None, b"cached-B"]
    assert provider.backend._raw_post_xfer_and_poll is fused_post
    assert len(fused_calls) == 3
    assert all(call[2] is selection.local_indices for call in provider.agent.make_calls)
    assert all(
        call[4] is selection.remote_indices for call in provider.agent.make_calls
    )
    assert selection.local_indices.readonly
    assert selection.remote_indices.readonly
    assert plan.cached_request_count == 1
    provider.backend.release_plan(plan)


def test_provider_hot_path_records_use_real_slots(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    work = _submit(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert work._request is not None
    assert not hasattr(work, "__dict__")
    assert not hasattr(work._request, "__dict__")
    work.close()

    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    assert not hasattr(slot, "__dict__")
    slot.start()
    assert slot._work is not None
    assert slot._work._request is not None
    assert not hasattr(slot._work, "__dict__")
    assert not hasattr(slot._work._request, "__dict__")
    assert slot.state is transfer.WorkState.COMPLETED
    slot.recycle()
    slot.close()
    provider.backend.release_plan(plan)


def test_prevalidated_one_shot_fuses_initial_and_continuation_polls(monkeypatch):
    success = object()
    in_progress = object()

    class FakeRawAgent:
        def __init__(self):
            self.post_status = success
            self.post_calls = []
            self.initial_status = in_progress
            self.initial_calls = []
            self.continuation_calls = []

        def postXferReq(self, handle):
            self.post_calls.append(handle)
            return self.post_status

        def getXferStatus(self, _handle):
            raise AssertionError("scalar status must not serve fused QD1")

        def postXferReqAndPoll(self, handle, max_polls, timeout_ns):
            self.initial_calls.append((handle, max_polls, timeout_ns))
            return self.initial_status

        def getXferStatusBatch(
            self, handle, max_polls, timeout_ns, timeout_check_interval
        ):
            self.continuation_calls.append(
                (handle, max_polls, timeout_ns, timeout_check_interval)
            )
            return success

    class RawOnlyAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.agent = FakeRawAgent()

        def make_prepped_xfer(self, *args, **kwargs):
            handle = super().make_prepped_xfer(*args, **kwargs)
            handle._handle = object()
            return handle

        def transfer(self, _handle):
            raise AssertionError("high-level post must not serve fused QD1")

        def check_xfer_state(self, _handle):
            raise AssertionError("high-level status must not serve fused QD1")

    fake_bindings = types.ModuleType("nixl._bindings")
    fake_bindings.nixlAgent = FakeRawAgent
    fake_bindings.NIXL_SUCCESS = success
    fake_bindings.NIXL_IN_PROG = in_progress
    package = sys.modules[NixlBackend.__module__.rpartition(".")[0]]
    monkeypatch.setitem(sys.modules, "nixl._bindings", fake_bindings)
    monkeypatch.setattr(package, "_bindings", fake_bindings, raising=False)

    local_agent = RawOnlyAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint(
        "fused-selection-local", backend="nixl", options={"_agent": local_agent}
    )
    remote = transfer.Endpoint(
        "fused-selection-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region(0, 8, stride=8, count=8)],
        remote=[peer.region("remote-buffer", 0, 8, stride=8, count=8)],
        indexed=True,
    )
    selection = plan.select_indices(
        transfer.TransferOp.WRITE,
        local_indices=np.asarray([7, 1, 4], dtype=np.int32),
        remote_indices=np.asarray([0, 5, 2], dtype=np.int32),
    )
    assert local.capabilities.fused_prevalidated_submit_poll
    assert local.capabilities.fused_work_poll

    for max_polls, timeout_ns in ((1 << 63, None), (1, 1 << 63)):
        with pytest.raises(ValueError):
            local._backend.submit_prevalidated_and_poll(
                transfer.TransferOp.WRITE,
                plan._handle,
                selection=selection._backend_selection,
                notification=None,
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                adopt_work=lambda _work: None,
            )

    work, terminal = selection.submit_and_test(
        notification=b"post-only", max_polls=0, timeout_ns=None
    )
    assert terminal
    assert work._state is transfer.WorkState.COMPLETED
    work.close()

    work, terminal = selection.submit_and_test(
        notification=b"pending", max_polls=64, timeout_ns=None
    )
    assert not terminal
    # Do not insert an ordinary scalar Work.state probe between the fused
    # initial and fused continuation boundaries under test.
    assert work._state is transfer.WorkState.RUNNING
    assert work.test_bounded(
        max_polls=4096,
        timeout_ns=100_000,
        timeout_check_interval=32,
    )
    assert work.state is transfer.WorkState.COMPLETED
    work.close()

    local_agent.agent.initial_status = success
    work, terminal = selection.submit_and_test(
        notification=b"immediate", max_polls=17, timeout_ns=23
    )
    assert terminal
    assert work.test()
    work.close()

    assert [call[1:] for call in local_agent.agent.initial_calls] == [
        (64, None),
        (17, 23),
    ]
    assert [call[1:] for call in local_agent.agent.continuation_calls] == [
        (4096, 100_000, 32)
    ]
    assert len(local_agent.agent.post_calls) == 1
    assert [call[5] for call in local_agent.make_calls] == [
        b"post-only",
        b"pending",
        b"immediate",
    ]

    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


def test_prevalidated_selection_requires_provider_alias_and_layout_agreement(
    monkeypatch,
):
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint(
        "proof-local", backend="nixl", options={"_agent": local_agent}
    )
    remote = transfer.Endpoint(
        "proof-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(32), name="local-buffer")
    remote_registration = remote.register(bytearray(32), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region(0, 8, stride=8, count=4)],
        remote=[peer.region("remote-buffer", 0, 8, stride=8, count=4)],
        indexed=True,
    )
    selection = plan.select_indices(
        "write",
        local_indices=np.asarray([3, 1], dtype=np.int32),
        remote_indices=np.asarray([0, 2], dtype=np.int32),
    )
    provider_plan = plan._handle
    provider_plan.uniform_pair_nbytes = None
    provider_plan.remote_full_nonoverlapping = False
    calls = {"layout": 0, "overlap": 0}
    original_layout = _selected_layout
    original_overlap = _reject_overlapping_destination

    def count_layout(*args, **kwargs):
        calls["layout"] += 1
        return original_layout(*args, **kwargs)

    def count_overlap(*args, **kwargs):
        calls["overlap"] += 1
        return original_overlap(*args, **kwargs)

    monkeypatch.setattr("nixl.torch_transfer._selected_layout", count_layout)
    monkeypatch.setattr(
        "nixl.torch_transfer._reject_overlapping_destination", count_overlap
    )
    work = selection.submit()
    work.close()
    assert calls == {"layout": 1, "overlap": 1}

    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


@pytest.mark.parametrize(
    "thread_mode", (transfer.ThreadMode.SINGLE, transfer.ThreadMode.SERIALIZED)
)
def test_provider_fused_request_slot_integrates_with_endpoint_contract(thread_mode):
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local = transfer.Endpoint(
        f"slot-{thread_mode.value}-local",
        backend="nixl",
        thread_mode=thread_mode,
        options={"_agent": local_agent},
    )
    remote = transfer.Endpoint(
        "slot-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region()],
        remote=[peer.region("remote-buffer")],
        indexed=True,
    )
    bound = plan.bind(
        transfer.TransferOp.WRITE,
        local_indices=[0],
        remote_indices=[0],
        notification=b"persistent-complete",
    )
    slot = bound.create_slot()

    identities = []
    for generation in range(1, 5):
        assert slot.start_and_wait().state is transfer.WorkState.COMPLETED
        assert slot.generation == generation
        work = slot._backend_slot._work
        identities.append((id(work), id(work._request), id(work._handle)))
        slot.recycle()
        assert slot.state is None

    # execute() makes only Core's public generation idle. The provider keeps a
    # clean COMPLETED receipt and folds it into the next start, or into close.
    for generation in range(5, 8):
        assert slot.execute() == generation
        assert slot.state is None
        provider_slot = slot._backend_slot
        assert provider_slot._active
        assert provider_slot._repostable
        assert provider_slot.poll_state() is transfer.WorkState.COMPLETED
        work = provider_slot._work
        identities.append((id(work), id(work._request), id(work._handle)))
    with pytest.raises(transfer.BusyError, match="no generation"):
        slot.recycle()

    assert len(set(identities)) == 1
    assert len(local_agent.make_calls) == 1
    assert local_agent.make_calls[0][5] == b"persistent-complete"
    handle = local_agent.make_calls[0][-1]
    assert handle.post_calls == 7
    assert handle.release_calls == 0

    slot.close()
    assert handle.release_calls == 1
    bound.close()
    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()


def test_deferred_success_close_retries_interrupted_release(provider):
    plan = _prepare(
        provider.backend,
        [_region(provider.registration)],
        [_region(provider.imported, remote=True)],
        indexed=False,
    )
    slot = _create_request_slot(
        provider.backend,
        transfer.TransferOp.WRITE,
        plan,
        local_indices=None,
        remote_indices=None,
        notification=None,
    )
    slot.start()
    assert slot.poll_state() is transfer.WorkState.COMPLETED
    assert slot._active and slot._repostable
    handle = slot._work._handle
    release = handle.release

    def interrupt_after_native_release():
        # Model the native owner clearing its pointer in the same C++ frame as
        # release: a later signal must make retry a native no-op.
        if handle.released:
            return
        release()
        raise KeyboardInterrupt()

    handle.release = interrupt_after_native_release
    with pytest.raises(KeyboardInterrupt):
        slot.close()
    assert not slot._active
    assert not slot._repostable
    assert not slot._closed
    assert handle.released
    assert handle.release_calls == 1

    slot.close()
    assert slot._closed
    assert handle.release_calls == 1
    provider.backend.release_plan(plan)


def test_execute_cancel_lost_race_retains_clean_deferred_receipt(monkeypatch):
    local_agent = _FakeAgent()
    remote_agent = _FakeAgent()
    local_agent.submit_status = "PROC"
    race = {"cancel_phase": False, "cancel_polls": 0}

    def check_xfer_state(_handle):
        if not race["cancel_phase"]:
            return "PROC"
        race["cancel_polls"] += 1
        # Core's pre-cancel refresh still observes RUNNING. NixlWork.cancel()
        # then observes that completion won the race and returns False.
        return "PROC" if race["cancel_polls"] == 1 else "DONE"

    local_agent.check_xfer_state = check_xfer_state
    local = transfer.Endpoint(
        "cancel-race-local",
        backend="nixl",
        thread_mode=transfer.ThreadMode.MULTIPLE,
        options={"_agent": local_agent},
    )
    remote = transfer.Endpoint(
        "cancel-race-remote", backend="nixl", options={"_agent": remote_agent}
    )
    local_agent.remote_name = _native_name(remote.endpoint_id, remote.incarnation)
    local_registration = local.register(bytearray(64), name="local-buffer")
    remote_registration = remote.register(bytearray(64), name="remote-buffer")
    peer = local.import_peer(remote.export_metadata())
    plan = local.prepare(
        local=[local_registration.region()],
        remote=[peer.region("remote-buffer")],
        indexed=True,
    )
    bound = plan.bind(
        transfer.TransferOp.WRITE,
        local_indices=[0],
        remote_indices=[0],
    )
    slot = bound.create_slot()
    progress_reached = threading.Event()
    cancel_finished = threading.Event()

    def coordinate_between_bounded_chunks():
        race["cancel_phase"] = True
        progress_reached.set()
        if not cancel_finished.wait(5):
            raise AssertionError("cancel did not run between execute poll chunks")

    monkeypatch.setattr(local, "progress", coordinate_between_bounded_chunks)
    outcome = {}

    def execute():
        try:
            outcome["generation"] = slot.execute()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert progress_reached.wait(5)
        assert not slot.cancel()
    finally:
        cancel_finished.set()
    thread.join(5)
    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["generation"] == 1
    assert race["cancel_polls"] == 2
    assert slot.state is None

    provider_slot = slot._backend_slot
    handle = provider_slot._work._handle
    assert provider_slot._active
    assert provider_slot._repostable
    assert provider_slot.poll_state() is transfer.WorkState.COMPLETED
    assert handle.post_calls == 1
    assert handle.release_calls == 0

    slot.close()
    assert handle.release_calls == 1
    bound.close()
    plan.close()
    peer.close()
    local_registration.close()
    remote_registration.close()
    local.close()
    remote.close()
