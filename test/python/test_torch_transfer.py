# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; these do not exercise native NIXL/UCX or CUDA."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("torch.distributed._transfer")
from nixl.torch_transfer import TorchTransferAgent


@pytest.fixture
def agent():
    native = Mock()
    native.name = "local"
    native.nixl_mems = {"DRAM": 0, "VRAM": 1}
    native.add_remote_agent.return_value = b"remote"
    native.prep_xfer_dlist.side_effect = lambda *args: object()
    native.transfer.return_value = "PROC"
    native.check_xfer_state.return_value = "PROC"
    shim = TorchTransferAgent(native, owner=bytearray(16))
    shim.register_memory([(1024, 16, 0, "")], "DRAM")
    shim.add_remote_agent(b"unchanged-metadata")
    yield shim, native
    native.check_xfer_state.return_value = "DONE"
    shim.close()


def test_indexed_requests_reuse_catalogs_and_preserve_arrays(agent):
    shim, native = agent
    local = shim.prep_xfer_dlist(
        "", np.array([[1024, 4, 0, 4, 4]], dtype=np.uint64), "DRAM"
    )
    remote = shim.prep_xfer_dlist("remote", [(2048, 4, 0)], "DRAM")
    indices = np.array([0], dtype=np.int32)
    for _ in range(2):
        request = shim.make_prepped_xfer(
            "WRITE", local, indices, remote, indices, b"room_kv"
        )
        assert shim.transfer(request) == "PROC"
        call = native.make_prepped_xfer.call_args.args
        assert call[0] == "WRITE" and call[2] is indices and call[4] is indices
        assert call[5] == b"room_kv"
        with pytest.raises(RuntimeError, match="not DONE"):
            shim.release_xfer_handle(request)
        native.check_xfer_state.return_value = "DONE"
        assert shim.check_xfer_state(request) == "DONE"
        shim.release_xfer_handle(request)  # idempotent after polling
        native.check_xfer_state.return_value = "PROC"
    assert native.prep_xfer_dlist.call_count == 2
    assert native.release_xfer_handle.call_count == 2


def test_read_fallback_keeps_native_wire_descriptors(agent):
    shim, native = agent
    descs = Mock()
    descs.getType.return_value = 0
    descs.descCount.return_value = 1
    native.get_xfer_descs.return_value = descs
    assert shim.get_xfer_descs(torch.zeros(4)) is descs
    request = shim.initialize_xfer("READ", descs, descs, "remote", b"bucket")
    assert shim.transfer(request) == "PROC"
    native.check_xfer_state.return_value = "DONE"
    assert shim.check_xfer_state(request) == "DONE"
    assert native.make_prepped_xfer.call_args.args[0] == "READ"
    assert native.release_dlist_handle.call_count == 2


def test_metadata_and_notifications_are_unchanged(agent):
    shim, native = agent
    native.get_agent_metadata.return_value = b"metadata"
    native.get_new_notifs.return_value = {"remote": [b"opaque\x00tag"]}
    assert shim.get_agent_metadata() == b"metadata"
    assert shim.get_new_notifs() == {"remote": [b"opaque\x00tag"]}
    assert shim.send_notif("remote", b"ack") is None
    native.send_notif.assert_called_once_with("remote", b"ack")


def test_raw_registration_requires_owner_but_tensor_is_its_own_owner():
    native = Mock()
    shim = TorchTransferAgent(native)
    with pytest.raises(ValueError):
        shim.register_memory([(1024, 4, 0, "")], "DRAM")
    reg = shim.register_memory(torch.zeros(4))
    reg.close()
    shim.close()
