# SPDX-License-Identifier: Apache-2.0
"""Optional provider and migration shim for torch.distributed._transfer.

The shim keeps existing framework metadata, descriptor and notification formats.
It is a review prototype: completed transfers only, not a new cancellation API.
"""

import numpy as np
import torch

from torch.distributed._transfer import Endpoint


def _state(state):
    return {"DONE": "done", "PROC": "pending", "PEND": "pending"}.get(state, "error")


class NixlBackend:
    """Wrap an existing nixl_agent; preserve its backend/thread configuration."""

    def __init__(self, agent):
        self.agent = agent

    def register(self, regions, kind):
        descs = [(int(a), int(n), int(d), "") for a, n, d in regions]
        return self.agent.register_memory(descs, "VRAM" if kind == "cuda" else "DRAM")

    def deregister(self, handle):
        self.agent.deregister_memory(handle)

    def metadata(self):
        return self.agent.get_agent_metadata()

    def connect(self, metadata):
        name = self.agent.add_remote_agent(metadata)
        return name.decode() if isinstance(name, bytes) else name

    def disconnect(self, peer):
        self.agent.remove_remote_agent(peer)

    def prepare(self, regions, kind, peer):
        return self.agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT" if peer is None else peer,
            regions,
            "VRAM" if kind == "cuda" else "DRAM",
        )

    def release_catalog(self, handle):
        self.agent.release_dlist_handle(handle)

    def create_work(
        self, operation, local, local_indices, remote, remote_indices, notification
    ):
        return self.agent.make_prepped_xfer(
            operation,
            local,
            local_indices,
            remote,
            remote_indices,
            notification,
        )

    def post(self, handle):
        return _state(self.agent.transfer(handle))

    def poll(self, handle):
        return _state(self.agent.check_xfer_state(handle))

    def release_work(self, handle):
        # Core permits release only after DONE; no version-specific active abort.
        self.agent.release_xfer_handle(handle)

    def notify(self, peer, message):
        self.agent.send_notif(peer, message)

    def notifications(self):
        return self.agent.get_new_notifs()


class _Request:
    def __init__(self, args, notification, temporary=()):
        self.args, self.notification, self.temporary = args, notification, temporary
        self.work = None

    def close(self):
        if self.work is not None:
            self.work.close()
        for catalog in self.temporary:
            catalog.close()


class TorchTransferAgent:
    """Small NIXL-shaped shim shared by the three framework experiments.

    Configuration and wire descriptor construction stay native. Registration,
    catalog preparation, submission, completion and notifications go via Core.
    Raw pointers require an owner that guarantees allocation stability. This
    shim does not make a framework's existing abort/reuse protocol safer.
    """

    def __init__(self, agent, *, owner=None):
        self._native, self._owner = agent, owner
        self.endpoint = Endpoint(NixlBackend(agent))
        self.name = agent.name

    def create_backend(self, *args, **kwargs):
        return self._native.create_backend(*args, **kwargs)

    def get_plugin_list(self):
        return self._native.get_plugin_list()

    def register_memory(self, regions, mem_type=None):
        if isinstance(regions, torch.Tensor):
            return self.endpoint.register_tensor(regions)
        return self.endpoint.register(
            [tuple(row[:3]) for row in regions],
            kind=self._kind(mem_type),
            owner=self._owner,
        )

    def deregister_memory(self, registration):
        registration.close()

    def get_agent_metadata(self):
        return self.endpoint.metadata()

    def add_remote_agent(self, metadata):
        return self.endpoint.connect(metadata)

    def remove_remote_agent(self, peer):
        self.endpoint.disconnect(peer)

    def get_xfer_descs(self, *args, **kwargs):
        # Retain NIXL's wire-picklable descriptor object (not a new wire format).
        return self._native.get_xfer_descs(*args, **kwargs)

    def _kind(self, mem_type=None, descs=None):
        if mem_type is None and descs is not None:
            mem_type = next(
                name
                for name in ("DRAM", "VRAM")
                if self._native.nixl_mems[name] == descs.getType()
            )
        if mem_type not in ("DRAM", "VRAM"):
            raise ValueError("prototype supports DRAM and VRAM only")
        return "cuda" if mem_type == "VRAM" else "host"

    def prep_xfer_dlist(self, peer, descs, mem_type=None):
        return self.endpoint.prepare(
            descs,
            kind=self._kind(mem_type, descs),
            peer=None if peer in ("", "NIXL_INIT_AGENT") else peer,
        )

    def release_dlist_handle(self, catalog):
        catalog.close()

    def make_prepped_xfer(
        self, operation, local, local_indices, remote, remote_indices, notif_msg=b""
    ):
        return _Request(
            (operation, local, local_indices, remote, remote_indices), notif_msg
        )

    def initialize_xfer(
        self, operation, local_descs, remote_descs, peer, notif_msg=b""
    ):
        local = self.prep_xfer_dlist("", local_descs)
        try:
            remote = self.prep_xfer_dlist(peer, remote_descs)
        except Exception:
            local.close()
            raise
        return _Request(
            (
                operation,
                local,
                np.arange(local_descs.descCount(), dtype=np.int32),
                remote,
                np.arange(remote_descs.descCount(), dtype=np.int32),
            ),
            notif_msg,
            (local, remote),
        )

    def transfer(self, request):
        if request.work is not None:
            raise RuntimeError("create a new request for each submission")
        request.work = self.endpoint.submit(
            *request.args, notification=request.notification
        )
        return {"done": "DONE", "pending": "PROC", "error": "ERR"}[request.work.state]

    def check_xfer_state(self, request):
        if request.work is None:
            return "PEND"
        state = request.work.poll()
        if state == "done":
            # SGLang relies on native handle finalizers; release explicitly here.
            request.close()
        return {"done": "DONE", "pending": "PROC", "error": "ERR"}[state]

    def release_xfer_handle(self, request):
        request.close()

    def send_notif(self, peer, message):
        self.endpoint.notify(peer, message)

    def get_new_notifs(self):
        return self.endpoint.notifications()

    def close(self):
        self.endpoint.close()
