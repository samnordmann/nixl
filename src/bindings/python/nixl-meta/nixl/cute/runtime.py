# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal host runtime for CuTe-DSL/NIXL experiments."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

import torch

from nixl._api import nixl_agent, nixl_agent_config

from .device import device_api_status


def _as_agent_name(name: str | bytes) -> str:
    return name.decode() if isinstance(name, bytes) else name


def _infer_memory_type(tensor: torch.Tensor, memory_type: str | None) -> str:
    if memory_type is not None:
        return memory_type
    if tensor.device.type == "cuda":
        return "VRAM"
    if tensor.device.type == "cpu":
        return "DRAM"
    raise ValueError(f"Cannot infer NIXL memory type for device {tensor.device!s}")


def _device_id(tensor: torch.Tensor) -> int:
    return tensor.get_device() if tensor.device.type == "cuda" else 0


@dataclass
class RegisteredTensor:
    """A tensor registered with a NIXL agent."""

    agent: "Agent"
    tensor: torch.Tensor
    descs: object
    memory_type: str
    backends: tuple[str, ...]
    released: bool = False

    @property
    def address(self) -> int:
        return self.tensor.data_ptr()

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()

    @property
    def device_id(self) -> int:
        return _device_id(self.tensor)

    def release(self) -> None:
        if not self.released:
            self.agent._agent.deregister_memory(self.descs, list(self.backends))
            self.released = True

    def __enter__(self) -> "RegisteredTensor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


@dataclass(frozen=True)
class RemoteAgent:
    """A remote NIXL agent loaded from metadata."""

    name: str
    metadata: bytes


@dataclass(frozen=True)
class LocalView:
    """Host-side description of a local registered tensor.

    This is not yet a device-consumable handle.  It is the Python control-plane
    object that will eventually map to a small C ABI handle for CuTe kernels.
    """

    registration: RegisteredTensor
    index: int
    handle: int | None = None

    @property
    def address(self) -> int:
        return self.registration.address

    @property
    def nbytes(self) -> int:
        return self.registration.nbytes

    @property
    def device_id(self) -> int:
        return self.registration.device_id


@dataclass(frozen=True)
class RemoteView:
    """Host-side description of a remote NIXL agent imported from metadata."""

    agent: RemoteAgent
    index: int
    handle: int | None = None

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def metadata(self) -> bytes:
        return self.agent.metadata


@dataclass(frozen=True)
class PreparedViews:
    """Explicit local/remote view bundle for future CuTe device wrappers."""

    local: tuple[LocalView, ...]
    remote: tuple[RemoteView, ...]
    device_status: object

    @property
    def device_api_available(self) -> bool:
        return bool(getattr(self.device_status, "available", False))


class Agent:
    """Small NIXL host runtime intended for CuTe-DSL experiments.

    The class wraps the existing high-level ``nixl._api.nixl_agent`` API and
    keeps the control plane explicit.  It does not yet expose a real
    device-consumable ``nixlMemViewH`` because that binding is not available in
    the Python API today.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        backends: Iterable[str] = ("UCX",),
        enable_prog_thread: bool = True,
        num_threads: int = 0,
        capture_telemetry: bool = False,
    ):
        self.name = name or f"nixl-cute-{uuid.uuid4()}"
        self.backends = tuple(backends)
        self._agent = nixl_agent(
            self.name,
            nixl_agent_config(
                enable_prog_thread=enable_prog_thread,
                backends=list(self.backends),
                num_threads=num_threads,
                capture_telemetry=capture_telemetry,
            ),
        )

    def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        memory_type: str | None = None,
        backends: Iterable[str] | None = None,
    ) -> RegisteredTensor:
        if not tensor.is_contiguous():
            raise ValueError("NIXL CuTe prototype only registers contiguous tensors")
        mem_type = _infer_memory_type(tensor, memory_type)
        backend_tuple = tuple(backends) if backends is not None else self.backends
        descs = self._agent.register_memory(tensor, mem_type, list(backend_tuple))
        return RegisteredTensor(self, tensor, descs, mem_type, backend_tuple)

    def export_metadata(
        self,
        registrations: Iterable[RegisteredTensor] | None = None,
        *,
        include_connection_info: bool = True,
        backends: Iterable[str] | None = None,
    ) -> bytes:
        regs = tuple(registrations or ())
        backend_list = list(tuple(backends) if backends is not None else self.backends)
        if not regs:
            return self._agent.get_agent_metadata()
        if len(regs) != 1:
            raise NotImplementedError(
                "Partial metadata export currently supports one registration"
            )
        return self._agent.get_partial_agent_metadata(
            regs[0].descs,
            inc_conn_info=include_connection_info,
            backends=backend_list,
        )

    def load_metadata(self, metadata: bytes) -> RemoteAgent:
        remote_name = _as_agent_name(self._agent.add_remote_agent(metadata))
        return RemoteAgent(name=remote_name, metadata=metadata)

    def remove_remote(self, remote: RemoteAgent | str) -> None:
        self._agent.remove_remote_agent(
            remote.name if isinstance(remote, RemoteAgent) else remote
        )

    def make_connection(self, remote: RemoteAgent | str) -> None:
        self._agent.make_connection(
            remote.name if isinstance(remote, RemoteAgent) else remote
        )

    def prepare_views(
        self,
        *,
        local: Iterable[RegisteredTensor] = (),
        remote: Iterable[RemoteAgent] = (),
    ) -> PreparedViews:
        """Prepare explicit local and remote views for a future device call.

        The returned views intentionally preserve NIXL's asymmetric model: local
        registrations and remote agents remain distinct instead of being forced
        into a symmetric window abstraction.
        """

        return PreparedViews(
            local=tuple(
                LocalView(registration=reg, index=i) for i, reg in enumerate(local)
            ),
            remote=tuple(
                RemoteView(agent=agent, index=i) for i, agent in enumerate(remote)
            ),
            device_status=device_api_status(),
        )

    def prepare_view(
        self,
        *,
        local: Iterable[RegisteredTensor] = (),
        remote: Iterable[RemoteAgent] = (),
    ) -> PreparedViews:
        """Backward-compatible singular spelling for ``prepare_views``."""

        return self.prepare_views(local=local, remote=remote)

