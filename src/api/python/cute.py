# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental CuTe-DSL host runtime for NIXL device communication.

This module is intentionally small and conservative. It gives Python kernel
authors a NIXL-shaped host control plane while the device ABI is finalized:

* named local agents;
* explicit tensor registration;
* explicit metadata export/import;
* prepared local and remote transfer views;
* host-side READ/WRITE smoke paths for validation;
* device bitcode discovery for future CuTe ``ffi`` declarations.

The device-side design follows the NCCL CuTe-DSL binding pattern: expose a
C-shaped wrapper ABI in a NIXL-owned device bitcode artifact, locate that
artifact with CUDA pathfinder when possible, and keep Python/CuTe wrappers away
from unstable C++ template internals.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from ._api import (
    nixl_agent,
    nixl_agent_config,
    nixl_prepped_dlist_handle,
    nixl_xfer_handle,
)

TransferOp = Literal["READ", "WRITE"]

_BITCODE_NAMES = ("libnixl_device.bc", "nixl_device.bc")
_HEADER_SUFFIX = Path("gpu/ucx/nixl_device.cuh")
_WRAPPER_HEADER_SUFFIX = Path("gpu/ucx/nixl_device_wrapper.cuh")


@dataclass(frozen=True)
class DeviceApiStatus:
    """Discovery result for the future CuTe device-link path."""

    header_path: Path | None
    wrapper_header_path: Path | None
    bitcode_path: Path | None
    diagnostics: tuple[str, ...] = ()

    @property
    def ready_for_cute(self) -> bool:
        return (
            self.header_path is not None
            and self.wrapper_header_path is not None
            and self.bitcode_path is not None
        )


@dataclass(frozen=True)
class RegisteredTensor:
    name: str
    tensor: Any
    memory_type: str
    reg_descs: Any


@dataclass(frozen=True)
class RemoteAgent:
    name: str
    metadata: bytes


@dataclass(frozen=True)
class PreparedView:
    """Prepared local or remote descriptor list.

    ``handle`` is intentionally opaque.  CuTe kernels should eventually consume
    a smaller device view handle produced by NIXL, not this Python object.
    """

    handle: nixl_prepped_dlist_handle
    count: int
    memory_type: str
    agent_name: str


@dataclass
class Agent:
    """NIXL host runtime for CuTe/cuTile experiments.

    The shape mirrors vLLM's NIXL usage: local registration is separate from
    metadata export, remote metadata is imported explicitly, and transfers are
    request-local.
    """

    name: str
    backends: Sequence[str] = ("UCX",)
    num_threads: int = 0
    capture_telemetry: bool = True
    _agent: nixl_agent = field(init=False, repr=False)
    _registrations: dict[str, RegisteredTensor] = field(default_factory=dict, init=False)
    _remotes: dict[str, RemoteAgent] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        config = nixl_agent_config(
            backends=list(self.backends),
            num_threads=self.num_threads,
            capture_telemetry=self.capture_telemetry,
        )
        self._agent = nixl_agent(self.name, config)

    @property
    def raw_agent(self) -> nixl_agent:
        return self._agent

    def register_tensor(
        self,
        tensor: Any,
        *,
        name: str,
        memory_type: str | None = None,
    ) -> RegisteredTensor:
        """Register one contiguous tensor and keep its registration handle."""

        reg_descs = self._agent.register_memory(
            tensor, mem_type=memory_type, backends=list(self.backends)
        )
        resolved_memory_type = memory_type or _tensor_memory_type(tensor)
        registration = RegisteredTensor(name, tensor, resolved_memory_type, reg_descs)
        self._registrations[name] = registration
        return registration

    def deregister_tensor(self, name: str) -> None:
        registration = self._registrations.pop(name)
        self._agent.deregister_memory(registration.reg_descs, backends=list(self.backends))

    def export_metadata(
        self,
        registration: RegisteredTensor | str | None = None,
        *,
        include_connection_info: bool = True,
    ) -> bytes:
        """Export full or registration-scoped NIXL metadata.

        Passing a registration follows the vLLM-style partial metadata path.
        Full metadata remains available for early experiments and compatibility
        with existing NIXL Python examples.
        """

        if registration is None:
            return self._agent.get_agent_metadata()
        if isinstance(registration, str):
            registration = self._registrations[registration]
        return self._agent.get_partial_agent_metadata(
            registration.reg_descs,
            inc_conn_info=include_connection_info,
            backends=list(self.backends),
        )

    def load_remote_metadata(self, metadata: bytes) -> RemoteAgent:
        remote_name = self._agent.add_remote_agent(metadata)
        remote = RemoteAgent(remote_name, metadata)
        self._remotes[remote_name] = remote
        return remote

    def invalidate_remote(self, remote: RemoteAgent | str) -> None:
        remote_name = remote.name if isinstance(remote, RemoteAgent) else remote
        self._agent.remove_remote_agent(remote_name)
        self._remotes.pop(remote_name, None)

    def prepare_local_view(
        self,
        registration: RegisteredTensor | str,
    ) -> PreparedView:
        if isinstance(registration, str):
            registration = self._registrations[registration]
        handle = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            registration.tensor,
            mem_type=registration.memory_type,
            backends=list(self.backends),
        )
        return PreparedView(handle, 1, registration.memory_type, "NIXL_INIT_AGENT")

    def prepare_remote_view(
        self,
        remote: RemoteAgent | str,
        descriptors: Any,
        *,
        memory_type: str,
        count: int | None = None,
    ) -> PreparedView:
        remote_name = remote.name if isinstance(remote, RemoteAgent) else remote
        handle = self._agent.prep_xfer_dlist(
            remote_name,
            descriptors,
            mem_type=memory_type,
            backends=list(self.backends),
        )
        return PreparedView(
            handle,
            _descriptor_count(descriptors) if count is None else count,
            memory_type,
            remote_name,
        )

    def make_transfer(
        self,
        op: TransferOp,
        local: PreparedView,
        local_indices: Sequence[int] | np.ndarray,
        remote: PreparedView,
        remote_indices: Sequence[int] | np.ndarray,
        *,
        notification: bytes = b"",
    ) -> nixl_xfer_handle:
        return self._agent.make_prepped_xfer(
            op,
            local.handle,
            _indices(local_indices),
            remote.handle,
            _indices(remote_indices),
            notif_msg=notification,
            backends=list(self.backends),
        )

    def transfer(self, handle: nixl_xfer_handle, *, notification: bytes = b"") -> str:
        return self._agent.transfer(handle, notif_msg=notification)

    def check(self, handle: nixl_xfer_handle) -> str:
        return self._agent.check_xfer_state(handle)

    def close(self) -> None:
        for remote_name in list(self._remotes):
            self.invalidate_remote(remote_name)
        for name in list(self._registrations):
            self.deregister_tensor(name)


def device_api_status(
    *,
    bitcode_path: str | os.PathLike[str] | None = None,
    header_path: str | os.PathLike[str] | None = None,
    wrapper_header_path: str | os.PathLike[str] | None = None,
) -> DeviceApiStatus:
    """Locate the pieces required for a CuTe external device binding."""

    diagnostics: list[str] = []
    header = _resolve_path(header_path) if header_path else _find_header(diagnostics)
    wrapper = (
        _resolve_path(wrapper_header_path)
        if wrapper_header_path
        else _find_wrapper_header(diagnostics)
    )
    bitcode = _resolve_path(bitcode_path) if bitcode_path else _find_bitcode(diagnostics)
    if header is None:
        diagnostics.append("nixl_device.cuh was not found in package or prefix paths")
    if wrapper is None:
        diagnostics.append(
            "nixl_device_wrapper.cuh was not found in package or prefix paths"
        )
    if bitcode is None:
        diagnostics.append("NIXL device bitcode was not found")
    return DeviceApiStatus(header, wrapper, bitcode, tuple(diagnostics))


def cute_bitcode(bitcode_path: str | os.PathLike[str] | None = None) -> Any:
    """Return a CuTe ``BitCode`` object for NIXL device wrappers.

    This is the future hook used by declarations such as
    ``cute.ffi(source=nixl.cute.cute_bitcode(), ...)``. It fails clearly until NIXL
    packages a compiler-linkable device artifact.
    """

    status = device_api_status(bitcode_path=bitcode_path)
    if status.bitcode_path is None:
        raise RuntimeError("; ".join(status.diagnostics))
    try:
        from cutlass.cute import BitCode
    except ImportError as exc:
        raise RuntimeError("CuTe-DSL BitCode support is not importable") from exc
    return BitCode(str(status.bitcode_path))


def cute_ffi(bitcode_path: str | os.PathLike[str] | None = None, **kwargs: Any) -> Any:
    """Return ``cute.ffi`` with the NIXL device bitcode source injected.

    This mirrors the NCCL CuTe-DSL binding style: users should not pass raw
    ``--link-libraries`` options to ``cute.compile`` once NIXL packages the
    device bitcode.
    """

    try:
        import cutlass.cute as cute
    except ImportError as exc:
        raise RuntimeError("CuTe-DSL is not importable") from exc
    return cute.ffi(source=cute_bitcode(bitcode_path), **kwargs)


def put_block(*args: Any, **kwargs: Any) -> None:
    """Placeholder for the future CuTe device wrapper declaration.

    The implementation belongs in a NIXL-owned bitcode/C wrapper artifact. The
    host runtime above is useful today; this device call intentionally fails so
    examples cannot accidentally benchmark a stub.
    """

    raise NotImplementedError(
        "nixl.cute.put_block requires a NIXL-owned device bitcode/C wrapper ABI"
    )


def _indices(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(indices, dtype=np.int32)


def _descriptor_count(descs: Any) -> int:
    if hasattr(descs, "shape") and len(descs.shape) > 0:
        return int(descs.shape[0])
    if isinstance(descs, Iterable) and not isinstance(descs, (bytes, bytearray, str)):
        return len(list(descs))
    return 1


def _tensor_memory_type(tensor: Any) -> str:
    device = getattr(tensor, "device", None)
    if device is None:
        return "DRAM"
    device_type = getattr(device, "type", str(device))
    return "VRAM" if device_type in ("cuda", "xpu") else "DRAM"


def _resolve_path(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    return candidate if candidate.exists() else None


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("NIXL_HOME", "NIXL_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        [
            Path(__file__).resolve().parent,
            Path(sys.prefix),
            Path("/usr/local"),
            Path("/usr"),
        ]
    )
    return roots


def _find_header(diagnostics: list[str]) -> Path | None:
    return _find_installed_header(_HEADER_SUFFIX, diagnostics)


def _find_wrapper_header(diagnostics: list[str]) -> Path | None:
    return _find_installed_header(_WRAPPER_HEADER_SUFFIX, diagnostics)


def _find_installed_header(suffix: Path, diagnostics: list[str]) -> Path | None:
    for root in _candidate_roots():
        for candidate in (
            root / suffix,
            root / "include" / "nixl" / suffix,
            root / "include" / suffix,
        ):
            if candidate.exists():
                return candidate
    diagnostics.append(
        f"searched for {suffix} in package, sys.prefix, /usr/local, and /usr"
    )
    return None


def _find_bitcode(diagnostics: list[str]) -> Path | None:
    explicit = os.environ.get("NIXL_DEVICE_BITCODE")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate
        diagnostics.append(f"NIXL_DEVICE_BITCODE does not exist: {candidate}")

    for root in _candidate_roots():
        for dirname in ("lib", "lib64", "share/nixl", "nixl"):
            for name in _BITCODE_NAMES:
                candidate = root / dirname / name
                if candidate.exists():
                    return candidate

    try:
        from cuda.pathfinder import find_bitcode_lib
    except ImportError:
        diagnostics.append("cuda.pathfinder is not importable")
        return None

    try:
        return Path(find_bitcode_lib("nixl_device"))
    except Exception as exc:  # pragma: no cover - depends on local CUDA wheels.
        diagnostics.append(f"cuda.pathfinder did not find nixl_device bitcode: {exc}")
    return None


__all__ = [
    "Agent",
    "DeviceApiStatus",
    "PreparedView",
    "RegisteredTensor",
    "RemoteAgent",
    "cute_ffi",
    "cute_bitcode",
    "device_api_status",
    "put_block",
]
