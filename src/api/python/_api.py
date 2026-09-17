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

from __future__ import annotations

import pickle
import sys
import threading
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from . import _bindings as nixlBind  # type: ignore
from .logging import get_logger

if TYPE_CHECKING:
    import torch

# Get logger using centralized configuration
logger = get_logger(__name__)


def _is_torch_tensor(obj: object) -> bool:
    """Return True if obj is a torch.Tensor and torch is already imported.

    Importing torch costs ~1s and dominates `import nixl`. A value can only be
    a torch.Tensor if torch has already been imported elsewhere, so we consult
    sys.modules instead of importing torch here.
    """
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


DEFAULT_COMM_PORT = nixlBind.DEFAULT_COMM_PORT
NIXL_NULL_AGENT = nixlBind.NIXL_NULL_AGENT


class nixl_prepped_dlist_handle:
    """Opaque handle wrapper for a prepared transfer descriptor list.
    Use release() to explicitly free resources; __del__ performs best-effort cleanup.

    Args:
        agent: Owning nixl_agent used to perform release operations.
        value: Internal handle
    """

    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return (
            f"nixl_prepped_dlist_handle(0x{self._handle:x}, released={self._released})"
        )

    def release(self):
        if not self._released:
            self._agent.releasedDlistH(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releasedDlistH(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_prepped_dlist_handle finalization failed for 0x%x",
                        self._handle,
                    )
                except Exception:
                    pass


class nixl_device_view_handle:
    """Owning handle for a device-visible NIXL memory view.

    A view keeps the high-level agent, the converted descriptor list, and the
    original input objects (including tensors) alive until :meth:`release`.
    :attr:`descriptor_lengths` exposes an immutable size snapshot for CuTe's
    trace-time bounds validation without exposing mutable native descriptors.
    The caller must synchronize every CUDA kernel that can use this view before
    releasing it or leaving its context manager; NIXL cannot infer CUDA stream
    dependencies from the host.

    Dropping a live handle triggers best-effort finalization and logs a warning.
    Explicit release, preferably via ``with``, is strongly recommended.
    """

    __slots__ = (
        "_count",
        "_descriptor_lengths",
        "_finalizer",
        "_handle",
        "_keepalive",
        "_kind",
        "_lock",
        "_local_regions",
        "_owner",
        "_regions",
        "_remote_agent",
        "_remote_agents",
        "__weakref__",
    )

    def __init__(
        self,
        owner,
        handle: int,
        *,
        kind: str,
        count: int,
        remote_agent: Optional[str],
        remote_agents: frozenset[str],
        regions: tuple[tuple[object, int, int, int], ...],
        local_regions: tuple[tuple[object, int, int, int], ...],
        keepalive: tuple[object, ...],
    ):
        if not handle:
            raise RuntimeError("NIXL returned a null device memory-view handle")
        self._owner = owner
        self._handle = int(handle)
        self._kind = kind
        self._count = count
        self._descriptor_lengths = tuple(end - start for _, _, start, end in regions)
        self._remote_agent = remote_agent
        self._remote_agents = remote_agents
        self._regions = regions
        self._local_regions = local_regions
        self._keepalive = keepalive
        self._lock = threading.Lock()
        self._finalizer = weakref.finalize(
            self,
            self._finalize,
            owner,
            self._handle,
            self._lock,
            keepalive,
        )

    @staticmethod
    def _finalize(owner, handle: int, lock, keepalive) -> None:
        # keepalive is intentionally captured by the finalizer so buffers remain
        # alive through the native release. It otherwise has no use here.
        with owner._device_view_lock, lock:
            try:
                logger.warning(
                    "Finalizing live NIXL device view 0x%x; CUDA work must be "
                    "synchronized before a view is released",
                    handle,
                )
            except Exception:
                # Logging modules can already be partially torn down during
                # interpreter shutdown. Native cleanup must still be attempted.
                pass
            try:
                owner.agent.releaseMemView(handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_device_view_handle finalization failed for 0x%x",
                        handle,
                        exc_info=True,
                    )
                except Exception:
                    pass
        del keepalive

    @property
    def valid(self) -> bool:
        """Whether the native memory-view handle is still live."""
        return self._finalizer.alive

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`valid`, used by CuTe JIT argument adapters."""
        return self.valid

    @property
    def handle(self) -> int:
        """Raw device-visible handle, suitable for CuTe DSL JIT arguments."""
        if not self.valid:
            raise RuntimeError("NIXL device memory view has been released")
        return self._handle

    @property
    def count(self) -> int:
        """Compatibility alias for :attr:`descriptor_count`."""
        return self._count

    @property
    def descriptor_count(self) -> int:
        """Number of descriptors in this view."""
        return self._count

    @property
    def descriptor_lengths(self) -> tuple[int, ...]:
        """Immutable byte lengths for trace-time descriptor bounds checks."""
        return self._descriptor_lengths

    @property
    def kind(self) -> str:
        """View locality: ``\"local\"`` or ``\"remote\"``."""
        return self._kind

    @property
    def remote_agents(self) -> frozenset[str]:
        """Remote agents referenced by this view, empty for a local view."""
        return self._remote_agents

    @property
    def descriptor_addresses(self) -> tuple[int, ...]:
        """Owner/original base address for every prepared descriptor.

        For a remote view these are the addresses advertised by each peer, not
        necessarily the process-local pointers returned by a device
        ``nixlGetPtr``. Numeric equality across process address spaces is not a
        correctness requirement. Applications using the direct mapped fast
        path must address a remote allocation from its non-null process-local
        ``nixlGetPtr`` base plus the descriptor-relative offset.
        """
        return tuple(start for _, _, start, _ in self._regions)

    def __repr__(self) -> str:
        handle = f"0x{self._handle:x}" if self.valid else "released"
        return (
            f"nixl_device_view_handle({handle}, kind={self._kind!r}, "
            f"count={self._count})"
        )

    def _overlaps_local_registration(self, dlist) -> bool:
        mem_type = dlist.getType()
        for i in range(dlist.descCount()):
            addr, length, device_id, *_ = dlist[i]
            start = int(addr)
            end = start + int(length)
            for view_mem, view_device, view_start, view_end in self._local_regions:
                if (
                    mem_type == view_mem
                    and int(device_id) == view_device
                    and start < view_end
                    and view_start < end
                ):
                    return True
        return False

    def release(self) -> None:
        """Release the native view after all CUDA users have synchronized.

        This operation is idempotent. If native release fails, the handle stays
        valid so the caller can synchronize or otherwise recover and retry.
        """
        with self._owner._device_view_lock, self._lock:
            if not self._finalizer.alive:
                return
            self._owner.agent.releaseMemView(self._handle)
            self._finalizer.detach()
            self._owner._device_views.discard(self)
            self._keepalive = ()

    def close(self) -> None:
        """Alias for :meth:`release` for conventional resource management."""
        self.release()

    def __enter__(self):
        if not self.valid:
            raise RuntimeError("Cannot enter a released NIXL device memory view")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class nixl_xfer_handle:
    """Opaque handle wrapper for a transfer request.
    Use release() to explicitly free resources. If transfer was not complete, this will initiate
    the abort process (if available) and will raise an exception.
    __del__ calls release() and if it fails, it logs the failure and defers release by queuing
    the handle in leaked xfer handles list, which will be re-released during agent destruction

    Args:
        agent: Owning nixl_agent used to perform release operations.
        value: Internal handle
    """

    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return f"nixl_xfer_handle(0x{self._handle:x}, released={self._released})"

    def release(self):
        if not self._released:
            self._agent.releaseXferReq(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releaseXferReq(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_xfer_handle finalization failed for 0x%x; keeping handle alive in agent leak list",
                        self._handle,
                    )
                except Exception:
                    pass
                try:
                    self._agent._leaked_xfer_handles.append(self._handle)
                except Exception:
                    pass
                return


# Opaque handle for backend can be just int, as it's not passed to the user
nixl_backend_handle = int


class nixl_thread_sync_t(Enum):
    """Enumeration of supported thread synchronization modes."""

    NIXL_THREAD_SYNC_NONE = nixlBind.NIXL_THREAD_SYNC_NONE
    NIXL_THREAD_SYNC_STRICT = nixlBind.NIXL_THREAD_SYNC_STRICT
    NIXL_THREAD_SYNC_RW = nixlBind.NIXL_THREAD_SYNC_RW
    NIXL_THREAD_SYNC_DEFAULT = nixlBind.NIXL_THREAD_SYNC_DEFAULT


class nixl_agent_config:
    """Configuration class for NIXL agent.

    Args:
        enable_prog_thread: Whether to enable the progress thread, if available.
        enable_listen_thread: Whether to enable the listener thread for metadata communication.
        listen_port: Specify the port for the listener thread to listen on.
        capture_telemetry: Whether to enable telemetry capture.
        num_threads: Specify number of threads for the supported multi-threaded backends.
        backends: List of backend names for agent to initialize.
            Default is UCX, other backends can be added to the list, or after
            agent creation, can be initialized with create_backend.
        sync_mode: Thread synchronization mode to use for the agent.
            If None, sync_mode is set based on the enable_listen flag.
    """

    def __init__(
        self,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = DEFAULT_COMM_PORT,
        capture_telemetry: bool = False,
        num_threads: int = 0,
        backends: list[str] = ["UCX"],
        sync_mode: Optional[nixl_thread_sync_t] = None,
    ):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread
        self.enable_listen = enable_listen_thread
        self.port = listen_port
        self.capture_telemetry = capture_telemetry
        self.num_threads = num_threads
        if sync_mode is not None and not isinstance(sync_mode, nixl_thread_sync_t):
            raise TypeError(
                f"sync_mode must be a nixl_thread_sync_t (got {type(sync_mode).__name__!r})"
            )

        self.sync_mode = sync_mode


class nixl_agent:
    """Main class for creating a NIXL agent and performing transfers.
    This class provides methods for initializing backends, creating descriptor lists,
    registering memory, performing data transfers, and destroying NIXL objects.

    Args:
        agent_name: Name of the agent, should be unique for clarity.
        nixl_conf: Optional configuration for the agent, described in nixl_agent_config.
        instantiate_all: Whether to instantiate all available backend plugins.
    """

    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if nixl_conf and instantiate_all:
            instantiate_all = False
            logger.warning(
                "Ignoring instantiate_all based on the provided config in agent creation."
            )
        if not nixl_conf:
            nixl_conf = nixl_agent_config()  # Using defaults set in nixl_agent_config

        default_sync_mode = (
            nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
            if nixl_conf.enable_listen
            else nixl_thread_sync_t.NIXL_THREAD_SYNC_NONE
        )

        # Set agent config and instantiate an agent
        agent_config = nixlBind.nixlAgentConfig()
        agent_config.useProgThread = nixl_conf.enable_pthread
        agent_config.useListenThread = nixl_conf.enable_listen
        agent_config.listenPort = nixl_conf.port
        agent_config.syncMode = (nixl_conf.sync_mode or default_sync_mode).value
        agent_config.pthrDelay = 0
        agent_config.lthrDelay = 100000
        agent_config.captureTelemetry = nixl_conf.capture_telemetry
        self.agent = nixlBind.nixlAgent(agent_name, agent_config)

        self.name = agent_name
        self._leaked_xfer_handles: list[int] = []
        self._device_views: weakref.WeakSet[nixl_device_view_handle] = weakref.WeakSet()
        self._device_view_lock = threading.RLock()
        self.notifs: dict[str, list[bytes]] = {}
        self.backends: dict[str, nixl_backend_handle] = {}
        self.backend_mems: dict[str, list[str]] = {}
        self.backend_options: dict[str, dict[str, str]] = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            logger.error("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options: dict[str, dict[str, str]] = {}
        self.plugin_mem_types: dict[str, list[str]] = {}

        if instantiate_all:
            nixl_conf.backends = self.plugin_list

        for bknd in nixl_conf.backends:
            if bknd not in self.plugin_list:
                logger.warning(
                    "Skipping backend registration %s due to the missing plugin.",
                    bknd,
                )
            else:
                # TODO: improve population of init from nixl_conf
                init: dict[str, str] = {}
                if nixl_conf.num_threads > 0:
                    if bknd == "UCX" or bknd == "OBJ" or bknd == "LIBFABRIC":
                        init["num_threads"] = str(nixl_conf.num_threads)
                    elif bknd == "GDS_MT":
                        init["thread_count"] = str(nixl_conf.num_threads)
                    elif bknd == "UCCL":
                        init["num_cpus"] = str(nixl_conf.num_threads)
                self.create_backend(bknd, init)

        self.nixl_mems = {
            "DRAM": nixlBind.DRAM_SEG,
            "VRAM": nixlBind.VRAM_SEG,
            "FILE": nixlBind.FILE_SEG,
            "BLOCK": nixlBind.BLK_SEG,
            "OBJ": nixlBind.OBJ_SEG,
            "cpu": nixlBind.DRAM_SEG,  # deprecated
            "cuda": nixlBind.VRAM_SEG,  # deprecated
        }
        self.nixl_ops = {
            "WRITE": nixlBind.NIXL_WRITE,
            "READ": nixlBind.NIXL_READ,
        }

        logger.info("Initialized NIXL agent: %s", agent_name)

    def __del__(self):
        # Best-effort cleanup of any leaked xfer handles belonging to this agent
        if getattr(self, "_leaked_xfer_handles", None):
            for h in list(self._leaked_xfer_handles):
                try:
                    self.agent.releaseXferReq(h)
                except Exception as e:
                    try:
                        logger.error(
                            "Failed to finalize leaked nixl_xfer_handle 0x%x: %s", h, e
                        )
                    except Exception:
                        pass
            self._leaked_xfer_handles.clear()

    def _load_plugin_params(self, plugin: str):
        if plugin not in self.plugin_list:
            return
        try:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types
        except Exception:
            logger.warning("Failed to load params for plugin %s", plugin, exc_info=True)

    def get_plugin_list(self) -> list[str]:
        """Get the list of available plugins.

        Returns:
            List of plugin names.
        """
        return self.plugin_list

    def get_plugin_mem_types(self, backend: str) -> list[str]:
        """Get the memory types supported by a plugin.

        Args:
            backend: Name of the plugin.

        Returns:
            List of supported memory types.
        """
        if backend not in self.plugin_mem_types:
            self._load_plugin_params(backend)
        if backend not in self.plugin_mem_types:
            logger.warning(
                "Plugin %s is not available to get its supported mem types.", backend
            )
            return []
        return self.plugin_mem_types[backend]

    def get_plugin_params(self, backend: str) -> dict[str, str]:
        """Get the initialization parameters of a plugin.
        This is a dictionary of strings (option name) to strings (default value for that option).

        Args:
            backend: Name of the plugin to get params for.

        Returns:
            Dictionary of plugin parameters, described above.
        """
        if backend not in self.plugin_b_options:
            self._load_plugin_params(backend)
        if backend not in self.plugin_b_options:
            logger.warning("Plugin %s is not available to get its parameters.", backend)
            return {}
        return self.plugin_b_options[backend]

    def get_backend_mem_types(self, backend: str) -> list[str]:
        """Get the memory types supported by a backend.
        Here, a backend means an initialized plugin.
        After a plugin is initialized, the supported memory types might have changed.
        This function is for getting a refreshed list of those memory types.

        Args:
            backend: Name of the backend.

        Returns:
            List of supported memory types.
        """
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its supported mem types.", backend
            )
            return []

    def get_backend_params(self, backend: str) -> dict[str, str]:
        """Get the parameters of a backend.
        Here, a backend means an initialized plugin.
        Available initialization parameters (described above) might have changed after initialization.
        This function is for getting a refreshed list of those parameters.

        Args:
            backend: Name of the backend.

        Returns:
            Dictionary of backend parameters, described in get_plugin_params.
        """
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its parameters.", backend
            )
            return {}

    def create_backend(self, backend: str, initParams: dict[str, str] = {}):
        """Initialize a backend with the specified initialization parameters, described above.

        Args:
            backend: Name of the backend.
            initParams: Dictionary of initialization parameters.
        """
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_mems[backend] = mem_types
        self.backend_options[backend] = backend_options
        logger.info("Backend %s was instantiated", backend)

    def register_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixlBind.nixlRegDList:
        """Register memory regions, optionally with specified backends.

        Args:
            reg_list: List of either memory regions, tensors, or nixlRegDList to register.
            mem_type: Optional memory type, necessary if specifying a list of memory regions.
            backends: Optional list of backend names for registration, otherwise NIXL will try to
                register with all backends that support this memory type.

        Returns:
            nixlRegDList for the registered memory, can be used with deregister_memory.
        """
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.registerMem(reg_descs, handle_list)

        return reg_descs

    def deregister_memory(
        self, dereg_list: nixlBind.nixlRegDList, backends: list[str] = []
    ):
        """Deregister memory regions from the specified backends.

        Args:
            dereg_list: nixlRegDList of memory to deregister, received from register_memory or get_reg_descs.
            backends: Optional list of backend names for deregistration, otherwise NIXL will deregister
                with all the backends that have these memory regions registered.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        with self._device_view_lock:
            affects_ucx = not backends or "UCX" in backends
            if affects_ucx:
                for view in tuple(self._device_views):
                    if view.valid and view._overlaps_local_registration(dereg_list):
                        raise RuntimeError(
                            "Cannot deregister memory overlapping an active NIXL "
                            "device view; synchronize CUDA work and release the view first"
                        )
            self.agent.deregisterMem(dereg_list, handle_list)

    def query_memory(
        self, reg_list, backend: str, mem_type: Optional[str] = None
    ) -> list[Optional[dict[str, str]]]:
        """Query information about memory/storage for a specific backend.

        Args:
            reg_list: List of either memory regions, tensors, or nixlRegDList to query.
            backend: Backend name for querying.
            mem_type: Optional memory type, necessary if specifying a list of memory regions.

        Returns:
            List of query results where each item is either None if not found, or a dictionary with the info
        """
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        # Get the backend handle
        if backend not in self.backends:
            raise ValueError(
                f"Backend '{backend}' not found. Available backends: {list(self.backends.keys())}"
            )

        return self.agent.queryMem(reg_descs, self.backends[backend])

    def make_connection(self, remote_agent: str, backends: list[str] = []):
        """Proactively establish a connection with a remote agent,
        which will reduce the time spent in the first transfer between the two agents.
        NIXL will establish the connection for all the backends that talk to that remote
        agent, or limit to the set of backends passed through the backends argument.
        This function is optional.

        Args:
            backends: Optional list of backend names to limit the connections to specific backends
            remote_agent: Name of the remote agent.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        self.agent.makeConnection(remote_agent, handle_list)

    def prep_xfer_dlist(
        self,
        agent_name: str,
        xfer_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixl_prepped_dlist_handle:
        """Prepare a transfer descriptor list for data transfer.
        Later, elements from this list can be used to create a transfer request by index.
        It should be done on the initiator agent, and for both sides of a transfer.
        Considering loopback, there are 3 modes for agent_name:
        - For local descriptors, it is set to NIXL_INIT_AGENT,
        indicating that this is a local preparation to be used as local_side handle.
        - For remote descriptors, it is set to the remote name, indicating
        that this is remote side preparation to be used for remote_side handle.
        - For loopback descriptors, it is set to local agent's name, indicating that
        this is for a loopback (local) transfer to be used for remote_side handle
        Preparation succeeds if there exists at least one backend that can handle all
        elements in the descriptor list.

        Args:
            agent_name: Name of the agent. It can be "NIXL_INIT_AGENT", local agent name, or remote agent name
            xfer_list: List of transfer descriptors, can be list of memory region tuples, tensors,
                Nx3 numpy array, or nixlXferDList. See get_xfer_descs for more details on the structure.
                Nx5 numpy array is taken as a compressed (strided) list, where `count` blocks of `len`
                bytes are spaced `stride` bytes apart (stride == len is dense), with start address
                of `address`, in the form (address, len, device ID, stride, count).
            mem_type: Optional memory type necessary for list of memory regions, mandatory for Nx5 arrays.
            backends: Optional list of backend names to limit which backends are used during preparation

        Returns:
            Opaque handle to the prepared transfer descriptor list.
        """
        is_strided = (
            isinstance(xfer_list, np.ndarray)
            and xfer_list.ndim == 2
            and xfer_list.shape[1] == 5
        )

        is_local = agent_name == "NIXL_INIT_AGENT" or agent_name == ""
        if is_local:
            agent_name = nixlBind.NIXL_INIT_AGENT

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        if is_strided:
            if mem_type is None:
                raise ValueError("Please specify a mem type for strided descriptors")
            # An Nx5 array carries no memory type, so it is passed separately
            handle = self.agent.prepXferDlist(
                agent_name, self.nixl_mems[mem_type], xfer_list, handle_list
            )
        else:
            descs = self.get_xfer_descs(xfer_list, mem_type)
            handle = self.agent.prepXferDlist(agent_name, descs, handle_list)
        return nixl_prepped_dlist_handle(self.agent, handle)

    def estimate_xfer_cost(self, req_handle: nixl_xfer_handle) -> tuple[int, int, int]:
        """Estimate the cost of a transfer operation.
        Times are in microseconds and the method indicates how the estimation was performed.

        Args:
            req_handle: Handle to the transfer operation.

        Returns:
            Tuple of duration, error margin, method
        """
        duration, err_margin, method = self.agent.estimateXferCost(req_handle._handle)
        if method == nixlBind.NIXL_COST_ANALYTICAL_BACKEND:
            method = "ANALYTICAL_BACKEND"
        else:
            method = "UNKNOWN"
        return duration, err_margin, method

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: nixl_prepped_dlist_handle,
        local_indices: Union[list[int], np.ndarray],
        remote_xfer_side: nixl_prepped_dlist_handle,
        remote_indices: Union[list[int], np.ndarray],
        notif_msg: bytes = b"",
        backends: list[str] = [],
        skip_desc_merge: bool = False,
    ) -> nixl_xfer_handle:
        """Prepare a transfer operation using prep_xfer_dlist handles.

        Args:
            operation: Type of operation ("WRITE" or "READ").
            local_xfer_side: Handle to the local transfer descriptor list,
                received from prep_xfer_dlist.
            local_indices: List or numpy array (dtype=int32) of indices for selecting local descriptors.
            remote_xfer_side: Handle to the remote (or loopback) transfer descriptor list,
                received from prep_xfer_dlist.
            remote_indices: List or numpy array (dtype=int32) of indices for selecting remote descriptors.
            notif_msg: Optional notification message to send after transfer is done.
                notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
            backends: Optional list of backend names to limit which backends NIXL can use.
            skip_desc_merge: Deprecated: Whether to skip descriptor merging optimization.

        Returns:
            Opaque handle for posting/checking transfer.
                The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
        """
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.makeXferReq(
            op,
            local_xfer_side._handle,
            local_indices,
            remote_xfer_side._handle,
            remote_indices,
            notif_msg,
            handle_list,
            skip_desc_merge,
        )

        return nixl_xfer_handle(self.agent, handle)

    def initialize_xfer(
        self,
        operation: str,
        local_descs: nixlBind.nixlXferDList,
        remote_descs: nixlBind.nixlXferDList,
        remote_agent: str,
        notif_msg: bytes = b"",
        backends: list[str] = [],
    ) -> nixl_xfer_handle:
        """Initialize a transfer operation. This is a combined API, to create a transfer request
        from two descriptor lists, where NIXL prepares the descriptor lists and then the transfer.
        If there are common descriptors across different transfer requests, using
        this combined API will result in repeated computation, such as validity checks and
        pre-processing done in the preparation step.

        Args:
            operation: Type of operation ("WRITE" or "READ").
            local_descs: List of local transfer descriptors, from get_xfer_descs.
            remote_descs: List of remote (or loopback) transfer descriptors, from get_xfer_descs.
            remote_agent: Name of the remote agent.
            notif_msg: Optional notification message.
                notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
            backends: Optional list of backend names to limit which backends NIXL can use.

        Returns:
            Opaque handle for posting/checking transfer.
                The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
        """
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.createXferReq(
            op, local_descs, remote_descs, remote_agent, notif_msg, handle_list
        )

        return nixl_xfer_handle(self.agent, handle)

    def transfer(self, handle: nixl_xfer_handle, notif_msg: bytes = b"") -> str:
        """Initiate a data transfer operation.
        After calling this, the transfer state can be checked asynchronously till completion.
        In case of small transfers that are completed as part of the call itself, return value
        will be "DONE", otherwise "PROC" or "ERR".

        Args:
            handle: Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.
            notif_msg: Optional notification message can be specified or updated per transfer call.
                notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.

        Returns:
            Status of the transfer operation ("DONE", "PROC", or "ERR").
        """
        status = self.agent.postXferReq(handle._handle, notif_msg)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def check_xfer_state(self, handle: nixl_xfer_handle) -> str:
        """Check the state of a transfer operation.

        Args:
            handle: Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.

        Returns:
            Status of the transfer operation ("DONE", "PROC", or "ERR").
        """
        status = self.agent.getXferStatus(handle._handle)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    def get_xfer_telemetry(
        self, handle: nixl_xfer_handle
    ) -> nixlBind.nixlXferTelemetry:
        """Get telemetry information of a transfer request.
        The output object has three time values fields in microseconds
        (startTime, postDuration, xferDuration), as well as integer totalBytes transferred
        for the request, and integer descCount representing number of descriptors involved
        (for example if there was some merging of descriptors).

        Args:
            handle: Handle to the transfer operation, from make_prepped_xfer or initialize_xfer.

        Returns:
            nixlXferTelemetry object
        """
        return self.agent.getXferTelemetry(handle._handle)

    def query_xfer_backend(self, handle: nixl_xfer_handle) -> str:
        """Query the backend that was chosen for a transfer operation.

        Args:
            handle: Handle to the transfer operation.

        Returns:
            Name of the backend decided for the transfer.
        """
        b_handle = self.agent.queryXferBackend(handle._handle)
        # this works because there should not be multiple matching handles in the Dict
        return next(
            backendS
            for backendS, backendH in self.backends.items()
            if backendH == b_handle
        )

    def release_xfer_handle(self, handle: nixl_xfer_handle):
        """Releases a transfer handle, which internally frees the memory used for the handle.
        If the transfer is active, NIXL will attempt to cancel it.
        If it cannot be canceled, an error will be returned and the handle will not be freed.

        Args:
            handle: Handle to the transfer operation from initialize_xfer or make_xfer.
        """
        handle.release()

    def release_dlist_handle(self, handle: nixl_prepped_dlist_handle):
        """Release a descriptor list handle, which internally frees the memory used for the handle.

        Args:
            handle: Handle to the descriptor list from make_prepped_dlist.
        """
        handle.release()

    def prep_mem_view(
        self,
        dlist: Union[nixlBind.nixlXferDList, nixlBind.nixlRemoteDList],
        *,
        backends: Optional[list[str]] = None,
        worker_id: Optional[int] = None,
        connection_timeout_ms: Optional[int] = None,
    ) -> int:
        """Prepare a raw local or remote device memory-view handle.

        This low-level compatibility API does not retain buffers or protect
        registrations. New code should use :meth:`prepare_device_view`.
        ``connection_timeout_ms`` applies only to a remote descriptor list.
        """
        handle_list = []
        for backend_string in backends or []:
            handle_list.append(self.backends[backend_string])

        if isinstance(dlist, nixlBind.nixlRemoteDList):
            return self.agent.prepMemView(
                dlist, handle_list, worker_id, connection_timeout_ms
            )
        return self.agent.prepMemView(dlist, handle_list, worker_id)

    def release_mem_view(self, mvh: int) -> None:
        """Release a memory-view handle.

        If ``mvh`` belongs to an owning :meth:`prepare_device_view` result,
        route release through that object so its lifetime guard is invalidated
        atomically with the native handle. Raw handles returned by
        :meth:`prep_mem_view` retain the compatibility behavior.
        """
        with self._device_view_lock:
            for view in tuple(self._device_views):
                if view.valid and view._handle == mvh:
                    view.release()
                    return
            self.agent.releaseMemView(mvh)

    def prepare_device_view(
        self,
        xfer_list,
        remote_agent: Optional[str] = None,
        mem_type: Optional[str] = None,
        backend: str = "UCX",
        worker_id: Optional[int] = None,
        connection_timeout_ms: Optional[int] = 30000,
    ) -> nixl_device_view_handle:
        """Prepare an owning VRAM view for use by NIXL's CuTe device API.

        Version 3 supports only the UCX backend and VRAM descriptors. For a
        single peer, ``remote_agent`` applies that name to every descriptor.
        A ``nixlRemoteDList`` or list of ``(address, length, device_id,
        agent_name)`` tuples may instead name a different peer per descriptor,
        allowing one persistent kernel to index a composite multi-peer view.
        Every composite descriptor must name an addressable agent. Current UCX
        GPU device lists do not safely support ``NULL_AGENT`` gaps: an all-gap
        list cannot be created and transport operations on a gap are invalid.
        Fixed-capacity applications should pad inactive slots with a registered
        local loopback descriptor and guard them in device control flow.
        Metadata for every non-local peer must already be loaded. The returned
        handle retains the original input objects and converted descriptors.

        The caller must synchronize every CUDA kernel using the view before
        calling :meth:`nixl_device_view_handle.release`. For remote views,
        ``connection_timeout_ms`` bounds endpoint wire-up by default. Set it to
        ``None`` only when an unbounded wait is explicitly intended.
        """
        if backend != "UCX":
            raise ValueError(
                f"NIXL CuTe device views support only backend='UCX' (got {backend!r})"
            )
        if backend not in self.backends:
            raise RuntimeError(
                "UCX backend must be initialized before preparing a device view"
            )
        # getBackendParams() exposes canonical NIXL enum names (for example,
        # "VRAM_SEG"), not the shorter public Python input aliases.
        if "VRAM_SEG" not in self.backend_mems.get(backend, ()):
            raise RuntimeError("The initialized UCX backend does not support VRAM")
        if not getattr(nixlBind, "HAVE_UCX_GPU_DEVICE_API", False):
            raise RuntimeError(
                "NIXL was built without the UCX GPU device API required by CuTe views"
            )
        if mem_type not in (None, "VRAM"):
            raise ValueError(
                f"NIXL CuTe device views require mem_type='VRAM' (got {mem_type!r})"
            )
        if isinstance(worker_id, bool) or (
            worker_id is not None
            and (
                not isinstance(worker_id, int)
                or worker_id < 0
                or worker_id > sys.maxsize
            )
        ):
            raise ValueError(
                "worker_id must be a non-negative platform-sized integer or None"
            )
        if isinstance(connection_timeout_ms, bool) or (
            connection_timeout_ms is not None
            and (
                not isinstance(connection_timeout_ms, int)
                or connection_timeout_ms <= 0
                or connection_timeout_ms > sys.maxsize
            )
        ):
            raise ValueError(
                "connection_timeout_ms must be a positive platform-sized integer or None"
            )
        is_remote_coordinates = (
            isinstance(xfer_list, (list, tuple))
            and bool(xfer_list)
            and isinstance(xfer_list[0], tuple)
            and len(xfer_list[0]) == 4
        )
        is_explicit_remote = (
            isinstance(xfer_list, nixlBind.nixlRemoteDList) or is_remote_coordinates
        )
        if is_explicit_remote and remote_agent is not None:
            raise ValueError(
                "remote_agent must be None when xfer_list is a nixlRemoteDList"
            )
        if remote_agent is not None and (
            not isinstance(remote_agent, str) or not remote_agent
        ):
            raise ValueError("remote_agent must be a non-empty string or None")
        if remote_agent == NIXL_NULL_AGENT:
            raise ValueError(
                "remote_agent cannot be NULL_AGENT; use a registered local "
                "loopback descriptor for an inactive slot"
            )
        if xfer_list is None:
            raise TypeError("xfer_list must not be None")
        try:
            if len(xfer_list) == 0:
                raise ValueError("xfer_list must contain at least one descriptor")
        except TypeError:
            # Native descriptor lists do not expose __len__; isEmpty below is
            # authoritative after normalization.
            pass

        if is_remote_coordinates:
            normalized_remote = []
            for descriptor in xfer_list:
                if not isinstance(descriptor, tuple) or len(descriptor) != 4:
                    raise ValueError(
                        "every composite remote descriptor must be an "
                        "(address, length, device_id, agent_name) tuple"
                    )
                address, length, device_id, agent_name = descriptor
                if agent_name is None:
                    raise ValueError(
                        "composite remote descriptors cannot use None/NULL_AGENT; "
                        "pad inactive slots with a registered local loopback descriptor"
                    )
                elif not isinstance(agent_name, str) or not agent_name:
                    raise ValueError(
                        "composite remote agent names must be non-empty strings "
                        "or None"
                    )
                normalized_remote.append((address, length, device_id, agent_name))
            descs = self.get_remote_descs(normalized_remote, mem_type)
        elif is_explicit_remote:
            descs = xfer_list
        else:
            descs = self.get_xfer_descs(xfer_list, mem_type)
        if descs is None:
            raise ValueError("Could not convert xfer_list to NIXL transfer descriptors")
        if descs.isEmpty():
            raise ValueError("xfer_list must contain at least one descriptor")
        if descs.getType() != nixlBind.VRAM_SEG:
            raise ValueError("NIXL CuTe device views require VRAM descriptors")

        regions = []
        local_regions = []
        remote_agents: set[str] = set()
        for i in range(descs.descCount()):
            descriptor = descs[i]
            addr, length, device_id = descriptor[:3]
            if int(length) <= 0:
                raise ValueError(f"descriptor {i} must have a positive length")
            descriptor_agent = remote_agent
            if is_explicit_remote:
                descriptor_agent = descriptor[3]
                if not isinstance(descriptor_agent, str) or not descriptor_agent:
                    raise ValueError(
                        f"remote descriptor {i} must name a non-empty agent"
                    )
                if descriptor_agent == NIXL_NULL_AGENT:
                    raise ValueError(
                        "composite remote descriptors cannot use NULL_AGENT; "
                        "pad inactive slots with a registered local loopback descriptor"
                    )
                remote_agents.add(descriptor_agent)
            start = int(addr)
            region = (descs.getType(), int(device_id), start, start + int(length))
            regions.append(region)
            if descriptor_agent is None or descriptor_agent == self.name:
                local_regions.append(region)

        if remote_agent is not None:
            remote_agents.add(remote_agent)

        backend_handles = [self.backends[backend]]
        kind = "remote" if is_explicit_remote or remote_agents else "local"
        keepalive: tuple[object, ...]
        if isinstance(xfer_list, (list, tuple)):
            # Retain a snapshot of container members as well as the container
            # itself. A caller clearing a list of tensors must not free their
            # storage while a device view still references it.
            keepalive = (xfer_list, tuple(xfer_list), descs)
        else:
            keepalive = (xfer_list, descs)
        with self._device_view_lock:
            if is_explicit_remote:
                raw_handle = self.agent.prepMemView(
                    descs,
                    backend_handles,
                    worker_id,
                    connection_timeout_ms,
                )
            elif remote_agent is None:
                raw_handle = self.agent.prepLocalMemView(
                    descs, backend_handles, worker_id
                )
            else:
                raw_handle = self.agent.prepRemoteMemView(
                    remote_agent,
                    descs,
                    backend_handles,
                    worker_id,
                    connection_timeout_ms,
                )

            view = nixl_device_view_handle(
                self,
                raw_handle,
                kind=kind,
                count=descs.descCount(),
                remote_agent=remote_agent,
                remote_agents=frozenset(remote_agents),
                regions=tuple(regions),
                local_regions=tuple(local_regions),
                keepalive=keepalive,
            )
            self._device_views.add(view)
            return view

    def get_new_notifs(self, backends: list[str] = []) -> dict[str, list[bytes]]:
        """Get new notifications that have come to the agent.

        Args:
            backends: Optional list of backend names to limit which backends are checked for notifications.

        Returns:
            Dictionary of new notifications.
                Return Dict is a map of remote agent names to a list of notification messages from that agent.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        return self.agent.getNotifs({}, handle_list)

    def update_notifs(self, backends: list[str] = []) -> dict[str, list[bytes]]:
        """Update notifications in a map
        Same as get_new_notifs, but returns all unhandled notifications in agent.

        Args:
            backends: Optional list of backend names to limit which backends are checked for notifications.

        Returns:
            Dictionary of updated notifications.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.notifs = self.agent.getNotifs(self.notifs, handle_list)  # Adds new notifs
        return self.notifs

    def check_remote_xfer_done(
        self,
        remote_agent_name: str,
        lookup_tag: bytes,
        backends: list[str] = [],
        tag_is_prefix=True,
    ) -> bool:
        """Check if a remote transfer is done with a specific notification.
        Will only remove the notification that is found.

        Args:
            remote_agent_name: Name of the remote agent.
            lookup_tag: A tag to match against available messages in the notification map.
                The tag Can be the same as the entire expected message.
            backends: Optional list of backend names to limit which backends are checked for notifications.
            tag_is_prefix: Optionally specify that the tag you want to search with is just a prefix, or can be search as a substring of the full message.

        Returns:
            True if the notification is found, False otherwise.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.notifs = self.agent.getNotifs(self.notifs, handle_list)  # Adds new notifs
        found = False
        message = None

        if remote_agent_name in self.notifs:
            for msg in self.notifs[remote_agent_name]:
                if (tag_is_prefix and msg.startswith(lookup_tag)) or (
                    not tag_is_prefix and lookup_tag in msg
                ):
                    message = msg
                    found = True
                    break
        if message:
            self.notifs[remote_agent_name].remove(message)
        return found

    def send_notif(
        self, remote_agent_name: str, notif_msg: bytes, backend: Optional[str] = None
    ):
        """Send a standalone notification to a remote agent, not bound to a transfer.

        Args:
            remote_agent_name: Name of the remote agent.
            notif_msg: Message to send, it will be received as bytes.
                notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
            backend: Optional backend name to use to send the notification.
        """
        if backend is None:
            self.agent.genNotif(remote_agent_name, notif_msg)
        else:
            self.agent.genNotif(remote_agent_name, notif_msg, [self.backends[backend]])

    def get_agent_metadata(self) -> bytes:
        """Get the full metadata of the local agent.

        Returns:
            Metadata of the local agent, in bytes.
        """
        return self.agent.getLocalMD()

    def get_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
    ) -> bytes:
        """Get partial metadata of the local agent.

        Args:
            descs: The list of descriptors to include metadata about.
                List can be empty if only trying to send connection info.
            inc_conn_info: Whether to include connection info in the metadata.
            backends: List of backends to consider when constructing partial metadata.

        Returns:
            Metadata of the local agent, in bytes.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        return self.agent.getLocalPartialMD(descs, inc_conn_info, handle_list)

    def add_remote_agent(self, metadata: bytes) -> str:
        """Add a remote agent using its metadata. After this call, current agent can
        initiate transfers towards the remote agent.

        Args:
            metadata: Metadata of the remote agent, received out-of-band in bytes.

        Returns:
            Name of the added remote agent.
        """
        agent_name = self.agent.loadRemoteMD(metadata)
        return agent_name

    def remove_remote_agent(self, agent: str):
        """Remove a remote agent. After this call, current agent cannot initiate
        transfers towards the remote agent specified in the call anymore.
        This call will also result in a disconnect between the two agents.

        Args:
            agent: Name of the remote agent.
        """
        with self._device_view_lock:
            for view in tuple(self._device_views):
                if (
                    view.valid
                    and view.kind == "remote"
                    and agent in view._remote_agents
                ):
                    raise RuntimeError(
                        f"Cannot remove remote agent {agent!r} while a device view "
                        "references it; synchronize CUDA work and release the view first"
                    )
            self.agent.invalidateRemoteMD(agent)

    def send_local_metadata(self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT):
        """Send all of your metadata to a peer or central metadata server.

        Args:
            ip_addr: If specified, will only send metadata to one peer by IP address.
                Otherwise, metadata will be sent to central metadata server, if supported.
            port: If specified next to ip_addr, will try to send to this specific port of a peer.
                Ignored when sending to a central metadata server.
        """
        self.agent.sendLocalMD(ip_addr, port)

    def send_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        """Send partial metadata of the local agent to a peer or central metadata server.

        Args:
            descs: The list of descriptors to include metadata about.
                List can be empty if only trying to send connection info.
            inc_conn_info: Whether to include connection info in the metadata.
            backends: List of backends to consider when constructing partial metadata.
            ip_addr: If specified, will only send metadata to one peer by IP address.
                Otherwise, metadata will be sent to central metadata server, if supported.
            port: If specified next to ip_addr, will try to send to this specific port of a peer.
                Ignored when sending to a central metadata server.
            label: Label to use for the metadata when sending to central metadata server.
                Ignored when sending to a peer.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.sendLocalPartialMD(
            descs, inc_conn_info, handle_list, ip_addr, port, label
        )

    def fetch_remote_metadata(
        self,
        remote_agent: str,
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        """Request metadata be retrieved from central metadata server or sent by peer.

        Args:
            ip_addr: If specified, will request metadata from one peer by IP address.
            port: If specified, will try to request on specific port.
        """
        self.agent.fetchRemoteMD(remote_agent, ip_addr, port, label)

    def invalidate_local_metadata(
        self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT
    ):
        """Invalidate your own metadata in the central metadata server, or from a specific peer.

        Args:
            ip_addr: If specified, will only send invalidation to one peer by IP address.
            port: If specified, will try to send to specific port.
        """
        self.agent.invalidateLocalMD(ip_addr, port)

    def check_remote_metadata(
        self, agent: str, descs: nixlBind.nixlXferDList = None
    ) -> bool:
        """Check if the remote metadata for a specific agent is available.
        When partial metadata methods are used, the descriptor list in question can be specified.

        Args:
            agent: Name of the remote agent.

        Returns:
            True if available, False otherwise
        """
        if descs is None:  # Just empty list, mem_type not important
            descs = nixlBind.nixlXferDList(nixlBind.DRAM_SEG)
        if self.agent.checkRemoteMD(agent, descs) == nixlBind.NIXL_SUCCESS:
            return True
        else:
            return False

    @staticmethod
    def _tensor_mem_type(tensor: torch.Tensor) -> str:
        return "DRAM" if tensor.get_device() == -1 else "VRAM"

    def get_xfer_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlXferDList:
        # can add check for DLPack input

        """Get nixlXferDList from different input types:
        a) list of 3 element tuples (address, len, device ID) alongside a mandatory memory type
        b) a tensor
        c) a list of tensors
        d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
        alongside a mandatory memory type
        e) passes along if an xfer_dlist is given.

        Args:
            descs: List of any of the above types
            mem_type: Optional memory type necessary for (a).

        Returns:
            Transfer descriptor list, nixlXferDList.
        """
        if isinstance(descs, nixlBind.nixlXferDList):
            return descs
        elif isinstance(descs, nixlBind.nixlRegDList):
            logger.error("RegList type detected for transfer, please use XferList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("3-tuple list needed for transfer")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif _is_torch_tensor(descs):
            if descs.is_contiguous():
                mem_type = self._tensor_mem_type(descs)
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlXferDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id)]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif _is_torch_tensor(descs[0]):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = self._tensor_mem_type(descs[0])
            new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    """
    @brief Get nixlRemoteDList from different input types:
            a) list of 4 element tuples (address, len, device ID, remote agent name)
               alongside a mandatory memory type.
            b) passes along if a nixlRemoteDList is given.

    @param descs List of any of the above types.
    @param mem_type Optional memory type necessary for (a).
    @return Remote descriptor list, nixlRemoteDList.
    """

    def get_remote_descs(
        self,
        descs: Union[nixlBind.nixlRemoteDList, list[tuple]],
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlRemoteDList:
        if isinstance(descs, nixlBind.nixlRemoteDList):
            return descs
        elif isinstance(descs[0], tuple):
            if (mem_type is not None) and (len(descs[0]) == 4):
                new_descs = nixlBind.nixlRemoteDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type")
                new_descs = None
            else:
                logger.error(
                    "4-tuple list (addr, len, dev_id, agent_name) needed for remote descriptors"
                )
                new_descs = None
        else:
            new_descs = None

        return new_descs

    def get_reg_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlRegDList:
        # can add check for DLPack input

        """Get nixlRegDList from different input types:
        a) list of 4 element tuples (address, len, device ID, meta info) alongside a mandatory memory type
        b) a tensor
        c) a list of tensors
        d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
        alongside a mandatory memory type. Empty meta info will be considered for each descriptor.
        e) passes along if a reg_dlist is given.

        Args:
            descs: List of any of the above types
            mem_type: Optional memory type necessary for (a).

        Returns:
            Registration descriptor list, nixlRegDList.
        """
        if isinstance(descs, nixlBind.nixlRegDList):
            return descs
        elif isinstance(descs, nixlBind.nixlXferDList):
            logger.error("XferList type detected for registration, please use RegList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("4-tuple list needed for registration")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif _is_torch_tensor(descs):
            if descs.is_contiguous():
                mem_type = self._tensor_mem_type(descs)
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlRegDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id, "")]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif _is_torch_tensor(descs[0]):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = self._tensor_mem_type(descs[0])
            new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    def get_serialized_descs(self, descs) -> bytes:
        """Serialize NIXL descriptor list with pickle.

        Args:
            descs: NIXL list to serialize.

        Returns:
            Serialized descriptor list.
        """
        return pickle.dumps(descs)

    def deserialize_descs(self, serialized_descs: bytes):
        """Deserialize NIXL descriptor list.

        Args:
            serialized_descs: Serialized NIXL descriptor list.

        Returns:
            Deserialized NIXL descriptor list.
        """
        return pickle.loads(serialized_descs)
