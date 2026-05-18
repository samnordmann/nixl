# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe-DSL helpers for NIXL.

The current package is a production-shaped scaffold: host-side registration and
metadata helpers are available, while device-side CuTe calls are explicitly
gated until NIXL ships a bitcode-backed C wrapper ABI for ``cute.ffi``.
"""

from .device import DeviceApiStatus, device_api_status, is_device_api_available
from ._helpers import BitcodeStatus, device_bitcode_path, device_bitcode_status

_RUNTIME_EXPORTS = {
    "Agent",
    "LocalView",
    "PreparedViews",
    "RegisteredTensor",
    "RemoteAgent",
    "RemoteView",
}


def __getattr__(name: str):
    """Lazily import host-runtime helpers.

    Importing ``nixl.cute`` should be cheap enough for capability checks in
    minimal CUDA containers.  The runtime layer imports Torch and the NIXL
    Python extension, so defer it until callers request host-side objects.
    """

    if name in _RUNTIME_EXPORTS:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Agent",
    "BitcodeStatus",
    "DeviceApiStatus",
    "LocalView",
    "PreparedViews",
    "RegisteredTensor",
    "RemoteAgent",
    "RemoteView",
    "device_bitcode_path",
    "device_bitcode_status",
    "device_api_status",
    "is_device_api_available",
]

