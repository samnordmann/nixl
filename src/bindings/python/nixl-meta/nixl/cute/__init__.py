# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe-DSL helpers for NIXL.

This package is intentionally small for the first prototype.  The host runtime
wraps the existing Python NIXL agent API; device-side CuTe calls are capability
gated until a supported CuTe external-call path for ``nixl_device.cuh`` is
validated.
"""

from .device import (
    DeviceApiStatus,
    device_api_status,
    is_device_api_available,
)
from .runtime import Agent, PreparedView, RegisteredTensor, RemoteAgent

__all__ = [
    "Agent",
    "DeviceApiStatus",
    "PreparedView",
    "RegisteredTensor",
    "RemoteAgent",
    "device_api_status",
    "is_device_api_available",
]

