# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical FFI binding layer for NIXL CuTe device calls.

The NCCL CuTe binding uses this layer for one ``cute.ffi`` prototype per C ABI
symbol.  NIXL does not yet ship ``libnixl_device.bc`` or wrapper symbols, so the
public device functions fail explicitly instead of pretending to be available.
"""

from __future__ import annotations

from ._helpers import device_bitcode_status
from ._structs import has_mirrored_structs


def has_device_bindings() -> bool:
    """Return whether generated bitcode-backed FFI bindings are available."""

    return False


def _not_implemented(name: str) -> None:
    bitcode = device_bitcode_status()
    structs = "present" if has_mirrored_structs() else "missing"
    raise NotImplementedError(
        f"nixl.cute.device.{name} is scaffold-only. "
        "NIXL still needs a stable C device wrapper ABI, generated cute.ffi "
        f"prototypes, and packaged libnixl_device.bc. Bitcode status: "
        f"{bitcode.reason}; struct mirrors: {structs}."
    )


def put_block(*args, **kwargs):
    """Future CTA/block-level NIXL device WRITE/PUT."""

    _not_implemented("put_block")


def atomic_add(*args, **kwargs):
    """Future NIXL device remote atomic add."""

    _not_implemented("atomic_add")


def get_ptr(*args, **kwargs):
    """Future wrapper for NIXL device pointer translation."""

    _not_implemented("get_ptr")
