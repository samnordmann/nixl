# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe native-struct mirrors for the future NIXL device ABI.

This module is intentionally small until NIXL owns a stable C wrapper ABI.  The
NCCL CuTe binding mirrors only small value types and keeps large internal
objects opaque; NIXL should follow that rule once ``libnixl_device.bc`` exists.
"""

from __future__ import annotations


MIRRORED_STRUCTS: tuple[str, ...] = ()


def has_mirrored_structs() -> bool:
    """Return whether generated CuTe struct mirrors are present."""

    return bool(MIRRORED_STRUCTS)


def require_mirrored_structs() -> None:
    """Raise until NIXL defines stable structs for the CuTe FFI layer."""

    raise NotImplementedError(
        "NIXL CuTe native-struct mirrors have not been generated yet. "
        "Define a stable C device ABI first, then mirror only the small value "
        "types that must cross the cute.ffi boundary by value."
    )
