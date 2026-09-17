# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public compile-time controls and status values for NIXL CuTe DSL."""

from enum import IntEnum, IntFlag


class Scope(IntEnum):
    """Threads that cooperatively execute one device operation."""

    THREAD = 0
    WARP = 1
    BLOCK = 2  # Reserved; unsafe in the pinned UCX device implementation.
    GRID = 3  # Reserved by NIXL; intentionally unsupported by this binding.


class Flags(IntFlag):
    """NIXL device operation flags.

    ``DEFER`` is accepted only by nonblocking post operations. A deferred
    operation needs a later non-deferred post on the same channel to ring the
    device transport doorbell; synchronous wait operations reject it.
    """

    NONE = 0
    DEFER = 1


class Status(IntEnum):
    """Values returned by the NIXL device ABI."""

    IN_PROGRESS = 1
    SUCCESS = 0
    NOT_POSTED = -1
    INVALID_PARAM = -2
    BACKEND = -3
    NOT_FOUND = -4
    MISMATCH = -5
    NOT_ALLOWED = -6
    REPOST_ACTIVE = -7
    UNKNOWN = -8
    NOT_SUPPORTED = -9
    REMOTE_DISCONNECT = -10
    CANCELED = -11
    NO_TELEMETRY = -12


NIXL_IN_PROG = Status.IN_PROGRESS
NIXL_SUCCESS = Status.SUCCESS
NIXL_ERR_NOT_POSTED = Status.NOT_POSTED
NIXL_ERR_INVALID_PARAM = Status.INVALID_PARAM
NIXL_ERR_BACKEND = Status.BACKEND
NIXL_ERR_NOT_FOUND = Status.NOT_FOUND
NIXL_ERR_MISMATCH = Status.MISMATCH
NIXL_ERR_NOT_ALLOWED = Status.NOT_ALLOWED
NIXL_ERR_REPOST_ACTIVE = Status.REPOST_ACTIVE
NIXL_ERR_UNKNOWN = Status.UNKNOWN
NIXL_ERR_NOT_SUPPORTED = Status.NOT_SUPPORTED
NIXL_ERR_REMOTE_DISCONNECT = Status.REMOTE_DISCONNECT
NIXL_ERR_CANCELED = Status.CANCELED
NIXL_ERR_NO_TELEMETRY = Status.NO_TELEMETRY


__all__ = [
    "Flags",
    "NIXL_ERR_BACKEND",
    "NIXL_ERR_CANCELED",
    "NIXL_ERR_INVALID_PARAM",
    "NIXL_ERR_MISMATCH",
    "NIXL_ERR_NOT_ALLOWED",
    "NIXL_ERR_NOT_FOUND",
    "NIXL_ERR_NOT_POSTED",
    "NIXL_ERR_NOT_SUPPORTED",
    "NIXL_ERR_NO_TELEMETRY",
    "NIXL_ERR_REMOTE_DISCONNECT",
    "NIXL_ERR_REPOST_ACTIVE",
    "NIXL_ERR_UNKNOWN",
    "NIXL_IN_PROG",
    "NIXL_SUCCESS",
    "Scope",
    "Status",
]
