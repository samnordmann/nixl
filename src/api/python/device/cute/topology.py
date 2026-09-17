# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side topology qualification for NIXL CuTe mapped protocols.

The mapped CuTe examples publish and consume system-scope atomics in memory
owned by peer GPUs. CUDA requires native peer atomics in every direction that
the protocol uses. These helpers perform that one-time qualification before any
persistent kernel is launched; they never execute in a device hot path.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


_CAPABILITY = "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED"


def _validated_devices(devices: Sequence[int]) -> tuple[int, ...]:
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise TypeError("devices must be a sequence of CUDA device ordinals")
    normalized: list[int] = []
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device ordinals must be integers, not booleans")
        if device < 0:
            raise ValueError("CUDA device ordinals must be nonnegative")
        normalized.append(device)
    if not normalized:
        raise ValueError("devices must contain at least one CUDA device ordinal")
    if len(set(normalized)) != len(normalized):
        raise ValueError("devices must contain distinct CUDA device ordinals")
    return tuple(normalized)


def _validated_accessing_devices(
    accessing_devices: Sequence[int], devices: tuple[int, ...]
) -> tuple[int, ...]:
    if isinstance(accessing_devices, (str, bytes)) or not isinstance(
        accessing_devices, Sequence
    ):
        raise TypeError("accessing_devices must be a sequence of CUDA device ordinals")
    normalized: list[int] = []
    for device in accessing_devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError(
                "accessing_devices CUDA ordinals must be integers, not booleans"
            )
        if device < 0:
            raise ValueError("accessing_devices CUDA ordinals must be nonnegative")
        normalized.append(device)
    if not normalized:
        raise ValueError(
            "accessing_devices must contain at least one CUDA device ordinal"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("accessing_devices must contain distinct CUDA device ordinals")
    unknown = [device for device in normalized if device not in devices]
    if unknown:
        raise ValueError(
            "accessing_devices must be a subset of devices; unknown CUDA ordinals: "
            f"{unknown}"
        )
    return tuple(normalized)


def _load_cuda_driver() -> Any:
    """Load CUDA Python only when a mapped protocol requests qualification."""

    try:
        from cuda.bindings import driver  # type: ignore
    except Exception as error:  # pragma: no cover - exercised by installed wheels
        raise RuntimeError(
            "NIXL CuTe topology qualification requires cuda.bindings.driver; "
            "install the supported NIXL CuTe extra or cuda-python"
        ) from error
    return driver


def _checked_cuda(driver: Any, operation: str, result: object) -> Any:
    """Unwrap one CUDA Python Driver API result or raise with context."""

    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned a malformed CUDA Python result")
    status = result[0]
    try:
        success = driver.CUresult.CUDA_SUCCESS
    except AttributeError as error:
        raise RuntimeError(
            "cuda.bindings.driver has no CUresult.CUDA_SUCCESS"
        ) from error
    if status != success:
        error_name = repr(status)
        try:
            name_result = driver.cuGetErrorName(status)
            if (
                isinstance(name_result, tuple)
                and len(name_result) == 2
                and name_result[0] == success
            ):
                raw_name = name_result[1]
                error_name = (
                    raw_name.decode(errors="replace")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
        except Exception:
            pass
        raise RuntimeError(f"{operation} failed with {error_name}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _decoded_cuda_text(value: object, operation: str) -> str:
    if isinstance(value, bytes):
        decoded = value.split(b"\0", 1)[0].decode("ascii", errors="strict")
    elif isinstance(value, str):
        decoded = value.split("\0", 1)[0]
    else:
        raise RuntimeError(f"{operation} returned a non-text value")
    if not decoded:
        raise RuntimeError(f"{operation} returned an empty value")
    return decoded


def _query_with_driver(
    driver: Any,
    devices: tuple[int, ...],
    accessing_devices: tuple[int, ...],
) -> dict[str, object]:
    """Query selected directional rows after resolving every device exactly once."""

    _checked_cuda(driver, "cuInit", driver.cuInit(0))
    try:
        attribute = (
            driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
        )
    except AttributeError as error:
        raise RuntimeError(
            "cuda.bindings.driver does not expose the native peer-atomic attribute"
        ) from error

    handles: dict[int, object] = {}
    identities: list[dict[str, object]] = []
    for ordinal in devices:
        handle = _checked_cuda(
            driver, f"cuDeviceGet({ordinal})", driver.cuDeviceGet(ordinal)
        )
        handles[ordinal] = handle
        bus_id = _decoded_cuda_text(
            _checked_cuda(
                driver,
                f"cuDeviceGetPCIBusId({ordinal})",
                driver.cuDeviceGetPCIBusId(32, handle),
            ),
            f"cuDeviceGetPCIBusId({ordinal})",
        )
        identities.append({"device_ordinal": ordinal, "pci_bus_id": bus_id})

    ordered_pairs: list[dict[str, object]] = []
    for accessing_device in accessing_devices:
        for owner_device in devices:
            if accessing_device == owner_device:
                continue
            operation = (
                "cuDeviceGetP2PAttribute(native atomic supported, "
                f"accessing={accessing_device}, owner={owner_device})"
            )
            raw_supported = _checked_cuda(
                driver,
                operation,
                driver.cuDeviceGetP2PAttribute(
                    attribute,
                    handles[accessing_device],
                    handles[owner_device],
                ),
            )
            if isinstance(raw_supported, bool) or not isinstance(raw_supported, int):
                raise RuntimeError(f"{operation} returned a non-integer value")
            supported_value = raw_supported
            if supported_value not in (0, 1):
                raise RuntimeError(
                    f"{operation} returned {supported_value}, expected 0 or 1"
                )
            ordered_pairs.append(
                {
                    "accessing_device": accessing_device,
                    "owner_device": owner_device,
                    "native_atomics_supported": supported_value == 1,
                }
            )

    evidence: dict[str, object] = {
        "schema_version": 1,
        "capability": _CAPABILITY,
        "devices": list(devices),
        "accessing_devices": list(accessing_devices),
        "device_identities": identities,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ordered_pairs": ordered_pairs,
        "all_supported": all(
            pair["native_atomics_supported"] is True for pair in ordered_pairs
        ),
        "query_scope": (
            "selected accessing devices to every distinct participating owner device"
        ),
        "execution_scope": "host preflight only; no steady-state device-path cost",
        "driver_module": str(getattr(driver, "__file__", "<unknown>")),
    }
    return evidence


def query_peer_native_atomics(
    devices: Sequence[int],
    *,
    accessing_devices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return JSON-friendly native-peer-atomic evidence for ``devices``.

    CUDA's peer capability is directional, so every ordered pair is queried.
    ``accessing_devices`` may select a non-empty subset of directional rows;
    every selected accessor is still checked against every distinct owner in
    ``devices``. Self-pairs are deliberately omitted: the prerequisite concerns
    one GPU accessing another GPU's allocation. Omitting ``accessing_devices``
    preserves the original full-matrix query behavior by selecting every device.
    """

    normalized = _validated_devices(devices)
    normalized_accessors = (
        normalized
        if accessing_devices is None
        else _validated_accessing_devices(accessing_devices, normalized)
    )
    try:
        driver = _load_cuda_driver()
        return _query_with_driver(
            driver,
            normalized,
            normalized_accessors,
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"CUDA native peer-atomic preflight failed: {error}"
        ) from error


def require_peer_native_atomics(
    devices: Sequence[int],
    *,
    accessing_devices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Require native peer atomics and return the successful evidence record.

    Raises:
        RuntimeError: if CUDA cannot query a pair or any directed pair does not
            support native peer atomics.
        TypeError, ValueError: if ``devices`` or ``accessing_devices`` is malformed.
    """

    evidence = query_peer_native_atomics(devices, accessing_devices=accessing_devices)
    ordered_pairs = evidence.get("ordered_pairs")
    if not isinstance(ordered_pairs, list):  # pragma: no cover - internal invariant
        raise RuntimeError("native peer-atomic query returned malformed evidence")
    unsupported: list[tuple[int, int]] = []
    for pair in ordered_pairs:
        if not isinstance(pair, dict):  # pragma: no cover - internal invariant
            raise RuntimeError("native peer-atomic query returned a malformed pair")
        if pair.get("native_atomics_supported") is not True:
            unsupported.append(
                (int(pair["accessing_device"]), int(pair["owner_device"]))
            )
    if unsupported:
        formatted = ", ".join(
            f"accessing {accessing}->owner {owner}" for accessing, owner in unsupported
        )
        raise RuntimeError(
            "NIXL CuTe mapped protocols require CUDA native peer atomics for "
            f"every queried directed device pair; unsupported pairs: {formatted}"
        )
    return evidence


__all__ = ["query_peer_native_atomics", "require_peer_native_atomics"]
