# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[2]
_TOPOLOGY_SOURCE = _ROOT / "src" / "api" / "python" / "device" / "cute" / "topology.py"


def _load_topology_module():
    name = f"_nixl_test_cute_topology_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, _TOPOLOGY_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Result:
    CUDA_SUCCESS = 0


class _P2PAttribute:
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 8


class _FakeDriver:
    CUresult = _Result
    CUdevice_P2PAttribute = _P2PAttribute
    __file__ = "/cuda/bindings/driver.py"

    def __init__(self, *, unsupported=(), fault=None, supported_value=None):
        self.calls = []
        self.unsupported = set(unsupported)
        self.fault = fault
        self.supported_value = supported_value

    def _result(self, operation, *values):
        if self.fault == operation:
            return (17,)
        return (0, *values)

    def cuGetErrorName(self, status):
        assert status == 17
        return 0, b"CUDA_ERROR_INVALID_DEVICE"

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return self._result("cuInit")

    def cuDeviceGet(self, ordinal):
        self.calls.append(("cuDeviceGet", ordinal))
        return self._result("cuDeviceGet", ("device", ordinal))

    def cuDeviceGetPCIBusId(self, length, device):
        self.calls.append(("cuDeviceGetPCIBusId", length, device))
        ordinal = device[1]
        return self._result(
            "cuDeviceGetPCIBusId", f"00000000:{ordinal + 1:02x}:00.0".encode()
        )

    def cuDeviceGetP2PAttribute(self, attribute, accessing, owner):
        pair = (accessing[1], owner[1])
        self.calls.append(("cuDeviceGetP2PAttribute", attribute, *pair))
        value = (
            self.supported_value
            if self.supported_value is not None
            else int(pair not in self.unsupported)
        )
        return self._result("cuDeviceGetP2PAttribute", value)


def _use_driver(monkeypatch, topology, driver):
    monkeypatch.setattr(topology, "_load_cuda_driver", lambda: driver)


def test_query_peer_native_atomics_is_complete_batched_and_json_safe(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    evidence = topology.query_peer_native_atomics((2, 0, 5))

    pairs = [(2, 0), (2, 5), (0, 2), (0, 5), (5, 2), (5, 0)]
    assert driver.calls.count(("cuInit", 0)) == 1
    assert [call for call in driver.calls if call[0] == "cuDeviceGet"] == [
        ("cuDeviceGet", 2),
        ("cuDeviceGet", 0),
        ("cuDeviceGet", 5),
    ]
    assert [
        call[-2:] for call in driver.calls if call[0] == "cuDeviceGetP2PAttribute"
    ] == pairs
    assert evidence == {
        "schema_version": 1,
        "capability": "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED",
        "devices": [2, 0, 5],
        "accessing_devices": [2, 0, 5],
        "device_identities": [
            {"device_ordinal": 2, "pci_bus_id": "00000000:03:00.0"},
            {"device_ordinal": 0, "pci_bus_id": "00000000:01:00.0"},
            {"device_ordinal": 5, "pci_bus_id": "00000000:06:00.0"},
        ],
        "cuda_visible_devices": "GPU-a,GPU-b,GPU-c",
        "ordered_pairs": [
            {
                "accessing_device": accessing,
                "owner_device": owner,
                "native_atomics_supported": True,
            }
            for accessing, owner in pairs
        ],
        "all_supported": True,
        "query_scope": (
            "selected accessing devices to every distinct participating owner device"
        ),
        "execution_scope": "host preflight only; no steady-state device-path cost",
        "driver_module": "/cuda/bindings/driver.py",
    }
    assert json.loads(json.dumps(evidence)) == evidence


def test_query_peer_native_atomics_can_select_exact_directional_rows(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)

    evidence = topology.query_peer_native_atomics((2, 0, 5), accessing_devices=(0, 5))

    expected_pairs = [(0, 2), (0, 5), (5, 2), (5, 0)]
    assert driver.calls.count(("cuInit", 0)) == 1
    assert [call for call in driver.calls if call[0] == "cuDeviceGet"] == [
        ("cuDeviceGet", 2),
        ("cuDeviceGet", 0),
        ("cuDeviceGet", 5),
    ]
    assert [
        call[-2:] for call in driver.calls if call[0] == "cuDeviceGetP2PAttribute"
    ] == expected_pairs
    assert evidence["devices"] == [2, 0, 5]
    assert evidence["accessing_devices"] == [0, 5]
    assert evidence["device_identities"] == [
        {"device_ordinal": 2, "pci_bus_id": "00000000:03:00.0"},
        {"device_ordinal": 0, "pci_bus_id": "00000000:01:00.0"},
        {"device_ordinal": 5, "pci_bus_id": "00000000:06:00.0"},
    ]
    assert [
        (pair["accessing_device"], pair["owner_device"])
        for pair in evidence["ordered_pairs"]
    ] == expected_pairs
    assert evidence["query_scope"] == (
        "selected accessing devices to every distinct participating owner device"
    )
    assert evidence["all_supported"] is True
    assert json.loads(json.dumps(evidence)) == evidence


def test_explicit_all_accessor_rows_preserve_requested_order(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)

    evidence = topology.query_peer_native_atomics(
        (2, 0, 5), accessing_devices=(5, 2, 0)
    )

    assert evidence["accessing_devices"] == [5, 2, 0]
    assert [
        (pair["accessing_device"], pair["owner_device"])
        for pair in evidence["ordered_pairs"]
    ] == [(5, 2), (5, 0), (2, 0), (2, 5), (0, 2), (0, 5)]


def test_require_peer_native_atomics_returns_success_evidence(monkeypatch):
    topology = _load_topology_module()
    _use_driver(monkeypatch, topology, _FakeDriver())

    evidence = topology.require_peer_native_atomics((0, 1))

    assert evidence["all_supported"] is True
    assert len(evidence["ordered_pairs"]) == 2


def test_require_peer_native_atomics_limits_failure_to_selected_rows(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver(unsupported={(0, 2), (2, 1)})
    _use_driver(monkeypatch, topology, driver)

    evidence = topology.require_peer_native_atomics((0, 2, 1), accessing_devices=(1,))
    assert evidence["all_supported"] is True
    assert [
        (pair["accessing_device"], pair["owner_device"])
        for pair in evidence["ordered_pairs"]
    ] == [(1, 0), (1, 2)]

    with pytest.raises(
        RuntimeError,
        match="unsupported pairs: accessing 2->owner 1",
    ):
        topology.require_peer_native_atomics((0, 2, 1), accessing_devices=(2,))


def test_require_peer_native_atomics_fails_closed_with_all_unsupported_pairs(
    monkeypatch,
):
    topology = _load_topology_module()
    _use_driver(
        monkeypatch,
        topology,
        _FakeDriver(unsupported={(0, 2), (2, 1)}),
    )

    with pytest.raises(
        RuntimeError,
        match=("unsupported pairs: accessing 0->owner 2, accessing 2->owner 1"),
    ):
        topology.require_peer_native_atomics((0, 2, 1))


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("cuInit", "cuInit failed with CUDA_ERROR_INVALID_DEVICE"),
        ("cuDeviceGet", r"cuDeviceGet\(0\) failed"),
        ("cuDeviceGetPCIBusId", r"cuDeviceGetPCIBusId\(0\) failed"),
        (
            "cuDeviceGetP2PAttribute",
            "native atomic supported, accessing=0, owner=1",
        ),
    ],
)
def test_cuda_failures_are_contextualized(monkeypatch, fault, message):
    topology = _load_topology_module()
    _use_driver(monkeypatch, topology, _FakeDriver(fault=fault))

    with pytest.raises(RuntimeError, match=message):
        topology.query_peer_native_atomics((0, 1))


def test_missing_cuda_bindings_fails_closed(monkeypatch):
    topology = _load_topology_module()

    def missing():
        raise RuntimeError("cuda.bindings.driver is unavailable")

    monkeypatch.setattr(topology, "_load_cuda_driver", missing)
    with pytest.raises(RuntimeError, match="cuda.bindings.driver is unavailable"):
        topology.query_peer_native_atomics((0, 1))


def test_malformed_cuda_python_result_fails_closed(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    driver.cuInit = lambda _flags: 0
    _use_driver(monkeypatch, topology, driver)

    with pytest.raises(RuntimeError, match="malformed CUDA Python result"):
        topology.query_peer_native_atomics((0,))


def test_malformed_pci_identity_fails_closed(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    driver.cuDeviceGetPCIBusId = lambda _length, _device: (0, 42)
    _use_driver(monkeypatch, topology, driver)

    with pytest.raises(RuntimeError, match="returned a non-text value"):
        topology.query_peer_native_atomics((0,))


def test_out_of_domain_integer_capability_value_fails_closed(monkeypatch):
    topology = _load_topology_module()
    _use_driver(monkeypatch, topology, _FakeDriver(supported_value=2))

    with pytest.raises(RuntimeError, match="returned 2, expected 0 or 1"):
        topology.query_peer_native_atomics((0, 1))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_non_integer_capability_value_fails_closed(monkeypatch, value):
    topology = _load_topology_module()
    _use_driver(monkeypatch, topology, _FakeDriver(supported_value=value))

    with pytest.raises(RuntimeError, match="returned a non-integer value"):
        topology.query_peer_native_atomics((0, 1))


def test_single_device_needs_no_peer_query(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)

    evidence = topology.require_peer_native_atomics((7,))

    assert evidence["devices"] == [7]
    assert evidence["ordered_pairs"] == []
    assert evidence["all_supported"] is True
    assert not any(call[0] == "cuDeviceGetP2PAttribute" for call in driver.calls)


@pytest.mark.parametrize(
    ("devices", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ((0, 0), ValueError, "distinct"),
        ((-1,), ValueError, "nonnegative"),
        ((True,), TypeError, "not booleans"),
        ((0.0,), TypeError, "integers"),
        ("0,1", TypeError, "sequence"),
        ({0, 1}, TypeError, "sequence"),
    ],
)
def test_peer_native_atomic_device_validation(devices, error, message):
    topology = _load_topology_module()

    with pytest.raises(error, match=message):
        topology.query_peer_native_atomics(devices)


@pytest.mark.parametrize(
    ("accessing_devices", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ((0, 0), ValueError, "distinct"),
        ((-1,), ValueError, "nonnegative"),
        ((True,), TypeError, "not booleans"),
        ((0.0,), TypeError, "integers"),
        ("0,1", TypeError, "sequence"),
        ({0, 1}, TypeError, "sequence"),
        ((3,), ValueError, "subset of devices"),
    ],
)
def test_accessing_device_subset_validation(
    monkeypatch, accessing_devices, error, message
):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)

    with pytest.raises(error, match=message):
        topology.query_peer_native_atomics((0, 1), accessing_devices=accessing_devices)
    assert driver.calls == []


def test_accessing_devices_is_keyword_only(monkeypatch):
    topology = _load_topology_module()
    _use_driver(monkeypatch, topology, _FakeDriver())

    with pytest.raises(TypeError):
        topology.query_peer_native_atomics((0, 1), (0,))
    with pytest.raises(TypeError):
        topology.require_peer_native_atomics((0, 1), (0,))


def test_single_selected_accessor_omits_its_self_pair(monkeypatch):
    topology = _load_topology_module()
    driver = _FakeDriver()
    _use_driver(monkeypatch, topology, driver)

    evidence = topology.require_peer_native_atomics((7,), accessing_devices=(7,))

    assert evidence["devices"] == [7]
    assert evidence["accessing_devices"] == [7]
    assert evidence["ordered_pairs"] == []
    assert evidence["all_supported"] is True
    assert not any(call[0] == "cuDeviceGetP2PAttribute" for call in driver.calls)


def test_topology_query_is_lazy_and_does_not_link_core_binding_to_cuda():
    source = _TOPOLOGY_SOURCE.read_text(encoding="utf-8")
    binding = (_ROOT / "src" / "bindings" / "python" / "nixl_bindings.cpp").read_text(
        encoding="utf-8"
    )
    meson = (_ROOT / "src" / "bindings" / "python" / "meson.build").read_text(
        encoding="utf-8"
    )

    loader = source.split("def _load_cuda_driver", 1)[1].split("def _checked_cuda", 1)[
        0
    ]
    assert "from cuda.bindings import driver" in loader
    assert (
        "from cuda.bindings import driver"
        not in source.split("def _load_cuda_driver", 1)[0]
    )
    assert "cuDeviceGetP2PAttribute" not in binding
    assert "NIXL_PYTHON_HAVE_CUDA" not in binding
    assert "cuda_dep" not in meson
    assert "dependencies: [nixl_dep, serdes_interface, pybind_dep]" in meson
