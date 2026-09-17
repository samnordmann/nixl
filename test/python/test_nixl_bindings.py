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

import array
import ctypes
import gc
import os
import pickle
import shutil
import struct
import tempfile
from collections import Counter

import numpy as np
import pytest

import nixl._bindings as nixl
import nixl._utils as nixl_utils
from nixl.logging import get_logger

logger = get_logger(__name__)

# These should automatically be run by pytest because of function names


@pytest.mark.parametrize(
    "mode",
    [nixl.NIXL_THREAD_SYNC_STRICT, nixl.NIXL_THREAD_SYNC_RW],
)
def test_effective_sync_mode_binding(mode):
    config = nixl.nixlAgentConfig()
    config.syncMode = mode
    agent = nixl.nixlAgent(f"effective-sync-{mode}", config)
    assert agent.getEffectiveSyncMode() == mode


def test_list():
    descs = [(1000, 105, 0), (2000, 30, 0), (1010, 20, 0)]
    test_list = nixl.nixlXferDList(nixl.DRAM_SEG, descs)

    assert test_list.descCount() == 3

    test_list.print()

    pickled_list = pickle.dumps(test_list)

    logger.info("Pickled list: %s", pickled_list)

    unpickled_list = pickle.loads(pickled_list)

    assert unpickled_list == test_list

    assert test_list.getType() == nixl.DRAM_SEG

    logger.info("Descriptor count: %s", test_list.descCount())
    assert test_list.descCount() == 3

    test_list.remDesc(1)
    assert test_list.descCount() == 2

    assert test_list[0] == descs[0]

    test_list.clear()

    assert test_list.isEmpty()

    test_list.addDesc((2000, 100, 0))


def test_agent():
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
    # getXferTelemetry() needs an active telemetry sink; without one, telemetry
    # is disabled and the call returns NIXL_ERR_NO_TELEMETRY. Point telemetry at
    # a temporary directory so the buffer exporter is created.
    telemetry_dir = tempfile.mkdtemp(prefix="nixl-telemetry-test-")
    os.environ["NIXL_TELEMETRY_DIR"] = telemetry_dir
    name1 = "Agent1"
    name2 = "Agent2"

    devices = nixl.nixlAgentConfig()
    devices.useProgThread = False

    agent1 = nixl.nixlAgent(name1, devices)
    agent2 = nixl.nixlAgent(name2, devices)

    ucx1 = agent1.createBackend("UCX", {})
    ucx2 = agent2.createBackend("UCX", {})

    size = 256
    addr1 = nixl_utils.malloc_passthru(size)
    addr2 = nixl_utils.malloc_passthru(size)

    nixl_utils.ba_buf(addr1, size)

    reg_list1 = nixl.nixlRegDList(nixl.DRAM_SEG)
    reg_list1.addDesc((addr1, size, 0, "dead"))

    reg_list2 = nixl.nixlRegDList(nixl.DRAM_SEG)
    reg_list2.addDesc((addr2, size, 0, "dead"))

    with pytest.raises(ValueError, match="exactly one explicit UCX"):
        agent1.prepareDeregisterMem(reg_list1, [])
    with pytest.raises(ValueError, match="exactly one explicit UCX"):
        agent1.prepareDeregisterMem(reg_list1, [ucx1, ucx1])
    with pytest.raises(ValueError, match="does not belong to this NIXL agent"):
        agent1.prepareDeregisterMem(reg_list1, [ucx2])
    with pytest.raises(ValueError, match="does not belong to this NIXL agent"):
        agent1.createNotifReceiver([ucx2])
    with pytest.raises(ValueError, match="does not belong to this NIXL agent"):
        agent1.createNotifSender(name2, [ucx2])

    ret = agent1.registerMem(reg_list1, [ucx1])
    assert ret == nixl.NIXL_SUCCESS

    ret = agent2.registerMem(reg_list2, [ucx2])
    assert ret == nixl.NIXL_SUCCESS

    meta1 = agent1.getLocalMD()
    meta2 = agent2.getLocalMD()

    ret_name = agent1.loadRemoteMD(meta2)
    assert ret_name.decode(encoding="UTF-8") == name2
    ret_name = agent2.loadRemoteMD(meta1)
    assert ret_name.decode(encoding="UTF-8") == name1

    offset = 8
    req_size = 8

    src_list = nixl.nixlXferDList(nixl.DRAM_SEG)
    src_list.addDesc((addr1 + offset, req_size, 0))

    dst_list = nixl.nixlXferDList(nixl.DRAM_SEG)
    dst_list.addDesc((addr2 + offset, req_size, 0))

    logger.info("Transfer from %s to %s", str(addr1 + offset), str(addr2 + offset))

    noti_str = "n\0tification"
    logger.info("Notification string: %s", noti_str)

    logger.info("Source list: %s", src_list)
    logger.info("Destination list: %s", dst_list)

    handle_owner = agent1.createXferReqOwned(
        nixl.NIXL_WRITE, src_list, dst_list, name2, noti_str
    )
    handle = handle_owner.value
    assert handle != 0
    assert not handle_owner.released

    logger.info("Transfer handle: %s", handle)

    with pytest.raises(ValueError, match="reqh must be a non-zero"):
        agent1.postXferReqAndPoll(0, 0)
    with pytest.raises(ValueError, match="max_polls must be non-negative"):
        agent1.postXferReqAndPoll(handle, -1)
    with pytest.raises(ValueError, match="timeout_ns must be non-negative"):
        agent1.postXferReqAndPoll(handle, 1, -1)
    with pytest.raises(ValueError, match="reqh must be a non-zero"):
        agent1.getXferStatusBatch(0, 1)
    with pytest.raises(ValueError, match="max_polls must be positive"):
        agent1.getXferStatusBatch(handle, 0)
    with pytest.raises(ValueError, match="max_polls must be positive"):
        agent1.getXferStatusBatch(handle, -1)
    with pytest.raises(ValueError, match="timeout_ns must be non-negative"):
        agent1.getXferStatusBatch(handle, 1, -1)
    with pytest.raises(ValueError, match="timeout_check_interval must be positive"):
        agent1.getXferStatusBatch(handle, 1, None, 0)
    with pytest.raises(ValueError, match="timeout_check_interval must be positive"):
        agent1.getXferStatusBatch(handle, 1, None, -1)

    # A zero poll budget is deliberately post-only. Ordinary status polling
    # below completes that generation and verifies the exact same handle can
    # subsequently use the bounded fused loop.
    status = agent1.postXferReqAndPoll(handle, 0)
    assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    logger.info("Transfer posted")

    notifMap = {}

    while status != nixl.NIXL_SUCCESS or len(notifMap) == 0:
        if status != nixl.NIXL_SUCCESS:
            # timeout_ns=0 still performs exactly one status observation.
            status = agent1.getXferStatusBatch(handle, 1, 0)

        if len(notifMap) == 0:
            notifMap = agent2.getNotifs(notifMap)

        assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG

    nixl_utils.verify_transfer(addr1 + offset, addr2 + offset, req_size)
    assert len(notifMap[name1]) == 1
    logger.info("Received notification: %s", notifMap[name1][0])
    assert notifMap[name1][0] == noti_str.encode()

    # The timed post+poll path likewise observes status once even at a zero
    # timeout, then a larger poll-only batch can continue the same generation.
    status = agent1.postXferReqAndPoll(handle, 64, 0)
    notification_receiver = agent2.createNotifReceiver([ucx2])
    assert notification_receiver.poll() == {}
    second_notif_map = {}
    for _ in range(10_000):
        if status != nixl.NIXL_SUCCESS:
            status = agent1.getXferStatusBatch(
                handle, 64, 1_000_000_000, timeout_check_interval=1
            )
        if len(second_notif_map) == 0:
            second_notif_map = notification_receiver.poll()
        assert status == nixl.NIXL_SUCCESS or status == nixl.NIXL_IN_PROG
        if status == nixl.NIXL_SUCCESS and second_notif_map:
            break
    else:
        pytest.fail("timed post-and-poll notification did not complete")
    assert second_notif_map[name1] == (noti_str.encode(),)

    with pytest.raises(ValueError, match="max_items must be non-negative"):
        notification_receiver.poll_bounded(
            max_items=-1,
            max_batch_items=1,
            max_batch_bytes=1,
            max_payload_bytes=1,
        )
    with pytest.raises(ValueError, match="max_batch_items must be positive"):
        notification_receiver.poll_bounded(
            max_items=0,
            max_batch_items=0,
            max_batch_bytes=1,
            max_payload_bytes=1,
        )
    assert (
        notification_receiver.poll_bounded(
            max_items=0,
            max_batch_items=1,
            max_batch_bytes=1,
            max_payload_bytes=1,
        )
        == {}
    )

    # Explicit notification overrides differ from the legacy post API: None
    # is a deliberate clear, while non-empty bytes replace the retained tag.
    # Repost one native handle through both transitions so a provider can use
    # request-local tags without leaking a prior generation's notification.
    with pytest.raises(ValueError, match="reqh must be a non-zero"):
        agent1.postXferReqWithNotifOverride(0, None)
    with pytest.raises(ValueError, match="non-empty or None"):
        agent1.postXferReqWithNotifOverride(handle, b"")

    observed_override_notifs = []
    expected_override_notifs = []

    def collect_override_notifs():
        polled = agent2.getNotifsGrouped([ucx2])
        for source, values in polled.items():
            assert source == name1
            assert isinstance(values, tuple)
            observed_override_notifs.extend(values)

    def complete_override(payload):
        if payload is not None:
            expected_override_notifs.append(payload)
        expected = Counter(expected_override_notifs)
        override_status = agent1.postXferReqWithNotifOverride(handle, payload)
        for _ in range(10_000):
            if override_status != nixl.NIXL_SUCCESS:
                override_status = agent1.getXferStatusBatch(
                    handle, 64, 1_000_000_000, timeout_check_interval=32
                )
            collect_override_notifs()
            observed = Counter(observed_override_notifs)
            if override_status == nixl.NIXL_SUCCESS and all(
                observed[value] >= count for value, count in expected.items()
            ):
                break
        else:
            pytest.fail("explicit-notification override did not complete")
        # Reject every extra already observed, but do not infer ordering or
        # future absence from completion of this or any later request.
        assert not (Counter(observed_override_notifs) - expected)

    def complete_legacy(payload, expected_payload):
        expected_override_notifs.append(expected_payload)
        expected = Counter(expected_override_notifs)
        legacy_status = agent1.postXferReq(handle, payload)
        for _ in range(10_000):
            if legacy_status != nixl.NIXL_SUCCESS:
                legacy_status = agent1.getXferStatusBatch(
                    handle, 64, 1_000_000_000, timeout_check_interval=32
                )
            collect_override_notifs()
            observed = Counter(observed_override_notifs)
            if legacy_status == nixl.NIXL_SUCCESS and all(
                observed[value] >= count for value, count in expected.items()
            ):
                break
        else:
            pytest.fail("legacy notification post did not complete")
        assert not (Counter(observed_override_notifs) - expected)

    for override in (b"override-a", b"override-b", b"override-a"):
        complete_override(override)
    legacy_large_payload = b"legacy-" + b"l" * 4097 + b"\x00\xff-tail"
    complete_legacy(legacy_large_payload, legacy_large_payload)
    # An empty legacy value passes no options and therefore retains the tag
    # installed by the preceding non-empty legacy replacement.
    complete_legacy(b"", legacy_large_payload)
    # These clear-to-tag transitions exercise request-local mutation. NIXL does
    # not order separate requests, so no later tagged repost is an absence
    # barrier for a clear; the assertion is only the order-insensitive Counter
    # union observed by this test run.
    complete_override(None)
    complete_override(b"override-after-clear")
    complete_override(None)
    complete_override(b"\x00\xffbinary-after-clear")
    complete_override(None)
    complete_override(b"final-after-clear")
    collect_override_notifs()
    assert Counter(observed_override_notifs) == Counter(expected_override_notifs)

    logger.info("Transfer verified")

    # Verify transfer telemetry
    telem = agent1.getXferTelemetry(handle)
    assert telem.descCount == 1
    assert telem.totalBytes == req_size
    assert telem.startTime > 0
    assert telem.postDuration > 0
    assert telem.xferDuration > 0
    assert telem.xferDuration >= telem.postDuration

    assert handle_owner.release() == nixl.NIXL_SUCCESS
    assert handle_owner.released
    # The native owner makes a retry idempotent without dereferencing the
    # already-deleted pointer.
    assert handle_owner.release() == nixl.NIXL_SUCCESS

    owned_local = agent1.prepXferDlistOwned(src_list, [ucx1])
    owned_remote = agent1.prepXferDlistOwned(name2, dst_list, [ucx1])
    large_owned_payload = b"owned-prepped-" + b"x" * 4097 + b"\x00\xff-tail"
    large_owned_request = agent1.makeXferReqOwned(
        nixl.NIXL_WRITE,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        large_owned_payload,
        [ucx1],
        False,
    )
    large_owned_status = agent1.postXferReqAndPoll(large_owned_request.value, 64, None)
    large_owned_notifications = []
    for _ in range(10_000):
        if large_owned_status != nixl.NIXL_SUCCESS:
            large_owned_status = agent1.getXferStatusBatch(
                large_owned_request.value,
                64,
                1_000_000_000,
                timeout_check_interval=32,
            )
        for source, values in agent2.getNotifs({}).items():
            assert source == name1
            large_owned_notifications.extend(values)
        if large_owned_status == nixl.NIXL_SUCCESS and large_owned_notifications:
            break
    else:
        pytest.fail("large owned-request notification did not complete")
    assert large_owned_notifications == [large_owned_payload]
    assert large_owned_request.release() == nixl.NIXL_SUCCESS
    assert large_owned_request.released
    writable_native_owner = array.array("i", [0])
    assert writable_native_owner.itemsize == 4
    writable_native = memoryview(writable_native_owner)
    readonly_native = writable_native.toreadonly()
    unaligned_storage = bytearray(5)
    struct.pack_into("=i", unaligned_storage, 1, 0)
    unaligned_address = ctypes.addressof(
        ctypes.c_char.from_buffer(unaligned_storage, 1)
    )
    assert unaligned_address % ctypes.alignment(ctypes.c_int) != 0
    unaligned_native = memoryview(unaligned_storage)[1:5].cast("i")
    forcecast_numpy = np.asarray([0], dtype=np.int64)
    noncontiguous_numpy = np.asarray([0, 7, 0], dtype=np.int32)[::2]
    assert not noncontiguous_numpy.flags.c_contiguous

    for indices in (
        [0],
        writable_native,
        readonly_native,
        unaligned_native,
        forcecast_numpy,
        noncontiguous_numpy,
    ):
        owned_made = agent1.makeXferReqOwned(
            nixl.NIXL_WRITE,
            owned_local.value,
            indices,
            owned_remote.value,
            indices,
        )
        assert owned_made.value != 0
        assert owned_made.release() == nixl.NIXL_SUCCESS
        assert owned_made.release() == nixl.NIXL_SUCCESS

    with pytest.raises(ValueError, match="indices numpy array must be 1D"):
        agent1.makeXferReqOwned(
            nixl.NIXL_WRITE,
            owned_local.value,
            np.asarray([[0]], dtype=np.int32),
            owned_remote.value,
            [0],
        )

    running_state = object()
    completed_state = object()
    failed_state = object()
    with pytest.raises(
        ValueError,
        match=("local request-slot descriptor owner belongs to a different NIXL agent"),
    ):
        agent2.createXferRequestSlotExecution(
            nixl.NIXL_WRITE,
            owned_local.value,
            [0],
            owned_remote.value,
            [0],
            "",
            [ucx2],
            running_state,
            completed_state,
            failed_state,
            owned_local,
            owned_remote,
        )

    foreign_remote_owner = agent2.prepXferDlistOwned(dst_list, [ucx2])
    with pytest.raises(
        ValueError,
        match=(
            "remote request-slot descriptor owner belongs to a different NIXL agent"
        ),
    ):
        agent1.createXferRequestSlotExecution(
            nixl.NIXL_WRITE,
            owned_local.value,
            [0],
            foreign_remote_owner.value,
            [0],
            "",
            [ucx1],
            running_state,
            completed_state,
            failed_state,
            owned_local,
            foreign_remote_owner,
        )
    assert foreign_remote_owner.release() == nixl.NIXL_SUCCESS

    dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_WRITE,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    # A strong Python reference is insufficient because callers can invoke the
    # public owner's release method. The dispatcher leases both lists until
    # close, so these calls fail before touching the native descriptors.
    with pytest.raises(RuntimeError, match="leased by a request-slot execution"):
        owned_local.release()
    with pytest.raises(RuntimeError, match="leased by a request-slot execution"):
        owned_remote.release()
    assert not owned_local.released
    assert not owned_remote.released
    second_lease = agent1.createXferRequestSlotExecution(
        nixl.NIXL_WRITE,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    second_lease.close()
    # Closing one of two users must not make the shared owners releasable.
    with pytest.raises(RuntimeError, match="leased by a request-slot execution"):
        owned_local.release()
    with pytest.raises(ValueError, match="max_polls must be non-negative"):
        dispatch.start_and_poll(max_polls=-1, timeout_ns=None)
    for generation in range(2):
        state = dispatch.start_and_poll(
            max_polls=0 if generation == 0 else 64,
            timeout_ns=None if generation == 0 else 0,
        )
        for _ in range(10_000):
            if state is completed_state:
                break
            assert state is running_state
            agent2.getNotifs({})
            state = dispatch.poll_bounded(
                max_polls=64,
                timeout_ns=1_000_000_000,
                timeout_check_interval=32,
            )
        else:
            pytest.fail("native request-slot execution did not complete")
        assert dispatch.active
        if generation == 0:
            dispatch.recycle()
            assert not dispatch.active
    assert dispatch.failure_epoch == 0
    assert not dispatch.failed
    dispatch.close()
    dispatch.close()
    with pytest.raises(RuntimeError, match="closed"):
        dispatch.poll_state()

    fixed_slot_payload = b"fixed-slot-" + b"f" * 4097 + b"\x00\xff-tail"
    fixed_dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_WRITE,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        fixed_slot_payload,
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    fixed_slot_notifications = []
    for _ in range(2):
        fixed_state = fixed_dispatch.start_and_poll(max_polls=64, timeout_ns=0)
        for _ in range(10_000):
            if fixed_state is not completed_state:
                assert fixed_state is running_state
                fixed_state = fixed_dispatch.poll_bounded(
                    max_polls=64,
                    timeout_ns=1_000_000_000,
                    timeout_check_interval=32,
                )
            for source, values in agent2.getNotifs({}).items():
                assert source == name1
                fixed_slot_notifications.extend(values)
            if fixed_state is completed_state:
                break
        else:
            pytest.fail("fixed-notification request slot did not complete")
    for _ in range(10_000):
        for source, values in agent2.getNotifs({}).items():
            assert source == name1
            fixed_slot_notifications.extend(values)
        if len(fixed_slot_notifications) == 2:
            break
    else:
        pytest.fail("fixed request-slot notifications did not drain")
    assert fixed_slot_notifications == [fixed_slot_payload, fixed_slot_payload]
    fixed_dispatch.close()

    dynamic_dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_WRITE,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    with pytest.raises(ValueError, match="non-empty or None"):
        dynamic_dispatch.start_with_notification(b"")
    assert dynamic_dispatch.poll_state() is None

    start_only_payload = b"start-only-" + b"s" * 4097 + b"\x00\xff-tail"
    dynamic_payloads = (
        b"slot-A" + b"y" * 4097 + b"\x00\xff-tail",
        b"slot-B",
        b"slot-A",
        None,
        b"after-clear-1",
        None,
        b"slot-\x00\xff",
        None,
        b"after-clear-2",
    )
    expected_dynamic = [start_only_payload] + [
        payload for payload in dynamic_payloads if payload is not None
    ]
    observed_dynamic = []
    dynamic_dispatch.start_with_notification(start_only_payload)
    start_only_state = dynamic_dispatch.poll_state()
    for _ in range(10_000):
        if start_only_state is not completed_state:
            assert start_only_state is running_state
            start_only_state = dynamic_dispatch.poll_bounded(
                max_polls=64,
                timeout_ns=1_000_000_000,
                timeout_check_interval=32,
            )
        for source, values in agent2.getNotifs({}).items():
            assert source == name1
            observed_dynamic.extend(values)
        if start_only_state is completed_state:
            break
    else:
        pytest.fail("start_with_notification request did not complete")
    for generation, payload in enumerate(dynamic_payloads):
        dynamic_state = dynamic_dispatch.start_and_poll_with_notification(
            payload,
            max_polls=0 if generation == 0 else 64,
            timeout_ns=None if generation == 0 else 0,
        )
        for _ in range(10_000):
            if dynamic_state is not completed_state:
                assert dynamic_state is running_state
                dynamic_state = dynamic_dispatch.poll_bounded(
                    max_polls=64,
                    timeout_ns=1_000_000_000,
                    timeout_check_interval=32,
                )
            polled = agent2.getNotifs({})
            for source, values in polled.items():
                assert source == name1
                observed_dynamic.extend(values)
            if dynamic_state is completed_state:
                break
        else:
            pytest.fail("dynamic native request-slot execution did not complete")

    for _ in range(10_000):
        polled = agent2.getNotifs({})
        for source, values in polled.items():
            assert source == name1
            observed_dynamic.extend(values)
        if len(observed_dynamic) == len(expected_dynamic):
            break
    else:
        pytest.fail("dynamic native request-slot notifications did not drain")
    # Completion lets every expected payload be drained, but NIXL does not order
    # notifications across requests. No later payload is evidence that a clear
    # emitted nothing; compare exact delivery multiplicity without arrival order.
    assert Counter(observed_dynamic) == Counter(expected_dynamic)
    assert dynamic_dispatch.failure_epoch == 0
    assert not dynamic_dispatch.failed
    dynamic_dispatch.close()

    ctypes.memset(addr1 + offset, 0, req_size)
    read_dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_READ,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    read_state = read_dispatch.start_and_poll(max_polls=64, timeout_ns=0)
    for _ in range(10_000):
        if read_state is completed_state:
            break
        assert read_state is running_state
        agent2.getNotifs({})
        read_state = read_dispatch.poll_bounded(
            max_polls=64,
            timeout_ns=1_000_000_000,
            timeout_check_interval=32,
        )
    else:
        pytest.fail("native READ request-slot execution did not complete")
    nixl_utils.verify_transfer(addr1 + offset, addr2 + offset, req_size)
    read_dispatch.close()

    cold_dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_READ,
        owned_local.value,
        [0],
        owned_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        owned_local,
        owned_remote,
    )
    cold_dispatch.close()

    # The dispatcher owns the prepared-list guards, not only their integer
    # values. Dropping every external owner before lazy request creation must
    # therefore leave a valid transfer recipe.
    retained_local = agent1.prepXferDlistOwned(src_list, [ucx1])
    retained_remote = agent1.prepXferDlistOwned(name2, dst_list, [ucx1])
    retained_dispatch = agent1.createXferRequestSlotExecution(
        nixl.NIXL_WRITE,
        retained_local.value,
        [0],
        retained_remote.value,
        [0],
        "",
        [ucx1],
        running_state,
        completed_state,
        failed_state,
        retained_local,
        retained_remote,
    )
    del retained_local, retained_remote
    gc.collect()
    retained_state = retained_dispatch.start_and_poll(max_polls=64, timeout_ns=0)
    for _ in range(10_000):
        if retained_state is completed_state:
            break
        assert retained_state is running_state
        agent2.getNotifs({})
        retained_state = retained_dispatch.poll_bounded(
            max_polls=64,
            timeout_ns=1_000_000_000,
            timeout_check_interval=32,
        )
    else:
        pytest.fail("owner-retaining request-slot execution did not complete")
    nixl_utils.verify_transfer(addr1 + offset, addr2 + offset, req_size)
    retained_dispatch.close()
    del retained_dispatch
    gc.collect()

    with pytest.raises(ValueError, match="local request-slot descriptor owner"):
        agent1.createXferRequestSlotExecution(
            nixl.NIXL_WRITE,
            owned_local.value,
            [0],
            owned_remote.value,
            [0],
            "",
            [ucx1],
            running_state,
            completed_state,
            failed_state,
            owned_remote,
            owned_remote,
        )

    for invalid_indices in ([True], [1 << 40]):
        with pytest.raises(ValueError, match="request-slot index"):
            agent1.createXferRequestSlotExecution(
                nixl.NIXL_WRITE,
                owned_local.value,
                invalid_indices,
                owned_remote.value,
                [0],
                "",
                [ucx1],
                running_state,
                completed_state,
                failed_state,
                owned_local,
                owned_remote,
            )

    assert owned_remote.release() == nixl.NIXL_SUCCESS
    assert owned_remote.release() == nixl.NIXL_SUCCESS
    assert owned_local.release() == nixl.NIXL_SUCCESS
    assert owned_local.release() == nixl.NIXL_SUCCESS

    with pytest.raises(ValueError, match="release handle must be non-zero"):
        nixl.nixlXferReleaseState(0)

    # Legacy raw creation/release remains available for compatibility.
    prepared_handle = agent1.prepXferDlist(src_list, [ucx1])
    dlist_release_state = nixl.nixlDlistReleaseState(prepared_handle)
    assert not dlist_release_state.released
    assert (
        agent1.releasedDlistHOnce(dlist_release_state, prepared_handle)
        == nixl.NIXL_SUCCESS
    )
    assert dlist_release_state.released
    assert (
        agent1.releasedDlistHOnce(dlist_release_state, prepared_handle)
        == nixl.NIXL_SUCCESS
    )

    notification_sender = agent1.createNotifSender(name2, [ucx1])
    prepared_payload = b"prepared-sender\x00payload"
    assert notification_sender.send(prepared_payload) is None
    for _ in range(10_000):
        prepared_batches = notification_receiver.poll()
        if prepared_batches:
            break
    else:
        pytest.fail("prepared sender notification was not received")
    assert prepared_batches == {name1: (prepared_payload,)}

    bounded_expected = (
        b"bounded-native-one",
        b"bounded-native-two",
        b"bounded-native-three",
    )
    for payload in bounded_expected:
        assert notification_sender.send(payload) is None
    bounded_observed = []
    for _ in range(10_000):
        bounded_batch = notification_receiver.poll_bounded(
            max_items=1,
            max_batch_items=8,
            max_batch_bytes=1024,
            max_payload_bytes=128,
        )
        returned = [payload for source in bounded_batch.values() for payload in source]
        assert len(returned) <= 1
        bounded_observed.extend(returned)
        if Counter(bounded_observed) == Counter(bounded_expected):
            break
    else:
        pytest.fail("bounded prepared receiver did not deliver every payload")
    assert Counter(bounded_observed) == Counter(bounded_expected)

    deregistration1 = agent1.prepareDeregisterMem(reg_list1, [ucx1])
    assert not deregistration1.completed
    with pytest.raises(ValueError, match="different NIXL agent"):
        agent2.executeDeregisterMem(deregistration1)
    assert not deregistration1.completed
    # The receipt owns a descriptor snapshot: mutation of the Python dlist
    # after preparation cannot redirect native deregistration.
    reg_list1.clear()
    assert deregistration1.execute() == nixl.NIXL_SUCCESS
    assert deregistration1.completed
    assert agent1.executeDeregisterMem(deregistration1) == nixl.NIXL_SUCCESS

    deregistration2 = agent2.prepareDeregisterMem(reg_list2, [ucx2])
    assert agent2.executeDeregisterMem(deregistration2) == nixl.NIXL_SUCCESS
    assert deregistration2.completed
    assert deregistration2.execute() == nixl.NIXL_SUCCESS

    missing = nixl.nixlRegDList(nixl.DRAM_SEG, [(addr1, size, 0, b"never-registered")])
    missing_receipt = agent1.prepareDeregisterMem(missing, [ucx1])
    with pytest.raises(nixl.nixlNotFoundError):
        missing_receipt.execute()
    assert missing_receipt.completed
    assert missing_receipt.execute() == nixl.NIXL_SUCCESS

    # createNotifReceiver's keep_alive edge, not a Python alias, owns agent2.
    # The prepared receiver must remain usable after the caller drops its last
    # direct agent reference.
    del agent2
    gc.collect()
    keep_alive_payload = b"receiver-keeps-agent-alive\x00"
    agent1.genNotif(name2, keep_alive_payload, [ucx1])
    for _ in range(10_000):
        retained_batches = notification_receiver.poll()
        if retained_batches:
            break
    else:
        pytest.fail("receiver did not keep its target agent alive")
    assert retained_batches == {name1: (keep_alive_payload,)}

    # After the receiver is gone, the completed receipt is the only Python
    # object retaining agent2. Its bound idempotent retry must remain safe.
    del notification_receiver
    gc.collect()
    assert deregistration2.execute() == nixl.NIXL_SUCCESS

    # Only initiator should call invalidate
    agent1.invalidateRemoteMD(name2)
    # agent2.invalidateRemoteMD(name1)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)

    # Restore the no-sink default so telemetry stays inactive for later tests.
    os.environ.pop("NIXL_TELEMETRY_DIR", None)
    shutil.rmtree(telemetry_dir, ignore_errors=True)


def test_deregistration_receipt_rejects_owned_non_ucx_backend():
    config = nixl.nixlAgentConfig()
    config.useProgThread = False
    config.useListenThread = False
    agent = nixl.nixlAgent("non-ucx-receipt-scope", config)
    try:
        params, _ = agent.getPluginParams("POSIX")
        backend = agent.createBackend("POSIX", params)
    except Exception as error:
        pytest.skip(f"POSIX backend unavailable for owned non-UCX test: {error}")

    descs = nixl.nixlRegDList(nixl.DRAM_SEG)
    with pytest.raises(ValueError, match="requires a UCX backend handle"):
        agent.prepareDeregisterMem(descs, [backend])


def test_query_mem():
    """Test basic queryMem functionality"""

    os.makedirs("files_for_query", exist_ok=True)
    # Create temporary test files
    temp_file1 = tempfile.NamedTemporaryFile(dir="files_for_query", delete=False)
    temp_file1.write(b"Test content for queryMem file 1")
    temp_file1.close()

    temp_file2 = tempfile.NamedTemporaryFile(dir="files_for_query", delete=False)
    temp_file2.write(b"Test content for queryMem file 2")
    temp_file2.close()

    # Create a non-existent file path
    non_existent_file = "./nixl_test_nonexistent_file_12345.txt"

    try:
        # Create an agent
        config = nixl.nixlAgentConfig()
        config.useProgThread = False
        config.useListenThread = False
        agent = nixl.nixlAgent("test_agent", config)

        try:
            params, mems = agent.getPluginParams("POSIX")
            backend = agent.createBackend("POSIX", params)
        except Exception as e:
            logger.exception("POSIX backend creation failed: %s", e)
            try:
                params, mems = agent.getPluginParams("MOCK_DRAM")
                backend = agent.createBackend("MOCK_DRAM", params)
                logger.info("Using MOCK_DRAM backend as fallback")
            except Exception as e2:
                pytest.skip(
                    f"No working backends available (POSIX: {e}, MOCK_DRAM: {e2})"
                )

        descs = nixl.nixlRegDList(nixl.FILE_SEG)

        # Test 1: Query with empty descriptor list
        resp = agent.queryMem(descs, backend)
        assert len(resp) == 0

        # Test 2: Query with actual file descriptors
        # Existing file 1
        descs.addDesc((0, 0, 0, temp_file1.name))
        # Non-existent file
        descs.addDesc((0, 0, 0, non_existent_file))
        # Existing file 2
        descs.addDesc((0, 0, 0, temp_file2.name))

        resp = agent.queryMem(descs, backend)

        # Verify results
        assert len(resp) == 3

        # First file should be accessible (returns dict with info)
        assert resp[0] is not None
        assert isinstance(resp[0], dict)
        assert "size" in resp[0]
        assert "mode" in resp[0]

        # Second file should not be accessible (returns None)
        assert resp[1] is None

        # Third file should be accessible (returns dict with info)
        assert resp[2] is not None
        assert isinstance(resp[2], dict)
        assert "size" in resp[2]
        assert "mode" in resp[2]

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file1.name):
            os.unlink(temp_file1.name)
        if os.path.exists(temp_file2.name):
            os.unlink(temp_file2.name)
