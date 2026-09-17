#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish GPU data with NIXL PUT followed by a remote atomic signal.

Two local processes use distinct GPUs. Rank 0 writes rank 1's payload and then
increments rank 1's 64-bit completion counter on the same NIXL device channel.
Rank 1 waits on that counter with a host-side deadline before consuming data.
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.multiprocessing as mp

import nixl.device.cute as nixl_cute
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

if __package__:
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_put_then_signal_host,
        wait_for_value,
        wait_until,
    )
else:
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_put_then_signal_host,
        wait_for_value,
        wait_until,
    )

PAYLOAD_ELEMENTS = 1024
PAYLOAD_BYTES = PAYLOAD_ELEMENTS * 4
CHANNEL = 0
WORLD_SIZE = 2


def _region(tensor: torch.Tensor) -> DeviceRegion:
    return DeviceRegion(
        address=tensor.data_ptr(),
        length=tensor.numel() * tensor.element_size(),
        device_id=tensor.get_device(),
    )


def _wait_for_notification(
    agent: nixl_agent, peer_name: str, message: bytes, timeout_s: float
) -> None:
    def received() -> bool:
        return message in agent.get_new_notifs().get(peer_name, ())

    wait_until(
        received,
        timeout_s=timeout_s,
        description=f"notification {message!r} from {peer_name!r}",
        poll_interval_s=0.05,
    )


def _worker(
    rank: int,
    devices: tuple[int, int],
    directory: str,
    timeout_s: float,
) -> None:
    device_index = devices[rank]
    torch.cuda.set_device(device_index)
    stream = torch.cuda.Stream(device=device_index)
    control = FileControlPlane(directory, rank, WORLD_SIZE, timeout_s)

    # Allocate and initialize before creating the UCX worker. The target keeps
    # payload and signal as separate descriptors so the atomic has an explicit,
    # naturally aligned 64-bit destination.
    with torch.cuda.stream(stream):
        if rank == 0:
            payload = torch.arange(
                PAYLOAD_ELEMENTS, dtype=torch.int32, device=f"cuda:{device_index}"
            )
            signal = None
            statuses = torch.full(
                (2,),
                int(nixl_cute.NIXL_ERR_NOT_POSTED),
                dtype=torch.int32,
                device=payload.device,
            )
            registered_tensors = [payload]
        else:
            payload = torch.zeros(
                PAYLOAD_ELEMENTS, dtype=torch.int32, device=f"cuda:{device_index}"
            )
            signal = torch.zeros(1, dtype=torch.int64, device=payload.device)
            statuses = None
            registered_tensors = [payload, signal]
    stream.synchronize()

    run_id = Path(directory).name
    name = f"cute_put_signal_{run_id}_{rank}"
    peer_name = f"cute_put_signal_{run_id}_{1 - rank}"
    agent = nixl_agent(
        name,
        nixl_agent_config(
            enable_prog_thread=True,
            backends=["UCX"],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        ),
    )

    with ExitStack() as resources:
        registration = agent.register_memory(registered_tensors, backends=["UCX"])
        resources.callback(agent.deregister_memory, registration, backends=["UCX"])

        local_coordinates = PeerCoordinates(
            name, tuple(_region(tensor) for tensor in registered_tensors)
        )
        metadata = control.exchange("metadata", agent.get_agent_metadata())
        coordinates = control.exchange("coordinates", local_coordinates.to_bytes())
        peer_coordinates = PeerCoordinates.from_bytes(coordinates[1 - rank])
        if peer_coordinates.agent_name != peer_name:
            raise RuntimeError(
                f"rank {rank} expected peer {peer_name!r}, got "
                f"{peer_coordinates.agent_name!r}"
            )

        agent.add_remote_agent(metadata[1 - rank])
        resources.callback(agent.remove_remote_agent, peer_name)
        control.barrier("metadata-loaded")

        # Force real endpoint wire-up before preparing a remote device view.
        agent.make_connection(peer_name, backends=["UCX"])
        agent.send_notif(peer_name, b"connected", backend="UCX")
        _wait_for_notification(agent, peer_name, b"connected", timeout_s)
        control.barrier("connected")

        if rank == 0:
            if statuses is None:
                raise AssertionError("rank 0 status allocation is missing")
            if len(peer_coordinates.regions) != 2:
                raise RuntimeError("target must publish payload and signal regions")
            with ExitStack() as views:
                local_view = views.enter_context(
                    agent.prepare_device_view(registered_tensors, backend="UCX")
                )
                remote_view = views.enter_context(
                    agent.prepare_device_view(
                        [region.descriptor for region in peer_coordinates.regions],
                        remote_agent=peer_name,
                        mem_type="VRAM",
                        backend="UCX",
                        connection_timeout_ms=max(1, int(timeout_s * 1000)),
                    )
                )
                try:
                    launch_put_then_signal_host(
                        local_view,
                        remote_view,
                        statuses,
                        stream,
                        size=PAYLOAD_BYTES,
                        remote_index=0,
                        signal_index=1,
                        channel=CHANNEL,
                        scope=nixl_cute.Scope.THREAD,
                    )
                finally:
                    # Both owning views and all registrations remain live until
                    # the exact launch stream is quiescent.
                    stream.synchronize()

                put_status, signal_status = (int(value) for value in statuses.tolist())
                check_status(put_status, "device PUT")
                check_status(signal_status, "remote completion atomic-add")

            # The target must not deregister while our remote view is live.
            control.publish("views-released")
            control.await_rank("payload-validated", 1)
        else:
            if signal is None:
                raise AssertionError("rank 1 signal allocation is missing")
            wait_for_value(
                lambda: int(signal.item()),
                1,
                timeout_s=timeout_s,
                description="remote GPU completion counter to become 1",
                poll_interval_s=0.01,
            )
            expected = torch.arange(
                PAYLOAD_ELEMENTS, dtype=torch.int32, device=payload.device
            )
            if not torch.equal(payload, expected):
                raise RuntimeError(
                    "remote signal arrived before a valid payload was visible"
                )
            control.publish("payload-validated")
            control.await_rank("views-released", 0)

        # remove_remote_agent and deregistration happen only after every remote
        # view has been released and the consumer has finished reading.
        control.barrier("safe-to-teardown")

    control.barrier("cleanup-complete")
    if rank == 0:
        print(
            f"NIXL CuTe PUT+signal published {PAYLOAD_BYTES} bytes from "
            f"cuda:{devices[0]} to cuda:{devices[1]} on channel {CHANNEL}"
        )


def run(devices: tuple[int, int], timeout_s: float = 30.0) -> None:
    """Spawn the producer and consumer after validating local GPU selection.

    ``timeout_s`` bounds control-plane and receiver polling only. The v1 device
    ABI waits synchronously and cannot cancel a kernel stalled by peer loss;
    use an external process/job timeout when fault injection is possible.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this example requires at least two CUDA GPUs")
    if len(devices) != WORLD_SIZE:
        raise ValueError("--devices must name exactly two CUDA devices")
    if any(
        isinstance(device, bool) or not isinstance(device, int) for device in devices
    ):
        raise TypeError("CUDA device IDs must be integers")
    if len(set(devices)) != WORLD_SIZE:
        raise ValueError("--devices must name two distinct CUDA devices")
    for device in devices:
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device} is unavailable; found "
                f"{torch.cuda.device_count()} devices"
            )
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a positive finite number")

    with TemporaryDirectory(prefix=f"nixl_cute_signal_{os.getpid()}_") as directory:
        mp.spawn(
            _worker,
            args=(devices, directory, timeout_s),
            nprocs=WORLD_SIZE,
            join=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The timeout cannot interrupt the synchronous device wait if a peer "
            "is lost; use an external process or scheduler timeout for fault tests."
        ),
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("SOURCE_GPU", "DESTINATION_GPU"),
        help="two distinct local CUDA device indices (default: 0 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds allowed for each control-plane or signal wait (default: 30)",
    )
    args = parser.parse_args()
    run(tuple(args.devices), args.timeout)


if __name__ == "__main__":
    main()
