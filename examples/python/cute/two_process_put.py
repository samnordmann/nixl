#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a NIXL PUT from a CuTe DSL kernel across two local GPU processes.

Rank 0 owns the source and launches the CuTe kernel. Rank 1 owns the remote
destination. A small file rendezvous stands in for the application's control
plane and exchanges NIXL metadata plus remote transfer coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import torch.multiprocessing as mp
from cutlass.cute.runtime import from_dlpack

import nixl.device.cute as nixl_cute
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

NUM_ELEMENTS = 1024
NUM_BYTES = NUM_ELEMENTS * 4  # torch.int32
RENDEZVOUS_TIMEOUT_SECONDS = 30.0


@cute.kernel
def put_kernel(
    local: nixl_cute.MemoryView,
    remote: nixl_cute.MemoryView,
    status: cute.Tensor,
):
    """Have the only launched thread copy the complete source descriptor."""
    status[0] = nixl_cute.put(
        local,
        remote,
        NUM_BYTES,
        scope=nixl_cute.Scope.THREAD,
        flags=nixl_cute.Flags.NONE,
    )


@cute.jit
def launch_put(
    local: nixl_cute.MemoryView,
    remote: nixl_cute.MemoryView,
    status: cute.Tensor,
    stream: cuda.CUstream,
):
    """Launch on the caller-owned stream; this function does not synchronize."""
    put_kernel(local, remote, status).launch(
        grid=[1, 1, 1], block=[1, 1, 1], stream=stream
    )


class FileRendezvous:
    """Minimal bounded rendezvous used only by this local-node example."""

    def __init__(self, directory: str, rank: int):
        self._directory = Path(directory)
        self.rank = rank
        self.peer = 1 - rank

    def _path(self, tag: str, rank: int) -> Path:
        return self._directory / f"{tag}.{rank}"

    def publish(self, tag: str, payload: bytes = b"") -> None:
        final = self._path(tag, self.rank)
        temporary = final.with_suffix(final.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, final)

    def await_peer(self, tag: str) -> bytes:
        path = self._path(tag, self.peer)
        deadline = time.monotonic() + RENDEZVOUS_TIMEOUT_SECONDS
        while True:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"rank {self.rank} timed out waiting for peer {self.peer} "
                        f"rendezvous {tag!r}"
                    )
                time.sleep(0.05)

    def barrier(self, tag: str) -> None:
        self.publish(tag)
        self.await_peer(tag)


def _wait_for_notification(agent: nixl_agent, peer_name: str) -> None:
    deadline = time.monotonic() + RENDEZVOUS_TIMEOUT_SECONDS
    while True:
        messages = agent.get_new_notifs().get(peer_name, ())
        if b"connected" in messages:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for a notification from {peer_name!r}"
            )
        time.sleep(0.05)


def _worker(rank: int, devices: tuple[int, int], directory: str) -> None:
    device_index = devices[rank]
    torch.cuda.set_device(device_index)
    stream = torch.cuda.Stream(device=device_index)
    cu_stream = cuda.CUstream(stream.cuda_stream)
    rendezvous = FileRendezvous(directory, rank)

    # Establish the CUDA context and finish allocation initialization before
    # creating UCX workers. Each process owns exactly one registered region.
    with torch.cuda.stream(stream):
        if rank == 0:
            buffer = torch.arange(
                NUM_ELEMENTS, dtype=torch.int32, device=f"cuda:{device_index}"
            )
            device_status = torch.full(
                (1,),
                int(nixl_cute.NIXL_IN_PROG),
                dtype=torch.int32,
                device=buffer.device,
            )
        else:
            buffer = torch.zeros(
                NUM_ELEMENTS, dtype=torch.int32, device=f"cuda:{device_index}"
            )
            device_status = None
    stream.synchronize()

    name = f"cute_put_{os.getppid()}_{rank}"
    peer_name = f"cute_put_{os.getppid()}_{1 - rank}"
    config = nixl_agent_config(
        enable_prog_thread=True,
        backends=["UCX"],
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
    )
    agent = nixl_agent(name, config)

    with ExitStack() as resources:
        registration = agent.register_memory(buffer, backends=["UCX"])
        resources.callback(agent.deregister_memory, registration, backends=["UCX"])

        # Metadata must be generated after registration; the peer also needs
        # the address tuple because NIXL metadata deliberately does not expose
        # application transfer coordinates.
        coordinates = {
            "name": name,
            "address": buffer.data_ptr(),
            "length": NUM_BYTES,
            "device_id": device_index,
        }
        rendezvous.publish("metadata", agent.get_agent_metadata())
        rendezvous.publish("coordinates", json.dumps(coordinates).encode("utf-8"))
        peer_metadata = rendezvous.await_peer("metadata")
        peer_coordinates = json.loads(rendezvous.await_peer("coordinates"))
        if peer_coordinates["name"] != peer_name:
            raise RuntimeError(
                f"rank {rank} expected peer {peer_name!r}, got "
                f"{peer_coordinates['name']!r}"
            )
        agent.add_remote_agent(peer_metadata)
        resources.callback(agent.remove_remote_agent, peer_name)
        rendezvous.barrier("metadata_loaded")

        # A notification round trip forces real inter-process CUDA-IPC lane
        # wire-up before rank 0 prepares its remote device memory list.
        agent.make_connection(peer_name, backends=["UCX"])
        agent.send_notif(peer_name, b"connected", backend="UCX")
        _wait_for_notification(agent, peer_name)
        rendezvous.barrier("connected")

        if rank == 0:
            assert device_status is not None
            try:
                with ExitStack() as views:
                    local = views.enter_context(
                        agent.prepare_device_view(buffer, backend="UCX")
                    )
                    remote = views.enter_context(
                        agent.prepare_device_view(
                            [
                                (
                                    peer_coordinates["address"],
                                    peer_coordinates["length"],
                                    peer_coordinates["device_id"],
                                )
                            ],
                            remote_agent=peer_name,
                            mem_type="VRAM",
                            backend="UCX",
                            connection_timeout_ms=30_000,
                        )
                    )
                    try:
                        launch_put(
                            nixl_cute.MemoryView(local),
                            nixl_cute.MemoryView(remote),
                            from_dlpack(device_status).mark_layout_dynamic(),
                            cu_stream,
                        )
                    finally:
                        # Registrations, peer metadata, and both views remain
                        # live until the explicit launch stream is quiescent.
                        stream.synchronize()

                    status = int(device_status.item())
                    if status != int(nixl_cute.NIXL_SUCCESS):
                        try:
                            status_name = nixl_cute.Status(status).name
                        except ValueError:
                            status_name = "UNKNOWN_STATUS"
                        raise RuntimeError(
                            f"device PUT failed: {status_name} ({status})"
                        )
            finally:
                # Always wake rank 1; if launch failed it will report a payload
                # mismatch instead of hanging indefinitely in the example.
                rendezvous.publish("transfer_complete")

            # Both owning handles have now been released.
            rendezvous.publish("views_released")
            rendezvous.await_peer("payload_validated")
        else:
            rendezvous.await_peer("transfer_complete")
            torch.cuda.synchronize(device_index)
            expected = torch.arange(
                NUM_ELEMENTS, dtype=torch.int32, device=buffer.device
            )
            if not torch.equal(buffer, expected):
                raise RuntimeError("remote payload validation failed")
            rendezvous.publish("payload_validated")
            rendezvous.await_peer("views_released")

        # Neither process may invalidate metadata or destroy its endpoint while
        # the peer can still be releasing a view that references it.
        rendezvous.barrier("safe_to_teardown")

    # ExitStack first invalidates peer metadata, then deregisters local memory.
    rendezvous.barrier("cleanup_complete")
    if rank == 0:
        print(
            f"NIXL CuTe PUT copied {NUM_BYTES} bytes from cuda:{devices[0]} "
            f"to cuda:{devices[1]}"
        )


def run(devices: tuple[int, int]) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this example requires at least two CUDA GPUs")
    if len(set(devices)) != 2:
        raise ValueError("the two ranks must use distinct CUDA devices")
    for device in devices:
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device} is unavailable; found "
                f"{torch.cuda.device_count()} devices"
            )

    with TemporaryDirectory(prefix="nixl_cute_put_") as directory:
        mp.spawn(_worker, args=(devices, directory), nprocs=2, join=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("SOURCE_GPU", "DESTINATION_GPU"),
        help="two distinct local CUDA device indices (default: 0 1)",
    )
    args = parser.parse_args()
    run(tuple(args.devices))


if __name__ == "__main__":
    main()
