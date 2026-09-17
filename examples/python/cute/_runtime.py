#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free runtime helpers shared by the CuTe examples."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_UINT64_MAX = (1 << 64) - 1
_STATUS_NAMES = {
    1: "IN_PROGRESS",
    0: "SUCCESS",
    -1: "NOT_POSTED",
    -2: "INVALID_PARAM",
    -3: "BACKEND",
    -4: "NOT_FOUND",
    -5: "MISMATCH",
    -6: "NOT_ALLOWED",
    -7: "REPOST_ACTIVE",
    -8: "UNKNOWN",
    -9: "NOT_SUPPORTED",
    -10: "REMOTE_DISCONNECT",
    -11: "CANCELED",
    -12: "NO_TELEMETRY",
}
_T = TypeVar("_T")

# Keep the control-plane portion importable on login nodes and in CPU-only
# unit tests. When CuTe is installed, these definitions become normal JIT
# callables; otherwise the stubs below fail only when device work is requested.
try:
    _CUTE_AVAILABLE = importlib.util.find_spec("cutlass.cute") is not None
except ModuleNotFoundError:
    _CUTE_AVAILABLE = False
if _CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    import nixl.device.cute as nixl_cute


if _CUTE_AVAILABLE:

    @cute.kernel
    def put_then_signal_kernel(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        statuses: cute.Tensor,
        size: cutlass.Constexpr[int],
        local_index: cutlass.Constexpr[int],
        local_offset: cutlass.Constexpr[int],
        remote_index: cutlass.Constexpr[int],
        remote_offset: cutlass.Constexpr[int],
        signal_index: cutlass.Constexpr[int],
        signal_offset: cutlass.Constexpr[int],
        signal_value: cutlass.Constexpr[int],
        channel: cutlass.Constexpr[int],
        with_signal: cutlass.Constexpr[bool],
        scope: cutlass.Constexpr,
    ):
        """Issue one PUT and, when selected, signal success on the same channel."""
        put_status = nixl_cute.put(
            local,
            remote,
            size,
            local_index=local_index,
            local_offset=local_offset,
            remote_index=remote_index,
            remote_offset=remote_offset,
            channel=channel,
            scope=scope,
        )
        if cutlass.const_expr(with_signal):
            if put_status == int(nixl_cute.NIXL_SUCCESS):
                signal_status = nixl_cute.atomic_add(
                    remote,
                    signal_value,
                    index=signal_index,
                    offset=signal_offset,
                    channel=channel,
                    scope=scope,
                )
                tidx, _, _ = cute.arch.thread_idx()
                if tidx == 0:
                    statuses[1] = signal_status
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            statuses[0] = put_status

    @cute.jit
    def launch_put_then_signal(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        size: cutlass.Constexpr[int],
        local_index: cutlass.Constexpr[int],
        local_offset: cutlass.Constexpr[int],
        remote_index: cutlass.Constexpr[int],
        remote_offset: cutlass.Constexpr[int],
        signal_index: cutlass.Constexpr[int],
        signal_offset: cutlass.Constexpr[int],
        signal_value: cutlass.Constexpr[int],
        channel: cutlass.Constexpr[int],
        with_signal: cutlass.Constexpr[bool],
        scope: cutlass.Constexpr,
    ):
        """Launch one specialized PUT or PUT+signal kernel without synchronizing."""
        if cutlass.const_expr(scope == nixl_cute.Scope.THREAD):
            block_size = 1
        elif cutlass.const_expr(scope == nixl_cute.Scope.WARP):
            block_size = 32
        else:
            raise ValueError("scope must be Scope.THREAD or Scope.WARP")
        put_then_signal_kernel(
            local,
            remote,
            statuses,
            size,
            local_index,
            local_offset,
            remote_index,
            remote_offset,
            signal_index,
            signal_offset,
            signal_value,
            channel,
            with_signal,
            scope,
        ).launch(grid=[1, 1, 1], block=[block_size, 1, 1], stream=stream)

    @cute.kernel
    def masked_thread_put_then_signal_kernel(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        rank_mask: cute.Tensor,
        statuses: cute.Tensor,
        destination_rank: cutlass.Constexpr[int],
        size: cutlass.Constexpr[int],
        local_index: cutlass.Constexpr[int],
        local_offset: cutlass.Constexpr[int],
        remote_index: cutlass.Constexpr[int],
        remote_offset: cutlass.Constexpr[int],
        signal_index: cutlass.Constexpr[int],
        signal_offset: cutlass.Constexpr[int],
        signal_value: cutlass.Constexpr[int],
        channel: cutlass.Constexpr[int],
    ):
        """Publish only when the committed device rank mask marks the peer active."""
        if rank_mask[destination_rank] == 0:
            put_status = nixl_cute.put(
                local,
                remote,
                size,
                local_index=local_index,
                local_offset=local_offset,
                remote_index=remote_index,
                remote_offset=remote_offset,
                channel=channel,
                scope=nixl_cute.Scope.THREAD,
            )
            statuses[0] = put_status
            if put_status == int(nixl_cute.NIXL_SUCCESS):
                statuses[1] = nixl_cute.atomic_add(
                    remote,
                    signal_value,
                    index=signal_index,
                    offset=signal_offset,
                    channel=channel,
                    scope=nixl_cute.Scope.THREAD,
                )
        else:
            statuses[0] = int(nixl_cute.NIXL_ERR_NOT_ALLOWED)
            statuses[1] = int(nixl_cute.NIXL_ERR_NOT_ALLOWED)

    @cute.jit
    def launch_masked_thread_put_then_signal(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        rank_mask: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        destination_rank: cutlass.Constexpr[int],
        size: cutlass.Constexpr[int],
        local_index: cutlass.Constexpr[int],
        local_offset: cutlass.Constexpr[int],
        remote_index: cutlass.Constexpr[int],
        remote_offset: cutlass.Constexpr[int],
        signal_index: cutlass.Constexpr[int],
        signal_offset: cutlass.Constexpr[int],
        signal_value: cutlass.Constexpr[int],
        channel: cutlass.Constexpr[int],
    ):
        """Launch one mask-gated THREAD PUT+signal without synchronizing."""
        masked_thread_put_then_signal_kernel(
            local,
            remote,
            rank_mask,
            statuses,
            destination_rank,
            size,
            local_index,
            local_offset,
            remote_index,
            remote_offset,
            signal_index,
            signal_offset,
            signal_value,
            channel,
        ).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

else:

    def put_then_signal_kernel(*args, **kwargs):
        raise RuntimeError("CuTe DSL is required for device PUT+signal kernels")

    def launch_put_then_signal(*args, **kwargs):
        raise RuntimeError("CuTe DSL is required for device PUT+signal kernels")

    def masked_thread_put_then_signal_kernel(*args, **kwargs):
        raise RuntimeError("CuTe DSL is required for mask-gated device kernels")

    def launch_masked_thread_put_then_signal(*args, **kwargs):
        raise RuntimeError("CuTe DSL is required for mask-gated device kernels")


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _non_negative_int(name: str, value: int, maximum: int = _UINT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class DeviceRegion:
    """Transfer coordinates for one registered VRAM region."""

    address: int
    length: int
    device_id: int

    def __post_init__(self) -> None:
        _non_negative_int("address", self.address)
        if self.address == 0:
            raise ValueError("address must be nonzero")
        _non_negative_int("length", self.length)
        if self.length == 0:
            raise ValueError("length must be positive")
        _non_negative_int("device_id", self.device_id)

    @property
    def descriptor(self) -> tuple[int, int, int]:
        """Return the tuple accepted by ``prepare_device_view``."""
        return (self.address, self.length, self.device_id)


@dataclass(frozen=True, slots=True)
class PeerCoordinates:
    """An agent name and its ordered registered-device coordinates."""

    agent_name: str
    regions: tuple[DeviceRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_name, str) or not self.agent_name:
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(self.regions, tuple) or not self.regions:
            raise ValueError("regions must be a non-empty tuple")
        if not all(isinstance(region, DeviceRegion) for region in self.regions):
            raise TypeError("regions must contain only DeviceRegion objects")

    def to_bytes(self) -> bytes:
        """Serialize a versioned, non-executable control-plane payload."""
        document = {
            "version": 1,
            "agent_name": self.agent_name,
            "regions": [
                {
                    "address": region.address,
                    "length": region.length,
                    "device_id": region.device_id,
                }
                for region in self.regions
            ],
        }
        return json.dumps(document, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> PeerCoordinates:
        """Validate and deserialize a payload produced by :meth:`to_bytes`."""
        if not isinstance(payload, bytes):
            raise TypeError("coordinate payload must be bytes")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid coordinate payload") from error
        if not isinstance(document, dict) or set(document) != {
            "version",
            "agent_name",
            "regions",
        }:
            raise ValueError("coordinate payload has an unsupported schema")
        if document["version"] != 1 or isinstance(document["version"], bool):
            raise ValueError("unsupported coordinate payload version")
        raw_regions = document["regions"]
        if not isinstance(raw_regions, list) or not raw_regions:
            raise ValueError("coordinate payload must contain at least one region")
        regions = []
        for raw in raw_regions:
            if not isinstance(raw, dict) or set(raw) != {
                "address",
                "length",
                "device_id",
            }:
                raise ValueError("coordinate payload contains an invalid region")
            regions.append(
                DeviceRegion(raw["address"], raw["length"], raw["device_id"])
            )
        return cls(document["agent_name"], tuple(regions))


class FileControlPlane:
    """A bounded, atomic file store for local multi-process examples.

    Tags are single-assignment by convention. Elastic examples should include
    their membership generation in every tag so a prior generation cannot be
    mistaken for current state. ``ranks`` may be any sparse subset of the
    configured world, but the calling rank must participate.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        rank: int,
        world_size: int,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.05,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.is_dir():
            raise ValueError(f"control-plane path is not a directory: {directory}")
        self.rank = rank
        self.world_size = world_size
        self.timeout_s = _positive_finite("timeout_s", timeout_s)
        self.poll_interval_s = _positive_finite("poll_interval_s", poll_interval_s)

    @staticmethod
    def _validate_tag(tag: str) -> str:
        if not isinstance(tag, str):
            raise TypeError("tag must be a string")
        if not _TAG.fullmatch(tag):
            raise ValueError(
                "tag must start with an alphanumeric character and contain only "
                "letters, digits, '.', '_' or '-'"
            )
        return tag

    def _validate_rank(self, rank: int) -> int:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank IDs must be integers")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside world size {self.world_size}")
        return rank

    def _participants(self, ranks: Iterable[int] | None) -> tuple[int, ...]:
        if ranks is None:
            return tuple(range(self.world_size))
        values = tuple(self._validate_rank(rank) for rank in ranks)
        if not values:
            raise ValueError("ranks must not be empty")
        if len(set(values)) != len(values):
            raise ValueError("ranks must not contain duplicates")
        if self.rank not in values:
            raise ValueError(f"calling rank {self.rank} must be present in ranks")
        return values

    def _path(self, tag: str, rank: int) -> Path:
        return self.directory / f"{self._validate_tag(tag)}.{self._validate_rank(rank)}"

    def publish(self, tag: str, payload: bytes = b"") -> None:
        """Atomically publish bytes under ``(tag, self.rank)``."""
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        final = self._path(tag, self.rank)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.directory,
                prefix=f".{final.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, final)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def _await_many(self, tag: str, ranks: tuple[int, ...]) -> dict[int, bytes]:
        pending = {rank: self._path(tag, rank) for rank in ranks}
        found: dict[int, bytes] = {}
        deadline = time.monotonic() + self.timeout_s
        while pending:
            for rank, path in tuple(pending.items()):
                try:
                    found[rank] = path.read_bytes()
                except FileNotFoundError:
                    continue
                del pending[rank]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"rank {self.rank} timed out waiting for tag {tag!r} from "
                    f"ranks {sorted(pending)}"
                )
            time.sleep(min(self.poll_interval_s, remaining))
        return found

    def await_rank(self, tag: str, rank: int) -> bytes:
        """Wait at most ``timeout_s`` for one rank's atomic publication."""
        rank = self._validate_rank(rank)
        return self._await_many(self._validate_tag(tag), (rank,))[rank]

    def exchange(
        self,
        tag: str,
        payload: bytes,
        ranks: Iterable[int] | None = None,
    ) -> dict[int, bytes]:
        """Publish and collect one payload from every participating rank."""
        participants = self._participants(ranks)
        self.publish(tag, payload)
        return self._await_many(self._validate_tag(tag), participants)

    def barrier(self, tag: str, ranks: Iterable[int] | None = None) -> None:
        """Bounded barrier over all ranks or an arbitrary sparse rank set."""
        participants = self._participants(ranks)
        self.publish(tag)
        self._await_many(self._validate_tag(tag), participants)


def wait_until(
    predicate: Callable[[], _T],
    *,
    timeout_s: float,
    description: str,
    poll_interval_s: float = 0.01,
) -> _T:
    """Poll until ``predicate`` is truthy, with one monotonic deadline."""
    timeout = _positive_finite("timeout_s", timeout_s)
    poll_interval = _positive_finite("poll_interval_s", poll_interval_s)
    if not isinstance(description, str) or not description:
        raise ValueError("description must be a non-empty string")
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(min(poll_interval, remaining))


def wait_for_value(
    read: Callable[[], _T],
    expected: _T,
    *,
    timeout_s: float,
    description: str,
    poll_interval_s: float = 0.01,
) -> _T:
    """Wait for an exact value and report the final observation on timeout."""
    last: list[object] = ["<not read>"]

    def matches() -> bool:
        last[0] = read()
        return last[0] == expected

    try:
        wait_until(
            matches,
            timeout_s=timeout_s,
            description=description,
            poll_interval_s=poll_interval_s,
        )
    except TimeoutError as error:
        raise TimeoutError(f"{error}; last observed value was {last[0]!r}") from None
    return expected


def check_status(status: int, operation: str) -> None:
    """Raise a readable error unless a host-observed NIXL status is success."""
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    try:
        value = int(status)
    except (TypeError, ValueError) as error:
        raise TypeError("status must be integer-convertible") from error
    if value != 0:
        name = _STATUS_NAMES.get(value, "UNRECOGNIZED")
        raise RuntimeError(f"{operation} failed: {name} ({value})")


def normalize_agent_name(value: str | bytes) -> str:
    """Normalize binding versions that return loaded agent names as bytes."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("agent name bytes must be valid UTF-8") from error
    if not isinstance(value, str):
        raise TypeError("agent name must be str or bytes")
    if not value:
        raise ValueError("agent name must not be empty")
    return value


def _setup_handshake_payload(
    *,
    kind: str,
    nonce: str,
    generation: int,
    source_rank: int,
    source_name: str,
    destination_rank: int | None = None,
    destination_name: str | None = None,
    observed_hello_source_ranks: tuple[int, ...] | None = None,
) -> bytes:
    document: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "nonce": nonce,
        "generation": generation,
        "source_rank": source_rank,
        "source_name": source_name,
    }
    if destination_rank is not None or destination_name is not None:
        if destination_rank is None or destination_name is None:
            raise AssertionError("setup handshake destination must be complete")
        document.update(
            {
                "destination_rank": destination_rank,
                "destination_name": destination_name,
            }
        )
    if observed_hello_source_ranks is not None:
        document["observed_hello_source_ranks"] = list(observed_hello_source_ranks)
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def complete_ucx_setup_handshake(
    agent,
    control: FileControlPlane,
    participant_names: Mapping[int, str],
    *,
    generation: int,
    nonce: str,
    notification_peer_ranks: Iterable[int] | None = None,
    timeout_s: float,
    poll_interval_s: float = 0.02,
    poll_callback: Callable[[], None] | None = None,
) -> None:
    """Complete one bounded UCX hello/receipt handshake before device setup.

    ``send_notif`` has no completion handle. With the UCX progress thread off,
    receiving every peer's hello is therefore not proof that this process's
    outbound hello has reached that peer. Each participant publishes a file
    receipt only after receiving its exact, directed hello set, then continues
    polling UCX while collecting every participant's receipt. Observing a
    peer's receipt proves that peer observed this process's hello before either
    side leaves setup. One monotonic deadline covers sends, inbound hellos, and
    receipt convergence.

    ``participant_names`` is the complete rank-to-agent identity map for the
    sparse setup group. ``notification_peer_ranks`` may select just the peers
    whose connections are new in an elastic generation; it defaults to every
    remote participant. ``poll_callback`` is setup-only and may raise to abort
    the handshake, for example when an elastic coordinator publishes ABORT.
    """

    if not isinstance(control, FileControlPlane):
        raise TypeError("control must be a FileControlPlane")
    if not isinstance(participant_names, Mapping) or not participant_names:
        raise ValueError("participant_names must be a non-empty rank/name mapping")
    participants_by_rank: dict[int, str] = {}
    for raw_rank, raw_name in participant_names.items():
        rank = control._validate_rank(raw_rank)
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"setup participant rank {rank} has an invalid name")
        participants_by_rank[rank] = raw_name
    participants = tuple(sorted(participants_by_rank))
    if len(participants) != len(participant_names):
        raise ValueError("participant_names contains ambiguous rank keys")
    if control.rank not in participants_by_rank:
        raise ValueError(
            f"calling rank {control.rank} must be present in participant_names"
        )
    if len(set(participants_by_rank.values())) != len(participants_by_rank):
        raise ValueError("setup participant agent names must be distinct")
    local_name = participants_by_rank[control.rank]
    if getattr(agent, "name", None) != local_name:
        raise ValueError(
            f"calling rank {control.rank} agent name does not match "
            f"participant_names: {getattr(agent, 'name', None)!r} != {local_name!r}"
        )

    generation = _non_negative_int("generation", generation)
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("nonce must be a non-empty string")
    timeout = _positive_finite("timeout_s", timeout_s)
    poll_interval = _positive_finite("poll_interval_s", poll_interval_s)
    if poll_callback is not None and not callable(poll_callback):
        raise TypeError("poll_callback must be callable")

    if notification_peer_ranks is None:
        notification_peers = tuple(
            rank for rank in participants if rank != control.rank
        )
    else:
        if isinstance(notification_peer_ranks, (str, bytes)):
            raise TypeError("notification_peer_ranks must contain integer ranks")
        try:
            raw_notification_peers = tuple(notification_peer_ranks)
        except TypeError as error:
            raise TypeError(
                "notification_peer_ranks must be an iterable of integer ranks"
            ) from error
        notification_peers = tuple(
            control._validate_rank(rank) for rank in raw_notification_peers
        )
        if len(set(notification_peers)) != len(notification_peers):
            raise ValueError("notification_peer_ranks must not contain duplicates")
        unknown = set(notification_peers) - set(participants)
        if unknown:
            raise ValueError(
                "notification_peer_ranks must be participant ranks; unknown "
                f"ranks {sorted(unknown)}"
            )
        if control.rank in notification_peers:
            raise ValueError("notification_peer_ranks must not contain the local rank")
        notification_peers = tuple(sorted(notification_peers))

    expected_by_name = {
        participants_by_rank[peer_rank]: (
            peer_rank,
            _setup_handshake_payload(
                kind="ucx-setup-hello",
                nonce=nonce,
                generation=generation,
                source_rank=peer_rank,
                source_name=participants_by_rank[peer_rank],
                destination_rank=control.rank,
                destination_name=local_name,
            ),
        )
        for peer_rank in notification_peers
    }
    deadline = time.monotonic() + timeout
    for peer_rank in notification_peers:
        peer_name = participants_by_rank[peer_rank]
        agent.send_notif(
            peer_name,
            _setup_handshake_payload(
                kind="ucx-setup-hello",
                nonce=nonce,
                generation=generation,
                source_rank=control.rank,
                source_name=local_name,
                destination_rank=peer_rank,
                destination_name=peer_name,
            ),
            backend="UCX",
        )

    participant_document = json.dumps(
        [[rank, participants_by_rank[rank]] for rank in participants],
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_nonce = hashlib.sha256(
        nonce.encode("utf-8") + b"\0" + participant_document
    ).hexdigest()[:24]
    receipt_tag = f"ucx-setup-receipt-g{generation}-{receipt_nonce}"
    local_receipt = _setup_handshake_payload(
        kind="ucx-setup-receipt",
        nonce=nonce,
        generation=generation,
        source_rank=control.rank,
        source_name=local_name,
        observed_hello_source_ranks=notification_peers,
    )
    receipt_paths = {rank: control._path(receipt_tag, rank) for rank in participants}
    pending_hellos = set(notification_peers)
    pending_receipts = set(participants)
    receipt_published = False

    while True:
        if poll_callback is not None:
            poll_callback()
        notifications = agent.get_new_notifs(backends=["UCX"])
        if not isinstance(notifications, dict):
            raise RuntimeError("UCX setup handshake returned malformed notifications")
        observed_names: set[str] = set()
        for raw_name, messages in notifications.items():
            name = normalize_agent_name(raw_name)
            if name in observed_names:
                raise RuntimeError(
                    f"UCX setup handshake returned duplicate source name {name!r}"
                )
            observed_names.add(name)
            expected = expected_by_name.get(name)
            if expected is None:
                raise RuntimeError(
                    "UCX setup handshake received a notification from unexpected "
                    f"agent {name!r} at rank {control.rank}"
                )
            if not isinstance(messages, list):
                raise RuntimeError(
                    f"UCX setup handshake notifications from {name!r} are malformed"
                )
            peer_rank, expected_payload = expected
            for message in messages:
                if message != expected_payload:
                    raise RuntimeError(
                        "UCX setup handshake received a stale or malformed "
                        f"notification from rank {peer_rank} ({name!r}); expected "
                        f"generation {generation} and nonce {nonce!r}"
                    )
                pending_hellos.discard(peer_rank)

        if not pending_hellos and not receipt_published:
            control.publish(receipt_tag, local_receipt)
            receipt_published = True

        if receipt_published:
            for peer_rank in tuple(pending_receipts):
                try:
                    receipt = receipt_paths[peer_rank].read_bytes()
                except FileNotFoundError:
                    continue
                try:
                    receipt_document = json.loads(receipt.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        "UCX setup handshake received a stale or malformed receipt "
                        f"from rank {peer_rank} ({participants_by_rank[peer_rank]!r}); "
                        f"expected generation {generation} and nonce {nonce!r}"
                    ) from error
                receipt_fields = {
                    "schema_version",
                    "kind",
                    "nonce",
                    "generation",
                    "source_rank",
                    "source_name",
                    "observed_hello_source_ranks",
                }
                hello_sources = (
                    receipt_document.get("observed_hello_source_ranks")
                    if isinstance(receipt_document, dict)
                    else None
                )
                hello_source_ranks = (
                    tuple(hello_sources)
                    if isinstance(hello_sources, list)
                    and all(type(source) is int for source in hello_sources)
                    and hello_sources == sorted(hello_sources)
                    and len(set(hello_sources)) == len(hello_sources)
                    and peer_rank not in hello_sources
                    and set(hello_sources) <= set(participants)
                    else None
                )
                local_edge_is_symmetric = hello_source_ranks is not None and (
                    (control.rank in hello_source_ranks)
                    is (peer_rank in notification_peers)
                )
                local_receipt_is_exact = peer_rank != control.rank or (
                    hello_source_ranks == notification_peers
                )
                if (
                    not isinstance(receipt_document, dict)
                    or set(receipt_document) != receipt_fields
                    or type(receipt_document.get("schema_version")) is not int
                    or receipt_document.get("schema_version") != 1
                    or receipt_document.get("kind") != "ucx-setup-receipt"
                    or receipt_document.get("nonce") != nonce
                    or type(receipt_document.get("generation")) is not int
                    or receipt_document.get("generation") != generation
                    or type(receipt_document.get("source_rank")) is not int
                    or receipt_document.get("source_rank") != peer_rank
                    or receipt_document.get("source_name")
                    != participants_by_rank[peer_rank]
                    or hello_source_ranks is None
                    or not local_edge_is_symmetric
                    or not local_receipt_is_exact
                ):
                    raise RuntimeError(
                        "UCX setup handshake received a stale, asymmetric, or "
                        f"malformed receipt from rank {peer_rank} "
                        f"({participants_by_rank[peer_rank]!r}); expected generation "
                        f"{generation} and nonce {nonce!r}"
                    )
                pending_receipts.remove(peer_rank)

        if receipt_published and not pending_receipts:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending_hello_labels = [
                (rank, participants_by_rank[rank]) for rank in sorted(pending_hellos)
            ]
            pending_receipt_labels = [
                (rank, participants_by_rank[rank]) for rank in sorted(pending_receipts)
            ]
            raise TimeoutError(
                f"rank {control.rank} timed out in UCX setup handshake for "
                f"generation {generation}, nonce {nonce!r}; pending hello peers "
                f"{pending_hello_labels}; pending receipt peers "
                f"{pending_receipt_labels}"
            )
        time.sleep(min(poll_interval, remaining))


def launch_put_then_signal_host(
    local_view,
    remote_view,
    statuses,
    stream,
    *,
    size: int,
    local_index: int = 0,
    local_offset: int = 0,
    remote_index: int = 0,
    remote_offset: int = 0,
    signal_index: int = 1,
    signal_offset: int = 0,
    signal_value: int = 1,
    channel: int = 0,
    with_signal: bool = True,
    scope: str = "thread",
    reset_statuses: bool = True,
) -> None:
    """Adapt host objects and enqueue a specialized PUT+signal kernel.

    Numeric arguments are CuTe ``Constexpr`` values. The first call for a new
    combination compiles a specialization; subsequent calls reuse CuTe's JIT
    cache. This function deliberately does not synchronize. The caller must
    keep both owning NIXL views, their registrations, peer metadata, the status
    tensor, and ``stream`` alive until that stream is synchronized.

    ``reset_statuses=False`` is intended for timing harnesses that enqueue the
    reset before their start event. In that mode the caller must initialize
    both status entries to ``NIXL_ERR_NOT_POSTED`` before every launch.
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for device PUT+signal kernels")
    import torch
    from cutlass.cute.runtime import from_dlpack

    if isinstance(scope, nixl_cute.Scope):
        scope_value = scope
    else:
        try:
            scope_value = {
                "thread": nixl_cute.Scope.THREAD,
                "warp": nixl_cute.Scope.WARP,
            }[scope.lower()]
        except (AttributeError, KeyError) as error:
            raise ValueError(
                "scope must be Scope.THREAD, Scope.WARP, 'thread', or 'warp'"
            ) from error
    if scope_value not in (nixl_cute.Scope.THREAD, nixl_cute.Scope.WARP):
        raise ValueError("only THREAD and WARP scope are supported")
    if not isinstance(with_signal, bool):
        raise TypeError("with_signal must be bool")
    if not isinstance(reset_statuses, bool):
        raise TypeError("reset_statuses must be bool")
    if not isinstance(statuses, torch.Tensor):
        raise TypeError("statuses must be a torch.Tensor")
    if (
        not statuses.is_cuda
        or statuses.dtype != torch.int32
        or not statuses.is_contiguous()
        or statuses.ndim != 1
        or statuses.numel() != 2
    ):
        raise ValueError(
            "statuses must be a contiguous, 1-D CUDA int32 tensor with 2 elements"
        )
    if statuses.device != stream.device:
        raise ValueError("statuses and stream must belong to the same CUDA device")
    _non_negative_int("size", size)
    if size == 0:
        raise ValueError("size must be positive")
    _non_negative_int("local_index", local_index, (1 << 32) - 1)
    _non_negative_int("local_offset", local_offset)
    _non_negative_int("remote_index", remote_index, (1 << 32) - 1)
    _non_negative_int("remote_offset", remote_offset)
    _non_negative_int("signal_index", signal_index, (1 << 32) - 1)
    _non_negative_int("signal_offset", signal_offset)
    if with_signal and signal_offset % 8:
        raise ValueError("signal_offset must be 8-byte aligned")
    _non_negative_int("signal_value", signal_value)
    _non_negative_int("channel", channel, (1 << 32) - 1)

    local = (
        local_view
        if isinstance(local_view, nixl_cute.MemoryView)
        else nixl_cute.MemoryView(local_view)
    )
    remote = (
        remote_view
        if isinstance(remote_view, nixl_cute.MemoryView)
        else nixl_cute.MemoryView(remote_view)
    )
    if reset_statuses:
        with torch.cuda.stream(stream):
            statuses.fill_(int(nixl_cute.NIXL_ERR_NOT_POSTED))
    launch_put_then_signal(
        local,
        remote,
        from_dlpack(statuses).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        size,
        local_index,
        local_offset,
        remote_index,
        remote_offset,
        signal_index,
        signal_offset,
        signal_value,
        channel,
        with_signal,
        scope_value,
    )


def launch_masked_thread_put_then_signal_host(
    local_view,
    remote_view,
    rank_mask,
    statuses,
    stream,
    *,
    destination_rank: int,
    size: int,
    local_index: int = 0,
    local_offset: int = 0,
    remote_index: int = 0,
    remote_offset: int = 0,
    signal_index: int = 1,
    signal_offset: int = 0,
    signal_value: int = 1,
    channel: int = 0,
) -> None:
    """Enqueue a THREAD PUT+signal gated by a committed device rank mask.

    The NIXL EP mask convention is zero for active and nonzero for inactive.
    ``destination_rank`` is specialized by CuTe, while the mask value is read
    by the GPU at execution time. This makes a stream-ordered mask copy the
    activation point; an inactive destination returns ``NIXL_ERR_NOT_ALLOWED``
    without posting either operation.
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for mask-gated device kernels")
    import torch
    from cutlass.cute.runtime import from_dlpack

    for name, tensor in (("rank_mask", rank_mask), ("statuses", statuses)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tensor.ndim != 1
        ):
            raise ValueError(f"{name} must be a contiguous, 1-D CUDA int32 tensor")
        if tensor.device != stream.device:
            raise ValueError(f"{name} and stream must belong to the same CUDA device")
    if statuses.numel() != 2:
        raise ValueError("statuses must contain exactly 2 elements")
    _non_negative_int("destination_rank", destination_rank, (1 << 32) - 1)
    if destination_rank >= rank_mask.numel():
        raise ValueError("destination_rank is outside rank_mask")
    _non_negative_int("size", size)
    if size == 0:
        raise ValueError("size must be positive")
    _non_negative_int("local_index", local_index, (1 << 32) - 1)
    _non_negative_int("local_offset", local_offset)
    _non_negative_int("remote_index", remote_index, (1 << 32) - 1)
    _non_negative_int("remote_offset", remote_offset)
    _non_negative_int("signal_index", signal_index, (1 << 32) - 1)
    _non_negative_int("signal_offset", signal_offset)
    if signal_offset % 8:
        raise ValueError("signal_offset must be 8-byte aligned")
    _non_negative_int("signal_value", signal_value)
    _non_negative_int("channel", channel, (1 << 32) - 1)

    local = (
        local_view
        if isinstance(local_view, nixl_cute.MemoryView)
        else nixl_cute.MemoryView(local_view)
    )
    remote = (
        remote_view
        if isinstance(remote_view, nixl_cute.MemoryView)
        else nixl_cute.MemoryView(remote_view)
    )
    with torch.cuda.stream(stream):
        statuses.fill_(int(nixl_cute.NIXL_ERR_NOT_POSTED))
    launch_masked_thread_put_then_signal(
        local,
        remote,
        from_dlpack(rank_mask).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        destination_rank,
        size,
        local_index,
        local_offset,
        remote_index,
        remote_offset,
        signal_index,
        signal_offset,
        signal_value,
        channel,
    )


__all__ = [
    "DeviceRegion",
    "FileControlPlane",
    "PeerCoordinates",
    "check_status",
    "complete_ucx_setup_handshake",
    "launch_masked_thread_put_then_signal",
    "launch_masked_thread_put_then_signal_host",
    "launch_put_then_signal",
    "launch_put_then_signal_host",
    "masked_thread_put_then_signal_kernel",
    "normalize_agent_name",
    "put_then_signal_kernel",
    "wait_for_value",
    "wait_until",
]
