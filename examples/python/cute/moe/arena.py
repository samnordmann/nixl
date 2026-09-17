# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked fixed-capacity memory layout for the CuTe NIXL MoE examples.

Each destination rank owns one arena.  The arena has one slab per possible
origin rank, including inactive ranks.  An origin is therefore the sole writer
of its slab in every destination arena, which avoids remote atomic allocation.
Elastic reconfiguration changes the active mask, never these offsets.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

# generation:u64, origin_rank:u32, origin_token:u32, route_slot:u32, expert:u32
DISPATCH_HEADER_NBYTES = 24

SLAB_PREAMBLE_MAGIC = 0x454D584E  # Little-endian bytes spell "NXME".
SLAB_PREAMBLE_VERSION = 2
# magic:u32, version:u16, wire_nbytes:u16, generation:u64,
# record_count:u32, origin_rank:u32, destination_rank:u32, element_size:u32
_SLAB_PREAMBLE = struct.Struct("<IHHQIIII")
SLAB_PREAMBLE_WIRE_NBYTES = _SLAB_PREAMBLE.size


def _checked_uint(name: str, value: int, maximum: int = UINT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _checked_positive(name: str, value: int, maximum: int = UINT64_MAX) -> int:
    _checked_uint(name, value, maximum)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _checked_add(lhs: int, rhs: int, limit: int, expression: str) -> int:
    if lhs > limit - rhs:
        raise OverflowError(f"{expression} exceeds the layout limit {limit}")
    return lhs + rhs


def _checked_mul(lhs: int, rhs: int, limit: int, expression: str) -> int:
    if lhs and rhs > limit // lhs:
        raise OverflowError(f"{expression} exceeds the layout limit {limit}")
    return lhs * rhs


def _require_power_of_two(name: str, value: int) -> int:
    _checked_positive(name, value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value


def align_up(value: int, alignment: int, *, limit: int = UINT64_MAX) -> int:
    """Return ``value`` rounded up, rejecting invalid or overflowing inputs."""

    _checked_uint("value", value, limit)
    _require_power_of_two("alignment", alignment)
    padding = (-value) & (alignment - 1)
    return _checked_add(value, padding, limit, "aligned value")


@dataclass(frozen=True)
class SlabPreamble:
    """Committed-record description at the start of each origin slab."""

    generation: int
    record_count: int
    origin_rank: int
    destination_rank: int
    element_size: int

    def __post_init__(self) -> None:
        _checked_uint("generation", self.generation, UINT64_MAX)
        _checked_uint("record_count", self.record_count, UINT32_MAX)
        _checked_uint("origin_rank", self.origin_rank, UINT32_MAX)
        _checked_uint("destination_rank", self.destination_rank, UINT32_MAX)
        _checked_positive("element_size", self.element_size, UINT32_MAX)

    def to_bytes(self, preamble_size: int = 32) -> bytes:
        _checked_positive("preamble_size", preamble_size, UINT32_MAX)
        if preamble_size < SLAB_PREAMBLE_WIRE_NBYTES:
            raise ValueError(
                f"preamble_size must be at least {SLAB_PREAMBLE_WIRE_NBYTES} bytes"
            )
        encoded = _SLAB_PREAMBLE.pack(
            SLAB_PREAMBLE_MAGIC,
            SLAB_PREAMBLE_VERSION,
            SLAB_PREAMBLE_WIRE_NBYTES,
            self.generation,
            self.record_count,
            self.origin_rank,
            self.destination_rank,
            self.element_size,
        )
        return encoded + bytes(preamble_size - len(encoded))

    @classmethod
    def from_bytes(
        cls, data: bytes, *, require_zero_padding: bool = True
    ) -> "SlabPreamble":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) < SLAB_PREAMBLE_WIRE_NBYTES:
            raise ValueError(
                f"preamble has {len(data)} bytes, expected at least "
                f"{SLAB_PREAMBLE_WIRE_NBYTES}"
            )
        (
            magic,
            version,
            wire_nbytes,
            generation,
            count,
            origin,
            destination,
            element_size,
        ) = _SLAB_PREAMBLE.unpack_from(data)
        if magic != SLAB_PREAMBLE_MAGIC:
            raise ValueError(f"invalid slab preamble magic 0x{magic:08x}")
        if version != SLAB_PREAMBLE_VERSION:
            raise ValueError(f"unsupported slab preamble version {version}")
        if wire_nbytes != SLAB_PREAMBLE_WIRE_NBYTES:
            raise ValueError(f"invalid slab preamble wire size {wire_nbytes}")
        if require_zero_padding and any(data[SLAB_PREAMBLE_WIRE_NBYTES:]):
            raise ValueError("slab preamble padding must be zero")
        return cls(
            generation=generation,
            record_count=count,
            origin_rank=origin,
            destination_rank=destination,
            element_size=element_size,
        )


@dataclass(frozen=True)
class RecordLocation:
    """Byte ranges for one record in a destination's arena."""

    peer_rank: int
    slot: int
    header_offset: int
    header_nbytes: int
    payload_offset: int
    payload_nbytes: int

    @property
    def header_end(self) -> int:
        return self.header_offset + self.header_nbytes

    @property
    def payload_end(self) -> int:
        return self.payload_offset + self.payload_nbytes


@dataclass(frozen=True)
class PeerSlab:
    """The immutable slab assigned to one peer rank."""

    peer_rank: int
    offset: int
    nbytes: int
    preamble_offset: int
    preamble_nbytes: int
    headers_offset: int
    headers_nbytes: int
    payloads_offset: int
    payloads_nbytes: int
    record_capacity: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class PeerSlabLayout:
    """Padded, fixed-capacity layout used by dispatch remote writes.

    ``peer_rank`` means the origin rank when this layout addresses a receive
    arena.  Headers and payloads live in separate arrays so payload alignment
    does not make every 24-byte header consume a full payload cache line.
    """

    max_ranks: int
    max_tokens_per_rank: int
    top_k: int
    hidden_size: int
    element_size: int
    preamble_size: int = 32
    header_size: int = DISPATCH_HEADER_NBYTES
    header_alignment: int = 16
    payload_alignment: int = 128
    slab_alignment: int = 4096
    address_limit: int = field(default=UINT64_MAX, repr=False)

    record_capacity: int = field(init=False)
    header_stride: int = field(init=False)
    payload_stride: int = field(init=False)
    payload_nbytes: int = field(init=False)
    headers_region_offset: int = field(init=False)
    payload_region_offset: int = field(init=False)
    slab_nbytes: int = field(init=False)
    arena_nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        _checked_positive("max_ranks", self.max_ranks, UINT32_MAX + 1)
        _checked_positive("max_tokens_per_rank", self.max_tokens_per_rank, UINT32_MAX)
        _checked_positive("top_k", self.top_k, UINT32_MAX)
        _checked_positive("hidden_size", self.hidden_size, UINT32_MAX)
        _checked_positive("element_size", self.element_size, UINT32_MAX)
        _checked_positive("preamble_size", self.preamble_size, UINT32_MAX)
        if self.preamble_size < SLAB_PREAMBLE_WIRE_NBYTES:
            raise ValueError(
                f"preamble_size must be at least {SLAB_PREAMBLE_WIRE_NBYTES} bytes"
            )
        _checked_positive("header_size", self.header_size, UINT32_MAX)
        if self.header_size < DISPATCH_HEADER_NBYTES:
            raise ValueError(
                f"header_size must be at least {DISPATCH_HEADER_NBYTES} bytes"
            )
        _require_power_of_two("header_alignment", self.header_alignment)
        _require_power_of_two("payload_alignment", self.payload_alignment)
        _require_power_of_two("slab_alignment", self.slab_alignment)
        _checked_positive("address_limit", self.address_limit, UINT64_MAX)

        capacity = _checked_mul(
            self.max_tokens_per_rank,
            self.top_k,
            self.address_limit,
            "record capacity",
        )
        payload_nbytes = _checked_mul(
            self.hidden_size,
            self.element_size,
            self.address_limit,
            "payload size",
        )
        header_stride = align_up(
            self.header_size, self.header_alignment, limit=self.address_limit
        )
        payload_stride = align_up(
            payload_nbytes, self.payload_alignment, limit=self.address_limit
        )
        headers_region_offset = align_up(
            self.preamble_size, self.header_alignment, limit=self.address_limit
        )
        headers_nbytes = _checked_mul(
            capacity, header_stride, self.address_limit, "header region"
        )
        headers_end = _checked_add(
            headers_region_offset,
            headers_nbytes,
            self.address_limit,
            "preamble and header regions",
        )
        payload_region_offset = align_up(
            headers_end, self.payload_alignment, limit=self.address_limit
        )
        payloads_nbytes = _checked_mul(
            capacity, payload_stride, self.address_limit, "payload region"
        )
        slab_unaligned = _checked_add(
            payload_region_offset,
            payloads_nbytes,
            self.address_limit,
            "peer slab",
        )
        slab_nbytes = align_up(
            slab_unaligned, self.slab_alignment, limit=self.address_limit
        )
        arena_nbytes = _checked_mul(
            self.max_ranks, slab_nbytes, self.address_limit, "arena"
        )

        object.__setattr__(self, "record_capacity", capacity)
        object.__setattr__(self, "header_stride", header_stride)
        object.__setattr__(self, "payload_stride", payload_stride)
        object.__setattr__(self, "payload_nbytes", payload_nbytes)
        object.__setattr__(self, "headers_region_offset", headers_region_offset)
        object.__setattr__(self, "payload_region_offset", payload_region_offset)
        object.__setattr__(self, "slab_nbytes", slab_nbytes)
        object.__setattr__(self, "arena_nbytes", arena_nbytes)

    def slab(self, peer_rank: int) -> PeerSlab:
        """Return the peer slab, validating its fixed-capacity rank ID."""

        _checked_uint("peer_rank", peer_rank, UINT32_MAX)
        if peer_rank >= self.max_ranks:
            raise IndexError(
                f"peer_rank {peer_rank} is outside max_ranks={self.max_ranks}"
            )
        offset = peer_rank * self.slab_nbytes
        headers_nbytes = self.record_capacity * self.header_stride
        payloads_offset = offset + self.payload_region_offset
        return PeerSlab(
            peer_rank=peer_rank,
            offset=offset,
            nbytes=self.slab_nbytes,
            preamble_offset=offset,
            preamble_nbytes=self.preamble_size,
            headers_offset=offset + self.headers_region_offset,
            headers_nbytes=headers_nbytes,
            payloads_offset=payloads_offset,
            payloads_nbytes=self.record_capacity * self.payload_stride,
            record_capacity=self.record_capacity,
        )

    def encode_preamble(
        self,
        *,
        origin_rank: int,
        destination_rank: int,
        generation: int,
        record_count: int,
    ) -> bytes:
        """Encode a slab preamble after validating this layout's capacity."""

        self.slab(origin_rank)
        self.slab(destination_rank)
        _checked_uint("generation", generation, UINT64_MAX)
        _checked_uint("record_count", record_count, UINT32_MAX)
        if record_count > self.record_capacity:
            raise ValueError(
                f"record_count {record_count} exceeds capacity {self.record_capacity}"
            )
        return SlabPreamble(
            generation,
            record_count,
            origin_rank,
            destination_rank,
            self.element_size,
        ).to_bytes(self.preamble_size)

    def decode_preamble(
        self,
        data: bytes,
        *,
        expected_origin_rank: int | None = None,
        expected_destination_rank: int | None = None,
        expected_generation: int | None = None,
    ) -> SlabPreamble:
        """Decode a complete layout-sized preamble and validate expectations."""

        if len(data) != self.preamble_size:
            raise ValueError(
                f"preamble has {len(data)} bytes, expected {self.preamble_size}"
            )
        preamble = SlabPreamble.from_bytes(data)
        self.slab(preamble.origin_rank)
        self.slab(preamble.destination_rank)
        if preamble.element_size != self.element_size:
            raise ValueError(
                f"preamble element size {preamble.element_size} does not match layout "
                f"element size {self.element_size}"
            )
        if preamble.record_count > self.record_capacity:
            raise ValueError(
                f"record_count {preamble.record_count} exceeds capacity "
                f"{self.record_capacity}"
            )
        if expected_origin_rank is not None:
            self.slab(expected_origin_rank)
            if preamble.origin_rank != expected_origin_rank:
                raise ValueError(
                    f"preamble origin {preamble.origin_rank} does not match expected "
                    f"origin {expected_origin_rank}"
                )
        if expected_destination_rank is not None:
            self.slab(expected_destination_rank)
            if preamble.destination_rank != expected_destination_rank:
                raise ValueError(
                    f"preamble destination {preamble.destination_rank} does not match "
                    f"expected destination {expected_destination_rank}"
                )
        if expected_generation is not None:
            _checked_uint("expected_generation", expected_generation, UINT64_MAX)
            if preamble.generation != expected_generation:
                raise ValueError(
                    f"preamble generation {preamble.generation} does not match "
                    f"expected generation {expected_generation}"
                )
        return preamble

    def record(self, peer_rank: int, slot: int) -> RecordLocation:
        """Return the header and payload byte ranges for a peer-local slot."""

        slab = self.slab(peer_rank)
        _checked_uint("slot", slot, UINT32_MAX)
        if slot >= self.record_capacity:
            raise IndexError(
                f"slot {slot} is outside record_capacity={self.record_capacity}"
            )
        return RecordLocation(
            peer_rank=peer_rank,
            slot=slot,
            header_offset=slab.headers_offset + slot * self.header_stride,
            header_nbytes=self.header_size,
            payload_offset=slab.payloads_offset + slot * self.payload_stride,
            payload_nbytes=self.payload_nbytes,
        )

    def validate_span(self, offset: int, nbytes: int) -> None:
        """Reject a byte span that is not fully contained by this arena."""

        _checked_uint("offset", offset, self.address_limit)
        _checked_uint("nbytes", nbytes, self.address_limit)
        end = _checked_add(offset, nbytes, self.address_limit, "byte span")
        if end > self.arena_nbytes:
            raise ValueError(
                f"byte span [{offset}, {end}) exceeds arena size {self.arena_nbytes}"
            )
