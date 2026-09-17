# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked host model for the banked NIXL CuTe low-latency MoE protocol.

This module deliberately depends only on the Python standard library.  It is
the executable contracts shared by the CuTe examples and their host runtime:

* rank and expert identifiers remain stable across sparse membership changes;
* the general request-based model has two banks and returned credits;
* the compact mapped model has one production bank by default (or two for A/B)
  and a closed dispatch/combine-ready reuse proof;
* even an empty bucket publishes a count-plus-one ready value; and
* a wire operation is accepted only for the expected membership generation and
  source-process incarnation.

No class below owns memory or performs communication.  Byte offsets, wire
stamps, schedules, and state transitions are kept here so they can be tested on
a machine without CUDA, CuTe DSL, or a NIXL device backend.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from itertools import pairwise
from typing import Iterable, Mapping, Sequence

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

NUM_BANKS = 2
# Correctness-only expert transform shared by the import-safe host oracle and
# the CuTe device kernel.  Keep it in this standard-library-only contract
# module so importing the process launcher cannot initialize CuTe before each
# spawned worker installs its rank-private code-generation environment.
STANDIN_EXPERT_BIAS_SCALE = 64
DOORBELL_INTERVAL = 4
DEFAULT_GDA_CHANNELS = 4
MAX_GDA_CHANNELS = 256
COUNTER_NBYTES = 8
MESSAGE_STAMP_NBYTES = 16
PIPELINE_BUCKET_STAMP_NBYTES = 32
DISPATCH_METADATA_NBYTES = 16
DISPATCH_HEADER_NBYTES = 32
SOURCE_INFO_NBYTES = 16

_MESSAGE_STAMP = struct.Struct("<QQ")
_PIPELINE_BUCKET_STAMP = struct.Struct("<QQQQ")
_DISPATCH_HEADER = struct.Struct("<QQII")
_PIPELINE_SOURCE_INFO = struct.Struct("<IIII")
assert _MESSAGE_STAMP.size == MESSAGE_STAMP_NBYTES
assert _PIPELINE_BUCKET_STAMP.size == PIPELINE_BUCKET_STAMP_NBYTES
assert _DISPATCH_HEADER.size <= DISPATCH_HEADER_NBYTES
assert _PIPELINE_SOURCE_INFO.size == SOURCE_INFO_NBYTES


class ProtocolError(RuntimeError):
    """An ordering, identity, or lifecycle invariant was violated."""


def _uint(name: str, value: int, maximum: int = UINT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _positive(name: str, value: int, maximum: int = UINT64_MAX) -> int:
    _uint(name, value, maximum)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _power_of_two(name: str, value: int) -> int:
    _positive(name, value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value


def _checked_add(lhs: int, rhs: int, limit: int, expression: str) -> int:
    if lhs > limit - rhs:
        raise OverflowError(f"{expression} exceeds address limit {limit}")
    return lhs + rhs


def _checked_mul(lhs: int, rhs: int, limit: int, expression: str) -> int:
    if lhs and rhs > limit // lhs:
        raise OverflowError(f"{expression} exceeds address limit {limit}")
    return lhs * rhs


def _checked_product(*factors: int, limit: int, expression: str) -> int:
    """Multiply non-negative factors without exceeding ``limit``."""

    result = 1
    for factor in factors:
        result = _checked_mul(result, factor, limit, expression)
    return result


def _align_up(value: int, alignment: int, limit: int, expression: str) -> int:
    _uint("value", value, limit)
    _power_of_two("alignment", alignment)
    padding = (-value) & (alignment - 1)
    return _checked_add(value, padding, limit, expression)


def _index(name: str, value: int, bound: int) -> int:
    _uint(name, value, UINT32_MAX)
    if value >= bound:
        raise IndexError(f"{name} {value} is outside [0, {bound})")
    return value


class Direction(str, Enum):
    """The independent dispatch and combine credit domains."""

    DISPATCH = "dispatch"
    COMBINE = "combine"


class PostFlag(IntEnum):
    """Values intentionally matching ``nixl.device.cute.Flags``."""

    NONE = 0
    DEFER = 1


@dataclass(frozen=True, slots=True, order=True)
class OperationEpoch:
    """A membership generation and operation step packed into one u64.

    A step is local to a membership generation.  Packing both halves makes a
    stale message distinguishable after a bank wraps from step ``n`` to
    ``n + 2``.
    """

    membership_generation: int
    step: int

    def __post_init__(self) -> None:
        _uint("membership_generation", self.membership_generation, UINT32_MAX)
        _uint("step", self.step, UINT32_MAX)

    @property
    def wire_value(self) -> int:
        return (self.membership_generation << 32) | self.step

    @property
    def bank(self) -> int:
        return self.step & 1

    @property
    def bank_sequence(self) -> int:
        """Monotonic one-based publication sequence within this bank."""

        return self.step // NUM_BANKS + 1

    @classmethod
    def from_wire(cls, value: int) -> "OperationEpoch":
        _uint("wire operation epoch", value, UINT64_MAX)
        return cls(value >> 32, value & UINT32_MAX)


@dataclass(frozen=True, slots=True)
class StableSparseTopology:
    """Fixed rank/expert capacity plus one elastic membership generation.

    ``rank_incarnations`` identifies the process currently associated with each
    stable rank slot.  Zero means that no process incarnation has ever been
    installed in that slot; active ranks therefore require a positive value.
    A still-live standby may rejoin with the same incarnation.  A restarted
    process must use a larger value.
    """

    max_ranks: int
    experts_per_rank: int
    active_ranks: tuple[int, ...]
    membership_generation: int
    rank_incarnations: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, UINT32_MAX)
        _positive("experts_per_rank", self.experts_per_rank, UINT32_MAX)
        _uint("membership_generation", self.membership_generation, UINT32_MAX)
        if self.max_ranks > (UINT32_MAX + 1) // self.experts_per_rank:
            raise OverflowError("fixed expert namespace does not fit in u32")

        ranks = tuple(self.active_ranks)
        if not ranks:
            raise ValueError("at least one rank must be active")
        for rank in ranks:
            _index("active rank", rank, self.max_ranks)
        if len(set(ranks)) != len(ranks):
            raise ValueError("active_ranks contains duplicates")
        object.__setattr__(self, "active_ranks", tuple(sorted(ranks)))

        incarnations = tuple(self.rank_incarnations)
        if len(incarnations) != self.max_ranks:
            raise ValueError(
                "rank_incarnations must contain exactly one value per stable "
                f"rank slot ({self.max_ranks}), got {len(incarnations)}"
            )
        for rank, incarnation in enumerate(incarnations):
            _uint(f"rank_incarnations[{rank}]", incarnation, UINT64_MAX)
        for rank in ranks:
            if incarnations[rank] == 0:
                raise ValueError(f"active rank {rank} has no process incarnation")
        object.__setattr__(self, "rank_incarnations", incarnations)

    @property
    def active_mask(self) -> tuple[bool, ...]:
        active = set(self.active_ranks)
        return tuple(rank in active for rank in range(self.max_ranks))

    @property
    def nixl_mask(self) -> tuple[int, ...]:
        """Return the native NIXL EP convention: zero active, one masked."""

        return tuple(0 if active else 1 for active in self.active_mask)

    @property
    def rank_bound(self) -> int:
        """Exclusive sparse loop bound; holes below it remain addressable."""

        return self.active_ranks[-1] + 1

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    @property
    def active_experts(self) -> tuple[int, ...]:
        return tuple(
            self.expert_id(rank, local_expert)
            for rank in self.active_ranks
            for local_expert in range(self.experts_per_rank)
        )

    def is_active(self, rank: int) -> bool:
        _index("rank", rank, self.max_ranks)
        # ``active_ranks`` is already canonical and immutable.  Avoid building
        # a transient set plus a full-capacity tuple on every hot host enqueue.
        return rank in self.active_ranks

    def incarnation(self, rank: int) -> int:
        _index("rank", rank, self.max_ranks)
        return self.rank_incarnations[rank]

    def expert_id(self, rank: int, local_expert: int) -> int:
        _index("rank", rank, self.max_ranks)
        _index("local_expert", local_expert, self.experts_per_rank)
        return rank * self.experts_per_rank + local_expert

    def owner(self, expert: int) -> int:
        _index("expert", expert, self.max_experts)
        return expert // self.experts_per_rank

    def local_expert(self, expert: int) -> int:
        _index("expert", expert, self.max_experts)
        return expert % self.experts_per_rank

    def transition(
        self,
        active_ranks: Iterable[int],
        *,
        membership_generation: int | None = None,
        rank_incarnations: Sequence[int] | None = None,
    ) -> "StableSparseTopology":
        """Create a later membership without renumbering any stable slot."""

        generation = (
            self.membership_generation + 1
            if membership_generation is None
            else membership_generation
        )
        _uint("membership_generation", generation, UINT32_MAX)
        if generation <= self.membership_generation:
            raise ValueError(
                "membership_generation must advance beyond "
                f"{self.membership_generation}"
            )
        incarnations = (
            self.rank_incarnations
            if rank_incarnations is None
            else tuple(rank_incarnations)
        )
        next_topology = StableSparseTopology(
            max_ranks=self.max_ranks,
            experts_per_rank=self.experts_per_rank,
            active_ranks=tuple(active_ranks),
            membership_generation=generation,
            rank_incarnations=tuple(incarnations),
        )
        for rank, (old, new) in enumerate(
            zip(self.rank_incarnations, next_topology.rank_incarnations)
        ):
            if new < old:
                raise ValueError(
                    f"rank {rank} incarnation moved backwards from {old} to {new}"
                )
        return next_topology


@dataclass(frozen=True, slots=True)
class MessageStamp:
    """Generation/incarnation identity for one release-published message unit."""

    operation: OperationEpoch
    source_incarnation: int

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationEpoch):
            raise TypeError("operation must be an OperationEpoch")
        _positive("source_incarnation", self.source_incarnation, UINT64_MAX)

    def to_bytes(self) -> bytes:
        return _MESSAGE_STAMP.pack(self.operation.wire_value, self.source_incarnation)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MessageStamp":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) != MESSAGE_STAMP_NBYTES:
            raise ValueError(
                f"message stamp has {len(data)} bytes, expected "
                f"{MESSAGE_STAMP_NBYTES}"
            )
        epoch, incarnation = _MESSAGE_STAMP.unpack(data)
        return cls(OperationEpoch.from_wire(epoch), incarnation)

    def validate(
        self,
        topology: StableSparseTopology,
        *,
        source_rank: int,
        expected_operation: OperationEpoch,
    ) -> None:
        """Reject stale banks, inactive senders, and replaced processes."""

        if not isinstance(topology, StableSparseTopology):
            raise TypeError("topology must be a StableSparseTopology")
        if not isinstance(expected_operation, OperationEpoch):
            raise TypeError("expected_operation must be an OperationEpoch")
        _index("source_rank", source_rank, topology.max_ranks)
        if expected_operation.membership_generation != topology.membership_generation:
            raise ProtocolError(
                "expected operation does not belong to the committed membership "
                "generation"
            )
        if self.operation != expected_operation:
            raise ProtocolError(
                f"stale operation epoch {self.operation.wire_value}, expected "
                f"{expected_operation.wire_value}"
            )
        if not topology.is_active(source_rank):
            raise ProtocolError(f"source rank {source_rank} is masked")
        expected_incarnation = topology.incarnation(source_rank)
        if self.source_incarnation != expected_incarnation:
            raise ProtocolError(
                f"source rank {source_rank} incarnation "
                f"{self.source_incarnation} does not match committed "
                f"incarnation {expected_incarnation}"
            )


@dataclass(frozen=True, slots=True)
class PipelineBucketStamp:
    """Ordered identity and record count for one split-phase bucket.

    The split-phase network protocol cannot encode ``count + 1`` directly in
    its publication word: requestless NIXL publication uses an atomic add, so
    the ready word is a monotonic sequence number.  The producer therefore
    writes this stamp first and publishes the matching sequence last on the
    same transport channel.  A mapped producer performs the equivalent
    release store.  The consumer acquire-waits for the sequence, then validates
    the operation, incarnation, and count before touching payload bytes.

    The final word is a fail-closed transport bit.  A producer still publishes
    the bucket after a copy failure so its peer cannot deadlock waiting for a
    sequence that will never arrive; the consumer observes the publication and
    rejects the stamped operation before reading payload bytes.
    """

    stamp: MessageStamp
    record_count: int
    failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stamp, MessageStamp):
            raise TypeError("stamp must be a MessageStamp")
        _uint("record_count", self.record_count, UINT32_MAX)
        if not isinstance(self.failed, bool):
            raise TypeError("failed must be a bool")

    def to_bytes(self) -> bytes:
        encoded = _PIPELINE_BUCKET_STAMP.pack(
            self.stamp.operation.wire_value,
            self.stamp.source_incarnation,
            self.record_count,
            int(self.failed),
        )
        return encoded

    @classmethod
    def from_bytes(cls, data: bytes) -> "PipelineBucketStamp":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) != PIPELINE_BUCKET_STAMP_NBYTES:
            raise ValueError(
                f"pipeline bucket stamp has {len(data)} bytes, expected "
                f"{PIPELINE_BUCKET_STAMP_NBYTES}"
            )
        epoch, incarnation, record_count, failed = _PIPELINE_BUCKET_STAMP.unpack(data)
        if failed not in (0, 1):
            raise ValueError("pipeline bucket stamp failure word must be zero or one")
        return cls(
            MessageStamp(OperationEpoch.from_wire(epoch), incarnation),
            record_count,
            bool(failed),
        )

    def validate(
        self,
        topology: StableSparseTopology,
        *,
        source_rank: int,
        expected_operation: OperationEpoch,
        capacity: int,
    ) -> None:
        """Validate identity and ensure a peer cannot overrun its fixed bucket."""

        _uint("capacity", capacity, UINT32_MAX)
        self.stamp.validate(
            topology,
            source_rank=source_rank,
            expected_operation=expected_operation,
        )
        if self.failed:
            raise ProtocolError("producer reported a bucket transport failure")
        if self.record_count > capacity:
            raise ProtocolError(
                f"bucket count {self.record_count} exceeds capacity {capacity}"
            )


@dataclass(frozen=True, slots=True)
class PipelineSourceInfo:
    """Origin identity carried beside one expert-major payload row."""

    origin_rank: int
    origin_token: int
    route_slot: int

    def __post_init__(self) -> None:
        _uint("origin_rank", self.origin_rank, UINT32_MAX)
        _uint("origin_token", self.origin_token, UINT32_MAX)
        _uint("route_slot", self.route_slot, UINT32_MAX)

    def to_bytes(self) -> bytes:
        return _PIPELINE_SOURCE_INFO.pack(
            self.origin_rank,
            self.origin_token,
            self.route_slot,
            0,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PipelineSourceInfo":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) != SOURCE_INFO_NBYTES:
            raise ValueError(
                f"pipeline source info has {len(data)} bytes, expected "
                f"{SOURCE_INFO_NBYTES}"
            )
        origin_rank, origin_token, route_slot, reserved = _PIPELINE_SOURCE_INFO.unpack(
            data
        )
        if reserved:
            raise ValueError("pipeline source info reserved word must be zero")
        return cls(origin_rank, origin_token, route_slot)


def pack_layout_range(begin: int, count: int) -> int:
    """Pack one expert/source ``(begin, count)`` pair into a u64 word."""

    _uint("layout range begin", begin, UINT32_MAX)
    _uint("layout range count", count, UINT32_MAX)
    if count > UINT32_MAX - begin:
        raise OverflowError("layout range end exceeds uint32")
    return (begin << 32) | count


def unpack_layout_range(value: int) -> tuple[int, int]:
    """Decode the exact u64 representation produced by pack_layout_range."""

    _uint("layout range word", value, UINT64_MAX)
    return value >> 32, value & UINT32_MAX


@dataclass(frozen=True, slots=True)
class DispatchHeader:
    """Minimal fixed-size dispatch control header.

    Source rank is implied by the single-writer receive slab, and destination
    expert is implied by that slab's local-expert bucket.  The reserved field
    and all alignment padding must remain zero so the wire format can evolve
    without making old readers silently misinterpret new data.
    """

    stamp: MessageStamp
    origin_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.stamp, MessageStamp):
            raise TypeError("stamp must be a MessageStamp")
        _uint("origin_token", self.origin_token, UINT32_MAX)

    def to_bytes(self) -> bytes:
        encoded = _DISPATCH_HEADER.pack(
            self.stamp.operation.wire_value,
            self.stamp.source_incarnation,
            self.origin_token,
            0,
        )
        return encoded + bytes(DISPATCH_HEADER_NBYTES - len(encoded))

    @classmethod
    def from_bytes(cls, data: bytes) -> "DispatchHeader":
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) != DISPATCH_HEADER_NBYTES:
            raise ValueError(
                f"dispatch header has {len(data)} bytes, expected "
                f"{DISPATCH_HEADER_NBYTES}"
            )
        epoch, incarnation, origin_token, reserved = _DISPATCH_HEADER.unpack_from(data)
        if reserved != 0 or any(data[_DISPATCH_HEADER.size :]):
            raise ValueError("dispatch header reserved bytes must be zero")
        return cls(
            MessageStamp(OperationEpoch.from_wire(epoch), incarnation),
            origin_token,
        )


@dataclass(frozen=True, slots=True)
class BucketKey:
    """A channel has an independent ordering and publication domain."""

    direction: Direction
    bank: int
    peer_rank: int
    channel: int

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")
        _index("bank", self.bank, NUM_BANKS)
        _uint("peer_rank", self.peer_rank, UINT32_MAX)
        _uint("channel", self.channel, UINT32_MAX)


@dataclass(frozen=True, slots=True)
class DataPost:
    slot: int
    flag: PostFlag

    def __post_init__(self) -> None:
        _uint("slot", self.slot, UINT32_MAX)
        if not isinstance(self.flag, PostFlag):
            raise TypeError("flag must be a PostFlag")
        expected = data_post_flag(self.slot)
        if self.flag != expected:
            raise ValueError(
                f"slot {self.slot} requires {expected.name}, got {self.flag.name}"
            )


@dataclass(frozen=True, slots=True)
class PublicationPost:
    """The final same-channel atomic; ``value - 1`` is the record count."""

    value: int
    flag: PostFlag = PostFlag.NONE

    def __post_init__(self) -> None:
        _positive("publication value", self.value, UINT32_MAX + 1)
        if not isinstance(self.flag, PostFlag):
            raise TypeError("flag must be a PostFlag")
        if self.flag != PostFlag.NONE:
            raise ValueError("a publication post must be non-deferred")


@dataclass(frozen=True, slots=True)
class BucketSchedule:
    """All data posts and the mandatory final publication for one bucket."""

    key: BucketKey
    data_posts: tuple[DataPost, ...]
    publication: PublicationPost

    def __post_init__(self) -> None:
        if not isinstance(self.key, BucketKey):
            raise TypeError("key must be a BucketKey")
        posts = tuple(self.data_posts)
        if any(not isinstance(post, DataPost) for post in posts):
            raise TypeError("data_posts must contain only DataPost values")
        for slot, post in enumerate(posts):
            if post.slot != slot:
                raise ValueError("data post slots must be contiguous and start at zero")
        if not isinstance(self.publication, PublicationPost):
            raise TypeError("publication must be a PublicationPost")
        if self.publication.value != len(posts) + 1:
            raise ValueError("publication must encode record_count + 1")
        object.__setattr__(self, "data_posts", posts)

    @property
    def record_count(self) -> int:
        return len(self.data_posts)


def data_post_flag(slot: int) -> PostFlag:
    """Return the native EP three-DEFER/one-NONE cadence for a bucket slot."""

    _uint("slot", slot, UINT32_MAX)
    return PostFlag.NONE if (slot + 1) % DOORBELL_INTERVAL == 0 else PostFlag.DEFER


def make_bucket_schedule(key: BucketKey, record_count: int) -> BucketSchedule:
    """Build one independent peer/channel schedule.

    The final publication is present for ``record_count == 0`` and carries one,
    preserving the native ``count + 1`` encoding where zero means not ready.
    """

    if not isinstance(key, BucketKey):
        raise TypeError("key must be a BucketKey")
    _uint("record_count", record_count, UINT32_MAX)
    posts = tuple(DataPost(slot, data_post_flag(slot)) for slot in range(record_count))
    return BucketSchedule(key, posts, PublicationPost(record_count + 1))


def normalize_gda_channel_count(requested: int) -> int:
    """Mirror UCX GDA's power-of-two channel normalization, fail closed.

    UCX masks device channels and supports at most 256.  Exposing the normalized
    value in the host protocol keeps payload, stamp, publication, and returned
    credit on the exact same channel without paying modulo or policy decisions
    in the generated kernel.
    """

    _positive("requested channel count", requested, MAX_GDA_CHANNELS)
    return 1 << (requested - 1).bit_length()


def gda_bucket_channel(
    *, peer_rank: int, bucket_ordinal: int, channel_count: int
) -> int:
    """Choose one stable same-bucket channel for ordered requestless posts."""

    _uint("peer_rank", peer_rank, UINT32_MAX)
    _uint("bucket_ordinal", bucket_ordinal, UINT32_MAX)
    _power_of_two("channel_count", channel_count)
    if channel_count > MAX_GDA_CHANNELS:
        raise ValueError(f"channel_count must not exceed {MAX_GDA_CHANNELS}")
    return (peer_rank + bucket_ordinal) & (channel_count - 1)


@dataclass(frozen=True, slots=True)
class ByteSpan:
    """One checked half-open byte interval in the registered arena."""

    name: str
    offset: int
    nbytes: int
    alignment: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("span name must be a non-empty string")
        _uint("offset", self.offset, UINT64_MAX)
        _positive("nbytes", self.nbytes, UINT64_MAX)
        _power_of_two("alignment", self.alignment)
        _checked_add(self.offset, self.nbytes, UINT64_MAX, f"span {self.name!r}")
        if self.offset % self.alignment:
            raise ValueError(
                f"span {self.name!r} offset {self.offset} is not aligned to "
                f"{self.alignment}"
            )

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def overlaps(self, other: "ByteSpan") -> bool:
        if not isinstance(other, ByteSpan):
            raise TypeError("other must be a ByteSpan")
        return self.offset < other.end and other.offset < self.end


_BANKED_REGIONS = frozenset(
    {
        "dispatch_stage",
        "dispatch_recv",
        "dispatch_ready",
        "dispatch_credit",
        "expert_input",
        "dispatch_src_info",
        "dispatch_layout_range",
        "combine_stage",
        "combine_recv",
        "combine_ready",
        "combine_credit",
    }
)


@dataclass(frozen=True, slots=True)
class LLArenaLayout:
    """Fixed-capacity two-bank byte layout for one rank's registered arena."""

    max_ranks: int
    experts_per_rank: int
    max_tokens_per_rank: int
    top_k: int
    hidden_size: int
    element_size: int
    record_alignment: int = 16
    region_alignment: int = 128
    arena_alignment: int = 4096
    address_limit: int = field(default=UINT64_MAX, repr=False)

    payload_nbytes: int = field(init=False)
    payload_stride: int = field(init=False)
    dispatch_record_nbytes: int = field(init=False)
    dispatch_record_stride: int = field(init=False)
    combine_record_nbytes: int = field(init=False)
    combine_record_stride: int = field(init=False)
    regions: tuple[ByteSpan, ...] = field(init=False)
    arena_nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, UINT32_MAX)
        _positive("experts_per_rank", self.experts_per_rank, UINT32_MAX)
        _positive("max_tokens_per_rank", self.max_tokens_per_rank, UINT32_MAX)
        _positive("top_k", self.top_k, UINT32_MAX)
        _positive("hidden_size", self.hidden_size, UINT32_MAX)
        _positive("element_size", self.element_size, UINT32_MAX)
        _power_of_two("record_alignment", self.record_alignment)
        _power_of_two("region_alignment", self.region_alignment)
        _power_of_two("arena_alignment", self.arena_alignment)
        _positive("address_limit", self.address_limit, UINT64_MAX)
        if self.record_alignment < 16:
            raise ValueError("record_alignment must satisfy NIXL's 16-byte PUT ABI")
        if self.record_alignment > self.region_alignment:
            raise ValueError("record_alignment must not exceed region_alignment")
        if self.region_alignment > self.arena_alignment:
            raise ValueError("region_alignment must not exceed arena_alignment")
        if self.max_ranks > (UINT32_MAX + 1) // self.experts_per_rank:
            raise OverflowError("fixed expert namespace does not fit in u32")
        max_experts = self.max_ranks * self.experts_per_rank
        if self.top_k > max_experts:
            raise ValueError(
                f"top_k {self.top_k} exceeds fixed expert capacity {max_experts}"
            )

        payload_nbytes = _checked_mul(
            self.hidden_size,
            self.element_size,
            self.address_limit,
            "token payload",
        )
        payload_stride = _align_up(
            payload_nbytes,
            self.record_alignment,
            self.address_limit,
            "token payload stride",
        )
        dispatch_record_nbytes = _checked_add(
            DISPATCH_HEADER_NBYTES,
            payload_nbytes,
            self.address_limit,
            "dispatch record",
        )
        dispatch_record_stride = _align_up(
            dispatch_record_nbytes,
            self.record_alignment,
            self.address_limit,
            "dispatch record stride",
        )
        combine_record_nbytes = _checked_add(
            MESSAGE_STAMP_NBYTES,
            payload_nbytes,
            self.address_limit,
            "combine record",
        )
        combine_record_stride = _align_up(
            combine_record_nbytes,
            self.record_alignment,
            self.address_limit,
            "combine record stride",
        )

        object.__setattr__(self, "payload_nbytes", payload_nbytes)
        object.__setattr__(self, "payload_stride", payload_stride)
        object.__setattr__(self, "dispatch_record_nbytes", dispatch_record_nbytes)
        object.__setattr__(self, "dispatch_record_stride", dispatch_record_stride)
        object.__setattr__(self, "combine_record_nbytes", combine_record_nbytes)
        object.__setattr__(self, "combine_record_stride", combine_record_stride)

        rank_expert_slots = _checked_mul(
            self.max_ranks,
            self.experts_per_rank,
            self.address_limit,
            "rank/expert slots",
        )
        rank_expert_token_slots = _checked_mul(
            rank_expert_slots,
            self.max_tokens_per_rank,
            self.address_limit,
            "rank/expert/token slots",
        )

        def size(name: str, *factors: int) -> tuple[str, int]:
            return (
                name,
                _checked_product(
                    *factors,
                    limit=self.address_limit,
                    expression=f"{name} size",
                ),
            )

        sizes = (
            size("control", 3, COUNTER_NBYTES),
            size("rank_mask", self.max_ranks, COUNTER_NBYTES),
            size("rank_incarnation", self.max_ranks, COUNTER_NBYTES),
            size(
                "dispatch_stage",
                NUM_BANKS,
                self.max_tokens_per_rank,
                dispatch_record_stride,
            ),
            size(
                "dispatch_recv",
                NUM_BANKS,
                rank_expert_token_slots,
                dispatch_record_stride,
            ),
            size(
                "dispatch_ready",
                NUM_BANKS,
                rank_expert_slots,
                COUNTER_NBYTES,
            ),
            size(
                "dispatch_credit",
                NUM_BANKS,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
            size(
                "expert_input",
                NUM_BANKS,
                rank_expert_token_slots,
                payload_stride,
            ),
            size(
                "dispatch_src_info",
                NUM_BANKS,
                rank_expert_token_slots,
                SOURCE_INFO_NBYTES,
            ),
            size(
                "dispatch_layout_range",
                NUM_BANKS,
                rank_expert_slots,
                COUNTER_NBYTES,
            ),
            size(
                "combine_stage",
                NUM_BANKS,
                rank_expert_token_slots,
                combine_record_stride,
            ),
            size(
                "combine_recv",
                NUM_BANKS,
                rank_expert_token_slots,
                combine_record_stride,
            ),
            size(
                "combine_ready",
                NUM_BANKS,
                rank_expert_slots,
                COUNTER_NBYTES,
            ),
            size(
                "combine_credit",
                NUM_BANKS,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
        )

        cursor = 0
        regions: list[ByteSpan] = []
        for name, nbytes in sizes:
            _positive(f"{name} size", nbytes, self.address_limit)
            cursor = _align_up(
                cursor,
                self.region_alignment,
                self.address_limit,
                f"{name} start",
            )
            end = _checked_add(cursor, nbytes, self.address_limit, name)
            regions.append(ByteSpan(name, cursor, nbytes, self.region_alignment))
            cursor = end
        arena_nbytes = _align_up(
            cursor,
            self.arena_alignment,
            self.address_limit,
            "arena size",
        )

        for left, right in pairwise(regions):
            if left.overlaps(right):
                raise AssertionError(f"arena regions overlap: {left} and {right}")
        object.__setattr__(self, "regions", tuple(regions))
        object.__setattr__(self, "arena_nbytes", arena_nbytes)

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    def region(self, name: str) -> ByteSpan:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown arena region {name!r}")

    def bank_region(self, name: str, bank: int) -> ByteSpan:
        """Return one half of a banked top-level region."""

        _index("bank", bank, NUM_BANKS)
        if name not in _BANKED_REGIONS:
            raise ValueError(f"region {name!r} is not banked")
        region = self.region(name)
        if region.nbytes % NUM_BANKS:
            raise AssertionError(f"banked region {name!r} is not evenly divisible")
        bank_nbytes = region.nbytes // NUM_BANKS
        # The grouping span carries no access-granularity promise.  Individual
        # records below retain their required 8- or 16-byte alignment.
        span_offset = region.offset + bank * bank_nbytes
        span_alignment = self.region_alignment
        while span_offset % span_alignment:
            span_alignment //= 2
        return ByteSpan(
            f"{name}[{bank}]",
            span_offset,
            bank_nbytes,
            span_alignment,
        )

    def _item(
        self,
        region_name: str,
        item_name: str,
        item_index: int,
        item_nbytes: int,
        item_alignment: int,
    ) -> ByteSpan:
        region = self.region(region_name)
        offset = region.offset + item_index * item_nbytes
        end = offset + item_nbytes
        if offset < region.offset or end > region.end:
            raise AssertionError(
                f"computed {item_name} [{offset}, {end}) outside {region_name} "
                f"[{region.offset}, {region.end})"
            )
        return ByteSpan(item_name, offset, item_nbytes, item_alignment)

    def control_word(self, index: int) -> ByteSpan:
        _index("control index", index, 3)
        return self._item(
            "control", f"control[{index}]", index, COUNTER_NBYTES, COUNTER_NBYTES
        )

    def rank_mask_word(self, rank: int) -> ByteSpan:
        _index("rank", rank, self.max_ranks)
        return self._item(
            "rank_mask",
            f"rank_mask[{rank}]",
            rank,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def rank_incarnation_word(self, rank: int) -> ByteSpan:
        _index("rank", rank, self.max_ranks)
        return self._item(
            "rank_incarnation",
            f"rank_incarnation[{rank}]",
            rank,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def dispatch_stage_record(self, bank: int, token: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("token", token, self.max_tokens_per_rank)
        item = bank * self.max_tokens_per_rank + token
        return self._item(
            "dispatch_stage",
            f"dispatch_stage[{bank}][{token}]",
            item,
            self.dispatch_record_stride,
            self.record_alignment,
        )

    def dispatch_receive_record(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        _index("slot", slot, self.max_tokens_per_rank)
        item = (
            (bank * self.experts_per_rank + local_expert) * self.max_ranks + source_rank
        ) * self.max_tokens_per_rank + slot
        return self._item(
            "dispatch_recv",
            (f"dispatch_recv[{bank}][{local_expert}]" f"[{source_rank}][{slot}]"),
            item,
            self.dispatch_record_stride,
            self.record_alignment,
        )

    def dispatch_ready_word(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        item = (
            bank * self.experts_per_rank + local_expert
        ) * self.max_ranks + source_rank
        return self._item(
            "dispatch_ready",
            f"dispatch_ready[{bank}][{local_expert}][{source_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def dispatch_credit_word(self, bank: int, peer_rank: int) -> ByteSpan:
        return self._credit_word("dispatch_credit", bank, peer_rank)

    def expert_input_record(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> ByteSpan:
        item = self._rank_expert_token_item(bank, local_expert, source_rank, slot)
        return self._item(
            "expert_input",
            f"expert_input[{bank}][{local_expert}][{source_rank}][{slot}]",
            item,
            self.payload_stride,
            self.record_alignment,
        )

    def dispatch_source_info(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> ByteSpan:
        item = self._rank_expert_token_item(bank, local_expert, source_rank, slot)
        return self._item(
            "dispatch_src_info",
            (f"dispatch_src_info[{bank}][{local_expert}]" f"[{source_rank}][{slot}]"),
            item,
            SOURCE_INFO_NBYTES,
            self.record_alignment,
        )

    def dispatch_layout_word(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        item = (
            bank * self.experts_per_rank + local_expert
        ) * self.max_ranks + source_rank
        return self._item(
            "dispatch_layout_range",
            f"dispatch_layout_range[{bank}][{local_expert}][{source_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def combine_stage_record(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> ByteSpan:
        item = self._rank_expert_token_item(bank, local_expert, source_rank, slot)
        return self._item(
            "combine_stage",
            f"combine_stage[{bank}][{local_expert}][{source_rank}][{slot}]",
            item,
            self.combine_record_stride,
            self.record_alignment,
        )

    def combine_receive_record(
        self, bank: int, global_expert: int, origin_token: int
    ) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("global_expert", global_expert, self.max_experts)
        _index("origin_token", origin_token, self.max_tokens_per_rank)
        item = (
            bank * self.max_experts + global_expert
        ) * self.max_tokens_per_rank + origin_token
        return self._item(
            "combine_recv",
            f"combine_recv[{bank}][{global_expert}][{origin_token}]",
            item,
            self.combine_record_stride,
            self.record_alignment,
        )

    def combine_ready_word(self, bank: int, global_expert: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("global_expert", global_expert, self.max_experts)
        item = bank * self.max_experts + global_expert
        return self._item(
            "combine_ready",
            f"combine_ready[{bank}][{global_expert}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def combine_credit_word(self, bank: int, peer_rank: int) -> ByteSpan:
        return self._credit_word("combine_credit", bank, peer_rank)

    def _rank_expert_token_item(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> int:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        _index("slot", slot, self.max_tokens_per_rank)
        return (
            (bank * self.experts_per_rank + local_expert) * self.max_ranks + source_rank
        ) * self.max_tokens_per_rank + slot

    def _credit_word(self, region_name: str, bank: int, peer_rank: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("peer_rank", peer_rank, self.max_ranks)
        item = bank * self.max_ranks + peer_rank
        return self._item(
            region_name,
            f"{region_name}[{bank}][{peer_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )


_PIPELINE_BANKED_REGIONS = frozenset(
    {
        "dispatch_stage",
        "dispatch_stage_src_info",
        "dispatch_recv",
        "dispatch_recv_src_info",
        "expert_input",
        "dispatch_src_info",
        "dispatch_stamp",
        "dispatch_ready_seq",
        "dispatch_credit",
        "dispatch_layout_range",
        "combine_stage",
        "combine_recv",
        "combine_stamp",
        "combine_ready_seq",
        "combine_credit",
        "bank_state",
    }
)


@dataclass(frozen=True, slots=True)
class PipelineLLArenaLayout:
    """Two-bank arena for a split dispatch/expert/combine pipeline.

    Unlike :class:`CompactLLArenaLayout`, this layout does not fuse expert
    compute into the communication kernel. The dispatched payload first lands
    in source-sharded receive buckets and is then compacted into
    the dense expert-major prefix handed to external grouped GEMM.  This is the
    same essential shape split used by mature NIXL EP: independent sources can
    write without a remote allocation handshake, while expert compute still
    receives ``[local_expert, packed_row, hidden]`` plus device counts. Keeping
    route metadata in a parallel 16-byte plane gives both representations a
    normal payload stride. The compaction is an explicit GPU data-path cost;
    it must be vectorized and measured, never hidden behind a zero-copy claim.

    The large payload regions are dimensioned only by facts the protocol
    actually needs.  A source contributes at most ``num_tokens`` rows to one
    destination expert, and an origin receives exactly one result for each
    ``(route_slot, token)``.  Every steady-state region is physically
    double-buffered.  A returned dispatch credit protects raw receive storage
    and dense expert input; a returned combine credit protects expert output
    and the origin's route-major receive plane.
    """

    max_ranks: int
    experts_per_rank: int
    num_tokens: int
    top_k: int
    hidden_size: int
    element_size: int
    record_alignment: int = 16
    region_alignment: int = 128
    arena_alignment: int = 4096
    address_limit: int = field(default=UINT64_MAX, repr=False)

    route_capacity: int = field(init=False)
    payload_nbytes: int = field(init=False)
    payload_stride: int = field(init=False)
    regions: tuple[ByteSpan, ...] = field(init=False)
    arena_nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, UINT32_MAX)
        _positive("experts_per_rank", self.experts_per_rank, UINT32_MAX)
        _positive("num_tokens", self.num_tokens, UINT32_MAX)
        _positive("top_k", self.top_k, UINT32_MAX)
        _positive("hidden_size", self.hidden_size, UINT32_MAX)
        _positive("element_size", self.element_size, UINT32_MAX)
        _power_of_two("record_alignment", self.record_alignment)
        _power_of_two("region_alignment", self.region_alignment)
        _power_of_two("arena_alignment", self.arena_alignment)
        _positive("address_limit", self.address_limit, UINT64_MAX)
        if self.record_alignment < 16:
            raise ValueError("record_alignment must satisfy NIXL's 16-byte PUT ABI")
        if self.record_alignment > self.region_alignment:
            raise ValueError("record_alignment must not exceed region_alignment")
        if self.region_alignment > self.arena_alignment:
            raise ValueError("region_alignment must not exceed arena_alignment")
        if self.max_ranks > (UINT32_MAX + 1) // self.experts_per_rank:
            raise OverflowError("fixed expert namespace does not fit in u32")
        max_experts = self.max_ranks * self.experts_per_rank
        if self.top_k > max_experts:
            raise ValueError(
                f"top_k {self.top_k} exceeds fixed expert capacity {max_experts}"
            )

        route_capacity = _checked_mul(
            self.num_tokens,
            self.top_k,
            self.address_limit,
            "route capacity",
        )
        if route_capacity > UINT32_MAX:
            raise OverflowError("num_tokens * top_k does not fit in u32")
        rank_token_capacity = _checked_mul(
            self.max_ranks,
            self.num_tokens,
            self.address_limit,
            "expert input row capacity",
        )
        if rank_token_capacity > UINT32_MAX:
            raise OverflowError("max_ranks * num_tokens does not fit in u32")
        payload_nbytes = _checked_mul(
            self.hidden_size,
            self.element_size,
            self.address_limit,
            "token payload",
        )
        if payload_nbytes % self.record_alignment:
            raise ValueError(
                "pipeline token payload bytes must be a multiple of "
                "record_alignment so expert_input and combine_stage can be "
                "exposed as contiguous [expert, row, hidden] tensors"
            )
        payload_stride = _align_up(
            payload_nbytes,
            self.record_alignment,
            self.address_limit,
            "token payload stride",
        )
        object.__setattr__(self, "route_capacity", route_capacity)
        object.__setattr__(self, "payload_nbytes", payload_nbytes)
        object.__setattr__(self, "payload_stride", payload_stride)

        rank_expert_slots = _checked_mul(
            self.max_ranks,
            self.experts_per_rank,
            self.address_limit,
            "rank/expert slots",
        )
        rank_expert_token_slots = _checked_mul(
            rank_expert_slots,
            self.num_tokens,
            self.address_limit,
            "rank/expert/token slots",
        )

        def size(name: str, *factors: int) -> tuple[str, int]:
            return (
                name,
                _checked_product(
                    *factors,
                    limit=self.address_limit,
                    expression=f"{name} size",
                ),
            )

        sizes = (
            # committed generation, next operation, admission state, fatal state
            size("control", 4, COUNTER_NBYTES),
            size("rank_mask", self.max_ranks, COUNTER_NBYTES),
            size("rank_incarnation", self.max_ranks, COUNTER_NBYTES),
            size(
                "dispatch_stage",
                NUM_BANKS,
                self.num_tokens,
                payload_stride,
            ),
            size(
                "dispatch_stage_src_info",
                NUM_BANKS,
                route_capacity,
                SOURCE_INFO_NBYTES,
            ),
            # Independent producers first land in fixed source-sharded
            # buckets.  A device compaction pass turns these raw records into
            # the dense expert-major planes below without a host allocation or
            # count readback.
            size(
                "dispatch_recv",
                NUM_BANKS,
                rank_expert_token_slots,
                payload_stride,
            ),
            size(
                "dispatch_recv_src_info",
                NUM_BANKS,
                rank_expert_token_slots,
                SOURCE_INFO_NBYTES,
            ),
            size(
                "expert_input",
                NUM_BANKS,
                rank_expert_token_slots,
                payload_stride,
            ),
            size(
                "dispatch_src_info",
                NUM_BANKS,
                rank_expert_token_slots,
                SOURCE_INFO_NBYTES,
            ),
            size(
                "dispatch_stamp",
                NUM_BANKS,
                rank_expert_slots,
                PIPELINE_BUCKET_STAMP_NBYTES,
            ),
            size(
                "dispatch_ready_seq",
                NUM_BANKS,
                rank_expert_slots,
                COUNTER_NBYTES,
            ),
            size(
                "dispatch_credit",
                NUM_BANKS,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
            size(
                "dispatch_layout_range",
                NUM_BANKS,
                rank_expert_slots,
                COUNTER_NBYTES,
            ),
            size(
                "combine_stage",
                NUM_BANKS,
                rank_expert_token_slots,
                payload_stride,
            ),
            size(
                "combine_recv",
                NUM_BANKS,
                route_capacity,
                payload_stride,
            ),
            size(
                "combine_stamp",
                NUM_BANKS,
                self.max_ranks,
                PIPELINE_BUCKET_STAMP_NBYTES,
            ),
            size(
                "combine_ready_seq",
                NUM_BANKS,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
            size(
                "combine_credit",
                NUM_BANKS,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
            size("bank_state", NUM_BANKS, COUNTER_NBYTES),
            size("abort_state", COUNTER_NBYTES),
        )

        cursor = 0
        regions: list[ByteSpan] = []
        for name, nbytes in sizes:
            _positive(f"{name} size", nbytes, self.address_limit)
            cursor = _align_up(
                cursor,
                self.region_alignment,
                self.address_limit,
                f"{name} start",
            )
            end = _checked_add(cursor, nbytes, self.address_limit, name)
            regions.append(ByteSpan(name, cursor, nbytes, self.region_alignment))
            cursor = end
        arena_nbytes = _align_up(
            cursor,
            self.arena_alignment,
            self.address_limit,
            "arena size",
        )
        for left, right in pairwise(regions):
            if left.overlaps(right):
                raise AssertionError(f"arena regions overlap: {left} and {right}")
        object.__setattr__(self, "regions", tuple(regions))
        object.__setattr__(self, "arena_nbytes", arena_nbytes)

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    def region(self, name: str) -> ByteSpan:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown arena region {name!r}")

    def bank_region(self, name: str, bank: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        if name not in _PIPELINE_BANKED_REGIONS:
            raise ValueError(f"region {name!r} is not banked")
        region = self.region(name)
        if region.nbytes % NUM_BANKS:
            raise AssertionError(f"banked region {name!r} is not evenly divisible")
        bank_nbytes = region.nbytes // NUM_BANKS
        span_offset = region.offset + bank * bank_nbytes
        span_alignment = self.region_alignment
        while span_offset % span_alignment:
            span_alignment //= 2
        return ByteSpan(
            f"{name}[{bank}]",
            span_offset,
            bank_nbytes,
            span_alignment,
        )

    def _item(
        self,
        region_name: str,
        item_name: str,
        item_index: int,
        item_nbytes: int,
        item_alignment: int,
    ) -> ByteSpan:
        region = self.region(region_name)
        relative = _checked_mul(
            item_index,
            item_nbytes,
            self.address_limit,
            f"{item_name} offset",
        )
        offset = _checked_add(
            region.offset,
            relative,
            self.address_limit,
            f"{item_name} address",
        )
        end = _checked_add(
            offset,
            item_nbytes,
            self.address_limit,
            f"{item_name} span",
        )
        if offset < region.offset or end > region.end:
            raise AssertionError(
                f"computed {item_name} [{offset}, {end}) outside {region_name} "
                f"[{region.offset}, {region.end})"
            )
        return ByteSpan(item_name, offset, item_nbytes, item_alignment)

    def control_word(self, index: int) -> ByteSpan:
        _index("control index", index, 4)
        return self._item(
            "control", f"control[{index}]", index, COUNTER_NBYTES, COUNTER_NBYTES
        )

    def rank_mask_word(self, rank: int) -> ByteSpan:
        _index("rank", rank, self.max_ranks)
        return self._item(
            "rank_mask", f"rank_mask[{rank}]", rank, COUNTER_NBYTES, COUNTER_NBYTES
        )

    def rank_incarnation_word(self, rank: int) -> ByteSpan:
        _index("rank", rank, self.max_ranks)
        return self._item(
            "rank_incarnation",
            f"rank_incarnation[{rank}]",
            rank,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def dispatch_stage_record(self, bank: int, token: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        _index("token", token, self.num_tokens)
        item = bank * self.num_tokens + token
        return self._item(
            "dispatch_stage",
            f"dispatch_stage[{bank}][{token}]",
            item,
            self.payload_stride,
            self.record_alignment,
        )

    def dispatch_stage_source_info(
        self, bank: int, token: int, route_slot: int
    ) -> ByteSpan:
        route = self._route_item(bank, token, route_slot)
        return self._item(
            "dispatch_stage_src_info",
            f"dispatch_stage_src_info[{bank}][{token}][{route_slot}]",
            route,
            SOURCE_INFO_NBYTES,
            SOURCE_INFO_NBYTES,
        )

    def dispatch_receive_record(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        source_slot: int,
    ) -> ByteSpan:
        """Return one raw source-sharded receive payload slot."""

        item = self._rank_expert_token_item(
            bank, local_expert, source_rank, source_slot
        )
        return self._item(
            "dispatch_recv",
            f"dispatch_recv[{bank}][{local_expert}]" f"[{source_rank}][{source_slot}]",
            item,
            self.payload_stride,
            self.record_alignment,
        )

    def dispatch_receive_source_info(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        source_slot: int,
    ) -> ByteSpan:
        """Return metadata beside one raw source-sharded receive slot."""

        item = self._rank_expert_token_item(
            bank, local_expert, source_rank, source_slot
        )
        return self._item(
            "dispatch_recv_src_info",
            f"dispatch_recv_src_info[{bank}][{local_expert}]"
            f"[{source_rank}][{source_slot}]",
            item,
            SOURCE_INFO_NBYTES,
            SOURCE_INFO_NBYTES,
        )

    def expert_input_record(
        self,
        bank: int,
        local_expert: int,
        packed_slot: int,
    ) -> ByteSpan:
        """Return one dense expert-major row produced by GPU compaction."""

        item = self._expert_packed_item(bank, local_expert, packed_slot)
        return self._item(
            "expert_input",
            f"expert_input[{bank}][{local_expert}][{packed_slot}]",
            item,
            self.payload_stride,
            self.record_alignment,
        )

    def dispatch_source_info(
        self,
        bank: int,
        local_expert: int,
        packed_slot: int,
    ) -> ByteSpan:
        """Return origin metadata beside one compacted expert row."""

        item = self._expert_packed_item(bank, local_expert, packed_slot)
        return self._item(
            "dispatch_src_info",
            f"dispatch_src_info[{bank}][{local_expert}][{packed_slot}]",
            item,
            SOURCE_INFO_NBYTES,
            SOURCE_INFO_NBYTES,
        )

    def dispatch_stamp(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        item = self._rank_expert_item(bank, local_expert, source_rank)
        return self._item(
            "dispatch_stamp",
            f"dispatch_stamp[{bank}][{local_expert}][{source_rank}]",
            item,
            PIPELINE_BUCKET_STAMP_NBYTES,
            16,
        )

    def dispatch_ready_sequence_word(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        item = self._rank_expert_item(bank, local_expert, source_rank)
        return self._item(
            "dispatch_ready_seq",
            f"dispatch_ready_seq[{bank}][{local_expert}][{source_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def dispatch_credit_word(self, bank: int, peer_rank: int) -> ByteSpan:
        return self._rank_word("dispatch_credit", bank, peer_rank)

    def dispatch_layout_word(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        item = self._rank_expert_item(bank, local_expert, source_rank)
        return self._item(
            "dispatch_layout_range",
            f"dispatch_layout_range[{bank}][{local_expert}][{source_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def combine_stage_record(
        self,
        bank: int,
        local_expert: int,
        packed_slot: int,
    ) -> ByteSpan:
        """Return the registered expert-output row for one compacted input."""

        item = self._expert_packed_item(bank, local_expert, packed_slot)
        return self._item(
            "combine_stage",
            f"combine_stage[{bank}][{local_expert}][{packed_slot}]",
            item,
            self.payload_stride,
            self.record_alignment,
        )

    def combine_receive_record(
        self, bank: int, route_slot: int, origin_token: int
    ) -> ByteSpan:
        route = self._combine_item(bank, route_slot, origin_token)
        return self._item(
            "combine_recv",
            f"combine_recv[{bank}][{route_slot}][{origin_token}]",
            route,
            self.payload_stride,
            self.record_alignment,
        )

    def combine_stamp(self, bank: int, peer_rank: int) -> ByteSpan:
        item = self._rank_item(bank, peer_rank)
        return self._item(
            "combine_stamp",
            f"combine_stamp[{bank}][{peer_rank}]",
            item,
            PIPELINE_BUCKET_STAMP_NBYTES,
            16,
        )

    def combine_ready_sequence_word(self, bank: int, peer_rank: int) -> ByteSpan:
        return self._rank_word("combine_ready_seq", bank, peer_rank)

    def combine_credit_word(self, bank: int, peer_rank: int) -> ByteSpan:
        return self._rank_word("combine_credit", bank, peer_rank)

    def bank_state_word(self, bank: int) -> ByteSpan:
        _index("bank", bank, NUM_BANKS)
        return self._item(
            "bank_state",
            f"bank_state[{bank}]",
            bank,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def abort_state_word(self) -> ByteSpan:
        return self._item(
            "abort_state",
            "abort_state[0]",
            0,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def _route_item(self, bank: int, token: int, route_slot: int) -> int:
        _index("bank", bank, NUM_BANKS)
        _index("token", token, self.num_tokens)
        _index("route_slot", route_slot, self.top_k)
        return (bank * self.num_tokens + token) * self.top_k + route_slot

    def _rank_item(self, bank: int, peer_rank: int) -> int:
        _index("bank", bank, NUM_BANKS)
        _index("peer_rank", peer_rank, self.max_ranks)
        return bank * self.max_ranks + peer_rank

    def _combine_item(self, bank: int, route_slot: int, origin_token: int) -> int:
        """Use the measured fused kernel's route-major receive ordering."""

        _index("bank", bank, NUM_BANKS)
        _index("route_slot", route_slot, self.top_k)
        _index("origin_token", origin_token, self.num_tokens)
        return (bank * self.top_k + route_slot) * self.num_tokens + origin_token

    def _rank_expert_item(self, bank: int, local_expert: int, rank: int) -> int:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("rank", rank, self.max_ranks)
        return (bank * self.experts_per_rank + local_expert) * self.max_ranks + rank

    def _rank_expert_token_item(
        self,
        bank: int,
        local_expert: int,
        rank: int,
        source_slot: int,
    ) -> int:
        bucket = self._rank_expert_item(bank, local_expert, rank)
        _index("source_slot", source_slot, self.num_tokens)
        return bucket * self.num_tokens + source_slot

    def _expert_packed_item(
        self, bank: int, local_expert: int, packed_slot: int
    ) -> int:
        _index("bank", bank, NUM_BANKS)
        _index("local_expert", local_expert, self.experts_per_rank)
        packed_capacity = self.max_ranks * self.num_tokens
        _index("packed_slot", packed_slot, packed_capacity)
        return (
            bank * self.experts_per_rank + local_expert
        ) * packed_capacity + packed_slot

    def _rank_word(self, region_name: str, bank: int, peer_rank: int) -> ByteSpan:
        item = self._rank_item(bank, peer_rank)
        return self._item(
            region_name,
            f"{region_name}[{bank}][{peer_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )


_COMPACT_BANKED_REGIONS = frozenset(
    {
        "dispatch_recv",
        "dispatch_stamp",
        "dispatch_ready",
        "combine_consumed",
        "combine_recv",
        "combine_stamp",
        "combine_ready",
    }
)


@dataclass(frozen=True, slots=True)
class CompactLLArenaLayout:
    """Minimal configurable-bank arena used by the production mapped MoE example.

    The dimensions are deliberately protocol-specific rather than conservative:

    * immutable dispatch stage: ``[stage_copy, token * top_k]`` records;
    * dispatch receive: ``[bank, local_expert, source_rank, token]`` bodies;
    * dispatch stamp: ``[bank, local_expert, source_rank]`` operation/incarnation;
    * dispatch ready: ``[bank, local_expert, source_rank]`` u64 words;
    * worker state: ``[bank, logical_worker_warp]`` u64 words (the historical
      ``combine_consumed`` region name is retained for layout compatibility);
    * abort state: one monotonic local u64 word shared by the cooperative grid;
    * combine receive: ``[bank, route_slot, token]`` payloads;
    * combine stamp: ``[bank, source_rank]`` operation/incarnation;
    * combine ready: ``[bank, source_rank]`` aggregate u64 words.

    A token cannot select the same expert twice, so one source contributes at
    most ``num_tokens`` records to one expert. The origin needs exactly one
    combine record for each ``(route_slot, token)`` pair. Keeping those facts in
    the address model prevents fixed expert/rank capacity from multiplying the
    large payload regions unnecessarily.
    """

    max_ranks: int
    experts_per_rank: int
    num_tokens: int
    top_k: int
    hidden_size: int
    element_size: int
    workers_per_peer: int = 1
    dispatch_stage_copies: int = 1
    num_banks: int = 1
    record_alignment: int = 16
    region_alignment: int = 128
    arena_alignment: int = 4096
    address_limit: int = field(default=UINT64_MAX, repr=False)

    route_capacity: int = field(init=False)
    payload_nbytes: int = field(init=False)
    payload_stride: int = field(init=False)
    dispatch_record_nbytes: int = field(init=False)
    dispatch_record_stride: int = field(init=False)
    combine_record_nbytes: int = field(init=False)
    combine_record_stride: int = field(init=False)
    regions: tuple[ByteSpan, ...] = field(init=False)
    arena_nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, UINT32_MAX)
        _positive("experts_per_rank", self.experts_per_rank, UINT32_MAX)
        _positive("num_tokens", self.num_tokens, UINT32_MAX)
        _positive("top_k", self.top_k, UINT32_MAX)
        _positive("hidden_size", self.hidden_size, UINT32_MAX)
        _positive("element_size", self.element_size, UINT32_MAX)
        _positive("workers_per_peer", self.workers_per_peer, UINT32_MAX)
        _positive("dispatch_stage_copies", self.dispatch_stage_copies, UINT32_MAX)
        _positive("num_banks", self.num_banks, NUM_BANKS)
        if self.num_banks not in (1, NUM_BANKS):
            raise ValueError("num_banks must be one or two")
        _power_of_two("record_alignment", self.record_alignment)
        _power_of_two("region_alignment", self.region_alignment)
        _power_of_two("arena_alignment", self.arena_alignment)
        _positive("address_limit", self.address_limit, UINT64_MAX)
        if self.record_alignment < 16:
            raise ValueError("record_alignment must satisfy NIXL's 16-byte PUT ABI")
        if self.record_alignment > self.region_alignment:
            raise ValueError("record_alignment must not exceed region_alignment")
        if self.region_alignment > self.arena_alignment:
            raise ValueError("region_alignment must not exceed arena_alignment")
        if self.max_ranks > (UINT32_MAX + 1) // self.experts_per_rank:
            raise OverflowError("fixed expert namespace does not fit in u32")
        max_experts = self.max_ranks * self.experts_per_rank
        if self.top_k > max_experts:
            raise ValueError(
                f"top_k {self.top_k} exceeds fixed expert capacity {max_experts}"
            )

        route_capacity = _checked_mul(
            self.num_tokens,
            self.top_k,
            self.address_limit,
            "route capacity",
        )
        if route_capacity > UINT32_MAX:
            raise OverflowError("num_tokens * top_k does not fit in u32")
        payload_nbytes = _checked_mul(
            self.hidden_size,
            self.element_size,
            self.address_limit,
            "token payload",
        )
        payload_stride = _align_up(
            payload_nbytes,
            self.record_alignment,
            self.address_limit,
            "token payload stride",
        )
        # The production mapped protocol validates operation/incarnation once
        # per released bucket. Repeating that stamp in every record would copy
        # and then overwrite 16 remote bytes per route. Records therefore carry
        # only immutable route metadata followed by the BF16 payload.
        dispatch_record_nbytes = _checked_add(
            DISPATCH_METADATA_NBYTES,
            payload_nbytes,
            self.address_limit,
            "dispatch record",
        )
        dispatch_record_stride = _align_up(
            dispatch_record_nbytes,
            self.record_alignment,
            self.address_limit,
            "dispatch record stride",
        )
        # Route slot and token already select a unique combine destination;
        # identity lives in one aggregate stamp per publishing peer.
        combine_record_nbytes = payload_nbytes
        combine_record_stride = _align_up(
            combine_record_nbytes,
            self.record_alignment,
            self.address_limit,
            "combine record stride",
        )

        object.__setattr__(self, "route_capacity", route_capacity)
        object.__setattr__(self, "payload_nbytes", payload_nbytes)
        object.__setattr__(self, "payload_stride", payload_stride)
        object.__setattr__(self, "dispatch_record_nbytes", dispatch_record_nbytes)
        object.__setattr__(self, "dispatch_record_stride", dispatch_record_stride)
        object.__setattr__(self, "combine_record_nbytes", combine_record_nbytes)
        object.__setattr__(self, "combine_record_stride", combine_record_stride)

        def size(name: str, *factors: int) -> tuple[str, int]:
            return (
                name,
                _checked_product(
                    *factors,
                    limit=self.address_limit,
                    expression=f"{name} size",
                ),
            )

        sizes = (
            size(
                "dispatch_stage",
                self.dispatch_stage_copies,
                route_capacity,
                dispatch_record_stride,
            ),
            size(
                "dispatch_recv",
                self.num_banks,
                self.experts_per_rank,
                self.max_ranks,
                self.num_tokens,
                dispatch_record_stride,
            ),
            size(
                "dispatch_stamp",
                self.num_banks,
                self.experts_per_rank,
                self.max_ranks,
                MESSAGE_STAMP_NBYTES,
            ),
            size(
                "dispatch_ready",
                self.num_banks,
                self.experts_per_rank,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
            size(
                "combine_consumed",
                self.num_banks,
                self.max_ranks,
                self.workers_per_peer,
                COUNTER_NBYTES,
            ),
            size("abort_state", COUNTER_NBYTES),
            size(
                "combine_recv",
                self.num_banks,
                self.top_k,
                self.num_tokens,
                combine_record_stride,
            ),
            size(
                "combine_stamp",
                self.num_banks,
                self.max_ranks,
                MESSAGE_STAMP_NBYTES,
            ),
            size(
                "combine_ready",
                self.num_banks,
                self.max_ranks,
                COUNTER_NBYTES,
            ),
        )

        cursor = 0
        regions: list[ByteSpan] = []
        for name, nbytes in sizes:
            _positive(f"{name} size", nbytes, self.address_limit)
            cursor = _align_up(
                cursor,
                self.region_alignment,
                self.address_limit,
                f"{name} start",
            )
            end = _checked_add(cursor, nbytes, self.address_limit, name)
            regions.append(ByteSpan(name, cursor, nbytes, self.region_alignment))
            cursor = end
        arena_nbytes = _align_up(
            cursor,
            self.arena_alignment,
            self.address_limit,
            "arena size",
        )

        for left, right in pairwise(regions):
            if left.overlaps(right):
                raise AssertionError(f"arena regions overlap: {left} and {right}")
        object.__setattr__(self, "regions", tuple(regions))
        object.__setattr__(self, "arena_nbytes", arena_nbytes)

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    def region(self, name: str) -> ByteSpan:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown arena region {name!r}")

    def bank_region(self, name: str, bank: int) -> ByteSpan:
        """Return one physical bank of a compact banked region."""

        _index("bank", bank, self.num_banks)
        if name not in _COMPACT_BANKED_REGIONS:
            raise ValueError(f"region {name!r} is not banked")
        region = self.region(name)
        if region.nbytes % self.num_banks:
            raise AssertionError(f"banked region {name!r} is not evenly divisible")
        bank_nbytes = region.nbytes // self.num_banks
        span_offset = region.offset + bank * bank_nbytes
        span_alignment = self.region_alignment
        while span_offset % span_alignment:
            span_alignment //= 2
        return ByteSpan(
            f"{name}[{bank}]",
            span_offset,
            bank_nbytes,
            span_alignment,
        )

    def _item(
        self,
        region_name: str,
        item_name: str,
        item_index: int,
        item_nbytes: int,
        item_alignment: int,
    ) -> ByteSpan:
        region = self.region(region_name)
        relative = _checked_mul(
            item_index,
            item_nbytes,
            self.address_limit,
            f"{item_name} offset",
        )
        offset = _checked_add(
            region.offset,
            relative,
            self.address_limit,
            f"{item_name} address",
        )
        end = _checked_add(
            offset,
            item_nbytes,
            self.address_limit,
            f"{item_name} span",
        )
        if offset < region.offset or end > region.end:
            raise AssertionError(
                f"computed {item_name} [{offset}, {end}) outside {region_name} "
                f"[{region.offset}, {region.end})"
            )
        return ByteSpan(item_name, offset, item_nbytes, item_alignment)

    def dispatch_stage_record(self, route: int, stage_copy: int = 0) -> ByteSpan:
        _index("route", route, self.route_capacity)
        _index("stage_copy", stage_copy, self.dispatch_stage_copies)
        item = stage_copy * self.route_capacity + route
        return self._item(
            "dispatch_stage",
            f"dispatch_stage[{stage_copy}][{route}]",
            item,
            self.dispatch_record_stride,
            self.record_alignment,
        )

    def dispatch_receive_record(
        self,
        bank: int,
        local_expert: int,
        source_rank: int,
        slot: int,
    ) -> ByteSpan:
        _index("bank", bank, self.num_banks)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        _index("slot", slot, self.num_tokens)
        item = (
            (bank * self.experts_per_rank + local_expert) * self.max_ranks + source_rank
        ) * self.num_tokens + slot
        return self._item(
            "dispatch_recv",
            f"dispatch_recv[{bank}][{local_expert}][{source_rank}][{slot}]",
            item,
            self.dispatch_record_stride,
            self.record_alignment,
        )

    def dispatch_ready_word(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        _index("bank", bank, self.num_banks)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        item = (
            bank * self.experts_per_rank + local_expert
        ) * self.max_ranks + source_rank
        return self._item(
            "dispatch_ready",
            f"dispatch_ready[{bank}][{local_expert}][{source_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def dispatch_stamp(
        self, bank: int, local_expert: int, source_rank: int
    ) -> ByteSpan:
        """Return one bucket-level dispatch operation/incarnation stamp."""

        _index("bank", bank, self.num_banks)
        _index("local_expert", local_expert, self.experts_per_rank)
        _index("source_rank", source_rank, self.max_ranks)
        item = (
            bank * self.experts_per_rank + local_expert
        ) * self.max_ranks + source_rank
        return self._item(
            "dispatch_stamp",
            f"dispatch_stamp[{bank}][{local_expert}][{source_rank}]",
            item,
            MESSAGE_STAMP_NBYTES,
            MESSAGE_STAMP_NBYTES,
        )

    def combine_consumed_word(
        self, bank: int, physical_rank_slot: int, worker: int = 0
    ) -> ByteSpan:
        """Return one monotonic local worker-state word.

        ``combine_consumed`` is the region's legacy name.  The production
        protocol uses four ordered values for dispatch-copy completion, inbound
        ready fanout, expert-shard completion, and expert completion. Dynamic
        membership maps active communication tasks onto the flat
        logical-worker-warp ordinal
        ``physical_rank_slot * workers_per_peer + worker``.
        """

        _index("bank", bank, self.num_banks)
        _index("physical_rank_slot", physical_rank_slot, self.max_ranks)
        _index("worker", worker, self.workers_per_peer)
        item = (
            bank * self.max_ranks + physical_rank_slot
        ) * self.workers_per_peer + worker
        return self._item(
            "combine_consumed",
            f"combine_consumed[{bank}][{physical_rank_slot}][{worker}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def abort_state_word(self) -> ByteSpan:
        """Return the cooperative grid's monotonic local abort word."""

        return self._item(
            "abort_state",
            "abort_state[0]",
            0,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def combine_receive_record(
        self, bank: int, route_slot: int, origin_token: int
    ) -> ByteSpan:
        _index("bank", bank, self.num_banks)
        _index("route_slot", route_slot, self.top_k)
        _index("origin_token", origin_token, self.num_tokens)
        item = (bank * self.top_k + route_slot) * self.num_tokens + origin_token
        return self._item(
            "combine_recv",
            f"combine_recv[{bank}][{route_slot}][{origin_token}]",
            item,
            self.combine_record_stride,
            self.record_alignment,
        )

    def combine_ready_word(self, bank: int, peer_rank: int) -> ByteSpan:
        _index("bank", bank, self.num_banks)
        _index("peer_rank", peer_rank, self.max_ranks)
        item = bank * self.max_ranks + peer_rank
        return self._item(
            "combine_ready",
            f"combine_ready[{bank}][{peer_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )

    def combine_stamp(self, bank: int, peer_rank: int) -> ByteSpan:
        """Return one aggregate combine operation/incarnation stamp per peer."""

        _index("bank", bank, self.num_banks)
        _index("peer_rank", peer_rank, self.max_ranks)
        item = bank * self.max_ranks + peer_rank
        return self._item(
            "combine_stamp",
            f"combine_stamp[{bank}][{peer_rank}]",
            item,
            MESSAGE_STAMP_NBYTES,
            MESSAGE_STAMP_NBYTES,
        )

    def _rank_word(self, region_name: str, bank: int, peer_rank: int) -> ByteSpan:
        _index("bank", bank, self.num_banks)
        _index("peer_rank", peer_rank, self.max_ranks)
        item = bank * self.max_ranks + peer_rank
        return self._item(
            region_name,
            f"{region_name}[{bank}][{peer_rank}]",
            item,
            COUNTER_NBYTES,
            COUNTER_NBYTES,
        )


class BankPhase(str, Enum):
    FREE = "free"
    PUBLISHING = "publishing"
    WAITING_CREDITS = "waiting_credits"


@dataclass(frozen=True, slots=True)
class BankSnapshot:
    bank: int
    phase: BankPhase
    operation: OperationEpoch | None
    expected_peers: frozenset[int]
    published_peers: frozenset[int]
    credited_peers: frozenset[int]


@dataclass(slots=True)
class _MutableBank:
    phase: BankPhase = BankPhase.FREE
    operation: OperationEpoch | None = None
    expected_peers: set[int] = field(default_factory=set)
    published_peers: set[int] = field(default_factory=set)
    credited_peers: set[int] = field(default_factory=set)
    publishing_finished: bool = False


class TwoBankCreditLifecycle:
    """Sender-side bank ownership and returned-credit state machine.

    A fast receiver may return a credit while the sender is still publishing to
    other peers.  Such an early credit is accepted only after that particular
    peer was marked published.  The bank becomes free when publication is
    closed and every expected peer has credited consumption.
    """

    def __init__(self, direction: Direction, max_ranks: int) -> None:
        if not isinstance(direction, Direction):
            raise TypeError("direction must be a Direction")
        _positive("max_ranks", max_ranks, UINT32_MAX)
        self.direction = direction
        self.max_ranks = max_ranks
        self._banks = [_MutableBank() for _ in range(NUM_BANKS)]
        self._last_started: OperationEpoch | None = None
        self._last_completed: list[OperationEpoch | None] = [None] * NUM_BANKS

    @property
    def quiescent(self) -> bool:
        return all(bank.phase == BankPhase.FREE for bank in self._banks)

    def assert_quiescent(self) -> None:
        if not self.quiescent:
            busy = [
                index
                for index, bank in enumerate(self._banks)
                if bank.phase != BankPhase.FREE
            ]
            raise ProtocolError(
                f"{self.direction.value} banks {busy} still have outstanding credits"
            )

    def begin(self, operation: OperationEpoch, expected_peers: Iterable[int]) -> int:
        if not isinstance(operation, OperationEpoch):
            raise TypeError("operation must be an OperationEpoch")
        peers = tuple(expected_peers)
        for peer in peers:
            _index("expected peer", peer, self.max_ranks)
        if len(set(peers)) != len(peers):
            raise ValueError("expected_peers contains duplicates")
        if self._last_started is not None and operation <= self._last_started:
            raise ProtocolError(
                f"operation {operation.wire_value} does not advance beyond "
                f"{self._last_started.wire_value}"
            )
        if (
            self._last_started is not None
            and operation.membership_generation
            != self._last_started.membership_generation
            and not self.quiescent
        ):
            raise ProtocolError(
                "membership generation may change only at a quiescent bank boundary"
            )

        bank_id = operation.bank
        bank = self._banks[bank_id]
        if bank.phase != BankPhase.FREE:
            active = bank.operation.wire_value if bank.operation else None
            raise ProtocolError(
                f"{self.direction.value} bank {bank_id} is still owned by "
                f"operation {active}"
            )

        bank.phase = BankPhase.PUBLISHING
        bank.operation = operation
        bank.expected_peers = set(peers)
        bank.published_peers.clear()
        bank.credited_peers.clear()
        bank.publishing_finished = False
        self._last_started = operation
        return bank_id

    def mark_published(self, operation: OperationEpoch, peer_rank: int) -> None:
        """Mark one peer after every channel, including empty ones, published."""

        bank = self._owned_bank(operation)
        if bank.phase != BankPhase.PUBLISHING:
            raise ProtocolError("cannot publish after publication has been closed")
        _index("peer_rank", peer_rank, self.max_ranks)
        if peer_rank not in bank.expected_peers:
            raise ProtocolError(f"peer {peer_rank} was not expected for this operation")
        if peer_rank in bank.published_peers:
            raise ProtocolError(f"peer {peer_rank} was published more than once")
        bank.published_peers.add(peer_rank)

    def finish_publishing(self, operation: OperationEpoch) -> None:
        bank = self._owned_bank(operation)
        if bank.phase != BankPhase.PUBLISHING:
            raise ProtocolError("publication is already closed")
        missing = bank.expected_peers - bank.published_peers
        if missing:
            raise ProtocolError(
                f"cannot close publication; missing peers {sorted(missing)}"
            )
        bank.publishing_finished = True
        bank.phase = BankPhase.WAITING_CREDITS
        self._retire_if_complete(operation.bank)

    def return_credit(self, operation: OperationEpoch, peer_rank: int) -> None:
        bank = self._owned_bank(operation)
        _index("peer_rank", peer_rank, self.max_ranks)
        if peer_rank not in bank.published_peers:
            raise ProtocolError(
                f"credit from peer {peer_rank} arrived before its publication"
            )
        if peer_rank in bank.credited_peers:
            raise ProtocolError(f"duplicate credit from peer {peer_rank}")
        bank.credited_peers.add(peer_rank)
        self._retire_if_complete(operation.bank)

    def snapshot(self, bank: int) -> BankSnapshot:
        _index("bank", bank, NUM_BANKS)
        state = self._banks[bank]
        return BankSnapshot(
            bank,
            state.phase,
            state.operation,
            frozenset(state.expected_peers),
            frozenset(state.published_peers),
            frozenset(state.credited_peers),
        )

    def last_completed(self, bank: int) -> OperationEpoch | None:
        _index("bank", bank, NUM_BANKS)
        return self._last_completed[bank]

    def _owned_bank(self, operation: OperationEpoch) -> _MutableBank:
        if not isinstance(operation, OperationEpoch):
            raise TypeError("operation must be an OperationEpoch")
        bank = self._banks[operation.bank]
        if bank.phase == BankPhase.FREE or bank.operation != operation:
            active = bank.operation.wire_value if bank.operation else None
            raise ProtocolError(
                f"operation {operation.wire_value} does not own bank "
                f"{operation.bank}; active operation is {active}"
            )
        return bank

    def _retire_if_complete(self, bank_id: int) -> None:
        bank = self._banks[bank_id]
        if not bank.publishing_finished:
            return
        if bank.credited_peers != bank.expected_peers:
            return
        if bank.operation is None:
            raise AssertionError("busy bank has no operation")
        self._last_completed[bank_id] = bank.operation
        bank.phase = BankPhase.FREE
        bank.operation = None
        bank.expected_peers.clear()
        bank.published_peers.clear()
        bank.credited_peers.clear()
        bank.publishing_finished = False


class ReceivePhase(str, Enum):
    WAITING_PUBLICATIONS = "waiting_publications"
    READY_TO_CONSUME = "ready_to_consume"
    CONSUMED = "consumed"
    CREDIT_ISSUED = "credit_issued"


@dataclass(frozen=True, slots=True)
class ReceiveSnapshot:
    phase: ReceivePhase
    observed_counts: Mapping[int, int]
    reset_channels: frozenset[int]


class ReceiveCreditLifecycle:
    """Receiver-side gate enforcing acquire, consume, reset, then ACK."""

    def __init__(
        self,
        direction: Direction,
        operation: OperationEpoch,
        source_rank: int,
        channels: Iterable[int],
    ) -> None:
        if not isinstance(direction, Direction):
            raise TypeError("direction must be a Direction")
        if not isinstance(operation, OperationEpoch):
            raise TypeError("operation must be an OperationEpoch")
        _uint("source_rank", source_rank, UINT32_MAX)
        channel_tuple = tuple(channels)
        if not channel_tuple:
            raise ValueError("at least one channel must be expected")
        for channel in channel_tuple:
            _uint("channel", channel, UINT32_MAX)
        if len(set(channel_tuple)) != len(channel_tuple):
            raise ValueError("channels contains duplicates")

        self.direction = direction
        self.operation = operation
        self.source_rank = source_rank
        self.channels = frozenset(channel_tuple)
        self._counts: dict[int, int] = {}
        self._reset_channels: set[int] = set()
        self._phase = ReceivePhase.WAITING_PUBLICATIONS

    @property
    def all_publications_observed(self) -> bool:
        return self._counts.keys() == self.channels

    @property
    def total_records(self) -> int:
        if not self.all_publications_observed:
            raise ProtocolError("not every channel publication has arrived")
        return sum(self._counts.values())

    def observe_publication(self, channel: int, value: int) -> int:
        if self._phase not in (
            ReceivePhase.WAITING_PUBLICATIONS,
            ReceivePhase.READY_TO_CONSUME,
        ):
            raise ProtocolError("cannot observe publication after consumption")
        _uint("channel", channel, UINT32_MAX)
        _positive("publication value", value, UINT32_MAX + 1)
        if channel not in self.channels:
            raise ProtocolError(f"unexpected channel {channel}")
        if channel in self._counts:
            raise ProtocolError(f"duplicate publication on channel {channel}")
        count = value - 1
        self._counts[channel] = count
        if self.all_publications_observed:
            self._phase = ReceivePhase.READY_TO_CONSUME
        return count

    def mark_consumed(self) -> None:
        if self._phase != ReceivePhase.READY_TO_CONSUME:
            raise ProtocolError("all channel publications must arrive before consume")
        self._phase = ReceivePhase.CONSUMED

    def mark_ready_reset(self, channel: int) -> None:
        if self._phase != ReceivePhase.CONSUMED:
            raise ProtocolError("ready words may be reset only after consumption")
        _uint("channel", channel, UINT32_MAX)
        if channel not in self.channels:
            raise ProtocolError(f"unexpected channel {channel}")
        if channel in self._reset_channels:
            raise ProtocolError(f"channel {channel} ready word reset twice")
        self._reset_channels.add(channel)

    def issue_credit(self) -> None:
        if self._phase != ReceivePhase.CONSUMED:
            raise ProtocolError("credit may be issued only after consumption")
        missing = self.channels - self._reset_channels
        if missing:
            raise ProtocolError(
                f"credit cannot precede ready-word resets for channels "
                f"{sorted(missing)}"
            )
        self._phase = ReceivePhase.CREDIT_ISSUED

    def snapshot(self) -> ReceiveSnapshot:
        return ReceiveSnapshot(
            self._phase,
            dict(self._counts),
            frozenset(self._reset_channels),
        )


__all__ = [
    "BankPhase",
    "BankSnapshot",
    "BucketKey",
    "BucketSchedule",
    "ByteSpan",
    "CompactLLArenaLayout",
    "COUNTER_NBYTES",
    "DEFAULT_GDA_CHANNELS",
    "DISPATCH_HEADER_NBYTES",
    "DOORBELL_INTERVAL",
    "DataPost",
    "Direction",
    "DispatchHeader",
    "LLArenaLayout",
    "MAX_GDA_CHANNELS",
    "MESSAGE_STAMP_NBYTES",
    "MessageStamp",
    "NUM_BANKS",
    "OperationEpoch",
    "PIPELINE_BUCKET_STAMP_NBYTES",
    "PipelineBucketStamp",
    "PipelineLLArenaLayout",
    "PipelineSourceInfo",
    "PostFlag",
    "ProtocolError",
    "PublicationPost",
    "ReceiveCreditLifecycle",
    "ReceivePhase",
    "ReceiveSnapshot",
    "SOURCE_INFO_NBYTES",
    "STANDIN_EXPERT_BIAS_SCALE",
    "StableSparseTopology",
    "TwoBankCreditLifecycle",
    "UINT32_MAX",
    "UINT64_MAX",
    "data_post_flag",
    "gda_bucket_channel",
    "make_bucket_schedule",
    "normalize_gda_channel_count",
    "pack_layout_range",
    "unpack_layout_range",
]
