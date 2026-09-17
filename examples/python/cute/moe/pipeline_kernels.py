# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe device kernels for the split-phase, all-mapped MoE pipeline.

The public host state machine lives in :mod:`pipeline`; this module is the
lower-level device building block.  Its three steady-state launches are:

``launch_mapped_dispatch``
    Route *live* BF16 activations into fixed-source receive segments, publish
    one release sequence per source/expert bucket, and compact the acquired
    segments into dense expert-major tensors entirely on the device.

``launch_standin_expert``
    Apply a deterministic BF16 expert to every valid dense row.  It exists to
    exercise the split boundary; production applications replace this launch
    with their grouped GEMM on the same stream.

``launch_mapped_combine``
    Scatter real expert output back to route-slot-major receive storage, wait
    for peer publications, reduce with FP32 accumulation, and return credits.

The fast path is intentionally narrow.  Every active arena must have a non-null
process-local pointer produced by ``launch_resolve_mapped_peers`` at a drained
membership boundary, and native peer atomics must have been qualified for every
directed GPU pair.  Hot kernels use those cached pointers and only the coherent
``mapped_copy_warp_ptr`` operation: live activations and expert output do not
satisfy the stronger whole-kernel immutability precondition of the read-only
copy.  There is no transparent network fallback here.  An RDMA specialization
needs ordered NIXL PUT/publication/credit operations and independent retained-
CUBIN qualification; silently mixing that protocol into this kernel would make
its bank-reuse proof false.

Every operation uses exactly two physical banks.  Dispatch credit means the
destination has finished reading a prior expert-input bank.  Combine credit
means the origin has finished reducing a prior combine-receive bank.  Both are
peer-originated device publications, so no tensor scalar is copied to the host
and no host synchronization or progress loop appears in the steady state.
Generation changes remain a drained control-plane operation.  Abrupt mapped-
owner loss is fail-stop because a direct peer load/store may fault and the
current NIXL device ABI cannot cancel or flush outstanding requestless work.

The module remains importable without CUDA, CuTe, Torch, or NIXL.  The checked
:class:`KernelContract` is therefore usable by host tests and by backends that
want to validate tensor geometry before importing accelerator libraries.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional


def _load_package_less_protocol_constant() -> int:
    """Load the shared constant for an isolated source-file import.

    Normal package and direct-example imports use the relative import below.
    A small number of source-contract tools deliberately execute this file via
    ``spec_from_file_location`` without a package context, so retain that
    dependency-free mode without adding the example root to ``sys.path``.
    """

    module_name = "_nixl_cute_moe_ll_protocol_contract"
    protocol_path = Path(__file__).with_name("ll_protocol.py").resolve()
    protocol = sys.modules.get(module_name)
    if protocol is None:
        spec = importlib.util.spec_from_file_location(module_name, protocol_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load shared MoE protocol from {protocol_path}")
        protocol = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = protocol
        try:
            spec.loader.exec_module(protocol)
        except BaseException:
            if sys.modules.get(module_name) is protocol:
                del sys.modules[module_name]
            raise
    elif Path(getattr(protocol, "__file__", "")).resolve() != protocol_path:
        raise ImportError(f"conflicting isolated protocol module {module_name}")
    value = getattr(protocol, "STANDIN_EXPERT_BIAS_SCALE", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ImportError(
            "shared stand-in expert bias scale must be a positive integer"
        )
    return value


if __package__:
    from .ll_protocol import STANDIN_EXPERT_BIAS_SCALE
else:
    try:  # Package-less direct example-root import.
        from moe.ll_protocol import STANDIN_EXPERT_BIAS_SCALE  # type: ignore[no-redef]
    except ModuleNotFoundError as error:
        if error.name != "moe":
            raise
        STANDIN_EXPERT_BIAS_SCALE = _load_package_less_protocol_constant()

NUM_BANKS = 2
WARP_SIZE = 32
BF16_NBYTES = 2
BF16_PER_VECTOR = 8
VECTOR_NBYTES = BF16_NBYTES * BF16_PER_VECTOR
COMBINE_VECTOR_UNROLL = 2
SOURCE_INFO_WORDS = 4
SOURCE_INFO_NBYTES = SOURCE_INFO_WORDS * 4
BUCKET_STAMP_WORDS = 4
BUCKET_STAMP_NBYTES = BUCKET_STAMP_WORDS * 8
COUNTER_NBYTES = 8
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
INT32_MAX = (1 << 31) - 1


def _positive(name: str, value: int, maximum: int = UINT32_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 < value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _index(name: str, value: int, bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < bound:
        raise IndexError(f"{name} {value} is outside [0, {bound})")
    return value


@dataclass(frozen=True, slots=True)
class KernelContract:
    """Checked logical geometry shared by the mapped launchers.

    Payload rows have no hidden padding in this specialization: requiring
    ``hidden_size`` to be a multiple of eight makes each BF16 row a multiple of
    the mapped-copy ABI's 16-byte vector.  ``source_info`` has three logical
    fields and one zero padding word.  ``layout_range`` is one u64 encoded as
    ``(begin << 32) | count``.
    """

    max_ranks: int
    experts_per_rank: int
    token_capacity: int
    top_k: int
    hidden_size: int

    def __post_init__(self) -> None:
        _positive("max_ranks", self.max_ranks, INT32_MAX)
        _positive("experts_per_rank", self.experts_per_rank, INT32_MAX)
        _positive("token_capacity", self.token_capacity, INT32_MAX)
        _positive("top_k", self.top_k, INT32_MAX)
        _positive("hidden_size", self.hidden_size, INT32_MAX)
        if self.max_ranks > WARP_SIZE:
            raise ValueError(f"max_ranks must not exceed warp size {WARP_SIZE}")
        if self.top_k > WARP_SIZE:
            raise ValueError(f"top_k must not exceed warp size {WARP_SIZE}")
        if self.max_ranks > INT32_MAX // self.experts_per_rank:
            raise OverflowError("fixed expert namespace does not fit in int32")
        if self.top_k > self.max_experts:
            raise ValueError("top_k exceeds the fixed expert namespace")
        if self.hidden_size % BF16_PER_VECTOR:
            raise ValueError("hidden_size must be divisible by eight BF16 values")
        if self.token_capacity > INT32_MAX // self.top_k:
            raise OverflowError("token_capacity * top_k does not fit in int32")
        if self.token_capacity > INT32_MAX // self.max_ranks:
            raise OverflowError("max_ranks * token_capacity does not fit in int32")
        dense_rows = self.max_ranks * self.token_capacity
        if dense_rows > INT32_MAX // self.experts_per_rank:
            raise OverflowError("stand-in expert CUDA grid x does not fit in int32")

    @property
    def max_experts(self) -> int:
        return self.max_ranks * self.experts_per_rank

    @property
    def route_capacity(self) -> int:
        return self.token_capacity * self.top_k

    @property
    def payload_nbytes(self) -> int:
        return self.hidden_size * BF16_NBYTES

    @property
    def dispatch_ctas(self) -> int:
        """Minimum worker grid covering every destination/expert bucket."""

        return self.max_ranks * self.experts_per_rank

    def validate_cooperative_grid(
        self,
        *,
        dispatch_resident_limit: int,
        worker_ctas: int,
        combine_resident_limit: int,
    ) -> int:
        """Validate occupancy results before either cooperative launch.

        The limits are hardware- and retained-CUBIN-specific totals obtained by
        the backend's occupancy query.  Selection happens at compile/admission
        time, never in a captured or steady-state launch.
        """

        _positive("dispatch_resident_limit", dispatch_resident_limit, INT32_MAX)
        _positive("worker_ctas", worker_ctas, INT32_MAX)
        _positive("combine_resident_limit", combine_resident_limit, INT32_MAX)
        if worker_ctas < self.dispatch_ctas:
            raise ValueError(
                "worker_ctas must cover every rank/expert communication bucket"
            )
        if worker_ctas > dispatch_resident_limit:
            raise ValueError(
                "worker_ctas exceeds the qualified dispatch resident limit"
            )
        if worker_ctas > combine_resident_limit:
            raise ValueError("worker_ctas exceeds the qualified combine resident limit")
        return worker_ctas

    @staticmethod
    def validate_nonoverlapping_spans(
        spans: Mapping[str, tuple[int, int]],
    ) -> None:
        """Reject hazardous operand aliases before compiling a launcher call.

        Values are checked half-open device-address intervals. Empty logical
        tensors may use ``begin == end`` and do not overlap any interval.  The
        sole allowed alias is an exact ``expert_output``/``combine_stage`` span,
        which is the registered zero-copy expert-epilogue path.  Callers pass
        individual arena destination subregions, not the owning arena span.
        """

        if not isinstance(spans, Mapping):
            raise TypeError("spans must be a mapping")
        checked: list[tuple[str, int, int]] = []
        for name, span in spans.items():
            if not isinstance(name, str) or not name:
                raise ValueError("span names must be non-empty strings")
            if not isinstance(span, tuple) or len(span) != 2:
                raise TypeError(f"span {name!r} must be a (begin, end) tuple")
            begin, end = span
            for field, value in (("begin", begin), ("end", end)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"span {name!r} {field} must be an integer")
                if not 0 <= value <= UINT64_MAX:
                    raise ValueError(f"span {name!r} {field} must fit in u64")
            if end < begin:
                raise ValueError(f"span {name!r} end precedes begin")
            if begin != end and begin % VECTOR_NBYTES:
                raise ValueError(
                    f"span {name!r} base must be {VECTOR_NBYTES}-byte aligned"
                )
            checked.append((name, begin, end))
        for index, (left_name, left_begin, left_end) in enumerate(checked):
            if left_begin == left_end:
                continue
            for right_name, right_begin, right_end in checked[index + 1 :]:
                if right_begin == right_end:
                    continue
                if left_begin < right_end and right_begin < left_end:
                    names = frozenset((left_name, right_name))
                    if names == frozenset(("expert_output", "combine_stage")):
                        if left_begin == right_begin and left_end == right_end:
                            continue
                    raise ValueError(
                        f"device spans {left_name!r} and {right_name!r} overlap"
                    )

    @property
    def tensor_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Maximum-capacity shapes for arena allocation and ahead-of-time JIT."""

        return self.tensor_shapes_for(self.token_capacity)

    def tensor_shapes_for(self, live_tokens: int) -> Mapping[str, tuple[int, ...]]:
        """Shapes for one live batch backed by this fixed-capacity arena."""

        if isinstance(live_tokens, bool) or not isinstance(live_tokens, int):
            raise TypeError("live_tokens must be an integer")
        if not 0 <= live_tokens <= self.token_capacity:
            raise ValueError(f"live_tokens must be in [0, {self.token_capacity}]")

        dense_rows = self.max_ranks * self.token_capacity
        return MappingProxyType(
            {
                "activations": (live_tokens, self.hidden_size),
                "topk_indices": (live_tokens, self.top_k),
                "topk_weights": (live_tokens, self.top_k),
                "dispatch_recv": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    self.max_ranks,
                    self.token_capacity,
                    self.hidden_size,
                ),
                "dispatch_recv_src_info": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    self.max_ranks,
                    self.token_capacity,
                    SOURCE_INFO_WORDS,
                ),
                "expert_input": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    dense_rows,
                    self.hidden_size,
                ),
                # Explicit current-bank grouped-GEMM input, outside the arena.
                "expert_counts": (self.experts_per_rank,),
                # Explicit current-bank route allocators, outside the arena.
                "route_counts": (self.max_experts,),
                "source_info": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    dense_rows,
                    SOURCE_INFO_WORDS,
                ),
                "layout_range": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    self.max_ranks,
                ),
                "expert_output": (
                    self.experts_per_rank,
                    dense_rows,
                    self.hidden_size,
                ),
                "combine_stage": (
                    NUM_BANKS,
                    self.experts_per_rank,
                    dense_rows,
                    self.hidden_size,
                ),
                "combine_recv": (
                    NUM_BANKS,
                    self.top_k,
                    self.token_capacity,
                    self.hidden_size,
                ),
                "combine_output": (live_tokens, self.hidden_size),
            }
        )

    def owner(self, global_expert: int) -> int:
        _index("global_expert", global_expert, self.max_experts)
        return global_expert // self.experts_per_rank

    def local_expert(self, global_expert: int) -> int:
        _index("global_expert", global_expert, self.max_experts)
        return global_expert % self.experts_per_rank

    def raw_receive_slot(self, source_rank: int, source_slot: int) -> int:
        """Return the fixed source-sharded slot before GPU compaction."""

        _index("source_rank", source_rank, self.max_ranks)
        _index("source_slot", source_slot, self.token_capacity)
        return source_rank * self.token_capacity + source_slot

    def compaction_shards(
        self, bucket: int, count: int, worker_ctas: int
    ) -> tuple[tuple[int, int], ...]:
        """Model the contiguous worker shards owned by one communication bucket.

        CTA ``bucket + shard * (R*E)`` owns each returned half-open range.  This
        import-safe model is also the executable specification for the device
        kernels' dispatch-compaction and combine-scatter partitions.
        """

        _index("bucket", bucket, self.dispatch_ctas)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if not 0 <= count <= self.token_capacity:
            raise ValueError(f"count must be in [0, {self.token_capacity}]")
        _positive("worker_ctas", worker_ctas, INT32_MAX)
        if worker_ctas < self.dispatch_ctas:
            raise ValueError(
                "worker_ctas must cover every rank/expert communication bucket"
            )
        shard_count = ((worker_ctas - 1 - bucket) // self.dispatch_ctas) + 1
        return tuple(
            (
                (count * shard) // shard_count,
                (count * (shard + 1)) // shard_count,
            )
            for shard in range(shard_count)
        )

    def combine_slot(self, route_slot: int, origin_token: int) -> int:
        _index("route_slot", route_slot, self.top_k)
        _index("origin_token", origin_token, self.token_capacity)
        return route_slot * self.token_capacity + origin_token

    def bank(self, operation: int) -> int:
        self._operation(operation)
        return operation & 1

    def wire_epoch(self, generation: int, operation: int) -> int:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise TypeError("generation must be an integer")
        if not 0 <= generation <= UINT32_MAX:
            raise ValueError("generation must fit in u32")
        self._operation(operation)
        return (generation << 32) | operation

    def ready_sequence(self, generation: int, operation: int) -> int:
        # Generation is validated in the adjacent stamp.  The ready/credit
        # word mirrors requestless atomic-add publication: one monotonic count
        # per physical bank, rebased to zero only after a generation drain.
        self.wire_epoch(generation, operation)
        return operation // NUM_BANKS + 1

    def previous_bank_credit(self, generation: int, operation: int) -> int | None:
        self._operation(operation)
        if operation < NUM_BANKS:
            return None
        return self.ready_sequence(generation, operation - NUM_BANKS)

    def pack_layout_range(self, begin: int, count: int) -> int:
        if isinstance(begin, bool) or not isinstance(begin, int):
            raise TypeError("begin must be an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if not 0 <= begin <= UINT32_MAX:
            raise ValueError("begin must fit in u32")
        if not 0 <= count <= self.token_capacity:
            raise ValueError("count exceeds the per-source expert capacity")
        if begin > self.max_ranks * self.token_capacity:
            raise ValueError("begin exceeds the dense expert row capacity")
        if begin + count > self.max_ranks * self.token_capacity:
            raise ValueError("layout range exceeds the dense expert row capacity")
        return (begin << 32) | count

    def unpack_layout_range(self, value: int) -> tuple[int, int]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("layout range must be an integer")
        if not 0 <= value <= UINT64_MAX:
            raise ValueError("layout range must fit in u64")
        begin, count = value >> 32, value & UINT32_MAX
        if count > self.token_capacity:
            raise ValueError("encoded count exceeds the per-source expert capacity")
        if begin > self.max_ranks * self.token_capacity:
            raise ValueError("encoded begin exceeds the dense expert row capacity")
        if begin + count > self.max_ranks * self.token_capacity:
            raise ValueError("encoded range exceeds the dense expert row capacity")
        return begin, count

    @staticmethod
    def _operation(operation: int) -> int:
        if isinstance(operation, bool) or not isinstance(operation, int):
            raise TypeError("operation must be an integer")
        if not 0 <= operation <= UINT32_MAX:
            raise ValueError("operation must fit in u32")
        return operation


def kernel_contract() -> Mapping[str, object]:
    """Return import-safe implementation and qualification metadata."""

    return MappingProxyType(
        {
            "implementation": "all-mapped-split-phase",
            "physical_banks": NUM_BANKS,
            "payload_dtype": "bfloat16",
            "weight_accumulator_dtype": "float32",
            "expert_counts_dtype": "int32",
            "route_counts_dtype": "uint64 current-bank [max_ranks*experts_per_rank]",
            "source_info_words": (
                "origin_rank",
                "origin_token",
                "route_slot",
                "reserved_zero",
            ),
            "layout_range_encoding": "(begin << 32) | count",
            "routing_precondition": (
                "each route is -1 (dropped) or an in-range expert; nonnegative "
                "experts with active owners are distinct per token, while routes "
                "to inactive owners contribute zero; topk_indices remains immutable "
                "through combine completion"
            ),
            "specialization_limits": (
                "max_ranks and top_k are at most one warp; token_capacity fixes "
                "arena strides while live_tokens independently specializes each "
                "rank's local batch"
            ),
            "zero_live_batch": (
                "live_tokens=0 still launches and publishes zero buckets; runtimes "
                "that cannot wrap empty DLPack tensors pass a retained aligned "
                "one-row dummy pointer which the kernels never dereference"
            ),
            "launch_precondition": (
                "backend rejects unless the same exact worker_ctas grid fits both "
                "retained dispatch and combine CUBIN cooperative residency limits; "
                "worker_ctas >= R*E and is specialization-cache-keyed; payload_stride "
                "equals the 16-byte-aligned BF16 row size; hazardous operand and "
                "arena-subregion spans have 16-byte-aligned bases and do not overlap, "
                "except exact expert_output == current-bank combine_stage"
            ),
            "mapped_copy": "coherent mapped_copy_warp_ptr",
            "combine_vector_unroll": COMBINE_VECTOR_UNROLL,
            "dispatch_compaction": (
                "O(N*K) route-parallel atomic packing grid-strided across worker_ctas "
                "into fixed-source dispatch_recv, then device compaction into a "
                "dense expert-major prefix using deterministic contiguous worker-CTA "
                "shards and constexpr divisors; no host count extraction"
            ),
            "combine_scatter": (
                "expert-output rows are scattered through deterministic contiguous "
                "worker-CTA shards with constexpr divisors before one grid-wide "
                "publication boundary"
            ),
            "standin_expert": (
                "correctness-only BF16 output = input + "
                "64 * (global_expert + 1); excluded from production perf claims"
            ),
            "standin_expert_bias_scale": STANDIN_EXPERT_BIAS_SCALE,
            "staging_policy": (
                "mapped dispatch bypasses source dispatch_stage; combine may "
                "read a same-stream expert_output directly or accept an expert "
                "epilogue that writes the registered combine_stage"
            ),
            "hot_path_host_sync": False,
            "hot_path_d2h_scalar": False,
            "hot_path_host_progress": False,
            "runtime_backend_adapter": False,
            "cuda_graph_capture": (
                "unsupported: replay would reuse by-value operation steps and can "
                "let stale ready or credit words satisfy waits; a future graph "
                "backend must update generation, operation, and incarnation "
                "kernel-node parameters before every replay and qualify that protocol"
            ),
            "membership_change": (
                "stage and pointer-resolve a private candidate while the old "
                "generation runs, then drain, rebase, globally commit, and swap"
            ),
            "collective_order": (
                "every active rank submits identical generation/operation order, "
                "including ranks whose local live_tokens is zero"
            ),
            "counter_rebase": (
                "ready and credit words reset only after the prior generation "
                "has cumulatively drained"
            ),
            "abrupt_peer_loss": "fail-stop; no in-place survivor recovery",
            "device_error_policy": (
                "dispatch exposes zero expert counts after error; combine publishes "
                "failed stamps and credits, then an unconditional PTX trap makes its CUDA "
                "completion event fail instead of exposing stale output. Abandoning "
                "an errored dispatch handle before combine is non-drainable misuse"
            ),
            "network_fallback": False,
            "performance_qualified": False,
        }
    )


try:
    _CUTE_AVAILABLE = importlib.util.find_spec("cutlass.cute") is not None
except ModuleNotFoundError:
    _CUTE_AVAILABLE = False


def is_cute_available() -> bool:
    """Whether this interpreter can import the CuTe implementation."""

    return _CUTE_AVAILABLE


if _CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op

    import nixl.device.cute as nixl_cute

    @dsl_user_op
    def _device_trap(
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Emit an unconditional PTX trap independent of assertion flags."""

        llvm.inline_asm(
            None,
            [],
            "trap;",
            "",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _wire_epoch(generation, operation):
        return cutlass.Uint64(generation) * cutlass.Uint64(1 << 32) + cutlass.Uint64(
            operation
        )

    @cute.jit
    def _ready_sequence(operation):
        return cutlass.Uint64(operation // NUM_BANKS) + cutlass.Uint64(1)

    @cute.jit
    def _warp_sum_u64(value):
        """Return a full-warp u64 sum without shared memory or an atomic."""

        value += cute.arch.shuffle_sync_bfly(value, 16)
        value += cute.arch.shuffle_sync_bfly(value, 8)
        value += cute.arch.shuffle_sync_bfly(value, 4)
        value += cute.arch.shuffle_sync_bfly(value, 2)
        value += cute.arch.shuffle_sync_bfly(value, 1)
        return value

    @cute.jit
    def _record_status(status_address, peer, value):
        if value != int(nixl_cute.NIXL_SUCCESS):
            nixl_cute.compare_exchange_status_gpu_i32(
                status_address + cutlass.Uint64(peer) * cutlass.Uint64(4), value
            )

    @cute.kernel
    def _resolve_mapped_peers_kernel(
        remote: nixl_cute.MemoryView,
        arena: cute.Tensor,
        rank_mask: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
    ):
        """Resolve process-local peer pointers once per drained generation."""

        tidx, _, _ = cute.arch.thread_idx()
        if tidx < max_ranks:
            statuses[tidx] = cutlass.Int32(0)
            peer_bases[tidx] = cutlass.Uint64(0)
            if rank_mask[tidx] == 0:
                if tidx == rank:
                    peer_bases[tidx] = cutlass.Uint64(arena.iterator.toint())
                else:
                    address = cutlass.Uint64(
                        cute.make_ptr(
                            cutlass.Uint8,
                            nixl_cute.get_ptr(remote, tidx),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).toint()
                    )
                    peer_bases[tidx] = address
                    if address == 0:
                        statuses[tidx] = cutlass.Int32(
                            int(nixl_cute.NIXL_ERR_NOT_SUPPORTED)
                        )

    @cute.jit
    def launch_resolve_mapped_peers(
        remote: nixl_cute.MemoryView,
        arena: cute.Tensor,
        rank_mask: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
    ):
        """Launch phase-boundary pointer resolution; caller validates status."""

        _resolve_mapped_peers_kernel(
            remote, arena, rank_mask, peer_bases, statuses, rank, max_ranks
        ).launch(
            grid=[1, 1, 1],
            block=[max_ranks, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _mapped_dispatch_kernel(
        arena: cute.Tensor,
        activations: cute.Tensor,
        topk_indices: cute.Tensor,
        expert_counts: cute.Tensor,
        route_counts: cute.Tensor,
        rank_mask: cute.Tensor,
        rank_incarnations: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        generation: cutlass.Uint32,
        operation: cutlass.Uint32,
        source_incarnation: cutlass.Uint64,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        live_tokens: cutlass.Constexpr[int],
        worker_ctas: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        dispatch_recv_base: cutlass.Constexpr[int],
        dispatch_recv_src_info_base: cutlass.Constexpr[int],
        expert_input_base: cutlass.Constexpr[int],
        dispatch_src_info_base: cutlass.Constexpr[int],
        dispatch_stamp_base: cutlass.Constexpr[int],
        dispatch_ready_base: cutlass.Constexpr[int],
        dispatch_credit_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Receive fixed-source buckets, then compact dense expert prefixes."""

        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        arena_address = cutlass.Uint64(arena.iterator.toint())
        activation_address = cutlass.Uint64(activations.iterator.toint())
        route_counts_address = cutlass.Uint64(route_counts.iterator.toint())
        status_address = cutlass.Uint64(statuses.iterator.toint())
        bank = operation % NUM_BANKS
        epoch = _wire_epoch(generation, operation)
        sequence = _ready_sequence(operation)

        rank_active = cutlass.Int32(0)
        if lane == 0:
            if rank_mask[rank] == 0:
                rank_active = cutlass.Int32(1)
        rank_active = cute.arch.shuffle_sync(rank_active, 0)

        # A source may reuse bank N only after each destination has finished
        # reading operation N-2's expert input.  One waiter per peer plus this
        # conditional grid join avoids E redundant system-scope polls.
        if operation >= NUM_BANKS:
            if cta < max_ranks:
                peer = cta
                if rank_active != 0:
                    if rank_mask[peer] == 0:
                        if peer != rank:
                            previous = _ready_sequence(operation - NUM_BANKS)
                            credit_offset = cutlass.Uint64(dispatch_credit_base) + (
                                cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(peer)
                            ) * cutlass.Uint64(COUNTER_NBYTES)
                            observed = nixl_cute.wait_acquire_system_u64(
                                arena_address + cutlass.Uint64(credit_offset),
                                previous,
                                scope=nixl_cute.Scope.WARP,
                            )
                            if observed != previous:
                                if lane == 0:
                                    _record_status(
                                        status_address,
                                        peer,
                                        int(nixl_cute.NIXL_ERR_MISMATCH),
                                    )
            nixl_cute.sync_grid()

        # Clear this bank's route allocators and validate every route before
        # issuing any mapped copy.  -1 and inactive owners are normal drops.
        # Malformed IDs or duplicate active experts fail every outgoing bucket
        # with count=0, allowing all receivers to finish without stale input.
        if cta < max_ranks * experts_per_rank:
            if lane == 0:
                route_counts[cta] = cutlass.Uint64(0)
        route_token = cutlass.Uint32(cta)
        while route_token < live_tokens:
            if rank_active != 0:
                if lane < top_k:
                    # DLPack preserves the logical 2-D tensor layout.  A lone
                    # scalar is therefore a coordinate, not a raw row-major
                    # offset: CuTe's idx2crd would otherwise remap this value
                    # across both modes (and introduce dynamic div/rem).  Keep
                    # token and route slot explicit at every external access.
                    selected = cutlass.Int32(topk_indices[(route_token, lane)])
                    if selected >= 0:
                        if selected < max_ranks * experts_per_rank:
                            owner = selected // experts_per_rank
                            if rank_mask[owner] == 0:
                                for prior_slot in cutlass.range_constexpr(top_k):
                                    if prior_slot < lane:
                                        prior = cutlass.Int32(
                                            topk_indices[(route_token, prior_slot)]
                                        )
                                        if prior == selected:
                                            _record_status(
                                                status_address,
                                                rank,
                                                int(nixl_cute.NIXL_ERR_MISMATCH),
                                            )
                        else:
                            _record_status(
                                status_address,
                                rank,
                                int(nixl_cute.NIXL_ERR_MISMATCH),
                            )
                    else:
                        if selected != -1:
                            _record_status(
                                status_address,
                                rank,
                                int(nixl_cute.NIXL_ERR_MISMATCH),
                            )
            route_token += cutlass.Uint32(worker_ctas)
        nixl_cute.sync_grid()
        routes_ok = cutlass.Int32(0)
        if lane == 0:
            if rank_active != 0:
                if statuses[rank] == 0:
                    routes_ok = cutlass.Int32(1)
        routes_ok = cute.arch.shuffle_sync(routes_ok, 0)

        # Route-parallel packing is O(N*K), independent of R*E.  Each logical
        # route is owned by one warp; lane zero atomically allocates its slot in
        # the current-bank destination bucket, then the converged warp performs
        # one coherent payload copy.  The per-token uniqueness check above
        # proves that no source/expert counter can exceed token_capacity.
        flat_route = cutlass.Uint32(cta)
        while flat_route < live_tokens * top_k:
            if routes_ok != 0:
                token = flat_route // top_k
                route_slot = flat_route % top_k
                selected = cutlass.Int32(-1)
                if lane == 0:
                    selected = cutlass.Int32(topk_indices[(token, route_slot)])
                selected = cute.arch.shuffle_sync(selected, 0)
                if selected >= 0:
                    owner = selected // experts_per_rank
                    owner_active = cutlass.Int32(0)
                    peer_address = cutlass.Uint64(0)
                    if lane == 0:
                        if rank_mask[owner] == 0:
                            if statuses[owner] == 0:
                                owner_active = cutlass.Int32(1)
                        peer_address = cutlass.Uint64(peer_bases[owner])
                    owner_active = cute.arch.shuffle_sync(owner_active, 0)
                    peer_address = cute.arch.shuffle_sync(peer_address, 0)
                    if owner_active != 0:
                        local_expert = selected % experts_per_rank
                        bucket = owner * experts_per_rank + local_expert
                        source_slot = cutlass.Uint64(0)
                        if lane == 0:
                            counter_offset = cutlass.Uint64(bucket) * cutlass.Uint64(
                                COUNTER_NBYTES
                            )
                            source_slot = nixl_cute.atomic_add_release_gpu_u64(
                                route_counts_address + counter_offset,
                                cutlass.Uint64(1),
                            )
                        source_slot = cute.arch.shuffle_sync(source_slot, 0)
                        if source_slot < token_capacity:
                            destination_item = (
                                (
                                    cutlass.Uint64(bank)
                                    * cutlass.Uint64(experts_per_rank)
                                    + cutlass.Uint64(local_expert)
                                )
                                * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(rank)
                            ) * cutlass.Uint64(token_capacity) + source_slot
                            destination_offset = cutlass.Uint64(
                                dispatch_recv_base
                            ) + destination_item * cutlass.Uint64(payload_stride)
                            source_offset = (
                                cutlass.Uint64(token)
                                * cutlass.Uint64(hidden_size)
                                * cutlass.Uint64(BF16_NBYTES)
                            )
                            copy_status = nixl_cute.mapped_copy_warp_ptr(
                                activation_address + source_offset,
                                peer_address + destination_offset,
                                cutlass.Uint64(hidden_size * BF16_NBYTES),
                            )
                            if copy_status != int(nixl_cute.NIXL_SUCCESS):
                                if lane == 0:
                                    _record_status(status_address, owner, copy_status)
                            if lane == 0:
                                source_info_offset = cutlass.Uint64(
                                    dispatch_recv_src_info_base
                                ) + destination_item * cutlass.Uint64(
                                    SOURCE_INFO_NBYTES
                                )
                                source_info = cute.make_tensor(
                                    cute.make_ptr(
                                        cutlass.Uint32,
                                        peer_address + source_info_offset,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(SOURCE_INFO_WORDS),
                                )
                                source_info[0] = cutlass.Uint32(rank)
                                source_info[1] = cutlass.Uint32(token)
                                source_info[2] = cutlass.Uint32(route_slot)
                                source_info[3] = cutlass.Uint32(0)
                        else:
                            if lane == 0:
                                _record_status(
                                    status_address,
                                    owner,
                                    int(nixl_cute.NIXL_ERR_MISMATCH),
                                )
            flat_route += cutlass.Uint32(worker_ctas)

        # All atomics and mapped copies complete before one warp per bucket
        # reads the final count and release-publishes its exact wire record.
        nixl_cute.sync_grid()
        if cta < max_ranks * experts_per_rank:
            peer = cta // experts_per_rank
            local_expert = cta % experts_per_rank
            peer_active = cutlass.Int32(0)
            peer_address = cutlass.Uint64(0)
            if lane == 0:
                if rank_mask[peer] == 0:
                    peer_active = cutlass.Int32(1)
                peer_address = cutlass.Uint64(peer_bases[peer])
            peer_active = cute.arch.shuffle_sync(peer_active, 0)
            peer_address = cute.arch.shuffle_sync(peer_address, 0)

            if rank_active != 0:
                if peer_active != 0:
                    if lane == 0:
                        counter_offset = cutlass.Uint64(cta) * cutlass.Uint64(
                            COUNTER_NBYTES
                        )
                        record_count = nixl_cute.load_acquire_gpu_u64(
                            route_counts_address + counter_offset
                        )
                        bucket_ok = routes_ok
                        if record_count > token_capacity:
                            bucket_ok = cutlass.Int32(0)
                            _record_status(
                                status_address,
                                peer,
                                int(nixl_cute.NIXL_ERR_MISMATCH),
                            )
                        if statuses[peer] != 0:
                            bucket_ok = cutlass.Int32(0)
                        if bucket_ok == 0:
                            record_count = cutlass.Uint64(0)
                        bucket_item = (
                            cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                            + cutlass.Uint64(local_expert)
                        ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(rank)
                        layout_offset = cutlass.Uint64(
                            dispatch_layout_base
                        ) + bucket_item * cutlass.Uint64(COUNTER_NBYTES)
                        begin = cutlass.Uint64(rank) * cutlass.Uint64(token_capacity)
                        layout_word = (begin << 32) + record_count
                        nixl_cute.store_release_system_u64(
                            peer_address + cutlass.Uint64(layout_offset), layout_word
                        )
                        stamp_offset = cutlass.Uint64(
                            dispatch_stamp_base
                        ) + bucket_item * cutlass.Uint64(BUCKET_STAMP_NBYTES)
                        stamp = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint64,
                                peer_address + cutlass.Uint64(stamp_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(BUCKET_STAMP_WORDS),
                        )
                        stamp[0] = epoch
                        stamp[1] = source_incarnation
                        failed = cutlass.Uint64(0)
                        if bucket_ok == 0:
                            failed = cutlass.Uint64(1)
                        stamp[2] = record_count
                        stamp[3] = failed
                        ready_offset = cutlass.Uint64(
                            dispatch_ready_base
                        ) + bucket_item * cutlass.Uint64(COUNTER_NBYTES)
                        nixl_cute.store_release_system_u64(
                            peer_address + cutlass.Uint64(ready_offset), sequence
                        )

            # Acquire the reciprocal source bucket.  The publication covers
            # raw payload, raw source metadata, source count, and stamp.
            incoming_count = cutlass.Uint64(0)
            if rank_active != 0:
                if peer_active != 0:
                    incoming_item = (
                        cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                        + cutlass.Uint64(local_expert)
                    ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(peer)
                    ready_offset = cutlass.Uint64(
                        dispatch_ready_base
                    ) + incoming_item * cutlass.Uint64(COUNTER_NBYTES)
                    observed = nixl_cute.wait_acquire_system_u64(
                        arena_address + cutlass.Uint64(ready_offset),
                        sequence,
                        scope=nixl_cute.Scope.WARP,
                    )
                    incoming_ok = cutlass.Int32(0)
                    if observed == sequence:
                        incoming_ok = cutlass.Int32(1)
                    if lane == 0:
                        stamp_offset = cutlass.Uint64(
                            dispatch_stamp_base
                        ) + incoming_item * cutlass.Uint64(BUCKET_STAMP_NBYTES)
                        stamp = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint64,
                                arena_address + cutlass.Uint64(stamp_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(BUCKET_STAMP_WORDS),
                        )
                        # CuTe tensor loads use a signless MLIR integer even
                        # when the pointer element is Uint64.  A same-width
                        # constructor is a no-op, so use a representation-
                        # preserving bitcast to give this branch-carried value
                        # an explicit unsigned DSL type.  The nested reset below
                        # must have exactly the same type at its control-flow
                        # merge.
                        incoming_count = cutlass.Int64(stamp[2]).bitcast(
                            cutlass.Uint64
                        )
                        if stamp[0] != epoch:
                            incoming_ok = cutlass.Int32(0)
                        if stamp[1] != rank_incarnations[peer]:
                            incoming_ok = cutlass.Int32(0)
                        # Peers may submit different live batch sizes.  The
                        # fixed source segment, not this rank's local batch,
                        # bounds a remote source's count.
                        if incoming_count > cutlass.Uint64(token_capacity):
                            incoming_ok = cutlass.Int32(0)
                        if stamp[3] != 0:
                            incoming_ok = cutlass.Int32(0)
                        if incoming_ok == 0:
                            incoming_count = cutlass.Uint64(0)
                            _record_status(
                                status_address,
                                peer,
                                int(nixl_cute.NIXL_ERR_MISMATCH),
                            )
                        # The acquired stamp is the sole trusted count.  Rewrite
                        # the adjacent range even on success so a torn, stale,
                        # or corrupted publisher-side layout word can never
                        # drive prefix overflow or an out-of-bounds compaction.
                        layout_offset = cutlass.Uint64(
                            dispatch_layout_base
                        ) + incoming_item * cutlass.Uint64(COUNTER_NBYTES)
                        sanitized_layout = (
                            cutlass.Uint64(peer) * cutlass.Uint64(token_capacity)
                        ) << 32
                        sanitized_layout += incoming_count
                        nixl_cute.store_release_gpu_u64(
                            arena_address + cutlass.Uint64(layout_offset),
                            sanitized_layout,
                        )
                    incoming_count = cute.arch.shuffle_sync(incoming_count, 0)
                else:
                    if lane == 0:
                        incoming_item = (
                            cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                            + cutlass.Uint64(local_expert)
                        ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(peer)
                        layout_offset = cutlass.Uint64(
                            dispatch_layout_base
                        ) + incoming_item * cutlass.Uint64(COUNTER_NBYTES)
                        nixl_cute.store_release_gpu_u64(
                            arena_address + cutlass.Uint64(layout_offset),
                            (cutlass.Uint64(peer) * cutlass.Uint64(token_capacity))
                            << 32,
                        )

        # Build a dense per-expert prefix only after every source/expert receiver
        # has completed its acquire.  Counts and ranges stay on device.
        nixl_cute.sync_grid()
        if cta < experts_per_rank:
            if lane == 0:
                total = cutlass.Uint64(0)
                for peer in cutlass.range(max_ranks, unroll=1):
                    bucket_item = (
                        cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                        + cutlass.Uint64(cta)
                    ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(peer)
                    layout_offset = cutlass.Uint64(
                        dispatch_layout_base
                    ) + bucket_item * cutlass.Uint64(COUNTER_NBYTES)
                    count = cutlass.Uint64(0)
                    if rank_active != 0:
                        if rank_mask[peer] == 0:
                            layout_value = nixl_cute.load_acquire_gpu_u64(
                                arena_address + cutlass.Uint64(layout_offset)
                            )
                            count = layout_value & cutlass.Uint64(UINT32_MAX)
                    packed_range = (total << 32) + count
                    nixl_cute.store_release_gpu_u64(
                        arena_address + cutlass.Uint64(layout_offset), packed_range
                    )
                    total += count
                expert_counts[cta] = cutlass.Int32(total)

        # Compact raw fixed-source segments into the dense prefix consumed by
        # grouped GEMM.  Every worker warp owns one deterministic contiguous
        # shard of a source/expert bucket and performs at most two coherent
        # copies (payload and metadata).  Integer-proportional boundaries cover
        # every row exactly once even when worker_ctas is not divisible by R*E.
        nixl_cute.sync_grid()
        bucket_count = max_ranks * experts_per_rank
        bucket = cta % bucket_count
        shard = cta // bucket_count
        shard_count = ((worker_ctas - 1 - bucket) // bucket_count) + 1
        source_rank = bucket // experts_per_rank
        local_expert = bucket % experts_per_rank
        bucket_item = (
            cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
            + cutlass.Uint64(local_expert)
        ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(source_rank)
        layout_value = cutlass.Uint64(0)
        if lane == 0:
            layout_value = nixl_cute.load_acquire_gpu_u64(
                arena_address
                + cutlass.Uint64(
                    cutlass.Uint64(dispatch_layout_base)
                    + bucket_item * cutlass.Uint64(COUNTER_NBYTES)
                )
            )
        layout_value = cute.arch.shuffle_sync(layout_value, 0)
        begin = layout_value >> 32
        count = layout_value & cutlass.Uint64(UINT32_MAX)
        # B and W are compile-time constants.  For W=q*B+r, each bucket has
        # either q or q+1 shards.  Keep the exact ownership formula above, but
        # select a constexpr divisor so CUDA never needs an out-of-line dynamic
        # u64 division helper in this launch-bound cooperative kernel.
        base_shard_count = worker_ctas // bucket_count
        shard_begin = cutlass.Uint64(0)
        shard_end = cutlass.Uint64(0)
        if cutlass.const_expr(worker_ctas % bucket_count == 0):
            shard_begin = (count * cutlass.Uint64(shard)) // cutlass.Uint64(
                base_shard_count
            )
            shard_end = (count * cutlass.Uint64(shard + 1)) // cutlass.Uint64(
                base_shard_count
            )
        else:
            if shard_count > base_shard_count:
                shard_begin = (count * cutlass.Uint64(shard)) // cutlass.Uint64(
                    base_shard_count + 1
                )
                shard_end = (count * cutlass.Uint64(shard + 1)) // cutlass.Uint64(
                    base_shard_count + 1
                )
            else:
                shard_begin = (count * cutlass.Uint64(shard)) // cutlass.Uint64(
                    base_shard_count
                )
                shard_end = (count * cutlass.Uint64(shard + 1)) // cutlass.Uint64(
                    base_shard_count
                )
        shard_rows = shard_end - shard_begin
        if shard_rows > 0:
            raw_item = (
                (
                    cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                    + cutlass.Uint64(local_expert)
                )
                * cutlass.Uint64(max_ranks)
                + cutlass.Uint64(source_rank)
            ) * cutlass.Uint64(token_capacity) + shard_begin
            dense_item = (
                (
                    cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                    + cutlass.Uint64(local_expert)
                )
                * (cutlass.Uint64(max_ranks) * cutlass.Uint64(token_capacity))
                + begin
                + shard_begin
            )
            payload_status = nixl_cute.mapped_copy_warp_ptr(
                arena_address
                + cutlass.Uint64(dispatch_recv_base)
                + raw_item * cutlass.Uint64(payload_stride),
                arena_address
                + cutlass.Uint64(expert_input_base)
                + dense_item * cutlass.Uint64(payload_stride),
                shard_rows * cutlass.Uint64(payload_stride),
            )
            info_status = nixl_cute.mapped_copy_warp_ptr(
                arena_address
                + cutlass.Uint64(
                    cutlass.Uint64(dispatch_recv_src_info_base)
                    + raw_item * cutlass.Uint64(SOURCE_INFO_NBYTES)
                ),
                arena_address
                + cutlass.Uint64(
                    cutlass.Uint64(dispatch_src_info_base)
                    + dense_item * cutlass.Uint64(SOURCE_INFO_NBYTES)
                ),
                shard_rows * cutlass.Uint64(SOURCE_INFO_NBYTES),
            )
            if lane == 0:
                _record_status(status_address, source_rank, payload_status)
                _record_status(status_address, source_rank, info_status)

        # A grouped GEMM may start as soon as the dispatch event completes.
        # Fail closed by exposing zero rows after any local/peer protocol error;
        # the required terminal CUDA failure is surfaced by the matching
        # combine, after its failed stamps have released every peer waiter.
        nixl_cute.sync_grid()
        if cta < experts_per_rank:
            if lane == 0:
                counts_ok = cutlass.Int32(1)
                for peer in cutlass.range(max_ranks, unroll=1):
                    if rank_mask[peer] == 0:
                        if statuses[peer] != 0:
                            counts_ok = cutlass.Int32(0)
                if counts_ok == 0:
                    expert_counts[cta] = cutlass.Int32(0)

    @cute.jit
    def launch_mapped_dispatch(
        arena: cute.Tensor,
        activations: cute.Tensor,
        topk_indices: cute.Tensor,
        expert_counts: cute.Tensor,
        route_counts: cute.Tensor,
        rank_mask: cute.Tensor,
        rank_incarnations: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        generation: cutlass.Uint32,
        operation: cutlass.Uint32,
        source_incarnation: cutlass.Uint64,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        live_tokens: cutlass.Constexpr[int],
        worker_ctas: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        dispatch_recv_base: cutlass.Constexpr[int],
        dispatch_recv_src_info_base: cutlass.Constexpr[int],
        expert_input_base: cutlass.Constexpr[int],
        dispatch_src_info_base: cutlass.Constexpr[int],
        dispatch_stamp_base: cutlass.Constexpr[int],
        dispatch_ready_base: cutlass.Constexpr[int],
        dispatch_credit_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Enqueue dispatch on ``stream`` without synchronizing the host."""

        _mapped_dispatch_kernel(
            arena,
            activations,
            topk_indices,
            expert_counts,
            route_counts,
            rank_mask,
            rank_incarnations,
            peer_bases,
            statuses,
            generation,
            operation,
            source_incarnation,
            rank,
            max_ranks,
            experts_per_rank,
            token_capacity,
            live_tokens,
            worker_ctas,
            top_k,
            hidden_size,
            dispatch_recv_base,
            dispatch_recv_src_info_base,
            expert_input_base,
            dispatch_src_info_base,
            dispatch_stamp_base,
            dispatch_ready_base,
            dispatch_credit_base,
            dispatch_layout_base,
            payload_stride,
        ).launch(
            grid=[worker_ctas, 1, 1],
            block=[WARP_SIZE, 1, 1],
            stream=stream,
            cooperative=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def _standin_expert_kernel(
        arena: cute.Tensor,
        expert_output: cute.Tensor,
        operation: cutlass.Uint32,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        expert_input_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Correctness stand-in: output = input + 64 * (global_expert + 1)."""

        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        bank = operation % NUM_BANKS
        rows_per_expert = max_ranks * token_capacity
        local_expert = cta // rows_per_expert
        row = cta % rows_per_expert
        source_rank = row // token_capacity
        source_slot = row % token_capacity
        arena_address = cutlass.Uint64(arena.iterator.toint())
        output_address = cutlass.Uint64(expert_output.iterator.toint())

        layout_item = (
            cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
            + cutlass.Uint64(local_expert)
        ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(source_rank)
        layout_value = cutlass.Uint64(0)
        if lane == 0:
            layout_value = nixl_cute.load_acquire_gpu_u64(
                arena_address
                + cutlass.Uint64(dispatch_layout_base)
                + layout_item * cutlass.Uint64(COUNTER_NBYTES)
            )
        layout_value = cute.arch.shuffle_sync(layout_value, 0)
        begin = layout_value >> 32
        count = layout_value & cutlass.Uint64(UINT32_MAX)
        if cutlass.Uint64(source_slot) < count:
            dense_slot = begin + cutlass.Uint64(source_slot)
            row_item = (
                cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                + cutlass.Uint64(local_expert)
            ) * (
                cutlass.Uint64(max_ranks) * cutlass.Uint64(token_capacity)
            ) + dense_slot
            input_address = (
                arena_address
                + cutlass.Uint64(expert_input_base)
                + row_item * cutlass.Uint64(payload_stride)
            )
            output_item = (
                cutlass.Uint64(local_expert)
                * (cutlass.Uint64(max_ranks) * cutlass.Uint64(token_capacity))
                + dense_slot
            )
            output_row = output_address + (
                output_item * cutlass.Uint64(hidden_size) * cutlass.Uint64(BF16_NBYTES)
            )
            vector_layout = cute.make_layout((4,))
            load = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                cutlass.Uint32,
                num_bits_per_copy=128,
                memory_order=cute.nvgpu.MemoryOrder.WEAK,
                memory_scope=cute.nvgpu.MemoryScope.CTA,
                l2_prefetch_size=cute.nvgpu.L2PrefetchSize.SIZE_256B,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            store = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                cutlass.Uint32,
                num_bits_per_copy=128,
                memory_order=cute.nvgpu.MemoryOrder.WEAK,
                memory_scope=cute.nvgpu.MemoryScope.CTA,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            fragment = cute.make_rmem_tensor(vector_layout, cutlass.Uint32)
            vector = cutlass.Uint32(lane)
            vectors = hidden_size // BF16_PER_VECTOR
            bias = cutlass.BFloat16(
                STANDIN_EXPERT_BIAS_SCALE * (rank * experts_per_rank + local_expert + 1)
            )
            while vector < vectors:
                byte_offset = cutlass.Uint64(vector) * VECTOR_NBYTES
                source = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        input_address + byte_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    vector_layout,
                )
                destination = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        output_row + byte_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    vector_layout,
                )
                cute.copy_atom_call(load, source, fragment)
                fragment_bf16 = cute.recast_tensor(fragment, cutlass.BFloat16)
                fragment_bf16.store(fragment_bf16.load() + bias)
                cute.copy_atom_call(store, fragment, destination)
                vector += cutlass.Uint32(WARP_SIZE)

    @cute.jit
    def launch_standin_expert(
        arena: cute.Tensor,
        expert_output: cute.Tensor,
        stream: cuda.CUstream,
        operation: cutlass.Uint32,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        expert_input_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Enqueue the example expert; replace with grouped GEMM in production."""

        _standin_expert_kernel(
            arena,
            expert_output,
            operation,
            rank,
            max_ranks,
            experts_per_rank,
            token_capacity,
            hidden_size,
            expert_input_base,
            dispatch_layout_base,
            payload_stride,
        ).launch(
            grid=[experts_per_rank * max_ranks * token_capacity, 1, 1],
            block=[WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _mapped_combine_kernel(
        arena: cute.Tensor,
        expert_output: cute.Tensor,
        topk_indices: cute.Tensor,
        topk_weights: cute.Tensor,
        combine_output: cute.Tensor,
        route_counts: cute.Tensor,
        rank_mask: cute.Tensor,
        rank_incarnations: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        generation: cutlass.Uint32,
        operation: cutlass.Uint32,
        source_incarnation: cutlass.Uint64,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        live_tokens: cutlass.Constexpr[int],
        worker_ctas: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        dispatch_src_info_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        dispatch_credit_base: cutlass.Constexpr[int],
        combine_recv_base: cutlass.Constexpr[int],
        combine_stamp_base: cutlass.Constexpr[int],
        combine_ready_base: cutlass.Constexpr[int],
        combine_credit_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Return expert output, reduce route weights, and release both banks."""

        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        arena_address = cutlass.Uint64(arena.iterator.toint())
        expert_output_address = cutlass.Uint64(expert_output.iterator.toint())
        combine_output_address = cutlass.Uint64(combine_output.iterator.toint())
        route_counts_address = cutlass.Uint64(route_counts.iterator.toint())
        status_address = cutlass.Uint64(statuses.iterator.toint())
        bank = operation % NUM_BANKS
        epoch = _wire_epoch(generation, operation)
        sequence = _ready_sequence(operation)
        rank_active = cutlass.Int32(0)
        if lane == 0:
            if rank_mask[rank] == 0:
                rank_active = cutlass.Int32(1)
        rank_active = cute.arch.shuffle_sync(rank_active, 0)

        # Do not overwrite any origin's route-slot bank until that origin has
        # completed the prior same-bank FP32 reduction.
        if operation >= NUM_BANKS:
            if cta < max_ranks:
                peer = cta
                if rank_active != 0:
                    if rank_mask[peer] == 0:
                        if peer != rank:
                            previous = _ready_sequence(operation - NUM_BANKS)
                            credit_offset = cutlass.Uint64(combine_credit_base) + (
                                cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(peer)
                            ) * cutlass.Uint64(COUNTER_NBYTES)
                            observed = nixl_cute.wait_acquire_system_u64(
                                arena_address + cutlass.Uint64(credit_offset),
                                previous,
                                scope=nixl_cute.Scope.WARP,
                            )
                            if observed != previous:
                                if lane == 0:
                                    _record_status(
                                        status_address,
                                        peer,
                                        int(nixl_cute.NIXL_ERR_MISMATCH),
                                    )
            nixl_cute.sync_grid()

        # Every worker warp owns one deterministic contiguous shard of an
        # (origin rank, local expert) source segment.  Coherent loads are
        # mandatory because expert_output was written by the immediately
        # preceding device kernel.  The grid join below covers every shard
        # before any origin receives its completion publication.
        bucket_count = max_ranks * experts_per_rank
        bucket = cta % bucket_count
        shard = cta // bucket_count
        shard_count = ((worker_ctas - 1 - bucket) // bucket_count) + 1
        base_shard_count = worker_ctas // bucket_count
        peer = bucket // experts_per_rank
        local_expert = bucket % experts_per_rank
        peer_active = cutlass.Int32(0)
        peer_address = cutlass.Uint64(0)
        if lane == 0:
            if rank_mask[peer] == 0:
                peer_active = cutlass.Int32(1)
            peer_address = cutlass.Uint64(peer_bases[peer])
        peer_active = cute.arch.shuffle_sync(peer_active, 0)
        peer_address = cute.arch.shuffle_sync(peer_address, 0)
        if rank_active != 0:
            if peer_active != 0:
                layout_item = (
                    cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                    + cutlass.Uint64(local_expert)
                ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(peer)
                layout_value = cutlass.Uint64(0)
                if lane == 0:
                    layout_value = nixl_cute.load_acquire_gpu_u64(
                        arena_address
                        + cutlass.Uint64(dispatch_layout_base)
                        + layout_item * cutlass.Uint64(COUNTER_NBYTES)
                    )
                layout_value = cute.arch.shuffle_sync(layout_value, 0)
                begin = layout_value >> 32
                record_count = layout_value & cutlass.Uint64(UINT32_MAX)
                shard_begin = cutlass.Uint64(0)
                shard_end = cutlass.Uint64(0)
                if cutlass.const_expr(worker_ctas % bucket_count == 0):
                    shard_begin = (
                        record_count * cutlass.Uint64(shard)
                    ) // cutlass.Uint64(base_shard_count)
                    shard_end = (
                        record_count * cutlass.Uint64(shard + 1)
                    ) // cutlass.Uint64(base_shard_count)
                else:
                    if shard_count > base_shard_count:
                        shard_begin = (
                            record_count * cutlass.Uint64(shard)
                        ) // cutlass.Uint64(base_shard_count + 1)
                        shard_end = (
                            record_count * cutlass.Uint64(shard + 1)
                        ) // cutlass.Uint64(base_shard_count + 1)
                    else:
                        shard_begin = (
                            record_count * cutlass.Uint64(shard)
                        ) // cutlass.Uint64(base_shard_count)
                        shard_end = (
                            record_count * cutlass.Uint64(shard + 1)
                        ) // cutlass.Uint64(base_shard_count)
                source_slot = shard_begin
                bucket_ok = cutlass.Int32(1)
                while source_slot < shard_end:
                    dense_slot = begin + source_slot
                    row_item = (
                        cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                        + cutlass.Uint64(local_expert)
                    ) * (
                        cutlass.Uint64(max_ranks) * cutlass.Uint64(token_capacity)
                    ) + dense_slot
                    origin_rank = cutlass.Uint32(0)
                    origin_token = cutlass.Uint32(0)
                    route_slot = cutlass.Uint32(0)
                    if lane == 0:
                        info_offset = cutlass.Uint64(
                            dispatch_src_info_base
                        ) + row_item * cutlass.Uint64(SOURCE_INFO_NBYTES)
                        source_info = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint32,
                                arena_address + cutlass.Uint64(info_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(SOURCE_INFO_WORDS),
                        )
                        # As with the Uint64 stamp loads above, make the
                        # signless tensor elements explicitly unsigned before
                        # they cross this dynamic branch and the following warp
                        # broadcast.
                        origin_rank = cutlass.Int32(source_info[0]).bitcast(
                            cutlass.Uint32
                        )
                        origin_token = cutlass.Int32(source_info[1]).bitcast(
                            cutlass.Uint32
                        )
                        route_slot = cutlass.Int32(source_info[2]).bitcast(
                            cutlass.Uint32
                        )
                        if origin_rank != peer:
                            bucket_ok = cutlass.Int32(0)
                        # This row originated on ``peer``; peer live extent is
                        # intentionally absent from the hot contract. Its fixed
                        # registered capacity is the safe bound.
                        if origin_token >= cutlass.Uint32(token_capacity):
                            bucket_ok = cutlass.Int32(0)
                        if route_slot >= cutlass.Uint32(top_k):
                            bucket_ok = cutlass.Int32(0)
                        if source_info[3] != 0:
                            bucket_ok = cutlass.Int32(0)
                    origin_token = cute.arch.shuffle_sync(origin_token, 0)
                    route_slot = cute.arch.shuffle_sync(route_slot, 0)
                    bucket_ok = cute.arch.shuffle_sync(bucket_ok, 0)
                    if bucket_ok != 0:
                        output_item = (
                            cutlass.Uint64(local_expert)
                            * (
                                cutlass.Uint64(max_ranks)
                                * cutlass.Uint64(token_capacity)
                            )
                            + dense_slot
                        )
                        combine_item = (
                            cutlass.Uint64(bank) * cutlass.Uint64(top_k)
                            + cutlass.Uint64(route_slot)
                        ) * cutlass.Uint64(token_capacity) + cutlass.Uint64(
                            origin_token
                        )
                        copy_status = nixl_cute.mapped_copy_warp_ptr(
                            expert_output_address
                            + output_item
                            * cutlass.Uint64(hidden_size)
                            * cutlass.Uint64(BF16_NBYTES),
                            peer_address
                            + cutlass.Uint64(combine_recv_base)
                            + combine_item * cutlass.Uint64(payload_stride),
                            cutlass.Uint64(hidden_size * BF16_NBYTES),
                        )
                        if copy_status != int(nixl_cute.NIXL_SUCCESS):
                            bucket_ok = cutlass.Int32(0)
                            if lane == 0:
                                _record_status(status_address, peer, copy_status)
                    source_slot += cutlass.Uint64(1)
                if bucket_ok == 0:
                    if lane == 0:
                        _record_status(
                            status_address,
                            peer,
                            int(nixl_cute.NIXL_ERR_MISMATCH),
                        )

        # All expert CTAs have finished their mapped writes before one leader
        # per origin performs the cumulative system-release publication.
        nixl_cute.sync_grid()
        if cta < max_ranks:
            peer = cta
            if rank_active != 0:
                if rank_mask[peer] == 0:
                    peer_address = cutlass.Uint64(0)
                    if lane == 0:
                        peer_address = cutlass.Uint64(peer_bases[peer])
                    peer_address = cute.arch.shuffle_sync(peer_address, 0)
                    if lane == 0:
                        stamp_item = cutlass.Uint64(bank) * cutlass.Uint64(
                            max_ranks
                        ) + cutlass.Uint64(rank)
                        stamp_offset = cutlass.Uint64(
                            combine_stamp_base
                        ) + stamp_item * cutlass.Uint64(BUCKET_STAMP_NBYTES)
                        stamp = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint64,
                                peer_address + cutlass.Uint64(stamp_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(BUCKET_STAMP_WORDS),
                        )
                        total = cutlass.Uint64(0)
                        for local_expert in cutlass.range(experts_per_rank, unroll=1):
                            layout_item = (
                                cutlass.Uint64(bank) * cutlass.Uint64(experts_per_rank)
                                + cutlass.Uint64(local_expert)
                            ) * cutlass.Uint64(max_ranks) + cutlass.Uint64(peer)
                            layout_value = nixl_cute.load_acquire_gpu_u64(
                                arena_address
                                + cutlass.Uint64(dispatch_layout_base)
                                + layout_item * cutlass.Uint64(COUNTER_NBYTES)
                            )
                            total += layout_value & cutlass.Uint64(UINT32_MAX)
                        stamp[0] = epoch
                        stamp[1] = source_incarnation
                        failed = cutlass.Uint64(0)
                        for candidate in cutlass.range(max_ranks, unroll=1):
                            if rank_mask[candidate] == 0:
                                if statuses[candidate] != 0:
                                    failed = cutlass.Uint64(1)
                        stamp[2] = total
                        stamp[3] = failed
                        ready_offset = cutlass.Uint64(combine_ready_base) + (
                            cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                            + cutlass.Uint64(rank)
                        ) * cutlass.Uint64(COUNTER_NBYTES)
                        nixl_cute.store_release_system_u64(
                            peer_address + cutlass.Uint64(ready_offset), sequence
                        )
                        # The combine launch is ordered after external expert
                        # compute on the same stream, so this also safely returns
                        # the peer's dispatch bank.
                        dispatch_credit_offset = cutlass.Uint64(
                            dispatch_credit_base
                        ) + (
                            cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                            + cutlass.Uint64(rank)
                        ) * cutlass.Uint64(
                            COUNTER_NBYTES
                        )
                        nixl_cute.store_release_system_u64(
                            peer_address + cutlass.Uint64(dispatch_credit_offset),
                            sequence,
                        )

                    incoming_ready_offset = cutlass.Uint64(combine_ready_base) + (
                        cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                        + cutlass.Uint64(peer)
                    ) * cutlass.Uint64(COUNTER_NBYTES)
                    observed = nixl_cute.wait_acquire_system_u64(
                        arena_address + cutlass.Uint64(incoming_ready_offset),
                        sequence,
                        scope=nixl_cute.Scope.WARP,
                    )
                    incoming_ok = cutlass.Int32(0)
                    if observed == sequence:
                        incoming_ok = cutlass.Int32(1)
                    if lane == 0:
                        incoming_stamp_offset = cutlass.Uint64(combine_stamp_base) + (
                            cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                            + cutlass.Uint64(peer)
                        ) * cutlass.Uint64(BUCKET_STAMP_NBYTES)
                        incoming_stamp = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint64,
                                arena_address + cutlass.Uint64(incoming_stamp_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(BUCKET_STAMP_WORDS),
                        )
                        if incoming_stamp[0] != epoch:
                            incoming_ok = cutlass.Int32(0)
                        if incoming_stamp[1] != rank_incarnations[peer]:
                            incoming_ok = cutlass.Int32(0)
                        published_count = cutlass.Int64(incoming_stamp[2]).bitcast(
                            cutlass.Uint64
                        )
                        # Unsigned comparison is fail-closed for corrupt wire
                        # counts with the high bit set; treating those bits as a
                        # negative Int64 would let them evade the capacity gate.
                        if published_count > cutlass.Uint64(live_tokens * top_k):
                            incoming_ok = cutlass.Int32(0)
                        if incoming_stamp[3] != 0:
                            incoming_ok = cutlass.Int32(0)
                        if incoming_ok == 0:
                            _record_status(
                                status_address,
                                peer,
                                int(nixl_cute.NIXL_ERR_MISMATCH),
                            )

        # A local cooperative join distributes every peer leader's system
        # acquire before comparing its published total with the current bank's
        # retained route allocators.  Dispatch clears these only after prior
        # same-bank credit, so they remain immutable through this combine.
        nixl_cute.sync_grid()
        if cta == 0:
            if rank_active != 0:
                incoming_partial = cutlass.Uint64(0)
                if lane < max_ranks:
                    peer = lane
                    if rank_mask[peer] == 0:
                        incoming_stamp_offset = cutlass.Uint64(combine_stamp_base) + (
                            cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                            + cutlass.Uint64(peer)
                        ) * cutlass.Uint64(BUCKET_STAMP_NBYTES)
                        incoming_stamp = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint64,
                                arena_address + cutlass.Uint64(incoming_stamp_offset),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(BUCKET_STAMP_WORDS),
                        )
                        # See the dispatch-side stamp load above.  This value is
                        # carried out of two dynamic branches before the warp
                        # reduction, so materialize its unsigned type too.
                        incoming_partial = cutlass.Int64(incoming_stamp[2]).bitcast(
                            cutlass.Uint64
                        )
                incoming_total = _warp_sum_u64(incoming_partial)

                expected_partial = cutlass.Uint64(0)
                # Keep this reduction cursor distinct from the Int32 worker
                # geometry ``bucket`` above.  CuTe tracks names across dynamic
                # regions and rejects reusing one with different signedness.
                route_bucket = cutlass.Uint32(lane)
                while route_bucket < max_ranks * experts_per_rank:
                    expected_partial += nixl_cute.load_acquire_gpu_u64(
                        route_counts_address
                        + cutlass.Uint64(route_bucket)
                        * cutlass.Uint64(COUNTER_NBYTES)
                    )
                    route_bucket += cutlass.Uint32(WARP_SIZE)
                expected_total = _warp_sum_u64(expected_partial)
                if lane == 0:
                    if incoming_total != expected_total:
                        _record_status(
                            status_address,
                            rank,
                            int(nixl_cute.NIXL_ERR_MISMATCH),
                        )

        # This join distributes the aggregate route-count check.  It is also
        # the fail-closed gate preventing stale combine slots from being read.
        nixl_cute.sync_grid()
        combine_ok = cutlass.Int32(0)
        if lane == 0:
            if rank_active != 0:
                combine_ok = cutlass.Int32(1)
                for peer in cutlass.range(max_ranks, unroll=1):
                    if rank_mask[peer] == 0:
                        if statuses[peer] != 0:
                            combine_ok = cutlass.Int32(0)
        combine_ok = cute.arch.shuffle_sync(combine_ok, 0)
        if combine_ok != 0:
            vector_layout = cute.make_layout((4,))
            combine_words_layout = cute.make_layout(
                (4, COMBINE_VECTOR_UNROLL), stride=(1, 4)
            )
            combine_values_layout = cute.make_layout(
                (BF16_PER_VECTOR, COMBINE_VECTOR_UNROLL),
                stride=(1, BF16_PER_VECTOR),
            )
            load = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                cutlass.Uint32,
                num_bits_per_copy=128,
                memory_order=cute.nvgpu.MemoryOrder.WEAK,
                memory_scope=cute.nvgpu.MemoryScope.CTA,
                l2_prefetch_size=cute.nvgpu.L2PrefetchSize.SIZE_256B,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            store = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                cutlass.Uint32,
                num_bits_per_copy=128,
                memory_order=cute.nvgpu.MemoryOrder.WEAK,
                memory_scope=cute.nvgpu.MemoryScope.CTA,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            combine_words = cute.make_rmem_tensor(combine_words_layout, cutlass.Uint32)
            accumulator_values = cute.make_rmem_tensor(
                combine_values_layout, cutlass.Float32
            )
            route_weights = cute.make_rmem_tensor(
                cute.make_layout(top_k), cutlass.Float32
            )
            total_vectors = hidden_size // BF16_PER_VECTOR
            combine_warp_chunk = WARP_SIZE * COMBINE_VECTOR_UNROLL
            full_vectors = (total_vectors // combine_warp_chunk) * combine_warp_chunk
            token = cutlass.Uint32(cta)
            while token < live_tokens:
                lane_weight = cutlass.Float32(0.0)
                if lane < top_k:
                    lane_weight = topk_weights[(token, lane)]
                for route_slot in cutlass.range_constexpr(top_k):
                    route_weights[route_slot] = cute.arch.shuffle_sync(
                        lane_weight, route_slot
                    )

                # U2 is the retained elastic kernel's qualified latency-hiding
                # shape: issue two independent LDG.128 instructions before the
                # dependent BF16-to-FP32 math.  The generic U1 tail below keeps
                # every H%8 contract valid without scalar loads.
                vector_base = cutlass.Uint32(lane)
                while vector_base < full_vectors:
                    accumulator_values.fill(0.0)
                    for route_slot in cutlass.range_constexpr(top_k):
                        selected = cutlass.Int32(
                            topk_indices[(token, route_slot)]
                        )
                        if selected >= 0:
                            if selected < max_ranks * experts_per_rank:
                                owner = selected // experts_per_rank
                                if rank_mask[owner] == 0:
                                    combine_item = (
                                        cutlass.Uint64(bank) * cutlass.Uint64(top_k)
                                        + cutlass.Uint64(route_slot)
                                    ) * cutlass.Uint64(token_capacity) + cutlass.Uint64(
                                        token
                                    )
                                    route_base = (
                                        arena_address
                                        + cutlass.Uint64(combine_recv_base)
                                        + combine_item * cutlass.Uint64(payload_stride)
                                    )
                                    for vector_unroll in cutlass.range_constexpr(
                                        COMBINE_VECTOR_UNROLL
                                    ):
                                        vector = vector_base + cutlass.Uint32(
                                            vector_unroll * WARP_SIZE
                                        )
                                        byte_offset = cutlass.Uint64(
                                            vector
                                        ) * cutlass.Uint64(VECTOR_NBYTES)
                                        source = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                route_base + byte_offset,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            vector_layout,
                                        )
                                        cute.copy_atom_call(
                                            load,
                                            source,
                                            combine_words[(None, vector_unroll)],
                                        )
                                    for vector_unroll in cutlass.range_constexpr(
                                        COMBINE_VECTOR_UNROLL
                                    ):
                                        accumulator = accumulator_values[
                                            (None, vector_unroll)
                                        ]
                                        fragment_bf16 = cute.recast_tensor(
                                            combine_words[(None, vector_unroll)],
                                            cutlass.BFloat16,
                                        )
                                        accumulator.store(
                                            accumulator.load()
                                            + fragment_bf16.load().to(cutlass.Float32)
                                            * route_weights[route_slot]
                                        )
                    for vector_unroll in cutlass.range_constexpr(COMBINE_VECTOR_UNROLL):
                        vector = vector_base + cutlass.Uint32(vector_unroll * WARP_SIZE)
                        byte_offset = cutlass.Uint64(vector) * cutlass.Uint64(
                            VECTOR_NBYTES
                        )
                        destination = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint32,
                                combine_output_address
                                + cutlass.Uint64(token)
                                * cutlass.Uint64(hidden_size)
                                * cutlass.Uint64(BF16_NBYTES)
                                + byte_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            vector_layout,
                        )
                        fragment = combine_words[(None, vector_unroll)]
                        fragment_bf16 = cute.recast_tensor(fragment, cutlass.BFloat16)
                        fragment_bf16.store(
                            accumulator_values[(None, vector_unroll)]
                            .load()
                            .to(cutlass.BFloat16)
                        )
                        cute.copy_atom_call(store, fragment, destination)
                    vector_base += cutlass.Uint32(combine_warp_chunk)

                if cutlass.const_expr(full_vectors != total_vectors):
                    vector = cutlass.Uint32(full_vectors) + cutlass.Uint32(lane)
                    while vector < total_vectors:
                        byte_offset = cutlass.Uint64(vector) * cutlass.Uint64(
                            VECTOR_NBYTES
                        )
                        accumulator = accumulator_values[(None, 0)]
                        accumulator.fill(0.0)
                        for route_slot in cutlass.range_constexpr(top_k):
                            selected = cutlass.Int32(
                                topk_indices[(token, route_slot)]
                            )
                            if selected >= 0:
                                if selected < max_ranks * experts_per_rank:
                                    owner = selected // experts_per_rank
                                    if rank_mask[owner] == 0:
                                        combine_item = (
                                            cutlass.Uint64(bank) * cutlass.Uint64(top_k)
                                            + cutlass.Uint64(route_slot)
                                        ) * cutlass.Uint64(
                                            token_capacity
                                        ) + cutlass.Uint64(
                                            token
                                        )
                                        source = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                arena_address
                                                + cutlass.Uint64(combine_recv_base)
                                                + combine_item
                                                * cutlass.Uint64(payload_stride)
                                                + byte_offset,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            vector_layout,
                                        )
                                        fragment = combine_words[(None, 0)]
                                        cute.copy_atom_call(load, source, fragment)
                                        fragment_bf16 = cute.recast_tensor(
                                            fragment, cutlass.BFloat16
                                        )
                                        accumulator.store(
                                            accumulator.load()
                                            + fragment_bf16.load().to(cutlass.Float32)
                                            * route_weights[route_slot]
                                        )
                        destination = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint32,
                                combine_output_address
                                + cutlass.Uint64(token)
                                * cutlass.Uint64(hidden_size)
                                * cutlass.Uint64(BF16_NBYTES)
                                + byte_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            vector_layout,
                        )
                        fragment = combine_words[(None, 0)]
                        fragment_bf16 = cute.recast_tensor(fragment, cutlass.BFloat16)
                        fragment_bf16.store(accumulator.load().to(cutlass.BFloat16))
                        cute.copy_atom_call(store, fragment, destination)
                        vector += cutlass.Uint32(WARP_SIZE)
                token += cutlass.Uint32(worker_ctas)

        # Every reducer is done before the origin releases this combine bank to
        # its expert owners.  The next same-bank combine waits on these words.
        nixl_cute.sync_grid()
        if cta < max_ranks:
            peer = cta
            if rank_active != 0:
                if rank_mask[peer] == 0:
                    if peer != rank:
                        peer_address = cutlass.Uint64(0)
                        if lane == 0:
                            peer_address = cutlass.Uint64(peer_bases[peer])
                        peer_address = cute.arch.shuffle_sync(peer_address, 0)
                        if lane == 0:
                            credit_offset = cutlass.Uint64(combine_credit_base) + (
                                cutlass.Uint64(bank) * cutlass.Uint64(max_ranks)
                                + cutlass.Uint64(rank)
                            ) * cutlass.Uint64(COUNTER_NBYTES)
                            nixl_cute.store_release_system_u64(
                                peer_address + cutlass.Uint64(credit_offset),
                                sequence,
                            )

        # Failure stamps and both classes of returned credit are visible before
        # the single terminal assertion.  Consequently a recorded output event
        # cannot report normal completion with stale data, and peer waits for
        # this operation are not silently stranded.
        nixl_cute.sync_grid()
        if cta == 0:
            if lane == 0:
                if combine_ok == 0:
                    _device_trap()

    @cute.jit
    def launch_mapped_combine(
        arena: cute.Tensor,
        expert_output: cute.Tensor,
        topk_indices: cute.Tensor,
        topk_weights: cute.Tensor,
        combine_output: cute.Tensor,
        route_counts: cute.Tensor,
        rank_mask: cute.Tensor,
        rank_incarnations: cute.Tensor,
        peer_bases: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        generation: cutlass.Uint32,
        operation: cutlass.Uint32,
        source_incarnation: cutlass.Uint64,
        rank: cutlass.Constexpr[int],
        max_ranks: cutlass.Constexpr[int],
        experts_per_rank: cutlass.Constexpr[int],
        token_capacity: cutlass.Constexpr[int],
        live_tokens: cutlass.Constexpr[int],
        worker_ctas: cutlass.Constexpr[int],
        top_k: cutlass.Constexpr[int],
        hidden_size: cutlass.Constexpr[int],
        dispatch_src_info_base: cutlass.Constexpr[int],
        dispatch_layout_base: cutlass.Constexpr[int],
        dispatch_credit_base: cutlass.Constexpr[int],
        combine_recv_base: cutlass.Constexpr[int],
        combine_stamp_base: cutlass.Constexpr[int],
        combine_ready_base: cutlass.Constexpr[int],
        combine_credit_base: cutlass.Constexpr[int],
        payload_stride: cutlass.Constexpr[int],
    ):
        """Enqueue reverse scatter and weighted reduction on ``stream``."""

        _mapped_combine_kernel(
            arena,
            expert_output,
            topk_indices,
            topk_weights,
            combine_output,
            route_counts,
            rank_mask,
            rank_incarnations,
            peer_bases,
            statuses,
            generation,
            operation,
            source_incarnation,
            rank,
            max_ranks,
            experts_per_rank,
            token_capacity,
            live_tokens,
            worker_ctas,
            top_k,
            hidden_size,
            dispatch_src_info_base,
            dispatch_layout_base,
            dispatch_credit_base,
            combine_recv_base,
            combine_stamp_base,
            combine_ready_base,
            combine_credit_base,
            payload_stride,
        ).launch(
            grid=[worker_ctas, 1, 1],
            block=[WARP_SIZE, 1, 1],
            stream=stream,
            cooperative=True,
            min_blocks_per_mp=1,
        )

else:

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "CuTe DSL and the NIXL CuTe device package are required for "
            "pipeline kernel launches"
        )

    launch_resolve_mapped_peers = _unavailable
    launch_mapped_dispatch = _unavailable
    launch_standin_expert = _unavailable
    launch_mapped_combine = _unavailable


__all__ = [
    "BF16_PER_VECTOR",
    "BUCKET_STAMP_NBYTES",
    "COUNTER_NBYTES",
    "INT32_MAX",
    "KernelContract",
    "NUM_BANKS",
    "SOURCE_INFO_NBYTES",
    "STANDIN_EXPERT_BIAS_SCALE",
    "is_cute_available",
    "kernel_contract",
    "launch_mapped_combine",
    "launch_mapped_dispatch",
    "launch_resolve_mapped_peers",
    "launch_standin_expert",
]
