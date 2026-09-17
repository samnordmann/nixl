#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark persistent NIXL transfers from CuTe DSL without host pacing.

Two processes use one GPU each.  Rank 0 launches one persistent producer
kernel and rank 1 launches one persistent receiver kernel.  No CPU operation,
CUDA event, or kernel launch occurs inside the measured loop.

Both mutually waiting grids use cooperative launch. Each process retains the
exact compiled CUBIN, binds that specialization on its target GPU, and rejects
the launch unless the CUDA occupancy API proves that the full grid is resident.

``wait`` measures request-backed PUT completion; with ``batch_size=1`` this
matches NIXLBench's complete PUT sample.  ``defer`` measures requestless PUT
posting with the production EP doorbell cadence: three deferred PUTs followed
by one non-deferred PUT.  ``mapped`` resolves the peer payload pointer once
before timing, then performs one coalesced warp-vectorized direct copy per group
with coherent, L1-no-allocate source loads. A copy is complete when the
primitive's final warp barrier returns. Measurements stop before publishing
completion: wait/defer use an ordered NIXL atomic, whereas mapped uses a
lane-zero system-release store.
The receiver observes completion with a timer-free system-acquire wait by
default and returns a GPU credit through the same transport class. Bit 62 is an
in-band abort sentinel, while successful iteration counters remain below it.
The producer never mutates/reuses a source generation until that credit is
observed on the GPU. Every batch slot has its own 64-bit generation word. The
persistent benchmark kernel validates counters continuously and validates every
slot's final state
after completion; it deliberately does not scan payloads before every credit,
which would contaminate the performance path. Registered source writes are
followed by a device system-release fence before the next transport PUT.

Mapped mode first requires CUDA native peer atomics for both directed GPU pairs,
then runs one setup-only device preflight against the exact peer payload and
counter descriptors. Both ranks exchange that result before either persistent
grid is launched. Every descriptor must yield a non-null process-local pointer,
but the importer and owner may use different numeric virtual addresses; all
direct accesses are formed from the pointer returned by ``nixlGetPtr`` plus a
descriptor-relative offset. ``--allow-unverified-mapped`` is retained as a
deprecated compatibility no-op.

The JSON result deliberately separates producer API latency from safe-reuse
round-trip latency. In ``defer`` mode the former is posting time, not transfer
completion. Credit waits are timer-free unless a positive diagnostic device
timeout is selected. Request-backed ``wait`` mode can still block inside UCX
request progress after abrupt endpoint loss; always retain an external process
or Slurm timeout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import socket
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Sequence

import torch

try:
    from ._cooperative import bind_and_validate_cooperative_launch
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from .benchmark_put import parse_size
except ImportError:
    from _cooperative import (  # type: ignore[no-redef]
        bind_and_validate_cooperative_launch,
    )
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        complete_ucx_setup_handshake,
        normalize_agent_name,
    )
    from benchmark_put import parse_size  # type: ignore[no-redef]


WORLD_SIZE = 2
GUARD_BYTES = 64
TIMESTAMPS_PER_SAMPLE = 4
MAX_GROUPS = 4096
THREAD_GROUPS_PER_BLOCK_LIMIT = 128
WARP_GROUPS_PER_BLOCK_LIMIT = 8
EP_DOORBELL_INTERVAL = 4
SOURCE_GENERATION_MODULUS = 1 << 64
UCX_NUM_WORKERS = 2
UCX_POST_THREADS = 0
_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1
_MAX_DEVICE_TIMEOUT_NS = (1 << 63) - 1
_STATUS_MISMATCH = -5
_STATUS_NOT_SUPPORTED = -9
_MAPPED_DESCRIPTOR_ROLES = ("payload", "counter")
_MAPPED_DESCRIPTOR_COUNT = 2
# CuTe DSL 4.5 lowers unsigned constants through a signed IntegerAttr path, so
# bit 63 cannot be materialized as a compile-time Uint64.  Valid counters are
# capped at Int32; bit 62 is therefore an equally disjoint, zero-cost sentinel.
_ABORT_COUNTER_BIT = 1 << 62
_MAX_DEVICE_ITERATIONS = _INT32_MAX

try:
    _CUTE_AVAILABLE = importlib.util.find_spec("cutlass.cute") is not None
except ModuleNotFoundError:
    _CUTE_AVAILABLE = False

if _CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    import nixl.device.cute as nixl_cute


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Choose an exact launch tile without idle threads or an inner-loop guard."""

    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one divides every positive integer")


@dataclass(frozen=True, slots=True)
class PersistentCase:
    """One compile-time-specialized persistent benchmark configuration."""

    size: int
    batch_size: int
    groups: int
    channels: int
    scope: str
    mode: str
    warmup: int
    iterations: int
    # Deprecated compatibility input. Native-peer-atomic qualification, not
    # cross-process numeric address equality, gates mapped execution.
    allow_unverified_mapped: bool = False
    device_timeout_ns: int = 0

    def __post_init__(self) -> None:
        for name in ("size", "batch_size", "groups", "channels", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int):
            raise ValueError("warmup must be a non-negative integer")
        if self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if self.groups > MAX_GROUPS:
            raise ValueError(f"groups must not exceed {MAX_GROUPS}")
        if self.channels > self.groups:
            raise ValueError("channels must not exceed groups")
        if self.scope not in {"thread", "warp"}:
            raise ValueError("scope must be 'thread' or 'warp'")
        if self.mode not in {"wait", "defer", "mapped"}:
            raise ValueError("mode must be 'wait', 'defer', or 'mapped'")
        if self.mode == "mapped" and self.scope != "warp":
            raise ValueError("mapped mode requires warp scope")
        if self.mode == "mapped" and self.size % 16:
            raise ValueError("mapped mode requires size to be a multiple of 16 bytes")
        if self.size < 8 or self.size % 8:
            raise ValueError(
                "size must be an 8-byte multiple for generation validation"
            )
        if not isinstance(self.allow_unverified_mapped, bool):
            raise TypeError("allow_unverified_mapped must be bool")
        if (
            isinstance(self.device_timeout_ns, bool)
            or not isinstance(self.device_timeout_ns, int)
            or self.device_timeout_ns < 0
            or self.device_timeout_ns > _MAX_DEVICE_TIMEOUT_NS
        ):
            raise ValueError("device_timeout_ns must fit CuTe's signed literal range")
        if self.total_iterations > _MAX_DEVICE_ITERATIONS:
            raise ValueError("warmup + iterations exceeds the Int32 device loop limit")
        if self.iterations * self.groups * TIMESTAMPS_PER_SAMPLE > _INT32_MAX:
            raise ValueError("measurement timestamp indexing exceeds Int32")
        if self.payload_bytes > _UINT64_MAX - 2 * GUARD_BYTES:
            raise ValueError("payload allocation exceeds uint64")

    @property
    def total_iterations(self) -> int:
        return self.warmup + self.iterations

    @property
    def payload_bytes(self) -> int:
        return self.size * self.batch_size * self.groups

    @property
    def bytes_per_iteration(self) -> int:
        return self.payload_bytes

    @property
    def generation_words_per_group(self) -> int:
        return self.batch_size

    @property
    def generation_words_total(self) -> int:
        return self.groups * self.generation_words_per_group

    @property
    def threads(self) -> int:
        return self.groups_per_block * (1 if self.scope == "thread" else 32)

    @property
    def groups_per_block(self) -> int:
        limit = (
            THREAD_GROUPS_PER_BLOCK_LIMIT
            if self.scope == "thread"
            else WARP_GROUPS_PER_BLOCK_LIMIT
        )
        return _largest_divisor_at_most(self.groups, limit)

    @property
    def blocks(self) -> int:
        return self.groups // self.groups_per_block


def _delta_ns(start: int, end: int) -> int:
    """Return a uint64 timer delta and reject reversed host-side samples."""

    if end < start:
        raise RuntimeError("%globaltimer sample moved backwards")
    return end - start


def nixlbench_percentile(values: Sequence[int], q: float) -> int:
    """Return NIXLBench's upper-order-statistic percentile.

    NIXLBench sorts its samples and selects ``min(int(n * q), n - 1)``.
    Keeping that exact index rule avoids interpolation artifacts in paired
    CuTe/NIXLBench comparisons.
    """

    if not values:
        raise ValueError("cannot summarize an empty sample")
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise TypeError("q must be a real number")
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0, 1]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("samples must be non-negative integers")
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def defer_put_for_slot(slot: int) -> bool:
    """Match ``nixl_ep_ll.cu::doorbell_flag`` for a zero-based PUT slot."""

    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError("slot must be a non-negative integer")
    return (slot + 1) % EP_DOORBELL_INTERVAL != 0


def source_generation(iteration: int) -> int:
    """Return the uint64 generation sent by one group during ``iteration``."""

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    return iteration % SOURCE_GENERATION_MODULUS


def summarize_timestamps(
    case: PersistentCase,
    timestamps: Sequence[int],
    *,
    source_device: int,
    target_device: int,
    backend_parameters: dict[str, str] | None = None,
    cooperative_occupancy: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Summarize ``(start, API done, credit, next-cycle boundary)`` timestamps."""

    expected = case.iterations * case.groups * TIMESTAMPS_PER_SAMPLE
    if len(timestamps) != expected:
        raise ValueError(f"expected {expected} timestamps, received {len(timestamps)}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in timestamps
    ):
        raise ValueError("timestamps must be non-negative integers")

    producer_api_ns: list[int] = []
    safe_reuse_ns: list[int] = []
    full_cycle_ns: list[int] = []
    cohort_api_span_ns: list[int] = []
    cohort_safe_reuse_span_ns: list[int] = []
    cohort_full_cycle_span_ns: list[int] = []
    raw_iterations: list[list[list[int]]] = []
    cursor = 0
    for _ in range(case.iterations):
        iteration_rows: list[list[int]] = []
        typed_rows: list[tuple[int, int, int, int]] = []
        for _ in range(case.groups):
            start, api_done, credit, cycle = (
                int(timestamps[cursor]),
                int(timestamps[cursor + 1]),
                int(timestamps[cursor + 2]),
                int(timestamps[cursor + 3]),
            )
            cursor += TIMESTAMPS_PER_SAMPLE
            if not (start <= api_done <= credit <= cycle):
                raise RuntimeError("invalid GPU timestamp ordering")
            row = (start, api_done, credit, cycle)
            typed_rows.append(row)
            iteration_rows.append([start, api_done, credit, cycle])
            producer_api_ns.append(_delta_ns(start, api_done))
            safe_reuse_ns.append(_delta_ns(start, credit))
            full_cycle_ns.append(_delta_ns(start, cycle))
        first_start = min(row[0] for row in typed_rows)
        cohort_api_span_ns.append(max(row[1] for row in typed_rows) - first_start)
        cohort_safe_reuse_span_ns.append(
            max(row[2] for row in typed_rows) - first_start
        )
        cohort_full_cycle_span_ns.append(
            max(row[3] for row in typed_rows) - first_start
        )
        raw_iterations.append(iteration_rows)

    def metrics(samples: Sequence[int], payload_bytes: int) -> dict[str, float | int]:
        p50 = nixlbench_percentile(samples, 0.50)
        p90 = nixlbench_percentile(samples, 0.90)
        p99 = nixlbench_percentile(samples, 0.99)
        return {
            "p50_ns": p50,
            "p90_ns": p90,
            "p99_ns": p99,
            "max_ns": max(samples),
            "logical_GBps_at_p50": payload_bytes / p50 if p50 else math.inf,
        }

    group_payload = case.size * case.batch_size
    run_start = min(row[0] for row in raw_iterations[0])
    run_end = max(row[3] for row in raw_iterations[-1])
    run_duration_ns = _delta_ns(run_start, run_end)
    run_payload_bytes = case.iterations * case.bytes_per_iteration
    return {
        "schema_version": 6,
        "case": "nixl_cute_persistent_transfer",
        "mode": case.mode,
        "scope": case.scope,
        "size_bytes": case.size,
        "batch_size": case.batch_size,
        "groups": case.groups,
        "launch": {
            "blocks": case.blocks,
            "groups_per_block": case.groups_per_block,
            "threads_per_block": case.threads,
            "multi_cta": case.blocks > 1,
            "cooperative": True,
            "residency_contract": (
                "planned CTAs <= exact active-block capacity of each bound "
                "producer/receiver CUfunction; enforced from retained CUBIN "
                "before cooperative launch"
            ),
            "occupancy": {
                role: dict(evidence)
                for role, evidence in (cooperative_occupancy or {}).items()
            },
        },
        "channels": case.channels,
        "channel_mapping": (
            "not used by direct mapped path"
            if case.mode == "mapped"
            else "group_id % channels"
        ),
        "warmup": case.warmup,
        "iterations": case.iterations,
        "compatibility": {
            "allow_unverified_mapped": {
                "value": case.allow_unverified_mapped,
                "deprecated": True,
                "affects_execution": False,
            },
        },
        "logical_bytes_per_iteration": case.bytes_per_iteration,
        "source_device": source_device,
        "target_device": target_device,
        "producer_api": {
            "samples_ns": producer_api_ns,
            **metrics(producer_api_ns, group_payload),
            "semantics": (
                "elapsed time around request-backed PUTs, including local request "
                "completion; WARP request completion returns converged; completion "
                "atomic excluded"
                if case.mode == "wait"
                else (
                    "elapsed time around requestless PUT posts using the EP 3-DEFER/"
                    "1-NONE cadence; completion atomic excluded and outstanding "
                    "transport work may remain"
                    if case.mode == "defer"
                    else "one completed warp-vectorized direct mapped copy of the "
                    "contiguous batch through a pre-resolved peer pointer using "
                    "coherent L1-no-allocate source loads, including one final warp "
                    "barrier; pointer resolution and release-store publish excluded"
                )
            ),
        },
        "safe_source_reuse_round_trip": {
            "samples_ns": safe_reuse_ns,
            **metrics(safe_reuse_ns, group_payload),
            "semantics": (
                "producer start through receiver system-acquire and returned "
                "GPU credit observed with system-acquire"
            ),
        },
        "full_iteration_cycle": {
            "samples_ns": full_cycle_ns,
            **metrics(full_cycle_ns, group_payload),
            "semantics": (
                "producer start through returned credit, source generation update, "
                "warp convergence, any required system-release fence, and the "
                "next-iteration boundary"
            ),
        },
        "iteration_cohort_producer_api_span": {
            "samples_ns": cohort_api_span_ns,
            **metrics(cohort_api_span_ns, case.bytes_per_iteration),
            "metric_kind": "cohort envelope, not steady-state throughput",
        },
        "iteration_cohort_safe_source_reuse_span": {
            "samples_ns": cohort_safe_reuse_span_ns,
            **metrics(cohort_safe_reuse_span_ns, case.bytes_per_iteration),
            "metric_kind": "cohort envelope, not steady-state throughput",
        },
        "iteration_cohort_full_cycle_span": {
            "samples_ns": cohort_full_cycle_span_ns,
            **metrics(cohort_full_cycle_span_ns, case.bytes_per_iteration),
            "metric_kind": "cohort envelope, not steady-state throughput",
        },
        "steady_state_run": {
            "duration_ns": run_duration_ns,
            "logical_bytes": run_payload_bytes,
            "logical_GBps": (
                run_payload_bytes / run_duration_ns if run_duration_ns else math.inf
            ),
            "semantics": (
                "all measured group-iterations divided by the interval from the "
                "earliest first measured start to the latest final cycle boundary"
            ),
        },
        "raw_gpu_timestamps_ns": raw_iterations,
        "metric_definitions": {
            "producer_api": (
                "per group: first-PUT start through last-PUT return; WARP wait mode "
                "returns converged at the same completion boundary as NIXLBench; "
                "excludes publish atomic and returned credit"
                if case.mode != "mapped"
                else "per group: one coalesced direct-copy start through completion; "
                "includes one warp barrier and excludes one-time pointer resolution, "
                "release-store publish, and returned credit"
            ),
            "safe_source_reuse_round_trip": (
                "per group: operation start through receiver acquire, returned GPU "
                "credit, and producer system-acquire"
            ),
            "iteration_cohort_producer_api_span": (
                "per iteration: earliest group start through latest group "
                "data-operation return; groups advance independently, so this is a "
                "straggler envelope, not a steady-state throughput interval"
            ),
            "iteration_cohort_safe_source_reuse_span": (
                "per iteration: earliest group start through latest returned credit; "
                "groups advance independently, so this is a straggler envelope"
            ),
            "full_iteration_cycle": (
                "per group: operation start through safe source reuse, recurring "
                "source mutation/convergence/fence, and the next operation boundary"
            ),
            "iteration_cohort_full_cycle_span": (
                "per iteration: earliest group start through latest next-operation "
                "boundary; adjacent group iterations may overlap"
            ),
            "steady_state_run": (
                "all logical bytes from every measured group-iteration divided by "
                "the complete measured run interval; use this for steady-state "
                "throughput"
            ),
            "raw_gpu_timestamps_ns_axes": "[iteration][group][timestamp]",
            "percentile_rule": "sorted[min(int(sample_count * q), sample_count - 1)]",
            "throughput_units": "decimal GB/s computed as bytes/ns",
        },
        "timing": {
            "clock": "CUDA %globaltimer",
            "timestamps_per_group_sample": [
                "producer_start",
                "producer_api_done",
                "return_credit_observed",
                "next_iteration_boundary",
            ],
            "jit_compilation_excluded": True,
            "kernel_launch_excluded": True,
            "cpu_synchronization_inside_measured_loop": False,
            "warmup_executes_inside_same_persistent_kernel": True,
            "device_operation_timeout_ns": case.device_timeout_ns or None,
            "credit_wait_mode": (
                "timer-free; externally bounded"
                if case.device_timeout_ns == 0
                else "diagnostic device timeout"
            ),
            "timeout_clock_check_period_failed_loads": (
                None if case.device_timeout_ns == 0 else 256
            ),
            "publish_atomic_excluded_from_producer_api": case.mode != "mapped",
            "publish_signal_excluded_from_producer_api": True,
            "warp_wait_returns_converged": (
                case.scope == "warp" and case.mode == "wait"
            ),
        },
        "protocol": {
            "data_path": (
                "direct_mapped_peer_memory"
                if case.mode == "mapped"
                else "nixl_device_ucx"
            ),
            "mapped_load_policy": (
                "coherent global L1::no_allocate loads"
                if case.mode == "mapped"
                else "not applicable"
            ),
            "mapped_pointer_resolution": (
                "once per group before the persistent measured loop"
                if case.mode == "mapped"
                else "not applicable"
            ),
            "mapped_batch_copy": (
                "one contiguous copy and one warp barrier per group-iteration"
                if case.mode == "mapped"
                else "not applicable"
            ),
            "publish": (
                "lane-zero system-release store after mapped-copy warp barrier"
                if case.mode == "mapped"
                else "same-channel PUT batch followed by ordered atomic-add"
            ),
            "defer_put_cadence": (
                "not applicable"
                if case.mode == "mapped"
                else "DEFER,DEFER,DEFER,NONE; repeat; final atomic NONE"
            ),
            "receiver": "system-scope acquire load of completion counter",
            "credit": (
                "lane-zero system-release store through mapped peer counter"
                if case.mode == "mapped"
                else "requestless non-deferred remote atomic-add"
            ),
            "source_reuse": "only after system-scope acquire observes returned credit",
            "abort": (
                "bit 62 of the ordered completion/credit counter; healthy counters "
                "are capped at signed 32-bit iterations"
            ),
            "mapping_contract": (
                "every descriptor must yield a non-null process-local nixlGetPtr "
                "base; mapped addresses use that base plus descriptor-relative "
                "offsets, numeric equality with the owner VA is not required, and "
                "every directed peer pair is qualified for native GPU atomics"
                if case.mode == "mapped"
                else "NIXL transport-owned"
            ),
        },
        "backend": "UCX",
        "backend_parameters": dict(backend_parameters or {}),
        "agent_configuration": {
            "enable_progress_thread": case.mode != "mapped",
            "thread_sync": "NIXL_THREAD_SYNC_DEFAULT",
            "ucx_num_workers": UCX_NUM_WORKERS,
            "ucx_post_threads": UCX_POST_THREADS,
        },
        "nixlbench_alignment": {
            "timer_boundary_comparable": (case.mode == "wait" and case.batch_size == 1),
            "required_case": "mode=wait,batch_size=1",
            "workload_equivalent": False,
            "note": (
                "The wait/batch-one timer surrounds the same request-backed PUT "
                "boundary, but this benchmark uses a per-iteration peer credit and "
                "NIXLBench does not. Queue pacing is therefore not "
                "workload-equivalent. "
                "For batch_size > 1, this benchmark also completes each PUT separately."
            ),
        },
        "validation": {
            "payload": "exact final full working set",
            "guard_bytes_each_side": GUARD_BYTES,
            "completion_counter_per_group": case.total_iterations,
            "return_credit_per_group": case.total_iterations,
            "source_generation_layout": "first uint64 word of every batch slot",
            "source_generation_words_per_group": case.generation_words_per_group,
            "source_generation_words_total": case.generation_words_total,
            "initial_source_generation": source_generation(0),
            "final_target_generation": source_generation(case.total_iterations - 1),
            "final_source_generation": source_generation(case.total_iterations),
            "source_reuse_ordering": (
                "every target batch slot retains the generation sent before the final "
                "returned credit; every source slot advances once after that credit"
            ),
            "coverage": (
                "completion and credit counters are checked after all iterations; "
                "payload bytes and every batch-slot generation word are checked for "
                "the final iteration only so validation adds no timed-loop traffic"
            ),
            "correctness": "PASS",
        },
    }


if _CUTE_AVAILABLE:

    @cute.kernel
    def _mapped_preflight_kernel(
        remote: nixl_cute.MemoryView,
        advertised_bases: cute.Tensor,
        statuses: cute.Tensor,
    ):
        """Classify the exact remote payload and counter descriptors."""

        tidx, _, _ = cute.arch.thread_idx()
        if tidx < _MAPPED_DESCRIPTOR_COUNT:
            process_local_base = cutlass.Uint64(
                cute.make_ptr(
                    cutlass.Uint8,
                    nixl_cute.get_ptr(remote, tidx),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ).toint()
            )
            statuses[tidx] = cutlass.Int32(0)
            if process_local_base == 0:
                statuses[tidx] = cutlass.Int32(int(nixl_cute.NIXL_ERR_NOT_SUPPORTED))
            if process_local_base != 0:
                if process_local_base != advertised_bases[tidx]:
                    statuses[tidx] = cutlass.Int32(int(nixl_cute.NIXL_ERR_MISMATCH))

    @cute.jit
    def _launch_mapped_preflight(
        remote: nixl_cute.MemoryView,
        advertised_bases: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
    ):
        _mapped_preflight_kernel(remote, advertised_bases, statuses).launch(
            grid=[1, 1, 1],
            block=[_MAPPED_DESCRIPTOR_COUNT, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _persistent_producer_kernel(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        source: cute.Tensor,
        credits: cute.Tensor,
        statuses: cute.Tensor,
        timestamps: cute.Tensor,
        size: cutlass.Constexpr[int],
        batch_size: cutlass.Constexpr[int],
        groups: cutlass.Constexpr[int],
        groups_per_block: cutlass.Constexpr[int],
        channels: cutlass.Constexpr[int],
        warmup: cutlass.Constexpr[int],
        iterations: cutlass.Constexpr[int],
        deferred: cutlass.Constexpr[bool],
        mapped: cutlass.Constexpr[bool],
        device_timeout_ns: cutlass.Constexpr[int],
        scope: cutlass.Constexpr,
    ):
        """Run every benchmark iteration on the GPU and retain raw timestamps."""

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(scope == nixl_cute.Scope.THREAD):
            group = block * groups_per_block + tidx
            leader = True
        else:
            group = block * groups_per_block + tidx // 32
            leader = tidx % 32 == 0
        channel = group % channels
        group_offset = cutlass.Uint64(group) * cutlass.Uint64(batch_size * size)
        counter_offset = cutlass.Uint64(group) * cutlass.Uint64(8)
        credit_address = (credits.iterator + group).toint()
        # Bit 62 is compile-time materializable through CuTe 4.5's signed host
        # IntegerAttr path and cannot collide with an Int32-bounded counter.
        abort_counter_bit = cutlass.Uint64(1 << 62)

        mapped_payload_base = cutlass.Uint64(0)
        mapped_counter_base = cutlass.Uint64(0)
        mapped_ready = True
        if cutlass.const_expr(mapped):
            # Resolve stable CUDA-IPC mappings once. Only lane zero touches the
            # descriptor, then broadcasts so every lane takes the same branch.
            if leader:
                mapped_payload_base = cutlass.Uint64(
                    cute.make_ptr(
                        cutlass.Int8,
                        nixl_cute.get_ptr(remote, index=0),
                    ).toint()
                )
                mapped_counter_base = cutlass.Uint64(
                    cute.make_ptr(
                        cutlass.Int8,
                        nixl_cute.get_ptr(remote, index=1),
                    ).toint()
                )
            mapped_payload_base = cute.arch.shuffle_sync(mapped_payload_base, 0)
            mapped_counter_base = cute.arch.shuffle_sync(mapped_counter_base, 0)
            mapped_ready = mapped_payload_base != 0
            if mapped_counter_base == 0:
                mapped_ready = False
            if mapped_payload_base == 0:
                if leader:
                    statuses[group * 2] = int(nixl_cute.NIXL_ERR_NOT_SUPPORTED)
            if mapped_counter_base == 0:
                if leader:
                    statuses[group * 2 + 1] = int(nixl_cute.NIXL_ERR_NOT_SUPPORTED)

        if mapped_ready:
            mapped_counter_address = mapped_counter_base + counter_offset
            group_source_address = (
                cutlass.Uint64(source.iterator.toint()) + group_offset
            )
            source_generation_words = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint64,
                    group_source_address,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                ),
                cute.make_layout(batch_size * size // 8),
            )
            iteration = cutlass.Int32(0)
            running = cutlass.Boolean(True)
            # ``&`` lowers directly to an i1 conjunction in CuTe DSL 4.5;
            # Python ``and`` adds an avoidable Boolean i1/i32 round-trip.
            while running & (iteration < warmup + iterations):
                started = cutlass.Uint64(0)
                if iteration >= warmup:
                    if leader:
                        started = nixl_cute.globaltimer_ns()

                if cutlass.const_expr(mapped):
                    # Batch slots are contiguous on both peers. Copy the whole
                    # group with one vector loop and one terminal warp barrier;
                    # descriptor lookup and pointer broadcast stay outside the
                    # measured loop. Coherent loads are required because each
                    # slot's generation word changes between iterations.
                    copy_status = nixl_cute.mapped_copy_warp_ptr(
                        group_source_address,
                        mapped_payload_base
                        + cutlass.Uint64(GUARD_BYTES)
                        + group_offset,
                        batch_size * size,
                    )
                    if copy_status != int(nixl_cute.NIXL_SUCCESS):
                        running = cutlass.Boolean(False)
                        if leader:
                            statuses[group * 2] = copy_status

                elif cutlass.const_expr(deferred):
                    # Match nixl_ep_ll.cu::doorbell_flag without emitting one
                    # call site per batch entry. A runtime chunk loop reuses four
                    # statically flagged calls; only the zero-to-three tail calls
                    # are specialized. The final ordered atomic rings any tail.
                    for chunk in cutlass.range(
                        batch_size // EP_DOORBELL_INTERVAL,
                        unroll=1,
                    ):
                        for chunk_slot in cutlass.range_constexpr(EP_DOORBELL_INTERVAL):
                            deferred_slot = chunk * EP_DOORBELL_INTERVAL + chunk_slot
                            deferred_offset = group_offset + cutlass.Uint64(
                                deferred_slot
                            ) * cutlass.Uint64(size)
                            put_flags = (
                                nixl_cute.Flags.DEFER
                                if chunk_slot + 1 < EP_DOORBELL_INTERVAL
                                else nixl_cute.Flags.NONE
                            )
                            deferred_status = nixl_cute.put_post(
                                local,
                                remote,
                                size,
                                local_index=0,
                                local_offset=deferred_offset,
                                remote_index=0,
                                remote_offset=cutlass.Uint64(GUARD_BYTES)
                                + deferred_offset,
                                channel=channel,
                                flags=put_flags,
                                scope=scope,
                            )
                            if deferred_status != int(nixl_cute.NIXL_IN_PROG):
                                running = cutlass.Boolean(False)
                                if leader:
                                    statuses[group * 2] = deferred_status

                    for tail_slot in cutlass.range_constexpr(
                        batch_size % EP_DOORBELL_INTERVAL
                    ):
                        tail_deferred_slot = (
                            batch_size // EP_DOORBELL_INTERVAL
                        ) * EP_DOORBELL_INTERVAL + tail_slot
                        tail_deferred_offset = group_offset + cutlass.Uint64(
                            tail_deferred_slot * size
                        )
                        tail_deferred_status = nixl_cute.put_post(
                            local,
                            remote,
                            size,
                            local_index=0,
                            local_offset=tail_deferred_offset,
                            remote_index=0,
                            remote_offset=cutlass.Uint64(GUARD_BYTES)
                            + tail_deferred_offset,
                            channel=channel,
                            flags=nixl_cute.Flags.DEFER,
                            scope=scope,
                        )
                        if tail_deferred_status != int(nixl_cute.NIXL_IN_PROG):
                            running = cutlass.Boolean(False)
                            if leader:
                                statuses[group * 2] = tail_deferred_status

                else:
                    # Request-backed completion can use one runtime loop body;
                    # its flag is constant and therefore needs no specialization.
                    for request_slot in cutlass.range(batch_size, unroll=1):
                        request_offset = group_offset + cutlass.Uint64(
                            request_slot
                        ) * cutlass.Uint64(size)
                        request_status = nixl_cute.put(
                            local,
                            remote,
                            size,
                            local_index=0,
                            local_offset=request_offset,
                            remote_index=0,
                            remote_offset=cutlass.Uint64(GUARD_BYTES) + request_offset,
                            channel=channel,
                            scope=scope,
                        )
                        if request_status != int(nixl_cute.NIXL_SUCCESS):
                            running = cutlass.Boolean(False)
                            if leader:
                                statuses[group * 2] = request_status

                api_done = cutlass.Uint64(0)
                if iteration >= warmup:
                    if leader:
                        api_done = nixl_cute.globaltimer_ns()
                expected_credit = cutlass.Uint64(iteration + 1)
                publish_value = expected_credit
                atomic_increment = cutlass.Uint64(1)
                if not running:
                    # One high-bit publication aborts the peer without
                    # colliding with any valid iteration counter.
                    publish_value = abort_counter_bit
                    atomic_increment = abort_counter_bit

                if cutlass.const_expr(mapped):
                    # mapped_copy_warp_ptr returned after its final barrier, so
                    # lane zero can now publish payload completion directly.
                    if leader:
                        nixl_cute.store_release_system_u64(
                            mapped_counter_address,
                            publish_value,
                        )
                else:
                    # Same-channel ordering publishes both request-backed
                    # and requestless PUTs. Keep it outside API timing.
                    signal_status = nixl_cute.atomic_add_post(
                        remote,
                        atomic_increment,
                        index=1,
                        offset=counter_offset,
                        channel=channel,
                        flags=nixl_cute.Flags.NONE,
                        scope=scope,
                    )
                    if signal_status != int(nixl_cute.NIXL_IN_PROG):
                        running = cutlass.Boolean(False)
                        if leader:
                            statuses[group * 2 + 1] = signal_status

                if cutlass.const_expr(device_timeout_ns == 0):
                    observed_credit = nixl_cute.wait_acquire_system_u64(
                        credit_address,
                        expected_credit,
                        scope=scope,
                    )
                else:
                    observed_credit = nixl_cute.wait_acquire_system_u64_for(
                        credit_address,
                        expected_credit,
                        device_timeout_ns,
                        scope=scope,
                    )
                if observed_credit < expected_credit:
                    running = cutlass.Boolean(False)
                    if leader:
                        statuses[group * 2 + 1] = int(
                            nixl_cute.NIXL_ERR_REMOTE_DISCONNECT
                        )
                if observed_credit >= abort_counter_bit:
                    running = cutlass.Boolean(False)
                    if leader:
                        statuses[group * 2 + 1] = int(
                            nixl_cute.NIXL_ERR_REMOTE_DISCONNECT
                        )
                credit_done = cutlass.Uint64(0)
                if iteration >= warmup:
                    if leader:
                        credit_done = nixl_cute.globaltimer_ns()

                if iteration >= warmup:
                    if leader:
                        sample = (iteration - warmup) * groups + group
                        base = sample * TIMESTAMPS_PER_SAMPLE
                        timestamps[base] = started
                        timestamps[base + 1] = api_done
                        timestamps[base + 2] = credit_done
                if iteration > warmup:
                    # Reuse this iteration's mandatory start timestamp as the
                    # prior iteration's full-cycle boundary. Store it only
                    # after the current operation's end timestamp so benchmark
                    # bookkeeping cannot bias producer API latency.
                    if leader:
                        previous_sample = (iteration - warmup - 1) * groups + group
                        timestamps[previous_sample * TIMESTAMPS_PER_SAMPLE + 3] = (
                            started
                        )

                next_iteration = iteration + 1
                # Advance every slot only after the safe-reuse timestamp. The
                # next iteration must transfer this generation for every PUT.
                if running:
                    if leader:
                        # Keep this loop induction variable distinct from the
                        # request/defer branch-local induction names above.
                        # CuTe's AST lowering otherwise merges the compile-time
                        # branches and sees a None-to-Int32 loop-carried type
                        # change when compiling the mapped specialization.
                        for generation_slot in cutlass.range(batch_size, unroll=1):
                            source_generation_words[generation_slot * (size // 8)] = (
                                cutlass.Uint64(iteration + 1)
                            )
                    if next_iteration < warmup + iterations:
                        if cutlass.const_expr(scope == nixl_cute.Scope.WARP):
                            # Only lane zero wrote source data. Its release
                            # fence orders those writes for the next UCX/NIC
                            # read; the rendezvous then holds every lane before
                            # the next converged WARP post. This replaces 32
                            # redundant system fences.
                            if cutlass.const_expr(not mapped):
                                if leader:
                                    nixl_cute.fence_release_system()
                            cute.arch.sync_warp()
                        else:
                            if cutlass.const_expr(not mapped):
                                nixl_cute.fence_release_system()
                if next_iteration == warmup + iterations:
                    # There is no next loop-start timestamp for the final
                    # sample. Pay one terminal timer read per complete run.
                    if leader:
                        final_sample = (iteration - warmup) * groups + group
                        timestamps[final_sample * TIMESTAMPS_PER_SAMPLE + 3] = (
                            nixl_cute.globaltimer_ns()
                        )
                iteration = next_iteration

    @cute.jit
    def _launch_persistent_producer(
        local: nixl_cute.MemoryView,
        remote: nixl_cute.MemoryView,
        source: cute.Tensor,
        credits: cute.Tensor,
        statuses: cute.Tensor,
        timestamps: cute.Tensor,
        stream: cuda.CUstream,
        size: cutlass.Constexpr[int],
        batch_size: cutlass.Constexpr[int],
        groups: cutlass.Constexpr[int],
        groups_per_block: cutlass.Constexpr[int],
        channels: cutlass.Constexpr[int],
        warmup: cutlass.Constexpr[int],
        iterations: cutlass.Constexpr[int],
        deferred: cutlass.Constexpr[bool],
        mapped: cutlass.Constexpr[bool],
        device_timeout_ns: cutlass.Constexpr[int],
        scope: cutlass.Constexpr,
    ):
        if cutlass.const_expr(scope == nixl_cute.Scope.THREAD):
            threads = groups_per_block
        else:
            threads = groups_per_block * 32
        _persistent_producer_kernel(
            local,
            remote,
            source,
            credits,
            statuses,
            timestamps,
            size,
            batch_size,
            groups,
            groups_per_block,
            channels,
            warmup,
            iterations,
            deferred,
            mapped,
            device_timeout_ns,
            scope,
        ).launch(
            grid=[groups // groups_per_block, 1, 1],
            block=[threads, 1, 1],
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def _persistent_receiver_kernel(
        remote: nixl_cute.MemoryView,
        completions: cute.Tensor,
        statuses: cute.Tensor,
        groups: cutlass.Constexpr[int],
        groups_per_block: cutlass.Constexpr[int],
        channels: cutlass.Constexpr[int],
        total_iterations: cutlass.Constexpr[int],
        mapped: cutlass.Constexpr[bool],
        device_timeout_ns: cutlass.Constexpr[int],
        scope: cutlass.Constexpr,
    ):
        """Turn ordered completion counters into GPU-to-GPU reuse credits."""

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(scope == nixl_cute.Scope.THREAD):
            group = block * groups_per_block + tidx
            leader = True
        else:
            group = block * groups_per_block + tidx // 32
            leader = tidx % 32 == 0
        channel = group % channels
        counter_offset = cutlass.Uint64(group) * cutlass.Uint64(8)
        completion_address = (completions.iterator + group).toint()
        # See the producer: bit 62 is the compile-safe disjoint abort domain.
        abort_counter_bit = cutlass.Uint64(1 << 62)

        mapped_counter_base = cutlass.Uint64(0)
        receiver_ready = True
        if cutlass.const_expr(mapped):
            if leader:
                mapped_counter_base = cutlass.Uint64(
                    cute.make_ptr(
                        cutlass.Int8,
                        nixl_cute.get_ptr(remote, index=1),
                    ).toint()
                )
            mapped_counter_base = cute.arch.shuffle_sync(mapped_counter_base, 0)
            receiver_ready = mapped_counter_base != 0
            if mapped_counter_base == 0:
                if leader:
                    statuses[group] = int(nixl_cute.NIXL_ERR_NOT_SUPPORTED)

        if receiver_ready:
            mapped_counter_address = mapped_counter_base + counter_offset
            iteration = cutlass.Int32(0)
            running = cutlass.Boolean(True)
            while running & (iteration < total_iterations):
                expected = cutlass.Uint64(iteration + 1)
                if cutlass.const_expr(device_timeout_ns == 0):
                    observed = nixl_cute.wait_acquire_system_u64(
                        completion_address,
                        expected,
                        scope=scope,
                    )
                else:
                    observed = nixl_cute.wait_acquire_system_u64_for(
                        completion_address,
                        expected,
                        device_timeout_ns,
                        scope=scope,
                    )
                if observed < expected:
                    running = cutlass.Boolean(False)
                    if leader:
                        statuses[group] = int(nixl_cute.NIXL_ERR_REMOTE_DISCONNECT)
                if observed >= abort_counter_bit:
                    running = cutlass.Boolean(False)
                    if leader:
                        statuses[group] = int(nixl_cute.NIXL_ERR_REMOTE_DISCONNECT)

                credit_value = expected
                atomic_increment = cutlass.Uint64(1)
                if not running:
                    credit_value = abort_counter_bit
                    atomic_increment = abort_counter_bit
                if cutlass.const_expr(mapped):
                    if leader:
                        nixl_cute.store_release_system_u64(
                            mapped_counter_address,
                            credit_value,
                        )
                else:
                    credit_status = nixl_cute.atomic_add_post(
                        remote,
                        atomic_increment,
                        index=1,
                        offset=counter_offset,
                        channel=channel,
                        flags=nixl_cute.Flags.NONE,
                        scope=scope,
                    )
                    if credit_status != int(nixl_cute.NIXL_IN_PROG):
                        running = cutlass.Boolean(False)
                        if leader:
                            statuses[group] = credit_status
                iteration += 1

    @cute.jit
    def _launch_persistent_receiver(
        remote: nixl_cute.MemoryView,
        completions: cute.Tensor,
        statuses: cute.Tensor,
        stream: cuda.CUstream,
        groups: cutlass.Constexpr[int],
        groups_per_block: cutlass.Constexpr[int],
        channels: cutlass.Constexpr[int],
        total_iterations: cutlass.Constexpr[int],
        mapped: cutlass.Constexpr[bool],
        device_timeout_ns: cutlass.Constexpr[int],
        scope: cutlass.Constexpr,
    ):
        if cutlass.const_expr(scope == nixl_cute.Scope.THREAD):
            threads = groups_per_block
        else:
            threads = groups_per_block * 32
        _persistent_receiver_kernel(
            remote,
            completions,
            statuses,
            groups,
            groups_per_block,
            channels,
            total_iterations,
            mapped,
            device_timeout_ns,
            scope,
        ).launch(
            grid=[groups // groups_per_block, 1, 1],
            block=[threads, 1, 1],
            stream=stream,
            cooperative=True,
        )


def _region(tensor: torch.Tensor) -> DeviceRegion:
    return DeviceRegion(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.get_device()
    )


def _pattern(size: int, device: int) -> torch.Tensor:
    base = torch.arange(251, dtype=torch.uint8, device=f"cuda:{device}")
    return base.repeat((size + base.numel() - 1) // base.numel())[:size].clone()


def _payload_generation_pattern(
    case: PersistentCase,
    generation: int,
    device: int,
) -> torch.Tensor:
    payload = _pattern(case.payload_bytes, device)
    # One 64-bit generation per logical transfer catches a missing or stale
    # batch slot without the 256-iteration aliasing of a byte sentinel.
    payload.view(torch.uint64)[:: case.size // 8] = generation
    return payload


def _validate_statuses(name: str, statuses: torch.Tensor) -> None:
    values = [int(value) for value in statuses.cpu().tolist()]
    failures = [(index, value) for index, value in enumerate(values) if value != 0]
    if failures:
        raise RuntimeError(f"{name} device operations failed: {failures}")


def _validate_payload(target: torch.Tensor, expected: torch.Tensor) -> None:
    if not bool(torch.all(target[:GUARD_BYTES] == 0xA5).item()):
        raise RuntimeError("persistent PUT corrupted the target prefix guard")
    if not torch.equal(target[GUARD_BYTES:-GUARD_BYTES], expected):
        raise RuntimeError("persistent transfer payload validation failed")
    if not bool(torch.all(target[-GUARD_BYTES:] == 0xA5).item()):
        raise RuntimeError("persistent PUT corrupted the target suffix guard")


def _validate_generation_words(
    name: str,
    payload: torch.Tensor,
    case: PersistentCase,
    expected: int,
) -> None:
    observed = [
        int(value)
        for value in payload.view(torch.uint64)[:: case.size // 8].cpu().tolist()
    ]
    expected_values = [expected] * case.generation_words_total
    if observed != expected_values:
        raise RuntimeError(
            f"{name} generation words are {observed}, expected {expected_values}"
        )


def _launch_then_publish_and_synchronize(
    launch: Callable[[], None],
    publish: Callable[[], None],
    synchronize: Callable[[], None],
) -> None:
    """Keep prepared views alive until an asynchronous persistent launch exits."""

    launched = False
    try:
        launch()
        launched = True
        publish()
    finally:
        if launched:
            synchronize()


def _scope(case: PersistentCase):
    return nixl_cute.Scope.THREAD if case.scope == "thread" else nixl_cute.Scope.WARP


def _classify_mapped_preflight(
    statuses: Sequence[int] | torch.Tensor,
    *,
    rank: int,
    allow_unverified_mapped: bool,
) -> dict[str, object]:
    """Validate exact process-local mappings without requiring equal VAs."""

    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank not in range(WORLD_SIZE)
    ):
        raise ValueError("mapped-preflight rank must be 0 or 1")
    if not isinstance(allow_unverified_mapped, bool):
        raise TypeError("allow_unverified_mapped must be bool")
    raw_values = (
        statuses.tolist() if isinstance(statuses, torch.Tensor) else list(statuses)
    )
    if not isinstance(raw_values, list) or len(raw_values) != _MAPPED_DESCRIPTOR_COUNT:
        raise ValueError("mapped-preflight status vector must contain two descriptors")

    descriptors: dict[str, object] = {}
    translated_va_observed = False
    for index, (role, raw_status) in enumerate(
        zip(_MAPPED_DESCRIPTOR_ROLES, raw_values, strict=True)
    ):
        if isinstance(raw_status, bool) or not isinstance(raw_status, int):
            raise TypeError("mapped-preflight statuses must be integers")
        status = int(raw_status)
        if status == 0:
            classification = "owner_va_coincident"
        elif status == _STATUS_MISMATCH:
            # CUDA IPC virtual addresses are process-local. A translated,
            # non-null import is valid and is the base used by the hot path.
            classification = "translated_process_local_va"
            translated_va_observed = True
        elif status == _STATUS_NOT_SUPPORTED:
            raise RuntimeError(
                f"rank {rank} peer {1 - rank} {role} descriptor is not locally "
                "mapped; persistent mapped mode has no transport fallback"
            )
        else:
            raise RuntimeError(
                f"rank {rank} peer {1 - rank} {role} descriptor returned "
                f"unexpected mapped-preflight status {status}"
            )
        descriptors[role] = {
            "index": index,
            "classification": classification,
            "status": status,
        }

    return {
        "rank": rank,
        "peer_rank": 1 - rank,
        "policy": "require_non_null_process_local_pointer",
        "evidence_scope": (
            "exact persistent benchmark remote device view and registered peer "
            "payload/counter allocations"
        ),
        "remote_descriptors": descriptors,
        "all_remote_descriptors_mapped": True,
        "translated_va_observed": translated_va_observed,
        "legacy_allow_unverified_mapped_ignored": allow_unverified_mapped,
    }


def _validated_mapped_preflight_evidence(
    evidence: object, rank: int
) -> dict[str, object]:
    """Validate untrusted peer evidence before admitting mapped execution."""

    expected_keys = {
        "rank",
        "peer_rank",
        "policy",
        "evidence_scope",
        "remote_descriptors",
        "all_remote_descriptors_mapped",
        "translated_va_observed",
        "legacy_allow_unverified_mapped_ignored",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise RuntimeError(f"rank {rank} published malformed mapping evidence")
    published_rank = evidence["rank"]
    peer_rank = evidence["peer_rank"]
    if (
        isinstance(published_rank, bool)
        or not isinstance(published_rank, int)
        or published_rank != rank
        or isinstance(peer_rank, bool)
        or not isinstance(peer_rank, int)
        or peer_rank != 1 - rank
    ):
        raise RuntimeError(f"rank {rank} published mismatched mapping identities")
    if evidence["policy"] != "require_non_null_process_local_pointer" or not isinstance(
        evidence["evidence_scope"], str
    ):
        raise RuntimeError(f"rank {rank} published malformed mapping policy")
    if evidence["all_remote_descriptors_mapped"] is not True:
        raise RuntimeError(f"rank {rank} did not prove every remote descriptor")
    translated = evidence["translated_va_observed"]
    legacy = evidence["legacy_allow_unverified_mapped_ignored"]
    if not isinstance(translated, bool) or not isinstance(legacy, bool):
        raise RuntimeError(f"rank {rank} published non-boolean mapping evidence")

    descriptors = evidence["remote_descriptors"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(
        _MAPPED_DESCRIPTOR_ROLES
    ):
        raise RuntimeError(f"rank {rank} did not cover both remote descriptors")
    expected_classifications = {
        0: "owner_va_coincident",
        _STATUS_MISMATCH: "translated_process_local_va",
    }
    observed_translation = False
    for index, role in enumerate(_MAPPED_DESCRIPTOR_ROLES):
        descriptor = descriptors[role]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "index",
            "classification",
            "status",
        }:
            raise RuntimeError(f"rank {rank} published malformed {role} evidence")
        descriptor_index = descriptor["index"]
        status = descriptor["status"]
        if (
            isinstance(descriptor_index, bool)
            or not isinstance(descriptor_index, int)
            or descriptor_index != index
            or isinstance(status, bool)
            or not isinstance(status, int)
            or status not in expected_classifications
            or descriptor["classification"] != expected_classifications[status]
        ):
            raise RuntimeError(f"rank {rank} published inconsistent {role} evidence")
        observed_translation |= status == _STATUS_MISMATCH
    if translated != observed_translation:
        raise RuntimeError(f"rank {rank} published inconsistent translation evidence")
    return dict(evidence)


def _mapped_preflight_status_payload(
    rank: int,
    error: Exception | None,
    evidence: dict[str, object] | None,
) -> bytes:
    """Encode one bounded, fail-closed outcome before any persistent launch."""

    if error is None:
        try:
            evidence = _validated_mapped_preflight_evidence(evidence, rank)
        except Exception as validation_error:
            error = validation_error
    payload: dict[str, object] = {"rank": rank, "ok": error is None}
    if error is None:
        payload["evidence"] = evidence
    else:
        payload["error_type"] = type(error).__name__
        payload["message"] = str(error)[:4096]
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_mapped_preflight_exchange(
    payloads: dict[int, bytes],
) -> tuple[tuple[dict[str, object], ...], dict[str, object] | None]:
    """Validate both envelopes and combine exact-view mapping evidence."""

    expected_ranks = set(range(WORLD_SIZE))
    if set(payloads) != expected_ranks:
        raise RuntimeError("mapped-preflight exchange must cover both ranks")
    failures: list[dict[str, object]] = []
    evidence_by_rank: dict[str, object] = {}
    translated_va_observed = False
    for rank in range(WORLD_SIZE):
        encoded = payloads[rank]
        if not isinstance(encoded, bytes):
            raise RuntimeError(f"rank {rank} published a non-bytes preflight envelope")
        try:
            document = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"rank {rank} published an invalid preflight envelope"
            ) from error
        if not isinstance(document, dict):
            raise RuntimeError(f"rank {rank} published a malformed preflight envelope")
        published_rank = document.get("rank")
        ok = document.get("ok")
        if (
            isinstance(published_rank, bool)
            or not isinstance(published_rank, int)
            or published_rank != rank
            or not isinstance(ok, bool)
        ):
            raise RuntimeError(f"rank {rank} published a malformed preflight identity")
        if ok:
            if set(document) != {"rank", "ok", "evidence"}:
                raise RuntimeError(
                    f"rank {rank} published a malformed successful preflight"
                )
            evidence = _validated_mapped_preflight_evidence(document["evidence"], rank)
            evidence_by_rank[str(rank)] = evidence
            translated_va_observed |= bool(evidence["translated_va_observed"])
        else:
            if (
                set(document) != {"rank", "ok", "error_type", "message"}
                or not isinstance(document["error_type"], str)
                or not isinstance(document["message"], str)
            ):
                raise RuntimeError(
                    f"rank {rank} published a malformed preflight failure"
                )
            failures.append(document)

    if failures:
        return tuple(failures), None
    return (), {
        "policy": "require_non_null_process_local_pointer",
        "evidence_scope": (
            "both directed exact persistent benchmark device views and their "
            "payload/counter descriptors"
        ),
        "descriptor_roles": list(_MAPPED_DESCRIPTOR_ROLES),
        "by_rank": evidence_by_rank,
        "all_remote_descriptors_mapped": True,
        "translated_va_observed": translated_va_observed,
    }


def _decode_occupancy_exchange(
    payloads: dict[int, bytes]
) -> dict[str, dict[str, object]]:
    """Decode exact producer/receiver occupancy evidence from both ranks."""

    expected_roles = {0: "producer", 1: "receiver"}
    if set(payloads) != set(expected_roles):
        raise ValueError("occupancy exchange must contain producer and receiver ranks")
    decoded: dict[str, dict[str, object]] = {}
    for rank, role in expected_roles.items():
        payload = payloads[rank]
        if not isinstance(payload, bytes):
            raise TypeError("occupancy exchange payloads must be bytes")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {role} occupancy payload") from error
        if (
            not isinstance(document, dict)
            or set(document) != {"role", "occupancy"}
            or document["role"] != role
            or not isinstance(document["occupancy"], dict)
        ):
            raise ValueError(f"invalid {role} occupancy document")
        decoded[role] = dict(document["occupancy"])
    return decoded


def _compile_mapped_preflight(
    remote_lengths: tuple[int, int],
    advertised_bases: torch.Tensor,
    statuses: torch.Tensor,
    stream: torch.cuda.Stream,
):
    """Compile the setup-only proof for the exact prepared descriptor layout."""

    return nixl_cute.compile(
        _launch_mapped_preflight,
        nixl_cute.make_fake_memory_view("remote", remote_lengths),
        from_dlpack(advertised_bases).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
    )


def _compile_producer(
    case: PersistentCase,
    tensors: tuple[torch.Tensor, ...],
    stream,
):
    source, credits, statuses, timestamps = tensors
    local_lengths = (source.numel(), credits.numel() * credits.element_size())
    remote_lengths = (
        source.numel() + 2 * GUARD_BYTES,
        credits.numel() * credits.element_size(),
    )
    compiled = nixl_cute.compile(
        _launch_persistent_producer,
        nixl_cute.make_fake_memory_view("local", local_lengths),
        nixl_cute.make_fake_memory_view("remote", remote_lengths),
        from_dlpack(source).mark_layout_dynamic(),
        from_dlpack(credits).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        from_dlpack(timestamps).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        case.size,
        case.batch_size,
        case.groups,
        case.groups_per_block,
        case.channels,
        case.warmup,
        case.iterations,
        case.mode == "defer",
        case.mode == "mapped",
        case.device_timeout_ns,
        _scope(case),
        options="--keep-cubin",
    )
    return bind_and_validate_cooperative_launch(
        compiled,
        device_ordinal=source.get_device(),
        block_threads=case.threads,
        planned_ctas=case.blocks,
    )


def _compile_receiver(
    case: PersistentCase,
    tensors: tuple[torch.Tensor, ...],
    stream,
):
    target, completions, statuses = tensors
    remote_lengths = (
        case.payload_bytes,
        completions.numel() * completions.element_size(),
    )
    compiled = nixl_cute.compile(
        _launch_persistent_receiver,
        nixl_cute.make_fake_memory_view("remote", remote_lengths),
        from_dlpack(completions).mark_layout_dynamic(),
        from_dlpack(statuses).mark_layout_dynamic(),
        cuda.CUstream(stream.cuda_stream),
        case.groups,
        case.groups_per_block,
        case.channels,
        case.total_iterations,
        case.mode == "mapped",
        case.device_timeout_ns,
        _scope(case),
        options="--keep-cubin",
    )
    return bind_and_validate_cooperative_launch(
        compiled,
        device_ordinal=target.get_device(),
        block_threads=case.threads,
        planned_ctas=case.blocks,
    )


def _worker(
    rank: int,
    devices: tuple[int, int],
    directory: str,
    case: PersistentCase,
    timeout_s: float,
) -> None:
    if not _CUTE_AVAILABLE:
        raise RuntimeError("CuTe DSL is required for the persistent benchmark")

    from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    device = devices[rank]
    torch.cuda.set_device(device)
    native_atomic_preflight = (
        nixl_cute.require_peer_native_atomics(devices)
        if case.mode == "mapped"
        else None
    )
    stream = torch.cuda.Stream(device=device)
    control = FileControlPlane(directory, rank, WORLD_SIZE, timeout_s)
    mapped_advertised_bases = None
    mapped_preflight_statuses = None
    with torch.cuda.stream(stream):
        if rank == 0:
            payload = _payload_generation_pattern(case, source_generation(0), device)
            counter = torch.zeros(case.groups, dtype=torch.int64, device=payload.device)
            statuses = torch.zeros(
                case.groups * 2, dtype=torch.int32, device=payload.device
            )
            timestamps = torch.zeros(
                case.iterations * case.groups * TIMESTAMPS_PER_SAMPLE,
                # NIXL retags CuTe's signless extern result as the public
                # unsigned timer type; LLVM/NVVM erase the scalar bitcast.
                dtype=torch.uint64,
                device=payload.device,
            )
            registered = [payload, counter]
        else:
            payload = torch.full(
                (case.payload_bytes + 2 * GUARD_BYTES,),
                0xA5,
                dtype=torch.uint8,
                device=f"cuda:{device}",
            )
            counter = torch.zeros(case.groups, dtype=torch.int64, device=payload.device)
            statuses = torch.zeros(
                case.groups, dtype=torch.int32, device=payload.device
            )
            timestamps = None
            registered = [payload, counter]
        if case.mode == "mapped":
            mapped_advertised_bases = torch.empty(
                _MAPPED_DESCRIPTOR_COUNT, dtype=torch.uint64, device=payload.device
            )
            mapped_preflight_statuses = torch.empty(
                _MAPPED_DESCRIPTOR_COUNT, dtype=torch.int32, device=payload.device
            )
    stream.synchronize()
    mapped_advertised_bases_host = (
        torch.empty(_MAPPED_DESCRIPTOR_COUNT, dtype=torch.uint64, pin_memory=True)
        if case.mode == "mapped"
        else None
    )
    mapped_preflight_statuses_host = (
        torch.empty(_MAPPED_DESCRIPTOR_COUNT, dtype=torch.int32, pin_memory=True)
        if case.mode == "mapped"
        else None
    )

    run_id = Path(directory).name
    name = f"cute_persistent_{run_id}_{rank}"
    peer_name = f"cute_persistent_{run_id}_{1 - rank}"
    agent = nixl_agent(
        name,
        nixl_agent_config(
            # Direct mapped traffic never requires host progress. Keep the
            # background thread for NIXL wait/defer modes only.
            enable_prog_thread=case.mode != "mapped",
            num_threads=UCX_POST_THREADS,
            backends=[],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_DEFAULT,
        ),
    )
    agent.create_backend(
        "UCX",
        {
            "num_threads": str(UCX_POST_THREADS),
            "num_workers": str(UCX_NUM_WORKERS),
            "ucx_num_device_channels": str(case.channels),
        },
    )

    with ExitStack() as resources:
        registration = agent.register_memory(registered, backends=["UCX"])
        resources.callback(agent.deregister_memory, registration, backends=["UCX"])
        coordinates = PeerCoordinates(
            name, tuple(_region(tensor) for tensor in registered)
        )
        all_metadata = control.exchange("metadata", agent.get_agent_metadata())
        all_coordinates = control.exchange("coordinates", coordinates.to_bytes())
        peer = PeerCoordinates.from_bytes(all_coordinates[1 - rank])
        if peer.agent_name != peer_name or len(peer.regions) != 2:
            raise RuntimeError(f"rank {rank} received invalid peer coordinates")
        loaded_name = normalize_agent_name(
            agent.add_remote_agent(all_metadata[1 - rank])
        )
        if loaded_name != peer_name:
            raise RuntimeError(f"rank {rank} loaded unexpected peer {loaded_name!r}")
        resources.callback(agent.remove_remote_agent, peer_name)
        control.barrier("metadata-loaded")
        agent.make_connection(peer_name, backends=["UCX"])
        complete_ucx_setup_handshake(
            agent,
            control,
            {rank: name, 1 - rank: peer_name},
            generation=0,
            nonce=run_id,
            timeout_s=timeout_s,
        )

        mapped_preflight_failure: Exception | None = None
        mapped_preflight_evidence: dict[str, object] | None = None
        with ExitStack() as views:
            local_view = None
            remote_view = None
            preflight_error: Exception | None = None
            try:
                local_view = views.enter_context(
                    agent.prepare_device_view(registered, backend="UCX")
                )
                remote_view = views.enter_context(
                    agent.prepare_device_view(
                        [region.descriptor for region in peer.regions],
                        remote_agent=peer_name,
                        mem_type="VRAM",
                        backend="UCX",
                        connection_timeout_ms=max(1, int(timeout_s * 1000)),
                    )
                )
            except Exception as error:
                if case.mode != "mapped":
                    raise
                preflight_error = error

            if case.mode == "mapped":
                local_evidence: dict[str, object] | None = None
                try:
                    if preflight_error is None:
                        if (
                            remote_view is None
                            or mapped_advertised_bases is None
                            or mapped_preflight_statuses is None
                            or mapped_advertised_bases_host is None
                            or mapped_preflight_statuses_host is None
                        ):
                            raise AssertionError(
                                "mapped preflight is missing a setup-only resource"
                            )
                        mapped_advertised_bases_host.copy_(
                            torch.as_tensor(
                                tuple(region.address for region in peer.regions),
                                dtype=torch.uint64,
                            )
                        )
                        compiled_preflight = _compile_mapped_preflight(
                            tuple(region.length for region in peer.regions),
                            mapped_advertised_bases,
                            mapped_preflight_statuses,
                            stream,
                        )
                        with torch.cuda.stream(stream):
                            mapped_advertised_bases.copy_(
                                mapped_advertised_bases_host, non_blocking=True
                            )
                            compiled_preflight(
                                remote_view,
                                from_dlpack(
                                    mapped_advertised_bases
                                ).mark_layout_dynamic(),
                                from_dlpack(
                                    mapped_preflight_statuses
                                ).mark_layout_dynamic(),
                                cuda.CUstream(stream.cuda_stream),
                            )
                            mapped_preflight_statuses_host.copy_(
                                mapped_preflight_statuses, non_blocking=True
                            )
                except Exception as error:
                    preflight_error = error

                # This is the only post-preflight CPU drain. It covers the H2D
                # owner bases, tiny proof kernel, and pinned D2H status copy.
                # No preflight operation appears in the measured persistent loop.
                try:
                    stream.synchronize()
                except Exception as drain_error:
                    if preflight_error is None:
                        preflight_error = drain_error
                    else:
                        preflight_error = RuntimeError(
                            f"{preflight_error}; mapped-preflight stream drain also "
                            f"failed: {drain_error}"
                        )
                if preflight_error is None:
                    try:
                        assert mapped_preflight_statuses_host is not None
                        local_evidence = _classify_mapped_preflight(
                            mapped_preflight_statuses_host,
                            rank=rank,
                            allow_unverified_mapped=case.allow_unverified_mapped,
                        )
                    except Exception as error:
                        preflight_error = error
                preflight_payloads = control.exchange(
                    "mapped-preflight-status",
                    _mapped_preflight_status_payload(
                        rank, preflight_error, local_evidence
                    ),
                )
                try:
                    preflight_failures, mapped_preflight_evidence = (
                        _decode_mapped_preflight_exchange(preflight_payloads)
                    )
                except Exception as error:
                    mapped_preflight_failure = RuntimeError(
                        f"mapped exact-view preflight exchange was invalid: {error}"
                    )
                else:
                    if preflight_failures:
                        mapped_preflight_failure = RuntimeError(
                            "mapped exact-view preflight failed across ranks: "
                            f"{preflight_failures}"
                        )
                    elif mapped_preflight_evidence is None:
                        mapped_preflight_failure = AssertionError(
                            "successful mapped preflight has no combined evidence"
                        )

            if mapped_preflight_failure is None:
                if local_view is None or remote_view is None:
                    raise AssertionError(
                        "persistent launch has no prepared device view"
                    )
                if rank == 0:
                    assert timestamps is not None
                    compiled, occupancy = _compile_producer(
                        case, (payload, counter, statuses, timestamps), stream
                    )
                else:
                    compiled, occupancy = _compile_receiver(
                        case, (payload, counter, statuses), stream
                    )
                occupancy_payloads = control.exchange(
                    "compiled-occupancy",
                    json.dumps(
                        {
                            "role": "producer" if rank == 0 else "receiver",
                            "occupancy": occupancy.as_dict(),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8"),
                )
                occupancy_evidence = _decode_occupancy_exchange(occupancy_payloads)

            if mapped_preflight_failure is None and rank == 1:
                _launch_then_publish_and_synchronize(
                    lambda: compiled(
                        remote_view,
                        from_dlpack(counter).mark_layout_dynamic(),
                        from_dlpack(statuses).mark_layout_dynamic(),
                        cuda.CUstream(stream.cuda_stream),
                    ),
                    lambda: control.publish("receiver-launched"),
                    stream.synchronize,
                )
                control.await_rank("producer-done", 0)
                _validate_statuses("receiver", statuses)
                if any(
                    int(value) != case.total_iterations for value in counter.tolist()
                ):
                    raise RuntimeError(
                        "receiver completion counters have unexpected values"
                    )
                _validate_payload(
                    payload,
                    _payload_generation_pattern(
                        case,
                        source_generation(case.total_iterations - 1),
                        device,
                    ),
                )
                target_payload = payload[GUARD_BYTES:-GUARD_BYTES]
                _validate_generation_words(
                    "target",
                    target_payload,
                    case,
                    source_generation(case.total_iterations - 1),
                )
                control.publish("receiver-validated")
            elif mapped_preflight_failure is None:
                assert timestamps is not None
                control.await_rank("receiver-launched", 1)
                compiled(
                    local_view,
                    remote_view,
                    from_dlpack(payload).mark_layout_dynamic(),
                    from_dlpack(counter).mark_layout_dynamic(),
                    from_dlpack(statuses).mark_layout_dynamic(),
                    from_dlpack(timestamps).mark_layout_dynamic(),
                    cuda.CUstream(stream.cuda_stream),
                )
                stream.synchronize()
                _validate_statuses("producer", statuses)
                if any(
                    int(value) != case.total_iterations for value in counter.tolist()
                ):
                    raise RuntimeError(
                        "producer return-credit counters have unexpected values"
                    )
                _validate_generation_words(
                    "source",
                    payload,
                    case,
                    source_generation(case.total_iterations),
                )
                timestamp_values = [int(value) for value in timestamps.cpu().tolist()]
                result = summarize_timestamps(
                    case,
                    timestamp_values,
                    source_device=devices[0],
                    target_device=devices[1],
                    backend_parameters=agent.get_backend_params("UCX"),
                    cooperative_occupancy=occupancy_evidence,
                )
                result["environment"] = {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "source_gpu": torch.cuda.get_device_name(devices[0]),
                    "target_gpu": torch.cuda.get_device_name(devices[1]),
                    "native_peer_atomics": native_atomic_preflight,
                    "mapped_preflight": mapped_preflight_evidence,
                    "nixl_cute_abi": nixl_cute.NIXL_CUTE_ABI_VERSION,
                }
                control.publish("producer-done")
                control.await_rank("receiver-validated", 1)
                print("RESULT " + json.dumps(result, sort_keys=True), flush=True)

        control.barrier("views-released")
        if mapped_preflight_failure is not None:
            raise mapped_preflight_failure
    control.barrier("cleanup-complete")


def run(
    *,
    devices: tuple[int, int],
    case: PersistentCase,
    timeout_s: float,
) -> None:
    """Validate the host configuration and spawn the two persistent peers."""

    if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE:
        raise ValueError("--devices must name two distinct CUDA devices")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a positive finite number")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("this benchmark requires at least two visible CUDA GPUs")
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device indices must be integers")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    target_memory = torch.cuda.get_device_properties(devices[1]).total_memory
    if case.payload_bytes + 2 * GUARD_BYTES > target_memory:
        raise ValueError("target working set exceeds total device memory")

    import torch.multiprocessing as mp

    with TemporaryDirectory(prefix=f"nixl_cute_persistent_{os.getpid()}_") as directory:
        mp.spawn(
            _worker,
            args=(devices, directory, case, float(timeout_s)),
            nprocs=WORLD_SIZE,
            join=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--size", default="64KiB", help="one transfer size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--scope", choices=("thread", "warp"), default="warp")
    parser.add_argument("--mode", choices=("wait", "defer", "mapped"), default="defer")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--device-timeout-ms",
        type=int,
        default=0,
        help=(
            "diagnostic per-operation GPU credit bound; zero uses the timer-free "
            "production wait and relies on the external job timeout"
        ),
    )
    parser.add_argument(
        "--allow-unverified-mapped",
        action="store_true",
        help=(
            "deprecated compatibility no-op; mapped execution is qualified by "
            "non-null process-local pointers and directed native peer atomics"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="control-plane timeout; also use an external job timeout",
    )
    args = parser.parse_args()
    run(
        devices=tuple(args.devices),
        case=PersistentCase(
            size=parse_size(args.size),
            batch_size=args.batch_size,
            groups=args.groups,
            channels=args.channels,
            scope=args.scope,
            mode=args.mode,
            warmup=args.warmup,
            iterations=args.iterations,
            allow_unverified_mapped=args.allow_unverified_mapped,
            device_timeout_ns=args.device_timeout_ms * 1_000_000,
        ),
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
