#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure two-GPU NIXL PUT kernels launched from CuTe DSL.

This intentionally small benchmark reports per-operation CUDA-event samples
for THREAD and WARP scope, with and without a same-channel completion atomic.
It validates every case and keeps compilation, metadata exchange, connection
setup, and view preparation outside timed samples.  Use NIXLBench's native
Device API mode as the transport baseline; this script is the CuTe-side peer,
not a replacement for NIXLBench.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import torch

try:
    from ._runtime import (
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_put_then_signal_host,
        normalize_agent_name,
        wait_for_value,
        wait_until,
    )
except ImportError:
    from _runtime import (  # type: ignore[no-redef]
        DeviceRegion,
        FileControlPlane,
        PeerCoordinates,
        check_status,
        launch_put_then_signal_host,
        normalize_agent_name,
        wait_for_value,
        wait_until,
    )

_SIZE = re.compile(r"\s*([0-9]+)\s*([KMGT]?I?B)?\s*", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "TB": 1_000_000_000_000,
    "KIB": 1 << 10,
    "MIB": 1 << 20,
    "GIB": 1 << 30,
    "TIB": 1 << 40,
}
_UINT64_MAX = (1 << 64) - 1
WORLD_SIZE = 2
CHANNEL = 0
PAYLOAD_GUARD_BYTES = 64
_MAX_DISTINCT_CASES = 255


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    size: int
    scope: str
    with_signal: bool

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError("benchmark size must be an integer")
        if not 0 < self.size <= _UINT64_MAX:
            raise ValueError("benchmark size must be in [1, UINT64_MAX]")
        if self.scope not in {"thread", "warp"}:
            raise ValueError("scope must be 'thread' or 'warp'")
        if not isinstance(self.with_signal, bool):
            raise TypeError("with_signal must be bool")

    @property
    def mode(self) -> str:
        return "put-signal" if self.with_signal else "put"


def parse_size(value: str) -> int:
    """Parse exact bytes or SI/IEC suffixes such as ``64KiB`` and ``1MB``."""

    if not isinstance(value, str):
        raise TypeError("size must be a string")
    match = _SIZE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid transfer size {value!r}")
    count = int(match.group(1))
    suffix = (match.group(2) or "").upper()
    size = count * _SIZE_MULTIPLIERS[suffix]
    if not 0 < size <= _UINT64_MAX:
        raise ValueError(f"transfer size {value!r} is outside uint64")
    return size


def parse_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, duplicate-free transfer-size sweep."""

    if not isinstance(value, str):
        raise TypeError("sizes must be a string")
    fields = value.split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("sizes must be a comma-separated non-empty list")
    sizes = tuple(parse_size(field) for field in fields)
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must not contain duplicates")
    return sizes


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("samples must be finite and non-negative")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_case(
    case: BenchmarkCase,
    samples_us: Sequence[float],
    *,
    warmup: int,
    source_device: int,
    target_device: int,
    final_payload_nonce: int,
) -> dict[str, object]:
    """Build one raw, machine-readable benchmark result."""

    p50 = percentile(samples_us, 0.50)
    p90 = percentile(samples_us, 0.90)
    p99 = percentile(samples_us, 0.99)
    return {
        "case": "nixl_cute_put",
        "size_bytes": case.size,
        "put_payload_bytes": case.size,
        "mode": case.mode,
        "scope": case.scope,
        "channel": CHANNEL,
        "source_device": source_device,
        "target_device": target_device,
        "preflight_compiles": 1,
        "warmup": warmup,
        "iterations": len(samples_us),
        "samples_us": list(samples_us),
        "p50_us": p50,
        "p90_us": p90,
        "p99_us": p99,
        "max_us": max(samples_us),
        "logical_payload_GBps_at_p50": case.size / (p50 * 1e3),
        "timing": (
            "CUDA events around queued launch-to-completion; Python dispatch "
            "idle may be included; status reset excluded"
        ),
        "validation": {
            "final_payload_nonce": final_payload_nonce,
            "target_poison_byte": case_target_poison(final_payload_nonce),
            "fresh_payload_per_case": True,
            "target_poisoned_per_case": True,
            "prefix_suffix_guard_bytes": PAYLOAD_GUARD_BYTES,
            "unwritten_tail_checked": True,
        },
        "correctness": "PASS",
    }


def make_cases(
    sizes: Sequence[int], scopes: Sequence[str], modes: Sequence[str]
) -> tuple[BenchmarkCase, ...]:
    """Return a deterministic size-major benchmark matrix."""

    if not sizes or not scopes or not modes:
        raise ValueError("sizes, scopes, and modes must all be non-empty")
    normalized_scopes = tuple(scope.lower() for scope in scopes)
    normalized_modes = tuple(mode.lower() for mode in modes)
    unknown_modes = set(normalized_modes) - {"put", "put-signal"}
    if unknown_modes:
        raise ValueError(f"unsupported benchmark modes: {sorted(unknown_modes)}")
    if len(set(normalized_scopes)) != len(normalized_scopes):
        raise ValueError("scopes must not contain duplicates")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("modes must not contain duplicates")
    return tuple(
        BenchmarkCase(size, scope, mode == "put-signal")
        for size in sizes
        for scope in normalized_scopes
        for mode in normalized_modes
    )


def _region(tensor: torch.Tensor) -> DeviceRegion:
    return DeviceRegion(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        tensor.get_device(),
    )


def _wait_for_notification(agent, peer_name: str, timeout_s: float) -> None:
    def received() -> bool:
        return b"connected" in agent.get_new_notifs().get(peer_name, ())

    wait_until(
        received,
        timeout_s=timeout_s,
        description=f"notification from {peer_name!r}",
        poll_interval_s=0.02,
    )


def _make_pattern(size: int, device: int) -> torch.Tensor:
    base = torch.arange(251, dtype=torch.uint8, device=f"cuda:{device}")
    return base.repeat((size + base.numel() - 1) // base.numel())[:size].clone()


def case_payload_nonce(case_index: int) -> int:
    """Return a non-zero byte nonce unique within a qualified run."""

    if isinstance(case_index, bool) or not isinstance(case_index, int):
        raise TypeError("case index must be an integer")
    if not 0 <= case_index < _MAX_DISTINCT_CASES:
        raise ValueError(
            f"case index must be in [0, {_MAX_DISTINCT_CASES - 1}] so final "
            "payloads remain distinct"
        )
    return case_index + 1


def case_target_poison(final_payload_nonce: int) -> int:
    """Choose a byte guaranteed to differ from the final pattern's byte zero."""

    if not 1 <= final_payload_nonce <= 0xFF:
        raise ValueError("final payload nonce must be in [1, 255]")
    # _make_pattern()[0] is zero, so the final transfer writes the nonce to
    # byte zero. Its complement cannot accidentally validate a skipped PUT,
    # even for the smallest accepted one-byte case.
    return final_payload_nonce ^ 0xFF


def _write_pattern(
    payload: torch.Tensor,
    base_pattern: torch.Tensor,
    size: int,
    nonce: int,
    stream: torch.cuda.Stream,
) -> None:
    """Queue a deterministic nonce-bearing source payload outside timing."""

    with torch.cuda.stream(stream):
        torch.bitwise_xor(base_pattern[:size], nonce, out=payload[:size])


def _validate_target_payload(
    target: torch.Tensor,
    expected_base: torch.Tensor,
    *,
    size: int,
    nonce: int,
    poison: int,
    case_index: int,
) -> None:
    """Validate fresh contents, exact transfer extent, and both guards."""

    prefix = target[:PAYLOAD_GUARD_BYTES]
    body = target[PAYLOAD_GUARD_BYTES:-PAYLOAD_GUARD_BYTES]
    suffix = target[-PAYLOAD_GUARD_BYTES:]
    expected = torch.bitwise_xor(expected_base[:size], nonce)
    if not torch.equal(body[:size], expected):
        raise RuntimeError(
            f"case {case_index} final nonce payload validation failed for "
            f"{size} bytes"
        )
    if not bool(torch.all(prefix == poison).item()):
        raise RuntimeError(f"case {case_index} corrupted the prefix guard")
    if not bool(torch.all(suffix == poison).item()):
        raise RuntimeError(f"case {case_index} corrupted the suffix guard")
    if not bool(torch.all(body[size:] == poison).item()):
        raise RuntimeError(f"case {case_index} wrote beyond its requested size")


def _run_producer_case(
    case: BenchmarkCase,
    *,
    payload: torch.Tensor,
    base_pattern: torch.Tensor,
    final_payload_nonce: int,
    local_view,
    remote_view,
    statuses: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Compile once, warm up, then retain every CUDA-event sample."""

    def launch() -> None:
        launch_put_then_signal_host(
            local_view,
            remote_view,
            statuses,
            stream,
            size=case.size,
            remote_index=0,
            remote_offset=PAYLOAD_GUARD_BYTES,
            signal_index=1,
            channel=CHANNEL,
            with_signal=case.with_signal,
            scope=case.scope,
            reset_statuses=False,
        )

    def reset_statuses() -> None:
        with torch.cuda.stream(stream):
            statuses.fill_(-1)

    # The preflight is mandatory even when --warmup=0. It both compiles the
    # specialization and qualifies its status outside the measured sample.
    nonfinal_nonce = final_payload_nonce ^ 0xFF
    reset_statuses()
    _write_pattern(payload, base_pattern, case.size, nonfinal_nonce, stream)
    launch()
    stream.synchronize()
    check_status(int(statuses[0].item()), f"{case.scope} {case.mode} preflight PUT")
    if case.with_signal:
        check_status(int(statuses[1].item()), "preflight completion atomic")

    for _ in range(warmup):
        reset_statuses()
        _write_pattern(payload, base_pattern, case.size, nonfinal_nonce, stream)
        launch()
        stream.synchronize()
        check_status(int(statuses[0].item()), f"{case.scope} {case.mode} warmup PUT")
        if case.with_signal:
            check_status(int(statuses[1].item()), "warmup completion atomic")

    samples: list[float] = []
    for iteration in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        reset_statuses()
        # Only the last measured launch writes the case-specific final nonce.
        # That makes validation distinguish timed work from preflight/warmup.
        nonce = final_payload_nonce if iteration == iterations - 1 else nonfinal_nonce
        _write_pattern(payload, base_pattern, case.size, nonce, stream)
        start.record(stream)
        launch()
        stop.record(stream)
        stop.synchronize()
        check_status(int(statuses[0].item()), f"{case.scope} {case.mode} timed PUT")
        if case.with_signal:
            check_status(int(statuses[1].item()), "timed completion atomic")
        samples.append(float(start.elapsed_time(stop)) * 1e3)
    return samples


def _worker(
    rank: int,
    devices: tuple[int, int],
    directory: str,
    cases: tuple[BenchmarkCase, ...],
    warmup: int,
    iterations: int,
    timeout_s: float,
) -> None:
    from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    device = devices[rank]
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    control = FileControlPlane(directory, rank, WORLD_SIZE, timeout_s)
    max_size = max(case.size for case in cases)
    with torch.cuda.stream(stream):
        if rank == 0:
            base_pattern = _make_pattern(max_size, device)
            payload = base_pattern.clone()
            signal = None
            statuses = torch.empty(2, dtype=torch.int32, device=f"cuda:{device}")
            registered = [payload]
        else:
            base_pattern = _make_pattern(max_size, device)
            payload = torch.empty(
                (max_size + 2 * PAYLOAD_GUARD_BYTES,),
                dtype=torch.uint8,
                device=f"cuda:{device}",
            )
            signal = torch.zeros(1, dtype=torch.int64, device=f"cuda:{device}")
            statuses = None
            registered = [payload, signal]
    stream.synchronize()

    run_id = Path(directory).name
    name = f"cute_bench_{run_id}_{rank}"
    peer_name = f"cute_bench_{run_id}_{1 - rank}"
    agent = nixl_agent(
        name,
        nixl_agent_config(
            enable_prog_thread=True,
            backends=["UCX"],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        ),
    )
    with ExitStack() as resources:
        registration = agent.register_memory(registered, backends=["UCX"])
        resources.callback(agent.deregister_memory, registration, backends=["UCX"])
        coordinates = PeerCoordinates(
            name, tuple(_region(tensor) for tensor in registered)
        )
        all_metadata = control.exchange("metadata", agent.get_agent_metadata())
        all_coordinates = control.exchange("coordinates", coordinates.to_bytes())
        peer_coordinates = PeerCoordinates.from_bytes(all_coordinates[1 - rank])
        if peer_coordinates.agent_name != peer_name:
            raise RuntimeError(
                f"rank {rank} expected {peer_name!r}, got "
                f"{peer_coordinates.agent_name!r}"
            )
        loaded_name = normalize_agent_name(
            agent.add_remote_agent(all_metadata[1 - rank])
        )
        if loaded_name != peer_name:
            raise RuntimeError(f"loaded unexpected peer metadata {loaded_name!r}")
        resources.callback(agent.remove_remote_agent, peer_name)
        control.barrier("metadata-loaded")
        agent.make_connection(peer_name, backends=["UCX"])
        agent.send_notif(peer_name, b"connected", backend="UCX")
        _wait_for_notification(agent, peer_name, timeout_s)
        control.barrier("connected")

        signal_expected = 0
        results: list[dict[str, object]] = []
        if rank == 0:
            if statuses is None or len(peer_coordinates.regions) != 2:
                raise RuntimeError("target did not publish payload and signal regions")
            with ExitStack() as views:
                local_view = views.enter_context(
                    agent.prepare_device_view(registered, backend="UCX")
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
                for case_index, case in enumerate(cases):
                    final_payload_nonce = case_payload_nonce(case_index)
                    control.barrier(f"case{case_index}.start")
                    samples = _run_producer_case(
                        case,
                        payload=payload,
                        base_pattern=base_pattern,
                        final_payload_nonce=final_payload_nonce,
                        local_view=local_view,
                        remote_view=remote_view,
                        statuses=statuses,
                        stream=stream,
                        warmup=warmup,
                        iterations=iterations,
                    )
                    control.publish(f"case{case_index}.producer-done")
                    control.await_rank(f"case{case_index}.validated", 1)
                    results.append(
                        summarize_case(
                            case,
                            samples,
                            warmup=warmup,
                            source_device=devices[0],
                            target_device=devices[1],
                            final_payload_nonce=final_payload_nonce,
                        )
                    )
            control.publish("views-released")
            control.await_rank("target-finished", 1)
        else:
            if signal is None:
                raise AssertionError("target completion counter is missing")
            expected_pattern = _make_pattern(max_size, device)
            for case_index, case in enumerate(cases):
                final_payload_nonce = case_payload_nonce(case_index)
                target_poison = case_target_poison(final_payload_nonce)
                with torch.cuda.stream(stream):
                    payload.fill_(target_poison)
                stream.synchronize()
                control.barrier(f"case{case_index}.start")
                control.await_rank(f"case{case_index}.producer-done", 0)
                if case.with_signal:
                    signal_expected += 1 + warmup + iterations
                    wait_for_value(
                        lambda: int(signal.item()),
                        signal_expected,
                        timeout_s=timeout_s,
                        description=(
                            f"case {case_index} completion counter to become "
                            f"{signal_expected}"
                        ),
                        poll_interval_s=0.005,
                    )
                torch.cuda.synchronize(device)
                _validate_target_payload(
                    payload,
                    expected_pattern,
                    size=case.size,
                    nonce=final_payload_nonce,
                    poison=target_poison,
                    case_index=case_index,
                )
                control.publish(f"case{case_index}.validated")
            control.publish("target-finished")
            control.await_rank("views-released", 0)

        control.barrier("safe-to-teardown")
    control.barrier("cleanup-complete")
    if rank == 0:
        for result in results:
            print("RESULT " + json.dumps(result, sort_keys=True), flush=True)


def run(
    *,
    devices: tuple[int, int],
    cases: tuple[BenchmarkCase, ...],
    warmup: int,
    iterations: int,
    timeout_s: float,
) -> None:
    """Validate the requested matrix and spawn one process per local GPU."""

    if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE:
        raise ValueError("--devices must name two distinct CUDA devices")
    if not cases:
        raise ValueError("at least one benchmark case is required")
    if len(cases) > _MAX_DISTINCT_CASES:
        raise ValueError(
            f"at most {_MAX_DISTINCT_CASES} cases are supported per qualified run"
        )
    for name, value in (("warmup", warmup), ("iterations", iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if iterations == 0:
        raise ValueError("iterations must be positive")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a positive finite number")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this benchmark requires at least two visible CUDA GPUs")
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("CUDA device indices must be integers")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")

    import torch.multiprocessing as mp

    with TemporaryDirectory(prefix=f"nixl_cute_bench_{os.getpid()}_") as directory:
        mp.spawn(
            _worker,
            args=(devices, directory, cases, warmup, iterations, timeout_s),
            nprocs=WORLD_SIZE,
            join=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, default=(0, 1))
    parser.add_argument(
        "--sizes",
        default="64B,4KiB,14KiB,64KiB,1MiB",
        help="comma-separated bytes or SI/IEC sizes",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("thread", "warp"),
        default=("thread", "warp"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("put", "put-signal"),
        default=("put", "put-signal"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    run(
        devices=tuple(args.devices),
        cases=make_cases(parse_sizes(args.sizes), args.scopes, args.modes),
        warmup=args.warmup,
        iterations=args.iterations,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
