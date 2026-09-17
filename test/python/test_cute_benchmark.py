# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.python.cute.benchmark_put import (
    PAYLOAD_GUARD_BYTES,
    BenchmarkCase,
    _validate_target_payload,
    case_payload_nonce,
    case_target_poison,
    make_cases,
    parse_size,
    parse_sizes,
    percentile,
    summarize_case,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("64", 64),
        ("4 KiB", 4096),
        ("14KiB", 14336),
        ("1MB", 1_000_000),
        ("2MiB", 2 << 20),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_parse_sizes_rejects_ambiguous_input():
    assert parse_sizes("64B,4KiB,1MiB") == (64, 4096, 1 << 20)
    with pytest.raises(ValueError, match="invalid"):
        parse_size("1.5MiB")
    with pytest.raises(ValueError, match="duplicates"):
        parse_sizes("1024,1KiB")
    with pytest.raises(ValueError, match="non-empty"):
        parse_sizes("64,")


def test_case_matrix_is_size_major_and_explicit():
    cases = make_cases((64, 4096), ("thread", "warp"), ("put", "put-signal"))

    assert len(cases) == 8
    assert cases[0] == BenchmarkCase(64, "thread", False)
    assert cases[1] == BenchmarkCase(64, "thread", True)
    assert cases[-1] == BenchmarkCase(4096, "warp", True)
    with pytest.raises(ValueError, match="unsupported"):
        make_cases((64,), ("thread",), ("get",))


def test_percentiles_and_bandwidth_summary_preserve_raw_samples():
    samples = [4.0, 1.0, 3.0, 2.0]
    assert percentile(samples, 0.5) == pytest.approx(2.5)
    assert percentile(samples, 0.9) == pytest.approx(3.7)

    result = summarize_case(
        BenchmarkCase(4096, "warp", True),
        samples,
        warmup=7,
        source_device=0,
        target_device=1,
        final_payload_nonce=7,
    )
    assert result["samples_us"] == samples
    assert result["p50_us"] == pytest.approx(2.5)
    assert result["logical_payload_GBps_at_p50"] == pytest.approx(
        4096 / (2.6 * 1e3), rel=0.05
    )
    assert result["mode"] == "put-signal"
    assert result["correctness"] == "PASS"
    assert "launch-to-completion" in result["timing"]
    assert "Python dispatch" in result["timing"]
    assert result["validation"]["final_payload_nonce"] == 7
    assert result["validation"]["prefix_suffix_guard_bytes"] == 64


def test_final_payload_nonces_are_distinct_and_bounded():
    assert [case_payload_nonce(index) for index in range(3)] == [1, 2, 3]
    assert [case_target_poison(nonce) for nonce in (1, 2, 255)] == [254, 253, 0]
    assert all(case_target_poison(nonce) != nonce for nonce in range(1, 256))
    with pytest.raises(ValueError, match="remain distinct"):
        case_payload_nonce(255)


def test_one_byte_case_cannot_pass_from_poison_or_cross_guard_writes():
    nonce = case_payload_nonce(0)
    poison = case_target_poison(nonce)
    expected_base = torch.arange(4, dtype=torch.uint8)
    target = torch.full(
        (2 * PAYLOAD_GUARD_BYTES + expected_base.numel(),),
        poison,
        dtype=torch.uint8,
    )

    with pytest.raises(RuntimeError, match="final nonce payload"):
        _validate_target_payload(
            target,
            expected_base,
            size=1,
            nonce=nonce,
            poison=poison,
            case_index=0,
        )

    target[PAYLOAD_GUARD_BYTES] = nonce
    _validate_target_payload(
        target,
        expected_base,
        size=1,
        nonce=nonce,
        poison=poison,
        case_index=0,
    )
    target[-1] ^= 1
    with pytest.raises(RuntimeError, match="suffix guard"):
        _validate_target_payload(
            target,
            expected_base,
            size=1,
            nonce=nonce,
            poison=poison,
            case_index=0,
        )


def test_invalid_case_and_sample_inputs_fail_closed():
    with pytest.raises(ValueError, match="scope"):
        BenchmarkCase(64, "block", False)
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="finite"):
        percentile([1.0, float("nan")], 0.5)
