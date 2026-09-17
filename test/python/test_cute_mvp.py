# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the tutorial model; GPU integration is exercised by torchrun."""

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).parents[2] / "examples/python/cute"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"cute_mvp_{name}", EXAMPLES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moe, build = load("moe"), load("build")


@pytest.mark.parametrize("text", ["", "0,0", "-1", "3", "0;;1"])
def test_invalid_membership(text):
    with pytest.raises(ValueError):
        moe.membership_plan(text, 3)


def test_sparse_expansion_contraction_rejoin():
    assert moe.membership_plan("0,1;0,1,2;0,2;0,1,2", 3) == (
        (0, 1),
        (0, 1, 2),
        (0, 2),
        (0, 1, 2),
    )


@pytest.mark.parametrize("active", [(0,), (0, 2), (0, 1, 2)])
def test_routes_keep_stable_expert_ids_and_bounded_slabs(active):
    _, weights, records = moe.route(0, active, 3, 0)
    assert sum(map(len, records)) == moe.TOKENS * moe.TOP_K
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(moe.TOKENS))
    seen = set()
    for owner, bucket in enumerate(records):
        assert len(bucket) <= moe.TOKENS * moe.TOP_K
        for token, slot, expert in bucket:
            assert owner in active and expert // moe.EXPERTS_PER_RANK == owner
            assert (token, expert) not in seen
            assert 0 <= token < moe.TOKENS and 0 <= slot < moe.TOP_K
            seen.add((token, expert))


def test_inactive_source_has_no_routes_and_zero_output():
    x, weights, records = moe.route(1, (0, 2), 3, 1)
    assert not any(records)
    assert moe.golden(x, weights, records).count_nonzero() == 0


def test_weighted_combine_keeps_token_and_route_correspondence():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    # Deliberately permute rows: source records, not receive position, own the token.
    records = [[(1, 0, 0), (0, 1, 1)], [(0, 0, 2), (1, 1, 3)]]
    expected = torch.tensor([[2.25, 4.5], [7.5, 10.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        moe.golden(x, weights, records), expected, rtol=0, atol=0
    )


def test_clean_ir_removes_only_nvvm_incompatible_metadata():
    source = (
        "define i32 @f() align 16 {\n ret i32 0\n}\n"
        "!llvm.module.flags = !{!0, !1}\n"
        '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}\n'
        '!1 = !{i32 1, !"keep-this", i32 7}\n'
    )
    cleaned = build.clean_ir(source)
    assert "nvvm-reflect-ftz" not in cleaned and "align 16" not in cleaned
    assert "!llvm.module.flags = !{!1}" in cleaned and '"keep-this"' in cleaned
