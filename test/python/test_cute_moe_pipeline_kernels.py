# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_KERNEL_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "python"
    / "cute"
    / "moe"
    / "pipeline_kernels.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_cute_moe_pipeline_kernels", _KERNEL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_KERNELS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _KERNELS
_SPEC.loader.exec_module(_KERNELS)

_PROTOCOL_PATH = _KERNEL_PATH.with_name("ll_protocol.py")
_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "_cute_moe_pipeline_kernel_protocol", _PROTOCOL_PATH
)
assert _PROTOCOL_SPEC is not None and _PROTOCOL_SPEC.loader is not None
_PROTOCOL = importlib.util.module_from_spec(_PROTOCOL_SPEC)
sys.modules[_PROTOCOL_SPEC.name] = _PROTOCOL
_PROTOCOL_SPEC.loader.exec_module(_PROTOCOL)

KernelContract = _KERNELS.KernelContract
INT32_MAX = _KERNELS.INT32_MAX
UINT32_MAX = _KERNELS.UINT32_MAX
UINT64_MAX = _KERNELS.UINT64_MAX


def _call_name(call: ast.Call) -> str:
    fields: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        fields.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        fields.append(node.id)
    return ".".join(reversed(fields))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name} definition"
    return matches[0]


def _calls(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        _call_name(candidate)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
    )


def test_contract_has_dense_expert_boundary_and_raw_receive_planes():
    contract = KernelContract(4, 2, 3, 2, 16)
    shapes = contract.tensor_shapes

    assert contract.max_experts == 8
    assert contract.route_capacity == 6
    assert contract.payload_nbytes == 32
    assert shapes["activations"] == (3, 16)
    assert shapes["dispatch_recv"] == (2, 2, 4, 3, 16)
    assert shapes["dispatch_recv_src_info"] == (2, 2, 4, 3, 4)
    assert shapes["expert_input"] == (2, 2, 12, 16)
    assert shapes["expert_counts"] == (2,)
    assert shapes["route_counts"] == (8,)
    assert shapes["source_info"] == (2, 2, 12, 4)
    assert shapes["layout_range"] == (2, 2, 4)
    assert shapes["expert_output"] == (2, 12, 16)
    assert shapes["combine_stage"] == (2, 2, 12, 16)
    assert shapes["combine_recv"] == (2, 2, 3, 16)
    assert shapes["combine_output"] == (3, 16)

    # A mapping proxy prevents accidental mutation of a compiled geometry.
    with pytest.raises(TypeError):
        shapes["expert_input"] = ()


def test_contract_indices_ranges_and_sequences_are_exact():
    contract = KernelContract(4, 2, 3, 2, 8)

    assert contract.owner(7) == 3
    assert contract.local_expert(7) == 1
    assert contract.raw_receive_slot(2, 1) == 7
    assert contract.combine_slot(1, 2) == 5
    assert [contract.bank(operation) for operation in range(6)] == [0, 1, 0, 1, 0, 1]
    assert contract.wire_epoch(7, 9) == (7 << 32) | 9
    assert contract.ready_sequence(7, 9) == 5
    assert contract.previous_bank_credit(7, 0) is None
    assert contract.previous_bank_credit(7, 1) is None
    assert contract.previous_bank_credit(7, 2) == contract.ready_sequence(7, 0)

    encoded = contract.pack_layout_range(7, 3)
    assert encoded == (7 << 32) | 3
    assert contract.unpack_layout_range(encoded) == (7, 3)

    with pytest.raises(IndexError, match="global_expert"):
        contract.owner(8)
    with pytest.raises(IndexError, match="source_slot"):
        contract.raw_receive_slot(0, 3)
    with pytest.raises(ValueError, match="per-source expert capacity"):
        contract.pack_layout_range(0, 4)
    with pytest.raises(ValueError, match="encoded count"):
        contract.unpack_layout_range(4)
    with pytest.raises(ValueError, match="dense expert row capacity"):
        contract.pack_layout_range(12, 1)
    with pytest.raises(ValueError, match="encoded range"):
        contract.unpack_layout_range((12 << 32) | 1)
    assert contract.ready_sequence(UINT32_MAX, UINT32_MAX) == (UINT32_MAX // 2) + 1
    with pytest.raises(ValueError, match="operation"):
        contract.ready_sequence(0, UINT32_MAX + 1)


def test_compaction_shards_exhaustively_cover_each_bucket_without_overlap():
    contract = KernelContract(3, 2, 17, 2, 8)
    bucket_count = contract.dispatch_ctas

    # Exercise exact multiples, uneven worker grids, more workers than rows,
    # empty buckets, and full-capacity buckets.
    for worker_ctas in range(bucket_count, 4 * bucket_count + 4):
        for bucket in range(bucket_count):
            expected_ctas = tuple(range(bucket, worker_ctas, bucket_count))
            for count in range(contract.token_capacity + 1):
                shards = contract.compaction_shards(bucket, count, worker_ctas)
                assert len(shards) == len(expected_ctas)
                assert shards[0][0] == 0
                assert shards[-1][1] == count
                assert all(
                    left_end == right_begin
                    for (_, left_end), (right_begin, _) in zip(shards, shards[1:])
                )
                assert all(0 <= begin <= end <= count for begin, end in shards)
                covered = [row for begin, end in shards for row in range(begin, end)]
                assert covered == list(range(count))
                lengths = [end - begin for begin, end in shards]
                assert max(lengths) - min(lengths) <= 1

    single_bucket = KernelContract(1, 1, 17, 1, 8)
    assert single_bucket.compaction_shards(0, 1, 128) == (
        *((0, 0),) * 127,
        (0, 1),
    )
    with pytest.raises(IndexError, match="bucket"):
        contract.compaction_shards(bucket_count, 1, bucket_count)
    with pytest.raises(TypeError, match="count"):
        contract.compaction_shards(0, True, bucket_count)
    with pytest.raises(ValueError, match="count"):
        contract.compaction_shards(0, contract.token_capacity + 1, bucket_count)
    with pytest.raises(ValueError, match="worker_ctas must cover"):
        contract.compaction_shards(0, 1, bucket_count - 1)


def test_live_extent_is_independent_from_fixed_arena_capacity():
    contract = KernelContract(4, 2, 4, 2, 16)
    shapes = contract.tensor_shapes_for(2)

    # Only local inputs/output use this rank's live extent.
    assert shapes["activations"] == (2, 16)
    assert shapes["topk_indices"] == (2, 2)
    assert shapes["topk_weights"] == (2, 2)
    assert shapes["combine_output"] == (2, 16)

    # Wire and expert planes keep the shared capacity in every physical stride.
    assert shapes["dispatch_recv"] == (2, 2, 4, 4, 16)
    assert shapes["expert_input"] == (2, 2, 16, 16)
    assert shapes["combine_recv"] == (2, 2, 4, 16)

    empty = contract.tensor_shapes_for(0)
    assert empty["activations"] == (0, 16)
    assert empty["topk_indices"] == (0, 2)
    assert empty["topk_weights"] == (0, 2)
    assert empty["combine_output"] == (0, 16)
    assert empty["dispatch_recv"] == shapes["dispatch_recv"]
    assert empty["expert_input"] == shapes["expert_input"]

    with pytest.raises(TypeError, match="live_tokens"):
        contract.tensor_shapes_for(True)
    with pytest.raises(ValueError, match="live_tokens"):
        contract.tensor_shapes_for(5)


def test_cooperative_grid_contract_rejects_deadlocking_geometry():
    contract = KernelContract(4, 2, 4, 2, 16)

    assert contract.dispatch_ctas == 8
    assert (
        contract.validate_cooperative_grid(
            dispatch_resident_limit=12,
            worker_ctas=12,
            combine_resident_limit=12,
        )
        == 12
    )
    with pytest.raises(ValueError, match="dispatch resident limit"):
        contract.validate_cooperative_grid(
            dispatch_resident_limit=8,
            worker_ctas=9,
            combine_resident_limit=9,
        )
    with pytest.raises(ValueError, match="worker_ctas must cover"):
        contract.validate_cooperative_grid(
            dispatch_resident_limit=12,
            worker_ctas=7,
            combine_resident_limit=12,
        )
    with pytest.raises(ValueError, match="combine resident limit"):
        contract.validate_cooperative_grid(
            dispatch_resident_limit=9,
            worker_ctas=9,
            combine_resident_limit=8,
        )


def test_launch_span_contract_rejects_aliasing_but_allows_empty_tensors():
    KernelContract.validate_nonoverlapping_spans(
        {
            "arena": (0x1000, 0x2000),
            "topk_indices": (0x3000, 0x3100),
            "topk_weights": (0x3100, 0x3200),
            "combine_output": (0x4000, 0x4000),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        KernelContract.validate_nonoverlapping_spans(
            {
                "topk_indices": (0x3000, 0x3100),
                "combine_output": (0x3080, 0x3200),
            }
        )
    with pytest.raises(ValueError, match="precedes"):
        KernelContract.validate_nonoverlapping_spans({"arena": (2, 1)})
    with pytest.raises(ValueError, match="16-byte aligned"):
        KernelContract.validate_nonoverlapping_spans(
            {"combine_output": (0x4004, 0x4100)}
        )


def test_launch_span_contract_allows_only_exact_zero_copy_stage_alias():
    KernelContract.validate_nonoverlapping_spans(
        {
            "expert_output": (0x5000, 0x6000),
            "combine_stage": (0x5000, 0x6000),
            "combine_output": (0x7000, 0x8000),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        KernelContract.validate_nonoverlapping_spans(
            {
                "expert_output": (0x5000, 0x6000),
                "combine_stage": (0x5800, 0x6800),
            }
        )


def test_contract_matches_checked_pipeline_arena_indexing():
    contract = KernelContract(4, 2, 3, 2, 8)
    layout = _PROTOCOL.PipelineLLArenaLayout(4, 2, 3, 2, 8, 2)

    assert layout.payload_stride == contract.payload_nbytes
    assert _PROTOCOL.PIPELINE_BUCKET_STAMP_NBYTES == _KERNELS.BUCKET_STAMP_NBYTES
    assert _PROTOCOL.SOURCE_INFO_NBYTES == _KERNELS.SOURCE_INFO_NBYTES
    for bank in range(2):
        for expert in range(2):
            for source_rank in range(4):
                for source_slot in range(3):
                    raw_item = ((bank * 2 + expert) * 4 + source_rank) * 3 + source_slot
                    assert (
                        layout.dispatch_receive_record(
                            bank, expert, source_rank, source_slot
                        ).offset
                        == layout.region("dispatch_recv").offset
                        + raw_item * layout.payload_stride
                    )
                packed_begin = source_rank * 3
                assert (
                    layout.expert_input_record(bank, expert, packed_begin).offset
                    == layout.region("expert_input").offset
                    + ((bank * 2 + expert) * 12 + packed_begin) * layout.payload_stride
                )
        for route_slot in range(2):
            for token in range(3):
                assert (
                    layout.combine_receive_record(bank, route_slot, token).offset
                    == layout.region("combine_recv").offset
                    + ((bank * 2 + route_slot) * 3 + token) * layout.payload_stride
                )
    for step in range(8):
        assert (
            contract.ready_sequence(9, step)
            == _PROTOCOL.OperationEpoch(9, step).bank_sequence
        )


@pytest.mark.parametrize(
    "args,error",
    [
        ((True, 1, 1, 1, 8), TypeError),
        ((1, 0, 1, 1, 8), ValueError),
        ((1, 1, 1, 2, 8), ValueError),
        ((1, 1, 1, 1, 10), ValueError),
        ((2, 1, UINT32_MAX, 2, 8), ValueError),
        ((2, 1, INT32_MAX, 2, 8), OverflowError),
        ((33, 1, 1, 1, 8), ValueError),
        ((2, 32, 1, 33, 8), ValueError),
    ],
)
def test_contract_rejects_uncompilable_or_unsafe_geometry(args, error):
    with pytest.raises(error):
        KernelContract(*args)


def test_import_safe_metadata_is_explicit_about_qualification_boundary():
    metadata = _KERNELS.kernel_contract()

    assert metadata["physical_banks"] == 2
    assert metadata["mapped_copy"] == "coherent mapped_copy_warp_ptr"
    assert "dense expert-major prefix" in metadata["dispatch_compaction"]
    assert metadata["hot_path_host_sync"] is False
    assert metadata["hot_path_d2h_scalar"] is False
    assert metadata["hot_path_host_progress"] is False
    assert metadata["network_fallback"] is False
    assert metadata["runtime_backend_adapter"] is False
    assert metadata["performance_qualified"] is False
    assert metadata["abrupt_peer_loss"].startswith("fail-stop")
    assert "distinct" in metadata["routing_precondition"]
    assert "-1 (dropped)" in metadata["routing_precondition"]
    assert "inactive owners contribute zero" in metadata["routing_precondition"]
    assert "token_capacity" in metadata["specialization_limits"]
    assert metadata["expert_counts_dtype"] == "int32"
    assert metadata["route_counts_dtype"].startswith("uint64 current-bank")
    assert "O(N*K) route-parallel" in metadata["dispatch_compaction"]
    assert "grid-strided across worker_ctas" in metadata["dispatch_compaction"]
    assert (
        "deterministic contiguous worker-CTA shards" in metadata["dispatch_compaction"]
    )
    assert "constexpr divisors" in metadata["dispatch_compaction"]
    assert "deterministic contiguous worker-CTA shards" in metadata["combine_scatter"]
    assert "grid-wide publication boundary" in metadata["combine_scatter"]
    assert _KERNELS.STANDIN_EXPERT_BIAS_SCALE == 64
    assert _KERNELS.STANDIN_EXPERT_BIAS_SCALE == _PROTOCOL.STANDIN_EXPERT_BIAS_SCALE
    assert metadata["standin_expert_bias_scale"] == 64
    assert "correctness-only BF16" in metadata["standin_expert"]
    assert "excluded from production perf claims" in metadata["standin_expert"]
    assert "same exact worker_ctas grid" in metadata["launch_precondition"]
    assert "dispatch and combine CUBIN" in metadata["launch_precondition"]
    assert (
        "exact expert_output == current-bank combine_stage"
        in metadata["launch_precondition"]
    )
    assert "one-row dummy" in metadata["zero_live_batch"]
    assert "identical generation/operation order" in metadata["collective_order"]
    assert "unsupported" in metadata["cuda_graph_capture"]
    assert "by-value operation steps" in metadata["cuda_graph_capture"]
    assert "before every replay" in metadata["cuda_graph_capture"]
    assert "topk_indices remains immutable" in metadata["routing_precondition"]
    assert "unconditional PTX trap" in metadata["device_error_policy"]
    with pytest.raises(TypeError):
        metadata["performance_qualified"] = True


def test_source_defines_real_split_phase_cute_kernels_and_launchers():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))

    expected_kernels = {
        "_resolve_mapped_peers_kernel",
        "_mapped_dispatch_kernel",
        "_standin_expert_kernel",
        "_mapped_combine_kernel",
    }
    expected_launchers = {
        "launch_resolve_mapped_peers",
        "launch_mapped_dispatch",
        "launch_standin_expert",
        "launch_mapped_combine",
    }
    definitions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert expected_kernels <= definitions.keys()
    assert expected_launchers <= definitions.keys()
    for name in expected_kernels:
        decorators = {
            (
                _call_name(ast.Call(decorator, [], []))
                if not isinstance(decorator, ast.Call)
                else _call_name(decorator)
            )
            for decorator in definitions[name].decorator_list
        }
        assert "cute.kernel" in decorators
    for name in expected_launchers:
        assert any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "cute"
            and decorator.attr == "jit"
            for decorator in definitions[name].decorator_list
        )


def test_standin_expert_uses_amplified_global_expert_bias_for_correctness():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    expert = _function(tree, "_standin_expert_kernel")
    bias_assignments = [
        node
        for node in ast.walk(expert)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "bias"
            for target in node.targets
        )
    ]
    assert len(bias_assignments) == 1
    assert ast.unparse(bias_assignments[0].value) == (
        "cutlass.BFloat16(STANDIN_EXPERT_BIAS_SCALE * "
        "(rank * experts_per_rank + local_expert + 1))"
    )
    expert_source = ast.get_source_segment(
        _KERNEL_PATH.read_text(encoding="utf-8"), expert
    )
    assert expert_source is not None
    assert "Correctness stand-in" in expert_source


def test_current_bank_route_count_abi_is_shared_by_dispatch_and_combine():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    dispatch_args = [
        arg.arg for arg in _function(tree, "launch_mapped_dispatch").args.args
    ]
    combine_args = [
        arg.arg for arg in _function(tree, "launch_mapped_combine").args.args
    ]

    assert dispatch_args[3:6] == ["expert_counts", "route_counts", "rank_mask"]
    assert combine_args[4:7] == ["combine_output", "route_counts", "rank_mask"]
    assert dispatch_args.count("route_counts") == 1
    assert combine_args.count("route_counts") == 1


def test_dispatch_worker_ctas_abi_and_launch_geometry_are_exact():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    kernel = _function(tree, "_mapped_dispatch_kernel")
    launcher = _function(tree, "launch_mapped_dispatch")

    for function in (kernel, launcher):
        arguments = [argument.arg for argument in function.args.args]
        live_index = arguments.index("live_tokens")
        assert arguments[live_index : live_index + 3] == [
            "live_tokens",
            "worker_ctas",
            "top_k",
        ]
        worker = function.args.args[live_index + 1]
        assert worker.annotation is not None
        assert ast.unparse(worker.annotation) == "cutlass.Constexpr[int]"

    kernel_calls = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call) and _call_name(node) == "_mapped_dispatch_kernel"
    ]
    assert len(kernel_calls) == 1
    forwarded = [ast.unparse(argument) for argument in kernel_calls[0].args]
    kernel_worker_index = [argument.arg for argument in kernel.args.args].index(
        "worker_ctas"
    )
    assert forwarded[kernel_worker_index] == "worker_ctas"
    assert forwarded.count("worker_ctas") == 1

    launches = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and (_call_name(node) == "launch" or _call_name(node).endswith(".launch"))
    ]
    assert len(launches) == 1
    keywords = {keyword.arg: keyword.value for keyword in launches[0].keywords}
    assert ast.unparse(keywords["grid"]) == "[worker_ctas, 1, 1]"
    assert isinstance(keywords["cooperative"], ast.Constant)
    assert keywords["cooperative"].value is True


def test_dispatch_is_live_coherent_published_and_device_compacted():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    dispatch = _function(tree, "_mapped_dispatch_kernel")
    calls = _calls(dispatch)
    source = ast.get_source_segment(_KERNEL_PATH.read_text(encoding="utf-8"), dispatch)
    assert source is not None

    assert calls.count("nixl_cute.mapped_copy_warp_ptr") >= 3
    assert not any("readonly" in call for call in calls)
    assert "nixl_cute.store_release_system_u64" in calls
    assert "nixl_cute.wait_acquire_system_u64" in calls
    assert "nixl_cute.sync_grid" in calls
    assert calls.count("nixl_cute.atomic_add_release_gpu_u64") == 1
    assert "route_counts[cta] = cutlass.Uint64(0)" in source
    assert "route_token += cutlass.Uint32(worker_ctas)" in source
    assert "while flat_route < live_tokens * top_k" in source
    assert "flat_route += cutlass.Uint32(worker_ctas)" in source
    assert "flat_route += cutlass.Uint32(max_ranks * experts_per_rank)" not in source
    assert "for token in cutlass.range(live_tokens" not in source
    assert source.index("nixl_cute.sync_grid()") < source.index(
        "nixl_cute.atomic_add_release_gpu_u64"
    )
    assert source.index("nixl_cute.atomic_add_release_gpu_u64") < source.index(
        "copy_status = nixl_cute.mapped_copy_warp_ptr"
    )
    assert "dispatch_recv_base" in source
    assert "dispatch_recv_src_info_base" in source
    assert "expert_input_base" in source
    assert "dispatch_src_info_base" in source
    assert "packed_range = (total << 32) + count" in source
    assert source.index("dispatch_recv_base") < source.index("expert_input_base")
    assert "if incoming_count > cutlass.Uint64(token_capacity)" in source
    assert "if incoming_count > live_tokens" not in source
    assert "sanitized_layout += incoming_count" in source
    assert source.index("sanitized_layout += incoming_count") < source.index(
        "# Build a dense per-expert prefix"
    )
    assert "if selected != -1" in source
    assert "expert_counts[cta] = cutlass.Int32(total)" in source
    assert "expert_count_base" not in source
    assert "expert_counts[cta] = cutlass.Int32(0)" in source


def test_external_route_tensors_use_explicit_two_dimensional_coordinates():
    """Prevent CuTe from treating flattened offsets as layout coordinates."""

    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))

    def coordinates(function: str, tensor: str) -> list[tuple[str, str]]:
        accesses = [
            node
            for node in ast.walk(_function(tree, function))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == tensor
        ]
        assert accesses, f"expected {function} to read {tensor}"
        result: list[tuple[str, str]] = []
        for access in accesses:
            # A DLPack (N, K) tensor keeps its 2-D CuTe layout.  Passing one
            # arithmetic scalar asks idx2crd to decode a coordinate; it is not
            # equivalent to a row-major linear offset and can add div/rem.
            assert isinstance(access.slice, ast.Tuple)
            assert len(access.slice.elts) == 2
            result.append(tuple(ast.unparse(value) for value in access.slice.elts))
        return result

    assert sorted(coordinates("_mapped_dispatch_kernel", "topk_indices")) == sorted(
        [
            ("route_token", "lane"),
            ("route_token", "prior_slot"),
            ("token", "route_slot"),
        ]
    )
    assert coordinates("_mapped_combine_kernel", "topk_weights") == [
        ("token", "lane")
    ]
    assert coordinates("_mapped_combine_kernel", "topk_indices") == [
        ("token", "route_slot"),
        ("token", "route_slot"),
    ]


def test_wire_integer_loads_keep_unsigned_type_across_dynamic_branches():
    """Guard CuTe's signless tensor-load control-flow boundary.

    Uint64 and Uint32 tensor elements are represented by signless MLIR integers
    and can otherwise surface as signed values.  The values below cross dynamic
    branch merges; explicit bitcasts prevent compile-time signedness mismatches
    and preserve fail-closed upper-bound comparisons.
    """

    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))

    def assigned_values(function: str, target: str) -> list[str]:
        return [
            ast.unparse(node.value)
            for node in ast.walk(_function(tree, function))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(candidate, ast.Name) and candidate.id == target
                for candidate in node.targets
            )
        ]

    assert "cutlass.Int64(stamp[2]).bitcast(cutlass.Uint64)" in assigned_values(
        "_mapped_dispatch_kernel", "incoming_count"
    )
    incoming_stamp_bitcast = (
        "cutlass.Int64(incoming_stamp[2]).bitcast(cutlass.Uint64)"
    )
    assert incoming_stamp_bitcast in assigned_values(
        "_mapped_combine_kernel", "incoming_partial"
    )
    assert incoming_stamp_bitcast in assigned_values(
        "_mapped_combine_kernel", "published_count"
    )
    for target, field in (("origin_rank", 0), ("origin_token", 1), ("route_slot", 2)):
        assert (
            f"cutlass.Int32(source_info[{field}]).bitcast(cutlass.Uint32)"
            in assigned_values("_mapped_combine_kernel", target)
        )

    source = _KERNEL_PATH.read_text(encoding="utf-8")
    assert "incoming_count > cutlass.Uint64(token_capacity)" in source
    assert "origin_token >= cutlass.Uint32(token_capacity)" in source
    assert "route_slot >= cutlass.Uint32(top_k)" in source
    assert "published_count > cutlass.Uint64(live_tokens * top_k)" in source


def test_combine_uses_a_distinct_unsigned_route_count_cursor():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    combine = _function(tree, "_mapped_combine_kernel")
    assignments = {
        target.id: ast.unparse(node.value)
        for node in ast.walk(combine)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"bucket", "route_bucket"}
    }
    assert assignments["bucket"] == "cta % bucket_count"
    assert assignments["route_bucket"] == "cutlass.Uint32(lane)"

    source = ast.get_source_segment(
        _KERNEL_PATH.read_text(encoding="utf-8"), combine
    )
    assert source is not None
    assert "route_bucket = cutlass.Uint32(lane)" in source
    assert "while route_bucket < max_ranks * experts_per_rank" in source


def test_dispatch_larger_worker_grid_guards_bucket_indices_and_all_grid_joins():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    dispatch = _function(tree, "_mapped_dispatch_kernel")
    parents = {
        child: parent
        for parent in ast.walk(dispatch)
        for child in ast.iter_child_nodes(parent)
    }

    clears = [
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "route_counts"
            and ast.unparse(target.slice) == "cta"
            for target in node.targets
        )
    ]
    assert len(clears) == 1
    ancestor = parents.get(clears[0])
    guards: list[str] = []
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            guards.append(ast.unparse(ancestor.test))
        ancestor = parents.get(ancestor)
    assert "cta < max_ranks * experts_per_rank" in guards

    strides = {
        ast.unparse(node.target): ast.unparse(node.value)
        for node in ast.walk(dispatch)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and ast.unparse(node.target) in {"route_token", "flat_route"}
    }
    assert strides == {
        "route_token": "cutlass.Uint32(worker_ctas)",
        "flat_route": "cutlass.Uint32(worker_ctas)",
    }

    grid_syncs = [
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Call) and _call_name(node) == "nixl_cute.sync_grid"
    ]
    assert grid_syncs
    for sync in grid_syncs:
        ancestor = parents.get(sync)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                assert "cta" not in {
                    candidate.id
                    for candidate in ast.walk(ancestor.test)
                    if isinstance(candidate, ast.Name)
                }
            ancestor = parents.get(ancestor)


def test_dispatch_compaction_uses_exact_full_worker_grid_partition():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    dispatch = _function(tree, "_mapped_dispatch_kernel")
    source = ast.get_source_segment(_KERNEL_PATH.read_text(encoding="utf-8"), dispatch)
    assert source is not None
    compaction = source.split(
        "# Compact raw fixed-source segments into the dense prefix", 1
    )[1].split("# A grouped GEMM may start", 1)[0]

    expected_assignments = {
        "bucket_count": "max_ranks * experts_per_rank",
        "bucket": "cta % bucket_count",
        "shard": "cta // bucket_count",
        "shard_count": "(worker_ctas - 1 - bucket) // bucket_count + 1",
        "base_shard_count": "worker_ctas // bucket_count",
        "shard_rows": "shard_end - shard_begin",
    }
    assignments = {
        (target.id, ast.unparse(node.value))
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in expected_assignments
    }
    assert set(expected_assignments.items()) <= assignments

    boundary_divisions = [
        ast.unparse(node.value)
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"shard_begin", "shard_end"}
            for target in node.targets
        )
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.FloorDiv)
    ]
    assert len(boundary_divisions) == 6
    assert set(boundary_divisions) == {
        "count * cutlass.Uint64(shard) // cutlass.Uint64(base_shard_count)",
        "count * cutlass.Uint64(shard + 1) // cutlass.Uint64(base_shard_count)",
        "count * cutlass.Uint64(shard) // cutlass.Uint64(base_shard_count + 1)",
        "count * cutlass.Uint64(shard + 1) // cutlass.Uint64(base_shard_count + 1)",
    }

    assert "if cta < max_ranks * experts_per_rank:" not in compaction
    assert "cutlass.Uint64(shard_count)" not in compaction
    assert "cutlass.const_expr(worker_ctas % bucket_count == 0)" in compaction
    assert ") * cutlass.Uint64(token_capacity) + shard_begin" in compaction
    assert re.search(r"\)\s*\+\s*begin\s*\+\s*shard_begin", compaction)
    assert "if shard_rows > 0:" in compaction
    assert compaction.count("nixl_cute.mapped_copy_warp_ptr") == 2
    assert "shard_rows * cutlass.Uint64(payload_stride)" in compaction
    assert "shard_rows * cutlass.Uint64(SOURCE_INFO_NBYTES)" in compaction
    assert compaction.count("_record_status(status_address, source_rank") == 2


def test_combine_uses_real_expert_output_fp32_weights_and_returned_credits():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    combine = _function(tree, "_mapped_combine_kernel")
    calls = _calls(combine)
    source = ast.get_source_segment(_KERNEL_PATH.read_text(encoding="utf-8"), combine)
    assert source is not None

    assert "nixl_cute.mapped_copy_warp_ptr" in calls
    assert not any("readonly" in call for call in calls)
    assert calls.count("nixl_cute.sync_grid") >= 3
    assert calls.count("nixl_cute.store_release_system_u64") >= 3
    assert "cutlass.Float32" in source
    assert "topk_weights" in source
    assert "topk_indices" in source
    assert "if origin_token >= cutlass.Uint32(token_capacity)" in source
    assert "if origin_token >= live_tokens" not in source
    assert "cutlass.Uint64(token_capacity)" in source
    assert "while validation_token < live_tokens" not in source
    assert "while route_bucket < max_ranks * experts_per_rank" in source
    assert "route_counts_address" in source
    assert "scratch[0]" not in source
    assert "token = cutlass.Uint32(cta)" in source
    assert "while token < live_tokens" in source
    assert "token += cutlass.Uint32(worker_ctas)" in source
    assert "dispatch_credit_base" in source
    assert "combine_credit_base" in source
    assert "combine_output_address" in source
    assert _KERNELS.COMBINE_VECTOR_UNROLL == 2
    assert "combine_words[(None, vector_unroll)]" in source
    assert "full_vectors != total_vectors" in source


def test_combine_scatter_uses_exact_full_worker_grid_partition():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    combine = _function(tree, "_mapped_combine_kernel")
    source = ast.get_source_segment(_KERNEL_PATH.read_text(encoding="utf-8"), combine)
    assert source is not None
    scatter = source.split(
        "# Every worker warp owns one deterministic contiguous shard", 1
    )[1].split("# All expert CTAs have finished their mapped writes", 1)[0]

    expected_assignments = {
        "bucket_count": "max_ranks * experts_per_rank",
        "bucket": "cta % bucket_count",
        "shard": "cta // bucket_count",
        "shard_count": "(worker_ctas - 1 - bucket) // bucket_count + 1",
        "base_shard_count": "worker_ctas // bucket_count",
        "source_slot": "shard_begin",
    }
    assignments = {
        (target.id, ast.unparse(node.value))
        for node in ast.walk(combine)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in expected_assignments
    }
    assert set(expected_assignments.items()) <= assignments

    boundary_divisions = [
        ast.unparse(node.value)
        for node in ast.walk(combine)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"shard_begin", "shard_end"}
            for target in node.targets
        )
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.FloorDiv)
    ]
    assert len(boundary_divisions) == 6
    assert set(boundary_divisions) == {
        "record_count * cutlass.Uint64(shard) // " "cutlass.Uint64(base_shard_count)",
        "record_count * cutlass.Uint64(shard + 1) // "
        "cutlass.Uint64(base_shard_count)",
        "record_count * cutlass.Uint64(shard) // "
        "cutlass.Uint64(base_shard_count + 1)",
        "record_count * cutlass.Uint64(shard + 1) // "
        "cutlass.Uint64(base_shard_count + 1)",
    }

    assert "if cta < max_ranks * experts_per_rank:" not in scatter
    assert "cutlass.Uint64(shard_count)" not in scatter
    assert "cutlass.const_expr(worker_ctas % bucket_count == 0)" in scatter
    assert "while source_slot < shard_end:" in scatter
    assert "source_slot += cutlass.Uint64(1)" in scatter
    assert scatter.count("nixl_cute.mapped_copy_warp_ptr") == 1
    assert "if origin_token >= cutlass.Uint32(token_capacity)" in scatter
    assert "if route_slot >= cutlass.Uint32(top_k)" in scatter
    assert "if source_info[3] != 0" in scatter
    assert scatter.count("_record_status(") == 2
    assert "_record_status(status_address, peer, copy_status)" in scatter
    assert "int(nixl_cute.NIXL_ERR_MISMATCH)" in scatter

    parents = {
        child: parent
        for parent in ast.walk(combine)
        for child in ast.iter_child_nodes(parent)
    }
    grid_syncs = [
        node
        for node in ast.walk(combine)
        if isinstance(node, ast.Call) and _call_name(node) == "nixl_cute.sync_grid"
    ]
    assert grid_syncs
    for sync in grid_syncs:
        ancestor = parents.get(sync)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                assert "cta" not in {
                    candidate.id
                    for candidate in ast.walk(ancestor.test)
                    if isinstance(candidate, ast.Name)
                }
            ancestor = parents.get(ancestor)

    publication = source.index("# All expert CTAs have finished their mapped writes")
    publication_join = source.index("nixl_cute.sync_grid()", publication)
    publication_release = source.index(
        "nixl_cute.store_release_system_u64", publication_join
    )
    assert source.rindex("nixl_cute.mapped_copy_warp_ptr", 0, publication) < publication
    assert publication < publication_join < publication_release


def test_all_large_arena_byte_offsets_are_widened_before_multiply():
    source = _KERNEL_PATH.read_text(encoding="utf-8")

    for index in ("destination_item", "raw_item", "dense_item", "combine_item"):
        assert re.search(
            rf"{index}\s*\*\s*cutlass\.Uint64\(\s*payload_stride\s*\)",
            source,
        )
    assert re.search(r"output_item\s*\*\s*cutlass\.Uint64\(\s*hidden_size\s*\)", source)
    assert re.search(
        r"cutlass\.Uint64\(\s*token\s*\)\s*\*\s*"
        r"cutlass\.Uint64\(\s*hidden_size\s*\)",
        source,
    )


def test_terminal_error_is_an_always_emitted_device_trap_after_publication():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    trap = _function(tree, "_device_trap")
    combine = _function(tree, "_mapped_combine_kernel")
    trap_source = ast.get_source_segment(_KERNEL_PATH.read_text(encoding="utf-8"), trap)
    combine_source = ast.get_source_segment(
        _KERNEL_PATH.read_text(encoding="utf-8"), combine
    )
    assert trap_source is not None and combine_source is not None

    assert "llvm.inline_asm" in _calls(trap)
    assert '"trap;"' in trap_source
    assert "cute_testing" not in _KERNEL_PATH.read_text(encoding="utf-8")
    assert "_device_trap" in _calls(combine)
    assert combine_source.rindex("nixl_cute.sync_grid()") < combine_source.rindex(
        "_device_trap()"
    )


def test_waits_are_one_poller_per_converged_warp_and_stamp_abi_is_exact():
    source = _KERNEL_PATH.read_text(encoding="utf-8")

    assert "scope=nixl_cute.Scope.WARP" in source
    assert "scope=nixl_cute.Scope.THREAD" not in source
    assert "stamp[2] = record_count" in source
    assert "stamp[2] = total" in source
    assert source.count("stamp[3] = failed") == 2
    assert "record_count + 1" not in source


def test_combine_credit_returns_to_the_origin_rank_slot_exactly_once():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    combine = _function(tree, "_mapped_combine_kernel")
    assignments = [
        node
        for node in ast.walk(combine)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "credit_offset"
            for target in node.targets
        )
    ]
    assert len(assignments) == 2
    returned_credit = max(assignments, key=lambda node: node.lineno)
    index_names = [
        node.id
        for node in ast.walk(returned_credit.value)
        if isinstance(node, ast.Name)
    ]
    assert index_names.count("rank") == 1
    assert "peer" not in index_names


def test_hot_launchers_have_no_cpu_sync_progress_or_scalar_materialization():
    tree = ast.parse(_KERNEL_PATH.read_text(encoding="utf-8"))
    forbidden_suffixes = {
        ".synchronize",
        ".item",
        ".tolist",
        ".cpu",
        ".numpy",
        ".wait",
        ".poll",
    }
    forbidden_exact = {
        "torch.cuda.synchronize",
        "cuda.cuStreamSynchronize",
        "nixl_cute.put",
        "nixl_cute.put_post",
        "nixl_cute.atomic_add",
        "nixl_cute.atomic_add_post",
    }

    for name in (
        "launch_mapped_dispatch",
        "launch_standin_expert",
        "launch_mapped_combine",
    ):
        calls = _calls(_function(tree, name))
        assert not forbidden_exact.intersection(calls)
        assert not any(
            call.endswith(suffix) for call in calls for suffix in forbidden_suffixes
        )
        assert any(call == "launch" or call.endswith(".launch") for call in calls)

    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imports


def test_unavailable_runtime_raises_without_importing_accelerator_stack():
    repo = Path(__file__).resolve().parents[2]
    example_root = repo / "examples" / "python" / "cute"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(example_root)
    script = textwrap.dedent(
        """
        import sys

        assert "site" not in sys.modules
        import moe.pipeline_kernels as kernels

        accelerator_roots = {"cuda", "cutlass", "nixl", "torch"}

        def imported_accelerators():
            return {
                name.partition(".")[0]
                for name in sys.modules
                if name.partition(".")[0] in accelerator_roots
            }

        assert not kernels.is_cute_available()
        assert imported_accelerators() == set()
        for name in (
            "launch_resolve_mapped_peers",
            "launch_mapped_dispatch",
            "launch_standin_expert",
            "launch_mapped_combine",
        ):
            try:
                getattr(kernels, name)()
            except RuntimeError as error:
                assert "CuTe DSL" in str(error)
            else:
                raise AssertionError(f"{name} unexpectedly accepted a device launch")
        assert imported_accelerators() == set()
        print("unavailable-runtime-launchers=PASS")
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"child stdout:\n{result.stdout}\nchild stderr:\n{result.stderr}"
    assert result.stdout.strip() == "unavailable-runtime-launchers=PASS"
