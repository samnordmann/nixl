# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe contract tests for the all-mapped pipeline backend."""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import struct
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.python.cute import elastic_moe_pipeline as elastic_pipeline
from examples.python.cute import elastic_moe_pipeline_run as runner
from examples.python.cute._runtime import FileControlPlane
from examples.python.cute.moe import mapped_pipeline_backend as mapped
from examples.python.cute.moe.ll_protocol import (
    OperationEpoch,
    PipelineLLArenaLayout,
    StableSparseTopology,
)
from examples.python.cute.moe.pipeline import PipelineBankBuffers, PipelineError
from examples.python.cute.moe.pipeline_kernels import KernelContract


class FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        address,
        *,
        element_size,
        device=0,
        values=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._address = address
        self._element_size = element_size
        self._device = device
        self.device = f"cuda:{device}"
        self.is_cuda = True
        self.recorded_streams = []
        self._values = values

    def data_ptr(self):
        return self._address

    def numel(self):
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def element_size(self):
        return self._element_size

    def get_device(self):
        return self._device

    def is_contiguous(self):
        return True

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        if self._values is not None:
            return list(self._values)
        return [0] * self.numel()


class MetadataGuardTensor(FakeTensor):
    """Fake tensor that can prove a hot path never touches tensor metadata."""

    _GUARDED_ATTRIBUTES = frozenset(
        {
            "shape",
            "dtype",
            "device",
            "is_cuda",
            "data_ptr",
            "numel",
            "element_size",
            "get_device",
            "is_contiguous",
            "record_stream",
            "untyped_storage",
        }
    )

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_reject_metadata", False)
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        if name in type(self)._GUARDED_ATTRIBUTES and object.__getattribute__(
            self, "_reject_metadata"
        ):
            raise AssertionError(f"hot path queried sealed tensor attribute {name}")
        return super().__getattribute__(name)

    def reject_metadata_reads(self):
        object.__setattr__(self, "_reject_metadata", True)


class FakeDynamicTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.address = tensor.data_ptr()

    def mark_layout_dynamic(self):
        return self


class FakeEvent:
    def __init__(self):
        self.recorded = []
        self.synchronize_calls = 0

    def record(self, stream):
        self.recorded.append(stream)

    def synchronize(self):
        self.synchronize_calls += 1


class FakeStream:
    def __init__(self, *, handle=123, device=0):
        self.cuda_stream = handle
        self.device = SimpleNamespace(type="cuda", index=device)
        self.waited = []
        self.synchronize_calls = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronize_calls += 1


class FakeLauncher:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


@dataclass(frozen=True)
class FakeDTypes:
    uint8: str = "uint8"
    int32: str = "int32"
    int64: str = "int64"
    uint64: str = "uint64"
    float32: str = "float32"
    bfloat16: str = "bfloat16"
    cuda: object | None = None


def _topology(*, generation=0, active=(0, 1), incarnations=(1, 1)):
    return StableSparseTopology(
        max_ranks=2,
        experts_per_rank=1,
        active_ranks=active,
        membership_generation=generation,
        rank_incarnations=incarnations,
    )


def _layout():
    return PipelineLLArenaLayout(
        max_ranks=2,
        experts_per_rank=1,
        num_tokens=4,
        top_k=2,
        hidden_size=8,
        element_size=2,
    )


def _tensor(shape, dtype, address, element_size, **kwargs):
    return FakeTensor(shape, dtype, address, element_size=element_size, **kwargs)


def _guarded_tensor(shape, dtype, address, element_size, **kwargs):
    return MetadataGuardTensor(
        shape, dtype, address, element_size=element_size, **kwargs
    )


def _backend(tmp_path, *, state=mapped.MappedBackendState.ACTIVE):
    control = FileControlPlane(tmp_path, rank=0, world_size=2, timeout_s=0.1)
    stopped = []
    backend = mapped.MappedPipelineBackend(
        control=control,
        devices=(0, 1),
        live_token_specializations=(0, 2),
        worker_ctas=2,
        run_id="unit",
        event_pool_depth=2,
        terminator=stopped.append,
    )
    layout = _layout()
    contract = KernelContract(2, 1, 4, 2, 8)
    stream_arg_calls = []
    uint32_calls = []
    capture_queries = []
    capture_state = {
        "current": False,
        "by_handle": {},
        "result": None,
        "raise": None,
    }

    def current_stream_capturing():
        return capture_state["current"]

    dtypes = FakeDTypes(
        cuda=SimpleNamespace(is_current_stream_capturing=current_stream_capturing)
    )
    dlpack_calls = []

    def from_dlpack(value):
        dlpack_calls.append(value)
        return FakeDynamicTensor(value)

    def wrap_stream(value):
        stream_arg_calls.append(value)
        return ("CUstream", value)

    def uint32(value):
        uint32_calls.append(value)
        return int(value)

    def stream_is_capturing(argument):
        capture_queries.append(argument)
        if capture_state["raise"] is not None:
            raise capture_state["raise"]
        if capture_state["result"] is not None:
            return capture_state["result"]
        return 0, capture_state["by_handle"].get(argument[1], 0)

    deps = SimpleNamespace(
        torch=dtypes,
        cutlass=SimpleNamespace(Uint32=uint32, Uint64=int),
        cuda=SimpleNamespace(
            CUstream=wrap_stream,
            CUresult=SimpleNamespace(CUDA_SUCCESS=0),
            CUstreamCaptureStatus=SimpleNamespace(
                CU_STREAM_CAPTURE_STATUS_NONE=0,
                CU_STREAM_CAPTURE_STATUS_ACTIVE=1,
                CU_STREAM_CAPTURE_STATUS_INVALIDATED=2,
            ),
            cuStreamIsCapturing=stream_is_capturing,
        ),
        from_dlpack=from_dlpack,
    )
    arena_base = 1_000_000
    arena = _tensor((layout.arena_nbytes,), dtypes.uint8, arena_base, 1)
    controls = {
        "mask": _tensor((2,), dtypes.int64, arena_base + 128, 8),
        "inc": _tensor((2,), dtypes.uint64, arena_base + 256, 8),
        "bases": _tensor((2,), dtypes.uint64, 4_000_000, 8),
        "statuses": _tensor((2,), dtypes.int32, 4_001_000, 4),
    }
    dense_rows = 8
    buffers = []
    for bank in range(2):
        expert = layout.bank_region("expert_input", bank)
        source = layout.bank_region("dispatch_src_info", bank)
        ranges = layout.bank_region("dispatch_layout_range", bank)
        combine = layout.bank_region("combine_stage", bank)
        buffers.append(
            PipelineBankBuffers(
                bank=bank,
                expert_input=_tensor(
                    (1, dense_rows, 8),
                    dtypes.bfloat16,
                    arena_base + expert.offset,
                    2,
                ),
                expert_counts=_tensor((1,), dtypes.int32, 5_000_000 + bank * 64, 4),
                source_info=_tensor(
                    (1, dense_rows, 4),
                    dtypes.int32,
                    arena_base + source.offset,
                    4,
                ),
                layout_ranges=_tensor(
                    (1, 2),
                    dtypes.uint64,
                    arena_base + ranges.offset,
                    8,
                ),
                combine_stage=_tensor(
                    (1, dense_rows, 8),
                    dtypes.bfloat16,
                    arena_base + combine.offset,
                    2,
                ),
            )
        )
    pool = tuple(
        tuple(
            mapped._EventSet(FakeEvent(), FakeEvent(), FakeEvent(), FakeEvent())
            for _ in range(2)
        )
        for _ in range(2)
    )
    dispatch = FakeLauncher()
    combine = FakeLauncher()
    backend._state = state
    backend._deps = deps
    backend._rank = 0
    backend._layout = layout
    backend._contract = contract
    backend._topology = _topology()
    backend._device = 0
    backend._arena = arena
    backend._rank_mask = controls["mask"]
    backend._rank_incarnations = controls["inc"]
    backend._peer_bases = controls["bases"]
    backend._statuses = controls["statuses"]
    backend._route_counts = (
        _tensor((2,), dtypes.uint64, 4_002_000, 8),
        _tensor((2,), dtypes.uint64, 4_002_064, 8),
    )
    backend._buffers = tuple(buffers)
    backend._event_pool = pool
    backend._drain_events = (FakeEvent(), FakeEvent())
    backend._compiled_dispatch = {0: dispatch, 2: dispatch}
    backend._compiled_combine = {0: combine, 2: combine}
    backend._dummy = {
        "activations": _tensor((1, 8), dtypes.bfloat16, 6_000_000, 2),
        "topk_indices": _tensor((1, 2), dtypes.int32, 6_001_000, 4),
        "topk_weights": _tensor((1, 2), dtypes.float32, 6_002_000, 4),
        "combine_output": _tensor((1, 8), dtypes.bfloat16, 6_003_000, 2),
    }
    backend._control_stream = FakeStream()
    backend._prepare_owned_descriptors()
    backend._activate_prepared_generation(_topology())
    backend._test_dlpack_calls = dlpack_calls
    backend._test_stream_arg_calls = stream_arg_calls
    backend._test_uint32_calls = uint32_calls
    backend._test_capture_queries = capture_queries
    backend._test_capture_state = capture_state
    return backend, stopped, dispatch, combine


def _dispatch_inputs(dtypes, *, tokens=2):
    return (
        _tensor((tokens, 8), dtypes.bfloat16, 10_000_000, 2),
        _tensor((tokens, 2), dtypes.int32, 10_001_000, 4),
    )


def _guarded_operation_inputs(dtypes, *, tokens=2):
    return (
        _guarded_tensor((tokens, 8), dtypes.bfloat16, 10_000_000, 2),
        _guarded_tensor((tokens, 2), dtypes.int32, 10_001_000, 4),
        _guarded_tensor((tokens, 2), dtypes.float32, 10_002_000, 4),
        _guarded_tensor((tokens, 8), dtypes.bfloat16, 10_003_008, 2),
    )


def _tensor_arg_operands(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return tuple(
        ast.unparse(call.args[0])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_tensor_arg"
    )


def test_module_imports_without_loading_torch():
    source = inspect.getsource(mapped)
    prefix = source.split("def _load_device_dependencies", 1)[0]
    assert "import torch" not in prefix
    assert "import nixl.device.cute" not in prefix


def test_constructor_checks_capacity_facts(tmp_path):
    control = FileControlPlane(tmp_path, rank=0, world_size=2)
    with pytest.raises(ValueError, match="one CUDA ordinal"):
        mapped.MappedPipelineBackend(
            control=control,
            devices=(0,),
            live_token_specializations=(1,),
            worker_ctas=2,
            run_id="x",
        )
    with pytest.raises(ValueError, match="distinct GPU"):
        mapped.MappedPipelineBackend(
            control=control,
            devices=(0, 0),
            live_token_specializations=(1,),
            worker_ctas=2,
            run_id="x",
        )


def test_constructor_always_admits_zero_token_specialization(tmp_path):
    control = FileControlPlane(tmp_path, rank=0, world_size=2)
    backend = mapped.MappedPipelineBackend(
        control=control,
        devices=(0, 1),
        live_token_specializations=(2, 4),
        worker_ctas=2,
        run_id="zero-normalized",
    )
    assert backend.live_token_specializations == (0, 2, 4)


def test_live_standby_admission_requires_incarnation_for_every_slot():
    topology = _topology(active=(0,), incarnations=(1, 0))
    with pytest.raises(PipelineError, match="every fixed slot"):
        mapped.MappedPipelineBackend._validate_live_standby_topology(
            topology, initial=True
        )


def test_dispatch_enqueues_waits_records_lifetimes_and_event(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    dtypes = backend._deps.torch
    activations, indices = _dispatch_inputs(dtypes)
    stream = FakeStream()
    input_ready, bank_ready = FakeEvent(), FakeEvent()
    predecessor = FakeEvent()
    event = backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=input_ready,
        bank_reuse_event=bank_ready,
        communication_predecessor_event=predecessor,
        buffers=backend._buffers[0],
        stream=stream,
    )
    assert stream.waited == [predecessor, input_ready, bank_ready]
    assert len(dispatch.calls) == 1
    assert dispatch.calls[0][4].tensor is backend._route_counts[0]
    assert event.recorded == [stream]
    assert activations.recorded_streams == [stream]
    assert indices.recorded_streams == [stream]


def test_admission_prepares_every_backend_owned_tensor_descriptor_once(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    expected = {
        id(backend._arena),
        id(backend._rank_mask),
        id(backend._rank_incarnations),
        id(backend._peer_bases),
        id(backend._statuses),
        *(id(tensor) for tensor in backend._dummy.values()),
    }
    for bank, buffers in enumerate(backend._buffers):
        expected.update(
            id(tensor)
            for tensor in (
                buffers.expert_input,
                buffers.expert_counts,
                buffers.source_info,
                buffers.layout_ranges,
                buffers.combine_stage,
                backend._prepared_owned.banks[bank].route_counts.owner,
            )
        )
    assert {id(tensor) for tensor in backend._test_dlpack_calls} == expected
    assert len(backend._test_dlpack_calls) == len(expected)
    with pytest.raises(PipelineError, match="already prepared"):
        backend._prepare_owned_descriptors()
    with pytest.raises(PipelineError, match="invalidated before activation"):
        backend._activate_prepared_generation(_topology(generation=1))


def test_dispatch_reuses_owned_descriptors_but_rewraps_caller_storage(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    activations, indices = _dispatch_inputs(backend._deps.torch)
    before = len(backend._test_dlpack_calls)

    for step in (0, 2):
        backend.enqueue_dispatch(
            operation=OperationEpoch(0, step),
            source_incarnation=1,
            topology=_topology(),
            activations=activations,
            topk_indices=indices,
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
        activations._address += 4096
        indices._address += 4096

    assert backend._test_dlpack_calls[before:] == [
        activations,
        indices,
        activations,
        indices,
    ]
    first, second = dispatch.calls
    assert first[0] is second[0] is backend._prepared_owned.arena.argument
    assert (
        first[3] is second[3] is backend._prepared_owned.banks[0].expert_counts.argument
    )
    assert (
        first[4] is second[4] is backend._prepared_owned.banks[0].route_counts.argument
    )
    assert first[1] is not second[1]
    assert first[2] is not second[2]
    assert first[1].address != second[1].address
    assert first[2].address != second[2].address


def test_explicit_external_seals_remove_hot_metadata_dlpack_and_lifetime_work(
    tmp_path,
):
    backend, _, dispatch, combine = _backend(tmp_path)
    dtypes = backend._deps.torch
    activations, indices, weights, output = _guarded_operation_inputs(dtypes)
    expert = _guarded_tensor((1, 8, 8), dtypes.bfloat16, 10_004_000, 2)
    bindings = (
        ("activations", activations),
        ("topk_indices", indices),
        ("topk_weights", weights),
        ("combine_output", output),
        ("expert_output", expert),
    )
    before = len(backend._test_dlpack_calls)
    for role, tensor in bindings:
        backend.prepare_external_tensor(tensor, role=role)
    assert len(backend._test_dlpack_calls) == before + len(bindings)
    for _, tensor in bindings:
        tensor.reject_metadata_reads()
    sealed_count = len(backend._test_dlpack_calls)

    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
    )
    backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=expert,
        zero_copy=False,
        topk_indices=indices,
        topk_weights=weights,
        combine_output=output,
        expert_ready_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
    )

    assert len(backend._test_dlpack_calls) == sealed_count
    assert (
        dispatch.calls[0][1]
        is backend._prepared_external["activations"][id(activations)].argument
    )
    assert (
        combine.calls[0][1]
        is backend._prepared_external["expert_output"][id(expert)].argument
    )
    assert all(tensor.recorded_streams == [] for _, tensor in bindings)


def test_prepared_operation_is_identity_only_and_bypasses_hot_cache_span_work(
    tmp_path,
):
    backend, _, dispatch, combine = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    activations, indices, weights, output = inputs
    token = backend.prepare_operation(*inputs)
    assert isinstance(token, mapped.PreparedMappedOperation)
    assert token.live_tokens == 2
    before = len(backend._test_dlpack_calls)
    for tensor in inputs:
        tensor.reject_metadata_reads()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepared operation entered a generic hot-path check")

    backend._find_prepared_external = forbidden
    backend._require_outside_owned_storage = forbidden
    backend._require_disjoint_spans = forbidden
    backend._memory_span = forbidden
    backend._tensor_arg = forbidden

    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
        prepared_operation=token,
    )
    backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=backend._buffers[0].combine_stage,
        zero_copy=True,
        topk_indices=indices,
        topk_weights=weights,
        combine_output=output,
        expert_ready_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
        prepared_operation=token,
    )

    assert len(backend._test_dlpack_calls) == before
    assert dispatch.calls[0][1] is token._activations.argument
    assert dispatch.calls[0][2] is token._topk_indices.argument
    assert combine.calls[0][2] is token._topk_indices.argument
    assert combine.calls[0][3] is token._topk_weights.argument
    assert combine.calls[0][4] is token._combine_output.argument
    assert all(tensor.recorded_streams == [] for tensor in inputs)


def test_direct_path_remains_independent_after_prepared_submission_context(
    tmp_path,
):
    backend, _, dispatch, combine = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    token = backend.prepare_operation(*inputs)
    communication_stream = FakeStream(handle=701)
    expert_stream = FakeStream(handle=702)
    backend.prepare_submission_context(
        streams=(communication_stream, expert_stream),
        first_step=0,
        step_count=2,
    )
    prepared = backend._prepared_submission
    assert prepared is not None
    assert backend._test_stream_arg_calls[-2:] == [701, 702]
    assert backend._test_uint32_calls[-2:] == [0, 1]
    backend.validate_submission_context(
        streams=(communication_stream, expert_stream, communication_stream),
        first_step=0,
        step_count=2,
    )
    assert backend._test_capture_queries == [
        prepared.bindings[id(communication_stream)].argument,
        prepared.bindings[id(expert_stream)].argument,
    ]
    stream_wrapper_count = len(backend._test_stream_arg_calls)
    step_wrapper_count = len(backend._test_uint32_calls)
    for tensor in inputs:
        tensor.reject_metadata_reads()

    expert_launcher = FakeLauncher()
    backend._compiled_expert = expert_launcher

    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=inputs[0],
        topk_indices=inputs[1],
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=communication_stream,
        prepared_operation=token,
    )
    expert_handle = SimpleNamespace(
        operation=OperationEpoch(0, 0),
        combine_buffer=backend._buffers[0].combine_stage,
        _backend_launch=None,
    )
    backend.enqueue_standin_expert(
        handle=expert_handle,
        output=backend._buffers[0].combine_stage,
        stream=expert_stream,
        input_ready_event=None,
    )
    events = backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=backend._buffers[0].combine_stage,
        zero_copy=True,
        topk_indices=inputs[1],
        topk_weights=inputs[2],
        combine_output=inputs[3],
        expert_ready_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=communication_stream,
        prepared_operation=token,
    )

    assert backend._test_stream_arg_calls[stream_wrapper_count:] == [701, 702, 701]
    assert backend._test_uint32_calls[step_wrapper_count:] == [0, 0, 0]
    assert dispatch.calls[0][9] == ("CUstream", 701)
    assert dispatch.calls[0][11] == 0
    assert expert_launcher.calls[0][2] == ("CUstream", 702)
    assert expert_launcher.calls[0][3] == 0
    assert combine.calls[0][10] == ("CUstream", 701)
    assert combine.calls[0][12] == 0
    assert events is backend._event_pool[0][0].combine_events


def test_cold_operation_launch_elides_identity_step_bank_and_event_derivation(
    tmp_path,
):
    backend, _, dispatch, combine = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    operation_token = backend.prepare_operation(*inputs)
    communication_stream = FakeStream(handle=731)
    expert_stream = FakeStream(handle=732)
    expert_launcher = FakeLauncher()
    backend._compiled_expert = expert_launcher
    context = backend.prepare_submission_context(
        streams=(communication_stream, expert_stream),
        first_step=0,
        step_count=1,
    )
    operation = OperationEpoch(0, 0)
    launch = backend.prepare_operation_launch(
        submission_context=context,
        operation=operation,
        source_incarnation=1,
        topology=backend._topology,
        buffers=backend._buffers[0],
        prepared_operation=operation_token,
        communication_stream=communication_stream,
        expert_stream=expert_stream,
    )
    wrapper_count = len(backend._test_stream_arg_calls)
    scalar_count = len(backend._test_uint32_calls)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cold-derived launch metadata was recomputed after D0")

    # The launch record retains the exact bound wait callables. Replacing the
    # owner attributes after cold preparation must not affect prepared posting.
    communication_stream.wait_event = forbidden
    expert_stream.wait_event = forbidden
    backend._events_for = forbidden
    backend._require_hot = forbidden
    backend._require_prepared_bank = forbidden
    for tensor in inputs:
        tensor.reject_metadata_reads()

    dispatch_ready = backend.enqueue_prepared_dispatch(
        launch,
        input_ready_event=None,
        communication_predecessor_event=None,
    )
    handle = SimpleNamespace(
        _backend_launch=launch,
        operation=operation,
        combine_buffer=backend._buffers[0].combine_stage,
    )
    expert_ready = backend.enqueue_standin_expert(
        handle=handle,
        output=backend._buffers[0].combine_stage,
        stream=expert_stream,
        input_ready_event=dispatch_ready,
    )
    events = backend.enqueue_prepared_combine(
        launch,
        expert_ready_event=expert_ready,
        communication_predecessor_event=dispatch_ready,
    )

    assert len(backend._test_stream_arg_calls) == wrapper_count
    assert len(backend._test_uint32_calls) == scalar_count
    assert dispatch.calls[0][9] is launch.communication_stream.argument
    assert dispatch.calls[0][11] is launch.step_argument
    assert expert_launcher.calls[0][2] is launch.expert_stream.argument
    assert expert_launcher.calls[0][3] is launch.step_argument
    assert combine.calls[0][10] is launch.communication_stream.argument
    assert combine.calls[0][12] is launch.step_argument
    assert expert_stream.waited == [dispatch_ready]
    assert communication_stream.waited == [dispatch_ready, expert_ready]
    assert events is launch.events.combine_events


def test_unprepared_submission_retains_per_launch_wrapper_fallback(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    activations, indices = _dispatch_inputs(backend._deps.torch)
    stream = FakeStream(handle=703)
    stream_wrapper_count = len(backend._test_stream_arg_calls)
    step_wrapper_count = len(backend._test_uint32_calls)

    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=stream,
    )

    assert backend._test_stream_arg_calls[stream_wrapper_count:] == [703]
    assert backend._test_uint32_calls[step_wrapper_count:] == [0]
    assert dispatch.calls[0][9] == ("CUstream", 703)
    assert dispatch.calls[0][11] == 0


def test_direct_path_falls_back_for_new_stream_and_step_after_preflight(
    tmp_path,
):
    backend, _, dispatch, _ = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    token = backend.prepare_operation(*inputs)
    admitted = FakeStream(handle=704)
    wrong = FakeStream(handle=705)
    backend.prepare_submission_context(streams=(admitted,), first_step=0, step_count=1)
    wrapper_count = len(backend._test_stream_arg_calls)
    scalar_count = len(backend._test_uint32_calls)

    wrong_ready = FakeEvent()
    admitted_ready = FakeEvent()
    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=inputs[0],
        topk_indices=inputs[1],
        input_ready_event=wrong_ready,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=wrong,
        prepared_operation=token,
    )
    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 2),
        source_incarnation=1,
        topology=_topology(),
        activations=inputs[0],
        topk_indices=inputs[1],
        input_ready_event=admitted_ready,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=admitted,
        prepared_operation=token,
    )

    assert len(dispatch.calls) == 2
    assert wrong.waited == [wrong_ready]
    assert admitted.waited == [admitted_ready]
    assert backend._test_stream_arg_calls[wrapper_count:] == [705, 704]
    assert backend._test_uint32_calls[scalar_count:] == [0, 2]


def test_prepared_submission_range_and_native_stream_alias_are_bounded(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    first = FakeStream(handle=706)
    second_alias = FakeStream(handle=706)

    with pytest.raises(ValueError, match="alias the same native"):
        backend.prepare_submission_context(
            streams=(first, second_alias), first_step=0, step_count=1
        )
    assert backend._prepared_submission is None
    with pytest.raises(ValueError, match="bounded prepared-wrapper"):
        backend.prepare_submission_context(
            streams=(first,),
            first_step=0,
            step_count=mapped._MAX_PREPARED_STEP_COUNT + 1,
        )
    with pytest.raises(ValueError, match="exceeds uint32"):
        backend.prepare_submission_context(
            streams=(first,), first_step=mapped.UINT32_MAX, step_count=2
        )
    assert backend._prepared_submission is None


def test_prepared_submission_context_reuses_exact_plan_and_replaces_next_plan(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=707)
    backend.prepare_submission_context(streams=(stream,), first_step=3, step_count=2)
    prepared = backend._prepared_submission
    wrapper_count = len(backend._test_stream_arg_calls)
    scalar_count = len(backend._test_uint32_calls)
    backend.prepare_submission_context(
        streams=(stream, stream), first_step=3, step_count=2
    )
    assert backend._prepared_submission is prepared
    assert len(backend._test_stream_arg_calls) == wrapper_count
    assert len(backend._test_uint32_calls) == scalar_count
    replacement = backend.prepare_submission_context(
        streams=(stream,), first_step=5, step_count=2
    )
    assert replacement is backend._prepared_submission
    assert replacement is not prepared
    with pytest.raises(PipelineError, match="replaced before use"):
        backend.validate_prepared_submission_context(prepared)

    stream.cuda_stream = 708
    with pytest.raises(PipelineError, match="handle/device changed"):
        backend.validate_prepared_bindings()
    assert backend.state == mapped.MappedBackendState.FAILED
    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert backend._prepared_submission is replacement
    assert prepared.bindings[id(stream)].owner is stream


def test_replaced_context_invalidates_unposted_cold_launch_before_device_post(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    operation_token = backend.prepare_operation(*inputs)
    stream = FakeStream(handle=733)
    first_context = backend.prepare_submission_context(
        streams=(stream,), first_step=0, step_count=1
    )
    launch = backend.prepare_operation_launch(
        submission_context=first_context,
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=backend._topology,
        buffers=backend._buffers[0],
        prepared_operation=operation_token,
        communication_stream=stream,
        expert_stream=stream,
    )
    backend.prepare_submission_context(streams=(stream,), first_step=1, step_count=1)

    with pytest.raises(PipelineError, match="replaced submission"):
        backend.enqueue_prepared_dispatch(
            launch,
            input_ready_event=None,
            communication_predecessor_event=None,
        )

    assert dispatch.calls == []
    assert stream.waited == []


def test_cold_launch_records_preserve_odd_first_step_bank_and_event_ring(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    operation_token = backend.prepare_operation(*inputs)
    communication = (FakeStream(handle=734), FakeStream(handle=735))
    experts = (FakeStream(handle=736), FakeStream(handle=737))
    context = backend.prepare_submission_context(
        streams=communication + experts,
        first_step=3,
        step_count=2,
    )

    launches = tuple(
        backend.prepare_operation_launch(
            submission_context=context,
            operation=OperationEpoch(0, step),
            source_incarnation=1,
            topology=backend._topology,
            buffers=backend._buffers[bank],
            prepared_operation=operation_token,
            communication_stream=communication[bank],
            expert_stream=experts[bank],
        )
        for step, bank in ((3, 1), (4, 0))
    )

    assert [launch.bank for launch in launches] == [1, 0]
    assert launches[0].events is backend._event_pool[1][1]
    assert launches[1].events is backend._event_pool[0][0]
    assert launches[0].communication_stream.owner is communication[1]
    assert launches[1].communication_stream.owner is communication[0]


def test_validate_submission_context_checks_current_and_every_unique_stream(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    first = FakeStream(handle=709)
    second = FakeStream(handle=710)
    backend.prepare_submission_context(
        streams=(first, second), first_step=0, step_count=1
    )

    backend.validate_submission_context(
        streams=(first, first, second), first_step=0, step_count=1
    )
    assert backend._test_capture_queries == [
        ("CUstream", 709),
        ("CUstream", 710),
    ]

    backend._test_capture_queries.clear()
    backend._test_capture_state["current"] = True
    with pytest.raises(PipelineError, match="Graph capture/replay"):
        backend.validate_submission_context(
            streams=(first, second), first_step=0, step_count=1
        )
    assert backend._test_capture_queries == []

    backend._test_capture_state["current"] = False
    backend._test_capture_state["by_handle"] = {709: 1}
    with pytest.raises(PipelineError, match="capture status"):
        backend.validate_submission_context(
            streams=(first, second), first_step=0, step_count=1
        )
    assert backend._test_capture_queries == [("CUstream", 709)]


def test_validate_submission_context_rejects_malformed_torch_capture_result(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=719)
    backend._test_capture_state["current"] = 1

    with pytest.raises(RuntimeError, match="malformed result"):
        backend.validate_submission_context(
            streams=(stream,), first_step=0, step_count=1
        )
    assert backend._test_capture_queries == []


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("not-a-tuple", "malformed"),
        ((0,), "malformed"),
        ((0, 9), "unknown capture status"),
        ((7, 0), "failed with"),
    ],
)
def test_validate_submission_context_rejects_malformed_driver_results(
    tmp_path, result, message
):
    backend, _, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=711)
    backend.prepare_submission_context(streams=(stream,), first_step=0, step_count=1)
    backend._test_capture_state["result"] = result

    with pytest.raises(RuntimeError, match=message):
        backend.validate_submission_context(
            streams=(stream,), first_step=0, step_count=1
        )


def test_validate_submission_context_accepts_prepared_subset_and_rejects_other(
    tmp_path,
):
    backend, _, _, _ = _backend(tmp_path)
    first = FakeStream(handle=712)
    second = FakeStream(handle=713)
    backend.prepare_submission_context(
        streams=(first, second), first_step=0, step_count=1
    )

    backend.validate_submission_context(streams=(first,), first_step=0, step_count=1)
    assert backend._test_capture_queries == [("CUstream", 712)]
    backend._test_capture_queries.clear()
    with pytest.raises(PipelineError, match="unprepared stream"):
        backend.validate_submission_context(
            streams=(first, FakeStream(handle=714)), first_step=0, step_count=1
        )
    assert backend._test_capture_queries == []


def test_submission_range_is_checked_before_any_stream_or_capture_query(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=720)
    backend.prepare_submission_context(streams=(stream,), first_step=3, step_count=4)

    def forbidden_capture_query():
        raise AssertionError("capture query ran before range rejection")

    backend._deps.torch.cuda.is_current_stream_capturing = forbidden_capture_query
    stream.cuda_stream = 999
    with pytest.raises(PipelineError, match="outside the prepared context"):
        backend.validate_submission_context(
            streams=(stream,), first_step=6, step_count=2
        )
    assert backend._test_capture_queries == []


@pytest.mark.parametrize(
    ("first_step", "step_count", "error", "message"),
    [
        (-1, 1, ValueError, "fit uint32"),
        (0, 0, ValueError, "positive integer"),
        (0, True, TypeError, "must be an integer"),
        (mapped.UINT32_MAX, 2, ValueError, "exceeds uint32"),
    ],
)
def test_unprepared_submission_range_rejects_before_stream_wrapping(
    tmp_path, first_step, step_count, error, message
):
    backend, _, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=721)
    wrappers = len(backend._test_stream_arg_calls)

    with pytest.raises(error, match=message):
        backend.validate_submission_context(
            streams=(stream,), first_step=first_step, step_count=step_count
        )

    assert len(backend._test_stream_arg_calls) == wrappers
    assert backend._test_capture_queries == []


def test_unprepared_submission_validation_is_cold_and_nonpersistent(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    stream = FakeStream(handle=715)
    before = len(backend._test_stream_arg_calls)

    backend.validate_submission_context(streams=(stream,), first_step=0, step_count=1)

    assert backend._test_stream_arg_calls[before:] == [715]
    assert backend._test_capture_queries == [("CUstream", 715)]
    assert backend._prepared_submission is None


def test_prepared_operation_zero_tokens_uses_owned_physical_dummies(tmp_path):
    backend, _, dispatch, combine = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch, tokens=0)
    before = len(backend._test_dlpack_calls)
    token = backend.prepare_operation(*inputs)
    assert token.live_tokens == 0
    assert len(backend._test_dlpack_calls) == before
    for tensor in inputs:
        tensor.reject_metadata_reads()

    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=inputs[0],
        topk_indices=inputs[1],
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
        prepared_operation=token,
    )
    backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=backend._buffers[0].combine_stage,
        zero_copy=True,
        topk_indices=inputs[1],
        topk_weights=inputs[2],
        combine_output=inputs[3],
        expert_ready_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
        prepared_operation=token,
    )

    owned = backend._prepared_owned
    assert dispatch.calls[0][1] is owned.dummy_activations.argument
    assert dispatch.calls[0][2] is owned.dummy_topk_indices.argument
    assert combine.calls[0][2] is owned.dummy_topk_indices.argument
    assert combine.calls[0][3] is owned.dummy_topk_weights.argument
    assert combine.calls[0][4] is owned.dummy_combine_output.argument
    assert len(backend._test_dlpack_calls) == before


def test_prepare_operation_is_transactional_and_rejects_joint_alias(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    dtypes = backend._deps.torch
    activations, indices, weights, _ = _guarded_operation_inputs(dtypes)
    output_alias = activations

    with pytest.raises(ValueError, match="must not overlap"):
        backend.prepare_operation(activations, indices, weights, output_alias)

    assert all(not bindings for bindings in backend._prepared_external.values())


def test_prepared_operation_rejects_foreign_token_identity_and_nonzero_copy(
    tmp_path,
):
    backend, _, dispatch, combine = _backend(tmp_path)
    inputs = _guarded_operation_inputs(backend._deps.torch)
    token = backend.prepare_operation(*inputs)
    other, _, other_dispatch, _ = _backend(tmp_path / "other")

    with pytest.raises(PipelineError, match="another backend"):
        other.enqueue_dispatch(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            activations=inputs[0],
            topk_indices=inputs[1],
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=other._buffers[0],
            stream=FakeStream(),
            prepared_operation=token,
        )
    assert other_dispatch.calls == []

    expert = _tensor((1, 8, 8), backend._deps.torch.bfloat16, 10_004_000, 2)
    with pytest.raises(PipelineError, match="combine_stage"):
        backend.enqueue_combine(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            expert_output=expert,
            zero_copy=False,
            topk_indices=inputs[1],
            topk_weights=inputs[2],
            combine_output=inputs[3],
            expert_ready_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
            prepared_operation=token,
        )
    assert dispatch.calls == []
    assert combine.calls == []


def test_external_seal_rejects_alias_with_nonarena_backend_control(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    alias = _tensor(
        (2, 2),
        backend._deps.torch.int32,
        backend._statuses.data_ptr(),
        4,
    )

    with pytest.raises(ValueError, match="backend-owned statuses"):
        backend.prepare_external_tensor(alias, role="topk_indices")
    assert backend._prepared_external["topk_indices"] == {}


def test_cold_binding_validation_fail_stops_on_external_or_owned_rebind(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path / "external")
    activations, _ = _dispatch_inputs(backend._deps.torch)
    backend.prepare_external_tensor(activations, role="activations")
    retained = backend._prepared_external["activations"][id(activations)]
    activations._address += 16

    with pytest.raises(PipelineError, match="storage changed"):
        backend.validate_prepared_bindings()
    assert backend.state == mapped.MappedBackendState.FAILED
    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert backend._prepared_external["activations"][id(activations)] is retained
    assert retained.owner is activations

    other, other_stopped, _, _ = _backend(tmp_path / "owned")
    other._buffers[0].expert_counts._address += 16
    with pytest.raises(PipelineError, match=r"expert_counts\[0\]"):
        other.validate_prepared_bindings()
    assert other.state == mapped.MappedBackendState.FAILED
    assert other_stopped == [mapped._FAIL_STOP_EXIT_CODE]


def test_dispatch_rejects_shape_alias_and_incarnation_before_launch(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    dtypes = backend._deps.torch
    activations, indices = _dispatch_inputs(dtypes)
    indices.shape = (2, 1)
    with pytest.raises(ValueError, match="topk_indices must have shape"):
        backend.enqueue_dispatch(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            activations=activations,
            topk_indices=indices,
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    indices.shape = (2, 2)
    indices._address = activations._address
    with pytest.raises(ValueError, match="must not overlap"):
        backend.enqueue_dispatch(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            activations=activations,
            topk_indices=indices,
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    indices._address = 10_001_000
    with pytest.raises(PipelineError, match="source incarnation"):
        backend.enqueue_dispatch(
            operation=OperationEpoch(0, 0),
            source_incarnation=2,
            topology=_topology(),
            activations=activations,
            topk_indices=indices,
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    assert dispatch.calls == []


def test_zero_token_dispatch_uses_physical_dummies_but_records_logical_inputs(
    tmp_path,
):
    backend, _, dispatch, _ = _backend(tmp_path)
    dtypes = backend._deps.torch
    activations, indices = _dispatch_inputs(dtypes, tokens=0)
    before = len(backend._test_dlpack_calls)
    event = backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
    )
    assert dispatch.calls[0][1].tensor is backend._dummy["activations"]
    assert dispatch.calls[0][2].tensor is backend._dummy["topk_indices"]
    assert len(backend._test_dlpack_calls) == before
    assert activations.recorded_streams
    assert indices.recorded_streams
    assert event is backend._event_pool[0][0].dispatch_ready


def test_combine_accepts_exact_zero_copy_span_and_records_every_operand(tmp_path):
    backend, _, _, combine = _backend(tmp_path)
    dtypes = backend._deps.torch
    _, indices = _dispatch_inputs(dtypes)
    weights = _tensor((2, 2), dtypes.float32, 10_002_000, 4)
    output = _tensor((2, 8), dtypes.bfloat16, 10_003_008, 2)
    expert = backend._buffers[0].combine_stage
    stream = FakeStream()
    ready = FakeEvent()
    predecessor = FakeEvent()
    before = len(backend._test_dlpack_calls)
    events = backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=expert,
        zero_copy=True,
        topk_indices=indices,
        topk_weights=weights,
        combine_output=output,
        expert_ready_event=ready,
        communication_predecessor_event=predecessor,
        buffers=backend._buffers[0],
        stream=stream,
    )
    assert stream.waited == [predecessor, ready]
    assert len(combine.calls) == 1
    assert backend._test_dlpack_calls[before:] == [indices, weights, output]
    assert combine.calls[0][0] is backend._prepared_owned.arena.argument
    assert (
        combine.calls[0][1] is backend._prepared_owned.banks[0].combine_stage.argument
    )
    assert combine.calls[0][5].tensor is backend._route_counts[0]
    assert events.output_ready.recorded == [stream]
    assert events.bank_reusable.recorded == [stream]
    assert expert.recorded_streams == []
    for tensor in (indices, weights, output):
        assert tensor.recorded_streams == [stream]


def test_zero_token_zero_copy_combine_uses_only_prepared_descriptors(tmp_path):
    backend, _, _, combine = _backend(tmp_path)
    dtypes = backend._deps.torch
    _, indices = _dispatch_inputs(dtypes, tokens=0)
    weights = _tensor((0, 2), dtypes.float32, 10_002_000, 4)
    output = _tensor((0, 8), dtypes.bfloat16, 10_003_000, 2)
    before = len(backend._test_dlpack_calls)

    backend.enqueue_combine(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        expert_output=backend._buffers[0].combine_stage,
        zero_copy=True,
        topk_indices=indices,
        topk_weights=weights,
        combine_output=output,
        expert_ready_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
    )

    assert len(backend._test_dlpack_calls) == before
    call = combine.calls[0]
    owned = backend._prepared_owned
    assert call[1] is owned.banks[0].combine_stage.argument
    assert call[2] is owned.dummy_topk_indices.argument
    assert call[3] is owned.dummy_topk_weights.argument
    assert call[4] is owned.dummy_combine_output.argument


def test_combine_rejects_unasserted_or_partial_arena_alias(tmp_path):
    backend, _, _, combine = _backend(tmp_path)
    dtypes = backend._deps.torch
    _, indices = _dispatch_inputs(dtypes)
    weights = _tensor((2, 2), dtypes.float32, 10_002_000, 4)
    output = _tensor((2, 8), dtypes.bfloat16, 10_003_008, 2)
    exact_view = _tensor(
        backend._buffers[0].combine_stage.shape,
        dtypes.bfloat16,
        backend._buffers[0].combine_stage.data_ptr(),
        2,
    )
    with pytest.raises(ValueError, match="only as the exact"):
        backend.enqueue_combine(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            expert_output=exact_view,
            zero_copy=False,
            topk_indices=indices,
            topk_weights=weights,
            combine_output=output,
            expert_ready_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    partial = _tensor(
        backend._buffers[0].combine_stage.shape,
        dtypes.bfloat16,
        backend._buffers[0].combine_stage.data_ptr() + 16,
        2,
    )
    with pytest.raises(PipelineError, match="exact registered"):
        backend.enqueue_combine(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            expert_output=partial,
            zero_copy=True,
            topk_indices=indices,
            topk_weights=weights,
            combine_output=output,
            expert_ready_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    assert combine.calls == []


def test_staging_keeps_old_generation_hot_and_rejects_restart(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    next_topology = _topology(generation=1, active=(0,))
    candidate = mapped._CandidateGeneration(
        topology=next_topology,
        view=SimpleNamespace(release=lambda: None, valid=False),
        rank_mask=object(),
        rank_incarnations=object(),
        peer_bases=object(),
        statuses=object(),
    )
    backend._make_candidate = lambda topology: candidate
    backend._resolve_candidate = lambda value: setattr(value, "resolved", True)
    backend._validate_configuration_quorum = lambda topology: "digest"
    backend.control.barrier = lambda _tag: None
    assert (
        backend.stage_generation(current=_topology(), next_topology=next_topology)
        is candidate
    )
    assert backend.state == mapped.MappedBackendState.STAGED
    activations, indices = _dispatch_inputs(backend._deps.torch)
    backend.enqueue_dispatch(
        operation=OperationEpoch(0, 0),
        source_incarnation=1,
        topology=_topology(),
        activations=activations,
        topk_indices=indices,
        input_ready_event=None,
        bank_reuse_event=None,
        communication_predecessor_event=None,
        buffers=backend._buffers[0],
        stream=FakeStream(),
    )
    assert len(dispatch.calls) == 1

    other, _, _, _ = _backend(tmp_path / "other")
    restarted = _topology(generation=1, active=(0, 1), incarnations=(1, 2))
    with pytest.raises(PipelineError, match="not process incarnation"):
        other.stage_generation(current=_topology(), next_topology=restarted)


def test_stage_resolves_candidate_before_exposing_it_and_commit_is_device_only():
    stage = inspect.getsource(mapped.MappedPipelineBackend.stage_generation)
    make = stage.index("candidate = self._make_candidate(next_topology)")
    resolve = stage.index("self._resolve_candidate(candidate)")
    expose = stage.index("self._staged_candidate = candidate")
    assert make < resolve < expose
    assert "self._topology = next_topology" not in stage

    commit = inspect.getsource(mapped.MappedPipelineBackend.commit_generation)
    rebase = inspect.getsource(mapped.MappedPipelineBackend._rebase_arena)
    for forbidden in (
        "_resolve_candidate(",
        "torch.tensor",
        "torch.empty",
        "torch.zeros",
        ".cpu(",
        ".tolist(",
        ".narrow(",
    ):
        assert forbidden not in commit
        assert forbidden not in rebase
    assert "self._rank_mask.copy_(candidate.rank_mask" in rebase
    assert "candidate.rank_incarnations" in rebase


def test_failed_stage_resolution_releases_candidate_without_touching_active(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    active_generation = backend._prepared_generation
    active_evidence = {"membership_generation": 0}
    backend._mapped_pointer_evidence = active_evidence
    released = []
    next_topology = _topology(generation=1, active=(0,))
    candidate = mapped._CandidateGeneration(
        topology=next_topology,
        view=SimpleNamespace(
            release=lambda: released.append(True),
            valid=False,
        ),
        rank_mask=object(),
        rank_incarnations=object(),
        peer_bases=object(),
        statuses=object(),
    )
    backend._make_candidate = lambda _topology: candidate
    backend._validate_configuration_quorum = lambda _topology: "digest"

    def fail_resolve(_candidate):
        raise RuntimeError("candidate mapping rejected")

    backend._resolve_candidate = fail_resolve
    with pytest.raises(RuntimeError, match="mapping rejected"):
        backend.stage_generation(current=_topology(), next_topology=next_topology)
    assert released == [True]
    assert candidate.released is True
    assert backend.state == mapped.MappedBackendState.ACTIVE
    assert backend._staged_candidate is None
    assert backend._prepared_generation is active_generation
    assert backend.mapped_pointer_evidence == active_evidence


def test_rebase_rebinds_generation_without_recreating_owned_descriptors(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    operation_inputs = _guarded_operation_inputs(backend._deps.torch)
    operation_token = backend.prepare_operation(*operation_inputs)
    owned = backend._prepared_owned
    original = backend._prepared_generation
    backend._prepared_generation = None
    next_topology = _topology(generation=1, active=(0,))

    backend._activate_prepared_generation(next_topology)

    rebound = backend._prepared_generation
    assert rebound is not original
    assert rebound.membership_generation == 1
    assert all(bank.owned is owned for bank in rebound.banks)
    assert all(
        rebound.banks[bank].storage is original.banks[bank].storage for bank in range(2)
    )
    assert rebound.banks[0].generation_arg == 1
    assert rebound.banks[0].source_incarnation_arg == 1
    assert (
        backend._require_dispatch_operation(
            operation_token, operation_inputs[0], operation_inputs[1]
        )
        is operation_token
    )
    with pytest.raises(PipelineError, match="stale generation"):
        backend._require_prepared_bank(OperationEpoch(0, 0))


def test_missing_prepared_generation_fails_before_hot_descriptor_work(tmp_path):
    backend, _, dispatch, _ = _backend(tmp_path)
    activations, indices = _dispatch_inputs(backend._deps.torch)
    before = len(backend._test_dlpack_calls)
    backend._prepared_generation = None

    with pytest.raises(PipelineError, match="no prepared CuTe descriptors"):
        backend.enqueue_dispatch(
            operation=OperationEpoch(0, 0),
            source_incarnation=1,
            topology=_topology(),
            activations=activations,
            topk_indices=indices,
            input_ready_event=None,
            bank_reuse_event=None,
            communication_predecessor_event=None,
            buffers=backend._buffers[0],
            stream=FakeStream(),
        )
    assert len(backend._test_dlpack_calls) == before
    assert dispatch.calls == []


def test_current_topology_uses_identity_fast_path_with_equal_value_fallback(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    backend._require_current_topology(backend._topology)
    backend._require_current_topology(_topology())
    source = inspect.getsource(mapped.MappedPipelineBackend._require_current_topology)
    assert "topology is not self._topology and topology != self._topology" in source


def test_generation_drain_is_event_ordered_and_nonblocking(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    next_topology = _topology(generation=1, active=(0,))
    candidate = mapped._CandidateGeneration(
        topology=next_topology,
        view=object(),
        rank_mask=object(),
        rank_incarnations=object(),
        peer_bases=object(),
        statuses=object(),
    )
    backend._staged_candidate = candidate
    backend._state = mapped.MappedBackendState.STAGED
    stream = FakeStream()
    first, second = FakeEvent(), FakeEvent()
    drain = backend.enqueue_generation_drain(
        current=_topology(),
        staged_token=candidate,
        completion_events=(first, second),
        stream=stream,
    )
    assert stream.waited == [first, second]
    assert drain.token.event.recorded == [stream]
    assert drain.token.event.synchronize_calls == 0
    assert backend.state == mapped.MappedBackendState.DRAINING


def test_fail_stop_quarantines_registration_and_uses_injected_terminator(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path)
    assert backend._prepared_generation is not None
    assert backend._prepared_owned is not None
    activations, _ = _dispatch_inputs(backend._deps.torch)
    backend.prepare_external_tensor(activations, role="activations")
    sealed = backend._prepared_external["activations"][id(activations)]
    launch_stream = FakeStream(handle=716)
    backend.prepare_submission_context(
        streams=(launch_stream,), first_step=0, step_count=1
    )
    prepared_submission = backend._prepared_submission
    operation_cookie = backend._operation_cookie
    registration = object()
    view = object()
    backend._registration = registration
    backend._active_candidate = view
    error = RuntimeError("posted work is uncertain")
    backend.fail_stop(error)
    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert backend.state == mapped.MappedBackendState.FAILED
    assert backend._registration is registration
    assert backend._active_candidate is view
    assert backend._prepared_generation is None
    assert backend._prepared_owned is None
    assert backend._prepared_external["activations"][id(activations)] is sealed
    assert backend._prepared_submission is prepared_submission
    assert prepared_submission.bindings[id(launch_stream)].owner is launch_stream
    assert backend._operation_cookie is not operation_cookie
    with pytest.raises(Exception, match="quarantined"):
        backend.close()


def test_fail_stop_writes_bounded_cause_after_quarantine_before_exit(
    tmp_path, monkeypatch
):
    backend, _, _, _ = _backend(tmp_path)
    writes = []
    terminations = []

    def write(file_descriptor, payload):
        assert file_descriptor == 2
        assert backend.state == mapped.MappedBackendState.FAILED
        assert backend._operation_cookie is None
        writes.append(payload)
        return len(payload)

    def terminate(exit_code):
        assert writes
        terminations.append(exit_code)

    monkeypatch.setattr(mapped.os, "write", write)
    backend._terminator = terminate
    backend.fail_stop(RuntimeError("rank-private causal detail"))

    assert terminations == [mapped._FAIL_STOP_EXIT_CODE]
    assert len(writes) == 1
    assert len(writes[0]) < 4096
    assert writes[0].endswith(b"\n")
    prefix = b"NIXL_CUTE_FAIL_STOP "
    assert writes[0].startswith(prefix)
    payload = json.loads(writes[0][len(prefix) :])
    assert payload == {
        "error": "rank-private causal detail",
        "error_type": "builtins.RuntimeError",
        "rank": 0,
        "schema_version": 1,
        "state": "failed",
    }


def test_fail_stop_diagnostic_failure_uses_constant_fallback(tmp_path, monkeypatch):
    backend, stopped, _, _ = _backend(tmp_path)
    writes = []

    class UnprintableError(RuntimeError):
        def __str__(self):
            raise MemoryError("cannot format failure")

    monkeypatch.setattr(
        mapped.os,
        "write",
        lambda file_descriptor, payload: writes.append(
            (file_descriptor, payload)
        )
        or len(payload),
    )
    backend.fail_stop(UnprintableError())

    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert writes == [(2, mapped._FAIL_STOP_DIAGNOSTIC_FALLBACK)]


def test_fail_stop_terminates_even_if_diagnostic_invocation_raises(
    tmp_path, monkeypatch
):
    backend, stopped, _, _ = _backend(tmp_path)
    error = MemoryError("post-enqueue allocation failed")

    def fail_before_helper_body(**_kwargs):
        assert backend.state == mapped.MappedBackendState.FAILED
        assert backend._operation_cookie is None
        raise MemoryError("cannot enter diagnostic helper")

    monkeypatch.setattr(mapped, "_emit_fail_stop_diagnostic", fail_before_helper_body)
    # The injected test terminator returns, so the original diagnostic failure
    # resumes after the finally block. Production uses ``os._exit`` and never
    # reaches this unwind point.
    with pytest.raises(MemoryError, match="cannot enter diagnostic helper"):
        backend.fail_stop(error)

    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert backend.state == mapped.MappedBackendState.FAILED
    assert backend._failed_error is error


def test_fail_stop_publishes_terminal_state_before_terminator_failure(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    operation_cookie = backend._operation_cookie

    def exhausted_terminator(exit_code):
        assert exit_code == mapped._FAIL_STOP_EXIT_CODE
        assert backend.state == mapped.MappedBackendState.FAILED
        assert backend._operation_cookie is None
        raise MemoryError("terminator allocation failed")

    backend._terminator = exhausted_terminator
    error = MemoryError("post-enqueue allocation failed")
    with pytest.raises(MemoryError, match="terminator allocation failed"):
        backend.fail_stop(error)

    assert backend.state == mapped.MappedBackendState.FAILED
    assert backend._failed_error is error
    assert backend._operation_cookie is not operation_cookie
    assert backend._operation_cookie is None
    source = inspect.getsource(mapped.MappedPipelineBackend.fail_stop)
    assert "object()" not in source
    assert "for candidate in (" not in source


@pytest.mark.parametrize("method_name", ["enqueue_dispatch", "enqueue_combine"])
def test_hot_methods_contain_no_cpu_progress_or_synchronization(method_name):
    source = inspect.getsource(getattr(mapped.MappedPipelineBackend, method_name))
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        ".synchronize(",
        "check_xfer_state",
        "get_new_notifs",
        "torch.empty",
        "torch.zeros",
    ):
        assert forbidden not in source


def test_hot_methods_convert_only_caller_owned_tensors():
    dispatch = _tensor_arg_operands(mapped.MappedPipelineBackend.enqueue_dispatch)
    combine = _tensor_arg_operands(mapped.MappedPipelineBackend.enqueue_combine)
    standin = _tensor_arg_operands(mapped.MappedPipelineBackend.enqueue_standin_expert)
    assert len(dispatch) == 2
    assert set(dispatch) == {"activations", "topk_indices"}
    assert len(combine) == 4
    assert set(combine) == {
        "expert_output",
        "topk_indices",
        "topk_weights",
        "combine_output",
    }
    assert standin == ()


def test_prepared_descriptor_records_never_capture_streams_or_events():
    for record in (
        mapped._PreparedTensor,
        mapped._PreparedBankStorage,
        mapped._PreparedOwnedStorage,
        mapped._PreparedBankGeneration,
        mapped._PreparedGeneration,
        mapped.PreparedMappedOperation,
    ):
        fields = set(record.__dataclass_fields__)
        assert all("stream" not in field and "event" not in field for field in fields)


def test_prepared_submission_cache_owns_only_cold_immutable_launch_state():
    stream_fields = set(mapped._PreparedLaunchStream.__dataclass_fields__)
    context_fields = set(mapped._PreparedSubmissionContext.__dataclass_fields__)
    assert stream_fields == {"owner", "argument", "wait_event", "handle", "device"}
    step_fields = set(mapped._PreparedStepLaunch.__dataclass_fields__)
    assert context_fields == {"first_step", "steps", "bindings"}
    assert step_fields == {"step", "bank", "step_argument", "events"}
    assert all(
        record.__dataclass_params__.frozen
        for record in (
            mapped._PreparedLaunchStream,
            mapped._PreparedStepLaunch,
            mapped._PreparedSubmissionContext,
            mapped._PreparedMappedLaunch,
        )
    )


def test_direct_launches_do_not_consume_a_finite_prepared_context():
    for method in (
        mapped.MappedPipelineBackend.enqueue_dispatch,
        mapped.MappedPipelineBackend.enqueue_combine,
        mapped.MappedPipelineBackend.enqueue_standin_expert,
    ):
        source = inspect.getsource(method)
        assert "self._prepared_submission" not in source
        assert "_prepared_stream_arg" not in source
        assert "_prepared_step_arg" not in source


def test_cold_bound_backend_entrypoints_have_no_hot_identity_or_scalar_math():
    hot_functions = (
        mapped.MappedPipelineBackend.enqueue_prepared_dispatch,
        mapped.MappedPipelineBackend.enqueue_prepared_combine,
        mapped.MappedPipelineBackend._enqueue_prepared_standin_expert,
        mapped.MappedPipelineBackend._require_prepared_mapped_launch,
        mapped.MappedPipelineBackend._require_active_backend,
    )
    forbidden_calls = {
        "id",
        "range",
        "list",
        "dict",
        "tuple",
        "OperationEpoch",
        "CombineEvents",
    }
    for function in hot_functions:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        definition = tree.body[0]
        definition.returns = None
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        ):
            argument.annotation = None
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not calls & forbidden_calls
        assert not any(
            isinstance(
                node,
                (
                    ast.List,
                    ast.ListComp,
                    ast.Dict,
                    ast.DictComp,
                    ast.Set,
                    ast.SetComp,
                    ast.Tuple,
                ),
            )
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
            for node in ast.walk(tree)
        )
    prepared_sources = "\n".join(
        inspect.getsource(function) for function in hot_functions[:3]
    )
    for cold_derivation in (
        "_prepared_stream_arg(",
        "_prepared_step_arg(",
        "_events_for(",
        "_require_hot(",
        "_require_prepared_bank(",
        "operation.bank",
        "handle.bank",
    ):
        assert cold_derivation not in prepared_sources


def test_submission_validator_uses_torch_and_driver_capture_queries():
    source = inspect.getsource(mapped.MappedPipelineBackend.validate_submission_context)
    range_check = source.index("if prepared is not None:")
    stream_checks = source.index("unique_streams = self._normalize_streams(streams)")
    capture_check = source.index("torch.cuda.is_current_stream_capturing()")
    assert range_check < stream_checks < capture_check
    assert "torch.cuda.is_current_stream_capturing()" in source
    assert "_checked_stream_capture_status(binding.argument)" in source
    assert "CUDA Graph capture/replay is unsupported" in source


def test_prepared_operation_api_is_optional_and_identity_validators_are_cold_free():
    for method in (
        mapped.MappedPipelineBackend.enqueue_dispatch,
        mapped.MappedPipelineBackend.enqueue_combine,
    ):
        parameter = inspect.signature(method).parameters["prepared_operation"]
        assert parameter.default is None
    for helper in (
        mapped.MappedPipelineBackend._require_dispatch_operation,
        mapped.MappedPipelineBackend._require_combine_operation,
    ):
        source = inspect.getsource(helper)
        for forbidden in (
            "_find_prepared_external",
            "_memory_span",
            "_tensor_arg",
            "_require_disjoint",
            "_validate_tensor",
            "_validate_shape",
            ".shape",
            ".dtype",
            ".data_ptr",
        ):
            assert forbidden not in source


def test_communication_predecessor_is_a_required_prelaunch_device_wait():
    for method in (
        mapped.MappedPipelineBackend.enqueue_dispatch,
        mapped.MappedPipelineBackend.enqueue_combine,
    ):
        signature = inspect.signature(method)
        parameter = signature.parameters["communication_predecessor_event"]
        assert parameter.default is inspect.Parameter.empty
        source = inspect.getsource(method)
        wait = source.index("stream.wait_event(communication_predecessor_event)")
        launch = source.index("launcher(")
        assert wait < launch


def test_event_ring_uses_distinct_expert_milestone(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    events = backend._events_for(OperationEpoch(0, 0))
    assert events.dispatch_ready is not events.expert_ready
    assert backend._events_for(OperationEpoch(0, 4)) is events


def test_combine_events_wrapper_is_preallocated_and_reused_with_event_ring(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    first = backend._events_for(OperationEpoch(0, 0))
    second = backend._events_for(OperationEpoch(0, 2))
    wrapped = first.combine_events

    assert backend._events_for(OperationEpoch(0, 4)).combine_events is wrapped
    assert second.combine_events is not wrapped
    assert wrapped.output_ready is first.output_ready
    assert wrapped.bank_reusable is first.bank_reusable
    source = inspect.getsource(mapped.MappedPipelineBackend.enqueue_combine)
    assert "return events.combine_events" in source
    assert "CombineEvents(" not in source


def test_direct_example_import_path_is_dependency_free():
    repo = Path(__file__).resolve().parents[2]
    example_root = repo / "examples" / "python" / "cute"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(example_root)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; assert 'site' not in sys.modules; "
                "import moe.mapped_pipeline_backend as m; "
                "assert 'moe.pipeline_kernels' not in sys.modules; "
                "accelerators = {'cuda', 'cutlass', 'nixl', 'torch'}; "
                "assert accelerators.isdisjoint("
                "name.partition('.')[0] for name in sys.modules); "
                "print(m.MappedBackendState.NEW)"
            ),
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"child stdout:\n{result.stdout}\nchild stderr:\n{result.stderr}"
    assert "MappedBackendState.NEW" in result.stdout


@pytest.mark.parametrize(
    (
        "pythonpath",
        "module_name",
        "runtime_module",
        "backend_module",
        "kernel_module",
    ),
    (
        (
            str(Path(__file__).resolve().parents[2]),
            "examples.python.cute.elastic_moe_pipeline_run",
            "examples.python.cute._runtime",
            "examples.python.cute.moe.mapped_pipeline_backend",
            "examples.python.cute.moe.pipeline_kernels",
        ),
        (
            str(Path(__file__).resolve().parents[2] / "examples" / "python" / "cute"),
            "elastic_moe_pipeline_run",
            "_runtime",
            "moe.mapped_pipeline_backend",
            "moe.pipeline_kernels",
        ),
    ),
    ids=("package-root", "direct-example-root"),
)
def test_runner_import_defers_cute_until_rank_codegen_environment(
    pythonpath, module_name, runtime_module, backend_module, kernel_module
):
    repo = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = pythonpath
    environment.pop("CUTE_DSL_KEEP", None)
    environment.pop("CUTE_DSL_DUMP_DIR", None)
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import inspect
        import os
        import sys
        import tempfile
        from pathlib import Path

        class CutlassImportSentinel(importlib.abc.MetaPathFinder):
            fired = False

            def find_spec(self, fullname, path, target=None):
                del path, target
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    assert os.environ.get("CUTE_DSL_KEEP") == "ptx,cubin"
                    dump_dir = Path(os.environ["CUTE_DSL_DUMP_DIR"])
                    assert dump_dir.is_dir()
                    assert dump_dir.name == "rank-3"
                    self.fired = True
                    raise RuntimeError("EXPECTED_CUTLASS_IMPORT_SENTINEL")
                return None

        sentinel = CutlassImportSentinel()
        sys.meta_path.insert(0, sentinel)
        module = importlib.import_module({module_name!r})
        assert not sentinel.fired
        assert {runtime_module!r} not in sys.modules
        assert {backend_module!r} not in sys.modules
        assert {kernel_module!r} not in sys.modules
        source = inspect.getsource(module._worker)
        assert (
            source.index("_configure_codegen_dump(")
            < source.index("if __package__:")
            < source.index("import torch")
            < source.index("MappedPipelineBackend(")
            < source.index("_prepare_worker(")
        )
        with tempfile.TemporaryDirectory() as root:
            try:
                module._worker(
                    3,
                    (0, 1, 2, 3),
                    root,
                    None,
                    1.0,
                    None,
                    None,
                    root,
                )
            except RuntimeError as error:
                assert str(error) == "EXPECTED_CUTLASS_IMPORT_SENTINEL"
            else:
                raise AssertionError("worker never attempted the guarded Cutlass import")
        assert sentinel.fired
        print(module.STANDIN_EXPERT_BIAS_SCALE)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "64"


def test_configuration_quorum_binds_generation_and_digest(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    calls = []

    def unanimous(tag, payload):
        calls.append((tag, payload))
        return {0: payload, 1: payload}

    backend.control.exchange = unanimous
    digest = backend._validate_configuration_quorum(_topology())
    assert len(digest) == 64
    assert f"-g0-d{digest}.unanimous" in calls[0][0]
    evidence = json.loads(calls[0][1].decode("utf-8"))
    assert evidence["digest"] == digest
    assert evidence["document"]["layout"]["arena_nbytes"] == _layout().arena_nbytes
    assert evidence["document"]["devices"] == [0, 1]


def test_ucx_handshake_nonce_binds_committed_configuration_digest():
    source = inspect.getsource(
        mapped.MappedPipelineBackend._initialize_agent_and_metadata
    )
    assert "generation=self._topology.membership_generation" in source
    assert "self._configuration_digest(self._topology)" in source


def test_configuration_quorum_rejects_adversarial_peer_document(tmp_path):
    backend, _, _, _ = _backend(tmp_path)

    def divergent(_tag, payload):
        peer = json.loads(payload.decode("utf-8"))
        peer["document"]["devices"] = [1, 0]
        return {0: payload, 1: json.dumps(peer).encode("utf-8")}

    backend.control.exchange = divergent
    with pytest.raises(RuntimeError, match="disagrees with configuration digest"):
        backend._validate_configuration_quorum(_topology())


def test_mapped_pointer_evidence_is_json_safe_and_never_leaks_addresses(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    backend._compiled_resolve = FakeLauncher()
    backend._control_stream = FakeStream()
    before = len(backend._test_dlpack_calls)
    candidate = mapped._CandidateGeneration(
        topology=_topology(),
        view=object(),
        rank_mask=_tensor((2,), "int64", 7_000_000, 8, values=[0, 0]),
        rank_incarnations=_tensor((2,), "uint64", 7_000_016, 8, values=[1, 1]),
        peer_bases=_tensor(
            (2,), "uint64", 7_000_032, 8, values=[1_000_000, 9_876_543_216]
        ),
        statuses=_tensor((2,), "int32", 7_000_048, 4, values=[0, 0]),
    )
    backend._resolve_candidate(candidate)
    assert candidate.prepared is not None
    assert len(backend._test_dlpack_calls) == before + 4
    with pytest.raises(PipelineError, match="already resolved"):
        backend._resolve_candidate(candidate)
    assert backend.mapped_pointer_evidence is None
    backend._mapped_pointer_evidence = candidate.mapped_pointer_evidence
    evidence = backend.mapped_pointer_evidence
    assert evidence["membership_generation"] == 0
    assert evidence["arena_nbytes"] == _layout().arena_nbytes
    assert len(evidence["configuration_digest"]) == 64
    assert evidence["peers"][0]["classification"] == "local_registered_arena"
    assert evidence["peers"][1]["classification"] == "mapped_process_local_peer"
    encoded = json.dumps(evidence, sort_keys=True)
    assert "9876543216" not in encoded
    assert "address" not in encoded
    evidence["peers"][0]["active"] = False
    assert backend.mapped_pointer_evidence["peers"][0]["active"] is True


@pytest.mark.parametrize(
    ("length_delta", "device", "address_delta", "message"),
    [
        (16, 0, 0, "arena length"),
        (0, 1, 0, "published device"),
        (0, 0, 1, "16-byte aligned"),
    ],
)
def test_coordinate_validation_rejects_malformed_owner_region(
    tmp_path, length_delta, device, address_delta, message
):
    backend, _, _, _ = _backend(tmp_path)
    layout = _layout()
    coordinates = (
        mapped.PeerCoordinates(
            "rank0",
            (
                mapped.DeviceRegion(
                    8_000_000 + address_delta,
                    layout.arena_nbytes + length_delta,
                    device,
                ),
            ),
        ),
        mapped.PeerCoordinates(
            "rank1",
            (mapped.DeviceRegion(9_000_000, layout.arena_nbytes, 1),),
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        backend._validate_coordinates(coordinates)


def test_coordinate_validation_accepts_complete_well_formed_vector(tmp_path):
    backend, _, _, _ = _backend(tmp_path)
    layout = _layout()
    coordinates = tuple(
        mapped.PeerCoordinates(
            f"rank{rank}",
            (
                mapped.DeviceRegion(
                    8_000_000 + rank * 1_000_000, layout.arena_nbytes, rank
                ),
            ),
        )
        for rank in range(layout.max_ranks)
    )
    backend._validate_coordinates(coordinates)


def test_initial_admission_quorum_precedes_storage_registration_and_view():
    source = inspect.getsource(mapped.MappedPipelineBackend.preallocate)
    quorum = source.index("self._validate_configuration_quorum(topology)")
    storage = source.index("self._initialize_storage()")
    metadata = source.index("self._initialize_agent_and_metadata()")
    candidate = source.index("self._make_candidate(topology)")
    assert quorum < storage < metadata < candidate


def test_all_pooled_events_are_materialized_before_activation():
    source = inspect.getsource(mapped.MappedPipelineBackend._initialize_storage)
    assert "events.dispatch_ready.record(self._control_stream)" in source
    assert "events.expert_ready.record(self._control_stream)" in source
    assert "events.output_ready.record(self._control_stream)" in source
    assert "events.bank_reusable.record(self._control_stream)" in source
    assert "event.record(self._control_stream)" in source
    assert source.rindex("self._control_stream.synchronize()") > source.index(
        "events.dispatch_ready.record"
    )


def test_compile_admits_one_exact_worker_grid_for_dispatch_and_combine():
    source = inspect.getsource(mapped.MappedPipelineBackend._compile_specializations)
    dispatch_compile, after_dispatch = source.split(
        "dispatch = deps.nixl_cute.compile(", 1
    )[1].split("combine = deps.nixl_cute.compile(", 1)
    combine_compile, occupancy = after_dispatch.split(
        "bound_dispatch, dispatch_occupancy", 1
    )

    assert "live_tokens,\n                self.worker_ctas,\n" in dispatch_compile
    assert "live_tokens,\n                self.worker_ctas,\n" in combine_compile
    dispatch_bind, combine_bind = occupancy.split("bound_combine, combine_occupancy", 1)
    assert "planned_ctas=self.worker_ctas" in dispatch_bind
    assert "planned_ctas=self.worker_ctas" in combine_bind
    assert "planned_ctas=contract.dispatch_ctas" not in source
    assert "--keep-cubin" not in source
    assert "--dump-dir" not in source


def test_commit_failure_enters_terminal_quarantine(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path)
    next_topology = _topology(generation=1, active=(0,))
    candidate = mapped._CandidateGeneration(
        topology=next_topology,
        view=object(),
        rank_mask=object(),
        rank_incarnations=object(),
        peer_bases=object(),
        statuses=object(),
    )
    backend._staged_candidate = candidate
    backend._state = mapped.MappedBackendState.DRAINING
    drain_event = FakeEvent()
    drain = mapped.GenerationDrain(mapped._DrainToken(candidate, drain_event))

    def fail_barrier(_tag):
        raise TimeoutError("quorum disappeared")

    backend.control.barrier = fail_barrier
    with pytest.raises(TimeoutError, match="quorum disappeared"):
        backend.commit_generation(
            current=_topology(),
            next_topology=next_topology,
            staged_token=candidate,
            drain=drain,
        )
    assert drain_event.synchronize_calls == 1
    assert backend.state == mapped.MappedBackendState.FAILED
    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    with pytest.raises(Exception, match="fail-stopped"):
        backend._require_active_backend()


def test_commit_invalidates_old_generation_before_rebase_and_rebinds_before_active():
    source = inspect.getsource(mapped.MappedPipelineBackend.commit_generation)
    invalidate = source.index("self._prepared_generation = None")
    rebase = source.index("self._rebase_arena(candidate)")
    activate = source.index("self._activate_prepared_generation(next_topology)")
    active = source.rindex("self._state = MappedBackendState.ACTIVE")
    assert invalidate < rebase < activate < active


def test_close_failure_enters_terminal_quarantine_without_dropping_refs(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path)
    activations, _ = _dispatch_inputs(backend._deps.torch)
    backend.prepare_external_tensor(activations, role="activations")
    sealed = backend._prepared_external["activations"][id(activations)]
    launch_stream = FakeStream(handle=717)
    backend.prepare_submission_context(
        streams=(launch_stream,), first_step=0, step_count=1
    )
    prepared_submission = backend._prepared_submission
    registration = object()
    arena = backend._arena
    backend._registration = registration
    backend._active_candidate = None
    backend._peer_names = ()
    backend.control.barrier = lambda _tag: None
    backend._control_stream = FakeStream()
    backend._drain_events = (FakeEvent(), FakeEvent())
    backend._agent = SimpleNamespace(
        deregister_memory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("deregister failed")
        )
    )
    with pytest.raises(RuntimeError, match="deregister failed"):
        backend.close()
    assert backend.state == mapped.MappedBackendState.FAILED
    assert stopped == [mapped._FAIL_STOP_EXIT_CODE]
    assert backend._registration is registration
    assert backend._arena is arena
    assert backend._prepared_generation is None
    assert backend._prepared_owned is None
    assert backend._prepared_external["activations"][id(activations)] is sealed
    assert backend._prepared_submission is prepared_submission


def test_healthy_close_drops_every_backend_device_owner(tmp_path):
    backend, stopped, _, _ = _backend(tmp_path)
    operation_inputs = _guarded_operation_inputs(backend._deps.torch)
    token = backend.prepare_operation(*operation_inputs)
    operation_cookie = token._backend_cookie
    launch_stream = FakeStream(handle=718)
    backend.prepare_submission_context(
        streams=(launch_stream,), first_step=0, step_count=1
    )
    backend._registration = object()
    backend._active_candidate = None
    backend._peer_names = ()
    backend.control.barrier = lambda _tag: None
    backend._control_stream = FakeStream()
    backend._drain_events = (FakeEvent(), FakeEvent())

    def deregister(*_args, **_kwargs):
        backend._registration = None

    backend._agent = SimpleNamespace(deregister_memory=deregister)
    backend.close()
    assert stopped == []
    assert backend.state == mapped.MappedBackendState.CLOSED
    for name in (
        "_arena",
        "_agent",
        "_deps",
        "_control_stream",
        "_rank_mask",
        "_rank_incarnations",
        "_peer_bases",
        "_statuses",
        "_route_counts",
        "_prepared_generation",
        "_prepared_owned",
    ):
        assert getattr(backend, name) is None
    assert backend._event_pool == ()
    assert backend._rebase_planes == ()
    assert all(not bindings for bindings in backend._prepared_external.values())
    assert backend._prepared_submission is None
    assert backend._operation_cookie is not operation_cookie
    assert backend._buffers == ()
    assert backend._compiled_dispatch == {}
    assert backend._compiled_combine == {}


def test_runner_preallocates_all_phase_resources_before_phase_loop():
    setup = inspect.getsource(runner._prepare_worker)
    source = inspect.getsource(runner._worker)
    measured_phases = source.split(
        "    try:\n        for generation, topology in enumerate(topologies):", 1
    )[1]
    assert "phase_tensors = tuple(" in setup
    assert "phase_measurements = tuple(" in setup
    assert "_bind_phase_schedules(pipeline, phase_tensors)" in setup
    assert "correctness_outputs = torch.full(" in inspect.getsource(
        runner._make_phase_tensors
    )
    assert "correctness_windows = tuple(" in inspect.getsource(
        runner._make_phase_tensors
    )
    assert "_prime_phase_measurements(phase_measurements, preparation_stream)" in setup
    assert "torch.cuda.Event" not in measured_phases
    assert "_make_phase_tensors(" not in measured_phases
    assert "prepare_two_bank_window(" not in measured_phases
    assert ".data_ptr(" not in measured_phases


def test_runner_correctness_pass_preserves_every_output_past_event_ring_wrap():
    source = inspect.getsource(runner._worker)
    submit = source.index("correctness_latest = _submit_correctness_windows(")
    drain = source.index("measurement.correctness_end.synchronize()", submit)
    validate = source.index("_validate_correctness_outputs(", drain)
    warmup = source.index("_submit_windows(", validate)
    assert submit < drain < validate < warmup
    make = inspect.getsource(runner._make_phase_tensors)
    assert "correctness_outputs[operation]" in make
    assert "range(case.correctness_iterations)" in make
    case_source = inspect.getsource(runner.PipelineRunCase.__post_init__)
    assert "correctness_iterations <= 2 * self.event_pool_depth" in case_source


def test_runner_operation_signatures_reject_same_parity_and_ring_wrap_stale_data(
    monkeypatch,
):
    case = SimpleNamespace(correctness_iterations=132, event_pool_depth=64)
    topology = _topology()
    attestation = runner._correctness_signature_attestation(
        case=case,
        topology=topology,
        live_tokens=4,
        routed_records=7,
    )
    signatures = [
        runner._correctness_operation_signature(operation, attestation["scale"])
        for operation in range(case.correctness_iterations)
    ]
    assert attestation == {
        "policy": "uint32_le_bfloat16_activation_channels_v1",
        "feature_offsets_mod_8": [4, 5, 6, 7],
        "scale": 2048.0,
        "exact_bfloat16": True,
        "operation_count": 132,
        "materialized_rows_per_operation": 4,
        "payload_observable": True,
        "same_parity_probe": {
            "operations": [0, 2],
            "signatures": [list(signatures[0]), list(signatures[2])],
            "distinct": True,
        },
        "event_ring_wrap_probe": {
            "event_pool_depth": 64,
            "operations": [0, 128],
            "signatures": [list(signatures[0]), list(signatures[128])],
            "distinct": True,
        },
    }
    assert signatures[0] != signatures[2]
    assert signatures[0] != signatures[128]
    for signature in signatures:
        for value in signature:
            float32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
            assert float32_bits & 0xFFFF == 0

    class HostBatch:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)

        def cpu(self):
            return self

        def __getitem__(self, index):
            return self.values[index]

    class HostOperand:
        def cpu(self):
            return self

    banks = tuple(
        SimpleNamespace(topk_indices=HostOperand(), topk_weights=HostOperand())
        for _ in range(2)
    )
    monkeypatch.setattr(
        runner,
        "_expected_output_from_host",
        lambda _torch, activations, _indices, _weights: activations,
    )
    monkeypatch.setattr(
        runner,
        "_compare_output",
        lambda _torch, observed, expected: (
            observed == expected,
            0.0 if observed == expected else 1.0,
        ),
    )

    def validate(observed):
        return runner._validate_correctness_outputs(
            object(),
            SimpleNamespace(
                correctness_outputs=HostBatch(observed),
                correctness_activations=HostBatch(signatures),
                inputs=banks,
            ),
        )

    assert validate(signatures) == (True, 0.0)
    same_parity_stale = list(signatures)
    same_parity_stale[2] = signatures[0]
    assert validate(same_parity_stale) == (False, 1.0)
    ring_wrap_stale = list(signatures)
    ring_wrap_stale[128] = signatures[0]
    assert validate(ring_wrap_stale) == (False, 1.0)


def test_runner_configures_fresh_rank_private_codegen_before_torch_import(
    tmp_path, monkeypatch
):
    root = tmp_path / "codegen"
    root.mkdir()
    monkeypatch.delenv("CUTE_DSL_DUMP_DIR", raising=False)
    monkeypatch.delenv("CUTE_DSL_KEEP", raising=False)

    configured = runner._configure_codegen_dump(str(root), 3)
    assert configured == str((root / "rank-3").resolve())
    assert Path(configured).is_dir()
    assert os.environ["CUTE_DSL_DUMP_DIR"] == configured
    assert os.environ["CUTE_DSL_KEEP"] == "ptx,cubin"
    with pytest.raises(FileExistsError):
        runner._configure_codegen_dump(str(root), 3)

    worker_source = inspect.getsource(runner._worker)
    assert worker_source.index("_configure_codegen_dump") < worker_source.index(
        "import torch"
    )


@pytest.mark.parametrize(
    "variable",
    (
        "CUTE_DSL_KEEP",
        "CUTE_DSL_KEEP_IR",
        "CUTE_DSL_KEEP_PTX",
        "CUTE_DSL_KEEP_CUBIN",
    ),
)
def test_runner_rejects_unscoped_inherited_codegen_retention(variable, monkeypatch):
    for name in (
        "CUTE_DSL_KEEP",
        "CUTE_DSL_KEEP_IR",
        "CUTE_DSL_KEEP_PTX",
        "CUTE_DSL_KEEP_CUBIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1" if variable != "CUTE_DSL_KEEP" else "cubin")
    with pytest.raises(ValueError, match="--codegen-dump-root"):
        runner._configure_codegen_dump(None, 0)


def test_direct_runner_reexecutes_the_canonical_package_before_imports(
    tmp_path,
):
    source_path = Path(runner.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    trampoline = source.index('if __name__ == "__main__" and not __package__:')
    bootstrap = source.index("runpy.run_module(", trampoline)
    reexecute = source.index("os.execve(", trampoline)
    canonical_module = source.index(
        'canonical_module = "examples.python.cute.elastic_moe_pipeline_run"',
        trampoline,
    )
    normal_imports = source.index(
        "if __package__:\n    from .elastic_moe_pipeline import ("
    )
    assert trampoline < canonical_module < bootstrap < reexecute < normal_imports
    assert '"-c"' in source[reexecute:normal_imports]
    assert "run_name='__main__'" in source[bootstrap:reexecute]

    environment = dict(os.environ)
    # Exercise promotion of a pre-existing late source-root entry as well as
    # the ordinary direct-file entry point.
    environment["PYTHONPATH"] = os.pathsep.join(
        ("/nonexistent/late-entry", os.fspath(source_path.parents[3]))
    )
    shadow = tmp_path / "examples"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        'raise RuntimeError("caller examples package shadowed NIXL")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, os.fspath(source_path), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run the split NIXL/CuTe" in result.stdout


def test_runner_times_host_enqueue_with_one_uninterrupted_bracket():
    source = inspect.getsource(runner._worker)
    preflight = source.index("throughput_preflight = preflight_rolling_moe_layers(")
    warm_barrier = source.index('control.barrier(f"runner-g{generation}-warm")')
    start = source.index("host_enqueue_start_ns = time.perf_counter_ns()")
    submit = source.index("latest = _submit_windows(", start)
    end = source.index("host_enqueue_end_ns = time.perf_counter_ns()", submit)
    assert preflight < warm_barrier < start
    between_issue_and_barrier = source[preflight:warm_barrier]
    assert ".record(" not in between_issue_and_barrier
    assert ".wait_event(" not in between_issue_and_barrier
    between = source[start:end]
    assert between.count("time.perf_counter_ns()") == 1
    assert "timing_events=None" in between
    assert "submission_preflight=throughput_preflight" in between
    assert "submission_ready_event=measurement.phase_start" in between
    assert "validate_submission_context" not in between
    assert "preflight_rolling_moe_layers" not in between
    assert "diagnostic_operation_events" not in between
    assert ".synchronize(" not in between
    assert ".cpu(" not in between
    assert "host_enqueue_total_ms" in source
    assert "host_enqueue_us_per_operation" in source

    submit_helper = inspect.getsource(runner._submit_windows)
    assert "submission_preflight=submission_preflight" in submit_helper
    assert "submission_ready_event=submission_ready_event" in submit_helper

    rolling = inspect.getsource(runner.enqueue_rolling_moe_layers)
    consume = rolling.index("_consume_rolling_preflight(")
    fast_submit = rolling.index("_enqueue_preflighted_fast_rolling(")
    assert consume < fast_submit
    fast_rolling = inspect.getsource(elastic_pipeline._enqueue_preflighted_fast_rolling)
    ready_wait = fast_rolling.index("admitted[0].wait_event(submission_ready_event)")
    first_dispatch = fast_rolling.index("pipeline._dispatch_preallocated(")
    assert ready_wait < first_dispatch


def test_runner_exposes_matched_throughput_submission_control_in_schema():
    defaults = runner._parser().parse_args([])
    convenience = runner._parser().parse_args(
        ["--throughput-submission-mode", "convenience"]
    )
    assert defaults.throughput_submission_mode == "preflight"
    assert convenience.throughput_submission_mode == "convenience"

    source = inspect.getsource(runner._worker)
    assert 'case.throughput_submission_mode == "preflight"' in source
    assert '"throughput_submission_mode": case.throughput_submission_mode' in source
    assert "throughput_preflight = None" in source
    assert "finite host plan built inside timing" in source

    with pytest.raises(ValueError, match="throughput_submission_mode"):
        runner.PipelineRunCase(
            max_ranks=1,
            experts_per_rank=1,
            token_capacity=1,
            top_k=1,
            hidden_size=8,
            worker_ctas=1,
            warmup=0,
            iterations=2,
            membership=((0,),),
            phase_live_tokens=((1,),),
            event_pool_depth=1,
            correctness_iterations=4,
            latency_iterations=2,
            throughput_submission_mode="unknown",
        )


def test_runner_cold_drains_preparation_and_omits_satisfied_hot_waits():
    source = inspect.getsource(runner._make_phase_tensors)
    assert "torch.cuda.stream(preparation_stream)" in source
    assert "input_ready_event=None" in source
    prime = inspect.getsource(runner._prime_phase_measurements)
    assert "preparation_stream.synchronize()" in prime
    assert "Real router adapters" in prime
    poison = inspect.getsource(mapped.MappedPipelineBackend.poison_data_planes)
    assert "torch.cuda.stream(self._control_stream)" in poison
    assert "self._control_stream.synchronize()" in poison


def test_phase_end_waits_both_bank_events_before_recording():
    measurement_stream = FakeStream()
    phase_end = FakeEvent()
    first, second = FakeEvent(), FakeEvent()

    def submission(event):
        return SimpleNamespace(
            enqueue_bank_reuse_wait=lambda stream, event=event: stream.wait_event(event)
        )

    submissions = (
        submission(first),
        submission(second),
    )
    runner._record_transitive_phase_end(
        measurement_stream=measurement_stream,
        latest_submissions=submissions,
        phase_end=phase_end,
    )
    assert measurement_stream.waited == [first, second]
    assert phase_end.recorded == [measurement_stream]


def test_runner_primes_every_lazy_timing_event_then_cold_drains():
    stream = FakeStream()
    first = runner._PhaseMeasurement(
        diagnostic_operation_events=(
            (FakeEvent(), FakeEvent()),
            (FakeEvent(), FakeEvent()),
        ),
        diagnostic_end=FakeEvent(),
        correctness_end=FakeEvent(),
        phase_start=FakeEvent(),
        phase_end=FakeEvent(),
    )
    second = runner._PhaseMeasurement(
        diagnostic_operation_events=((FakeEvent(), FakeEvent()),),
        diagnostic_end=FakeEvent(),
        correctness_end=FakeEvent(),
        phase_start=FakeEvent(),
        phase_end=FakeEvent(),
    )
    runner._prime_phase_measurements((first, second), stream)
    for measurement in (first, second):
        for start, end in measurement.diagnostic_operation_events:
            assert start.recorded == [stream]
            assert end.recorded == [stream]
        assert measurement.diagnostic_end.recorded == [stream]
        assert measurement.correctness_end.recorded == [stream]
        assert measurement.phase_start.recorded == [stream]
        assert measurement.phase_end.recorded == [stream]
    assert stream.synchronize_calls == 1


def test_runner_default_live_counts_include_unequal_and_zero_cases():
    membership = ((0, 1), (0,), (0, 1))
    live = runner.parse_phase_live_tokens(
        None,
        membership=membership,
        max_ranks=2,
        capacity=8,
    )
    assert len(live) == len(membership)
    assert all(len(phase) == 2 for phase in live)
    assert any(0 in phase for phase in live)
    assert any(len(set(phase)) > 1 for phase in live)


def test_route_plan_has_distinct_valid_routes_drops_and_empty_expert():
    topology = StableSparseTopology(
        max_ranks=2,
        experts_per_rank=2,
        active_ranks=(0, 1),
        membership_generation=0,
        rank_incarnations=(1, 1),
    )
    indices, weights, summary = runner._route_plan(
        rank=0,
        live_tokens=16,
        topology=topology,
        top_k=4,
    )
    assert summary["dropped_routes"] > 0
    assert summary["intentionally_empty_experts"] == 1
    assert len(summary["expert_route_counts"]) == 4
    assert sum(summary["expert_route_counts"]) == summary["routed_records"]
    assert summary["expert_route_counts"][-1] == 0
    assert summary["max_expert_route_count"] == max(summary["expert_route_counts"])
    assert summary["gate_weight_policy"] == (
        "normalized_positive_distinct_route_slot_plus_one_odd_normalizer"
    )
    assert len(set(summary["expert_route_counts"])) > 1
    assert all(
        len({value for value in row if value >= 0}) == sum(value >= 0 for value in row)
        for row in indices
    )
    assert all(
        (
            math.isclose(sum(row), 1.0)
            if any(value >= 0 for value in route)
            else sum(row) == 0.0
        )
        for route, row in zip(indices, weights)
    )
    assert all(
        len({weight for expert, weight in zip(route, row) if expert >= 0})
        == sum(expert >= 0 for expert in route)
        for route, row in zip(indices, weights)
    )
    for route, row in zip(indices, weights):
        coefficients, normalizer = runner._route_weight_coefficients(route)
        if normalizer == 0:
            assert all(expert < 0 for expert in route)
            assert all(weight == 0.0 for weight in row)
            continue
        assert normalizer % 2 == 1
        assert sum(coefficients) == normalizer
        assert all(
            weight == 0.0
            for expert, weight in zip(route, row)
            if expert < 0
        )
        assert all(
            math.isclose(weight, coefficient / normalizer)
            for expert, weight, coefficient in zip(route, row, coefficients)
            if expert >= 0
        )


def test_route_weight_coefficients_preserve_uniqueness_for_every_drop_mask():
    top_k = 8
    for valid_mask in range(1 << top_k):
        choices = [
            route_slot if valid_mask & (1 << route_slot) else -1
            for route_slot in range(top_k)
        ]
        coefficients, normalizer = runner._route_weight_coefficients(choices)
        valid_coefficients = [
            coefficient
            for expert, coefficient in zip(choices, coefficients)
            if expert >= 0
        ]
        assert all(
            coefficient == 0
            for expert, coefficient in zip(choices, coefficients)
            if expert < 0
        )
        if not valid_coefficients:
            assert normalizer == 0
            continue
        assert normalizer % 2 == 1
        assert normalizer == sum(valid_coefficients)
        assert all(coefficient > 0 for coefficient in valid_coefficients)
        assert valid_coefficients == sorted(valid_coefficients)
        assert len(set(valid_coefficients)) == len(valid_coefficients)


def test_odd_route_normalizer_removes_the_observed_bf16_midpoint_tie():
    # N=16/K=8 diagnostics hit this route row at token 10.  At operation 38,
    # expert BF16 outputs occupy the adjacent values below.  Legacy 1..K
    # coefficients assigned exactly half of the weight to the upper value;
    # a +/- 0.0078125 FP32 FMA delta therefore selected opposite BF16 results.
    choices = [0, -1, 1, 5, 4, 2, 3, 6]
    upper_slots = (3, 4, 7)
    legacy = [
        route_slot + 1 if expert >= 0 else 0
        for route_slot, expert in enumerate(choices)
    ]
    assert 2 * sum(legacy[slot] for slot in upper_slots) == sum(legacy) == 34

    coefficients, normalizer = runner._route_weight_coefficients(choices)
    assert coefficients == [1, 0, 3, 4, 5, 6, 7, 9]
    assert normalizer == 35
    lower = 77824
    upper = 78336
    weighted = sum(
        Fraction(coefficient, normalizer)
        * (upper if slot in upper_slots else lower)
        for slot, coefficient in enumerate(coefficients)
        if coefficient
    )
    midpoint = Fraction(lower + upper, 2)
    assert weighted - midpoint == Fraction(256, 35)


def test_runner_correctness_inputs_encode_rank_token_bank_and_feature_channels():
    source = inspect.getsource(runner._make_phase_tensors)
    assert "feature_index.remainder(3)" in source
    assert "token_signature = (token_index + 1.0) * 512.0" in source
    assert "float(rank + 1) * 512.0" in source
    assert "float(bank + 1) * 512.0" in source
    assert "feature_index.remainder(16)" in source
    assert "correctness_activations[0::2].copy_(typed_inputs[0].activations)" in source
    assert "correctness_activations[1::2].copy_(typed_inputs[1].activations)" in source
    assert "correctness_activations[:, :, feature_offset::8]" in source
    assert "activations=correctness_activations[operation]" in source
    golden = inspect.getsource(runner._expected_output_from_host)
    assert "STANDIN_EXPERT_BIAS_SCALE" in golden
