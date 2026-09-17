# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side layout and correctness model for the NIXL CuTe MoE examples.

Protocol control-plane imports intentionally do not import Torch.
The tensor-valued reference model is loaded lazily when one of its public
symbols is first requested, which keeps membership planning usable on login and
CPU-only nodes.
"""

from importlib import import_module

from .arena import (
    DISPATCH_HEADER_NBYTES,
    SLAB_PREAMBLE_MAGIC,
    SLAB_PREAMBLE_VERSION,
    SLAB_PREAMBLE_WIRE_NBYTES,
    UINT32_MAX,
    UINT64_MAX,
    PeerSlab,
    PeerSlabLayout,
    RecordLocation,
    SlabPreamble,
    align_up,
)
from .ll_protocol import PipelineLLArenaLayout, StableSparseTopology

_REFERENCE_EXPORTS = frozenset(
    {
        "DispatchMetadata",
        "DispatchRecord",
        "DistributedReferenceResult",
        "ElasticExpertTopology",
        "ExpertBatch",
        "ExpertOutput",
        "RoutingPlan",
        "distributed_moe_reference",
        "expert_transform",
        "pack_dispatch",
        "process_expert_batches",
        "route_topk",
        "unpack_dispatch",
        "weighted_combine",
    }
)


def __getattr__(name: str):
    if name not in _REFERENCE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.reference"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _REFERENCE_EXPORTS)


__all__ = [
    "DISPATCH_HEADER_NBYTES",
    "SLAB_PREAMBLE_MAGIC",
    "SLAB_PREAMBLE_VERSION",
    "SLAB_PREAMBLE_WIRE_NBYTES",
    "UINT32_MAX",
    "UINT64_MAX",
    "DispatchMetadata",
    "DispatchRecord",
    "DistributedReferenceResult",
    "ElasticExpertTopology",
    "ExpertBatch",
    "ExpertOutput",
    "PeerSlab",
    "PeerSlabLayout",
    "PipelineLLArenaLayout",
    "RecordLocation",
    "RoutingPlan",
    "SlabPreamble",
    "StableSparseTopology",
    "align_up",
    "distributed_moe_reference",
    "expert_transform",
    "pack_dispatch",
    "process_expert_batches",
    "route_topk",
    "unpack_dispatch",
    "weighted_combine",
]
