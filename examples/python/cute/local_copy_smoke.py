#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe-DSL smoke test for the NIXL CuTe package scaffold.

This does not use NIXL networking.  It validates the lowest-risk first layer:
`nixl.cute` imports, PyTorch tensors convert to CuTe tensors, and a simple CuTe
kernel can copy data between CUDA tensors.
"""

import argparse

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from nixl.cute import device_api_status


@cute.kernel
def _copy_kernel(
    g_src: cute.Tensor,
    g_dst: cute.Tensor,
    c_dst: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_src.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thr_src = thr_copy.partition_S(g_src[blk_coord])
    thr_dst = thr_copy.partition_D(g_dst[blk_coord])
    thr_crd = thr_copy.partition_S(c_dst[blk_coord])

    pred = cute.make_rmem_tensor(thr_crd.shape, cutlass.Boolean)
    for i in range(cute.size(pred)):
        pred[i] = cute.elem_less(thr_crd[i], shape)

    frag = cute.make_fragment_like(thr_src)
    cute.copy(copy_atom, thr_src, frag, pred=pred)
    cute.copy(copy_atom, frag, thr_dst, pred=pred)


@cute.jit
def copy_tensor(src: cute.Tensor, dst: cute.Tensor, copy_bits: cutlass.Constexpr = 128):
    dtype = src.element_type
    vector_size = copy_bits // dtype.width
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_src = cute.zipped_divide(src, tiler_mn)
    g_dst = cute.zipped_divide(dst, tiler_mn)
    c_dst = cute.zipped_divide(cute.make_identity_tensor(dst.shape), tiler_mn)

    _copy_kernel(g_src, g_dst, c_dst, dst.shape, thr_layout, val_layout).launch(
        grid=[cute.size(g_dst, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run(m: int, n: int) -> None:
    cutlass.cuda.initialize_cuda_context()

    src = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    dst = torch.empty_like(src)

    cute_src = from_dlpack(src).mark_layout_dynamic()
    cute_dst = from_dlpack(dst).mark_layout_dynamic()
    copy_tensor(cute_src, cute_dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src)

    status = device_api_status()
    print("PASS: CuTe local copy")
    print(f"NIXL device API available: {status.available} ({status.reason})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.m, args.n)

