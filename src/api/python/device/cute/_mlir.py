# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal MLIR type adapters used by the CuTe DSL extern ABI."""

from cutlass.cutlass_dsl import ir


class LLVMPtr:
    """Type annotation and matcher for an opaque ``!llvm.ptr`` value."""

    @staticmethod
    def mlir_type():
        return ir.Type.parse("!llvm.ptr")

    @staticmethod
    def __get_mlir_types__():
        return [LLVMPtr.mlir_type()]

    @classmethod
    def isinstance(cls, value):
        return isinstance(value, ir.Value) and value.type == cls.mlir_type()


__all__ = ["LLVMPtr"]
