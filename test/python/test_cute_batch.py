# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual private batch body against fake statuses, without CUDA/UCX."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_batch_submission_and_request_lifetime(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is needed for the fake-backend check")
    source = (Path(__file__).parents[2] / "examples/python/cute/device.cu").read_text()

    def body(name):
        start = source.index(f"{name}(")
        return "int\n" + source[start : source.index("\n}", start) + 2]

    harness = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
using nixl_status_t = int;
using nixlMemViewH = void*;
constexpr int NIXL_SUCCESS = 0, NIXL_IN_PROG = 1, NIXL_ERR_INVALID_PARAM = -2;
struct nixlGpuXferStatusH { alignas(16) int id = -1; char padding[60]; };
struct nixlMemViewElem { nixlMemViewH mvh; size_t index, offset; };
int issued, count, submit[8], finish[8], polls[8];
int nixlPut(nixlMemViewElem src, nixlMemViewElem dst, size_t bytes,
            unsigned channel, uint64_t flags, nixlGpuXferStatusH* request) {
    int i = issued++;
    assert(reinterpret_cast<uintptr_t>(request) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(src.mvh) == 0x1000);
    assert(reinterpret_cast<uintptr_t>(dst.mvh) == 0x2000 + i);
    assert(src.index == 0 && dst.index == 0 && bytes == 64);
    assert(src.offset == i * 128 && dst.offset == i * 256);
    assert(channel == 0 && flags == 0);
    if (submit[i] == NIXL_IN_PROG) request->id = i;
    return submit[i];
}
int nixlGpuGetXferStatus(nixlGpuXferStatusH& request) {
    assert(issued == count);  // No completion polling before all submissions.
    assert(request.id >= 0);  // Never poll immediate-success/error requests.
    return ++polls[request.id] == 1 ? NIXL_IN_PROG : finish[request.id];
}
"""
    harness += body("complete") + "\n" + body("cute_nixl_put_batch")
    harness += r"""
int main() {
    assert(cute_nixl_put_batch(0, 0, 64, -1) == NIXL_ERR_INVALID_PARAM);
    assert(cute_nixl_put_batch(0, 0, 64, 9) == NIXL_ERR_INVALID_PARAM);
    assert(cute_nixl_put_batch(0, 0, 64, 0) == NIXL_SUCCESS);
    for (count = 1; count <= 8; ++count) {
        for (int scenario = 0; scenario < 4; ++scenario) {
            uint64_t plan[24];
            issued = 0;
            for (int i = 0; i < count; ++i) {
                plan[3*i] = 0x2000 + i;
                plan[3*i+1] = i * 128;
                plan[3*i+2] = i * 256;
                submit[i] = scenario == 0 ? 0 : (i % 2 == 0 ? 1 : 0);
                finish[i] = polls[i] = 0;
            }
            if (scenario == 2) submit[0] = -3;  // Submit error, other requests still drain.
            if (scenario == 3) finish[0] = -4;  // Completion error, others still drain.
            int rc = cute_nixl_put_batch(0x1000, reinterpret_cast<uint64_t>(plan), 64, count);
            assert(rc == (scenario == 2 ? -3 : scenario == 3 ? -4 : 0));
            assert(issued == count);
            for (int i = 0; i < count; ++i) assert(polls[i] == (submit[i] == 1 ? 2 : 0));
        }
    }
}
"""
    cpp, executable = tmp_path / "batch.cpp", tmp_path / "batch"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True)
