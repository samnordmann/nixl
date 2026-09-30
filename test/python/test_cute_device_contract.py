# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the actual private shim against a deterministic mock backend.

Checks forwarding and request state transitions, NOT transport/memory ordering.
The runnable GPU examples cover the latter on the tested CUDA-IPC backend.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_request_contract(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required")
    device = (Path(__file__).parents[2] / "examples/python/cute/device.cu").read_text()
    # Only replace CUDA/transport declarations; keep the submission/progress
    # implementation verbatim. PTX fences/waits are deliberately not CPU-tested.
    body = device[device.index("template<nixl_gpu_level_t") :]
    body = body[
        : body.index(
            'extern "C" __device__ __attribute__((always_inline)) int\ncute_nixl_fence'
        )
    ]
    mock = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#define __device__
#define __forceinline__ inline
enum class nixl_gpu_level_t { THREAD, WARP, BLOCK };
using nixl_status_t = int;
constexpr int NIXL_SUCCESS=0, NIXL_IN_PROG=1, NIXL_ERR_INVALID_PARAM=-2;
using nixlMemViewH = void*;
struct nixlGpuXferStatusH { alignas(16) unsigned char storage[64]{}; };
struct nixlMemViewElem { void *mvh; size_t index, offset; };
int submits=0, polls=0, submission=1, polled_level=-1;
uint64_t recorded_value=0;
nixlGpuXferStatusH *recorded_request=nullptr;
template<nixl_gpu_level_t level>
int nixlPut(const nixlMemViewElem &src, const nixlMemViewElem &dst, uint64_t size,
            unsigned channel, uint64_t flags, nixlGpuXferStatusH *request) {
    assert(src.mvh==reinterpret_cast<void*>(11) && dst.mvh==reinterpret_cast<void*>(22));
    assert(src.index==3 && dst.index==4 && src.offset==5 && dst.offset==6);
    assert(size==17 && channel==2 && flags==1);
    recorded_request=request; ++submits; return submission;
}
template<nixl_gpu_level_t level> int nixlGpuGetXferStatus(nixlGpuXferStatusH&) {
    polled_level=int(level); return ++polls < 2 ? NIXL_IN_PROG : NIXL_SUCCESS;
}
template<nixl_gpu_level_t level>
int nixlAtomicAdd(uint64_t value, const nixlMemViewElem &dst, unsigned channel,
                  uint64_t flags, nixlGpuXferStatusH *request) {
    assert(dst.mvh==reinterpret_cast<void*>(22) && dst.index==4 && dst.offset==6);
    assert(channel==2 && flags==1);
    recorded_request=request; recorded_value=value; ++submits; return submission;
}
void *nixlGetPtr(nixlMemViewH view, size_t index) {
    assert(view==reinterpret_cast<void*>(22) && index==4);
    return reinterpret_cast<void*>(1234);
}
"""
    checks = r"""
int main() {
    alignas(64) nixlGpuXferStatusH req;
    auto address=reinterpret_cast<uint64_t>(&req);
    for(int level=0; level<3; level++) {
        polls=0; submission=NIXL_IN_PROG;
        assert(cute_nixl_put(11,22,17,3,4,5,6,2,1,address,level)==NIXL_IN_PROG);
        assert(polls==0 && recorded_request==&req);
        assert(cute_nixl_progress(address,1,level)==NIXL_IN_PROG);
        assert(cute_nixl_progress(address,1,level)==NIXL_SUCCESS);
        assert(polls==2 && polled_level==level);
        polls=0;
        assert(cute_nixl_put(11,22,17,3,4,5,6,2,1,0,level)==NIXL_SUCCESS);
        assert(polls==2 && polled_level==level);
        polls=0;
        assert(cute_nixl_atomic_add(7,22,4,6,2,1,address,level)==NIXL_IN_PROG);
        assert(polls==0 && recorded_request==&req && recorded_value==7);
        assert(cute_nixl_progress(address,1,level)==NIXL_IN_PROG);
        assert(cute_nixl_progress(address,1,level)==NIXL_SUCCESS);
        for(int terminal : {0,-5}) {
            submission=terminal; polls=0;
            assert(cute_nixl_put(11,22,17,3,4,5,6,2,1,address,level)==terminal);
            assert(cute_nixl_put(11,22,17,3,4,5,6,2,1,0,level)==terminal);
            assert(cute_nixl_progress(0,terminal,level)==terminal);
            assert(polls==0);
        }
    }
    assert(cute_nixl_put(11,22,17,3,4,5,6,2,1,address,3)==NIXL_ERR_INVALID_PARAM);
    assert(cute_nixl_get_ptr(22,4)==1234);
}
"""
    source = tmp_path / "contract.cpp"
    source.write_text("#include <initializer_list>\n" + mock + body + checks)
    binary = tmp_path / "contract"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
