/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nixl_device_cute.cuh"

// UCX 1.23's device-code qualifier currently recognizes NVCC and HIPCC but
// not Clang CUDA. Include that one header with its NVCC branch selected so
// UCS_F_DEVICE and the device atomics are emitted for the GPU. Keep the
// compatibility define scoped away from CUDA and C++ standard headers.
#if defined(__clang__) && defined(__CUDA__) && !defined(__NVCC__)
#pragma push_macro("__NVCC__")
#pragma push_macro("__builtin_ia32_prefetch")
#define __NVCC__
#include <ucs/sys/device_code.h>
#pragma pop_macro("__builtin_ia32_prefetch")
#pragma pop_macro("__NVCC__")
#endif

#include <gpu/nixl_device.cuh>

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {
constexpr uint32_t nixl_cute_device_abi_version = 3;
constexpr uint32_t nixl_cute_max_threads_per_block = 1024;
constexpr uint32_t nixl_cute_warp_size = 32;
constexpr uint32_t nixl_cute_timeout_check_interval = 256;
constexpr uint32_t nixl_cute_abort_check_interval = 4096;
constexpr uint64_t nixl_cute_uint64_max = ~uint64_t{0};
static_assert((nixl_cute_timeout_check_interval &
               (nixl_cute_timeout_check_interval - 1)) == 0);
static_assert((nixl_cute_abort_check_interval &
               (nixl_cute_abort_check_interval - 1)) == 0);
static_assert((nixl_cute_abort_check_interval %
               nixl_cute_timeout_check_interval) == 0);

#if defined(NIXL_CUTE_ENABLE_DEVICE_VALIDATION)
#define NIXL_CUTE_RETURN_INVALID_IF(condition) \
    do {                                         \
        if (condition) {                         \
            return static_cast<int32_t>(NIXL_ERR_INVALID_PARAM); \
        }                                        \
    } while (false)
#else
// CuTe validates every literal shape/index/span while tracing. Production
// bitcode trusts the remaining dynamic preconditions so no wrapper-only
// branches survive in a hot operation. A checked artifact is available via
// the Meson option for bring-up and dynamic-argument debugging.
#define NIXL_CUTE_RETURN_INVALID_IF(condition) \
    do {                                         \
    } while (false)
#endif

__device__ __forceinline__ uint64_t
readGlobaltimerNs() {
    uint64_t value;
    // The memory clobber makes this a compiler barrier as well as a hardware
    // timestamp, so loads/stores and inlined transport calls cannot migrate
    // across benchmark or timeout boundaries.
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) : : "memory");
    return value;
}

template<nixl_gpu_level_t level>
__device__ nixlGpuXferStatusH *
requestForCooperativeScope() {
    static_assert(level == nixl_gpu_level_t::WARP);
    __shared__ nixlGpuXferStatusH
        requests[nixl_cute_max_threads_per_block / nixl_cute_warp_size];
    return &requests[threadIdx.x / nixl_cute_warp_size];
}

template<nixl_gpu_level_t level>
__device__ bool
validBlockShape() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return true;
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((blockDim.y != 1) || (blockDim.z != 1) ||
            (blockDim.x > nixl_cute_max_threads_per_block)) {
            return false;
        }
        return (blockDim.x % nixl_cute_warp_size) == 0;
    }
}

template<nixl_gpu_level_t level>
__device__ int32_t
waitForRequest(nixlGpuXferStatusH &request, nixl_status_t status) {
    const bool progressed = status == NIXL_IN_PROG;
    while (status == NIXL_IN_PROG) {
        status = nixlGpuGetXferStatus<level>(request);
    }
    if constexpr (level == nixl_gpu_level_t::WARP) {
        // A synchronous UCX cooperative operation already completes its
        // level-wide rendezvous before returning UCS_OK (CUDA IPC, for example,
        // calls uct_cuda_ipc_level_sync()). Do not pay a second barrier on that
        // hot path. Preserve an explicit converged return after request polling.
        if (progressed) {
            __syncwarp();
        }
    }
    return static_cast<int32_t>(status);
}

template<nixl_gpu_level_t level>
__device__ int32_t
putWait(void *local_view,
        uint32_t local_index,
        uint64_t local_offset,
        void *remote_view,
        uint32_t remote_index,
        uint64_t remote_offset,
        uint64_t size,
        uint32_t channel,
        uint64_t flags) {
    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<level>() || (local_view == nullptr) ||
        (remote_view == nullptr) || (size == 0) || (flags != 0) ||
        (local_offset > (nixl_cute_uint64_max - size)) ||
        (remote_offset > (nixl_cute_uint64_max - size)));
    const nixlMemViewElem source{
        local_view, static_cast<size_t>(local_index), static_cast<size_t>(local_offset)};
    const nixlMemViewElem destination{
        remote_view, static_cast<size_t>(remote_index), static_cast<size_t>(remote_offset)};
    // UCX initializes the request payload on post.  Do not zero 64 bytes in
    // this per-operation hot path (the native Device API uses it uninitialized
    // as well).
    nixlGpuXferStatusH thread_request;
    auto *request = &thread_request;
    if constexpr (level != nixl_gpu_level_t::THREAD) {
        request = requestForCooperativeScope<level>();
    }
    const auto status = nixlPut<level>(source,
                                      destination,
                                      static_cast<size_t>(size),
                                      channel,
                                      flags,
                                      request);
    return waitForRequest<level>(*request, status);
}

template<nixl_gpu_level_t level>
__device__ int32_t
putPost(void *local_view,
        uint32_t local_index,
        uint64_t local_offset,
        void *remote_view,
        uint32_t remote_index,
        uint64_t remote_offset,
        uint64_t size,
        uint32_t channel,
        uint64_t flags) {
    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<level>() || (local_view == nullptr) ||
        (remote_view == nullptr) || (size == 0) ||
        ((flags & ~nixl_gpu_flags::defer) != 0) ||
        (local_offset > (nixl_cute_uint64_max - size)) ||
        (remote_offset > (nixl_cute_uint64_max - size)));
    const nixlMemViewElem source{
        local_view, static_cast<size_t>(local_index), static_cast<size_t>(local_offset)};
    const nixlMemViewElem destination{
        remote_view, static_cast<size_t>(remote_index), static_cast<size_t>(remote_offset)};
    return static_cast<int32_t>(nixlPut<level>(source,
                                               destination,
                                               static_cast<size_t>(size),
                                               channel,
                                               flags,
                                               nullptr));
}

template<nixl_gpu_level_t level>
__device__ int32_t
atomicAddWait(uint64_t value,
              void *remote_view,
              uint32_t remote_index,
              uint64_t remote_offset,
              uint32_t channel,
              uint64_t flags) {
    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<level>() || (remote_view == nullptr) ||
        ((remote_offset & (alignof(uint64_t) - 1)) != 0) ||
        (remote_offset > (nixl_cute_uint64_max - sizeof(uint64_t))) ||
        (flags != 0));
    const nixlMemViewElem counter{
        remote_view, static_cast<size_t>(remote_index), static_cast<size_t>(remote_offset)};
    nixlGpuXferStatusH thread_request;
    auto *request = &thread_request;
    if constexpr (level != nixl_gpu_level_t::THREAD) {
        request = requestForCooperativeScope<level>();
    }
    const auto status = nixlAtomicAdd<level>(value, counter, channel, flags, request);
    return waitForRequest<level>(*request, status);
}

template<nixl_gpu_level_t level>
__device__ int32_t
atomicAddPost(uint64_t value,
              void *remote_view,
              uint32_t remote_index,
              uint64_t remote_offset,
              uint32_t channel,
              uint64_t flags) {
    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<level>() || (remote_view == nullptr) ||
        ((remote_offset & (alignof(uint64_t) - 1)) != 0) ||
        (remote_offset > (nixl_cute_uint64_max - sizeof(uint64_t))) ||
        ((flags & ~nixl_gpu_flags::defer) != 0));
    const nixlMemViewElem counter{
        remote_view, static_cast<size_t>(remote_index), static_cast<size_t>(remote_offset)};
    return static_cast<int32_t>(
        nixlAtomicAdd<level>(value, counter, channel, flags, nullptr));
}

__device__ __forceinline__ uint64_t
loadAcquireSystemU64(uint64_t address) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint64_t
loadAcquireGpuU64(uint64_t address) {
    uint64_t value;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint64_t
loadRelaxedSystemU64(uint64_t address) {
    uint64_t value;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint64_t
loadRelaxedGpuU64(uint64_t address) {
    uint64_t value;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

template<bool system_scope>
__device__ __forceinline__ uint64_t
loadAcquireScopedU64(uint64_t address) {
    if constexpr (system_scope) {
        return loadAcquireSystemU64(address);
    } else {
        return loadAcquireGpuU64(address);
    }
}

template<bool system_scope>
__device__ __forceinline__ uint64_t
loadRelaxedScopedU64(uint64_t address) {
    if constexpr (system_scope) {
        return loadRelaxedSystemU64(address);
    } else {
        return loadRelaxedGpuU64(address);
    }
}

template<nixl_gpu_level_t level, bool system_scope = true>
__device__ uint64_t
waitAcquireScopedU64(uint64_t address, uint64_t least) {
    uint64_t observed = 0;
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        observed = loadAcquireScopedU64<system_scope>(address);
        while (observed < least) {
            do {
                observed = loadRelaxedScopedU64<system_scope>(address);
            } while (observed < least);
            observed = loadAcquireScopedU64<system_scope>(address);
        }
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((threadIdx.x & (nixl_cute_warp_size - 1)) == 0) {
            observed = loadAcquireScopedU64<system_scope>(address);
            while (observed < least) {
                do {
                    observed = loadRelaxedScopedU64<system_scope>(address);
                } while (observed < least);
                observed = loadAcquireScopedU64<system_scope>(address);
            }
        }
        observed = __shfl_sync(0xffffffffu, observed, 0);
        // __shfl_sync transfers the value but is not a memory fence. Propagate
        // lane zero's acquire edge to the payload-reading lanes through a warp
        // barrier before returning.
        __syncwarp();
    }
    return observed;
}

template<bool observe_peer_abort>
__device__ __forceinline__ bool
abortRequested(uint64_t local_abort_address,
               uint64_t local_abort_least,
               uint64_t peer_abort_address,
               uint64_t peer_abort_least) {
    const uint64_t local_candidate = loadRelaxedGpuU64(local_abort_address);
    if (local_candidate >= local_abort_least &&
        loadAcquireGpuU64(local_abort_address) >= local_abort_least) {
        return true;
    }
    if constexpr (observe_peer_abort) {
        const uint64_t peer_candidate = loadRelaxedSystemU64(peer_abort_address);
        if (peer_candidate >= peer_abort_least &&
            loadAcquireSystemU64(peer_abort_address) >= peer_abort_least) {
            return true;
        }
    }
    return false;
}

template<nixl_gpu_level_t level,
         bool system_scope,
         bool observe_peer_abort = false>
__device__ uint64_t
waitAcquireScopedU64OrAbort(uint64_t address,
                            uint64_t least,
                            uint64_t abort_address,
                            uint64_t abort_least,
                            uint64_t peer_abort_address = 0,
                            uint64_t peer_abort_least = 0) {
    constexpr uint32_t abort_interval =
        system_scope ? nixl_cute_abort_check_interval
                     : nixl_cute_timeout_check_interval;
    uint64_t observed = 0;
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        observed = loadAcquireScopedU64<system_scope>(address);
        uint32_t polls = 0;
        while (observed < least) {
            const uint64_t candidate = loadRelaxedScopedU64<system_scope>(address);
            if (candidate >= least) {
                observed = loadAcquireScopedU64<system_scope>(address);
                continue;
            }
            ++polls;
            if ((polls & (abort_interval - 1)) == 0) {
                if (abortRequested<observe_peer_abort>(
                        abort_address,
                        abort_least,
                        peer_abort_address,
                        peer_abort_least)) {
                    break;
                }
            }
        }
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((threadIdx.x & (nixl_cute_warp_size - 1)) == 0) {
            observed = loadAcquireScopedU64<system_scope>(address);
            uint32_t polls = 0;
            while (observed < least) {
                const uint64_t candidate =
                    loadRelaxedScopedU64<system_scope>(address);
                if (candidate >= least) {
                    observed = loadAcquireScopedU64<system_scope>(address);
                    continue;
                }
                ++polls;
                if ((polls & (abort_interval - 1)) == 0) {
                    if (abortRequested<observe_peer_abort>(
                            abort_address,
                            abort_least,
                            peer_abort_address,
                            peer_abort_least)) {
                        break;
                    }
                }
            }
        }
        observed = __shfl_sync(0xffffffffu, observed, 0);
        __syncwarp();
    }
    return observed;
}

template<nixl_gpu_level_t level>
__device__ uint64_t
waitAcquireSystemU64Until(uint64_t address, uint64_t least, uint64_t deadline_ns) {
    uint64_t observed = 0;
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        observed = loadAcquireSystemU64(address);
        if (observed < least) {
            // Keep the ready path to one acquire load and use relaxed loads for
            // failed polls. Confirm readiness and timeout with a final acquire
            // so every returned value preserves the API's acquire semantics.
            uint32_t polls = 1;
            while (true) {
                const uint64_t candidate = loadRelaxedSystemU64(address);
                if (candidate >= least) {
                    observed = loadAcquireSystemU64(address);
                    if (observed >= least) {
                        break;
                    }
                }
                ++polls;
                if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                    readGlobaltimerNs() >= deadline_ns) {
                    observed = loadAcquireSystemU64(address);
                    break;
                }
            }
        }
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((threadIdx.x & (nixl_cute_warp_size - 1)) == 0) {
            observed = loadAcquireSystemU64(address);
            if (observed < least) {
                uint32_t polls = 1;
                while (true) {
                    const uint64_t candidate = loadRelaxedSystemU64(address);
                    if (candidate >= least) {
                        observed = loadAcquireSystemU64(address);
                        if (observed >= least) {
                            break;
                        }
                    }
                    ++polls;
                    if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                        readGlobaltimerNs() >= deadline_ns) {
                        observed = loadAcquireSystemU64(address);
                        break;
                    }
                }
            }
        }
        observed = __shfl_sync(0xffffffffu, observed, 0);
        __syncwarp();
    }
    return observed;
}

template<nixl_gpu_level_t level, bool system_scope = true>
__device__ uint64_t
waitAcquireScopedU64For(uint64_t address, uint64_t least, uint64_t timeout_ns) {
    uint64_t observed = 0;
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        observed = loadAcquireScopedU64<system_scope>(address);
        if (observed < least) {
            // Defer the timer read until the first miss. The common ready path
            // is exactly one acquire load, while unsigned subtraction remains
            // correct if %globaltimer wraps during a very long-lived kernel.
            const uint64_t started = readGlobaltimerNs();
            uint32_t polls = 1;
            while (true) {
                const uint64_t candidate =
                    loadRelaxedScopedU64<system_scope>(address);
                if (candidate >= least) {
                    observed = loadAcquireScopedU64<system_scope>(address);
                    if (observed >= least) {
                        break;
                    }
                }
                ++polls;
                if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                    (readGlobaltimerNs() - started) >= timeout_ns) {
                    // Preserve the API contract that even a timeout returns
                    // an acquire-loaded value.
                    observed = loadAcquireScopedU64<system_scope>(address);
                    break;
                }
            }
        }
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((threadIdx.x & (nixl_cute_warp_size - 1)) == 0) {
            observed = loadAcquireScopedU64<system_scope>(address);
            if (observed < least) {
                const uint64_t started = readGlobaltimerNs();
                uint32_t polls = 1;
                while (true) {
                    const uint64_t candidate =
                        loadRelaxedScopedU64<system_scope>(address);
                    if (candidate >= least) {
                        observed = loadAcquireScopedU64<system_scope>(address);
                        if (observed >= least) {
                            break;
                        }
                    }
                    ++polls;
                    if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                        (readGlobaltimerNs() - started) >= timeout_ns) {
                        observed =
                            loadAcquireScopedU64<system_scope>(address);
                        break;
                    }
                }
            }
        }
        observed = __shfl_sync(0xffffffffu, observed, 0);
        __syncwarp();
    }
    return observed;
}

template<nixl_gpu_level_t level>
__device__ uint64_t
waitAcquireSystemU64ForOrAbort(uint64_t address,
                               uint64_t least,
                               uint64_t abort_address,
                               uint64_t abort_least,
                               uint64_t timeout_ns) {
    uint64_t observed = 0;
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        observed = loadAcquireSystemU64(address);
        if (observed < least) {
            // The target-ready path above is still exactly one acquire load.
            // Abort is a local GPU word and is sampled much less frequently
            // than the remote target so an ordinary peer delay does not add
            // material interconnect traffic or timer pressure.
            const uint64_t started = readGlobaltimerNs();
            uint32_t polls = 1;
            while (true) {
                const uint64_t candidate = loadRelaxedSystemU64(address);
                if (candidate >= least) {
                    observed = loadAcquireSystemU64(address);
                    if (observed >= least) {
                        break;
                    }
                }
                ++polls;
                if ((polls & (nixl_cute_abort_check_interval - 1)) == 0) {
                    const uint64_t abort_candidate =
                        loadRelaxedGpuU64(abort_address);
                    if (abort_candidate >= abort_least &&
                        loadAcquireGpuU64(abort_address) >= abort_least) {
                        break;
                    }
                }
                if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                    (readGlobaltimerNs() - started) >= timeout_ns) {
                    observed = loadAcquireSystemU64(address);
                    break;
                }
            }
        }
    } else {
        static_assert(level == nixl_gpu_level_t::WARP);
        if ((threadIdx.x & (nixl_cute_warp_size - 1)) == 0) {
            observed = loadAcquireSystemU64(address);
            if (observed < least) {
                const uint64_t started = readGlobaltimerNs();
                uint32_t polls = 1;
                while (true) {
                    const uint64_t candidate = loadRelaxedSystemU64(address);
                    if (candidate >= least) {
                        observed = loadAcquireSystemU64(address);
                        if (observed >= least) {
                            break;
                        }
                    }
                    ++polls;
                    if ((polls & (nixl_cute_abort_check_interval - 1)) == 0) {
                        const uint64_t abort_candidate =
                            loadRelaxedGpuU64(abort_address);
                        if (abort_candidate >= abort_least &&
                            loadAcquireGpuU64(abort_address) >= abort_least) {
                            break;
                        }
                    }
                    if ((polls & (nixl_cute_timeout_check_interval - 1)) == 0 &&
                        (readGlobaltimerNs() - started) >= timeout_ns) {
                        observed = loadAcquireSystemU64(address);
                        break;
                    }
                }
            }
        }
        observed = __shfl_sync(0xffffffffu, observed, 0);
        __syncwarp();
    }
    return observed;
}

struct alignas(16) NixlCuteInt4 {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
};

static_assert(sizeof(NixlCuteInt4) == 16);

__device__ __forceinline__ NixlCuteInt4
loadNoAllocate(const NixlCuteInt4 *address) {
    NixlCuteInt4 value;
    asm volatile("ld.global.L1::no_allocate.L2::256B.v4.s32 "
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ NixlCuteInt4
loadNcNoAllocate(const NixlCuteInt4 *address) {
    NixlCuteInt4 value;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 "
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address)
                 : "memory");
    return value;
}

template<bool source_is_readonly>
__device__ __forceinline__ NixlCuteInt4
loadMapped(const NixlCuteInt4 *address) {
    if constexpr (source_is_readonly) {
        return loadNcNoAllocate(address);
    } else {
        return loadNoAllocate(address);
    }
}

__device__ __forceinline__ void
storeNoAllocate(NixlCuteInt4 *address, const NixlCuteInt4 &value) {
    asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address),
                   "r"(value.x),
                   "r"(value.y),
                   "r"(value.z),
                   "r"(value.w)
                 : "memory");
}

/**
 * Copy one peer span cooperatively with a converged warp.
 *
 * Keep the load phase ahead of the store phase for eight vectors per lane.
 * This is the same latency-hiding shape used by the production elastic EP
 * kernel, while still handling any positive 16-byte multiple in the tail.
 */
template<bool source_is_readonly>
__device__ __forceinline__ int32_t
copyWarp(uint64_t source_address, uint64_t destination_address, uint64_t size) {
    constexpr uint32_t full_warp_mask = 0xffffffffu;
    constexpr uint64_t vector_bytes = sizeof(NixlCuteInt4);
    constexpr uint64_t unroll = 8;
    constexpr uint64_t vectors_per_iteration = nixl_cute_warp_size * unroll;

    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<nixl_gpu_level_t::WARP>() || (source_address == 0) ||
        (destination_address == 0) || (size == 0) ||
        (((source_address | destination_address | size) & (vector_bytes - 1)) !=
         0) ||
        (source_address > (nixl_cute_uint64_max - size)) ||
        (destination_address > (nixl_cute_uint64_max - size)));

    const uint32_t lane = threadIdx.x & (nixl_cute_warp_size - 1);
    const auto *const __restrict__ source =
        reinterpret_cast<const NixlCuteInt4 *>(source_address);
    auto *const __restrict__ destination =
        reinterpret_cast<NixlCuteInt4 *>(destination_address);
    const uint64_t num_vectors = size / vector_bytes;
    const uint64_t full_vectors =
        (num_vectors / vectors_per_iteration) * vectors_per_iteration;

    for (uint64_t i = lane; i < full_vectors; i += vectors_per_iteration) {
        NixlCuteInt4 values[unroll];
#pragma unroll
        for (uint32_t j = 0; j < unroll; ++j) {
            values[j] =
                loadMapped<source_is_readonly>(source + i + j * nixl_cute_warp_size);
        }
#pragma unroll
        for (uint32_t j = 0; j < unroll; ++j) {
            storeNoAllocate(destination + i + j * nixl_cute_warp_size, values[j]);
        }
    }

    const uint64_t tail = full_vectors + lane;
    NixlCuteInt4 tail_values[unroll];
#pragma unroll
    for (uint32_t j = 0; j < unroll; ++j) {
        if ((tail + j * nixl_cute_warp_size) < num_vectors) {
            tail_values[j] = loadMapped<source_is_readonly>(
                source + tail + j * nixl_cute_warp_size);
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < unroll; ++j) {
        if ((tail + j * nixl_cute_warp_size) < num_vectors) {
            storeNoAllocate(destination + tail + j * nixl_cute_warp_size,
                            tail_values[j]);
        }
    }

    // Orders every lane's payload stores before any lane can publish or reuse
    // the source.  The API deliberately performs no status/signal write.
    __syncwarp(full_warp_mask);
    return static_cast<int32_t>(NIXL_SUCCESS);
}

template<bool source_is_readonly>
__device__ __forceinline__ int32_t
mappedCopyWarp(uint64_t source_address,
               void *remote_view,
               uint32_t remote_index,
               uint64_t remote_offset,
               uint64_t size) {
    constexpr uint32_t full_warp_mask = 0xffffffffu;
    constexpr uint64_t vector_bytes = sizeof(NixlCuteInt4);

    NIXL_CUTE_RETURN_INVALID_IF(
        !validBlockShape<nixl_gpu_level_t::WARP>() || (remote_view == nullptr) ||
        ((remote_offset & (vector_bytes - 1)) != 0));

    const uint32_t lane = threadIdx.x & (nixl_cute_warp_size - 1);
    uint64_t destination_base = 0;
    if (lane == 0) {
        destination_base = reinterpret_cast<uint64_t>(
            nixlGetPtr(remote_view, static_cast<size_t>(remote_index)));
    }
    destination_base = __shfl_sync(full_warp_mask, destination_base, 0);
    if (destination_base == 0) {
        // A null mapping is an expected transport choice. Returning a uniform
        // status lets the converged caller issue the ordinary NIXL PUT path.
        return static_cast<int32_t>(NIXL_ERR_NOT_SUPPORTED);
    }
    NIXL_CUTE_RETURN_INVALID_IF(
        (destination_base > (nixl_cute_uint64_max - remote_offset)));
    return copyWarp<source_is_readonly>(
        source_address, destination_base + remote_offset, size);
}
} // namespace

#if defined(NIXL_CUTE_DISABLE_FORCEINLINE)
#define NIXL_CUTE_EXPORT extern "C" __device__ __attribute__((noinline))
#else
// CuTe links this module before its final device optimization.  Marking the
// public ABI symbol always-inline removes a device CALL and, without ``used``,
// lets its bitcode importer discard every unreferenced NIXL export.  The
// builder's public-API internalize list preserves these definitions in the
// standalone artifact.  The opt-in noinline form exists only for A/B checks.
#define NIXL_CUTE_EXPORT extern "C" __device__ __attribute__((always_inline))
#endif

NIXL_CUTE_EXPORT uint32_t
nixl_cute_abi_version() {
    return nixl_cute_device_abi_version;
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_put_thread_wait(void *local_view,
                          uint32_t local_index,
                          uint64_t local_offset,
                          void *remote_view,
                          uint32_t remote_index,
                          uint64_t remote_offset,
                          uint64_t size,
                          uint32_t channel,
                          uint64_t flags) {
    return putWait<nixl_gpu_level_t::THREAD>(local_view,
                                             local_index,
                                             local_offset,
                                             remote_view,
                                             remote_index,
                                             remote_offset,
                                             size,
                                             channel,
                                             flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_put_warp_wait(void *local_view,
                        uint32_t local_index,
                        uint64_t local_offset,
                        void *remote_view,
                        uint32_t remote_index,
                        uint64_t remote_offset,
                        uint64_t size,
                        uint32_t channel,
                        uint64_t flags) {
    return putWait<nixl_gpu_level_t::WARP>(local_view,
                                           local_index,
                                           local_offset,
                                           remote_view,
                                           remote_index,
                                           remote_offset,
                                           size,
                                           channel,
                                           flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_put_thread_post(void *local_view,
                          uint32_t local_index,
                          uint64_t local_offset,
                          void *remote_view,
                          uint32_t remote_index,
                          uint64_t remote_offset,
                          uint64_t size,
                          uint32_t channel,
                          uint64_t flags) {
    return putPost<nixl_gpu_level_t::THREAD>(local_view,
                                             local_index,
                                             local_offset,
                                             remote_view,
                                             remote_index,
                                             remote_offset,
                                             size,
                                             channel,
                                             flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_put_warp_post(void *local_view,
                        uint32_t local_index,
                        uint64_t local_offset,
                        void *remote_view,
                        uint32_t remote_index,
                        uint64_t remote_offset,
                        uint64_t size,
                        uint32_t channel,
                        uint64_t flags) {
    return putPost<nixl_gpu_level_t::WARP>(local_view,
                                           local_index,
                                           local_offset,
                                           remote_view,
                                           remote_index,
                                           remote_offset,
                                           size,
                                           channel,
                                           flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_add_thread_wait(uint64_t value,
                                 void *remote_view,
                                 uint32_t remote_index,
                                 uint64_t remote_offset,
                                 uint32_t channel,
                                 uint64_t flags) {
    return atomicAddWait<nixl_gpu_level_t::THREAD>(
        value, remote_view, remote_index, remote_offset, channel, flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_add_warp_wait(uint64_t value,
                               void *remote_view,
                               uint32_t remote_index,
                               uint64_t remote_offset,
                               uint32_t channel,
                               uint64_t flags) {
    return atomicAddWait<nixl_gpu_level_t::WARP>(
        value, remote_view, remote_index, remote_offset, channel, flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_add_thread_post(uint64_t value,
                                 void *remote_view,
                                 uint32_t remote_index,
                                 uint64_t remote_offset,
                                 uint32_t channel,
                                 uint64_t flags) {
    return atomicAddPost<nixl_gpu_level_t::THREAD>(
        value, remote_view, remote_index, remote_offset, channel, flags);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_add_warp_post(uint64_t value,
                               void *remote_view,
                               uint32_t remote_index,
                               uint64_t remote_offset,
                               uint32_t channel,
                               uint64_t flags) {
    return atomicAddPost<nixl_gpu_level_t::WARP>(
        value, remote_view, remote_index, remote_offset, channel, flags);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_globaltimer_ns() {
    return readGlobaltimerNs();
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_load_acquire_system_u64(uint64_t address) {
    return loadAcquireSystemU64(address);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_load_acquire_gpu_u64(uint64_t address) {
    return loadAcquireGpuU64(address);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_thread(uint64_t address, uint64_t least) {
    return waitAcquireScopedU64<nixl_gpu_level_t::THREAD>(address, least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_warp(uint64_t address, uint64_t least) {
    return waitAcquireScopedU64<nixl_gpu_level_t::WARP>(address, least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_thread(uint64_t address, uint64_t least) {
    return waitAcquireScopedU64<nixl_gpu_level_t::THREAD, false>(address, least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_warp(uint64_t address, uint64_t least) {
    return waitAcquireScopedU64<nixl_gpu_level_t::WARP, false>(address, least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_or_abort_thread(uint64_t address,
                                                   uint64_t least,
                                                   uint64_t abort_address,
                                                   uint64_t abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::THREAD, true>(
        address, least, abort_address, abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_or_abort_warp(uint64_t address,
                                                 uint64_t least,
                                                 uint64_t abort_address,
                                                 uint64_t abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::WARP, true>(
        address, least, abort_address, abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_or_aborts_thread(
    uint64_t address,
    uint64_t least,
    uint64_t local_abort_address,
    uint64_t local_abort_least,
    uint64_t peer_abort_address,
    uint64_t peer_abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::THREAD, true, true>(
        address,
        least,
        local_abort_address,
        local_abort_least,
        peer_abort_address,
        peer_abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_or_aborts_warp(
    uint64_t address,
    uint64_t least,
    uint64_t local_abort_address,
    uint64_t local_abort_least,
    uint64_t peer_abort_address,
    uint64_t peer_abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::WARP, true, true>(
        address,
        least,
        local_abort_address,
        local_abort_least,
        peer_abort_address,
        peer_abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_or_abort_thread(uint64_t address,
                                                uint64_t least,
                                                uint64_t abort_address,
                                                uint64_t abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::THREAD, false>(
        address, least, abort_address, abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_or_abort_warp(uint64_t address,
                                              uint64_t least,
                                              uint64_t abort_address,
                                              uint64_t abort_least) {
    return waitAcquireScopedU64OrAbort<nixl_gpu_level_t::WARP, false>(
        address, least, abort_address, abort_least);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_thread_until(uint64_t address,
                                                uint64_t least,
                                                uint64_t deadline_ns) {
    return waitAcquireSystemU64Until<nixl_gpu_level_t::THREAD>(
        address, least, deadline_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_warp_until(uint64_t address,
                                              uint64_t least,
                                              uint64_t deadline_ns) {
    return waitAcquireSystemU64Until<nixl_gpu_level_t::WARP>(
        address, least, deadline_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_thread_for(uint64_t address,
                                              uint64_t least,
                                              uint64_t timeout_ns) {
    return waitAcquireScopedU64For<nixl_gpu_level_t::THREAD>(
        address, least, timeout_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_warp_for(uint64_t address,
                                            uint64_t least,
                                            uint64_t timeout_ns) {
    return waitAcquireScopedU64For<nixl_gpu_level_t::WARP>(
        address, least, timeout_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_thread_for_or_abort(
    uint64_t address,
    uint64_t least,
    uint64_t abort_address,
    uint64_t abort_least,
    uint64_t timeout_ns) {
    return waitAcquireSystemU64ForOrAbort<nixl_gpu_level_t::THREAD>(
        address, least, abort_address, abort_least, timeout_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_system_u64_warp_for_or_abort(
    uint64_t address,
    uint64_t least,
    uint64_t abort_address,
    uint64_t abort_least,
    uint64_t timeout_ns) {
    return waitAcquireSystemU64ForOrAbort<nixl_gpu_level_t::WARP>(
        address, least, abort_address, abort_least, timeout_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_thread_for(uint64_t address,
                                           uint64_t least,
                                           uint64_t timeout_ns) {
    return waitAcquireScopedU64For<nixl_gpu_level_t::THREAD, false>(
        address, least, timeout_ns);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_wait_acquire_gpu_u64_warp_for(uint64_t address,
                                         uint64_t least,
                                         uint64_t timeout_ns) {
    return waitAcquireScopedU64For<nixl_gpu_level_t::WARP, false>(
        address, least, timeout_ns);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_store_release_system_u64(uint64_t address, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(address), "l"(value)
                 : "memory");
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_store_release_gpu_u64(uint64_t address, uint64_t value) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;"
                 :
                 : "l"(address), "l"(value)
                 : "memory");
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT uint64_t
nixl_cute_atomic_add_release_gpu_u64(uint64_t address, uint64_t value) {
    uint64_t prior;
    asm volatile("atom.add.release.gpu.global.u64 %0, [%1], %2;"
                 : "=l"(prior)
                 : "l"(address), "l"(value)
                 : "memory");
    return prior;
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_max_release_gpu_u64(uint64_t address, uint64_t value) {
    uint64_t prior;
    asm volatile("atom.max.release.gpu.global.u64 %0, [%1], %2;"
                 : "=l"(prior)
                 : "l"(address), "l"(value)
                 : "memory");
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_atomic_max_release_system_u64(uint64_t address, uint64_t value) {
    uint64_t prior;
    asm volatile("atom.max.release.sys.global.u64 %0, [%1], %2;"
                 : "=l"(prior)
                 : "l"(address), "l"(value)
                 : "memory");
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_compare_exchange_status_gpu_i32(uint64_t address, int32_t value) {
    uint32_t prior;
    const uint32_t desired = static_cast<uint32_t>(value);
    asm volatile("atom.cas.release.gpu.global.b32 %0, [%1], %2, %3;"
                 : "=r"(prior)
                 : "l"(address), "r"(uint32_t{0}), "r"(desired)
                 : "memory");
    return static_cast<int32_t>(prior);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_sync_grid() {
    cg::this_grid().sync();
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_fence_release_system() {
    // Make source data produced by this kernel visible to a following
    // device-initiated NIC/transport read. The memory clobber also prevents
    // the compiler from moving source writes below the fence.
    asm volatile("fence.release.sys;" : : : "memory");
    return static_cast<int32_t>(NIXL_SUCCESS);
}

NIXL_CUTE_EXPORT void *
nixl_cute_get_ptr(void *remote_view, uint32_t remote_index) {
    return nixlGetPtr(remote_view, static_cast<size_t>(remote_index));
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_mapped_copy_warp_ptr(uint64_t source_address,
                               uint64_t destination_address,
                               uint64_t size) {
    return copyWarp<false>(source_address, destination_address, size);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_mapped_copy_warp_ptr_readonly(uint64_t source_address,
                                        uint64_t destination_address,
                                        uint64_t size) {
    return copyWarp<true>(source_address, destination_address, size);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_mapped_copy_warp(uint64_t source_address,
                           void *remote_view,
                           uint32_t remote_index,
                           uint64_t remote_offset,
                           uint64_t size) {
    return mappedCopyWarp<false>(
        source_address, remote_view, remote_index, remote_offset, size);
}

NIXL_CUTE_EXPORT int32_t
nixl_cute_mapped_copy_warp_readonly(uint64_t source_address,
                                    void *remote_view,
                                    uint32_t remote_index,
                                    uint64_t remote_offset,
                                    uint64_t size) {
    return mappedCopyWarp<true>(
        source_address, remote_view, remote_index, remote_offset, size);
}

#undef NIXL_CUTE_EXPORT
#undef NIXL_CUTE_RETURN_INVALID_IF
