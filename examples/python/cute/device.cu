// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// UCX uses NVCC-specific checks when defining device qualifiers.
#pragma push_macro("__NVCC__")
#pragma push_macro("__builtin_ia32_prefetch")
#define __NVCC__
#include <ucs/sys/device_code.h>
#pragma pop_macro("__builtin_ia32_prefetch")
#pragma pop_macro("__NVCC__")

// Clang also requires device attributes on UCX's explicit specializations.
// Parse dependencies first so the pragma covers only the UCX CUDA-IPC header.
#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <ucs/type/status.h>
#include <cuda/atomic>
#pragma clang attribute push(__attribute__((device)), apply_to = function)
#include <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#pragma clang attribute pop

#include <gpu/nixl_device.cuh>

static_assert(sizeof(nixlGpuXferStatusH) == 64);
static_assert(alignof(nixlGpuXferStatusH) <= 64);

// These are private linking shims, rebuilt with NIXL; not a stable public ABI.
// Cooperative calls require every lane and uniform arguments, except request
// storage: each calling thread owns one private 64-byte slot.
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ int
complete(nixlGpuXferStatusH &request, nixl_status_t status) {
    while (status == NIXL_IN_PROG) {
        status = nixlGpuGetXferStatus<level>(request);
    }
    return status;
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ int
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    uint64_t bytes,
    unsigned channel,
    uint64_t flags,
    uint64_t request_address) {
    alignas(64) nixlGpuXferStatusH local_request;
    auto *request =
        request_address ? reinterpret_cast<nixlGpuXferStatusH *>(request_address) : &local_request;
    const auto status = nixlPut<level>(src, dst, bytes, channel, flags, request);
    return request_address ? status : complete<level>(*request, status);
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_put(uint64_t local,
              uint64_t remote,
              uint64_t bytes,
              uint64_t local_index,
              uint64_t remote_index,
              uint64_t local_offset,
              uint64_t remote_offset,
              unsigned channel,
              uint64_t flags,
              uint64_t request,
              int level) {
    const nixlMemViewElem src{reinterpret_cast<nixlMemViewH>(local), local_index, local_offset};
    const nixlMemViewElem dst{reinterpret_cast<nixlMemViewH>(remote), remote_index, remote_offset};
    switch (level) {
    case 0:
        return put<nixl_gpu_level_t::THREAD>(src, dst, bytes, channel, flags, request);
    case 1:
        return put<nixl_gpu_level_t::WARP>(src, dst, bytes, channel, flags, request);
    case 2:
        return put<nixl_gpu_level_t::BLOCK>(src, dst, bytes, channel, flags, request);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_progress(uint64_t address, int status, int level) {
    if (status != NIXL_IN_PROG) {
        return status; // Immediate success/error must not poll UCX completion storage.
    }
    auto &request = *reinterpret_cast<nixlGpuXferStatusH *>(address);
    switch (level) {
    case 0:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(request);
    case 1:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::WARP>(request);
    case 2:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK>(request);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ int
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel,
           uint64_t flags,
           uint64_t request_address) {
    alignas(64) nixlGpuXferStatusH request;
    auto *storage =
        request_address ? reinterpret_cast<nixlGpuXferStatusH *>(request_address) : &request;
    const auto status = nixlAtomicAdd<level>(value, counter, channel, flags, storage);
    return request_address ? status : complete<level>(*storage, status);
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_atomic_add(uint64_t value,
                     uint64_t remote,
                     uint64_t index,
                     uint64_t offset,
                     unsigned channel,
                     uint64_t flags,
                     uint64_t request,
                     int level) {
    const nixlMemViewElem counter{reinterpret_cast<nixlMemViewH>(remote), index, offset};
    switch (level) {
    case 0:
        return atomic_add<nixl_gpu_level_t::THREAD>(value, counter, channel, flags, request);
    case 1:
        return atomic_add<nixl_gpu_level_t::WARP>(value, counter, channel, flags, request);
    case 2:
        return atomic_add<nixl_gpu_level_t::BLOCK>(value, counter, channel, flags, request);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

extern "C" __device__ __attribute__((always_inline)) uint64_t
cute_nixl_get_ptr(uint64_t remote, uint64_t index) {
    return reinterpret_cast<uint64_t>(nixlGetPtr(reinterpret_cast<nixlMemViewH>(remote), index));
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_fence() {
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    return NIXL_SUCCESS;
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_wait(uint64_t address, uint64_t expected) {
    uint64_t observed;
    do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(observed)
                     : "l"(address)
                     : "memory");
    } while (observed < expected);
    return 0;
}
