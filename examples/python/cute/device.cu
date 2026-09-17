// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// UCX 1.23 recognizes NVCC, but not Clang, when defining device qualifiers.
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

// These are private linking shims, rebuilt with NIXL; not a stable public ABI.
// Each call is issued by one thread. Completion waits stay on that GPU thread.
__device__ __forceinline__ int
complete(nixlGpuXferStatusH &request, nixl_status_t status) {
    while (status == NIXL_IN_PROG) {
        status = nixlGpuGetXferStatus(request);
    }
    return status;
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_put(uint64_t local,
              uint64_t remote,
              uint64_t bytes,
              uint64_t local_offset,
              uint64_t remote_offset) {
    alignas(64) nixlGpuXferStatusH request;
    const nixlMemViewElem src{reinterpret_cast<nixlMemViewH>(local), 0, local_offset};
    const nixlMemViewElem dst{reinterpret_cast<nixlMemViewH>(remote), 0, remote_offset};
    return complete(request, nixlPut(src, dst, bytes, 0, 0, &request));
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_signal(uint64_t remote, uint64_t offset) {
    // Publish the preceding PUT before the counter, including CUDA-IPC stores.
    __threadfence_system();
    alignas(64) nixlGpuXferStatusH request;
    const nixlMemViewElem counter{reinterpret_cast<nixlMemViewH>(remote), 1, offset};
    return complete(request, nixlAtomicAdd(1, counter, 0, 0, &request));
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_wait(uint64_t address, uint64_t expected) {
    uint64_t observed;
    do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(observed) : "l"(address) : "memory");
    } while (observed < expected);
    return 0;
}
