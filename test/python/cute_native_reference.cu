// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Same credit protocol as exchange.py, compiled by NVCC as an independent baseline.
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <algorithm>

__device__ void
wait_counter(const uint64_t *address, uint64_t expected) {
    uint64_t value;
    do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    } while (value < expected);
}

template<nixl_gpu_level_t level>
__device__ int
finish(nixlGpuXferStatusH &req, nixl_status_t s) {
    while (s == NIXL_IN_PROG) {
        s = nixlGpuGetXferStatus<level>(req);
    }
    return s;
}

__device__ int
signal(uint64_t view, size_t offset) {
    nixlGpuXferStatusH req;
    return finish<nixl_gpu_level_t::THREAD>(
        req, nixlAtomicAdd(1, {reinterpret_cast<nixlMemViewH>(view), 1, offset}, 0, 0, &req));
}

__global__ void
begin(const uint64_t *plan, uint64_t *ready, uint64_t *step, int peers, int rank, int world) {
    uint64_t old = *step;
    __syncthreads();
    if (!threadIdx.x) {
        *step = old + 1;
    }
    if (old) {
        for (int i = threadIdx.x; i < peers; i += blockDim.x) {
            signal(plan[2 * i + 1], (world + rank) * 8);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < peers; i += blockDim.x) {
        wait_counter(ready + world + plan[2 * i], old);
    }
}

template<nixl_gpu_level_t level>
__global__ void
send(uint64_t local,
     const uint64_t *plan,
     int *status,
     uint64_t size,
     int rank,
     int tiles,
     int tile_bytes) {
    int tile = blockIdx.x, row = blockIdx.y, peer = plan[2 * row];
    uint64_t view = plan[2 * row + 1], offset = uint64_t(tile) * tile_bytes;
    uint64_t bytes = min(uint64_t(tile_bytes), size - offset);
    nixlGpuXferStatusH req;
    int result = finish<level>(
        req,
        nixlPut<level>({reinterpret_cast<nixlMemViewH>(local), 0, peer * size + offset},
                       {reinterpret_cast<nixlMemViewH>(view), 0, rank * size + offset},
                       bytes,
                       0,
                       0,
                       &req));
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    if (!threadIdx.x) {
        status[row * tiles + tile] = result;
    }
}

__global__ void
wait_all(const uint64_t *plan,
         uint64_t *ready,
         const uint64_t *step,
         int *status,
         int peers,
         int tiles,
         int rank) {
    for (int i = threadIdx.x; i < peers; i += blockDim.x) {
        int failed = 0;
        for (int t = 0; t < tiles; t++) {
            failed |= status[i * tiles + t];
        }
        if (!failed) {
            status[i * tiles] = signal(plan[2 * i + 1], rank * 8);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < peers; i += blockDim.x) {
        wait_counter(ready + plan[2 * i], *step);
    }
}

template<nixl_gpu_level_t level>
__global__ void
small(uint64_t local,
      uint64_t *plan,
      uint64_t *ready,
      uint64_t *step,
      int *status,
      uint64_t size,
      int rank,
      int world) {
    int tid = threadIdx.x;
    if (!tid) {
        uint64_t previous = *step;
        if (previous) {
            signal(plan[1], (world + rank) * 8);
        }
        wait_counter(ready + world + plan[0], previous);
        *step = previous + 1;
    }
    __syncthreads();
    nixlGpuXferStatusH req;
    int result =
        finish<level>(req,
                      nixlPut<level>({reinterpret_cast<nixlMemViewH>(local), 0, plan[0] * size},
                                     {reinterpret_cast<nixlMemViewH>(plan[1]), 0, rank * size},
                                     size,
                                     0,
                                     0,
                                     &req));
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    __syncthreads();
    if (!tid) {
        status[0] = result;
        if (!result) {
            status[0] = signal(plan[1], rank * 8);
        }
    }
    if (!tid) {
        wait_counter(ready + plan[0], *step);
    }
}

template<nixl_gpu_level_t level>
void
launch(uint64_t local,
       uint64_t *plan,
       uint64_t *ready,
       uint64_t *step,
       int *status,
       uint64_t size,
       int rank,
       int world,
       int tiles,
       int tile_bytes,
       int threads,
       cudaStream_t stream) {
    if (size <= 4096 && world == 2 && tiles == 1) {
        small<level>
            <<<1, threads, 0, stream>>>(local, plan, ready, step, status, size, rank, world);
        return;
    }
    begin<<<1, 128, 0, stream>>>(plan, ready, step, world - 1, rank, world);
    send<level><<<dim3(tiles, world - 1), threads, 0, stream>>>(
        local, plan, status, size, rank, tiles, tile_bytes);
    wait_all<<<1, 128, 0, stream>>>(plan, ready, step, status, world - 1, tiles, rank);
}

extern "C" int
exchange(uint64_t local,
         uint64_t *plan,
         uint64_t *ready,
         uint64_t *step,
         int *status,
         uint64_t size,
         int rank,
         int world,
         int tiles,
         int tile_bytes,
         int threads,
         int level,
         cudaStream_t stream) {
#define LAUNCH(L) \
    launch<L>(    \
        local, plan, ready, step, status, size, rank, world, tiles, tile_bytes, threads, stream)
    if (level == 0) {
        LAUNCH(nixl_gpu_level_t::THREAD);
    }
    if (level == 1) {
        LAUNCH(nixl_gpu_level_t::WARP);
    }
    if (level == 2) {
        LAUNCH(nixl_gpu_level_t::BLOCK);
    }
    return cudaGetLastError();
}
