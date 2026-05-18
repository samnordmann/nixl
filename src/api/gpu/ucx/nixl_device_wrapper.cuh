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
#ifndef _NIXL_DEVICE_WRAPPER_CUH
#define _NIXL_DEVICE_WRAPPER_CUH

#include <nixl_device.cuh>

#if defined(__cplusplus)
#define NIXL_IR_EXTERN_C extern "C"
#else
#define NIXL_IR_EXTERN_C
#endif

#define NIXL_IR_DEVICE_INLINE __device__ __forceinline__

/*
 * Experimental C-shaped wrapper surface for Python GPU DSL bindings.
 *
 * CuTe-DSL and similar systems bind most cleanly to named device symbols in a
 * bitcode artifact.  The public NIXL device API is currently C++ templates in
 * nixl_device.cuh, so this header provides stable names for the concrete
 * specializations a DSL binding can call.  A production binding should compile
 * this surface into a NIXL-owned libnixl_device.bc artifact and generate the
 * Python FFI declarations from this header.
 */

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE size_t
nixlGpuXferStatusH_C_size() {
    return sizeof(nixlGpuXferStatusH);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE size_t
nixlMemViewElem_C_size() {
    return sizeof(nixlMemViewElem);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE void *
nixlDeviceGetPtr(nixlMemViewH mvh, size_t index) {
    return nixlGetPtr(mvh, index);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDevicePutThread(nixlMemViewElem src,
                    nixlMemViewElem dst,
                    size_t size,
                    unsigned channel_id,
                    uint64_t flags,
                    nixlGpuXferStatusH *xfer_status) {
    return nixlPut<nixl_gpu_level_t::THREAD>(
        src, dst, size, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDevicePutWarp(nixlMemViewElem src,
                  nixlMemViewElem dst,
                  size_t size,
                  unsigned channel_id,
                  uint64_t flags,
                  nixlGpuXferStatusH *xfer_status) {
    return nixlPut<nixl_gpu_level_t::WARP>(
        src, dst, size, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDevicePutBlock(nixlMemViewElem src,
                   nixlMemViewElem dst,
                   size_t size,
                   unsigned channel_id,
                   uint64_t flags,
                   nixlGpuXferStatusH *xfer_status) {
    return nixlPut<nixl_gpu_level_t::BLOCK>(
        src, dst, size, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDevicePutGrid(nixlMemViewElem src,
                  nixlMemViewElem dst,
                  size_t size,
                  unsigned channel_id,
                  uint64_t flags,
                  nixlGpuXferStatusH *xfer_status) {
    return nixlPut<nixl_gpu_level_t::GRID>(
        src, dst, size, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceAtomicAddThread(uint64_t value,
                          nixlMemViewElem counter,
                          unsigned channel_id,
                          uint64_t flags,
                          nixlGpuXferStatusH *xfer_status) {
    return nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
        value, counter, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceAtomicAddWarp(uint64_t value,
                        nixlMemViewElem counter,
                        unsigned channel_id,
                        uint64_t flags,
                        nixlGpuXferStatusH *xfer_status) {
    return nixlAtomicAdd<nixl_gpu_level_t::WARP>(
        value, counter, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceAtomicAddBlock(uint64_t value,
                         nixlMemViewElem counter,
                         unsigned channel_id,
                         uint64_t flags,
                         nixlGpuXferStatusH *xfer_status) {
    return nixlAtomicAdd<nixl_gpu_level_t::BLOCK>(
        value, counter, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceAtomicAddGrid(uint64_t value,
                        nixlMemViewElem counter,
                        unsigned channel_id,
                        uint64_t flags,
                        nixlGpuXferStatusH *xfer_status) {
    return nixlAtomicAdd<nixl_gpu_level_t::GRID>(
        value, counter, channel_id, flags, xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceGetXferStatusThread(nixlGpuXferStatusH *xfer_status) {
    return nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(*xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceGetXferStatusWarp(nixlGpuXferStatusH *xfer_status) {
    return nixlGpuGetXferStatus<nixl_gpu_level_t::WARP>(*xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceGetXferStatusBlock(nixlGpuXferStatusH *xfer_status) {
    return nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK>(*xfer_status);
}

NIXL_IR_EXTERN_C NIXL_IR_DEVICE_INLINE nixl_status_t
nixlDeviceGetXferStatusGrid(nixlGpuXferStatusH *xfer_status) {
    return nixlGpuGetXferStatus<nixl_gpu_level_t::GRID>(*xfer_status);
}

#endif // _NIXL_DEVICE_WRAPPER_CUH
