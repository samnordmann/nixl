/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_CLANG_UCT_CUDA_IPC_COMPAT_CUH
#define NIXL_CLANG_UCT_CUDA_IPC_COMPAT_CUH

/*
 * UCX 1.23's CUDA-IPC explicit template specializations omit UCS_F_DEVICE.
 * NVCC inherits the primary template's device target, while Clang diagnoses
 * the specializations as host functions. Apply the device attribute only
 * while the original UCX header is parsed; include_next keeps UCX authoritative.
 */
#if defined(__clang__) && defined(__CUDA__)
/*
 * Parse the dependencies before entering the attribute region.  In
 * particular, <cuda/atomic> reaches CUDA's C++ runtime and <assert.h>;
 * marking those declarations as device functions changes their exception
 * specifications and makes Clang reject the CUDA wrapper declarations.
 * Their include guards make the corresponding includes in UCX a no-op, so
 * the pragma below is confined to declarations authored by cuda_ipc.cuh.
 */
#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>
#include <cuda/atomic>

/*
 * UCX through current master implements CUDA-IPC counter publication as a
 * relaxed fetch-add followed by a release fence.  A release sequence must be
 * established by the atomic write itself (or by a fence sequenced before it);
 * a later fence cannot make an acquire load that observed the earlier atomic
 * synchronize with preceding payload stores.  Rename the two upstream
 * definitions while parsing the header, then provide the ordering-correct
 * endpoint below.  This compatibility layer is compiled only into NIXL's
 * Clang CUDA bitcode and can be removed once the minimum supported UCX carries
 * a release fetch-add.
 */
#define uct_cuda_ipc_atomic_inc uct_cuda_ipc_atomic_inc_nixl_unordered
#define uct_cuda_ipc_ep_atomic_add uct_cuda_ipc_ep_atomic_add_nixl_unordered
#pragma clang attribute push(__attribute__((device)), apply_to = function)
#include_next <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#pragma clang attribute pop
#undef uct_cuda_ipc_ep_atomic_add
#undef uct_cuda_ipc_atomic_inc

#pragma clang attribute push(__attribute__((device)), apply_to = function)
UCS_F_DEVICE void
uct_cuda_ipc_atomic_inc(uint64_t *dst, uint64_t inc_value)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> dst_ref{*dst};
    dst_ref.fetch_add(inc_value, cuda::memory_order_release);
}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK, typename MemElement>
UCS_F_DEVICE ucs_status_t
uct_cuda_ipc_ep_atomic_add(uct_device_ep_h device_ep,
                           const MemElement *mem_elem,
                           uint64_t inc_value,
                           uint64_t remote_address,
                           uint64_t flags,
                           uct_device_completion_t *comp)
{
    auto cuda_ipc_mem_element =
        reinterpret_cast<const uct_cuda_ipc_md_device_mem_element_t *>(mem_elem);
    unsigned int lane_id, num_lanes;
    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    if (lane_id == 0) {
        auto *mapped_rem_addr = reinterpret_cast<uint64_t *>(
            uct_cuda_ipc_map_remote(cuda_ipc_mem_element, remote_address));
        uct_cuda_ipc_atomic_inc(mapped_rem_addr, inc_value);
    }
    uct_cuda_ipc_level_sync<level>();
    return UCS_OK;
}
#pragma clang attribute pop
#else
#include_next <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#endif

#endif // NIXL_CLANG_UCT_CUDA_IPC_COMPAT_CUH
