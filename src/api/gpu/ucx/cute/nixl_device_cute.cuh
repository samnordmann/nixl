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

#ifndef NIXL_DEVICE_CUTE_CUH
#define NIXL_DEVICE_CUTE_CUH

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define NIXL_CUTE_DEVICE __device__
#else
#define NIXL_CUTE_DEVICE
#endif

/**
 * Stable, C-shaped device ABI consumed by CuTe DSL.
 *
 * Memory-view pointers are opaque handles returned by nixlAgent::prepMemView.
 * The thread and warp entry points are deliberately separate: every thread
 * participating in a warp call must use identical arguments and non-divergent
 * control flow. The wait variants do not return until the UCX request has
 * completed or failed. Thread requests use thread-local storage; warp requests
 * use one shared object per warp. Warp entry points require a one-dimensional
 * block whose x dimension is a multiple of 32.
 * Wait entry points accept flags == 0 only. Post entry points accept DEFER and
 * intentionally allocate no request storage: the caller must eventually issue
 * a non-deferred ordered operation on the same channel. A completion atomic on
 * that channel is the normal release signal for a batch of deferred PUTs.
 */
extern "C" {

NIXL_CUTE_DEVICE uint32_t
nixl_cute_abi_version();

NIXL_CUTE_DEVICE int32_t
nixl_cute_put_thread_wait(void *local_view,
                          uint32_t local_index,
                          uint64_t local_offset,
                          void *remote_view,
                          uint32_t remote_index,
                          uint64_t remote_offset,
                          uint64_t size,
                          uint32_t channel,
                          uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_put_warp_wait(void *local_view,
                        uint32_t local_index,
                        uint64_t local_offset,
                        void *remote_view,
                        uint32_t remote_index,
                        uint64_t remote_offset,
                        uint64_t size,
                        uint32_t channel,
                        uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_put_thread_post(void *local_view,
                          uint32_t local_index,
                          uint64_t local_offset,
                          void *remote_view,
                          uint32_t remote_index,
                          uint64_t remote_offset,
                          uint64_t size,
                          uint32_t channel,
                          uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_put_warp_post(void *local_view,
                        uint32_t local_index,
                        uint64_t local_offset,
                        void *remote_view,
                        uint32_t remote_index,
                        uint64_t remote_offset,
                        uint64_t size,
                        uint32_t channel,
                        uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_add_thread_wait(uint64_t value,
                                 void *remote_view,
                                 uint32_t remote_index,
                                 uint64_t remote_offset,
                                 uint32_t channel,
                                 uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_add_warp_wait(uint64_t value,
                               void *remote_view,
                               uint32_t remote_index,
                               uint64_t remote_offset,
                               uint32_t channel,
                               uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_add_thread_post(uint64_t value,
                                 void *remote_view,
                                 uint32_t remote_index,
                                 uint64_t remote_offset,
                                 uint32_t channel,
                                 uint64_t flags);

NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_add_warp_post(uint64_t value,
                               void *remote_view,
                               uint32_t remote_index,
                               uint64_t remote_offset,
                               uint32_t channel,
                               uint64_t flags);

/** Read CUDA's system-wide nanosecond timer for in-kernel instrumentation. */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_globaltimer_ns();

/**
 * System-scope acquire load used to consume GPU-written completion signals.
 * Address must be non-null and naturally aligned; validation is intentionally
 * left to the CuTe tracing layer so this hot primitive has no runtime branch.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_load_acquire_system_u64(uint64_t address);

/** Same-device GPU-scope acquire load for local cooperative state. */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_load_acquire_gpu_u64(uint64_t address);

/**
 * Wait until an aligned GPU address reaches ``least`` using system-scope
 * acquire loads.  The warp form polls only from lane zero and broadcasts the
 * observed value, then performs a warp memory barrier so lane zero's acquire
 * orders every lane's subsequent payload reads.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_thread(uint64_t address, uint64_t least);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_warp(uint64_t address, uint64_t least);

/**
 * Unbounded GPU-scope waits for same-device CTA handoffs. Failed polls use
 * relaxed loads and the successful observation is reloaded with acquire
 * semantics. No ``%globaltimer`` instruction is executed.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_thread(uint64_t address, uint64_t least);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_warp(uint64_t address, uint64_t least);

/**
 * Wait on a system-scope target while periodically observing a same-GPU abort
 * word. Failed target polls are relaxed, successful target and abort
 * observations are acquire loads, and no ``%globaltimer`` instruction is
 * executed. This form is for performance kernels whose peer lifetime is
 * guaranteed by a graceful/drained control plane.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_or_abort_thread(uint64_t address,
                                                   uint64_t least,
                                                   uint64_t abort_address,
                                                   uint64_t abort_least);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_or_abort_warp(uint64_t address,
                                                 uint64_t least,
                                                 uint64_t abort_address,
                                                 uint64_t abort_least);

/**
 * Timer-free peer wait with both local and peer abort epochs. Target and peer
 * abort use system scope; local abort uses GPU scope. Abort words are sampled
 * only after each 4096 failed target loads, leaving the ready path as one
 * acquire. A value below least reports either abort.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_or_aborts_thread(
    uint64_t address,
    uint64_t least,
    uint64_t local_abort_address,
    uint64_t local_abort_least,
    uint64_t peer_abort_address,
    uint64_t peer_abort_least);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_or_aborts_warp(
    uint64_t address,
    uint64_t least,
    uint64_t local_abort_address,
    uint64_t local_abort_least,
    uint64_t peer_abort_address,
    uint64_t peer_abort_least);

/**
 * Wait for a same-GPU target, but return its last acquire-loaded value when a
 * monotonic abort word reaches ``abort_least``. The ready path is one target
 * acquire. Failed target polls are relaxed and inspect the abort word once per
 * 256 misses, with no timer instructions. A return below ``least`` reports
 * abort. The warp form has the same converged-call contract as other waits.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_or_abort_thread(uint64_t address,
                                                uint64_t least,
                                                uint64_t abort_address,
                                                uint64_t abort_least);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_or_abort_warp(uint64_t address,
                                              uint64_t least,
                                              uint64_t abort_address,
                                              uint64_t abort_least);

/**
 * Bounded variants of the acquire wait. ``deadline_ns`` is an absolute
 * ``%globaltimer`` value. They return the last acquire-loaded value; a value
 * below ``least`` means the deadline expired. The timer is sampled once per
 * 256 unsuccessful loads so the healthy ready path pays no timer-read cost.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_thread_until(uint64_t address,
                                                uint64_t least,
                                                uint64_t deadline_ns);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_warp_until(uint64_t address,
                                              uint64_t least,
                                              uint64_t deadline_ns);

/**
 * Relative-timeout variants with a one-load healthy path. The timer is first
 * read only after the counter misses ``least``; subsequent timeout checks are
 * amortized over groups of 256 failed loads. Unsigned elapsed-time arithmetic
 * handles a wrap of ``%globaltimer``.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_thread_for(uint64_t address,
                                              uint64_t least,
                                              uint64_t timeout_ns);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_warp_for(uint64_t address,
                                            uint64_t least,
                                            uint64_t timeout_ns);

/**
 * Relative system-scope wait that also observes a same-GPU monotonic abort
 * word. The ready path remains one target acquire. After a miss, target polls
 * are relaxed, timeout checks occur every 256 polls, and the local abort word
 * is checked every 4096 polls. A return below ``least`` means timeout or abort;
 * callers can acquire-load ``abort_address`` to distinguish them.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_thread_for_or_abort(
    uint64_t address,
    uint64_t least,
    uint64_t abort_address,
    uint64_t abort_least,
    uint64_t timeout_ns);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_system_u64_warp_for_or_abort(
    uint64_t address,
    uint64_t least,
    uint64_t abort_address,
    uint64_t abort_least,
    uint64_t timeout_ns);

/**
 * GPU-scope relative waits for counters whose producers and consumers are
 * CTAs on the same device. Do not use these to consume peer/NIC writes.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_thread_for(uint64_t address,
                                           uint64_t least,
                                           uint64_t timeout_ns);

NIXL_CUTE_DEVICE uint64_t
nixl_cute_wait_acquire_gpu_u64_warp_for(uint64_t address,
                                         uint64_t least,
                                         uint64_t timeout_ns);

/**
 * System-scope release store used to publish direct peer-memory writes.
 * Address must be non-null and naturally aligned.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_store_release_system_u64(uint64_t address, uint64_t value);

/** Publish to CTAs on this GPU only; peer/NIC consumers require system scope. */
NIXL_CUTE_DEVICE int32_t
nixl_cute_store_release_gpu_u64(uint64_t address, uint64_t value);

/**
 * Atomically add to a same-GPU counter with release ordering and return the
 * value observed before the addition. Peer/NIC counters require a transport
 * atomic or a system-scope protocol instead.
 */
NIXL_CUTE_DEVICE uint64_t
nixl_cute_atomic_add_release_gpu_u64(uint64_t address, uint64_t value);

/** Monotonically publish a multi-writer same-GPU abort epoch. */
NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_max_release_gpu_u64(uint64_t address, uint64_t value);

/** Monotonically publish an abort epoch that mapped peer GPUs may acquire. */
NIXL_CUTE_DEVICE int32_t
nixl_cute_atomic_max_release_system_u64(uint64_t address, uint64_t value);

/** Atomically record the first nonzero i32 status and return the prior value. */
NIXL_CUTE_DEVICE int32_t
nixl_cute_compare_exchange_status_gpu_i32(uint64_t address, int32_t value);

/**
 * Synchronize every thread in a cooperatively launched grid. Every CTA and
 * thread must execute the call in converged control flow. The launch must set
 * CUDA's cooperative attribute; violating either precondition can deadlock.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_sync_grid();

/**
 * Order prior GPU writes before a following device-initiated transport reads
 * those bytes. This is needed when a fused kernel fills a registered source
 * buffer and then submits a NIXL PUT from that buffer. It is not needed for a
 * source that was made visible at an earlier CUDA operation boundary.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_fence_release_system();

/**
 * Return the remote descriptor's mapping in the calling process, or null when
 * the transport cannot expose one. This process-local pointer may differ
 * numerically from the address advertised by the allocation owner; direct
 * accesses must use the returned base plus a descriptor-relative offset.
 * Protocols using system-scope atomics on peer GPU memory must separately
 * qualify native atomics for every directed accessing-GPU-to-owner-GPU pair.
 */
NIXL_CUTE_DEVICE void *
nixl_cute_get_ptr(void *remote_view, uint32_t remote_index);

/**
 * Lowest-overhead mapped copy after the caller resolves a peer pointer once
 * with ``nixl_cute_get_ptr``. Both addresses and size are 16-byte aligned.
 * Source and destination spans do not overlap. The destination is a pointer in
 * the calling process and must be derived from the non-null get-ptr result plus
 * an offset within that descriptor. Its numeric value need not equal the
 * allocation owner's address in another process.
 * The coherent form supports fused/persistent source writes; the read-only
 * form has the same whole-kernel immutability precondition described below.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_mapped_copy_warp_ptr(uint64_t source_address,
                               uint64_t destination_address,
                               uint64_t size);

NIXL_CUTE_DEVICE int32_t
nixl_cute_mapped_copy_warp_ptr_readonly(uint64_t source_address,
                                        uint64_t destination_address,
                                        uint64_t size);

/**
 * Copy a 16-byte-aligned span directly through a mapped remote descriptor.
 *
 * All 32 lanes must call this function convergently with identical arguments.
 * Lane zero resolves the mapping and broadcasts it.  The copy uses coherent
 * 128-bit no-allocate loads and no-allocate stores, then performs one final
 * warp rendezvous.  It is safe when a producer kernel wrote the source before
 * this call. Source and destination must not overlap. The resolved destination
 * is the process-local mapping base plus ``remote_offset``; cross-process
 * numeric address equality is not required. It does not write a completion
 * signal. A null mapping returns NIXL_ERR_NOT_SUPPORTED uniformly so the caller
 * can use a NIXL PUT instead.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_mapped_copy_warp(uint64_t source_address,
                           void *remote_view,
                           uint32_t remote_index,
                           uint64_t remote_offset,
                           uint64_t size);

/**
 * Higher-bandwidth mapped copy for a source that is read-only for the entire
 * kernel lifetime. This form uses ``ld.global.nc`` and must never be used for
 * a buffer written or reused by the same persistent/fused kernel.
 */
NIXL_CUTE_DEVICE int32_t
nixl_cute_mapped_copy_warp_readonly(uint64_t source_address,
                                    void *remote_view,
                                    uint32_t remote_index,
                                    uint64_t remote_offset,
                                    uint64_t size);

} // extern "C"

#undef NIXL_CUTE_DEVICE

#endif // NIXL_DEVICE_CUTE_CUH
