/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <iostream>
#include <optional>
#include <span>
#include <utility>

#include "nixl.h"
#include "serdes/serdes.h"

namespace py = pybind11;

typedef std::map<std::string, std::vector<py::bytes>> nixl_py_notifs_t;

class nixlNotPostedError : public std::runtime_error {
public:
    nixlNotPostedError(const char *what) : runtime_error(what) {}
};

class nixlInvalidParamError : public std::runtime_error {
public:
    nixlInvalidParamError(const char *what) : runtime_error(what) {}
};

class nixlBackendError : public std::runtime_error {
public:
    nixlBackendError(const char *what) : runtime_error(what) {}
};

class nixlNotFoundError : public std::runtime_error {
public:
    nixlNotFoundError(const char *what) : runtime_error(what) {}
};

class nixlMismatchError : public std::runtime_error {
public:
    nixlMismatchError(const char *what) : runtime_error(what) {}
};

class nixlNotAllowedError : public std::runtime_error {
public:
    nixlNotAllowedError(const char *what) : runtime_error(what) {}
};

class nixlRepostActiveError : public std::runtime_error {
public:
    nixlRepostActiveError(const char *what) : runtime_error(what) {}
};

class nixlUnknownError : public std::runtime_error {
public:
    nixlUnknownError(const char *what) : runtime_error(what) {}
};

class nixlNotSupportedError : public std::runtime_error {
public:
    nixlNotSupportedError(const char *what) : runtime_error(what) {}
};

class nixlRemoteDisconnectError : public std::runtime_error {
public:
    nixlRemoteDisconnectError(const char *what) : runtime_error(what) {}
};

class nixlCancelledError : public std::runtime_error {
public:
    nixlCancelledError(const char *what) : runtime_error(what) {}
};

class nixlNoTelemetryError : public std::runtime_error {
public:
    nixlNoTelemetryError(const char *what) : runtime_error(what) {}
};

void
throw_nixl_exception(const nixl_status_t &status) {
    switch (status) {
    case NIXL_IN_PROG:
        return; // not an error
    case NIXL_SUCCESS:
        return; // not an error
    case NIXL_ERR_NOT_POSTED:
        throw nixlNotPostedError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_INVALID_PARAM:
        throw nixlInvalidParamError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_BACKEND:
        throw nixlBackendError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_NOT_FOUND:
        throw nixlNotFoundError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_MISMATCH:
        throw nixlMismatchError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_NOT_ALLOWED:
        throw nixlNotAllowedError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_REPOST_ACTIVE:
        throw nixlRepostActiveError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_UNKNOWN:
        throw nixlUnknownError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_NOT_SUPPORTED:
        throw nixlNotSupportedError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_REMOTE_DISCONNECT:
        throw nixlRemoteDisconnectError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_CANCELED:
        throw nixlCancelledError(nixlEnumStrings::statusStr(status).c_str());
        break;
    case NIXL_ERR_NO_TELEMETRY:
        throw nixlNoTelemetryError(nixlEnumStrings::statusStr(status).c_str());
        break;
    default:
        throw std::runtime_error("BAD_STATUS");
    }
}

namespace {

template<typename Tag>
class nixl_py_release_state {
public:
    explicit nixl_py_release_state(uintptr_t handle) : handle_(handle) {
        if (handle_ == 0) {
            throw std::invalid_argument("release handle must be non-zero");
        }
    }

    bool
    begin(const nixlAgent &agent, uintptr_t handle) {
        if (handle != handle_) {
            throw std::invalid_argument("release state does not own this handle");
        }
        if (agent_ != nullptr && agent_ != &agent) {
            throw std::invalid_argument("release state does not own this agent");
        }
        if (released_) {
            return false;
        }
        agent_ = &agent;
        released_ = true;
        return true;
    }

    void
    rollback() noexcept {
        released_ = false;
    }

    bool
    released() const noexcept {
        return released_;
    }

private:
    uintptr_t handle_;
    const nixlAgent *agent_ = nullptr;
    bool released_ = false;
};

struct nixl_py_xfer_release_tag {};
struct nixl_py_dlist_release_tag {};
using nixl_py_xfer_release_state = nixl_py_release_state<nixl_py_xfer_release_tag>;
using nixl_py_dlist_release_state = nixl_py_release_state<nixl_py_dlist_release_tag>;

struct nixl_py_xfer_handle_traits {
    using handle_type = nixlXferReqH;

    static nixl_status_t
    release(const nixlAgent &agent, handle_type *handle) {
        return agent.releaseXferReq(handle);
    }
};

struct nixl_py_dlist_handle_traits {
    using handle_type = nixlDlistH;

    static nixl_status_t
    release(const nixlAgent &agent, handle_type *handle) {
        return agent.releasedDlistH(handle);
    }
};

template<bool LeaseAware>
struct nixl_py_owned_handle_lease_state {};

template<>
struct nixl_py_owned_handle_lease_state<true> {
    size_t lease_count = 0;
};

template<typename Traits, bool LeaseAware = false>
class nixl_py_owned_handle : private nixl_py_owned_handle_lease_state<LeaseAware> {
public:
    using handle_type = typename Traits::handle_type;

    explicit nixl_py_owned_handle(const nixlAgent &agent) : agent_(&agent) {}
    nixl_py_owned_handle(const nixl_py_owned_handle &) = delete;
    nixl_py_owned_handle &operator=(const nixl_py_owned_handle &) = delete;
    nixl_py_owned_handle &operator=(nixl_py_owned_handle &&) = delete;

    nixl_py_owned_handle(nixl_py_owned_handle &&other) noexcept
        : agent_(other.agent_), handle_(other.handle_) {
        other.handle_ = nullptr;
        if constexpr (LeaseAware) {
            this->lease_count = other.lease_count;
            other.lease_count = 0;
        }
    }

    ~nixl_py_owned_handle() noexcept {
        if (handle_ != nullptr) {
            if constexpr (LeaseAware) {
                if (this->lease_count != 0) {
                    // A disappearing dispatcher whose request could not be
                    // quiesced deliberately leaks its descriptors instead of
                    // risking release beneath native request state.
                    return;
                }
            }
            // Return-conversion unwind reaches here with an unposted object
            // and cannot report cleanup failure. Retained/posted owners must
            // use explicit release, which preserves the pointer and reports a
            // native failure so the caller can retry.
            try {
                (void)Traits::release(*agent_, handle_);
            }
            catch (...) {
            }
        }
    }

    void
    adopt(handle_type *handle) noexcept {
        handle_ = handle;
    }

    uintptr_t
    value() const noexcept {
        return reinterpret_cast<uintptr_t>(handle_);
    }

    bool
    released() const noexcept {
        return handle_ == nullptr;
    }

    bool
    owned_by(const nixlAgent &agent) const noexcept {
        return agent_ == &agent;
    }

    handle_type *
    get() const noexcept {
        return handle_;
    }

    nixl_status_t
    release_status() {
        if (handle_ == nullptr) {
            return NIXL_SUCCESS;
        }
        if constexpr (LeaseAware) {
            if (this->lease_count != 0) {
                throw std::runtime_error(
                    "cannot release a descriptor-list owner while it is leased by a "
                    "request-slot execution");
            }
        }
        const nixl_status_t ret = Traits::release(*agent_, handle_);
        if (ret == NIXL_SUCCESS) {
            handle_ = nullptr;
        }
        return ret;
    }

    nixl_status_t
    release() {
        const nixl_status_t ret = release_status();
        throw_nixl_exception(ret);
        return ret;
    }

    void
    acquire_lease(const uintptr_t expected_handle) {
        static_assert(LeaseAware, "leases are only supported by lease-aware handles");
        if (handle_ == nullptr || value() != expected_handle) {
            throw std::invalid_argument(
                "cannot lease a released or mismatched descriptor-list owner");
        }
        if (this->lease_count == std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("descriptor-list owner lease count overflow");
        }
        ++this->lease_count;
    }

    void
    release_lease() noexcept {
        static_assert(LeaseAware, "leases are only supported by lease-aware handles");
        if (this->lease_count != 0) {
            --this->lease_count;
        }
    }

private:
    const nixlAgent *agent_;
    handle_type *handle_ = nullptr;
};

using nixl_py_owned_xfer_handle = nixl_py_owned_handle<nixl_py_xfer_handle_traits>;
using nixl_py_owned_dlist_handle =
    nixl_py_owned_handle<nixl_py_dlist_handle_traits, true>;

template<typename State, typename Release>
nixl_status_t
release_once(State &state, const nixlAgent &agent, uintptr_t handle, Release &&release) {
    if (!state.begin(agent, handle)) {
        return NIXL_SUCCESS;
    }
    try {
        const nixl_status_t ret = release();
        throw_nixl_exception(ret);
        return ret;
    }
    catch (...) {
        // A normal native failure did not consume the pointer and remains
        // retryable. Python signals cannot run between begin() and release()
        // because these bindings deliberately retain the GIL for both calls.
        state.rollback();
        throw;
    }
}

// Builds a compressed (strided) descriptor list from an Nx5 numpy array, where each row is a run
// of `count` blocks of `len` bytes with consecutive block starts spaced `stride` bytes apart:
// (addr, len, dev_id, stride, count). A dense run has stride == len.
nixl_stride_dlist_t
to_stride_dlist(nixl_mem_t mem, const py::array &descs) {
    static_assert(sizeof(nixlStrideDesc) == 5 * sizeof(uint64_t), "nixlStrideDesc size mismatch");

    if (descs.ndim() != 2 || descs.shape(1) != 5) {
        throw std::invalid_argument("descs must be a Nx5 numpy array");
    }
    if (!py::dtype::of<uint64_t>().equal(descs.dtype()) &&
        !py::dtype::of<int64_t>().equal(descs.dtype())) {
        throw std::invalid_argument("descs must be a Nx5 numpy array of uint64 or int64");
    }
    if (!(descs.flags() & py::array::c_style)) {
        throw std::invalid_argument("descs must be a C-contiguous numpy array");
    }

    size_t n = descs.shape(0);
    nixl_stride_dlist_t new_list(mem, n);
    if (n > 0) {
        // The Nx5 array matches the nixlStrideDesc layout so we can simply memcpy
        std::memcpy(&new_list[0], descs.data(), descs.size() * sizeof(uint64_t));
    }

    return new_list;
}

nixl_opt_args_t
make_opt_args(const std::vector<uintptr_t> &backends) {
    nixl_opt_args_t extra_params;
    extra_params.backends.reserve(backends.size());
    for (uintptr_t backend : backends) {
        extra_params.backends.push_back(reinterpret_cast<nixlBackendH *>(backend));
    }
    return extra_params;
}

nixl_opt_args_t
make_validated_opt_args(const nixlAgent &agent,
                        const std::vector<uintptr_t> &backends,
                        const char *owner,
                        const char *required_single_type = nullptr) {
    if (required_single_type != nullptr && backends.size() != 1) {
        throw std::invalid_argument(std::string(owner) + " requires exactly one explicit " +
                                    required_single_type + " backend handle");
    }
    nixl_opt_args_t extra_params;
    extra_params.backends.reserve(backends.size());
    for (uintptr_t backend : backends) {
        auto *handle = reinterpret_cast<nixlBackendH *>(backend);
        nixl_backend_t type;
        if (agent.getBackendType(handle, type) != NIXL_SUCCESS) {
            throw std::invalid_argument(std::string(owner) +
                                        " contains a backend handle that does not belong to "
                                        "this NIXL agent");
        }
        if (required_single_type != nullptr && type != required_single_type) {
            throw std::invalid_argument(std::string(owner) + " requires a " +
                                        required_single_type + " backend handle");
        }
        extra_params.backends.push_back(handle);
    }
    return extra_params;
}

class nixl_py_mem_deregistration {
public:
    nixl_py_mem_deregistration(nixlAgent &agent,
                               const nixl_reg_dlist_t &descs,
                               const std::vector<uintptr_t> &backends)
        : agent_(&agent),
          descs_(descs),
          extra_params_(make_validated_opt_args(
              agent, backends, "memory-deregistration receipt", "UCX")) {}

    nixl_py_mem_deregistration(const nixl_py_mem_deregistration &) = delete;
    nixl_py_mem_deregistration &operator=(const nixl_py_mem_deregistration &) = delete;

    nixl_status_t
    execute(nixlAgent &agent) {
        if (&agent != agent_) {
            throw std::invalid_argument(
                "memory-deregistration receipt belongs to a different NIXL agent");
        }
        std::lock_guard<std::mutex> lock(execute_mutex_);
        if (completed_.load(std::memory_order_acquire)) {
            return NIXL_SUCCESS;
        }

        const nixl_status_t ret = agent_->deregisterMem(descs_, &extra_params_);
        if (ret == NIXL_SUCCESS || ret == NIXL_ERR_NOT_FOUND) {
            // This commit runs while the GIL is released. A pending Python
            // signal delivered when the binding reacquires the GIL therefore
            // cannot make a completed native deregistration look retryable.
            completed_.store(true, std::memory_order_release);
        }
        throw_nixl_exception(ret);
        return ret;
    }

    nixl_status_t
    execute_bound() {
        return execute(*agent_);
    }

    bool
    completed() const noexcept {
        return completed_.load(std::memory_order_acquire);
    }

private:
    nixlAgent *const agent_;
    const nixl_reg_dlist_t descs_;
    const nixl_opt_args_t extra_params_;
    std::mutex execute_mutex_;
    std::atomic<bool> completed_{false};
};

py::dict
get_notif_batches(nixlAgent &agent, const nixl_opt_args_t *extra_params) {
    nixl_notifs_t new_notifs;
    {
        py::gil_scoped_release release;
        const nixl_status_t ret = agent.getNotifs(new_notifs, extra_params);
        throw_nixl_exception(ret);
    }

    py::dict result;
    for (const auto &pair : new_notifs) {
        // Core and the provider expose immutable grouped payloads. Build that
        // final container directly: a generic STL caster would first create a
        // Python list, which the provider would immediately copy to a tuple.
        py::tuple payloads(pair.second.size());
        for (size_t index = 0; index < pair.second.size(); ++index) {
            py::bytes payload(pair.second[index]);
            // A new tuple steals the sole reference. Avoid the accessor's
            // redundant increment/decrement pair for every payload.
            PyTuple_SET_ITEM(payloads.ptr(),
                             static_cast<Py_ssize_t>(index),
                             payload.release().ptr());
        }
        result[py::str(pair.first)] = std::move(payloads);
    }
    return result;
}

class nixl_py_notification_receiver {
public:
    nixl_py_notification_receiver(nixlAgent &agent,
                                  const std::vector<uintptr_t> &backends)
        : agent_(&agent),
          extra_params_(
              make_validated_opt_args(agent, backends, "notification receiver")) {}

    nixl_py_notification_receiver(const nixl_py_notification_receiver &) = delete;
    nixl_py_notification_receiver &operator=(const nixl_py_notification_receiver &) = delete;

    py::dict
    poll() {
        return poll_impl(std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    }

    py::dict
    poll_bounded(const int64_t max_items,
                 const int64_t max_batch_items,
                 const int64_t max_batch_bytes,
                 const int64_t max_payload_bytes) {
        if (max_items < 0) {
            throw std::invalid_argument("max_items must be non-negative");
        }
        if (max_batch_items <= 0) {
            throw std::invalid_argument("max_batch_items must be positive");
        }
        if (max_batch_bytes <= 0) {
            throw std::invalid_argument("max_batch_bytes must be positive");
        }
        if (max_payload_bytes <= 0) {
            throw std::invalid_argument("max_payload_bytes must be positive");
        }
        return poll_impl(static_cast<size_t>(max_items),
                         static_cast<size_t>(max_batch_items),
                         static_cast<size_t>(max_batch_bytes),
                         static_cast<size_t>(max_payload_bytes));
    }

private:
    using source_batch_t = std::pair<std::string, std::vector<nixl_blob_t>>;

    bool
    spill_empty() const noexcept {
        return source_cursor_ == spill_.size();
    }

    void
    refill_spill(const std::optional<size_t> max_batch_items,
                 const std::optional<size_t> max_batch_bytes,
                 const std::optional<size_t> max_payload_bytes) {
        nixl_notifs_t drained;
        const nixl_status_t ret = agent_->getNotifs(drained, &extra_params_);
        if (ret == NIXL_SUCCESS) {
            size_t item_count = 0;
            size_t payload_bytes = 0;
            for (const auto &pair : drained) {
                for (const auto &payload : pair.second) {
                    if (max_payload_bytes.has_value() &&
                        payload.size() > *max_payload_bytes) {
                        throw std::length_error(
                            "notification payload exceeds max_payload_bytes");
                    }
                    if (max_batch_items.has_value() &&
                        item_count == *max_batch_items) {
                        throw std::length_error(
                            "native notification batch exceeds max_batch_items");
                    }
                    ++item_count;
                    if (max_batch_bytes.has_value() &&
                        payload.size() > *max_batch_bytes - payload_bytes) {
                        throw std::length_error(
                            "native notification batch exceeds max_batch_bytes");
                    }
                    payload_bytes += payload.size();
                }
            }

            spill_.clear();
            spill_.reserve(drained.size());
            for (auto &pair : drained) {
                if (!pair.second.empty()) {
                    spill_.emplace_back(pair.first, std::move(pair.second));
                }
            }
            source_cursor_ = 0;
            payload_cursor_ = 0;
        }
        throw_nixl_exception(ret);
    }

    py::dict
    materialize(const std::optional<size_t> max_items) {
        py::dict result;
        size_t source_cursor = source_cursor_;
        size_t payload_cursor = payload_cursor_;
        size_t remaining = max_items.value_or(std::numeric_limits<size_t>::max());

        while (source_cursor < spill_.size() && remaining != 0) {
            const auto &source = spill_[source_cursor];
            const size_t available = source.second.size() - payload_cursor;
            const size_t take = std::min(available, remaining);
            py::tuple payloads(take);
            for (size_t index = 0; index < take; ++index) {
                py::bytes payload(source.second[payload_cursor + index]);
                PyTuple_SET_ITEM(payloads.ptr(),
                                 static_cast<Py_ssize_t>(index),
                                 payload.release().ptr());
            }
            result[py::str(source.first)] = std::move(payloads);
            payload_cursor += take;
            remaining -= take;
            if (payload_cursor == source.second.size()) {
                ++source_cursor;
                payload_cursor = 0;
            }
        }

        // Cursor publication is deliberately after every Python allocation
        // and dict insertion above. Conversion failure leaves the native spill
        // untouched and retryable instead of silently skipping payloads.
        source_cursor_ = source_cursor;
        payload_cursor_ = payload_cursor;
        if (spill_empty()) {
            spill_.clear();
            source_cursor_ = 0;
        }
        return result;
    }

    py::dict
    poll_impl(const std::optional<size_t> max_items,
              const std::optional<size_t> max_batch_items,
              const std::optional<size_t> max_batch_bytes,
              const std::optional<size_t> max_payload_bytes) {
        if (max_items.has_value() && *max_items == 0) {
            return py::dict();
        }

        // Never wait on the receiver mutex while holding the GIL: another
        // polling thread may need it to finish Python result construction.
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(poll_mutex_);
        if (spill_empty()) {
            refill_spill(max_batch_items, max_batch_bytes, max_payload_bytes);
        }
        py::gil_scoped_acquire acquire;
        return materialize(max_items);
    }

    nixlAgent *const agent_;
    const nixl_opt_args_t extra_params_;
    std::mutex poll_mutex_;
    std::vector<source_batch_t> spill_;
    size_t source_cursor_ = 0;
    size_t payload_cursor_ = 0;
};

class nixl_py_notification_sender {
public:
    nixl_py_notification_sender(nixlAgent &agent,
                                std::string remote_agent,
                                const std::vector<uintptr_t> &backends)
        : agent_(&agent),
          remote_agent_(std::move(remote_agent)),
          extra_params_(make_validated_opt_args(agent, backends, "notification sender")) {}

    nixl_py_notification_sender(const nixl_py_notification_sender &) = delete;
    nixl_py_notification_sender &operator=(const nixl_py_notification_sender &) = delete;

    void
    send(const std::string &msg) const {
        const nixl_status_t ret = agent_->genNotif(remote_agent_, msg, &extra_params_);
        throw_nixl_exception(ret);
    }

private:
    nixlAgent *agent_;
    const std::string remote_agent_;
    const nixl_opt_args_t extra_params_;
};

bool
is_native_signed_int32_buffer(const py::buffer_info &info) noexcept;

std::vector<int>
copy_xfer_indices(const py::object &indices) {
    // Request slots own their index vectors for their complete lifetime.  Copy
    // a native int32 buffer in one operation while the GIL is held instead of
    // expanding it through a Python list and one Python integer per element.
    // Unlike make_xfer_req(), this path never borrows exporter storage.
    if (PyObject_CheckBuffer(indices.ptr())) {
        auto buffer = py::reinterpret_borrow<py::buffer>(indices);
        auto info = buffer.request(false);
        if (is_native_signed_int32_buffer(info)) {
            std::vector<int> result(static_cast<size_t>(info.size));
            if (!result.empty()) {
                std::memcpy(result.data(), info.ptr, result.size() * sizeof(int));
            }
            return result;
        }
    }

    py::sequence values;
    if (py::isinstance<py::array>(indices)) {
        const auto indices_array = indices.cast<py::array>();
        if (indices_array.ndim() != 1) {
            throw std::invalid_argument("indices numpy array must be 1D");
        }
        values = indices_array.attr("tolist")().cast<py::sequence>();
    } else {
        values = indices.cast<py::sequence>();
    }
    std::vector<int> result;
    result.reserve(values.size());
    for (const py::handle value : values) {
        if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
            throw std::invalid_argument("request-slot indices must be integers");
        }
        const int64_t native_value = py::cast<int64_t>(value);
        if (native_value < std::numeric_limits<int>::min() ||
            native_value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("request-slot index exceeds native int range");
        }
        result.push_back(static_cast<int>(native_value));
    }
    return result;
}

class nixl_py_request_slot_execution {
public:
    nixl_py_request_slot_execution(nixlAgent &agent,
                                   const nixl_xfer_op_t operation,
                                   const uintptr_t local_side,
                                   const py::object &local_indices,
                                   const uintptr_t remote_side,
                                   const py::object &remote_indices,
                                   std::string notif_msg,
                                   const std::vector<uintptr_t> &backends,
                                   py::object running_state,
                                   py::object completed_state,
                                   py::object failed_state,
                                   py::object local_owner,
                                   py::object remote_owner)
        : agent_(&agent),
          operation_(operation),
          local_side_(reinterpret_cast<nixlDlistH *>(local_side)),
          remote_side_(reinterpret_cast<nixlDlistH *>(remote_side)),
          local_indices_(copy_xfer_indices(local_indices)),
          remote_indices_(copy_xfer_indices(remote_indices)),
          extra_params_(
              make_validated_opt_args(agent, backends, "request-slot execution")),
          local_owner_(std::move(local_owner)),
          remote_owner_(std::move(remote_owner)),
          owner_(agent),
          running_state_(std::move(running_state)),
          completed_state_(std::move(completed_state)),
          failed_state_(std::move(failed_state)) {
        if (local_side_ == nullptr || remote_side_ == nullptr) {
            throw std::invalid_argument("request-slot descriptor handles must be non-zero");
        }
        if (local_indices_.size() != remote_indices_.size()) {
            throw std::invalid_argument("request-slot index selections must have equal length");
        }
        if (!notif_msg.empty()) {
            extra_params_.notif.emplace(std::move(notif_msg));
        }
        auto &local_native_owner = validate_dlist_owner(local_owner_, local_side, "local");
        auto &remote_native_owner = validate_dlist_owner(remote_owner_, remote_side, "remote");
        local_native_owner_ = &local_native_owner;
        remote_native_owner_ = &remote_native_owner;
        local_native_owner.acquire_lease(local_side);
        try {
            remote_native_owner.acquire_lease(remote_side);
        }
        catch (...) {
            local_native_owner.release_lease();
            throw;
        }
        dlist_leases_active_ = true;
    }

    nixl_py_request_slot_execution(const nixl_py_request_slot_execution &) = delete;
    nixl_py_request_slot_execution &operator=(const nixl_py_request_slot_execution &) = delete;

    ~nixl_py_request_slot_execution() noexcept {
        // Explicit close is required for reportable errors. Finalization still
        // attempts to quiesce the request first; if it cannot, the outstanding
        // descriptor leases make their non-throwing destructors leak safely
        // instead of releasing storage beneath ambiguous native state.
        if (!owner_.released()) {
            try {
                (void)owner_.release_status();
            }
            catch (...) {
            }
        }
        if (owner_.released()) {
            release_dlist_leases();
        }
    }

    void
    start() {
        check_startable();
        begin_start();
        {
            py::gil_scoped_release release;
            start_native<false>(0,
                                std::nullopt,
                                std::chrono::steady_clock::time_point{},
                                nullptr);
        }
    }

    void
    start_with_notification(std::optional<std::string> notification) {
        validate_notification_override(notification);
        check_startable();
        begin_start();
        {
            py::gil_scoped_release release;
            start_native<true>(0,
                               std::nullopt,
                               std::chrono::steady_clock::time_point{},
                               &notification);
        }
    }

    py::object
    start_and_poll(const int64_t max_polls, const std::optional<int64_t> timeout_ns) {
        validate_start_poll_args(max_polls, timeout_ns);
        check_startable();
        const auto started = timeout_ns.has_value() && max_polls > 0
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        begin_start();
        {
            py::gil_scoped_release release;
            start_native<false>(max_polls, timeout_ns, started, nullptr);
        }
        return state_object();
    }

    py::object
    start_and_poll_with_notification(std::optional<std::string> notification,
                                     const int64_t max_polls,
                                     const std::optional<int64_t> timeout_ns) {
        validate_notification_override(notification);
        validate_start_poll_args(max_polls, timeout_ns);
        check_startable();
        const auto started = timeout_ns.has_value() && max_polls > 0
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        begin_start();
        {
            py::gil_scoped_release release;
            start_native<true>(max_polls, timeout_ns, started, &notification);
        }
        return state_object();
    }

    py::object
    poll_state() {
        if (state_ == state_t::running || state_ == state_t::ambiguous) {
            py::gil_scoped_release release;
            poll_once_native();
        }
        return state_object();
    }

    py::object
    poll_bounded(const int64_t max_polls,
                 const std::optional<int64_t> timeout_ns,
                 const int64_t timeout_check_interval) {
        validate_poll_args(max_polls, timeout_ns);
        if (timeout_check_interval <= 0) {
            throw std::invalid_argument("timeout_check_interval must be positive");
        }
        if (state_ == state_t::running || state_ == state_t::ambiguous) {
            py::gil_scoped_release release;
            poll_bounded_native(max_polls, timeout_ns, timeout_check_interval);
        }
        return state_object();
    }

    bool
    cancel() {
        if (state_ == state_t::running || state_ == state_t::ambiguous) {
            py::gil_scoped_release release;
            poll_once_native();
        }
        // NIXL's release operation cannot distinguish a completed race from a
        // successful cancellation, so this dispatch remains observational.
        return false;
    }

    void
    recycle() {
        if (state_ == state_t::closed) {
            throw std::runtime_error("NIXL request-slot execution is closed");
        }
        if (state_ == state_t::running || state_ == state_t::ambiguous) {
            throw std::runtime_error("cannot recycle active NIXL request-slot execution");
        }
        // Keep COMPLETED and FAILED observable across the interruptible Core
        // idle publication. The next start consumes a completed receipt, or a
        // retired failure creates a fresh RAII-owned request.
        retired_ = true;
    }

    void
    close() {
        if (state_ == state_t::closed) {
            // A pending Python signal can arrive after the native request and
            // CLOSED state commit but before the first call reaches this
            // epilogue. A same-operation retry repairs the descriptor leases.
            release_dlist_leases();
            return;
        }
        if (state_ == state_t::running || state_ == state_t::ambiguous ||
            (state_ == state_t::failed && !retired_)) {
            throw std::runtime_error("cannot close active NIXL request-slot execution");
        }
        nixl_status_t release_status = NIXL_SUCCESS;
        bool release_threw = false;
        {
            py::gil_scoped_release release;
            try {
                release_status = owner_.release_status();
            }
            catch (...) {
                release_threw = true;
                release_status = NIXL_ERR_UNKNOWN;
            }
            if (!release_threw && release_status == NIXL_SUCCESS) {
                // Commit CLOSED before GIL reacquisition can deliver a pending
                // Python signal. Retrying close is then a safe no-op.
                state_ = state_t::closed;
                retired_ = true;
            }
        }
        if (release_threw) {
            throw std::runtime_error("NIXL request-slot release threw an exception");
        }
        throw_nixl_exception(release_status);
        release_dlist_leases();
    }

    bool
    active() const noexcept {
        return state_ == state_t::running || state_ == state_t::ambiguous ||
               (state_ == state_t::completed_receipt && !retired_) ||
               (state_ == state_t::failed && !retired_);
    }

    bool
    failed() const noexcept {
        return state_ == state_t::failed;
    }

    uint64_t
    failure_epoch() const noexcept {
        return failure_epoch_;
    }

    std::string
    failure_message() const {
        if (failure_stage_ == failure_stage_t::none) {
            return {};
        }
        std::string result = "NIXL request-slot ";
        switch (failure_stage_) {
        case failure_stage_t::make:
            result += "creation";
            break;
        case failure_stage_t::post:
            result += "post";
            break;
        case failure_stage_t::poll:
            result += "status poll";
            break;
        case failure_stage_t::release:
            result += "failure cleanup";
            break;
        case failure_stage_t::none:
            break;
        }
        result += " failed with ";
        result += nixlEnumStrings::statusStr(failure_status_);
        if (release_status_ < 0) {
            result += "; release remains ambiguous with ";
            result += nixlEnumStrings::statusStr(release_status_);
        }
        return result;
    }

private:
    enum class state_t { cold, running, completed_receipt, ambiguous, failed, closed };
    enum class failure_stage_t { none, make, post, poll, release };

    nixl_py_owned_dlist_handle &
    validate_dlist_owner(py::object &owner,
                         const uintptr_t expected_handle,
                         const char *side) {
        if (!py::isinstance<nixl_py_owned_dlist_handle>(owner)) {
            throw std::invalid_argument(std::string(side) +
                                        " request-slot descriptor owner must be a "
                                        "nixlOwnedDlistHandle");
        }
        auto &native_owner = owner.cast<nixl_py_owned_dlist_handle &>();
        if (!native_owner.owned_by(*agent_)) {
            throw std::invalid_argument(
                std::string(side) +
                " request-slot descriptor owner belongs to a different NIXL agent");
        }
        if (native_owner.released()) {
            throw std::invalid_argument(std::string(side) +
                                        " request-slot descriptor owner is released");
        }
        if (native_owner.value() != expected_handle) {
            throw std::invalid_argument(std::string(side) +
                                        " request-slot descriptor owner does not match handle");
        }
        return native_owner;
    }

    void
    release_dlist_leases() noexcept {
        if (!dlist_leases_active_) {
            return;
        }
        // Both references remain alive as members. This also works when both
        // sides intentionally use the same owner: construction acquires two
        // leases and teardown releases two leases.
        local_native_owner_->release_lease();
        remote_native_owner_->release_lease();
        dlist_leases_active_ = false;
    }

    void
    check_startable() const {
        if (state_ == state_t::closed) {
            throw std::runtime_error("NIXL request-slot execution is closed");
        }
        if (state_ == state_t::running || state_ == state_t::ambiguous ||
            (state_ == state_t::failed && !retired_)) {
            throw std::runtime_error("NIXL request-slot execution is already active");
        }
    }

    static void
    validate_start_poll_args(const int64_t max_polls,
                             const std::optional<int64_t> timeout_ns) {
        if (max_polls < 0) {
            throw std::invalid_argument("max_polls must be non-negative");
        }
        if (timeout_ns.has_value() && *timeout_ns < 0) {
            throw std::invalid_argument("timeout_ns must be non-negative or None");
        }
    }

    static void
    validate_poll_args(const int64_t max_polls,
                       const std::optional<int64_t> timeout_ns) {
        if (max_polls <= 0) {
            throw std::invalid_argument("max_polls must be positive");
        }
        if (timeout_ns.has_value() && *timeout_ns < 0) {
            throw std::invalid_argument("timeout_ns must be non-negative or None");
        }
    }

    static void
    validate_notification_override(const std::optional<std::string> &notification) {
        if (notification.has_value() && notification->empty()) {
            throw std::invalid_argument(
                "notification override must be non-empty or None");
        }
    }

    void
    clear_failure() noexcept {
        failure_stage_ = failure_stage_t::none;
        failure_status_ = NIXL_SUCCESS;
        release_status_ = NIXL_SUCCESS;
    }

    void
    begin_start() noexcept {
        const bool clean_receipt =
            state_ == state_t::completed_receipt && !retired_;
        state_ = state_t::running;
        if (!clean_receipt) {
            retired_ = false;
            clear_failure();
        }
    }

    template<bool HasNotificationOverride>
    void
    start_native(const int64_t max_polls,
                 const std::optional<int64_t> timeout_ns,
                 const std::chrono::steady_clock::time_point started,
                 std::optional<std::string> *notification_override) noexcept {
        try {
            if (owner_.released()) {
                nixlXferReqH *handle = nullptr;
                const nixl_status_t make_status = agent_->makeXferReq(
                    operation_,
                    *local_side_,
                    std::span<const int>(local_indices_),
                    *remote_side_,
                    std::span<const int>(remote_indices_),
                    handle,
                    &extra_params_);
                if (handle != nullptr) {
                    // Ownership becomes durable while the GIL is still
                    // released, before any signal can cross into Python.
                    owner_.adopt(handle);
                }
                if (make_status != NIXL_SUCCESS || owner_.released()) {
                    fail_and_quiesce(failure_stage_t::make,
                                     make_status < 0 ? make_status : NIXL_ERR_UNKNOWN);
                    return;
                }
            }

            nixl_status_t post_status;
            if constexpr (HasNotificationOverride) {
                if (notification_override == nullptr) {
                    fail_and_quiesce(failure_stage_t::post, NIXL_ERR_INVALID_PARAM);
                    return;
                }
                nixl_opt_args_t post_options;
                post_options.notif = std::move(*notification_override);
                // A non-null options pointer is the explicit override
                // boundary. A disengaged optional clears a tag retained by a
                // prior repost.
                post_status = agent_->postXferReq(owner_.get(), &post_options);
            }
            else {
                // Compile the fixed specialization without constructing
                // options or branching on request-local notification state.
                post_status = agent_->postXferReq(owner_.get());
            }
            if (post_status < 0) {
                fail_and_quiesce(failure_stage_t::post, post_status);
                return;
            }
            if (post_status == NIXL_SUCCESS) {
                state_ = state_t::completed_receipt;
                return;
            }
            if (post_status != NIXL_IN_PROG) {
                fail_and_quiesce(failure_stage_t::post, NIXL_ERR_UNKNOWN);
                return;
            }
            if (max_polls > 0) {
                poll_after_start_native(max_polls, timeout_ns, started);
            }
        }
        catch (...) {
            fail_and_quiesce(failure_stage_t::post, NIXL_ERR_UNKNOWN);
        }
    }

    void
    poll_after_start_native(const int64_t max_polls,
                            const std::optional<int64_t> timeout_ns,
                            const std::chrono::steady_clock::time_point started) noexcept {
        for (int64_t poll = 0; poll < max_polls; ++poll) {
            if (!observe_status_native(failure_stage_t::poll) || poll + 1 == max_polls) {
                return;
            }
            if (timeout_ns.has_value()) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                if (elapsed >= *timeout_ns) {
                    return;
                }
            }
        }
    }

    void
    poll_once_native() noexcept {
        if (state_ == state_t::ambiguous) {
            retry_ambiguous_release();
            return;
        }
        if (state_ == state_t::running) {
            observe_status_native(failure_stage_t::poll);
        }
    }

    void
    poll_bounded_native(const int64_t max_polls,
                        const std::optional<int64_t> timeout_ns,
                        const int64_t timeout_check_interval) noexcept {
        if (state_ == state_t::ambiguous) {
            retry_ambiguous_release();
            return;
        }
        if (state_ != state_t::running) {
            return;
        }
        if (!timeout_ns.has_value()) {
            for (int64_t poll = 0; poll < max_polls; ++poll) {
                if (!observe_status_native(failure_stage_t::poll)) {
                    return;
                }
            }
            return;
        }

        const auto started = std::chrono::steady_clock::now();
        if (!observe_status_native(failure_stage_t::poll) || max_polls == 1) {
            return;
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
        if (elapsed >= *timeout_ns) {
            return;
        }
        int64_t polls_until_timeout_check = timeout_check_interval - 1;
        if (polls_until_timeout_check == 0) {
            polls_until_timeout_check = 1;
        }
        for (int64_t poll = 1; poll < max_polls; ++poll) {
            if (!observe_status_native(failure_stage_t::poll) || poll + 1 == max_polls) {
                return;
            }
            if (--polls_until_timeout_check == 0) {
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - started)
                              .count();
                if (elapsed >= *timeout_ns) {
                    return;
                }
                polls_until_timeout_check = timeout_check_interval;
            }
        }
    }

    bool
    observe_status_native(const failure_stage_t stage) noexcept {
        try {
            const nixl_status_t status = agent_->getXferStatus(owner_.get());
            if (status == NIXL_SUCCESS) {
                state_ = state_t::completed_receipt;
                return false;
            } else if (status == NIXL_IN_PROG) {
                // begin_start() already published RUNNING, and every caller
                // enters this helper only while that state is current. Avoid
                // a redundant store on every continuing status observation.
                return true;
            } else {
                fail_and_quiesce(stage, status < 0 ? status : NIXL_ERR_UNKNOWN);
                return false;
            }
        }
        catch (...) {
            fail_and_quiesce(stage, NIXL_ERR_UNKNOWN);
            return false;
        }
    }

    void
    fail_and_quiesce(const failure_stage_t stage, const nixl_status_t status) noexcept {
        failure_stage_ = stage;
        failure_status_ = status;
        release_status_ = NIXL_SUCCESS;
        ++failure_epoch_;
        if (owner_.released()) {
            state_ = state_t::failed;
            return;
        }
        try {
            release_status_ = owner_.release_status();
        }
        catch (...) {
            release_status_ = NIXL_ERR_UNKNOWN;
        }
        state_ = owner_.released() ? state_t::failed : state_t::ambiguous;
    }

    void
    retry_ambiguous_release() noexcept {
        try {
            release_status_ = owner_.release_status();
        }
        catch (...) {
            release_status_ = NIXL_ERR_UNKNOWN;
        }
        if (owner_.released()) {
            state_ = state_t::failed;
        }
    }

    py::object
    state_object() const {
        switch (state_) {
        case state_t::cold:
            return py::none();
        case state_t::running:
        case state_t::ambiguous:
            return running_state_;
        case state_t::completed_receipt:
            return completed_state_;
        case state_t::failed:
            return failed_state_;
        case state_t::closed:
            throw std::runtime_error("NIXL request-slot execution is closed");
        }
        throw std::runtime_error("invalid NIXL request-slot execution state");
    }

    nixlAgent *agent_;
    nixl_xfer_op_t operation_;
    nixlDlistH *local_side_;
    nixlDlistH *remote_side_;
    std::vector<int> local_indices_;
    std::vector<int> remote_indices_;
    nixl_opt_args_t extra_params_;
    // Retain the RAII dlist owners until the transfer-request owner has been
    // destroyed. Raw integer handles alone cannot prevent a direct caller from
    // dropping the prepared descriptor lists before lazy request creation.
    py::object local_owner_;
    py::object remote_owner_;
    nixl_py_owned_xfer_handle owner_;
    py::object running_state_;
    py::object completed_state_;
    py::object failed_state_;
    state_t state_ = state_t::cold;
    failure_stage_t failure_stage_ = failure_stage_t::none;
    nixl_status_t failure_status_ = NIXL_SUCCESS;
    nixl_status_t release_status_ = NIXL_SUCCESS;
    uint64_t failure_epoch_ = 0;
    bool retired_ = false;
    // Lease-only bookkeeping is cold and deliberately follows every field
    // used by start/repost/poll, preserving the hot dispatch layout.
    nixl_py_owned_dlist_handle *local_native_owner_ = nullptr;
    nixl_py_owned_dlist_handle *remote_native_owner_ = nullptr;
    bool dlist_leases_active_ = false;
};

template<typename DlistT>
nixlDlistH *
prep_xfer_dlist(const nixlAgent &agent,
                const std::string &agent_name,
                const DlistT &descs,
                const std::vector<uintptr_t> &backends,
                nixl_py_owned_dlist_handle *owned = nullptr) {
    const nixl_opt_args_t extra_params = make_opt_args(backends);
    nixlDlistH *handle = nullptr;
    const nixl_status_t ret =
        agent.prepXferDlist(agent_name, descs, handle, &extra_params);
    if (owned != nullptr && handle != nullptr) {
        // Publish native ownership before a released-GIL caller can reacquire
        // and deliver a pending Python signal.
        owned->adopt(handle);
    }
    throw_nixl_exception(ret);
    return handle;
}

bool
is_native_signed_int32_buffer(const py::buffer_info &info) noexcept {
    static_assert(sizeof(int) == sizeof(std::int32_t), "NIXL indices require 32-bit int");

    const Py_buffer *view = info.view();
    if (view == nullptr || info.ndim != 1 || info.itemsize != sizeof(int) || info.size < 0 ||
        info.shape.size() != 1 || info.strides.size() != 1 || info.shape[0] != info.size ||
        (info.ptr == nullptr && info.size != 0) ||
        info.size > PY_SSIZE_T_MAX / info.itemsize ||
        view->len != info.size * info.itemsize || PyBuffer_IsContiguous(view, 'C') != 1 ||
        (view->suboffsets != nullptr && view->suboffsets[0] >= 0)) {
        return false;
    }

    if (info.format == "i" || info.format == "@i" || info.format == "=i") {
        return true;
    }
    const std::uint16_t endian_probe = 1;
    const bool native_is_little =
        *reinterpret_cast<const unsigned char *>(&endian_probe) == 1;
    return info.format == (native_is_little ? "<i" : ">i");
}

nixlXferReqH *
make_xfer_req(nixlAgent &agent,
              const nixl_xfer_op_t &operation,
              uintptr_t local_side,
              const py::object &local_indices,
              uintptr_t remote_side,
              const py::object &remote_indices,
              std::string notif_msg,
              const std::vector<uintptr_t> &backends,
              bool skip_desc_merge,
              nixl_py_owned_xfer_handle *owned = nullptr) {
    nixlXferReqH *handle = nullptr;
    nixl_opt_args_t extra_params = make_opt_args(backends);

    if (!local_side || !remote_side) {
        throw nixlInvalidParamError("local_side and remote_side must be valid pointers");
    }

    if (notif_msg.size() > 0) {
        extra_params.notif.emplace(std::move(notif_msg));
    }
    extra_params.skipDescMerge = skip_desc_merge;
    std::vector<int> local_indices_vec;
    std::vector<int> remote_indices_vec;
    // An active Py_buffer export, unlike a borrowed Python object reference,
    // pins resizable exporter storage while makeXferReq runs without the GIL.
    // These owners are deliberately scoped outside the released-GIL block so
    // PyBuffer_Release runs only after the GIL has been reacquired.
    std::optional<py::buffer_info> local_indices_export;
    std::optional<py::buffer_info> remote_indices_export;

    auto indices_to_span = [](const py::object &indices,
                              std::vector<int> &backing,
                              std::optional<py::buffer_info> &active_export)
        -> std::span<const int> {
        if (PyObject_CheckBuffer(indices.ptr())) {
            auto buffer = py::reinterpret_borrow<py::buffer>(indices);
            auto info = buffer.request(false);
            if (is_native_signed_int32_buffer(info)) {
                const auto count = static_cast<size_t>(info.size);
                const auto address = reinterpret_cast<std::uintptr_t>(info.ptr);
                // A writable exporter can be mutated by another Python thread
                // as soon as this binding releases the GIL. Borrow only an
                // immutable, naturally aligned buffer (the Core fast path);
                // snapshot writable or unaligned storage before GIL release.
                if (info.view()->readonly != 0 && address % alignof(int) == 0) {
                    active_export.emplace(std::move(info));
                    return std::span<const int>(
                        static_cast<const int *>(active_export->ptr), count);
                }

                // Forming and dereferencing an unaligned int pointer is
                // undefined behavior even when the host ISA tolerates it.
                // memcpy also gives a writable exporter a stable native
                // snapshot before another Python thread can run.
                backing.resize(count);
                if (count != 0) {
                    std::memcpy(backing.data(), info.ptr, count * sizeof(int));
                }
                return std::span<const int>(backing);
            }
        }

        if (py::isinstance<py::array>(indices)) {
            const auto indices_array = indices.cast<py::array>();
            if (indices_array.ndim() != 1)
                throw std::invalid_argument("indices numpy array must be 1D");

            // TODO: compatibility with previous version, to be removed
            using int_array_t = py::array_t<int, py::array::c_style | py::array::forcecast>;
            const auto converted = indices.cast<int_array_t>();
            backing.assign(converted.data(), converted.data() + converted.size());
            return std::span<const int>(backing);
        }
        backing = indices.cast<std::vector<int>>();
        return std::span<const int>(backing);
    };

    const auto local_span =
        indices_to_span(local_indices, local_indices_vec, local_indices_export);
    const auto remote_span =
        indices_to_span(remote_indices, remote_indices_vec, remote_indices_export);
    if (local_span.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        remote_span.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("indices length exceeds native int range");
    }

    const auto local_dlist = reinterpret_cast<nixlDlistH *>(local_side);
    const auto remote_dlist = reinterpret_cast<nixlDlistH *>(remote_side);

    {
        py::gil_scoped_release release;
        const nixl_status_t ret = agent.makeXferReq(operation,
                                                    *local_dlist,
                                                    local_span,
                                                    *remote_dlist,
                                                    remote_span,
                                                    handle,
                                                    &extra_params);
        if (owned != nullptr && handle != nullptr) {
            // This must precede gil_scoped_release destruction. Otherwise a
            // pending signal on GIL reacquire can strand the raw request.
            owned->adopt(handle);
        }
        throw_nixl_exception(ret);
    }
    return handle;
}

nixlXferReqH *
create_xfer_req(nixlAgent &agent,
                const nixl_xfer_op_t &operation,
                const nixl_xfer_dlist_t &local_descs,
                const nixl_xfer_dlist_t &remote_descs,
                const std::string &remote_agent,
                std::string notif_msg,
                const std::vector<uintptr_t> &backends,
                nixl_py_owned_xfer_handle *owned = nullptr) {
    nixlXferReqH *handle = nullptr;
    nixl_opt_args_t extra_params = make_opt_args(backends);

    if (notif_msg.size() > 0) {
        extra_params.notif.emplace(std::move(notif_msg));
    }
    const nixl_status_t ret = agent.createXferReq(
        operation, local_descs, remote_descs, remote_agent, handle, &extra_params);
    if (owned != nullptr && handle != nullptr) {
        owned->adopt(handle);
    }
    throw_nixl_exception(ret);
    return handle;
}

template<typename DlistT>
uintptr_t
prep_mem_view(const nixlAgent &agent, const DlistT &dlist, const std::vector<uintptr_t> &backends) {
    const nixl_opt_args_t extra_params = make_opt_args(backends);
    nixlMemViewH mvh;
    throw_nixl_exception(agent.prepMemView(dlist, mvh, &extra_params));
    return reinterpret_cast<uintptr_t>(mvh);
}
} // namespace

PYBIND11_MODULE(_bindings, m) {

    // TODO: each nixl class and/or function can be documented in place
    m.doc() = "pybind11 NIXL plugin: Implements NIXL descriptors and lists, as well as bindings of "
              "NIXL CPP APIs";

    m.attr("NIXL_INIT_AGENT") = NIXL_INIT_AGENT;

    m.attr("DEFAULT_COMM_PORT") = default_comm_port;

    // Whether NIXL was built against a UCX with the GPU device API, which the
    // prepMemView/releaseMemView path requires. Mirrors the meson
    // HAVE_UCX_GPU_DEVICE_API gate so callers (and tests) can probe support.
#ifdef HAVE_UCX_GPU_DEVICE_API
    m.attr("HAVE_UCX_GPU_DEVICE_API") = true;
#else
    m.attr("HAVE_UCX_GPU_DEVICE_API") = false;
#endif

    py::class_<nixl_py_xfer_release_state>(m, "nixlXferReleaseState")
        .def(py::init<uintptr_t>(), py::arg("handle"))
        .def_property_readonly("released", &nixl_py_xfer_release_state::released);
    py::class_<nixl_py_dlist_release_state>(m, "nixlDlistReleaseState")
        .def(py::init<uintptr_t>(), py::arg("handle"))
        .def_property_readonly("released", &nixl_py_dlist_release_state::released);
    py::class_<nixl_py_owned_xfer_handle>(m, "nixlOwnedXferHandle")
        .def_property_readonly("value", &nixl_py_owned_xfer_handle::value)
        .def_property_readonly("released", &nixl_py_owned_xfer_handle::released)
        .def("release", &nixl_py_owned_xfer_handle::release);
    py::class_<nixl_py_owned_dlist_handle>(m, "nixlOwnedDlistHandle")
        .def_property_readonly("value", &nixl_py_owned_dlist_handle::value)
        .def_property_readonly("released", &nixl_py_owned_dlist_handle::released)
        .def("release", &nixl_py_owned_dlist_handle::release);
    py::class_<nixl_py_mem_deregistration>(m, "nixlMemDeregistration")
        .def_property_readonly("completed", &nixl_py_mem_deregistration::completed)
        .def("execute",
             &nixl_py_mem_deregistration::execute_bound,
             py::call_guard<py::gil_scoped_release>());
    py::class_<nixl_py_notification_receiver>(m, "nixlNotificationReceiver")
        .def(
            "poll",
            &nixl_py_notification_receiver::poll,
            R"pbdoc(
Destructively drain new notifications into a fresh dict of source tuples.

Receivers and one-shot calls on one agent consume the same queues; they are not
subscriptions. An exception can follow partial native consumption. Concurrency
follows the agent sync mode: callers must serialize NONE mode. The receiver
keeps its agent alive, and every explicit backend handle supplied at creation
must belong to that agent.
)pbdoc")
        .def(
            "poll_bounded",
            &nixl_py_notification_receiver::poll_bounded,
            R"pbdoc(
Drain at most one native batch and materialize at most ``max_items`` payloads.

The receiver validates the complete just-drained native batch against the item,
payload-byte, and per-payload limits before constructing Python objects. Any
remainder stays in receiver-owned C++ spill storage. Consumed C++ strings remain
owned until the complete spill batch retires, so spill plus returned Python bytes
can transiently approach twice the byte cap. The limits bound one destructive
drain and each Python materialization, not a hard combined-memory peak or the
backend's queue before ``getNotifs``.
)pbdoc",
            py::arg("max_items"),
            py::arg("max_batch_items"),
            py::arg("max_batch_bytes"),
            py::arg("max_payload_bytes"));
    py::class_<nixl_py_notification_sender>(m, "nixlNotificationSender")
        .def(
            "send",
            &nixl_py_notification_sender::send,
            R"pbdoc(
Send one payload using the remote name and backend selection fixed at creation.

Successful ``None`` return means only that the selected NIXL backend accepted the
payload; it does not mean the remote application observed it. The sender keeps
its agent alive, releases the GIL around native submission, and follows the
agent sync-mode concurrency contract.
)pbdoc",
            py::arg("msg"),
            py::call_guard<py::gil_scoped_release>());
    py::class_<nixl_py_request_slot_execution>(m, "nixlRequestSlotExecution")
        .def("start", &nixl_py_request_slot_execution::start)
        .def("start_with_notification",
             &nixl_py_request_slot_execution::start_with_notification,
             py::arg("notification") = std::nullopt)
        .def("start_and_poll",
             &nixl_py_request_slot_execution::start_and_poll,
             py::arg("max_polls"),
             py::arg("timeout_ns") = std::nullopt)
        .def("start_and_poll_with_notification",
             &nixl_py_request_slot_execution::start_and_poll_with_notification,
             py::arg("notification"),
             py::arg("max_polls"),
             py::arg("timeout_ns") = std::nullopt)
        .def("poll_state", &nixl_py_request_slot_execution::poll_state)
        .def("poll_bounded",
             &nixl_py_request_slot_execution::poll_bounded,
             py::arg("max_polls"),
             py::arg("timeout_ns"),
             py::arg("timeout_check_interval"))
        .def("cancel", &nixl_py_request_slot_execution::cancel)
        .def("recycle", &nixl_py_request_slot_execution::recycle)
        .def("close", &nixl_py_request_slot_execution::close)
        .def_property_readonly("active", &nixl_py_request_slot_execution::active)
        .def_property_readonly("failed", &nixl_py_request_slot_execution::failed)
        .def_property_readonly("failure_epoch",
                               &nixl_py_request_slot_execution::failure_epoch)
        .def_property_readonly("failure_message",
                               &nixl_py_request_slot_execution::failure_message);

    // cast types
    py::enum_<nixl_thread_sync_t>(m, "nixl_thread_sync_t")
        .value("NIXL_THREAD_SYNC_NONE", nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE)
        .value("NIXL_THREAD_SYNC_STRICT", nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT)
        .value("NIXL_THREAD_SYNC_RW", nixl_thread_sync_t::NIXL_THREAD_SYNC_RW)
        .value("NIXL_THREAD_SYNC_DEFAULT", nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT)
        .export_values();

    py::enum_<nixl_mem_t>(m, "nixl_mem_t")
        .value("DRAM_SEG", DRAM_SEG)
        .value("VRAM_SEG", VRAM_SEG)
        .value("BLK_SEG", BLK_SEG)
        .value("OBJ_SEG", OBJ_SEG)
        .value("FILE_SEG", FILE_SEG)
        .export_values();

    py::enum_<nixl_xfer_op_t>(m, "nixl_xfer_op_t")
        .value("NIXL_READ", NIXL_READ)
        .value("NIXL_WRITE", NIXL_WRITE)
        .export_values();

    py::enum_<nixl_cost_t>(m, "nixl_cost_t")
        .value("NIXL_COST_ANALYTICAL_BACKEND", nixl_cost_t::ANALYTICAL_BACKEND)
        .export_values();

    py::enum_<nixl_status_t>(m, "nixl_status_t")
        .value("NIXL_IN_PROG", NIXL_IN_PROG)
        .value("NIXL_SUCCESS", NIXL_SUCCESS)
        .value("NIXL_ERR_NOT_POSTED", NIXL_ERR_NOT_POSTED)
        .value("NIXL_ERR_INVALID_PARAM", NIXL_ERR_INVALID_PARAM)
        .value("NIXL_ERR_BACKEND", NIXL_ERR_BACKEND)
        .value("NIXL_ERR_NOT_FOUND", NIXL_ERR_NOT_FOUND)
        .value("NIXL_ERR_MISMATCH", NIXL_ERR_MISMATCH)
        .value("NIXL_ERR_NOT_ALLOWED", NIXL_ERR_NOT_ALLOWED)
        .value("NIXL_ERR_REPOST_ACTIVE", NIXL_ERR_REPOST_ACTIVE)
        .value("NIXL_ERR_UNKNOWN", NIXL_ERR_UNKNOWN)
        .value("NIXL_ERR_NOT_SUPPORTED", NIXL_ERR_NOT_SUPPORTED)
        .export_values();

    py::class_<nixl_xfer_telem_t>(m, "nixlXferTelemetry")
        .def(py::init<>())
        .def_property_readonly("startTime",
                               [](const nixl_xfer_telem_t &t) {
                                   return std::chrono::duration_cast<chrono_period_us_t>(
                                              t.startTime.time_since_epoch())
                                       .count();
                               })
        .def_property_readonly("postDuration",
                               [](const nixl_xfer_telem_t &t) { return t.postDuration.count(); })
        .def_property_readonly("xferDuration",
                               [](const nixl_xfer_telem_t &t) { return t.xferDuration.count(); })
        .def_readonly("totalBytes", &nixl_xfer_telem_t::totalBytes)
        .def_readonly("descCount", &nixl_xfer_telem_t::descCount);


    py::register_exception<nixlNotPostedError>(m, "nixlNotPostedError");
    py::register_exception<nixlInvalidParamError>(m, "nixlInvalidParamError");
    py::register_exception<nixlBackendError>(m, "nixlBackendError");
    py::register_exception<nixlNotFoundError>(m, "nixlNotFoundError");
    py::register_exception<nixlMismatchError>(m, "nixlMismatchError");
    py::register_exception<nixlNotAllowedError>(m, "nixlNotAllowedError");
    py::register_exception<nixlRepostActiveError>(m, "nixlRepostActiveError");
    py::register_exception<nixlUnknownError>(m, "nixlUnknownError");
    py::register_exception<nixlNotSupportedError>(m, "nixlNotSupportedError");
    py::register_exception<nixlRemoteDisconnectError>(m, "nixlRemoteDisconnectError");
    py::register_exception<nixlCancelledError>(m, "nixlCancelledError");
    py::register_exception<nixlNoTelemetryError>(m, "nixlNoTelemetryError");

    py::class_<nixl_xfer_dlist_t>(m, "nixlXferDList")
        .def(py::init<nixl_mem_t, int>(), py::arg("type"), py::arg("init_size") = 0)
        .def(py::init([](nixl_mem_t mem, py::array descs) {
                 static_assert(sizeof(nixlBasicDesc) == 3 * sizeof(uint64_t),
                               "nixlBasicDesc size mismatch");
                 // Check array shape and dtype
                 if (descs.ndim() != 2 || descs.shape(1) != 3)
                     throw std::invalid_argument("descs must be a Nx3 numpy array");
                 if (!py::dtype::of<uint64_t>().equal(descs.dtype()) &&
                     !py::dtype::of<int64_t>().equal(descs.dtype()))
                     throw std::invalid_argument(
                         "descs must be a Nx3 numpy array of uint64 or int64");
                 if (!(descs.flags() & py::array::c_style)) {
                     throw std::invalid_argument("descs must be a C-contiguous numpy array");
                 }
                 size_t n = descs.shape(0);
                 nixl_xfer_dlist_t new_list(mem, n);
                 // We assume that the Nx3 array matches the nixlBasicDesc layout so we can simply
                 // memcpy
                 std::memcpy(&new_list[0], descs.data(), descs.size() * sizeof(uint64_t));

                 return new_list;
             }),
             py::arg("type"),
             py::arg("descs").noconvert())
        .def(py::init([](nixl_mem_t mem, py::list descs) {
                 nixl_xfer_dlist_t new_list(mem, descs.size());
                 for (size_t i = 0; i < descs.size(); i++) {
                     if (!py::isinstance<py::tuple>(descs[i])) {
                         throw py::type_error(
                             "Each descriptor must be a tuple when provided as a list");
                     }
                     auto desc = py::reinterpret_borrow<py::tuple>(descs[i]);
                     if (desc.size() != 3) {
                         throw py::value_error(
                             "Each descriptor tuple must have exactly 3 elements");
                     }
                     new_list[i] = nixlBasicDesc(desc[0].cast<uintptr_t>(),
                                                 desc[1].cast<size_t>(),
                                                 desc[2].cast<uint64_t>());
                 }

                 return new_list;
             }),
             py::arg("type"),
             py::arg("descs").noconvert())
        .def("getType", &nixl_xfer_dlist_t::getType)
        .def("descCount", &nixl_xfer_dlist_t::descCount)
        .def("isEmpty", &nixl_xfer_dlist_t::isEmpty)
        .def(py::self == py::self)
        .def("__getitem__",
             [](nixl_xfer_dlist_t &list, unsigned int i) -> py::tuple {
                 nixlBasicDesc &desc = list[i];
                 return py::make_tuple(desc.addr, desc.len, desc.devId);
             })
        .def("__setitem__",
             [](nixl_xfer_dlist_t &list, unsigned int i, const py::tuple &desc) {
                 list[i] = nixlBasicDesc(
                     desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint64_t>());
             })
        .def("addDesc",
             [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlBasicDesc(
                     desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint64_t>()));
             })
        .def("append",
             [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlBasicDesc(
                     desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint64_t>()));
             })
        .def("index",
             [](nixl_xfer_dlist_t &list, const py::tuple &desc) {
                 int ret = (nixl_status_t)list.getIndex(nixlBasicDesc(
                     desc[0].cast<uintptr_t>(), desc[1].cast<size_t>(), desc[2].cast<uint64_t>()));
                 if (ret < 0) throw_nixl_exception((nixl_status_t)ret);
                 return (int)ret;
             })
        .def("remDesc", &nixl_xfer_dlist_t::remDesc)
        .def("clear", &nixl_xfer_dlist_t::clear)
        .def("print", &nixl_xfer_dlist_t::print)
        .def(py::pickle(
            [](const nixl_xfer_dlist_t &self) { // __getstate__
                nixlSerDes serdes;
                self.serialize(&serdes);
                return py::bytes(serdes.exportStr());
            },
            [](py::bytes serdes_str) { // __setstate__
                nixlSerDes serdes;
                serdes.importStr(std::string(serdes_str));
                nixl_xfer_dlist_t newObj = nixl_xfer_dlist_t(&serdes);
                return newObj;
            }));

    py::class_<nixl_remote_dlist_t>(m, "nixlRemoteDList")
        .def(py::init<nixl_mem_t, int>(), py::arg("type"), py::arg("init_size") = 0)
        .def(py::init([](nixl_mem_t mem, py::list descs) {
                 nixl_remote_dlist_t new_list(mem, descs.size());
                 for (size_t i = 0; i < descs.size(); i++) {
                     if (!py::isinstance<py::tuple>(descs[i])) {
                         throw py::type_error(
                             "Each descriptor must be a tuple when provided as a list");
                     }
                     auto desc = py::reinterpret_borrow<py::tuple>(descs[i]);
                     if (desc.size() != 4) {
                         throw py::value_error(
                             "Each descriptor must be (addr, len, dev_id, agent_name)");
                     }
                     new_list[i] = nixlRemoteDesc(desc[0].cast<uintptr_t>(),
                                                  desc[1].cast<size_t>(),
                                                  desc[2].cast<uint64_t>(),
                                                  desc[3].cast<std::string>());
                 }

                 return new_list;
             }),
             py::arg("type"),
             py::arg("descs").noconvert())
        .def("getType", &nixl_remote_dlist_t::getType)
        .def("descCount", &nixl_remote_dlist_t::descCount)
        .def("isEmpty", &nixl_remote_dlist_t::isEmpty)
        .def(py::self == py::self)
        .def("__getitem__",
             [](nixl_remote_dlist_t &list, unsigned int i) -> py::tuple {
                 nixlRemoteDesc &desc = list[i];
                 return py::make_tuple(desc.addr, desc.len, desc.devId, desc.remoteAgent);
             })
        .def("__setitem__",
             [](nixl_remote_dlist_t &list, unsigned int i, const py::tuple &desc) {
                 list[i] = nixlRemoteDesc(desc[0].cast<uintptr_t>(),
                                          desc[1].cast<size_t>(),
                                          desc[2].cast<uint64_t>(),
                                          desc[3].cast<std::string>());
             })
        .def("addDesc",
             [](nixl_remote_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlRemoteDesc(desc[0].cast<uintptr_t>(),
                                             desc[1].cast<size_t>(),
                                             desc[2].cast<uint64_t>(),
                                             desc[3].cast<std::string>()));
             })
        .def("append",
             [](nixl_remote_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlRemoteDesc(desc[0].cast<uintptr_t>(),
                                             desc[1].cast<size_t>(),
                                             desc[2].cast<uint64_t>(),
                                             desc[3].cast<std::string>()));
             })
        .def("remDesc", &nixl_remote_dlist_t::remDesc)
        .def("clear", &nixl_remote_dlist_t::clear)
        .def("print", &nixl_remote_dlist_t::print);

    py::class_<nixl_reg_dlist_t>(m, "nixlRegDList")
        .def(py::init<nixl_mem_t, int>(), py::arg("type"), py::arg("init_size") = 0)
        .def(py::init([](nixl_mem_t mem, py::array descs) {
            if (descs.ndim() != 2 || descs.shape(1) != 3)
                throw std::invalid_argument("descs must be a Nx3 numpy array");
            if (!py::dtype::of<uint64_t>().equal(descs.dtype()) &&
                !py::dtype::of<int64_t>().equal(descs.dtype()))
                throw std::invalid_argument("descs must be a Nx3 numpy array of uint64 or int64");
            if (!(descs.flags() & py::array::c_style)) {
                throw std::invalid_argument("descs must be a C-contiguous numpy array");
            }
            size_t n = descs.shape(0);
            nixl_reg_dlist_t new_list(mem, n);
            if (py::dtype::of<uint64_t>().equal(descs.dtype())) {
                auto buffer = descs.unchecked<uint64_t, 2>();
                for (size_t i = 0; i < n; i++) {
                    new_list[i] = nixlBlobDesc(buffer(i, 0), buffer(i, 1), buffer(i, 2), "");
                }
            } else {
                auto buffer = descs.unchecked<int64_t, 2>();
                for (size_t i = 0; i < n; i++) {
                    new_list[i] = nixlBlobDesc(buffer(i, 0), buffer(i, 1), buffer(i, 2), "");
                }
            }

            return new_list;
        }))
        .def(py::init([](nixl_mem_t mem, py::list descs) {
                 nixl_reg_dlist_t new_list(mem, descs.size());
                 for (size_t i = 0; i < descs.size(); i++) {
                     if (!py::isinstance<py::tuple>(descs[i])) {
                         throw py::type_error(
                             "Each descriptor must be a tuple when provided as a list");
                     }
                     auto desc = descs[i].cast<py::tuple>();
                     if (desc.size() != 4) {
                         throw py::value_error(
                             "Each descriptor tuple must have exactly 4 elements");
                     }
                     new_list[i] = nixlBlobDesc(desc[0].cast<uintptr_t>(),
                                                desc[1].cast<size_t>(),
                                                desc[2].cast<uint64_t>(),
                                                desc[3].cast<std::string>());
                 }

                 return new_list;
             }),
             py::arg("type"),
             py::arg("descs"))
        .def("getType", &nixl_reg_dlist_t::getType)
        .def("descCount", &nixl_reg_dlist_t::descCount)
        .def("isEmpty", &nixl_reg_dlist_t::isEmpty)
        .def(py::self == py::self)
        .def("__getitem__",
             [](nixl_reg_dlist_t &list, unsigned int i) -> py::tuple {
                 nixlBlobDesc desc = list[i];
                 return py::make_tuple(desc.addr, desc.len, desc.devId, py::bytes(desc.metaInfo));
             })
        .def("__setitem__",
             [](nixl_reg_dlist_t &list, unsigned int i, const py::tuple &desc) {
                 list[i] = nixlBlobDesc(desc[0].cast<uintptr_t>(),
                                        desc[1].cast<size_t>(),
                                        desc[2].cast<uint64_t>(),
                                        desc[3].cast<std::string>());
             })
        .def("addDesc",
             [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlBlobDesc(desc[0].cast<uintptr_t>(),
                                           desc[1].cast<size_t>(),
                                           desc[2].cast<uint64_t>(),
                                           desc[3].cast<std::string>()));
             })
        .def("append",
             [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                 list.addDesc(nixlBlobDesc(desc[0].cast<uintptr_t>(),
                                           desc[1].cast<size_t>(),
                                           desc[2].cast<uint64_t>(),
                                           desc[3].cast<std::string>()));
             })
        .def("index",
             [](nixl_reg_dlist_t &list, const py::tuple &desc) {
                 int ret = list.getIndex(nixlBlobDesc(desc[0].cast<uintptr_t>(),
                                                      desc[1].cast<size_t>(),
                                                      desc[2].cast<uint64_t>(),
                                                      desc[3].cast<std::string>()));
                 if (ret < 0) throw_nixl_exception((nixl_status_t)ret);
                 return ret;
             })
        .def("trim", &nixl_reg_dlist_t::trim)
        .def("remDesc", &nixl_reg_dlist_t::remDesc)
        .def("clear", &nixl_reg_dlist_t::clear)
        .def("print", &nixl_reg_dlist_t::print)
        .def(py::pickle(
            [](const nixl_reg_dlist_t &self) { // __getstate__
                nixlSerDes serdes;
                self.serialize(&serdes);
                return py::bytes(serdes.exportStr());
            },
            [](py::bytes serdes_str) { // __setstate__
                nixlSerDes serdes;
                serdes.importStr(std::string(serdes_str));
                nixl_reg_dlist_t newObj = nixl_reg_dlist_t(&serdes);
                return newObj;
            }));

    py::class_<nixlAgentConfig>(m, "nixlAgentConfig")
        .def(py::init<>())
        // legacy constructors kept for compatibility
        .def(py::init<bool>())
        .def(py::init<bool, bool>())
        .def(py::init<bool, bool, int>())
        .def(py::init<bool, bool, int, nixl_thread_sync_t>())
        .def(py::init<bool, bool, int, nixl_thread_sync_t, int>())
        .def(py::init<bool, bool, int, nixl_thread_sync_t, int, uint64_t>())
        .def(py::init<bool, bool, int, nixl_thread_sync_t, int, uint64_t, uint64_t>())
        .def(py::init<bool, bool, int, nixl_thread_sync_t, int, uint64_t, uint64_t, bool>())
        .def_readwrite("useProgThread", &nixlAgentConfig::useProgThread)
        .def_readwrite("useListenThread", &nixlAgentConfig::useListenThread)
        .def_readwrite("listenPort", &nixlAgentConfig::listenPort)
        .def_readwrite("syncMode", &nixlAgentConfig::syncMode)
        .def_readwrite("captureTelemetry", &nixlAgentConfig::captureTelemetry)
        .def_readwrite("pthrDelay", &nixlAgentConfig::pthrDelay)
        .def_readwrite("lthrDelay", &nixlAgentConfig::lthrDelay)
        .def_readwrite("etcdWatchTimeout", &nixlAgentConfig::etcdWatchTimeout);

    // note: pybind will automatically convert notif_map to python types:
    // so, a Dictionary of string: List<string>

    py::class_<nixlAgent>(m, "nixlAgent")
        .def(py::init<std::string, nixlAgentConfig>())
        .def("getEffectiveSyncMode", &nixlAgent::getEffectiveSyncMode)
        .def("getAvailPlugins",
             [](nixlAgent &agent) -> std::vector<nixl_backend_t> {
                 std::vector<nixl_backend_t> backends;
                 throw_nixl_exception(agent.getAvailPlugins(backends));
                 return backends;
             })
        .def("getPluginParams",
             [](nixlAgent &agent,
                const nixl_backend_t type) -> std::pair<nixl_b_params_t, std::vector<std::string>> {
                 nixl_b_params_t params;
                 nixl_mem_list_t mems;
                 std::vector<std::string> mems_vec;
                 throw_nixl_exception(agent.getPluginParams(type, mems, params));
                 for (const auto &elm : mems)
                     mems_vec.push_back(nixlEnumStrings::memTypeStr(elm));
                 return std::make_pair(params, mems_vec);
             })
        .def("getBackendParams",
             [](nixlAgent &agent,
                uintptr_t backend) -> std::pair<nixl_b_params_t, std::vector<std::string>> {
                 nixl_b_params_t params;
                 nixl_mem_list_t mems;
                 std::vector<std::string> mems_vec;
                 throw_nixl_exception(
                     agent.getBackendParams((nixlBackendH *)backend, mems, params));
                 for (const auto &elm : mems)
                     mems_vec.push_back(nixlEnumStrings::memTypeStr(elm));
                 return std::make_pair(params, mems_vec);
             })
        .def(
            "createBackend",
            [](nixlAgent &agent,
               const nixl_backend_t &type,
               const nixl_b_params_t &initParams) -> uintptr_t {
                nixlBackendH *backend = nullptr;
                throw_nixl_exception(agent.createBackend(type, initParams, backend));
                return (uintptr_t)backend;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "registerMem",
            [](nixlAgent &agent,
               nixl_reg_dlist_t descs,
               const std::vector<uintptr_t> &backends) -> nixl_status_t {
                nixl_opt_args_t extra_params;
                nixl_status_t ret;
                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);

                ret = agent.registerMem(descs, &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("descs"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "deregisterMem",
            [](nixlAgent &agent,
               nixl_reg_dlist_t descs,
               const std::vector<uintptr_t> &backends) -> nixl_status_t {
                nixl_opt_args_t extra_params;
                nixl_status_t ret;
                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);

                ret = agent.deregisterMem(descs, &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("descs"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepareDeregisterMem",
            [](nixlAgent &agent,
               const nixl_reg_dlist_t &descs,
               const std::vector<uintptr_t> &backends) {
                return std::make_unique<nixl_py_mem_deregistration>(
                    agent, descs, backends);
            },
            R"pbdoc(
Create a persistent memory-deregistration receipt.

This interruption-safe contract requires exactly one explicit, agent-owned UCX
backend handle. Generic NIXL deregistration does not expose sufficient
per-backend ownership/progress for a sound durable receipt. The receipt copies
descriptors, retains the validated UCX handle, and keeps this exact agent alive.
Preparation does not mutate registration state. Execute it with
``executeDeregisterMem``; SUCCESS and NOT_FOUND become durable completion before
Python signal delivery, and subsequent execution is a no-op.
)pbdoc",
            py::arg("descs"),
            py::arg("backends"),
            py::keep_alive<0, 1>())
        .def(
            "executeDeregisterMem",
            [](nixlAgent &agent, nixl_py_mem_deregistration &receipt) {
                return receipt.execute(agent);
            },
            py::arg("receipt"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "queryMem",
            [](nixlAgent &agent,
               nixl_reg_dlist_t descs,
               uintptr_t backend) -> std::vector<nixl_query_resp_t> {
                std::vector<nixl_query_resp_t> resp;
                nixl_opt_args_t extra_params;

                extra_params.backends.push_back((nixlBackendH *)backend);

                nixl_status_t ret = agent.queryMem(descs, resp, &extra_params);
                throw_nixl_exception(ret);
                return resp;
            },
            py::arg("descs"),
            py::arg("backend"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "makeConnection",
            [](nixlAgent &agent,
               const std::string &remote_agent,
               const std::vector<uintptr_t> &backends) {
                nixl_opt_args_t extra_params;

                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);

                nixl_status_t ret = agent.makeConnection(remote_agent, &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepXferDlist",
            [](nixlAgent &agent,
               std::string &agent_name,
               const nixl_xfer_dlist_t &descs,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(
                    prep_xfer_dlist(agent, agent_name, descs, backends));
            },
            py::arg("agent_name"),
            py::arg("descs"),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepXferDlist",
            [](nixlAgent &agent,
               const nixl_xfer_dlist_t &descs,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(
                    prep_xfer_dlist(agent, NIXL_INIT_AGENT, descs, backends));
            },
            py::arg("descs"),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepXferDlist",
            [](nixlAgent &agent,
               std::string &agent_name,
               nixl_mem_t mem,
               const py::array &descs,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                const nixl_stride_dlist_t stride_descs = to_stride_dlist(mem, descs);

                py::gil_scoped_release release;
                return reinterpret_cast<uintptr_t>(
                    prep_xfer_dlist(agent, agent_name, stride_descs, backends));
            },
            py::arg("agent_name"),
            py::arg("mem_type"),
            py::arg("descs").noconvert(),
            py::arg("backend") = std::vector<uintptr_t>({}))
        .def(
            "prepXferDlistOwned",
            [](nixlAgent &agent,
               std::string &agent_name,
               const nixl_xfer_dlist_t &descs,
               const std::vector<uintptr_t> &backends) {
                auto owned = std::make_unique<nixl_py_owned_dlist_handle>(agent);
                (void)prep_xfer_dlist(
                    agent, agent_name, descs, backends, owned.get());
                return owned;
            },
            py::arg("agent_name"),
            py::arg("descs"),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>(),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepXferDlistOwned",
            [](nixlAgent &agent,
               const nixl_xfer_dlist_t &descs,
               const std::vector<uintptr_t> &backends) {
                auto owned = std::make_unique<nixl_py_owned_dlist_handle>(agent);
                (void)prep_xfer_dlist(
                    agent, NIXL_INIT_AGENT, descs, backends, owned.get());
                return owned;
            },
            py::arg("descs"),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>(),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepXferDlistOwned",
            [](nixlAgent &agent,
               std::string &agent_name,
               nixl_mem_t mem,
               const py::array &descs,
               const std::vector<uintptr_t> &backends) {
                const nixl_stride_dlist_t stride_descs = to_stride_dlist(mem, descs);
                auto owned = std::make_unique<nixl_py_owned_dlist_handle>(agent);
                {
                    py::gil_scoped_release release;
                    (void)prep_xfer_dlist(
                        agent, agent_name, stride_descs, backends, owned.get());
                }
                return owned;
            },
            py::arg("agent_name"),
            py::arg("mem_type"),
            py::arg("descs").noconvert(),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>())
        .def(
            "makeXferReq",
            [](nixlAgent &agent,
               const nixl_xfer_op_t &operation,
               uintptr_t local_side,
               py::object local_indices,
               uintptr_t remote_side,
               py::object remote_indices,
               std::string notif_msg,
               const std::vector<uintptr_t> &backends,
               bool skip_desc_merge) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(make_xfer_req(agent,
                                                                 operation,
                                                                 local_side,
                                                                 local_indices,
                                                                 remote_side,
                                                                 remote_indices,
                                                                 std::move(notif_msg),
                                                                 backends,
                                                                 skip_desc_merge));
            },
            py::arg("operation"),
            py::arg("local_side"),
            py::arg("local_indices"),
            py::arg("remote_side"),
            py::arg("remote_indices"),
            py::arg("notif_msg"),
            py::arg("backend"),
            py::arg("skip_desc_merg") = false)
        .def(
            "makeXferReqOwned",
            [](nixlAgent &agent,
               const nixl_xfer_op_t &operation,
               uintptr_t local_side,
               py::object local_indices,
               uintptr_t remote_side,
               py::object remote_indices,
               std::string notif_msg,
               const std::vector<uintptr_t> &backends,
               bool skip_desc_merge) {
                auto owned = std::make_unique<nixl_py_owned_xfer_handle>(agent);
                (void)make_xfer_req(agent,
                                    operation,
                                    local_side,
                                    local_indices,
                                    remote_side,
                                    remote_indices,
                                    std::move(notif_msg),
                                    backends,
                                    skip_desc_merge,
                                    owned.get());
                return owned;
            },
            py::arg("operation"),
            py::arg("local_side"),
            py::arg("local_indices"),
            py::arg("remote_side"),
            py::arg("remote_indices"),
            py::arg("notif_msg") = std::string(""),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::arg("skip_desc_merg") = false,
            py::keep_alive<0, 1>())
        .def(
            "createXferRequestSlotExecution",
            [](nixlAgent &agent,
               const nixl_xfer_op_t &operation,
               uintptr_t local_side,
               py::object local_indices,
               uintptr_t remote_side,
               py::object remote_indices,
               std::string notif_msg,
               const std::vector<uintptr_t> &backends,
               py::object running_state,
               py::object completed_state,
               py::object failed_state,
               py::object local_owner,
               py::object remote_owner) {
                return std::make_unique<nixl_py_request_slot_execution>(agent,
                                                                        operation,
                                                                        local_side,
                                                                        local_indices,
                                                                        remote_side,
                                                                        remote_indices,
                                                                        std::move(notif_msg),
                                                                        backends,
                                                                        std::move(running_state),
                                                                        std::move(completed_state),
                                                                        std::move(failed_state),
                                                                        std::move(local_owner),
                                                                        std::move(remote_owner));
            },
            py::arg("operation"),
            py::arg("local_side"),
            py::arg("local_indices"),
            py::arg("remote_side"),
            py::arg("remote_indices"),
            py::arg("notif_msg"),
            py::arg("backend"),
            py::arg("running_state"),
            py::arg("completed_state"),
            py::arg("failed_state"),
            py::arg("local_owner"),
            py::arg("remote_owner"),
            py::keep_alive<0, 1>())
        .def(
            "createXferReq",
            [](nixlAgent &agent,
               const nixl_xfer_op_t &operation,
               const nixl_xfer_dlist_t &local_descs,
               const nixl_xfer_dlist_t &remote_descs,
               const std::string &remote_agent,
               std::string notif_msg,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(create_xfer_req(agent,
                                                                   operation,
                                                                   local_descs,
                                                                   remote_descs,
                                                                   remote_agent,
                                                                   std::move(notif_msg),
                                                                   backends));
            },
            py::arg("operation"),
            py::arg("local_descs"),
            py::arg("remote_descs"),
            py::arg("remote_agent"),
            py::arg("notif_msg") = std::string(""),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "createXferReqOwned",
            [](nixlAgent &agent,
               const nixl_xfer_op_t &operation,
               const nixl_xfer_dlist_t &local_descs,
               const nixl_xfer_dlist_t &remote_descs,
               const std::string &remote_agent,
               std::string notif_msg,
               const std::vector<uintptr_t> &backends) {
                auto owned = std::make_unique<nixl_py_owned_xfer_handle>(agent);
                (void)create_xfer_req(agent,
                                      operation,
                                      local_descs,
                                      remote_descs,
                                      remote_agent,
                                      std::move(notif_msg),
                                      backends,
                                      owned.get());
                return owned;
            },
            py::arg("operation"),
            py::arg("local_descs"),
            py::arg("remote_descs"),
            py::arg("remote_agent"),
            py::arg("notif_msg") = std::string(""),
            py::arg("backend") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>(),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "estimateXferCost",
            [](nixlAgent &agent, uintptr_t reqh) -> std::tuple<int64_t, int64_t, int> {
                std::chrono::microseconds duration;
                std::chrono::microseconds err_margin;
                nixl_cost_t method;
                nixl_status_t ret = agent.estimateXferCost(
                    reinterpret_cast<const nixlXferReqH *>(reqh), duration, err_margin, method);
                throw_nixl_exception(ret);
                return std::make_tuple(duration.count(), err_margin.count(), int(method));
            },
            py::arg("req_handle"))
        .def(
            "postXferReq",
            [](nixlAgent &agent, uintptr_t reqh, std::string notif_msg) -> nixl_status_t {
                nixl_opt_args_t extra_params;
                nixl_status_t ret;
                if (notif_msg.size() > 0) {
                    extra_params.notif.emplace(std::move(notif_msg));
                    ret = agent.postXferReq((nixlXferReqH *)reqh, &extra_params);
                } else {
                    ret = agent.postXferReq((nixlXferReqH *)reqh);
                }
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("reqh"),
            py::arg("notif_msg") = std::string(""),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "postXferReqWithNotifOverride",
            [](nixlAgent &agent,
               uintptr_t reqh,
               std::optional<std::string> notif_msg) -> nixl_status_t {
                if (reqh == 0) {
                    throw std::invalid_argument("reqh must be a non-zero request handle");
                }
                nixl_opt_args_t extra_params;
                if (notif_msg.has_value() && notif_msg->empty()) {
                    throw std::invalid_argument(
                        "notification override must be non-empty or None");
                }
                extra_params.notif = std::move(notif_msg);
                // Passing the options object is intentional even for None:
                // NIXL then clears any notification retained by a prior post.
                nixl_status_t ret =
                    agent.postXferReq(reinterpret_cast<nixlXferReqH *>(reqh), &extra_params);
                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("reqh"),
            py::arg("notif_msg") = std::nullopt,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "postXferReqAndPoll",
            [](nixlAgent &agent,
               uintptr_t reqh,
               int64_t max_polls,
               std::optional<int64_t> timeout_ns) -> nixl_status_t {
                if (reqh == 0) {
                    throw std::invalid_argument("reqh must be a non-zero request handle");
                }
                if (max_polls < 0) {
                    throw std::invalid_argument("max_polls must be non-negative");
                }
                if (timeout_ns.has_value() && *timeout_ns < 0) {
                    throw std::invalid_argument("timeout_ns must be non-negative or None");
                }

                std::chrono::steady_clock::time_point started;
                if (timeout_ns.has_value() && max_polls != 0) {
                    started = std::chrono::steady_clock::now();
                }

                nixl_status_t ret = agent.postXferReq((nixlXferReqH *)reqh);
                if (ret < 0) {
                    throw_nixl_exception(ret);
                }
                if (ret != NIXL_IN_PROG || max_polls == 0) {
                    return ret;
                }

                if (!timeout_ns.has_value()) {
                    for (int64_t poll = 0; poll < max_polls; ++poll) {
                        ret = agent.getXferStatus((nixlXferReqH *)reqh);
                        if (ret < 0) {
                            throw_nixl_exception(ret);
                        }
                        if (ret != NIXL_IN_PROG) {
                            return ret;
                        }
                    }
                    return NIXL_IN_PROG;
                }

                for (int64_t poll = 0; poll < max_polls; ++poll) {
                    ret = agent.getXferStatus((nixlXferReqH *)reqh);
                    if (ret < 0) {
                        throw_nixl_exception(ret);
                    }
                    if (ret != NIXL_IN_PROG) {
                        return ret;
                    }
                    if (poll + 1 == max_polls) {
                        return NIXL_IN_PROG;
                    }
                    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - started)
                                             .count();
                    if (elapsed >= *timeout_ns) {
                        return NIXL_IN_PROG;
                    }
                }
                return NIXL_IN_PROG;
            },
            py::arg("reqh"),
            py::arg("max_polls"),
            py::arg("timeout_ns") = std::nullopt,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "getXferStatusBatch",
            [](nixlAgent &agent,
               uintptr_t reqh,
               int64_t max_polls,
               std::optional<int64_t> timeout_ns,
               int64_t timeout_check_interval) -> nixl_status_t {
                if (reqh == 0) {
                    throw std::invalid_argument("reqh must be a non-zero request handle");
                }
                if (max_polls <= 0) {
                    throw std::invalid_argument("max_polls must be positive");
                }
                if (timeout_ns.has_value() && *timeout_ns < 0) {
                    throw std::invalid_argument("timeout_ns must be non-negative or None");
                }
                if (timeout_check_interval <= 0) {
                    throw std::invalid_argument("timeout_check_interval must be positive");
                }

                nixl_status_t ret;
                if (!timeout_ns.has_value()) {
                    for (int64_t poll = 0; poll < max_polls; ++poll) {
                        ret = agent.getXferStatus((nixlXferReqH *)reqh);
                        if (ret < 0) {
                            throw_nixl_exception(ret);
                        }
                        if (ret != NIXL_IN_PROG) {
                            return ret;
                        }
                    }
                    return NIXL_IN_PROG;
                }

                // Observe status once before consulting the deadline.  This preserves
                // the timeout_ns=0 contract and also makes a terminal/error result at
                // every sampling boundary authoritative over the soft time bound.
                const auto started = std::chrono::steady_clock::now();
                ret = agent.getXferStatus((nixlXferReqH *)reqh);
                if (ret < 0) {
                    throw_nixl_exception(ret);
                }
                if (ret != NIXL_IN_PROG || max_polls == 1) {
                    return ret;
                }
                auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
                if (elapsed >= *timeout_ns) {
                    return NIXL_IN_PROG;
                }

                // steady_clock::now() is material on sub-microsecond status probes.
                // After the special first-observation check, sample only at exact
                // observation-count multiples of the interval. timeout_ns is
                // therefore a soft bound: it can overshoot by up to one interval of
                // non-preemptible getXferStatus calls (or one slow status call).
                int64_t polls_until_timeout_check = timeout_check_interval - 1;
                if (polls_until_timeout_check == 0) {
                    polls_until_timeout_check = 1;
                }
                for (int64_t poll = 1; poll < max_polls; ++poll) {
                    ret = agent.getXferStatus((nixlXferReqH *)reqh);
                    if (ret < 0) {
                        throw_nixl_exception(ret);
                    }
                    if (ret != NIXL_IN_PROG) {
                        return ret;
                    }
                    if (poll + 1 == max_polls) {
                        return NIXL_IN_PROG;
                    }
                    if (--polls_until_timeout_check == 0) {
                        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
                        if (elapsed >= *timeout_ns) {
                            return NIXL_IN_PROG;
                        }
                        polls_until_timeout_check = timeout_check_interval;
                    }
                }
                return NIXL_IN_PROG;
            },
            py::arg("reqh"),
            py::arg("max_polls"),
            py::arg("timeout_ns") = std::nullopt,
            py::arg("timeout_check_interval") = 32,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "getXferStatus",
            [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                nixl_status_t ret = agent.getXferStatus((nixlXferReqH *)reqh);
                throw_nixl_exception(ret);
                return ret;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "getXferTelemetry",
            [](nixlAgent &agent, uintptr_t reqh) -> nixl_xfer_telem_t {
                nixl_xfer_telem_t telemetry;
                nixl_status_t ret = agent.getXferTelemetry((nixlXferReqH *)reqh, telemetry);
                throw_nixl_exception(ret);
                return telemetry;
            },
            py::arg("reqh"))
        .def("queryXferBackend",
             [](nixlAgent &agent, uintptr_t reqh) -> uintptr_t {
                 nixlBackendH *backend = nullptr;
                 throw_nixl_exception(agent.queryXferBackend((nixlXferReqH *)reqh, backend));
                 return (uintptr_t)backend;
             })
        .def("releaseXferReq",
             [](nixlAgent &agent, uintptr_t reqh) -> nixl_status_t {
                 nixl_status_t ret = agent.releaseXferReq((nixlXferReqH *)reqh);
                 throw_nixl_exception(ret);
                 return ret;
             })
        .def("releaseXferReqOnce",
             [](nixlAgent &agent,
                nixl_py_xfer_release_state &state,
                uintptr_t reqh) -> nixl_status_t {
                 return release_once(state, agent, reqh, [&agent, reqh] {
                     return agent.releaseXferReq(reinterpret_cast<nixlXferReqH *>(reqh));
                 });
             })
        .def("releasedDlistH",
             [](nixlAgent &agent, uintptr_t handle) -> nixl_status_t {
                 nixl_status_t ret = agent.releasedDlistH((nixlDlistH *)handle);
                 throw_nixl_exception(ret);
                 return ret;
             })
        .def("releasedDlistHOnce",
             [](nixlAgent &agent,
                nixl_py_dlist_release_state &state,
                uintptr_t handle) -> nixl_status_t {
                 return release_once(state, agent, handle, [&agent, handle] {
                     return agent.releasedDlistH(reinterpret_cast<nixlDlistH *>(handle));
                 });
             })
        .def(
            "getNotifs",
            [](nixlAgent &agent,
               nixl_py_notifs_t &notif_map,
               const std::vector<uintptr_t> &backends) -> nixl_py_notifs_t {
                nixl_notifs_t new_notifs;
                nixl_opt_args_t extra_params;

                {
                    py::gil_scoped_release release;
                    for (uintptr_t backend : backends)
                        extra_params.backends.push_back((nixlBackendH *)backend);

                    nixl_status_t ret = agent.getNotifs(new_notifs, &extra_params);

                    throw_nixl_exception(ret);
                }

                for (const auto &pair : new_notifs) {
                    for (const auto &str : pair.second)
                        notif_map[pair.first].push_back(py::bytes(str));
                }
                return notif_map;
            },
            py::arg("notif_map"),
            py::arg("backends") = std::vector<uintptr_t>({}))
        .def(
            "getNotifsGrouped",
            [](nixlAgent &agent, const std::vector<uintptr_t> &backends) {
                const nixl_opt_args_t extra_params = make_validated_opt_args(
                    agent, backends, "grouped notification poll");
                return get_notif_batches(agent, &extra_params);
            },
            R"pbdoc(
Destructively drain new notifications into a fresh dict[str, tuple[bytes, ...]].

The supplied backend handles must belong to this agent. Calls compete for the
same queues, and an exception can follow partial native consumption.
)pbdoc",
            py::arg("backends") = std::vector<uintptr_t>({}))
        .def(
            "createNotifReceiver",
            [](nixlAgent &agent, const std::vector<uintptr_t> &backends) {
                return std::make_unique<nixl_py_notification_receiver>(agent, backends);
            },
            R"pbdoc(
Prepare a grouped destructive receiver with a fixed backend selection.

Every supplied backend handle is identity-validated against this exact agent
before it can be retained. The receiver keeps the agent alive and follows its
native sync-mode concurrency contract. An empty selection retains NIXL's legacy
default-selection behavior rather than claiming a frozen selection.
)pbdoc",
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>())
        .def(
            "createNotifSender",
            [](nixlAgent &agent,
               const std::string &remote_agent,
               const std::vector<uintptr_t> &backends) {
                return std::make_unique<nixl_py_notification_sender>(
                    agent, remote_agent, backends);
            },
            R"pbdoc(
Prepare a standalone notification sender with a fixed destination and backend selection.

Every supplied backend handle is identity-validated against this exact agent
before it can be retained. The sender keeps the agent alive and follows its
native sync-mode concurrency contract. Successful ``None`` return is local
acceptance, not remote observation.
)pbdoc",
            py::arg("remote_agent"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::keep_alive<0, 1>())
        .def(
            "genNotif",
            [](nixlAgent &agent,
               const std::string &remote_agent,
               const std::string &msg,
               const std::vector<uintptr_t> &backends) {
                nixl_opt_args_t extra_params;
                nixl_status_t ret;

                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);


                ret = agent.genNotif(remote_agent, msg, &extra_params);

                throw_nixl_exception(ret);
                return ret;
            },
            py::arg("remote_agent"),
            py::arg("msg"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def("getLocalMD",
             [](nixlAgent &agent) -> py::bytes {
                 // python can only interpret text strings
                 std::string ret_str("");
                 throw_nixl_exception(agent.getLocalMD(ret_str));
                 return py::bytes(ret_str);
             })
        .def(
            "getLocalPartialMD",
            [](nixlAgent &agent,
               nixl_reg_dlist_t descs,
               bool inc_conn_info,
               const std::vector<uintptr_t> &backends) -> py::bytes {
                std::string ret_str("");

                nixl_opt_args_t extra_params;

                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);
                extra_params.includeConnInfo = inc_conn_info;

                throw_nixl_exception(agent.getLocalPartialMD(descs, ret_str, &extra_params));
                return py::bytes(ret_str);
            },
            py::arg("descs"),
            py::arg("inc_conn_info") = false,
            py::arg("backends") = std::vector<uintptr_t>({}))
        .def("loadRemoteMD",
             [](nixlAgent &agent, const std::string &remote_metadata) -> py::bytes {
                 // python can only interpret text strings
                 std::string remote_name("");
                 {
                     py::gil_scoped_release release;
                     throw_nixl_exception(agent.loadRemoteMD(remote_metadata, remote_name));
                 }
                 return py::bytes(remote_name);
             })
        .def(
            "inspectRemoteMD",
            [](const nixlAgent &agent,
               const std::string &remote_metadata) -> py::bytes {
                 std::string remote_name("");
                 throw_nixl_exception(
                     agent.inspectRemoteMD(remote_metadata, remote_name));
                 return py::bytes(remote_name);
             })
        .def("invalidateRemoteMD", &nixlAgent::invalidateRemoteMD)
        .def(
            "sendLocalMD",
            [](nixlAgent &agent, std::string ip_addr, int port) {
                nixl_opt_args_t extra_params;

                extra_params.ipAddr = ip_addr;
                extra_params.port = port;

                throw_nixl_exception(agent.sendLocalMD(&extra_params));
            },
            py::arg("ip_addr") = std::string(""),
            py::arg("port") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sendLocalPartialMD",
            [](nixlAgent &agent,
               nixl_reg_dlist_t descs,
               bool inc_conn_info,
               const std::vector<uintptr_t> &backends,
               std::string ip_addr,
               int port,
               std::string label) {
                std::string ret_str("");

                nixl_opt_args_t extra_params;

                for (uintptr_t backend : backends)
                    extra_params.backends.push_back((nixlBackendH *)backend);
                extra_params.includeConnInfo = inc_conn_info;
                extra_params.ipAddr = ip_addr;
                extra_params.port = port;
                extra_params.metadataLabel = label;

                throw_nixl_exception(agent.sendLocalPartialMD(descs, &extra_params));
            },
            py::arg("descs"),
            py::arg("inc_conn_info") = false,
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::arg("ip_addr") = std::string(""),
            py::arg("port") = 0,
            py::arg("label") = std::string(""),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "fetchRemoteMD",
            [](nixlAgent &agent,
               std::string remote_agent,
               std::string ip_addr,
               int port,
               std::string label) {
                nixl_opt_args_t extra_params;

                extra_params.ipAddr = ip_addr;
                extra_params.port = port;
                extra_params.metadataLabel = label;

                throw_nixl_exception(agent.fetchRemoteMD(remote_agent, &extra_params));
            },
            py::arg("remote_agent"),
            py::arg("ip_addr") = std::string(""),
            py::arg("port") = 0,
            py::arg("label") = std::string(""),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "invalidateLocalMD",
            [](nixlAgent &agent, std::string ip_addr, int port) {
                nixl_opt_args_t extra_params;

                extra_params.ipAddr = ip_addr;
                extra_params.port = port;

                throw_nixl_exception(agent.invalidateLocalMD(&extra_params));
            },
            py::arg("ip_addr") = std::string(""),
            py::arg("port") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def("checkRemoteMD", &nixlAgent::checkRemoteMD)
        .def(
            "prepMemView",
            [](nixlAgent &agent,
               const nixl_xfer_dlist_t &dlist,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                return prep_mem_view(agent, dlist, backends);
            },
            py::arg("dlist"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "prepMemView",
            [](nixlAgent &agent,
               const nixl_remote_dlist_t &dlist,
               const std::vector<uintptr_t> &backends) -> uintptr_t {
                return prep_mem_view(agent, dlist, backends);
            },
            py::arg("dlist"),
            py::arg("backends") = std::vector<uintptr_t>({}),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "releaseMemView",
            [](nixlAgent &agent, uintptr_t mvh) {
                agent.releaseMemView(reinterpret_cast<nixlMemViewH>(mvh));
            },
            py::arg("mvh"),
            py::call_guard<py::gil_scoped_release>());
}
