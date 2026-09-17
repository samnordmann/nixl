# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[4] / "src" / "plugins" / "ucx" / "ucx_backend.cpp"
)


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace : index + 1]
    raise AssertionError(f"unterminated function body for {signature!r}")


def test_release_deletes_only_after_quiescence_commit() -> None:
    source = SOURCE.read_text()
    body = _function_body(source, "nixlUcxEngine::releaseReqH")

    try_release = body.index("int_handle->tryRelease()")
    non_success = body.index("status != NIXL_SUCCESS")
    deletion = body.index("delete int_handle")
    assert try_release < non_success < deletion
    assert "int_handle->release()" not in body


def test_data_requests_are_not_detached_while_active() -> None:
    source = SOURCE.read_text()
    send_body = _function_body(source, "\nnixlUcxEngine::sendXferRange(const")

    assert "int_handle->append(ret, req, conn)" in send_body
    assert "ucp_request_free(pending_req)" not in send_body


def test_release_path_never_cancel_frees_an_active_request() -> None:
    source = SOURCE.read_text()
    try_release_body = _function_body(source, "tryRelease() {")
    cancel_body = _function_body(source, "requestCancel() {")

    assert "reqCancel" not in try_release_body
    assert "reqRelease" not in try_release_body
    assert "worker_->reqCancel(req)" in cancel_body
    assert "worker_->reqRelease(req)" not in cancel_body


def test_chunk_append_error_drains_before_completing_shared_ownership() -> None:
    source = SOURCE.read_text()
    send_body = _function_body(source, "nixlUcxThreadPoolEngine::sendXferRange")

    assert "const nixl_status_t chunk_status = chunk_handle->status()" in send_body
    assert "chunk_status == NIXL_IN_PROG" in send_body
    assert "chunk_handle->complete(ret)" not in send_body


def test_chunk_pending_commit_keeps_state_alive_past_method_return() -> None:
    source = SOURCE.read_text()
    complete_body = _function_body(source, "nixlUcxChunkBackendReqH::complete")
    dedicated_source = source[source.index("class nixlUcxDedicatedThread") :]
    run_body = _function_body(dedicated_source, "run() override")

    assert "auto shared_state = std::move(sharedState_)" in complete_body
    assert "pendingReqs.fetch_sub" not in complete_body
    completion = run_body.index("auto completed_state = (*it)->complete(status)")
    erase = run_body.index("it = requests_.erase(it)", completion)
    pending_commit = run_body.index("completed_state->pendingReqs.fetch_sub(1)", erase)
    assert completion < erase < pending_commit
