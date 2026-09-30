# NIXL from CuTe: review prototype

Thin device bindings, incremental teaching examples, and a prepared stream-ordered
exchange. This is a source-only review prototype, not a stable production API.
MoE stays separate from the binding. See [the walkthrough](walkthrough.html)
for the tutorial, ordering argument, and code map.

## Build

Use NIXL built/installed from this checkout with UCX GPU-device support.
The bitcode must use the same UCX headers as the host NIXL build. Validated
setup: GB200, NVIDIA PyTorch 26.06 (CUDA 13.3), UCX 1.21.0, CUTLASS DSL 4.5.1,
and LLVM 20. This example does not introduce wheel/extras support.

```bash
# After building/installing NIXL with UCX, from the repository root:
export PYTHONPATH="$PWD/src/bindings/python/nixl-meta:$PYTHONPATH"
python examples/python/cute/build.py --ucx /opt/hpcx/ucx --arch sm_100
```

Select the actual GPU architecture (`sm_90` for Hopper, `sm_100` for GB200).
The build writes `device.bc` beside the example. Rebuild it when the NIXL/UCX
headers, GPU architecture, or compiler change. There is no stable ABI, artifact
manifest, automatic architecture selection, or binary distribution here.

## Incremental examples

```bash
# 1. GPU-side PUT; host barrier makes the teaching example synchronous.
torchrun --standalone --nproc-per-node=2 examples/python/cute/put.py

# 2. Same PUT, followed by a GPU signal and system-acquire wait.
torchrun --standalone --nproc-per-node=2 examples/python/cute/put.py --with-signal

# 3. Dispatch -> stand-in experts -> weighted combine; drain, remove, rejoin.
torchrun --standalone --nproc-per-node=2 examples/python/cute/moe.py

# 4. Also exercise expansion and a sparse active-rank set.
torchrun --standalone --nproc-per-node=3 examples/python/cute/moe.py \
  --membership '0,1;0,1,2;0,2;0,1,2'

# 5. Request-backed submit -> independent work -> GPU completion.
torchrun --standalone --nproc-per-node=2 examples/python/cute/async_put.py

# 6. Preallocated GPU pack/dispatch/expert/return/combine; graph replay and epochs.
torchrun --standalone --nproc-per-node=4 examples/python/cute/moe_stream.py \
  --membership '0,1;0,1,2,3;0,2;0,1,2,3'
```

Use separate GPUs on one node and an external job timeout. Device waits cannot
recover a killed peer. All examples validate their outputs; PUT repeats three
times, and MoE checks every generation against a CPU golden result.

## Native expressivity

- `put` and `atomic_add`: descriptor indices, byte offsets, channel, flags, and
  `THREAD` / `WARP` / `BLOCK` cooperation. All participating lanes call uniformly.
- `put_async(..., request)` defaults to native `DEFER`; it returns submission
  status without a wrapper wait. Each participating thread owns a separate
  64-byte, 64-byte-aligned request until `complete` succeeds. `progress` performs
  one native progress step. Terminal statuses are never polled again.
- `put(..., request=0)` remains the GPU-blocking convenience path. This does not
  mean CPU synchronization. `DEFER` permits deferral, but does not guarantee it.
- `get_ptr(remote, index)` exposes mapped remote memory when available (zero
  otherwise). It must not be called with a local view.
- `fence` is a per-thread system release fence; `signal` fences the caller then
  increments a counter; `wait` uses a system-acquire load. `atomic_add` is the
  raw native operation and adds no wrapper fence. None flush incomplete network PUTs.

GRID cooperation and requestless fire-and-forget are deliberately not provided:
the selected backend does not implement GRID copy, and requestless completion
needs a separate proven protocol. There is still no new public C++ API or stable ABI.

## Prepared exchange

[`Exchange`](exchange.py) precompiles launchers and allocates its plan/status/step
once per epoch. It uses one ready counter and one receive-credit counter per rank.
The GPU returns a credit only after previous consumers finish on the same stream,
waits for credits, copies tiles cooperatively, then signals once per peer and waits
for incoming data. All writers release-fence before publication. Separate kernels
provide a grid-wide join without a grid-residency assumption. Small one-peer
exchanges use one CTA; multi-peer transfers retain parallel CTAs.

`Exchange()` only enqueues work: no host barrier, synchronization, status readback,
or application GPU-buffer allocation. Produce, exchange and consume on the prepared
stream. Call `check()` only at a validation/drain boundary. Membership changes drain
first, rebuild the plan, and invalidate old Python launchers. Destroy old graphs
before releasing their views: a raw CUDA graph replay cannot run the Python guard.

The default geometry is a measured GB200 starting point, not an autotuner. Override
`tile_bytes`, `threads` and `level` for other shapes/devices. The prepared MoE example
keeps routing fixed within an epoch, uses GPU packing/combine, and validates changing
inputs during graph replay; its padding and stand-in experts remain teaching choices.

## Measure and validate

```bash
# Optional independent NVCC implementation of the same protocol (not a shim).
nvcc -std=c++17 -O3 -DNDEBUG -arch=sm_100 -shared -Xcompiler -fPIC \
  -I src/api/device -I src/api/cpp -I /opt/hpcx/ucx/include \
  test/python/cute_native_reference.cu -o /tmp/cute_native_reference.so
torchrun --standalone --nproc-per-node=4 examples/python/cute/benchmark.py \
  --bytes 67108864 --reference-so /tmp/cute_native_reference.so --nccl

PYTHONPATH="examples/python/cute:$PYTHONPATH" torchrun --standalone \
  --nproc-per-node=4 test/python/cute_exchange_smoke.py --bytes 65550 --tile-bytes 65536
python -m pytest -q test/python/test_cute_mvp.py test/python/test_cute_device_contract.py
```

The benchmark includes self-copy, reuse credits, PUTs, publication and receive waits.
CUDA events are **inside** the graph, after an untimed GPU rendezvous. It reports
every rank's samples and the median of slowest-rank times. Registration/JIT/capture
are excluded. Buffers and nonuniform BF16 payloads are reused. Compare matching
geometry; NCCL is an additional all-to-all baseline, not an elasticity equivalent.
`--threads 32 --level warp` and `--tile-bytes` allow explicit geometry sweeps.

## Deliberate limits and evidence boundary

- Gloo is only the CPU control plane (metadata, routing records, barriers).
  The introductory `put.py` / `moe.py` retain their synchronous teaching harness;
  do not use those programs as performance benchmarks. The expert is still
  `x * (expert_id + 1)`, not grouped GEMM.
- A fixed maximum of live processes; inactive ranks remain registered standbys.
  Membership changes stage views and commit after a drain. Stable expert IDs
  are `rank * experts_per_rank + local_expert`. No crash recovery or process replacement.
- The application keeps buffers, registrations, metadata and raw device views
  alive until consumers finish. GPU credits permit steady-state source/receive reuse;
  membership changes still require a coordinated drain. Use an external job timeout.
- Qualification here is single-node CUDA-IPC. The installed UCX build has no GDA:
  actual pending RDMA progress, inter-node ordering and overlap remain unverified.
  The CPU mock checks wrapper forwarding/state transitions, not transport semantics.
- No crash recovery, late process creation, persistent scheduler, stable ABI, wheel
  changes, or production MoE throughput claim. Graph replay is scoped to one epoch.

The only native change handles UCX synchronous completion correctly for the
request-backed calls. Without it, a successful CUDA-IPC PUT can poll a UCX
request that was never initialized. Other native cleanups from the full stack
are intentionally excluded.

The original four teaching commands passed on GB200 (2026-09-17).
See [performance evidence](performance.html) for this follow-up's measurements,
negative experiments and remaining gaps; proximity to native NIXL is not a proof
of a universal hardware speed-of-light limit.
