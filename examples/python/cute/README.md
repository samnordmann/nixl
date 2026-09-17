# NIXL from CuTe: review prototype

Three operations (`put`, `signal`, `wait`), two runnable examples, one small
device-bitcode build. This is a source-only prototype, not a production API.
The MoE example is separate from the binding. See [the walkthrough](walkthrough.html)
for the incremental tutorial and code map.

## Build

Use an installed NIXL built from this checkout with UCX GPU-device support,
CUDA, CUTLASS DSL 4.5.1, and LLVM 20. The bitcode must use the same UCX headers
as the host NIXL build. On CUDA 13, use a CUDA-13-compatible CUTLASS DSL source
installation; this example does not introduce wheel/extras support.

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
```

Use separate GPUs on one node and an external job timeout. Device waits cannot
recover a killed peer. All examples validate their outputs; PUT repeats three
times, and MoE checks every generation against a CPU golden result.

## Deliberate limits

- One-thread device calls and one channel. No tuning or throughput claim.
- Gloo is only the CPU control plane (metadata, routing records, barriers).
  NIXL/CuTe transfers BF16 activations and expert results. The router/packing are
  CPU-based and the expert is `x * (expert_id + 1)`, not grouped GEMM.
- A fixed maximum of live processes; inactive ranks remain registered standbys.
  Membership changes stage views and commit after a drain. Stable expert IDs
  are `rank * experts_per_rank + local_expert`. No crash recovery or process replacement.
- The application keeps buffers, registrations, metadata and raw device views
  alive until consumers finish. Sources are reused only after each exchange drains.
- No persistent scheduler, CUDA graphs, stable public ABI, wheel changes,
  topology framework, or performance-evidence infrastructure.

The only native change handles UCX synchronous completion correctly for the
request-backed calls. Without it, a successful CUDA-IPC PUT can poll a UCX
request that was never initialized. Other native cleanups from the full stack
are intentionally excluded.

CPU model/build-helper checks: `python -m pytest -q test/python/test_cute_mvp.py`.
