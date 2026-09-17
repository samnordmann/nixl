# NIXL device operations from CuTe DSL

This directory demonstrates the version 3 NIXL device ABI from CUTLASS CuTe
DSL 4.5 kernels. The examples progress from one PUT to ordered signaling,
THREAD/WARP measurement, and a complete elastic MoE dispatch/expert/combine
reference. Every remote example uses distinct GPU processes: modern UCX needs
a real CUDA-IPC peer lane for a remote device memory view; a same-process,
same-GPU loopback is not a valid substitute.

## Learning path

### 1. Minimal two-process PUT

Register one source and destination, exchange metadata and transfer
coordinates, prepare asymmetric local/remote views, PUT on an explicit stream,
validate on the receiver, and tear down in lifetime-safe order:

```bash
python examples/python/cute/two_process_put.py --devices 0 1
```

### 2. PUT followed by a visible completion signal

This adds a separate aligned 64-bit target counter. The producer performs PUT
and then atomic-add on the same channel; the receiver consumes the payload only
after observing that signal. All host/control-plane waits are bounded.

```bash
python examples/python/cute/put_signal.py --devices 0 1 --timeout 60
```

The timeout cannot cancel a blocking device kernel already spinning on a dead peer;
use an external process or Slurm timeout for fault injection.

### 3. THREAD/WARP microbenchmark

The benchmark runs PUT-only and PUT+signal, retains raw CUDA-event samples, and
reports p50/p90/p99/max plus logical payload GB/s. Events bracket the queued
launch-to-completion path, including any stream idle time while the Python
wrapper dispatches the kernel; the result is not kernel-only timing. Its
mandatory preflight compiles each specialization outside the timed sample.
Every case poisons a guarded target, then gives the final timed payload a
distinct nonce and checks the exact body, unwritten tail, and both guards before
reporting PASS.

```bash
python examples/python/cute/benchmark_put.py \
  --devices 0 1 \
  --sizes 64B,4KiB,14KiB,64KiB,1MiB \
  --scopes thread warp \
  --modes put put-signal \
  --warmup 200 --iterations 2000
```

Use native NIXLBench Device API mode as the principal transport baseline. This
is a same-address, hot-buffer latency/throughput measurement, not streaming
bandwidth. Do not compare its logical payload GB/s to NCCL `busbw`; the
accounting differs.

### 4. GPU-persistent transport benchmark

This benchmark removes Python pacing, CUDA events, and repeated launches from
the measured path. Two long-lived kernels execute warmup and measurement on the
GPU, time with `%globaltimer`, and use GPU-returned credits before source reuse.
`wait` is the request-backed baseline; `defer` uses the EP
`DEFER,DEFER,DEFER,NONE` doorbell cadence:

```bash
python examples/python/cute/benchmark_persistent.py \
  --devices 0 1 --size 64KiB --batch-size 4 \
  --groups 4 --channels 4 --scope warp \
  --mode defer --warmup 200 --iterations 2000
```

Credit polling is timer-free by default. `--device-timeout-ms` opts into a
diagnostic timer-reading wait; use the enclosing process or Slurm timeout as
the production failure bound. Mapped batches are copied as one contiguous span,
paying one warp rendezvous per group and iteration rather than one per slot.

The result keeps producer API latency and safe-reuse round-trip latency
separate. `full_iteration_cycle` also includes the recurring source-generation
update, warp convergence, required release fence, and the next iteration
boundary. Because groups advance independently, per-iteration cohort spans are
reported only as straggler envelopes; `steady_state_run` divides all measured
logical bytes by the complete first-start-to-last-boundary interval. Every
batch slot has a 64-bit generation word, so final-state validation does not
alias after 256 iterations, and both payload guards are checked. The timed path
does not scan each payload generation before returning credit. More than eight
warp groups (or 128 thread groups) launch as multiple exact-size CTAs with no
idle-lane branch. Because corresponding CTAs wait on one another across GPUs, both
kernels use cooperative launch. Preflight binds each exact specialization,
borrows the CUDA library already owned by that bound callable, resolves its
current-context function, queries true CTA-per-SM occupancy, and rejects a grid
larger than the resulting cooperative capacity. `mapped`
resolves peer pointers once, then uses the vectorized warp-copy primitive in its
inner loop. Before launching mapped mode, the example requires CUDA native peer
atomics for every directed pair of participating GPUs. A non-null `nixlGetPtr`
result is the peer allocation's base in the calling process; every access uses
that base plus a descriptor-relative offset. CUDA IPC may map it at a different
numeric address from the owner's process, and numeric equality is not required.
`--allow-unverified-mapped` remains a deprecated compatibility no-op.

### 5. Readable elastic MoE communication

The reference example performs deterministic active-expert top-k routing, BF16
dispatch, per-expert batching and transform, reverse combine communication,
weighted scatter/combine, and a CPU golden comparison on every active rank.
The default four-generation plan expands, sparsely removes rank 2, and rejoins
the same stable slot:

```text
[0, 1] -> [0, 1, 2, 3] -> [0, 1, 3] -> [0, 1, 2, 3]
```

```bash
python examples/python/cute/elastic_moe.py \
  --devices 0 1 2 3 \
  --plan examples/python/cute/elastic_expansion_contraction.json \
  --experts-per-rank 2 --top-k 2 \
  --num-tokens 8 --hidden-size 256 \
  --warmup 1 --iterations 3
```

It follows the in-tree NIXL EP elastic model where it can do so safely:

- Buffer capacity is fixed once at `max_ranks`; phase membership is a mask.
- Expert IDs are permanent: `rank * experts_per_rank + local_expert`.
- Every receiver has one padded, single-writer slab for every possible source,
  including sparse inactive holes.
- Each slab carries a versioned `NXME` preamble with generation, origin,
  destination, record count, and element size. Record headers carry token,
  route-slot, and expert identity. Stale or misrouted slabs fail validation,
  including empty slabs.
- Next-generation metadata and peer views are prepared while old views remain
  active. By default, one old-generation MoE round runs after staging to prove
  that preparation does not implicitly activate a join.
- Commit happens only after a full quiescent fence. A stable GPU rank-mask
  allocation is updated in stream order; every remote PUT and its completion
  atomic dynamically read that mask before posting. Expansion phases first
  probe staged join views while they are still masked and require
  `NIXL_ERR_NOT_ALLOWED`. Old views are then released, removed metadata is
  invalidated, and same-slot re-add waits the same five-second grace used by
  the mature EP elastic test.
- PUT and its remote counter atomic use one channel. The fixed preamble makes a
  zero-record message unambiguous, so the counter advances by exactly one.

This example keeps all maximum-capacity processes alive as standby ranks. It
demonstrates staged membership/view elasticity, sparse routing, contraction,
and same-slot rejoin; it is not equivalent to independently starting/exiting
worker processes, failure detection, or recovery from SIGKILL/node loss.
Negative killed-rank entries from the mature EP JSON plans are rejected.
Its rank mask is an application-owned local CUDA tensor updated only at a
quiescent generation boundary; it does not reproduce the mature EP path's
concurrent timeout/failure-mask update protocol.

For the optimized production data path and process-level elastic harness, see
[NIXL EP Buffer](../../device/ep/nixl_ep/buffer.py),
[elastic test](../../device/ep/tests/elastic/elastic.py), and
[low-latency kernels](../../device/ep/csrc/kernels/nixl_ep_ll.cu). The CuTe example
deliberately uses CPU routing/packing plus synchronous one-slab-per-peer PUTs so
the protocol and lifetimes stay readable. It should not be presented as an EP
performance competitor.

### 6. Persistent low-latency elastic MoE

`elastic_moe_ll.py` is the performance-oriented end-to-end example. It keeps
one cooperative CuTe kernel resident for the complete membership generation
and performs dispatch, a deterministic BF16 expert, reverse result transfer,
FP32 weighted top-k reduction, and versioned next-round handshakes without host work or
another launch inside the measured loop:

```bash
python examples/python/cute/elastic_moe_ll.py \
  --devices 0 1 2 3 \
  --membership '0,1;0,1,2,3;0,1,3;0,1,2,3' \
  --experts-per-rank 2 --top-k 2 \
  --num-tokens 8 --hidden-size 256 \
  --warmup 20 --iterations 100 --timing-mode none
```

The arena has stable sparse rank slots and one communication bank by default.
The reverse combine-ready/next-dispatch-ready handshake plus the final
all-combine-read grid barrier proves that bank is consumed before reuse. In the
uninstrumented two-bank specialization, the following round's retained
pre-reduction barrier joins every prior reader before round N+2 can reuse bank
N, so the otherwise redundant final barrier is compiled out. Timed two-bank
modes add only the joins their metric needs: envelope adds one terminal join,
while cadence and peer add end joins on measured rounds but not warmup.
Dispatch records carry
only token, route slot, gate, and payload; one separate bucket stamp validates
generation and source incarnation before the release-ready word is accepted.
Even an empty expert bucket publishes `record_count + 1`. Expert task leaders
join any shard warps with GPU-scope release/acquire operations. One peer leader
then joins all of that peer's experts and performs a single cumulative
system-release publication carrying the aggregate record count. The matching
peer leader acquires it once, validates one peer stamp, and a cooperative grid
barrier makes every route-slot payload visible before disjoint token reducers
accumulate in FP32 and cast once to BF16. The one-bank path uses a second grid
barrier to protect combine-bank reuse; the two-bank path uses the next round's
retained join described above and adds end joins only for requested timing
semantics. Thus system-scope publication and polling scale with active peers,
not experts, and there is no CPU synchronization or launch in the kernel loop.

The default device wait is the production, timer-free specialization: a ready
word costs one acquire load, and only repeated misses sample owner-local and
peer abort epochs. `--device-timeout-ms` opts into a timer-reading diagnostic
specialization. Device-detected protocol errors publish a phase-wide system
abort, but neither mode recovers abrupt mapped-owner loss. In particular, a
kernel that never launches cannot announce a device abort; use the scheduler or
another external job timeout as the final failure bound.

`--timing-mode none` is the default deployment specialization: all `%globaltimer`
reads, timing stores, and the timing-only cooperative barrier are compiled out.
Use `envelope` for only a first-start/final-output-ready pair, `cadence` for
round-start throughput analysis, and `peer` (or the compatibility alias
`--instrument-per-peer`) only for perturbing diagnostics. Compare the
uninstrumented and diagnostic specializations with same-stream CUDA events
around the whole persistent launch; per-rank `%globaltimer` epochs are never
subtracted across GPUs.

Every cooperative launch is checked against the exact CUDA library already
loaded and owned by its bound specialization before the registered view is
used. The host converts that borrowed runtime handle to its native Driver
handle, resolves the exact `kernel_info` symbol, proves that the kernel belongs
to the same library, obtains its current-context function, and queries block
residency for the requested block size. It neither loads nor unloads a second
module. The same bound callable is launched for measurement. Ordinary
execution therefore needs no retained compiler file or path-bearing compile
option, preserving CuTe's reusable JIT-cache key and eliminating cross-rank
dump-file races. This check adds no device instruction and is outside timing.
`--warps-per-cta` remains an explicit tuning control because the best grid
shape is specialization- and architecture-dependent.

Every worker compiles into a fresh rank-private directory. This is a correctness
requirement, not cosmetic logging: CuTe 4.5.1 truncates long specialization
names, so rank-specialized PTX/CUBIN files can otherwise share one basename.
The default directory is private to the temporary run and is removed afterward.
Pass an existing parent with `--codegen-dump-root PATH` to retain `rank-0/`,
`rank-1/`, and so on for inspection; those target rank directories must not
already exist and are rejected rather than reused. The process-elastic variant
additionally includes the incarnation in each private directory name.

This low-latency example intentionally implements only the mapped CUDA-IPC data
path. Before allocating or launching it requires CUDA native peer atomics for
every directed pair of participating GPUs. Each worker queries only its local
accessor row, so its control-plane record is linear in fixed rank capacity; the
phase summary validates and combines those rows into one canonical matrix. Its
device preflight then classifies each active pointer from the exact view and
arena allocation used by the timed phase and fails closed when `nixlGetPtr`
returns null. The returned address is
local to the importing process and may differ from the allocation owner's
numeric address; every remote access uses that process-local base plus an arena
offset. `--allow-unverified-mapped` is only a deprecated no-op for old
launch scripts. A future non-mapped path must add generated-buffer NIXL PUTs,
ordered transport publication, and returned credits as one coherent protocol;
the example does not silently mix an incomplete fallback into the mapped kernel.

#### Graceful OS-process elasticity

`elastic_moe_ll_process.py` applies the same persistent kernel to actual
process membership. A slot has no worker while inactive; removal destroys that
worker, and rejoin creates a new CUDA context, NIXL agent and registration with
a monotonically larger incarnation and unique agent name:

```bash
python examples/python/cute/elastic_moe_ll_process.py \
  --devices 0 1 \
  --membership '0,1;0;0,1' \
  --experts-per-rank 2 --top-k 2 \
  --num-tokens 8 --hidden-size 256 \
  --warmup 20 --iterations 100 --timing-mode none
```

The bounded local coordinator admits a generation only after every candidate
has registered, connected, staged its compact dispatch template, prepared its
sparse view and passed mapped-pointer preflight. Each worker publishes either
`CANDIDATE_PREPARED` or a failure while keeping its owner registered. All
prepared candidates wait for the coordinator's `COMMIT|ABORT` decision. On an
ordinary setup failure or pre-GO timeout, `ABORT` makes every active worker
drain queued CUDA work, explicitly release (and retry) any device view, unload
its known remote identities, publish `SAFE_TO_SHUTDOWN`, and remain registered
until the coordinator observes the complete safe quorum and publishes
`SHUTDOWN`. Staging, preflight, and the pinned status snapshot still share one
stream drain; the failure-only protocol adds no healthy-path rendezvous.

On success the coordinator publishes `COMMIT`, waits for every
`COMMIT_ACKNOWLEDGED`, and publishes `GO`. Every worker then durably publishes
`GO_OBSERVED` and waits on that same all-rank marker set before enqueueing, so
no marker fsync can overlap another rank's persistent GPU interval. After the
one persistent kernel drains, all old view contexts exit and every retiring remote identity is
unloaded before its owner deregisters; unchanged `(slot, incarnation, agent)`
peers retain their NIXL metadata and UCX connection across generations and
unchanged/removal-only generations skip metadata, coordinate, and connection
handshakes. A post-GO Python, marker, validation, or remote-removal error is
carried by the existing unload/result convergence; every live worker then
unloads retained identities and uses the same all-safe `SHUTDOWN` ordering.
Statuses, outputs, and optional timing data are copied to reusable pinned
buffers on the kernel stream before its single host drain, so validation adds
no implicit CUDA wait. The coordinator observes process exit before reusing a
stable GPU slot. Singleton generations pad every fixed-capacity descriptor with
the rank's registered loopback arena and run only the dedicated local MoE path.
This avoids UCX v1.23.x's unsafe `NULL_AGENT` device-list gap behavior without
adding an instruction to the persistent loop.

This lifecycle adds no instruction, launch or CPU synchronization to the
measured kernel loop. It is intentionally same-node and graceful, with
transition downtime. Actual process/SIGKILL or coordinator loss, inability to
publish or read the failure/decision/safety files, ambiguous `GO` observation,
an unquiescent CUDA stream, a device view that remains valid after its explicit
release retry, or a persistent remote-invalidation failure is catastrophic
fail-stop: the complete job must be terminated and no replacement is admitted.
The harness does not claim that CUDA-IPC access survives such abrupt owner
loss. Use an external Slurm timeout as the final bound around fault-injection
runs. The example coordinator uses filesystem markers, polling, and durability
calls for readable lifecycle evidence; it is not an online control plane or a
transition-latency reference. New workers are started only after the preceding
drained generation retires; a production control plane should pre-spawn,
register, compile, connect, stage, and preflight joiners on free slots while the
current generation is still serving.

## Version 3 contract

- Backend and memory: version 3 supports the **UCX** backend and **VRAM**
  descriptors only. NIXL must report `UCX GPU Device API: YES` at configure
  time. Owning CuTe views reject `NULL_AGENT` descriptors: current UCX device
  lists cannot create an all-gap handle and transport operations may not target
  a gap. Use a registered local-loopback descriptor for inactive fixed-capacity
  slots and exclude those slots with converged device control flow.
- Operations: kernels can issue blocking or requestless PUT and 64-bit
  atomic-add operations. There is **no device GET/RDMA-read operation**. `get_ptr()` only
  returns an address for a remote descriptor that UCX already mapped locally;
  it does not fetch data.
- Scope: `Scope.THREAD` makes each calling thread issue one operation.
  `Scope.WARP` is cooperative: all 32 lanes must execute the call, with
  identical arguments, in non-divergent control flow. Version 3 rejects BLOCK
  and GRID scope.
- Flags: blocking calls require `Flags.NONE`. Requestless `put_post()` and
  `atomic_add_post()` accept `Flags.DEFER`; every used peer/channel must
  eventually receive a non-deferred operation to ring its device doorbell.
  The normal batch is deferred PUTs followed by one non-deferred atomic on the
  same peer and channel.
- Completion: blocking PUT/atomic-add waits for its NIXL request, but it is not
  a CUDA block barrier and does not synchronize a host stream. Requestless
  posts return `NIXL_IN_PROG` when accepted; that means submitted, not locally
  complete. Source-buffer reuse needs a GPU credit/ack protocol.
- Visibility: a receiver must observe an RDMA- or peer-written counter with a
  system acquire before consuming its payload. Direct mapped peer copies are
  published with `store_release_system_u64()`. GPU-scope load/store and
  `atomic_add_release_gpu_u64()` / `atomic_max_release_gpu_u64()` support local
  allocation and handoff; the system-scope atomic max publishes a monotonic
  abort that mapped peers can observe. The fetch-add returns the prior `uint64`
  value and is same-GPU only; it is not a peer/NIXL transport atomic. These
  raw-address primitives require non-null, naturally aligned addresses.
- Fused source production: call `fence_release_system()` between GPU writes to
  a registered source buffer and a device-initiated NIXL PUT that may cause the
  transport to read it. The fence is device-only and does not synchronize a
  CUDA stream or the CPU.
- Polling: the unbounded system/GPU waits are timer-free. The abort-aware system
  wait can observe one local GPU abort word, or both a local GPU abort and a
  peer system abort; its ready path remains exactly one system-acquire load and
  checks abort words only after repeated misses. `wait_acquire_system_u64_for()`
  is the diagnostic relative-timeout form: a ready first load does not read the
  clock. Use `wait_acquire_system_u64_until()` when an absolute `%globaltimer`
  deadline is already available.
- Cooperative convergence: `sync_grid()` is valid only in a cooperatively
  launched kernel whose complete grid is resident. The production MoE helper
  validates exact bound-library CUfunction occupancy and launches the same bound
  executable; ordinary callers remain responsible for that launch contract.
- Direct mapping: `mapped_copy_warp_ptr()` uses coherent, no-allocate loads for
  data produced or reused in the current kernel. The `_readonly` form may use
  the non-coherent read-only path only when its entire source span is immutable
  for the kernel lifetime. Both forms copy 16-byte-aligned, non-overlapping
  spans and leave completion publication to the caller. Derive every mapped
  address from the process-local `get_ptr()` base plus a descriptor-relative
  offset. Protocols that use system-scope operations on peer GPU memory must
  qualify native peer atomics for every directed accessing-GPU-to-owner-GPU pair.

For WARP scope, launch complete warps and keep every lane on the same path:

```python
@cute.kernel
def warp_put(local: nixl_cute.MemoryView, remote: nixl_cute.MemoryView):
    # Every lane executes exactly this call with the same descriptor arguments.
    nixl_cute.put(local, remote, 4096, scope=nixl_cute.Scope.WARP)

@cute.jit
def launch(local, remote, stream: cuda.CUstream):
    warp_put(local, remote).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream
    )
```

Literal descriptor indices, offsets, and sizes are checked while CuTe traces
the program. Dynamic DSL values cannot be host-validated and must remain within
the prepared descriptor extents at runtime; they must also select an addressed
descriptor, never a raw low-level `NULL_AGENT` gap.

A remote 64-bit counter can be used as an explicit producer signal after PUTs
on the same channel:

```python
@cute.kernel
def publish(remote_counter: nixl_cute.MemoryView, status: cute.Tensor):
    status[0] = nixl_cute.atomic_add(
        remote_counter,
        1,
        index=0,
        offset=0,
        channel=0,
        scope=nixl_cute.Scope.THREAD,
    )
```

`get_ptr()` is not an RDMA GET. It exposes a peer descriptor only when UCX has
already mapped it locally (for example through CUDA IPC). Its non-null return is
the base in the calling process and may differ numerically from the descriptor
owner's address. Wrap the opaque LLVM pointer before using it as a CuTe iterator
and form accesses only from that returned base:

```python
raw_base = nixl_cute.get_ptr(remote, index=0)
peer_base = cute.make_ptr(cutlass.Int8, raw_base)
if peer_base.toint() != 0:
    peer_tensor = cute.make_tensor(
        peer_base + BYTE_OFFSET, cute.make_layout((NUM_BYTES,))
    )
    # Use peer_tensor only within the mapped branch and view lifetime.
```

Compilation can happen before NIXL resources exist. Type-only views fix the
compiled signature; invoke the result later with live handles of the same kind
and compatible descriptor extents:

```python
compiled = nixl_cute.compile(
    launch_put,
    nixl_cute.make_fake_memory_view("local", (4096,)),
    nixl_cute.make_fake_memory_view("remote", (4096,)),
    status_tensor,
    stream,
)
# local_handle and remote_handle are owning nixl_device_view_handle objects.
compiled(local_handle, remote_handle, status_tensor, stream)
```

`nixl_cute.compile()` adds a host-only launch contract for each view. Before
CuTe marshals any pointer, it verifies the live view's local/remote kind, exact
descriptor count, minimum compiled extents, and lifetime. It adds no kernel
argument or device instruction.

## Resource lifetime and ordering

Treat the view as a borrowed device capability. The following objects must all
outlive every kernel that can use it:

1. the `nixl_agent` that prepared the view;
2. the registered tensor/storage and its registration;
3. the loaded peer metadata for a remote view (NIXL retains the native remote
   section until release as a memory-safety backstop); and
4. the owning `nixl_device_view_handle` (and its CuTe `MemoryView` adapter).

Synchronize every participating CUDA stream before closing/releasing a device
view (context-manager exit calls `release()`).
Only after remote views are released may peer metadata be invalidated with
`remove_remote_agent()`. Only after local views are released may overlapping
memory be deregistered; this also applies to locally owned loopback regions in
remote/composite views. The high-level view records ownership per descriptor
and enforces both cases. The runnable example uses `ExitStack` to encode this
order even when setup or execution raises.

Agent metadata contains connection and registration information. Applications
must also exchange the destination transfer coordinates
`(address, length, device_id)` over their control plane; the example uses a
bounded file rendezvous to make that requirement explicit.

## Build and install

Prerequisites are CUDA, a UCX build exposing both host and GPU device APIs,
LLVM 20 (`clang`, `opt`, `llvm-dis`, and `llvm-as`), and PyTorch with CUDA
support. The CUDA 12 `cute` extra installs the validated
`nvidia-cutlass-dsl==4.5.1` and constrains `cuda-python` to the CUDA 12 major;
the example uses its `cuda.bindings.driver` API for topology qualification.

For the released meta-package, install the supported optional dependency:

```bash
python -m pip install 'nixl[cute-cu12]'
```

The direct CUDA 12 backend exposes the shorter `nixl-cu12[cute]` extra. This
release deliberately declares no CUDA 13 CuTe extra: CUTLASS DSL 4.5.1's
official `cu13` dependency set contains overlapping wheels
([NVIDIA/cutlass#3259](https://github.com/NVIDIA/cutlass/issues/3259)), while
the plain/base payload is not a CUDA 13 substitute. For a CUDA 13 source
environment, install CUTLASS DSL with its corresponding `setup.sh --cu13`
workflow, verify that environment independently, and install NIXL without an
extra. At import, `nixl.device.cute` verifies that the installed
`nvidia-cutlass-dsl` distribution is exactly 4.5.1. A source-only environment
without readable distribution metadata must opt in explicitly with
`NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL=1`; this unsafe escape hatch emits a
runtime warning and is not appropriate for release qualification. Do not turn
the upstream race into an implicit release dependency.

For a CUDA 12 source build, install NIXL with its UCX plugin and verified
device bitcode, for example:

```bash
python -m pip install 'cuda-python>=12.8,<13'
python -m pip install '.[cute]' \
  -Csetup-args=-Dbuild_cute_device=enabled \
  -Csetup-args=-Ducx_path=/opt/ucx \
  -Csetup-args=-Dcute_llvm_path=/opt/llvm-20/bin \
  -Csetup-args=-Dcute_bitcode_arch=sm_90
```

Production bitcode omits wrapper-only dynamic argument checks and marks every
public device entry point `alwaysinline`, so linked kernels carry neither a
wrapper call nor validation branches. The builder fails closed if the optimized
LLVM module loses that policy. For bring-up, build a separate checked artifact
with `-Csetup-args=-Dcute_device_validation=true`; the manifest records both the
validation and force-inline modes so benchmark artifacts cannot be confused.

`cute_bitcode_arch` accepts one SM, a comma-separated SM list, or `auto`. An
explicit single-SM source build retains the compatible `libnixl_device.bc` plus
`nixl_device_abi.json` names. Multi-SM and `auto` builds install pairs such as
`libnixl_device_sm_90.bc` plus `nixl_device_abi_sm_90.json`. Release wheels use
an explicit LLVM-20-compatible subset of the NIXL CUDA matrix. In particular,
ordinary NIXL CUDA objects retain `sm_103`, but CuTe bitcode omits it because
the qualified Clang 20 producer has no `sm_103` target. The CuTe binding fails
closed with an unsupported-SM diagnostic instead of loading a nearby artifact.

The CuTe binding resolves the current rank-local CUDA device only when an
extern is traced, after applications have selected their device. It requires an
exact matching artifact and validates its architecture, ABI version, LLVM
major, SHA-256, `forceinline=true`, and `device_validation=false`; unsupported
SMs fail explicitly. A generic single-SM source artifact is accepted only when
its manifest matches the active device. For development, an explicit
`NIXL_DEVICE_BITCODE` path is accepted with
`NIXL_DEVICE_ALLOW_UNSAFE_OVERRIDE=1`; it still requires an authenticated,
architecture-matched manifest but may intentionally enable debug validation.

Relevant implementation references:

- [High-level view preparation and lifetime guards](../../../src/api/python/_api.py)
- [CuTe-facing operations](../../../src/api/python/device/cute/ops.py)
- [MemoryView JIT adapter](../../../src/api/python/device/cute/memory.py)
- [Device ABI source](../../../src/api/gpu/ucx/cute/nixl_device_cute.cu)
- [Bitcode build and manifest installation](../../../src/api/gpu/ucx/cute/meson.build)
