# NIXL device operations from CuTe DSL

This directory demonstrates the version 3 NIXL device ABI from CUTLASS CuTe
DSL 4.5 kernels. The examples progress from one PUT to ordered signaling,
THREAD/WARP measurement, and a GPU-persistent transport benchmark. Every remote example uses distinct GPU processes: modern UCX needs
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
