# NIXL Python API

The Python API can be found at `src/api/python/_api.py`. These are the pythonic APIs for NIXL, if more direct access to C++ style methods are desired,
the exact header implementation of `src/api/cpp` is done through pybind11 that can be found in `src/bindings/python`.

## Python API Features

The Python bindings provide access to the full NIXL API including:

- **Agent Management**: Create and configure NIXL agents
- **Memory Registration**: Register and deregister memory/storage
- **Transfer Operations**: Create and manage data transfers
- **QueryMem API**: Query memory/storage information and accessibility
- **Backend Management**: Create and configure different backends (UCX, GDS, etc.)

## Installation

### From PyPI

The nixl python API and libraries, including UCX, are available directly through PyPI:

```bash
pip install nixl
```

### From Source

To build from source, follow the main build instructions in the README.md, then install the Python bindings:

```bash
# From the root nixl directory
pip install .
```

## Backend initialization parameters

Backends expose their initialization parameters through `get_plugin_params(backend)`, which
returns the defaults. Override the values you need before passing the map to `create_backend`:

```python
from nixl import nixl_agent, nixl_agent_config

# backends=[] leaves backend creation to the caller. The default config
# initializes UCX, and a backend can only be created once per agent.
agent = nixl_agent("example_agent", nixl_agent_config(backends=[]))

params = agent.get_plugin_params("UCX")
params["ucx_error_handling_mode"] = "peer"   # or "none"
agent.create_backend("UCX", params)
```

See [UCX backend initialization options](BackendGuide.md#ucx-backend-initialization-options)
for the supported UCX keys and their semantics. Note that `ucx_error_handling_mode` influences
UCP transport lane selection in addition to error reporting.

## Prepared standalone notification sender

Repeated notification sends can resolve the destination and backend handles
once, outside the hot path:

```python
sender = agent.create_notif_sender("remote-agent", backends=["UCX"])
assert sender.send(b"request-ready") is None
```

The prepared sender retains its native agent, follows the agent's configured
synchronization mode, and releases the Python GIL while submitting the native
notification. A successful `None` return means local NIXL acceptance; it does
not prove that the remote application received or processed the payload.
In reader/writer synchronization mode, independent prepared sends may enter
concurrently; integrations must not wrap that native interval in a global
Python mutex. PyTorch Core owns a pooled, interruption-durable peer-lifetime
reservation; the NIXL provider captures the prepared callable under its short
ownership lock and enters this method after releasing that lock, without a
second hot-path reservation.
Protocols requiring that guarantee need application sequence numbers,
acknowledgements, deduplication, and retry. `agent.send_notif(...)` remains the
compatible one-shot operation.

## Effective agent synchronization mode

The direct pybind agent exposes a read-only query for the synchronization mode
that NIXL actually retained, rather than merely echoing the requested
configuration:

```python
from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t

agent = nixl_agent(
    "example-agent",
    nixl_agent_config(sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_NONE),
)
effective = agent.agent.getEffectiveSyncMode()
```

`getEffectiveSyncMode()` reports the immutable agent-lock mode selected during
construction. If any metadata backend starts a worker thread, NIXL upgrades a
requested `NIXL_THREAD_SYNC_NONE` to `NIXL_THREAD_SYNC_STRICT`, and the getter
reports `NIXL_THREAD_SYNC_STRICT`. It does not report a backend plugin's separate
worker-thread configuration. This query is intended for setup diagnostics and
benchmark attestation; it need not be called on a transfer hot path.

## PyTorch transfer provider ownership contract

`nixl.torch_transfer` advertises
`TORCH_TRANSFER_FACTORY_API_VERSION = 2`. Its idempotent
`register_torch_backend()` requires PyTorch Core to advertise
`BACKEND_FACTORY_API_VERSION >= 2` and registers the exact version-2 factory;
there is no signature inspection or fallback to version 1. Framework adapters
that rely on adopt-before-activation should require the provider's exact value
to be `2` before constructing an Endpoint.

The registration retry is also safe if Core committed before an asynchronous
exception prevented NIXL from storing its local completion bit. Core accepts
the retry only when the already-registered callable is the identical object and
the API version is exactly `2`. A foreign callable or contract remains a hard
`BusyError`; NIXL never replaces it implicitly.
The module-local `_REGISTERED` value is not an authority: every public
`register_torch_backend()` call enters Core's exact idempotent registration
transaction under the provider lock. Consequently an external Core
`unregister_backend("nixl")` is repaired by the next registration call, while a
foreign replacement remains protected by Core's collision error.

FactoryV2 creates a pure Python backend record, transfers it to Core through
`adopt_backend`, and only then creates/publishes the native NIXL agent. Normal
operations have one lifecycle gate (`_available`). Close stores that gate false
before entering its retry-only teardown state. PyTorch-provider deregistration
requires NIXL's durable prepare/execute receipt APIs before native mutation;
the general high-level NIXL API retains its documented legacy compatibility
route.

Reusable-request eviction and plan drain mark a request `RETIRING` before native
release. Dirty reconciliation idempotently finishes such release and never
rebuilds the request as idle. Native request-slot construction also rejects an
owned descriptor list created by a different `nixlAgent`. These checks are all
construction/teardown or dirty-recovery work and do not change the cached
submit, repost, or polling calls.

In `ThreadMode.MULTIPLE`, asynchronous notification send publishes and adopts
its normal provider `Work` under the short ownership lock, captures the prepared
sender (or agent fallback) strongly, and releases that lock before native entry.
The Work blocks backend close and Core keeps the peer alive. Notification poll
and manual progress share a separate destructive-consumer lock because both can
drain the same NIXL ingress queue; they no longer hold the provider-global lock
and therefore overlap data submit/status and notification sends. A reusable
weak-identity admission token blocks close only while the underlying call is
active. Sequential warm calls reuse one token and add no steady-state token
allocation. Non-MULTIPLE modes keep their existing Core/caller serialization.

Agent-construction options do not use Python truthiness or integer coercion.
`enable_nixl_progress_thread` (when supplied), `enable_listen_thread`, and
`capture_telemetry` must be exact `bool` values. `listen_port` must be an integer
other than `bool` in `0..65535`, matching native `uint16_t`, and `num_threads`
must be an integer other than `bool` in `0..4294967295`, matching the UCX
backend's unsigned parameter parser. Invalid values fail before the agent
factory is called; this validation is setup-only and does not affect a transfer
hot path.

## Examples

See the [Python examples](../examples/python/) directory for complete working examples including:

- [query_mem_example.py](../examples/python/query_mem_example.py) - QueryMem API demonstration
- [nixl_gds_example.py](../examples/python/nixl_gds_example.py) - GDS backend usage
- [nixl_api_example.py](../examples/python/nixl_api_example.py) - General API usage
- [basic_two_peers.py](../examples/python/basic_two_peers.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling
