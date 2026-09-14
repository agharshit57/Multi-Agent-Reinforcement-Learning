# Real-First Deployment Architecture (corrected end-to-end design)

## 1. The architectural flaw (confirmed by full trace)

The old deployment path presented the **CC4 simulation topology as the
physical network**: `generate_default_map()` fabricates 112 hosts with
synthetic `10.0.x.x` addresses, and the GUI (`Network` page, topology
canvas, threat/agent tables) renders CC4 logical zones — Contractor
Network, Restricted Zone A/B, Operational Zone A/B, … — as if they were
real infrastructure. Running on one PC therefore displayed a fictitious
112-host enterprise. The simulation's logical geometry leaked into every
layer because telemetry, normalization, observation, action tables and
the UI all spoke CC4 slot names directly, with no boundary marking what
is physical and what is policy-internal.

Reality audit of the old path: **genuinely real** were only the weight
loader, the validator/executor gate *logic*, and the live-transport
rules; **simulated** were the demo pipeline loop and mock effects;
**mock** were the policy/collectors; everything else was **abstraction**
(CC4-faithful shapes with no I/O). Nothing sensed a real machine.

## 2. Corrected architecture

```text
REAL WORLD (source of truth)
  RealAsset / RealSegment / RealTopology .... Deployement/real_world.py
  LocalMachineCollector (stdlib sensing) ..... genuinely real, 1 PC = 1 asset
  EDR/SIEM adapters (via live.py contract) .. site-owned, future
            |  asset_id-keyed telemetry, never CC4 names
            v
POLICY-COMPATIBILITY LAYER (isolated, deterministic, audited)
  PolicyAdapter ............................. Deployement/policy_adapter.py
    real assets (sorted) -> 112 CC4 slots (sticky; overflow = inventory)
    unpopulated slots = explicit EMPTY (quiet) + TEST-NET-3 labels
    bound slots carry REAL ip/hostname (live backends resolve reality)
  + FROZEN, UNCHANGED: normalizer, state_builder, observation ([5,210]),
    action_table/mask (82/242), policy_engine, checkpoint contract
            |  policy decisions (advisory only)
            v
REAL-WORLD ACTION LAYER
  translate_decision -> RealActionPlan ...... Deployement/real_actions.py
    phantom target / no footprint = BLOCKED plan (loud refusal)
  + FROZEN, UNCHANGED: validator (modes/cooldowns/approvals/idempotency),
    executor + backends, audit trail
  PhantomGuardBackend (2nd net: refuses placeholder host targets live)
            |
            v
INTEGRATION BOUNDARY (loopback, schema real-first/v1)
  Python sidecar ............................ Deployement/sidecar.py
  + Deployement/inference_service.py (/health /contract /decide /cycle
    /approve; Bearer auth; contract guard 210/242/137)
  Torch/MAPPO stays in Python; systematic refusal on schema drift.
            |
            v
C# SOC CONSOLE (Deployement/CyberMarl.Deployment/, .NET 8 + WPF, MVVM)
  Real Assets hero view / Policy Mapping technical drawer (warning
  banner) / Actions with approvals / Audit / Telemetry & Model.
  No weights, no geometry internals, no in-process enforcement.
  Existing Python Tkinter console untouched as reference/fallback.
```

## 3. What was preserved vs what changed

PRESERVED (frozen, byte-identical contracts): 137-host vocab and order,
9-subnet order, agent→zone ownership, slot order, 92/210 obs + [5,210]
batch, 82/242 action tables + sorted command order, mask semantics
(Sleep net, alert gates), comms 128-dim + 7-field schema, checkpoint
format + vocab validation, frozen trust, one-step comm delay, all four
safety modes + live opt-in, cooldowns, approvals single-ownership,
idempotency, backend interface + audit, live transport/secrets rules,
all 172 pre-existing tests (still passing, unmodified files).

CHANGED (additive only — no existing file edited): `real_world.py`,
`policy_adapter.py`, `real_actions.py`, `real_pipeline.py`,
`inference_service.py` (+`POST /approve`), `sidecar.py`,
`tests/test_real_architecture.py` (21 tests), and the new
`CyberMarl.Deployment/` solution (Core net8.0 + WPF App). The old
demo/GUI path runs exactly as before.

## 4. What remains simulated / mock (honest list)

* Policy-view simulation: mock/shadow/supervised executors, demo
  timeline, MockPolicyEngine — kept deliberately for tests and dry runs.
* Unpopulated slots: explicit quiet padding (labelled, never shown as
  machines). A 1-asset site genuinely under-constrains a 112-slot
  enterprise policy — the adapter contains that truthfully instead of
  fabricating a network.
* Process/connection inventory without psutil/EDR: reported unknown via
  `partial_errors`, never synthesized. Intrusion verdicts require real
  detectors (`intrusion_confirmed`-class events); discovery invents none.
* Live enforcement: interface + guards done; per-site vendor adapters and
  real EDR/SIEM collectors are still site work (unchanged blocker).

## 5. Suitability as a real-world foundation

Yes, with the documented boundary: real sensing exists on both sides
(Python `LocalMachineCollector`, C# `LocalDiscovery`); the frozen model
serves through an audited, deterministic mapping that cannot hallucinate
infrastructure; phantom actions are refused at two independent layers;
all safety gates are preserved and the full suite (211 tests) passes.
What would graduate it to production: site EDR/SIEM collectors on the
`live.py` contract, baselines per site, a real enforcement backend
behind the phantom guard, TLS/auth termination if the sidecar ever
leaves loopback, and multi-host fleet validation.

## 6. Component reality labels (fleet/telemetry stage)

Every deployment component carries exactly one label. Enforced in code
(`COMPONENT_KIND`) and tests (`TestComponentLabels`):

| Component | Label | Meaning |
|---|---|---|
| `endpoint_sensors` (tasklist//proc, netstat//proc-net-tcp, sc/systemctl) | real | OS-observed; unavailable sources report loudly |
| `network_inventory` (ARP, interface subnets) | real | wire/OS-observed; no /24 guessing |
| `DeclaredInventory` (fleet JSON) | adapter-dependent | operator-asserted; validated strictly |
| `FileAlertFeed`, `SyslogAuthSensor` | adapter-dependent | real detector output; missing = no coverage |
| remote `agent_collectors` | adapter-dependent | interface only; stdlib ships no remote agent yet |
| uncovered/ARP-stub assets | real (observed) | declared-or-wire proof only; read STALE |
| `MockPolicyEngine`, `MockCollector`, demo timeline | mock | deterministic stand-ins, never presented as real |
| mock/supervised executors, `MockBackend` | simulated | policy-view effects only, no network I/O |
| `normalizer/state_builder/observation/masks/policy` on the policy view | adapter-fed | frozen training math, real-sourced inputs |
| `validator` modes/cooldowns/approvals, `executor` gates, audit | real (logic) | enforced identically in every mode |
| `model_loader` weights, `TrainedPolicyEngine` | real | frozen checkpoint inference |
| Tkinter console, WPF console, Avalonia console | real (display) | no enforcement in-process, approvals delegated |
| CC4 zones/host slots anywhere outside `policy_adapter` | FORBIDDEN | internal policy abstraction only |

Rules that are never bent: no invented processes/connections/services;
no inferred compromise (only explicit detector kinds, decided by the
frozen normalizer); no guessed subnets; broadcast/multicast never
become assets; corrupt feeds are refused, not interpreted; uncovered
reads STALE, never quiet; overflow stays inventory-only.

## 8. Live-path hardening (reported failure modes, fixed + pinned)

Three safety/consistency guarantees whose implementations quietly
defeated themselves, all in the live-execution/approval path
(`Deployement/tests/test_live_safety.py` pins each):

1. **Phantom guard type-mismatch** (`real_actions.py`): the guard
   inspected dict-shaped assets only, but `LiveExecutor._real_asset`
   collapses the asset to a plain string before backends see it -- so
   it never fired in production while a dict-only unit test claimed it
   worked. `is_phantom_asset` now accepts both shapes (TEST-NET-3 /
   `unpopulated-policy-slot-*` match either way); anything else is not
   phantom. Fail-closed: an operator machine literally named like a
   placeholder is refused, never executed against.
2. **Approval position-aliasing** (`pipeline.py`, `executor.py`,
   `inference_service.py`, both UIs): the C# console numbered queued
   actions 0..N per cycle while the server queue persists across
   cycles, so approving after a new cycle could pop an older, unrelated
   decision while the stale row silently vanished from the UI. Every
   queued entry now carries a stable `approval_id` (minted once at
   queue time, preserved across approve/re-queue retries);
   `approve_pending` resolves ids to current positions (legacy integer
   positions still work against a fresh snapshot); the sidecar exposes
   `GET /approvals` and id-based `POST /approve`; both consoles render
   the server queue (stable ids, failed attempts retryable) and can no
   longer address positions at all (empty id refused client-side).
3. **`run_cmd` fail-safe** (`endpoint_sensors.py`): any non-zero exit
   is now "unavailable source" even with stdout present (previously a
   failing-but-chatty command got parsed as telemetry), matching the
   documented contract and `network_inventory.py`'s existing check.

## 7. Linux (WSL/Ubuntu) support

* Python deployment is fully Linux-native: sensors dispatch
  (`/proc`, `/proc/net/tcp{,6}`, `systemctl`, `/proc/net/arp`,
  `ip neigh`, `ip addr`, auth.log), stdlib-only, no torch needed for
  mock paths. Verified: **222/222 tests pass on Ubuntu 24.04**
  (incl. torch inference tests with CPU torch).
* Linux GUI: `src/Avalonia/` (Avalonia 11, `net8.0`, shares Core +
  the same ViewModels file as WPF). Run on Ubuntu with the .NET 8 SDK:
  `dotnet run --project src/Avalonia/...` (needs X11 client libs:
  `sudo apt-get install -y libice6 libsm6 libx11-6 libxcursor1
  libxrandr2 libxi6`, plus WSLg/a desktop). Self-contained publish
  verified: `dotnet publish -r linux-x64 --self-contained` boots the
  managed runtime on Ubuntu (fails only at missing OS X11 libs).
* `Global\`-prefixed mutexes and `MessageBox` were removed from shared
  paths (portable single-instance name, async confirm dialog).
