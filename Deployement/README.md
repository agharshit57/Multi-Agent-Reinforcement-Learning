# Cyber MARL — Deployment Layer (`Deployement/`)

Real-world serving path for the trained CC4 MAPPO blue-team policy.
**Training/simulation code is never modified** — everything deployment
needs lives in this folder as adapters around the frozen `.pt` weights.

## Architecture

```
Real Network Telemetry
        |
        v
Telemetry Collector  (telemetry.py: Mock / File / Live-stub)
        |
        v
Telemetry Normalizer (normalizer.py: hosts, processes, connections,
        |             sessions, system state, security events
        v             -> compromised / process_event / connection_event)
CC4 State Builder    (state_builder.py: mission phase, blocks, flags)
        |
        v
Observation Builder  (observation.py: EXACT 92/210-dim training layout,
        |             padded to [5, 210] like train.py)
        v
Trained GNN + Attention  (reused Marl/mappo/gnn_attention.py, frozen)
        |  256-D representation (internal)
        v
Communication Decoder -> 7 structured fields (reused schema)
        |
        v
Communication Encoder -> message vectors (reused modules)
        |
        v
MAPPO policy -> per-agent action id  (greedy, mask applied)
        |
        v
Action Mask (action_mask.py: structural + alert-gated + Sleep net)
        |
        v
Validation/Translation (validator.py: modes, approvals, cooldowns)
        |
        v
Executor (executor.py: shadow / mock / supervised) -> Real Defensive Action
```

## Geometry (must match training — asserted in tests)

| agent | zones | obs native | obs model | actions |
|---|---|---|---|---|
| blue_agent_0..3 | 1 zone | 92 | 210 (padded) | 82 |
| blue_agent_4 | 3 zones | 210 | 210 | 242 |

Anchor indices (agent 0): `0 Analyse server_host_0`, `16 Monitor`,
`49 Sleep`, `50 Allow…`, `58 Block…`, `66 DeployDecoy`. HQ:
`48 Monitor`, `145 Sleep`, `146/170/194 Allow/Block/Deploy`.

## Install

Windows:
```bat
Deployement\install.bat
```
Linux/macOS/WSL:
```bash
bash Deployement/install.sh
```
This creates `deploy-venv`, installs training `Requirements.txt` plus
`requirements-deploy.txt` (numpy only), runs the test suite and a
headless demo. Model inference additionally needs the training venv
dependencies (torch, CybORG wrappers) — same environment as training.

## Run

```bash
python -m Deployement.app --demo --headless --cycles 5 --mode shadow
python -m Deployement.app --demo --gui --mode mock
python -m Deployement.app --assets site-assets.json \
    --checkpoint checkpoints/fixedMaybe/mappo_final.pt \
    --policy trained --mode supervised --gui
# or: Deployement/run_gui.bat --demo --mode mock
```

Modes: `shadow` (log-only, never mutates state), `mock` (simulated
effects on internal enforcement state, closed loop), `supervised`
(safe ops auto-apply simulated; destructive ops wait for human approval
in the GUI), `live` (approved ops execute for real via a configured
`EnforcementBackend`; **explicit opt-in only** via `--enable-live`,
default backend refuses everything loudly). There is deliberately **no**
fully-automatic enforcement mode for destructive actions.

Live mode safety contract:
- destructive ops need human approval, then execute exactly once
  (idempotency keys; no blind retries of enforcing ops);
- backend failures surface as `applied=False` + error, never silent
  success; `NullBackend` (default) refuses everything;
- only `collect_*` read-only ops retry (max 3 attempts);
- zone-block effects are the only auto-rollbackable ops; host
  remediation has no automatic inverse (documented manual step).
- host-enforcement backends always receive the RESOLVED real asset
  (IP/hostname from the asset map), never the simulated CC4 hostname;
  zone operations use real zone names shared with the map.

## Site enforcement backends (no source changes per site)

`--backend` (CLI) > site `"backend"` (`DEPLOY_BACKEND`) > `"null"`:
- `"null"` — refuses everything loudly (default, safe);
- `"mock"` — in-memory simulation with idempotency + audit + rollback;
- `"package.module:ClassOrFactory"` — imports your adapter and calls
  it with NO arguments (it reads its own credentials from the
  environment); the result must be an `EnforcementBackend` instance.
  Unimportable modules, missing attributes, non-backend objects, and
  constructors needing arguments all fail at startup with a clean
  error — a configured backend never silently falls back to mock/null,
  and live mode without any configured backend refuses loudly instead
  of pretending to enforce.

## Asset map

`assets.example.json` (112 bound slots) shows the format. Every CC4
slot hostname per agent zone must be bound once; extras go to
`unmapped` (inventoried, never fed to the model). Validation rejects:
duplicate slots/IPs/hostnames, invalid IPs, empty hostnames, invalid
roles, router/non-slot bindings, unknown agents, and ambiguous zone
names. Generate a starter:
```bash
python -c "from Deployement.asset_map import generate_default_map; \
generate_default_map().save('site-assets.json')"
```

Zone attribution is strict everywhere (`match_zone`): a hostname belongs
to a zone iff it starts with `zone + "_"`; zero or multiple matches
raise instead of guessing.

## Site configuration & secrets

Non-secret settings live in a site JSON file (`--site-config`), see
`site_config.py` for the schema (assets, checkpoint, policy, mode,
mission phase, cooldowns, collector, baselines path, backend).
Overrides via `DEPLOY_ASSETS`, `DEPLOY_CHECKPOINT`, `DEPLOY_POLICY`,
`DEPLOY_MODE`, `DEPLOY_MISSION_PHASE`, `DEPLOY_COOLDOWN_S`,
`DEPLOY_SESSION_DIR`, `DEPLOY_COLLECTOR_ENDPOINT`,
`DEPLOY_MAX_RESPONSE_BYTES`, `DEPLOY_STALE_AFTER_S`, `DEPLOY_BACKEND`.

## Site backend adapters (no source changes per site)

Enforcement backends load by configuration string (CLI `--backend` >
site `"backend"` > safe default `"null"`):
- `"null"` — refuses everything loudly (default);
- `"mock"` — in-memory simulation with idempotency + audit + rollback;
- `"module:Class"` — imports your adapter and constructs it with NO
  arguments (it reads its own credentials from the environment).
  Must be an `EnforcementBackend` instance, else startup fails loudly —
  a configured backend never silently falls back to mock/null.

Telemetry adapters work the same way via site `collector.translate`
(`"module:function"`, contract `dict -> dict[key, HostTelemetry]`).
`poll_interval_s` paces repeated live polls; `max_response_bytes`
(default 8 MiB) bounds payloads before parsing. Neither invents vendor
APIs: transport is implemented, vendor parsing/bodies are yours.

## Console tour (for normal users)

The GUI opens with an unmissable mode banner (SHADOW gray, MOCK amber,
SUPERVISED orange, LIVE red) plus a 4-step quick-start bar:
1) Rebuild, 2) Start/Step, 3) review Dashboard + Messages,
4) approve/deny destructive actions in Approvals (attempt counts shown
for failed-then-requeued items; full history in the Audit tab).
Switching to LIVE asks for explicit confirmation first. Health and
stale-host counts ride in the status bar every cycle.

Secrets (collector tokens etc.) are referenced BY VARIABLE NAME
(`collector.token_env`) and resolved from the environment at startup;
a missing variable fails fast naming the variable, never its value.
Config files containing secret values are rejected at load, and audit
records are scrubbed before logging.

## Telemetry health & baselines

Per-host health is explicit: `quiet` (fresh + no evidence),
`stale` (missing/stale/failed telemetry — forced alert posture, never
healthy), `compromised` (explicit intrusion evidence, sticky until a
recovery signal, a `recovered` remediation, or session reset).
A dead collector synthesizes all-stale cycles (visible in record
`health`/`stale_hosts` and GUI status) instead of quiet ones.

Baselines resolve per host with precedence host > role > zone >
global, per dimension (processes/ports/peers); undefined dimensions
are fail-safe (everything is an event). Telemetry from unmapped hosts
is inventoried (`unseen_keys`) and never fed to the model.

## Fully implemented vs mocked

Implemented: strict asset binding + validation; per-scope baselines;
explicit health (quiet/stale/compromised, sticky compromise,
fail-safe stale posture); compromise promotion into the trained alert
representation (dims unchanged); exact observation/action/mask
replication (tested index-for-index); checkpoint loading with vocab
validation; eval-semantics inference (`eval()` + full `no_grad()` +
finite/range output guards); validator (ranges, masks, bindings,
approval-time cooldowns, idempotency keys); shadow/mock/supervised/
live executors with audit trail; structured JSONL session logs;
tkinter console (mode banner, quick-start bar, dashboard/assets/
telemetry/obs/messages/approvals+audit/enforcement/log); demo timeline
with recovery; 150+-test suite (`pytest Deployement/tests/ -q`).

Mocked / stubbed: vendor telemetry parsing (adapter interface +
`translate` boundary; HTTP transport implemented); vendor enforcement
bodies (`EnforcementBackend` interface + `NullBackend`/`MockBackend`);
remediation/block effects outside mock/live-mock (internal state only).

## Blockers for a real network (honest list)

1. Site vendor adapters: a `translate(payload)` function and an
   `EnforcementBackend` implementation (EDR/SIEM/firewall APIs).
   Interfaces, timeouts, retries, idempotency, and audit hooks exist;
   vendor calls do not (by design — never faked).
2. Credentials via environment (`DEPLOY_*_TOKEN`-style vars) + network
   reachability to the collector endpoint.
3. Calibrated per-host/role/zone `baselines` — without them everything
   raises events (safe default, noisy).
4. Change-control sign-off + `--enable-live` for live enforcement;
   keep supervised mode until then.
5. A trained checkpoint from current code (137-host vocab); verified
   automatically on load (contract assert in the policy engine).
6. Display for the console (headless mode needs none).
