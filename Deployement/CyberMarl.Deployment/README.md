# CyberMarl.Deployment — C# SOC Console (.NET 8 + WPF)

Real-first desktop frontend for the Cyber MARL deployment layer.
The existing Python Tkinter console (`Deployement/gui.py`) is untouched
and remains the reference/fallback. This app does **not** recreate those
screens: it is redesigned around the corrected architecture where the
**real environment is the source of truth** and CC4/policy ids are a
quarantined technical view.

## What lives where (native C# vs Python services)

| Component | Home | Notes |
|---|---|---|
| Real topology UX (assets, actions, approvals, audit) | C# `App` (WPF, MVVM) | Hero views show physical machines only |
| Local discovery (hostname, NICs, MACs, OS) | C# `Core/LocalDiscovery` | Genuinely real sensing, inbox APIs only |
| Policy-mapping drawer (technical) | C# `App` Policy Mapping tab | Warning banner; never infrastructure |
| Safety labels, risk, approval gating (display) | C# `Core/SafetyGates` | Mirror only — enforcement stays server-side |
| Audit sinks (memory + JSONL) | C# `Core/AuditLog` | Mirrors decisions.jsonl habits |
| Sidecar HTTP client + contract guard | C# `Core/InferenceClient` | Refuses schema/contract mismatch loudly |
| Telemetry → CC4 slots, obs/masks, MAPPO inference | Python `Deployement/` sidecar | `inference_service.py` + frozen layers |
| Trained weights (PyTorch MAPPO, `mappo_final.pt`) | Python only | Never ported; served over loopback |
| Approval queue (single-ownership), validator, executor, backends | Python only | C# calls `POST /approve` with a stable id; owns nothing |

## Communication (schema `real-first/v1`, loopback only)

Base URL + token from environment (never hardcoded, never logged):

```bat
set INFERENCE_URL=http://127.0.0.1:8410
set INFERENCE_TOKEN=<from the sidecar operator>
CyberMarl.Deployment.App.exe
```

Start the sidecar (training venv for real weights, deploy venv for mock):

```bash
# mock policy, loopback sidecar on :8410 (genuine local OS sensing)
python -m Deployement.sidecar --port 8410 --policy mock --mode shadow
# fleet mode: local sensing + ARP peers + declared hosts (fleet.example.json)
python -m Deployement.sidecar --port 8410 --policy mock --mode shadow \
    --fleet --inventory Deployement/fleet.example.json
# with a real detector alert feed (JSONL appended by a real sensor)
python -m Deployement.sidecar --port 8410 --policy mock --mode shadow \
    --fleet --alerts /var/log/cybermarl/alerts.jsonl
# trained policy (training venv + checkpoint)
python -m Deployement.sidecar --port 8410 --policy trained \
    --checkpoint checkpoints/fixedMaybe/mappo_final.pt --mode supervised
```

Alert-feed line format (one JSON object per line; missing file means
NO compromise coverage, loudly — never empty evidence):

```json
{"kind": "intrusion_confirmed", "severity": "critical",
 "details": "red session on host", "timestamp": 1717248000.0,
 "asset": "host:srv-01"}
```

Only explicit detector kinds (`intrusion_confirmed`, `red_session`,
critical IDS alerts naming red activity) can ever mark compromise, and
that verdict is computed by the frozen normalizer — sensors only
deliver events. Anything else (port scans, failed logins, unknown
processes/services/connections) is activity signal, at most.

Endpoints: `GET /health`, `GET /contract` (client refuses anything but
`obs 210 / act 242 / vocab 137 / real-first/v1`), `POST /decide`,
`POST /cycle {topology, batch}`, `GET /approvals` (full pending queue
with stable ids), `POST /approve {approval_id, approver}`.

Approvals are addressed by STABLE `approval_id`, never by queue
position: the server queue persists across cycles while list positions
shift, so position-based approval could pop the wrong (older) decision.
The UI renders `GET /approvals` (nothing silently disappears between
cycles); unknown ids fail loudly instead of popping a stale position.
Legacy integer `index` is still accepted by the server against a
freshly-read snapshot, but no UI uses it.
`POST /cycle` is shadow-safe by pipeline mode; live execution additionally
needs the Python `--enable-live` equivalent (a configured real backend —
default refuses loudly) plus the C# live-mode confirmation.

## Build

```bat
cd Deployement\CyberMarl.Deployment
dotnet build CyberMarl.Deployment.sln -c Release
```

Requires the .NET 8 SDK and Windows (WPF). No external NuGet packages —
restore works offline. `Core` is `net8.0` (portable); only `App` needs
`net8.0-windows`.

## Linux frontend (separate, WSL/Ubuntu-compatible)

WPF cannot run on Linux, so `src/Avalonia/` is a separate Avalonia 11
app (`net8.0`, cross-platform) that shares `Core` **and the exact same
`ViewModels.cs` file** (compile-linked, not copied) — identical
navigation, checklist, approvals and audit semantics on both shells.
CC4 mapping stays quarantined to the technical tab on both.

```bash
# Ubuntu 24.04 with the .NET 8 SDK (sidecar reachable at the URL below)
sudo apt-get install -y libice6 libsm6 libx11-6 libxcursor1 libxrandr2 libxi6
export INFERENCE_URL=http://127.0.0.1:8410 INFERENCE_TOKEN=dev-token-123
dotnet run --project Deployement/CyberMarl.Deployment/src/Avalonia -c Release
# self-contained, no SDK needed at runtime:
dotnet publish Deployement/CyberMarl.Deployment/src/Avalonia \
    -c Release -r linux-x64 --self-contained
```

Verified: solution builds with 0 warnings/errors; linux-x64 publish
boots the managed runtime on Ubuntu 24.04 (only missing-OS-X11-libs
remains without the apt line above); the full Python suite passes
222/222 on the same Ubuntu box.

## Reality audit for this app

* REAL: local discovery, displayed assets/addresses, approval intent,
  audit records, contract/health reads.
* DELEGATED (real, but executed in Python): inference, masking,
  validation, approvals execution, backends.
* NEVER here: weights, obs/action geometry internals, CC4 topology as
  infrastructure, process/connection inventory claims, intrusion
  verdicts (those arrive as real telemetry or stay unknown).
