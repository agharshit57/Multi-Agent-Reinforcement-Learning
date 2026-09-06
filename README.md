# Multi-Agent Reinforcement Learning for Cyber Defense 

This repository implements a **Multi-Agent Proximal Policy Optimization (MAPPO)** system for autonomous cyber defense, built on the **CAGE Challenge 4 (CC4)** scenario in the [CybORG](https://github.com/cage-challenge/CybORG) simulation environment. A team of blue-team defender agents is trained under a centralized-training, decentralized-execution (CTDE) paradigm to detect, contain, and recover from network intrusions carried out by red-team adversaries of increasing sophistication.

Beyond a standard MAPPO baseline, this project explores how **structured communication** and **relational reasoning over network topology** can improve coordination between defenders that only observe a local slice of the network. Each agent sees a partial view of its own host/subnet; the system's goal is to let agents reason about the network as a graph and share the right information with each other to compensate for that partial observability.

## Environment

CAGE Challenge 4 models a realistic multi-subnet enterprise network with multiple cooperating blue agents, each responsible for defending a portion of the network, against red agents that laterally move, escalate privileges, and impact critical services. Because no single agent observes the whole network, effective defense depends on:

1. **Representing structure** — the network isn't a flat vector; it has hosts, subnets, and missions with meaningful relationships between them.
2. **Coordinating under partial observability** — agents need to share relevant information without flooding each other with noise.
3. **Stable centralized value estimation** — a shared critic needs to make sense of multiple agents' observations jointly, which is often the actual bottleneck in MARL training rather than raw policy network capacity.

This project addresses each of these directly, rather than treating MAPPO as an off-the-shelf algorithm applied to a flattened observation space.

### Simulated Network Representation:- 


<img width="2268" height="1835" alt="image" src="https://github.com/user-attachments/assets/b8448f8c-b422-4246-90c9-a6ef36ae8e3f" />



## Architecture

**Actor — Graph Neural Network.** Each agent's flat observation is decomposed into five semantic entity tokens, then passed through graph message-passing layers structured around a host → subnet → mission hierarchy. This replaces a naive single-token self-attention actor with one that can propagate information along the actual topology of the network being defended.

**Critic — Cross-Agent Attention.** The critic is centralized (as MAPPO requires) and attends jointly over all agents' `(NUM_AGENTS, OBS_DIM)` observation tokens, rather than concatenating them naively. Critic instability — not actor capacity — was identified as the primary failure mode during development, diagnosed via monotonic policy regression across checkpoints and growing critic loss variance across training. This motivated dedicated diagnostics (explained variance logging, separated entropy coefficients for action vs. communication policies) built directly into the training loop.

**Communication — Structured & Differentiable.** Agents exchange learned messages through an explicit encoder → schema → decoder → evaluator pipeline (`Marl/mappo/communication/`), with a dynamic trust mechanism weighting how much an agent relies on messages from each of its teammates. Messages are reconstructed at PPO update time from stored source observations to preserve correct gradient flow back through the communication channel — communication is trained end-to-end alongside the policy, not treated as a fixed side-channel.

**Curriculum Training.** Agents are trained against a progression of red agents of increasing capability — starting with `RandomSelectRedAgent` and advancing to the more deliberate `FiniteStateRedAgent` — so that policies learn robust general defense behavior before facing adversaries with explicit attack strategies.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Exploring the Environment](#exploring-the-environment)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints & Logs](#checkpoints--logs)
- [Deployment (real-world serving)](#deployment-real-world-serving)

## Repository Structure

```
.
├── checkpoints/            # Saved model checkpoints from training runs
├── CybORG/                 # CAGE Challenge 4 simulation environment
│   ├── Agents/              # Built-in simple agents and environment wrappers
│   ├── Evaluation/          # Official CC4 evaluation harness and example submissions
│   ├── Shared/Scenarios/    # Scenario definitions
│   ├── Simulator/           # Core simulator: actions (abstract/concrete/decoy/exploit/escalate), green actions
│   └── Tests/                # Unit and acceptance tests for CC4, Green/Red agents
├── evaluation/              # Training logs and evaluation plots/output
├── Deployement/             # Real-world deployment layer (see below)
│   ├── endpoint_sensors.py    # Genuine OS sensing: processes, TCP, services, alert feeds (Win/Linux)
│   ├── network_inventory.py   # ARP, subnets, declared fleet, multi-host topology, FleetCollector
│   ├── real_world.py          # REAL assets/topology (source of truth, never CC4 names)
│   ├── policy_adapter.py      # Isolated real→CC4 slot mapping (deterministic, audited)
│   ├── real_actions.py        # Policy decision → real-operation plans + phantom-target refusals
│   ├── real_pipeline.py       # RealWorldPipeline (frozen layers composed, all safety modes)
│   ├── inference_service.py + sidecar.py  # Loopback HTTP sidecar, schema real-first/v1
│   ├── gui.py                 # Legacy Python/Tkinter console (reference/fallback)
│   ├── fleet.example.json     # Declared-fleet example
│   ├── REAL_FIRST_ARCHITECTURE.md  # Corrected architecture + component reality labels
│   ├── CyberMarl.Deployment/  # C# SOC consoles: WPF App (Windows) + AvaloniaApp (Linux)
│   └── tests/                 # 222 deployment tests (mapping, sensors, failure cases, gates)
├── Marl/
│   ├── gnn/                  # GNN models and environment wrapper for the actor/critic
│   └── mappo/                # Core MAPPO implementation
│       ├── communication/     # Structured differentiable comms: encoder, decoder, schema, evaluator, trust
│       ├── action_mask.py     # Action masking logic
│       ├── buffer.py          # Rollout buffer
│       ├── config.py          # Training/environment configuration
│       ├── env.py             # CC4 environment wrapper for MARL training
│       ├── evaluate.py        # Checkpoint evaluation script
│       ├── gnn_attention.py   # Hierarchical GNN actor architecture
│       ├── mappo.py           # MAPPO algorithm (PPO updates, GAE, losses)
│       ├── network.py / network_attention.py  # Actor/critic network definitions
│       ├── train.py           # Main training entry point
│       ├── utils.py           # Shared utilities
│       └── value_norm.py      # Value normalization for the critic
└── playschool/              # Scratch utilities for exploring the CC4 environment (agents, actions, observations, network topology) — see below
```

## Setup & Installation :-

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anirudh110106/Multi-Agent-Reinforcement-Learning
   cd Multi-Agent-Reinforcement-Learning
   ```

2. **Create and activate a Python virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r Requirements.txt
   ```

4. **Install the CybORG package in editable mode:**

   ```bash
   pip install -e ./CybORG
   ```

5. **Install system dependencies** (required for some GUI/evaluation components):

   ```bash
   sudo apt install python3-tk --assume-yes
   ```

## Exploring the Environment

The `playschool/` directory holds small standalone scripts for poking at the CC4 environment before or alongside training — inspecting agents, listing available actions, viewing the network topology, and examining both raw and processed observation spaces. Run any script from the repository root, e.g.:

```bash
python3 -m playschool.run_cc4
```

This is the fastest way to sanity-check environment behavior (action spaces, observation shapes, agent rosters) without spinning up a full training run.

## Training

The MAPPO training implementation lives in `Marl/mappo/`. Agents are trained with a curriculum that progressively increases red-agent difficulty, from `RandomSelectRedAgent` to `FiniteStateRedAgent`. The actor is a hierarchical GNN (`gnn_attention.py`) operating over host/subnet/mission node types, and the critic (`network_attention.py`) uses cross-agent attention for centralized value estimation. Inter-agent communication is handled by the structured, differentiable pipeline in `Marl/mappo/communication/`.

To start training:

```bash
python3 -m Marl.mappo.train
```

Configuration (hyperparameters, curriculum schedule, network dimensions, etc.) is set in `Marl/mappo/config.py`.

Checkpoints are saved automatically to `checkpoints/`. Training logs and plots (episode return, actor/critic loss, entropy, explained variance) are written to `evaluation/`.

## Evaluation

Once you have trained checkpoints, use `Marl/mappo/evaluate.py` to run full episodes with a frozen policy and measure performance against different red agents.

- **Evaluate a single checkpoint:**

  ```bash
  python3 -m Marl.mappo.evaluate --checkpoint checkpoints/yourFolder/mappo_final.pt
  ```

- **Evaluate against a specific red agent** (e.g. `finite`):

  ```bash
  python3 -m Marl.mappo.evaluate --checkpoint checkpoints/yourFolder/mappo_final.pt --episodes 100 --red-agent finite
  ```

- **Evaluate a sweep of checkpoints:**

  ```bash
  python3 -m Marl.mappo.evaluate --sweep checkpoints/ --episodes 50
  ```

Available options for `--red-agent` are `random`, `finite`, or `both`.

## Checkpoints & Logs

- `checkpoints/` — serialized model weights saved periodically during training.
- `evaluation/` — training curves, evaluation results, and diagnostic plots (e.g. explained variance, per-red-agent-type performance breakdowns).

## Deployment (real-world serving)

`Deployement/` serves the frozen CC4 MAPPO policy (`checkpoints/fixedMaybe/mappo_final.pt`) against **real networks** — training/simulation code is never modified. Full design: `Deployement/REAL_FIRST_ARCHITECTURE.md`.

**Core idea (three-way separation).** The real environment is the source of truth; CC4 zones/host slots are strictly an internal policy abstraction:

```
REAL telemetry (OS sensors, ARP, fleet, detector feeds)
  → normalized real state → PolicyAdapter → exact 210-dim obs
  → MAPPO inference → action mask → policy action
  → real-world action translation → validation/approval → execution
```

- **REAL**: OS sensors (`tasklist`/`/proc`, `netstat`/`/proc/net/tcp`, `sc`/`systemctl`), ARP/subnet observation, frozen-checkpoint inference, validator/executor safety gates, audit.
- **Adapter-dependent**: declared fleet file, detector alert feeds, remote agents (interfaces provided; missing = loud "no coverage", never invented data).
- **Mock/simulated**: mock policy, demo timeline, mock executors — dry-run tools, never presented as real.
- **Never**: invented processes/connections, inferred compromise (only explicit detector kinds, decided by the frozen normalizer), guessed subnets, or CC4 names shown as infrastructure. Uncovered assets read STALE, never healthy. Safety modes `shadow/mock/supervised/live` (+ live opt-in, approvals, idempotency, phantom-target guards) are preserved end to end.

**Run (Windows):**

```bat
:: sidecar with genuine local sensing
set INFERENCE_TOKEN=dev-token-123
python -m Deployement.sidecar --port 8410 --policy mock --mode shadow
:: fleet + detector feed
python -m Deployement.sidecar --port 8410 --policy mock --mode shadow --fleet --inventory Deployement\fleet.example.json
:: legacy Tkinter console
python -m Deployement.app --demo --gui --mode mock
```

**Run (Linux / WSL Ubuntu):** same Python sidecar (native `/proc` sensors), plus the separate Avalonia GUI:

```bash
export INFERENCE_TOKEN=dev-token-123
python3 -m Deployement.sidecar --port 8410 --policy mock --mode shadow --fleet
# new terminal:
sudo apt-get install -y libice6 libsm6 libx11-6 libxcursor1 libxrandr2 libxi6  # one time
export INFERENCE_URL=http://127.0.0.1:8410 INFERENCE_TOKEN=dev-token-123
dotnet run --project Deployement/CyberMarl.Deployment/src/Avalonia -c Release
```

**C# consoles** (`Deployement/CyberMarl.Deployment/`): WPF `App` (Windows) and Avalonia `AvaloniaApp` (Linux) share Core + the same ViewModels — real-assets hero view, approvals, audit; CC4 mapping quarantined to a warning-bannered technical tab. Weights stay in Python (loopback sidecar, Bearer auth, contract guard 210/242/137). Build all: `dotnet build Deployement\CyberMarl.Deployment\CyberMarl.Deployment.sln -c Release`.

**Tests:** `python -m pytest Deployement/tests/ -q` — **222 passed on Windows and on WSL Ubuntu 24.04** (needs `numpy`, `pytest`, CPU `torch` there for the torch-gated tests), covering frozen contracts, real→policy mapping, sensor parsers (Win+Linux fixtures), telemetry failure cases, safety gates, and the sidecar protocol.

---

This project was developed as an academic Software Development Cycle (SDC-II) submission, centered on structured inter-agent communication, hierarchical GNN-based policies, and dynamic trust modeling for multi-agent cyber defense.
