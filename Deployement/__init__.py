"""Deployement: real-world deployment layer for the Cyber MARL project.

This package is intentionally independent from training/simulation code.
It converts real network telemetry into the exact observation/action
representation the trained CC4 MAPPO policy expects, runs inference with
the frozen ``.pt`` weights, and translates policy decisions into safe,
validated real-world defensive actions (shadow/mock/supervised only).

Pipeline:
    Telemetry Collector -> Normalizer -> CC4 State Builder
    -> Observation Builder -> Policy Engine (trained GNN+MAPPO)
    -> Action Mask -> Validation/Translation -> Executor

Modes: shadow (log-only), mock (simulated), supervised (human
approval for destructive ops), live (explicit opt-in real
enforcement via EnforcementBackend).
"""

__version__ = "0.2.0"
