"""Deployment action masking (numpy-only, mirrors Marl/mappo/action_mask.py).

Two layers, recomputed every decision cycle:
  1. structural: True for every entry of the fixed tables (fixed
     mapping => every slot exists, mirroring in-sim sessions on all
     allowed hosts).
  2. contextual: Restore/Remove only on hosts with an active
     process/connection alert in the CURRENT normalized state;
     BlockTrafficZone only when its source zone is known-quiet-gated
     exactly like training (suppress iff source visible AND quiet;
     unknown sources stay allowed).

A Sleep safety net guarantees the mask is never all-False (an all
``-inf`` categorical would produce NaNs).
"""

import numpy as np


DISRUPTIVE = frozenset({"Restore", "Remove"})


def compute_mask(table, normalized, agent_zones, match_zones=None):
    """Boolean mask over a per-agent action table.

    ``match_zones`` overrides the zone universe used for attribution
    (tests/adversarial configs); None means the CC4 SUBNETS default.
    """
    from .asset_map import match_zone
    mask = np.ones(len(table), dtype=bool)
    sleep_index = None
    # Visible zones = this agent's own zones (same observability rule
    # as training: flags only ever cover the agent's own zone).
    zone_flags = {}
    for cc4, host in normalized.hosts.items():
        zone = (match_zone(cc4, match_zones) if match_zones is not None
                else _zone_of(cc4))
        if zone in agent_zones:
            flagged = bool(host.process_event or host.connection_event)
            zone_flags[zone] = zone_flags.get(zone, False) or flagged
    host_flags = {cc4: bool(h.process_event or h.connection_event)
                  for cc4, h in normalized.hosts.items()}
    for i, entry in enumerate(table):
        command = entry["command"]
        if command == "Sleep" and sleep_index is None:
            sleep_index = i
        if command in DISRUPTIVE:
            if not host_flags.get(entry["target"], False):
                mask[i] = False
        elif command == "BlockTrafficZone":
            source = str(entry["target"]).lower()
            if source in zone_flags and not zone_flags[source]:
                mask[i] = False
    if not mask.any():
        mask[sleep_index if sleep_index is not None else 0] = True
    return mask


def pad_mask(mask, action_dim):
    """Pad a real-sized mask to the shared-policy width (tail=False)."""
    padded = np.zeros(action_dim, dtype=bool)
    padded[:len(mask)] = mask
    return padded


def _zone_of(cc4):
    # Single shared implementation (strict, unambiguous-or-raise).
    # asset_map imports only stdlib+json+config, so this adds no heavy
    # dependencies to the masking path.
    from .asset_map import match_zone
    return match_zone(cc4)
