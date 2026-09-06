"""CC4 action tables: index <-> (command, target) without the simulator.

Replicates ``BlueFixedActionWrapper._populate_action_space`` exactly:
  - commands in verified ``sorted(classes, key=str)`` order:
    Analyse, Monitor, Remove, Restore, Sleep, AllowTrafficZone,
    BlockTrafficZone, DeployDecoy
  - Sleep / Monitor: single entries, always valid
  - Allow/BlockTrafficZone: every (dst zone of this agent) x (all 9
    subnets as src, minus self) pair; ``from_subnet`` lowercased
  - host commands (Analyse/Remove/Restore/DeployDecoy): every slot
    hostname of the agent's zones in global sorted order, routers
    skipped (they become Sleep placeholders in-sim; here the slot is
    simply absent because routers are not observable slots)
  - every entry structurally valid (fixed mapping => blue presence on
    all slots, mirroring in-sim sessions on all allowed hosts)

Resulting sizes: 82 (agents 0-3), 242 (agent 4) -- asserted at build.
Each entry carries the CC4 slot identity plus the bound real asset so
the translation layer never guesses a target.

Label format mirrors the simulator's
``f"{cmd} {dst} ({cidr}) <- {src} ({cidr})"`` with deterministic
placeholder CIDRs (see config.PLACEHOLDER_CIDRS); real IPs travel in
the entry's ``asset`` field, not the label.
"""

from .asset_map import zone_slot_hostnames
from .config import (AGENT_ZONES, COMMAND_ORDER, LARGE_ACTION_DIM,
                     PLACEHOLDER_CIDRS, SMALL_ACTION_DIM, SUBNETS)


def _placeholder_cidr(zone):
    return PLACEHOLDER_CIDRS[SUBNETS.index(zone)]


def build_action_table(agent_id, asset_map):
    """List of action entries for one agent, index = policy action id."""
    agent_name = f"blue_agent_{agent_id}"
    zones = sorted(AGENT_ZONES[agent_id])
    table = []

    def hosts_in_order():
        names = []
        for zone in zones:
            names.extend(zone_slot_hostnames(zone))
        return sorted(names)

    zone_pairs = [(dst, src) for dst in zones for src in SUBNETS
                  if src != dst]

    for command in COMMAND_ORDER:
        if command == "Sleep":
            table.append(_entry(len(table), command, "none", None, None,
                                agent_name, asset_map, "Sleep"))
        elif command == "Monitor":
            table.append(_entry(len(table), command, "none", None, None,
                                agent_name, asset_map, "Monitor"))
        elif command in ("AllowTrafficZone", "BlockTrafficZone"):
            for dst, src in zone_pairs:
                label = (f"{command} {dst} ({_placeholder_cidr(dst)}) "
                         f"<- {src.lower()} ({_placeholder_cidr(src)})")
                table.append(_entry(len(table), command, "zone", dst, src,
                                    agent_name, asset_map, label))
        else:  # Analyse / Remove / Restore / DeployDecoy
            for cc4 in hosts_in_order():
                zone = _zone_of(cc4)
                table.append(_entry(len(table), command, "host", zone, cc4,
                                    agent_name, asset_map,
                                    f"{command} {cc4}"))
    expected = (LARGE_ACTION_DIM if agent_id == 4 else SMALL_ACTION_DIM)
    if len(table) != expected:
        raise AssertionError(
            f"agent {agent_id}: built {len(table)} actions, "
            f"expected {expected} -- layout drift vs training")
    return table


def _entry(index, command, kind, zone, target, agent_name, asset_map,
           label):
    asset = None
    if kind == "host":
        asset = dict(asset_map.agents[agent_name]["hosts"][target])
    return {"index": index, "command": command, "kind": kind,
            "zone": zone, "target": target, "label": label,
            "asset": asset}


def _zone_of(cc4):
    # Single shared implementation (strict, unambiguous-or-raise).
    from .asset_map import match_zone
    return match_zone(cc4)


def build_all_tables(asset_map):
    return {i: build_action_table(i, asset_map) for i in range(5)}
