"""Observation builder: CC4 state -> exact trained-policy vectors.

Replicates ``BlueFlatWrapper.observation_change`` plus
``Marl/mappo/train.py::pad_observation`` WITHOUT importing the
simulator, so the bytes the model sees in production equal the bytes
it saw in training:

native per agent:
    [mission(1) | zone block(s) | native messages(32, always zero here)]
zone block (59):
    [subnet one-hot(9) | blocked-subnets(9) | comms-policy(9)
     | process alerts(16) | connection alerts(16)]
shared-policy row (what the model consumes):
    210 dims -- mission + 3 block slots + message tail, with a
    single-zone agent's real block in slot 0 exactly like
    ``pad_observation`` does.

The comms-policy subvector is a pure-python port of
``BlueFlatWrapper._build_comms_policy_network`` (complete graph over
the 7 core subnets plus the two Restricted<->Operational links, with
the documented per-mission-phase edge removals).
"""

import numpy as np

from .config import (AGENT_ZONES, LARGE_OBS_DIM, MAX_HOSTS, MESSAGE_DIM,
                     MISSION_DIM, NUM_HQ_SUBNETS, NUM_SUBNETS, SMALL_OBS_DIM,
                     SUBNET_BLOCK_DIM, SUBNETS)

_CORE_SUBNETS = (
    "internet_subnet",
    "admin_network_subnet",
    "office_network_subnet",
    "public_access_zone_subnet",
    "contractor_network_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
)
_RESTRICTED_A = "restricted_zone_a_subnet"
_RESTRICTED_B = "restricted_zone_b_subnet"
_OPERATIONAL_A = "operational_zone_a_subnet"
_OPERATIONAL_B = "operational_zone_b_subnet"


def comms_policy_edges(phase):
    """Undirected edge set for a mission phase (frozenset pairs)."""
    edges = set()
    for i in range(len(_CORE_SUBNETS)):
        for j in range(i + 1, len(_CORE_SUBNETS)):
            edges.add(frozenset((_CORE_SUBNETS[i], _CORE_SUBNETS[j])))
    edges.add(frozenset((_RESTRICTED_A, _OPERATIONAL_A)))
    edges.add(frozenset((_RESTRICTED_B, _OPERATIONAL_B)))
    if phase == 1:  # MissionA: Restricted A links cut
        for other in (_OPERATIONAL_A, "contractor_network_subnet",
                      _RESTRICTED_B, "internet_subnet"):
            edges.discard(frozenset((_RESTRICTED_A, other)))
    elif phase == 2:  # MissionB: Restricted B links cut
        for other in (_OPERATIONAL_B, "contractor_network_subnet",
                      _RESTRICTED_A, "internet_subnet"):
            edges.discard(frozenset((_RESTRICTED_B, other)))
    return edges


def comms_policy_subvector(zone, phase):
    """9-dim inverted adjacency row for ``zone`` (ints, diagonal=1)."""
    edges = comms_policy_edges(phase)
    return np.array(
        [0 if frozenset((zone, other)) in edges else 1
         for other in SUBNETS],
        dtype=np.float32)


class ObservationBuilder:
    """Builds native + padded observation rows from a CC4State."""

    def __init__(self, asset_map):
        self.asset_map = asset_map

    # ------------------------------------------------------------- API --
    def build_native(self, agent_id, state):
        """Native-length vector: 92 dims (agents 0-3) / 210 (agent 4)."""
        zones = AGENT_ZONES[agent_id]
        parts = [np.array([state.mission_phase], dtype=np.float32)]
        for zone in zones:
            parts.append(self._zone_block(agent_id, zone, state))
        parts.append(np.zeros(MESSAGE_DIM, dtype=np.float32))
        return np.concatenate(parts)

    def build_padded(self, agent_id, state):
        """210-dim shared-policy row, replicating ``pad_observation``."""
        native = self.build_native(agent_id, state)
        if native.shape[0] == LARGE_OBS_DIM:
            return native.astype(np.float32)
        padded = np.zeros(LARGE_OBS_DIM, dtype=np.float32)
        padded[:MISSION_DIM] = native[:MISSION_DIM]
        n_blocks = ((native.shape[0] - MISSION_DIM - MESSAGE_DIM)
                    // SUBNET_BLOCK_DIM)
        real_end = MISSION_DIM + n_blocks * SUBNET_BLOCK_DIM
        padded[MISSION_DIM:real_end] = native[MISSION_DIM:real_end]
        tail = MISSION_DIM + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
        padded[tail:tail + MESSAGE_DIM] = native[-MESSAGE_DIM:]
        return padded

    def build_batch(self, state):
        """[5, 210] model-ready batch in blue_agent_0..4 order."""
        return np.stack([self.build_padded(i, state) for i in range(5)])

    # ---------------------------------------------------------- internals --
    def _zone_block(self, agent_id, zone, state):
        agent_name = f"blue_agent_{agent_id}"
        one_hot = np.array([zone == name for name in SUBNETS],
                           dtype=np.float32)
        blocked_here = state.blocks.get(zone, [])
        blocked = np.array([name in blocked_here for name in SUBNETS],
                           dtype=np.float32)
        policy = comms_policy_subvector(zone, state.mission_phase)
        proc = np.zeros(MAX_HOSTS, dtype=np.float32)
        conn = np.zeros(MAX_HOSTS, dtype=np.float32)
        for host_idx, cc4 in enumerate(self._slot_hostnames(agent_name,
                                                            zone)):
            flags = state.hosts.get(cc4)
            if flags is None:
                continue
            if flags.process_event:
                proc[host_idx] = 1.0
            if flags.connection_event:
                conn[host_idx] = 1.0
        return np.concatenate([one_hot, blocked, policy, proc, conn])

    def _slot_hostnames(self, agent_name, zone):
        """Slot hostnames in model order (server 0-5, user 0-9)."""
        from .asset_map import match_zone
        names = []
        for cc4 in sorted(self.asset_map.agents[agent_name]["hosts"]):
            if match_zone(cc4) == zone and "router" not in cc4:
                names.append(cc4)
        return names[:MAX_HOSTS]


def expected_native_dim(agent_id):
    zones = len(AGENT_ZONES[agent_id])
    return MISSION_DIM + zones * SUBNET_BLOCK_DIM + MESSAGE_DIM


def assert_layout():
    """Sanity: layout constants reproduce the training dimensions."""
    assert SUBNET_BLOCK_DIM == 59, SUBNET_BLOCK_DIM
    assert MESSAGE_DIM == 32, MESSAGE_DIM
    for agent in range(4):
        assert expected_native_dim(agent) == SMALL_OBS_DIM == 92
    assert expected_native_dim(4) == LARGE_OBS_DIM == 210
