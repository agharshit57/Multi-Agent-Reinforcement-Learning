"""
env.py

Environment interface for MAPPO.

This file is the ONLY place that interacts with CybORG.
The rest of the project only imports CC4Env.

Ground truth
------------

get_ground_truth(sender_id) reports what is ACTUALLY true in the sender's
zone, RIGHT NOW, as pure per-host facts (and, as of this revision, the
same facts aggregated per-subnet -- see "Subnet-level ground truth"
below). It is deliberately independent of any agent message: it does not
receive, inspect, or resolve a claimed target, and it never substitutes
one host (or subnet) for another. Grading a message against this truth is
entirely the MessageEvaluator's job.

    CybORG true state
          |
          v
        env.py  ->  PURE GROUND TRUTH  ->  MessageEvaluator
                                               ^
                                               |
                                     agent's StructuredMessage

This separation is what fixes the old wrong-target rescue: if an agent
claims COMPROMISE on Host_3 while Host_7 (not Host_3) is the compromised
one, env.py simply reports the true state of every relevant host in the
zone. Host_3 is absent from target_status, so the evaluator sees the
claim was about an uninvolved host and scores it wrong -- the claim is
NOT silently re-pointed at Host_7.

It distinguishes three tiers per host, using signals verified against
CybORG's own action source (see _compute_host_snapshot()):

    COMPROMISE          an active Red session exists on the host --
                         the one signal that cannot come from benign
                         Green activity.

    SUSPICIOUS_ACTIVITY a process/connection event exists but no Red
                         session is confirmed. NOT Red-exclusive --
                         GreenLocalWork/GreenAccessService write into
                         the same event fields as Red's exploit/
                         portscan actions -- so this is a "something
                         happened here" signal, not proof of Red
                         involvement.

    NONE                neither.

Still a heuristic, not a perfect oracle: it cannot yet distinguish
SCAN from LATERAL_MOVEMENT from PRIVILEGE_ESCALATION within the
SUSPICIOUS_ACTIVITY/COMPROMISE tiers (that would need per-action-type
event tagging CybORG doesn't currently expose at the Host.events
level).

Subnet-level ground truth (this revision)
--------------------------------------------
HOST and SUBNET target claims used to share a single host-indexed
ground truth, which meant SUBNET claims could never actually be graded
(see communication/evaluator.py's old handling -- always excluded from
scoring). get_ground_truth() now ALSO returns `subnet_status`, built by
aggregating the SAME per-host snapshot up to the subnet level: a subnet
is reported at whichever tier its most severe relevant host currently
sits at. Subnet IDs use the deterministic mapping
`sorted(state.subnet_name_to_cidr.keys())` -- the same alphabetical CC4
subnet ordering already used by gnn_attention.py's SUBNET_NAME_ORDER, so
"index 3" means the same subnet everywhere in this project that needs to
agree on it. See get_num_subnet_targets() and communication/encoder.py's
independent subnet-target embedding table for the other two pieces of
this fix.

Ground truth here is still host-and-subnet level only; there is no
finer-grained target type currently in the schema.
"""

import numpy as np

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents.Wrappers.BlueFlatWrapper import NUM_HQ_SUBNETS, MAX_HOSTS

from .config import EPISODE_LENGTH
# from .communication.schema import EventType, HostStatus, ThreatLevel
from .communication.schema import (
    EventType,
    HostStatus,
    ThreatLevel,
    TargetType,
)


class CC4Env:

    def get_num_targets(self):
            """
            Return the number of possible HOST communication targets.

            Target IDs use the deterministic mapping:

                sorted(state.hosts.keys())

            Therefore the target vocabulary size is the number
            of hosts currently known to CybORG.

            This is the HOST vocabulary specifically -- see
            get_num_subnet_targets() for the separate, independent
            SUBNET vocabulary. The two must never be conflated; see
            communication/schema.py's module docstring.
            """

            state = self.cyborg.environment_controller.state

            hostnames = sorted(state.hosts.keys())

            return len(hostnames)

    def get_num_subnet_targets(self):
        """
        Return the number of possible SUBNET communication targets.

        Subnet target IDs use the deterministic mapping:

            sorted(state.subnet_name_to_cidr.keys())

        the same alphabetical ordering CybORG's own BlueFlatWrapper
        uses for its per-slot subnet one-hot (and that
        gnn_attention.py's SUBNET_NAME_ORDER already reproduces) --
        reusing that convention keeps subnet identity consistent
        across every part of the pipeline that has to agree on
        "which index means which subnet".

        This is the SUBNET vocabulary specifically, an INDEPENDENT
        index space from get_num_targets()'s HOST vocabulary -- index
        3 here and index 3 there mean unrelated things. See
        communication/schema.py's module docstring and
        communication/encoder.py's separate subnet-target embedding
        table.
        """

        state = self.cyborg.environment_controller.state

        subnet_names = sorted(state.subnet_name_to_cidr.keys())

        return len(subnet_names)

    def __init__(self, red_agent_class=FiniteStateRedAgent):

        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent_class,
            steps=EPISODE_LENGTH,
        )

        cyborg = CybORG(scenario_generator=scenario)

        # Kept so get_ground_truth() (and anything else that needs true
        # state) can reach environment_controller.state directly, the
        # same way TrueStateTableWrapper does. Previously this was a
        # local variable in __init__ and was lost after construction.
        self.cyborg = cyborg

        # Official CC4 MARL wrapper
        self.env = EnterpriseMAE(cyborg)

        self.agent_names = list(self.env.agents)

        # ------------------------------------------------------------
        # Ground-truth state snapshot -- see step() / reset() and
        # get_ground_truth() below for why this has to be captured
        # BEFORE each env.step() call rather than read live.
        # ------------------------------------------------------------

        self._pre_step_host_snapshot = self._compute_host_snapshot()

    ############################################################

    def reset(self, seed=None):

        observations, info = self.env.reset(seed=seed)

        # Fresh episode -- re-snapshot from the post-reset state so
        # the first messages generated this episode are graded
        # against the state they actually saw.
        self._pre_step_host_snapshot = self._compute_host_snapshot()

        return observations, info

    ############################################################

    def step(self, actions, messages=None):

        if messages is None:
            messages = {}

        # ------------------------------------------------------------
        # Snapshot ground truth BEFORE stepping.
        #
        # Messages arriving in this call were generated by train.py
        # from the observation as of the START of this call (i.e.
        # right after the PREVIOUS step/reset) -- see get_ground_truth()
        # for the full explanation. self.cyborg.environment_controller
        # .state is mutated in place by self.env.step() below, so this
        # has to run first or get_ground_truth() ends up grading
        # messages against a state that didn't exist yet when they
        # were written.
        # ------------------------------------------------------------

        self._pre_step_host_snapshot = self._compute_host_snapshot()

        observations, rewards, terminated, truncated, info = \
            self.env.step(actions, messages)

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    @property
    def agents(self):

        return self.env.agents

    @property
    def possible_agents(self):

        return self.env.possible_agents

    def observation_space(self, agent):

        return self.env.observation_space(agent)

    def action_space(self, agent):

        return self.env.action_space(agent)

    def sample_actions(self):
        """
        Random action for every blue agent.
        Useful for testing.
        """

        actions = {}

        for agent in self.agents:
            actions[agent] = self.action_space(agent).sample()

        return actions

    def get_observation_dims(self):
        dims = {}
        for agent in self.agents:
            dims[agent] = self.observation_space(agent).shape[0]

        return dims

    ############################################################

    def get_action_dims(self):

        dims = {}

        for agent in self.agents:
            dims[agent] = self.action_space(agent).n

        return dims

    ############################################################
    # Passthrough: action-space introspection
    #
    # Needed by action_mask.py to build the Adaptive Action Mask
    # without any file outside env.py touching CybORG directly.
    ############################################################

    def action_mask(self, agent_name):
        """
        Structural validity mask for `agent_name`, as maintained by
        CybORG's BlueFixedActionWrapper.

        True  -> action at this index currently targets a real
                 host/subnet that exists this episode.
        False -> action is a padding/no-op placeholder (e.g. the
                 target host was not generated this episode).

        This is recomputed by CybORG at every env.reset() (see
        BlueFixedActionWrapper._populate_action_space), so it is
        already episode-fresh -- it does NOT change mid-episode.
        """

        return self.env.action_mask(agent_name)

    def action_labels(self, agent_name):
        """Human-readable label per action index (for logging/debugging)."""

        return self.env.action_labels(agent_name)

    def actions(self, agent_name):
        """Ordered list of the underlying CybORG Action objects."""

        return self.env.actions(agent_name)

    ############################################################
    # Live, agent-legitimate signals for the Adaptive Action Mask
    #
    # Both of the following are re-derived directly from CybORG's
    # host-event log -- the SAME event data BlueFlatWrapper.observation_change
    # projects into `malicious_processes` / `network_connections` in the
    # agent's own observation vector. This is NOT privileged red-team
    # ground truth (unlike get_ground_truth() below, which is only used
    # for trust supervision) -- it is exactly what the agent already
    # "sees", just re-read here instead of re-parsed out of a padded,
    # per-subnet-flattened observation array.
    ############################################################

    def get_host_alert_flags(self, agent_name):
        """
        Return {hostname: bool} for every host this agent's action space
        can target (routers excluded, they are never action targets).

        True means the host currently has an outstanding malicious-process
        or suspicious-network-connection event -- i.e. it is currently
        "flagged" from the agent's own point of view.
        """

        state = self.cyborg.environment_controller.state

        flags = {}

        for hostname in self.env.hosts(agent_name):

            if "router" in hostname:
                continue

            if hostname not in state.hosts:
                flags[hostname] = False
                continue

            events = state.hosts[hostname].events

            has_process_event = bool(
                events.old_process_creation or events.process_creation
            )

            has_connection_event = bool(
                events.old_network_connections or events.network_connections
            )

            flags[hostname] = has_process_event or has_connection_event

        return flags

    def get_zone_alert_flags(self, agent_name):
        """
        Return {subnet_name: bool}: True if ANY host in that subnet is
        currently flagged (see get_host_alert_flags). Used to gate
        BlockTrafficZone so an agent cannot cut off a zone with no
        observed malicious activity.
        """

        state = self.cyborg.environment_controller.state

        host_flags = self.get_host_alert_flags(agent_name)

        zone_flags = {}

        for hostname, flagged in host_flags.items():

            subnet = state.hostname_subnet_map.get(hostname)

            if subnet is None:
                continue

            subnet = str(subnet).lower()

            zone_flags[subnet] = zone_flags.get(subnet, False) or flagged

        return zone_flags

    def get_blocked_zone_pairs(self):
        """
        Return the set of (from_subnet, to_subnet) subnet-name pairs
        CURRENTLY blocked, exactly as CybORG's own BlockTrafficZone /
        AllowTrafficZone bookkeeping represents them:

            state.blocks[to_subnet] -> list of from_subnet names
            currently blocked from reaching to_subnet.

        Deliberately does NOT normalize case here. BlueFixedActionWrapper
        lowercases `from_subnet` at action-construction time but NOT
        `to_subnet` (see _populate_action_space: `srcname = srcname.lower()`
        is applied only to the source side), and ControlTraffic.py's
        BlockTrafficZone/AllowTrafficZone.execute_control_traffic() key
        state.blocks with self.to_subnet/self.from_subnet exactly as
        constructed. Returning the raw strings here means callers can
        compare directly against action.from_subnet/action.to_subnet
        with no risk of a case-mismatch silently breaking the lookup.

        Like get_host_alert_flags/get_zone_alert_flags, this is NOT
        privileged red-team ground truth -- every agent's own
        observation already carries a `blocked_subnets` flag for its
        own zone (BlueFlatWrapper.observation_change); this reads the
        same underlying fact directly, for every subnet pair, instead
        of re-deriving a single agent's slice of it from a padded
        observation vector.
        """

        state = self.cyborg.environment_controller.state

        blocks = getattr(state, "blocks", {})

        pairs = set()

        for to_subnet, from_subnets in blocks.items():

            for from_subnet in from_subnets:

                pairs.add((from_subnet, to_subnet))

        return pairs

    ############################################################
    # True per-episode host layout (for host_active_mask)
    ############################################################

    def get_host_active_mask(self, agent_name):
        """
        Return `agent_name`'s TRUE per-episode host-validity mask:

            [NUM_HQ_SUBNETS, MAX_HOSTS] bool

        True at (subnet_slot, host_slot) iff a REAL host occupies
        that position THIS EPISODE. This is the mask
        gnn_attention.py's SharedActor (via mappo.py's
        `host_active_mask` plumbing) needs to isolate padded/fake
        host nodes from the GNN -- see gnn_attention.py's module
        docstring, "Host-level padding".

        Why this can't be read off the observation vector
        ----------------------------------------------------
        CC4 randomizes 1-6 servers / 3-10 users per zone at reset
        (fixed for the episode), but BlueFlatWrapper.observation_change
        ANDs "host exists" together with "host currently has an
        alert" into a single bit before the flat vector is ever
        built (`h in state.hosts and 0 < len(...)`) -- so an absent
        host and a present-but-currently-quiet host are bit-for-bit
        identical (both read 0) at any given timestep. Existence must
        therefore be read directly from `state.hosts`, never inferred
        from alert values, or a real, currently-quiet host would be
        wrongly masked out as if it were padding.

        Ordering (must match gnn_attention.py + train.py exactly)
        -------------------------------------------------------------
        - Subnet slot order: `self.env.subnets(agent_name)` -- the
          same sorted per-agent subnet list BlueFlatWrapper.
          observation_change iterates over (`for subnet in
          self.subnets(agent_name)`) and that train.py's
          pad_observation() relies on when it places an agent's real
          subnet block(s) into the first `len(subnets)` of
          NUM_HQ_SUBNETS slots, in that same order. Agents with fewer
          than NUM_HQ_SUBNETS real subnets (every agent except the HQ
          agent) simply leave the remaining slots False here, which
          is exactly the "padding subnet slot" state
          `_build_batch_adjacency` already expects.

        - Host slot order, per subnet: `[h for h in
          self.env.hosts(agent_name) if subnet in h and "router" not
          in h]` -- the identical filter BlueFlatWrapper.
          observation_change uses to build `process_subvector`/
          `connection_subvector` for that same subnet block, over the
          same sorted, fixed-length (always MAX_HOSTS per subnet)
          hostname list `self.env.hosts(agent_name)` returns
          (BlueFixedActionWrapper pre-generates MAX_USER_HOSTS +
          MAX_SERVER_HOSTS formatted hostnames per subnet regardless
          of whether they exist this episode -- see
          _create_hardcoded_metadata). Reusing the exact same list +
          filter here guarantees host_slot `h` in this mask lines up
          with process_subvector[h]/connection_subvector[h] in the
          observation, for every subnet.

        Existence check: `hostname in state.hosts` -- state.hosts
        only contains the hosts CybORG actually generated this
        episode, which is exactly the ground truth this mask needs.
        """

        state = self.cyborg.environment_controller.state

        mask = np.zeros((NUM_HQ_SUBNETS, MAX_HOSTS), dtype=bool)

        subnet_slots = self.env.subnets(agent_name)
        hosts = self.env.hosts(agent_name)

        for slot, subnet in enumerate(subnet_slots[:NUM_HQ_SUBNETS]):

            subnet_hosts = [
                h for h in hosts if subnet in h and "router" not in h
            ]

            for host_idx, hostname in enumerate(subnet_hosts[:MAX_HOSTS]):

                mask[slot, host_idx] = hostname in state.hosts

        return mask

    def get_all_host_active_masks(self):
        """
        `get_host_active_mask` for every agent, stacked in the same
        `sorted(self.possible_agents)` order train.py's `agent_names`
        already uses everywhere else (obs_array, actions_arr, ...).

        Returns
        -------
        np.ndarray, shape [NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS], bool
        """

        agent_names = sorted(self.possible_agents)

        return np.stack(
            [
                self.get_host_active_mask(name)
                for name in agent_names
            ],
            axis=0,
        )

    ############################################################
    # Ground truth for MessageEvaluator / DynamicTrust
    ############################################################

    def _compute_host_snapshot(self):
        """
        Return {hostname: {"compromised", "has_process_event",
        "has_connection_event"}} for every host, from the live CybORG
        true state at the moment this is called.

        compromised:
            An active Red session exists on the host. Verified against
            CybORG source as the one signal here that CANNOT come from
            benign Green activity -- the authoritative check.

        has_process_event / has_connection_event:
            state.hosts[h].events shows a process_creation /
            network_connections entry this window (same fields
            get_host_alert_flags() reads). NOT Red-exclusive --
            GreenLocalWork.py and GreenAccessService.py write into
            these same fields, same as Red's ExploitAction/Portscan --
            so on their own these mean "something happened here", not
            "Red did this". Only used to distinguish SUSPICIOUS_ACTIVITY
            from NONE in get_ground_truth() when `compromised` is
            False; never used to claim COMPROMISE by themselves.
        """

        state = self.cyborg.environment_controller.state

        snapshot = {}

        for hostname, host in state.hosts.items():

            sessions = getattr(host, "sessions", {})

            compromised = False

            for owner, session_list in sessions.items():

                if "red" in str(owner).lower() and session_list:
                    compromised = True
                    break

            events = host.events

            has_process_event = bool(
                events.old_process_creation or events.process_creation
            )

            has_connection_event = bool(
                events.old_network_connections or events.network_connections
            )

            snapshot[hostname] = {
                "compromised": compromised,
                "has_process_event": has_process_event,
                "has_connection_event": has_connection_event,
            }

        return snapshot

    @staticmethod
    def _tier_for_snapshot(host_snapshot):
        """
        Map one host's raw snapshot booleans to the (event_type,
        threat_level, status) tier used by both target_status and
        subnet_status below. Returns None for the NONE tier (host is
        left out of the ground-truth dict entirely, same convention
        both callers already relied on before this was factored out).
        """

        if host_snapshot["compromised"]:

            return {
                "event_type": EventType.COMPROMISE,
                "threat_level": ThreatLevel.HIGH,
                "status": HostStatus.COMPROMISED,
            }

        if host_snapshot["has_process_event"] or host_snapshot["has_connection_event"]:

            # Suspicious tier: an event exists but no Red session is
            # confirmed. Not Red-exclusive (see _compute_host_snapshot()),
            # so reported as SUSPICIOUS_ACTIVITY, never COMPROMISE.
            return {
                "event_type": EventType.SUSPICIOUS_ACTIVITY,
                "threat_level": ThreatLevel.MEDIUM,
                "status": HostStatus.SUSPICIOUS,
            }

        return None

    def get_ground_truth(self, sender_id):
        """
        Pure, message-independent ground truth for the sender's zone.

        This reports what is ACTUALLY true about every relevant host (AND,
        as of this revision, every relevant subnet) in the sender's
        security zone, RIGHT NOW. It does NOT receive, inspect, or resolve
        any agent message, target, or receiver -- it answers only "what is
        true in the sender's zone?". Grading a specific claim against this
        truth is entirely MessageEvaluator's job (see
        communication/evaluator.py).

        This is the fix for the old wrong-target rescue: because we never
        see the claimed target here, we can never silently re-point a
        claim at a different, genuinely-compromised host or subnet. We
        simply report the truth for the whole zone; a claim about an
        uninvolved host/subnet will find it absent from
        target_status/subnet_status and be scored wrong by the evaluator.

        State source
        ------------
        Uses the snapshot captured BEFORE this step's env.step() ran
        (see step() / _compute_host_snapshot()), so a message generated
        from the start-of-step observation is graded against the state it
        was actually generated from -- not the mutated post-step state.

        Per-host tiers (target_status)
        -------------------------------
            active Red session
                -> EventType.COMPROMISE / ThreatLevel.HIGH
                   / HostStatus.COMPROMISED

            process/connection event but no confirmed Red session
                -> EventType.SUSPICIOUS_ACTIVITY / ThreatLevel.MEDIUM
                   / HostStatus.SUSPICIOUS

            neither
                -> absent from target_status (i.e. NONE / normal)

        Per-subnet tiers (subnet_status)
        -----------------------------------
        Each subnet is reported at whichever tier its most severe
        relevant host (by the same three tiers above) currently sits at.
        A subnet with no flagged hosts is absent from subnet_status,
        exactly mirroring target_status's own convention. This is a
        genuinely SEPARATE ground-truth dict from target_status -- a
        SUBNET-typed message's target_id is a subnet index (see
        communication/schema.py), never a host index, and must be graded
        against subnet_status, never target_status.

        Returns
        -------
        None
            If the sender id / agent cannot be resolved. (No trust update
            is performed in that case -- see train.py.)

        Otherwise a dict:

            {
                "target_status": {
                    target_id (int): {
                        "event_type":   EventType,
                        "threat_level": ThreatLevel,
                        "status":       HostStatus,
                    },
                    ...   # one entry per RELEVANT host in the zone
                },
                "subnet_status": {
                    subnet_id (int): {
                        "event_type":   EventType,
                        "threat_level": ThreatLevel,
                        "status":       HostStatus,
                    },
                    ...   # one entry per RELEVANT subnet in the zone
                },
                "zone_quiet": bool,   # True iff no relevant host exists
            }

        target_id in target_status uses the deterministic sorted host
        list (sorted(state.hosts.keys())) -- matching get_num_targets().
        subnet_id in subnet_status uses the deterministic sorted subnet
        list (sorted(state.subnet_name_to_cidr.keys())) -- matching
        get_num_subnet_targets(). These MUST remain consistent with the
        target-ID mappings used by the communication encoder/decoder.
        """

        # ------------------------------------------------------------
        # Resolve sender (message-independent)
        # ------------------------------------------------------------

        agent_names = sorted(self.possible_agents)

        if not (0 <= sender_id < len(agent_names)):
            return None

        sender_name = agent_names[sender_id]

        state = self.cyborg.environment_controller.state

        agent_meta = state.scenario.agents.get(sender_name)

        if agent_meta is None:
            return None

        # ------------------------------------------------------------
        # Sender's assigned subnets
        # ------------------------------------------------------------

        sender_subnets = {
            str(subnet).lower()
            for subnet in agent_meta.allowed_subnets
        }

        # ------------------------------------------------------------
        # Deterministic host <-> target_id mapping.
        #
        # IMPORTANT: must match the mapping used by the communication
        # module and get_num_targets().
        # ------------------------------------------------------------

        hostnames = sorted(state.hosts.keys())

        host_to_id = {
            hostname: index
            for index, hostname in enumerate(hostnames)
        }

        # ------------------------------------------------------------
        # Deterministic subnet <-> subnet_id mapping.
        #
        # IMPORTANT: must match get_num_subnet_targets() and
        # gnn_attention.py's SUBNET_NAME_ORDER.
        # ------------------------------------------------------------

        subnet_names = sorted(state.subnet_name_to_cidr.keys())

        subnet_to_id = {
            name: index
            for index, name in enumerate(subnet_names)
        }

        # ------------------------------------------------------------
        # Every host in the sender's zone, graded from the PRE-STEP
        # snapshot (not a fresh live query). Hosts with no relevant
        # activity are simply left out of target_status. Subnet-level
        # status is aggregated from the SAME per-host tiers as we go,
        # taking the most severe tier seen per subnet.
        # ------------------------------------------------------------

        target_status = {}
        subnet_status = {}

        for hostname in sorted(self._pre_step_host_snapshot):

            subnet = str(
                state.hostname_subnet_map.get(hostname, "")
            ).lower()

            if subnet not in sender_subnets:
                continue

            snapshot = self._pre_step_host_snapshot[hostname]

            tier = self._tier_for_snapshot(snapshot)

            if tier is None:
                # NONE tier -- host absent from target_status, and
                # contributes nothing to subnet_status either.
                continue

            target_id = host_to_id.get(hostname)

            if target_id is not None:
                target_status[target_id] = tier
            # else: in the snapshot but not in the current host list,
            # so it cannot be addressed by any HOST message target_id
            # -- skip target_status, but it can still count toward its
            # subnet's aggregated tier below.

            subnet_id = subnet_to_id.get(subnet)

            if subnet_id is not None:

                existing = subnet_status.get(subnet_id)

                if (
                    existing is None
                    or int(tier["threat_level"]) > int(existing["threat_level"])
                ):
                    subnet_status[subnet_id] = tier

        return {
            "target_status": target_status,
            "subnet_status": subnet_status,
            "zone_quiet": len(target_status) == 0,
        }


from ray.tune.registry import register_env


def env_creator(env_config=None):
    """
    RLlib environment creator.
    """
    return CC4Env()


def register_cc4_env():
    """
    Register the environment with RLlib.
    Safe to call multiple times.
    """
    register_env("CC4", lambda config: env_creator(config))