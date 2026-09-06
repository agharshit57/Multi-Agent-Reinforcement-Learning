"""Shared constants for the deployment layer.

Every value that must match the trained model is mirrored here with its
provenance. Sources:
  - dims/layouts: ``Marl/mappo/config.py`` (OBS_DIM=210, ACTION_DIM=242,
    SMALL_OBS_DIM=92, SMALL_ACTION_DIM=82)
  - subnet order: ``CybORG`` 9-subnet alphabetical order, identical to
    ``Marl/mappo/gnn_attention.py::SUBNET_NAME_ORDER``
  - observation block layout: ``BlueFlatWrapper.observation_change``
    (mission | subnet blocks | native messages) and
    ``Marl/mappo/train.py::pad_observation``
  - command order: ``BlueFixedActionWrapper._populate_action_space``,
    which sorts the 8 blue-action classes by ``str`` (full
    "<class 'module.Name'>" string). Verified module paths:
    Sleep               <- CybORG.Simulator.Actions.Action
    Analyse/Monitor/Remove/Restore
                        <- CybORG.Simulator.Actions.AbstractActions.*
    AllowTrafficZone/BlockTrafficZone
                        <- ...ConcreteActions.ControlTraffic
    DeployDecoy         <- ...ConcreteActions.DecoyActions.DeployDecoy
    Sorted result: Analyse, Monitor, Remove, Restore, Sleep,
    AllowTrafficZone, BlockTrafficZone, DeployDecoy.

This module imports NOTHING third-party so the telemetry/observation
layers stay runnable without the training venv.
"""

# ---------------------------------------------------------------- subnet --
SUBNETS = (
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
)
NUM_SUBNETS = len(SUBNETS)

# ------------------------------------------------- agent -> zones (fixed) --
# Mirrors EnterpriseScenarioGenerator._generate_blue_agents:
# blue_agent_0..3 defend one zone each, blue_agent_4 (HQ) defends three.
AGENT_ZONES = {
    0: ("restricted_zone_a_subnet",),
    1: ("operational_zone_a_subnet",),
    2: ("restricted_zone_b_subnet",),
    3: ("operational_zone_b_subnet",),
    4: ("admin_network_subnet",
         "office_network_subnet",
         "public_access_zone_subnet"),
}
NUM_AGENTS = 5
AGENT_NAMES = tuple(f"blue_agent_{i}" for i in range(NUM_AGENTS))

# ------------------------------------------------------- observation dims --
MISSION_DIM = 1
MAX_HOSTS = 16            # 10 users + 6 servers per zone (CC4 max)
MAX_USER_HOSTS = 10
MAX_SERVER_HOSTS = 6
SUBNET_BLOCK_DIM = 3 * NUM_SUBNETS + 2 * MAX_HOSTS  # 27 + 32 = 59
NUM_MESSAGES = 4
MESSAGE_LENGTH = 8
MESSAGE_DIM = NUM_MESSAGES * MESSAGE_LENGTH        # 32
NUM_HQ_SUBNETS = 3

SMALL_OBS_DIM = 92    # 1 + 1*59 + 32  (blue_agent_0..3, native)
LARGE_OBS_DIM = 210   # 1 + 3*59 + 32  (blue_agent_4, native + padded)
OBS_DIM = LARGE_OBS_DIM  # shared-policy width train.py pads everything to

# ------------------------------------------------------------ action dims --
SMALL_ACTION_DIM = 82
LARGE_ACTION_DIM = 242
ACTION_DIM = LARGE_ACTION_DIM  # shared-policy width

# Verified ``sorted(commands, key=str)`` order (see module docstring).
COMMAND_ORDER = (
    "Analyse",
    "Monitor",
    "Remove",
    "Restore",
    "Sleep",
    "AllowTrafficZone",
    "BlockTrafficZone",
    "DeployDecoy",
)

# ------------------------------------------------------- communication -----
COMMUNICATION_DIM = 128   # mirrors Marl/mappo/gnn_attention.py
COMMUNICATION_LATENT_DIM = 256

# Structured message fields (mirrors communication/schema.py MESSAGE_FIELDS
# in mappo/buffer field order: event, target_type, threat, status,
# priority, confidence, target_id).
MESSAGE_FIELD_NAMES = (
    "event_type", "target_type", "threat_level", "status",
    "priority", "confidence", "target_id",
)
EVENT_TYPES = ("NONE", "DISCOVERY", "SCAN", "SUSPICIOUS_ACTIVITY",
               "COMPROMISE", "LATERAL_MOVEMENT", "PRIVILEGE_ESCALATION",
               "RECOVERY")
TARGET_TYPES = ("NONE", "HOST", "SUBNET")
THREAT_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
HOST_STATUS = ("UNKNOWN", "NORMAL", "SUSPICIOUS", "COMPROMISED", "CONTAINED")
PRIORITIES = ("LOW", "MEDIUM", "HIGH", "URGENT")

# ------------------------------------------------------------------ misc --
MISSION_PHASES = ("Preplanning", "MissionA", "MissionB")
DEFAULT_MISSION_PHASE = 0

# Execution modes. No fully-automatic enforcement of destructive actions.
# "live" executes approved real operations through a configured
# EnforcementBackend and is explicitly opt-in only (CLI --enable-live):
# without it, selecting live is rejected before anything runs.
MODES = ("shadow", "mock", "supervised", "live")
DEFAULT_MODE = "shadow"

# Per-operation enforcement timeouts (seconds) for live backends.
# Mock/shadow ignore these; real backends MUST enforce them.
OP_TIMEOUT_S = {
    "noop": 5,
    "collect_status": 30,
    "collect_forensics": 60,
    "deploy_honeypot": 120,
    "terminate_suspicious": 60,
    "reimage_host": 600,
    "allow_zone_traffic": 60,
    "isolate_zone_traffic": 60,
}
DEFAULT_OP_TIMEOUT_S = 60

# Permitted asset roles in asset maps (empty string = unspecified).
# Deliberately broad: unknown-but-valid site roles must not be rejected;
# truly invalid values (non-strings, blanks) are still rejected.
ALLOWED_ROLES = frozenset({
    "", "server", "user", "workstation", "router", "network", "infra",
    "controller", "honeypot", "iot", "printer",
})

# Commands that may never auto-execute outside mock/shadow; in supervised
# mode they wait in the approval queue.
DESTRUCTIVE_COMMANDS = frozenset({"Restore", "Remove", "BlockTrafficZone"})
INVESTIGATIVE_COMMANDS = frozenset({"Analyse", "DeployDecoy"})
SAFE_COMMANDS = frozenset({"Sleep", "Monitor", "AllowTrafficZone"})

# Placeholder CIDRs for label readability (structural only, documented in
# action_table.py). Index = position in SUBNETS.
PLACEHOLDER_CIDRS = tuple(f"10.0.{i}.0/24" for i in range(NUM_SUBNETS))


def _stable_host_list():
    """Every possible CC4 hostname, sorted (byte-identical replica of
    ``Marl/mappo/env.py::STABLE_HOST_LIST``: one internet host plus, for
    each of the other 8 subnets, one router + 10 users + 6 servers)."""
    names = ["root_internet_host_0"]
    for zone in SUBNETS:
        if zone == "internet_subnet":
            continue
        names.append(f"{zone}_router")
        for i in range(MAX_USER_HOSTS):
            names.append(f"{zone}_user_host_{i}")
        for i in range(MAX_SERVER_HOSTS):
            names.append(f"{zone}_server_host_{i}")
    return tuple(sorted(names))


STABLE_HOST_LIST = _stable_host_list()
NUM_HOST_TARGETS = len(STABLE_HOST_LIST)  # 137
STABLE_HOST_INDEX = {name: i for i, name in enumerate(STABLE_HOST_LIST)}

assert NUM_HOST_TARGETS == 137, NUM_HOST_TARGETS
assert list(STABLE_HOST_LIST) == sorted(STABLE_HOST_LIST)
assert "root_internet_host_0" in STABLE_HOST_INDEX
assert "restricted_zone_a_subnet_server_host_0" in STABLE_HOST_INDEX

DEFAULT_CHECKPOINT = "checkpoints/fixedMaybe/mappo_final.pt"
