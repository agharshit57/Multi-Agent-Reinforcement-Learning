"""Fixed mapping between real network assets and the CC4 structure.

The trained policy has a FIXED geometry: every agent owns fixed zones,
every zone has fixed host slots (10 users + 6 servers). Deployment must
therefore assign each real asset to exactly one CC4 slot ONCE, at
configuration time -- observation/action sizes never change at runtime.

Slot order inside a zone replicates
``BlueFixedActionWrapper``/``BlueFlatWrapper`` exactly:
``sorted(hostnames minus routers)`` per zone, i.e. server_host_0..5
then user_host_0..9 (single-digit lexicographic order).

File format (JSON)::

    {"agents": {"blue_agent_0": {"hosts": [
        {"cc4": "restricted_zone_a_subnet_server_host_0",
         "ip": "10.1.0.11", "hostname": "srv-a-01", "role": "server"},
        ...]}}}

Only ``cc4`` is structural; ``ip``/``hostname``/``role`` describe the
real asset bound to that slot. Real hosts with no slot are reported as
``unmapped`` (observed for inventory, never fed to the model).
"""

import ipaddress
import json

from .config import (AGENT_NAMES, AGENT_ZONES, ALLOWED_ROLES, MAX_HOSTS,
                     MAX_SERVER_HOSTS, MAX_USER_HOSTS, SUBNETS)


class AssetMapError(ValueError):
    """Raised when an asset map is inconsistent with the CC4 geometry."""


def check_zone_names_unambiguous(zones):
    """Fail loudly if zone attribution could ever be ambiguous.

    Strict matching attributes ``cc4`` to ``zone`` iff
    ``cc4.startswith(zone + "_")``. That rule is ambiguous exactly when
    one zone's ``zone + "_"`` prefix is itself a prefix of another
    zone's ``zone + "_"`` (e.g. "x_subnet" vs "x_subnet_y": a host
    "x_subnet_y_host_0" startswith BOTH "x_subnet_" and
    "x_subnet_y_"). Such configurations are rejected here instead of
    silently attributing to whichever zone sorts first.
    """
    zones = list(zones)
    for anchor in zones:
        prefix = anchor + "_"
        for other in zones:
            if other != anchor and (other + "_").startswith(prefix):
                raise AssetMapError(
                    f"ambiguous zone names: {other!r} collides with "
                    f"{anchor!r} under prefix attribution "
                    f"({prefix!r} is a prefix of {other + '_'!r})")
    return True


def match_zone(cc4, zones=SUBNETS):
    """Attribute one CC4 hostname to exactly one zone (strict).

    Returns the single zone with ``cc4.startswith(zone + "_")``.
    Raises AssetMapError on zero matches (unknown zone) or more than
    one match (ambiguous configuration -- never silently pick one).
    This is the ONE zone-matching implementation; every module
    (asset map, action table/mask, observation, pipeline, GUI) routes
    through it so attribution cannot drift between layers.
    """
    if not isinstance(cc4, str) or not cc4:
        raise AssetMapError(f"cannot determine zone of {cc4!r}")
    matches = [zone for zone in zones if cc4.startswith(zone + "_")]
    if len(matches) != 1:
        raise AssetMapError(
            f"cannot determine zone of {cc4!r}: "
            f"{len(matches)} zone matches "
            f"({', '.join(matches) or 'none'})")
    return matches[0]


def cc4_hostname(zone, kind, index):
    """Canonical CC4 hostname, mirroring SUBNET_*_FORMAT in CybORG."""
    if kind == "router":
        return f"{zone}_router"
    if kind == "user":
        return f"{zone}_user_host_{index}"
    if kind == "server":
        return f"{zone}_server_host_{index}"
    raise AssetMapError(f"unknown host kind: {kind!r}")


def zone_slot_hostnames(zone):
    """All 16 observable slot hostnames of a zone, in model order."""
    names = ([cc4_hostname(zone, "server", i) for i in range(MAX_SERVER_HOSTS)]
             + [cc4_hostname(zone, "user", i) for i in range(MAX_USER_HOSTS)])
    ordered = sorted(names)
    assert len(ordered) == MAX_HOSTS
    return ordered


class AssetMap:
    """Fixed real-asset <-> CC4-slot binding."""

    def __init__(self, agents, unmapped=None):
        # agents: {agent_name: {"hosts": {cc4: {"ip","hostname","role"}}}}
        self.agents = agents
        self.unmapped = list(unmapped or [])
        # Precompute slot indices: {(agent, cc4): (slot, host_idx)}
        self._slot_index = {}
        self._stable_index = {}
        for agent, zones in AGENT_ZONES.items():
            name = f"blue_agent_{agent}"
            for slot, zone in enumerate(zones):
                for host_idx, cc4 in enumerate(zone_slot_hostnames(zone)):
                    self._slot_index[(name, cc4)] = (slot, host_idx)

    # ------------------------------------------------------------ loading --
    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "agents" not in data:
            raise AssetMapError("asset map needs a top-level 'agents' object")
        if not isinstance(data["agents"], dict):
            raise AssetMapError("'agents' must be an object keyed by agent")
        for key in data["agents"]:
            if key not in AGENT_NAMES:
                raise AssetMapError(
                    f"unknown agent {key!r}; valid agents: "
                    f"{', '.join(AGENT_NAMES)}")
        agents = {}
        seen_cc4 = {}
        seen_ip = {}
        seen_hostname = {}
        for agent in AGENT_NAMES:
            entry = data["agents"].get(agent, {})
            if not isinstance(entry, dict):
                raise AssetMapError(f"{agent}: entry must be an object")
            raw_hosts = entry.get("hosts", [])
            if not isinstance(raw_hosts, list):
                raise AssetMapError(f"{agent}: 'hosts' must be a list")
            hosts = {}
            for pos, item in enumerate(raw_hosts):
                if not isinstance(item, dict):
                    raise AssetMapError(
                        f"{agent}: host #{pos} must be an object")
                cc4 = item.get("cc4")
                if not cc4 or not isinstance(cc4, str):
                    raise AssetMapError(
                        f"{agent}: host #{pos} without a 'cc4' slot name")
                if cc4 in seen_cc4:
                    raise AssetMapError(
                        f"{cc4} bound twice ({seen_cc4[cc4]} and {agent})")
                seen_cc4[cc4] = agent
                ip = item.get("ip", "")
                hostname = item.get("hostname", cc4)
                role = item.get("role", "")
                _validate_ip(agent, cc4, ip)
                _validate_hostname(agent, cc4, hostname)
                _validate_role(agent, cc4, role)
                if ip:
                    if ip in seen_ip:
                        raise AssetMapError(
                            f"duplicate IP {ip!r} on {cc4} ({agent}) and "
                            f"{seen_ip[ip]}: one physical asset cannot "
                            f"occupy two CC4 slots")
                    seen_ip[ip] = f"{cc4} ({agent})"
                if hostname and hostname != cc4:
                    if hostname in seen_hostname:
                        raise AssetMapError(
                            f"duplicate hostname {hostname!r} on {cc4} "
                            f"({agent}) and {seen_hostname[hostname]}: "
                            f"physical-asset mapping is ambiguous")
                    seen_hostname[hostname] = f"{cc4} ({agent})"
                hosts[cc4] = {"ip": ip, "hostname": hostname, "role": role}
            agents[agent] = {"hosts": hosts}
        # Structural validation: every slot hostname must be bound, and
        # every bound cc4 must be a REAL slot of that agent's zones
        # (this rejects routers, the internet host, typos, and
        # out-of-zone names -- none of them are observable slots and
        # the model could never address them).
        used_zones = set()
        for agent, zones in AGENT_ZONES.items():
            name = f"blue_agent_{agent}"
            bound = agents[name]["hosts"]
            for zone in zones:
                used_zones.add(zone)
                for cc4 in zone_slot_hostnames(zone):
                    if cc4 not in bound:
                        raise AssetMapError(
                            f"{name}: missing binding for slot {cc4}")
            valid_slots = set()
            for zone in zones:
                valid_slots.update(zone_slot_hostnames(zone))
            for cc4 in bound:
                if cc4 not in valid_slots:
                    raise AssetMapError(
                        f"{name}: {cc4} is not an observable slot of "
                        f"zones {sorted(zones)} (routers and foreign "
                        f"hosts cannot be bound)")
        check_zone_names_unambiguous(used_zones)
        unmapped = data.get("unmapped", [])
        if not isinstance(unmapped, list) or any(
                not isinstance(u, dict) for u in unmapped):
            raise AssetMapError("'unmapped' must be a list of objects")
        return cls(agents, unmapped)

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self):
        return {"agents": {
            agent: {"hosts": [
                {"cc4": cc4, **info}
                for cc4, info in sorted(entry["hosts"].items())]}
            for agent, entry in self.agents.items()},
            "unmapped": list(self.unmapped)}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # ------------------------------------------------------------ lookup --
    def slot_of(self, agent_name, cc4):
        """(subnet_slot, host_slot) for a bound CC4 hostname."""
        return self._slot_index[(agent_name, cc4)]

    def is_bound(self, agent_name, cc4):
        return (agent_name, cc4) in self._slot_index

    def real_asset(self, agent_name, cc4):
        """Real-asset record bound to a slot (KeyError if unbound)."""
        return self.agents[agent_name]["hosts"][cc4]

    def bound_hosts(self, agent_name):
        return sorted(self.agents[agent_name]["hosts"])

    @staticmethod
    def stable_order():
        """All slot hostnames in global sorted order (matches training)."""
        names = []
        for zones in AGENT_ZONES.values():
            for zone in zones:
                names.extend(zone_slot_hostnames(zone))
        return sorted(set(names))


def _zone_of(cc4):
    """Back-compat alias for :func:`match_zone` (default zone set)."""
    return match_zone(cc4, SUBNETS)


def _validate_ip(agent, cc4, ip):
    if not ip:
        return  # unbound address slot; keyed by hostname instead
    if not isinstance(ip, str):
        raise AssetMapError(
            f"{agent}: {cc4} has non-string ip {ip!r}")
    try:
        ipaddress.ip_address(ip.strip())
    except ValueError:
        raise AssetMapError(
            f"{agent}: {cc4} has invalid IP address {ip!r}")


def _validate_hostname(agent, cc4, hostname):
    if not isinstance(hostname, str) or not hostname.strip():
        raise AssetMapError(
            f"{agent}: {cc4} has an empty/invalid hostname")


def _validate_role(agent, cc4, role):
    if not isinstance(role, str) or role not in ALLOWED_ROLES:
        raise AssetMapError(
            f"{agent}: {cc4} has invalid role {role!r}; allowed: "
            f"{sorted(ALLOWED_ROLES)}")


# The shipped CC4 zone names must themselves be unambiguous under the
# strict rule above; fail at import time, not mid-incident.
check_zone_names_unambiguous(SUBNETS)


def generate_default_map():
    """Build a placeholder map (synthetic IPs) so the app runs out of box."""
    agents = {}
    host_counter = {}
    for agent, zones in AGENT_ZONES.items():
        name = f"blue_agent_{agent}"
        hosts = []
        for zone in zones:
            octet = SUBNETS.index(zone)
            for i, cc4 in enumerate(zone_slot_hostnames(zone)):
                n = host_counter.get(zone, 11)
                host_counter[zone] = n + 1
                hosts.append({"cc4": cc4,
                              "ip": f"10.0.{octet}.{n}",
                              "hostname": cc4.replace("_subnet_", "-").replace(
                                  "_", "-"),
                              "role": "server" if "server" in cc4 else "user"})
        agents[name] = {"hosts": hosts}
    return AssetMap.from_dict({"agents": agents, "unmapped": []})
