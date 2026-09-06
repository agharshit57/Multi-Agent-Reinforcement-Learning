"""Scripted demo intrusion timeline (no network needed).

Five cycles over the default asset map:
  0. quiet baseline (known processes/ports only)
  1. port scan against Restricted A (suspicious connections)
  2. confirmed compromise on one Restricted A server
     (intrusion_confirmed -> compromised signal, like a Red session;
     compromise is sticky from here until recovery)
  3. lateral-movement indicators toward Operational A
     (compromise persists: no recovery signal yet)
  4. explicit recovery (recovery_confirmed on the compromised host)
     plus quiet baseline elsewhere

Baselines match ``demo_baselines()`` so cycle 0 is fully quiet.
"""

from .asset_map import generate_default_map
from .telemetry import HostTelemetry, MockCollector, SecurityEvent, TelemetryBatch

BASE_PROCESSES = ["sshd", "apache2", "mysqld", "cron", "systemd"]
BASE_PORTS = {22, 80, 443}


def demo_baselines():
    # "10.0.0.1" is the demo network's known controller: with it listed,
    # cycle 0 (baseline traffic to the controller only) is fully quiet,
    # while any outside peer still raises a connection event.
    return {"processes": set(BASE_PROCESSES),
            "ports": set(BASE_PORTS), "peers": {"10.0.0.1"}}


def _host(key, processes=None, connections=None, events=None):
    return HostTelemetry(key=key, processes=list(processes or []),
                         connections=list(connections or []),
                         sessions=[], up=True,
                         events=list(events or []))


def demo_batches(asset_map=None):
    """Build the 5-cycle MockCollector batch list."""
    asset_map = asset_map or generate_default_map()
    agent0 = asset_map.agents["blue_agent_0"]["hosts"]
    by_cc4 = {cc4: info for cc4, info in agent0.items()}

    def key(cc4):
        return by_cc4[cc4]["ip"]

    srv0 = key("restricted_zone_a_subnet_server_host_0")
    srv1 = key("restricted_zone_a_subnet_server_host_1")
    user0 = key("restricted_zone_a_subnet_user_host_0")
    agent1 = asset_map.agents["blue_agent_1"]["hosts"]
    op_srv0 = list(agent1.values())[0]["ip"]

    quiet = {info["ip"]: _host(info["ip"], processes=["sshd", "cron"],
                               connections=["tcp:22->10.0.0.1"])
             for a in asset_map.agents.values() for info in a["hosts"].values()}

    def clone(**over):
        batch = {k: _host(k, processes=list(v.processes),
                          connections=list(v.connections))
                 for k, v in quiet.items()}
        batch.update(over)
        return batch

    batches = [
        TelemetryBatch(timestamp=1.0, hosts=clone(), notes="quiet baseline"),
        TelemetryBatch(timestamp=2.0, hosts=clone(**{
            srv0: _host(srv0, processes=["sshd"],
                       connections=["tcp:22->10.0.0.1",
                                    "tcp:443->198.51.100.7",
                                    "tcp:3390->198.51.100.7"],
                       events=[SecurityEvent(kind="port_scan",
                                             severity="medium",
                                             details="scan from 198.51.100.7",
                                             timestamp=2.0)])}),
            notes="port scan Restricted A"),
        TelemetryBatch(timestamp=3.0, hosts=clone(**{
            srv0: _host(srv0, processes=["sshd", "nc_backdoor"],
                       connections=["tcp:4444->198.51.100.7"],
                       events=[SecurityEvent(kind="intrusion_confirmed",
                                             severity="critical",
                                             details="red session on host",
                                             timestamp=3.0)])}),
            notes="confirmed compromise server_host_0"),
        TelemetryBatch(timestamp=4.0, hosts=clone(**{
            srv0: _host(srv0, processes=["sshd", "nc_backdoor"],
                       connections=["tcp:4444->198.51.100.7"]),
            srv1: _host(srv1, processes=["sshd"],
                       connections=["tcp:22->10.0.0.1",
                                    "tcp:445->" + op_srv0],
                       events=[SecurityEvent(kind="suspicious_connection",
                                             severity="medium",
                                             details="lateral movement",
                                             timestamp=4.0)])}),
            notes="lateral movement indicators"),
        TelemetryBatch(timestamp=5.0, hosts=clone(**{
            srv0: _host(srv0, processes=["sshd", "cron"],
                       connections=["tcp:22->10.0.0.1"],
                       events=[SecurityEvent(kind="recovery_confirmed",
                                             severity="low",
                                             details="analyst reimaged host",
                                             timestamp=5.0)])}),
            notes="recovery quiet"),
    ]
    return batches


def demo_collector(asset_map=None):
    return MockCollector(demo_batches(asset_map))
