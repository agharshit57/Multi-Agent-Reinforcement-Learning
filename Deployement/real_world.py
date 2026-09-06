"""Real-world source of truth: physical assets, segments, discovery.

THREE-WAY SEPARATION -- read before touching deployment code::

    REAL ASSET / REAL TOPOLOGY  (this module) ..... what physically exists.
    POLICY REPRESENTATION       (policy_adapter.py) . CC4 slots / obs /
                                                    masks for the frozen
                                                    model. Technical view
                                                    only -- never presented
                                                    as infrastructure.
    REAL-WORLD ACTION           (real_actions.py) ... validated operations
                                                    on REAL assets; policy
                                                    decisions are input,
                                                    never authority.

Nothing in this module knows CC4 zone names, slot order, observation
dims, or action ids. Telemetry keys here are stable ``asset_id`` values
(``"host:<lower-hostname>"``), never CC4 hostnames or placeholder IPs.

Reality audit for this module:

* ``LocalMachineCollector`` performs GENUINELY REAL sensing of the
  machine it runs on (stdlib only: hostname, interface addresses, MAC,
  OS release). Process/connection inventory uses ``psutil`` ONLY when
  installed; otherwise it degrades loudly via ``partial_errors`` --
  never by fabricating activity.
* Anything this module cannot observe is reported as unknown/partial,
  never synthesized. In particular it NEVER emits intrusion evidence:
  compromise verdicts come from real detectors downstream, not from
  discovery.
"""

import platform
import socket
import time
import uuid
from dataclasses import dataclass, field

from .telemetry import (HostTelemetry, SecurityEvent, TelemetryBatch,
                        TelemetryCollector)


# ------------------------------------------------------------------ model --

@dataclass
class RealInterface:
    """One physical/virtual NIC address on a real asset."""
    address: str
    family: str = "ipv4"          # "ipv4" | "ipv6"
    scope: str = "unknown"        # loopback|link-local|private|public|other


@dataclass
class RealLink:
    """One observed/declared relationship between real assets.

    ``via`` is always labeled: "arp" (L2-seen), "subnet" (shared
    observed subnet), "declared" (shared operator-declared segment).
    """
    asset_id: str
    peer: str                 # asset_id (or "ip:<addr>" stub)
    via: str = "arp"          # arp|subnet|declared


@dataclass
class RealAsset:
    """One physical machine. Identity is ``asset_id``, stable across
    reboots for the same hostname (``"host:<lower-hostname>"``).

    ``coverage`` is honest sensing state (never a health claim):

    * "covered-local" ..... this machine, genuinely sensed.
    * "covered-agent" ..... remote agent reports (adapter-dependent).
    * "uncovered" ......... declared but no sensor (reads STALE).
    * "observed-arp-only" . wire proof only (ARP stub, nothing more).
    * "unknown" ........... legacy/unspecified.
    """
    asset_id: str
    hostname: str
    ips: tuple = ()               # non-loopback addresses first
    macs: tuple = ()              # observed MACs (empty when unavailable)
    os_name: str = ""
    os_version: str = ""
    role: str = ""                # free text; "" = unspecified
    segment_id: str = "host-only"
    services: tuple = ()          # supervised service names (inventory)
    last_seen: float = 0.0
    coverage: str = "unknown"
    coverage_detail: str = ""


@dataclass
class RealSegment:
    """One real network segment (what the site's switching/L3 says, not
    a CC4 logical zone). A single unmanaged PC is ``host-only``."""
    segment_id: str
    name: str
    kind: str = "host-only"       # host-only|lan|unknown
    cidrs: tuple = ()


@dataclass
class RealTopology:
    """The deployment's physical inventory at a point in time."""
    assets: list = field(default_factory=list)     # RealAsset
    segments: list = field(default_factory=list)   # RealSegment
    links: list = field(default_factory=list)      # RealLink (labeled)
    source: str = ""              # collector that produced it
    discovered_at: float = 0.0

    def by_id(self, asset_id):
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    def primary_ip(self, asset):
        """Best address for display/connection: first non-loopback."""
        for ip in asset.ips or ():
            if not _is_loopback(ip):
                return ip
        return asset.ips[0] if asset.ips else ""


# ---------------------------------------------------------------- helpers --

def asset_id_for_hostname(hostname):
    """Deterministic stable id for a hostname (never a CC4 name)."""
    return "host:" + str(hostname or "unknown").strip().lower()


def _is_loopback(address):
    try:
        import ipaddress
        return ipaddress.ip_address(address.split("%")[0]).is_loopback
    except Exception:
        text = str(address).lower()
        return text == "localhost" or text.startswith("127.")


def _scope_of(address):
    try:
        import ipaddress
        parsed = ipaddress.ip_address(address.split("%")[0])
        if parsed.is_loopback:
            return "loopback"
        if parsed.is_link_local:
            return "link-local"
        if parsed.is_private:
            return "private"
        if parsed.is_global:
            return "public"
        return "other"
    except Exception:
        return "unknown"


def _local_addresses():
    """(ips, interfaces) via stdlib only. Never raises: worst case []."""
    ips, interfaces = [], []
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = ""
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(
                hostname or "localhost", None, proto=socket.IPPROTO_TCP):
            address = sockaddr[0]
            if not address or address in ips:
                continue
            ips.append(address)
            kind = "ipv6" if family == socket.AF_INET6 else "ipv4"
            interfaces.append(RealInterface(
                address=address, family=kind, scope=_scope_of(address)))
    except Exception:
        pass
    # Non-loopback first so primary_ip() is meaningful; stable order.
    ips.sort(key=lambda ip: (_is_loopback(ip), ip))
    interfaces.sort(key=lambda i: (_is_loopback(i.address), i.address))
    return ips, interfaces


def _local_macs():
    """Observed MACs via stdlib. [] when unavailable (never fabricated:
    a locally-administered multicast bit means uuid offers no MAC)."""
    try:
        node = uuid.getnode()
    except Exception:
        return []
    if (node >> 40) & 0x01:  # multicast bit set -> random, not a MAC
        return []
    mac = ":".join(f"{(node >> shift) & 0xFF:02x}"
                   for shift in (40, 32, 24, 16, 8, 0))
    return [mac]


def _psutil_inventory():
    """(processes, connections, note). psutil is OPTIONAL: without it
    the collector reports unknown inventory loudly, never fake data."""
    try:
        import psutil  # type: ignore
    except Exception:
        return [], [], "process/connection inventory unavailable (psutil not installed)"  # noqa: E501
    try:
        processes = sorted({p.info["name"] for p in
                            psutil.process_iter(["name"])
                            if p.info.get("name")})
    except Exception:
        processes, note_p = [], "process enumeration failed"
    else:
        note_p = ""
    try:
        connections = []
        for conn in psutil.net_connections(kind="tcp"):
            try:
                port = conn.laddr.port if conn.laddr else None
                peer = conn.raddr.ip if conn.raddr else ""
            except Exception:
                continue
            if port is None:
                continue
            connections.append(f"tcp:{port}->{peer}" if peer
                               else f"tcp:{port}")
        connections = sorted(set(connections))
    except Exception:
        connections, note_c = [], "connection enumeration failed"
    else:
        note_c = ""
    note = "; ".join(n for n in (note_p, note_c) if n)
    return processes, connections, note


# -------------------------------------------------------------- collector --

class LocalMachineCollector(TelemetryCollector):
    """Genuinely-real sensing of THIS machine (the deployment host).

    Each ``next_batch()`` re-reads discovery (cheap stdlib calls) and
    returns one full-snapshot ``TelemetryBatch`` keyed by ``asset_id``.
    ``exhausted`` is always False: a live sensor has no end-of-stream.

    Honesty rules: no intrusion evidence is ever synthesized (events
    only from real detectors wired by the site); missing inventory is
    a ``partial_errors`` entry, never silence and never fiction.
    """

    name = "local-machine"

    def __init__(self, role="", clock=None, builder=None):
        self.role = role
        self._clock = clock or time.time
        # Optional EndpointTelemetryBuilder (genuine OS sensing beyond
        # psutil: native process/connection/service sensors + detector
        # alert feeds). None keeps the legacy psutil-only path.
        self.builder = builder
        self._topology = self._discover()

    # -- discovery (REAL) --
    def _discover(self):
        try:
            hostname = socket.gethostname() or "unknown"
        except Exception:
            hostname = "unknown"
        ips, _ = _local_addresses()
        macs = _local_macs()
        asset_id = asset_id_for_hostname(hostname)
        asset = RealAsset(
            asset_id=asset_id, hostname=hostname,
            ips=tuple(ips), macs=tuple(macs),
            os_name=platform.system(), os_version=platform.version(),
            role=self.role, segment_id="host-only",
            last_seen=float(self._clock()),
            coverage="covered-local",
            coverage_detail="sensed on this machine")
        segment = RealSegment(segment_id="host-only",
                              name="This machine (host-only)",
                              kind="host-only", cidrs=())
        return RealTopology(assets=[asset], segments=[segment],
                            source=self.name,
                            discovered_at=float(self._clock()))

    @property
    def topology(self):
        """Current real topology (re-discovered every batch)."""
        return self._topology

    # -- TelemetryCollector contract (REAL) --
    @property
    def exhausted(self):
        return False

    def next_batch(self):
        now = float(self._clock())
        self._topology = self._discover()
        asset = self._topology.assets[0]
        if self.builder is not None:
            host, partial = self.builder.build(now=now)
            host.key = asset.asset_id
            services = list(getattr(host, "services", []) or [])
            asset.services = tuple(services)
            return TelemetryBatch(
                timestamp=now, hosts={asset.asset_id: host},
                notes=(f"endpoint-sensor telemetry for "
                       f"{asset.hostname}"),
                source=self.name, partial_errors=partial)
        processes, connections, note = _psutil_inventory()
        partial = ([{"key": asset.asset_id, "error": note}] if note
                   else [])
        host = HostTelemetry(
            key=asset.asset_id, processes=processes,
            connections=connections, sessions=[], up=True, events=[])
        return TelemetryBatch(
            timestamp=now, hosts={asset.asset_id: host},
            notes=f"local-machine telemetry for {asset.hostname}",
            source=self.name, partial_errors=partial)

    def close(self):
        pass


def describe_real_asset(asset):
    """One-line operator summary (real identifiers only)."""
    ip = ""
    for candidate in asset.ips or ():
        if not _is_loopback(candidate):
            ip = candidate
            break
    location = ip or (asset.ips[0] if asset.ips else "no address")
    return (f"{asset.hostname} [{asset.asset_id}] @ {location} "
            f"({asset.os_name or 'unknown OS'})")


__all__ = [
    "RealAsset", "RealInterface", "RealLink", "RealSegment",
    "RealTopology", "LocalMachineCollector",
    "asset_id_for_hostname", "describe_real_asset",
]
