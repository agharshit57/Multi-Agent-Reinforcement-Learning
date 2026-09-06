"""Network inventory: REAL multi-host topology (no fabrication).

COMPONENT KIND: real (OS-observed: ARP, interface addresses) +
adapter-dependent (operator-declared fleet file, agent collectors).

Sources, in honesty order:

1. LOCAL (real_world.LocalMachineCollector): full sensing, covered.
2. DECLARED (DeclaredInventory JSON): operator-asserted real machines.
   Coverage ``uncovered`` until an agent reports for them -- their
   policy slots read STALE (fail-safe), never quiet, never healthy.
3. ARP-OBSERVED (read_arp_table): IPs seen on the wire. Stub assets
   ``ip:<addr>`` with coverage ``observed-arp-only``: we know SOMETHING
   answered there, nothing more. Never enriched with guesses.
4. AGENT-BACKED (agent_collectors): remote hosts running a compatible
   telemetry agent (interface; stdlib has no remote agent yet --
   documented future, same shape as EndpointTelemetryBuilder output).

Segments: computed ONLY from observed netmasks (``ip addr`` / netsh)
or explicit declared segments. No mask = no grouping (per-host
segments + ARP links), never a /24 guess. Links: ``arp`` (L2-seen),
``subnet`` (shared observed subnet), ``declared`` (shared declared
segment).

``FleetCollector`` yields asset_id-keyed batches containing ONLY
covered assets; uncovered assets are omitted so the frozen normalizer
marks them STALE (unknown, investigate) -- silence is never quiet.
"""

import ipaddress
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .real_world import (RealAsset, RealLink, RealSegment, RealTopology,
                         asset_id_for_hostname)
from .telemetry import TelemetryBatch, TelemetryCollector

COMPONENT_KIND = "real"


class InventoryError(ValueError):
    """Declared fleet file is invalid (loud, with path + reason)."""


# ------------------------------------------------------------------- ARP --

@dataclass
class ArpEntry:
    ip: str
    mac: str = ""
    iface: str = ""


def parse_arp_windows(text):
    """``arp -a`` -> [ArpEntry]. Skips headers/incomplete entries."""
    entries = []
    for line in (text or "").splitlines():
        parts = line.split()
        # "192.168.1.1  aa-bb-cc-dd-ee-ff  dynamic"
        if len(parts) < 3:
            continue
        ip, mac, kind = parts[0], parts[1], parts[2].lower()
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        if not re.fullmatch(r"[0-9a-fA-F-]{17}", mac):
            continue
        if kind not in ("dynamic", "static"):
            continue
        entries.append(ArpEntry(ip=ip, mac=mac.lower()))
    # Deduplicate, stable order.
    seen, out = set(), []
    for entry in sorted(entries, key=lambda e: e.ip):
        if entry.ip not in seen:
            seen.add(entry.ip)
            out.append(entry)
    return out


def parse_proc_net_arp(text):
    """Linux /proc/net/arp -> [ArpEntry] (complete numeric entries)."""
    entries = []
    lines = (text or "").splitlines()
    for line in lines[1:]:  # header
        parts = line.split()
        # IP HWtype Flags HWaddr Mask Device
        if len(parts) < 6:
            continue
        ip, flags, mac, device = parts[0], parts[2], parts[3], parts[5]
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        if flags == "0x0" or mac == "00:00:00:00:00:00":
            continue  # incomplete/failed resolution: not evidence
        entries.append(ArpEntry(ip=ip, mac=mac.lower(), iface=device))
    seen, out = set(), []
    for entry in sorted(entries, key=lambda e: e.ip):
        if entry.ip not in seen:
            seen.add(entry.ip)
            out.append(entry)
    return out


def parse_ip_neigh(text):
    """Linux ``ip neigh`` -> [ArpEntry] (REACHABLE/STALE/DELAY only)."""
    entries = []
    for line in (text or "").splitlines():
        # "192.168.1.1 dev eth0 lladdr aa:bb:... REACHABLE"
        match = re.match(
            r"(\S+)\s+dev\s+(\S+)(?:\s+lladdr\s+(\S+))?\s+(\S+)",
            line.strip())
        if not match:
            continue
        ip, device, mac, state = match.groups()
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        if state.upper() in ("FAILED", "INCOMPLETE"):
            continue  # no L2 proof: not evidence
        entries.append(ArpEntry(ip=ip, mac=(mac or "").lower(),
                                iface=device))
    seen, out = set(), []
    for entry in sorted(entries, key=lambda e: e.ip):
        if entry.ip not in seen:
            seen.add(entry.ip)
            out.append(entry)
    return out


def read_arp_table(timeout_s=15):
    """(entries, available, reason), platform-dispatched, best-effort."""
    if sys.platform.startswith("win"):
        try:
            proc = subprocess.run(["arp", "-a"], capture_output=True,
                                  timeout=timeout_s, check=False)
        except (FileNotFoundError, subprocess.SubprocessError,
                OSError) as exc:
            return [], False, f"arp unavailable: {exc}"
        if proc.returncode != 0:
            return [], False, "arp -a failed"
        return (parse_arp_windows(
            proc.stdout.decode("utf-8", errors="replace")), True, "")
    # Linux: /proc first (rootless), ip neigh fallback.
    try:
        with open("/proc/net/arp", "r", encoding="utf-8",
                  errors="replace") as fh:
            return parse_proc_net_arp(fh.read()), True, ""
    except OSError:
        pass
    try:
        proc = subprocess.run(["ip", "neigh"], capture_output=True,
                              timeout=timeout_s, check=False)
    except (FileNotFoundError, subprocess.SubprocessError,
            OSError) as exc:
        return [], False, f"no ARP source (/proc + ip neigh failed: {exc})"  # noqa: E501
    if proc.returncode != 0:
        return [], False, "ip neigh failed"
    return (parse_ip_neigh(
        proc.stdout.decode("utf-8", errors="replace")), True, "")


# -------------------------------------------------------------- subnets --

def parse_ip_addr(text):
    """``ip -o -f inet addr show`` -> {iface: [(ip, prefixlen)]}."""
    found = {}
    for line in (text or "").splitlines():
        # "2: eth0    inet 172.24.16.5/20 brd ... scope global eth0"
        match = re.search(r"^\d+:\s+(\S+)\s+inet\s+(\S+)/(\d+)",
                          line.strip())
        if not match:
            continue
        iface, ip, prefix = match.groups()
        try:
            ipaddress.ip_address(ip)
            prefixlen = int(prefix)
        except ValueError:
            continue
        if ipaddress.ip_address(ip).is_loopback:
            continue
        found.setdefault(iface, []).append((ip, prefixlen))
    return found


def parse_netsh_addresses(text):
    """``netsh interface ipv4 show addresses`` -> {iface: [(ip, plen)]}.

    Locale-fragile by nature (English labels); unparseable output
    yields {} (unknown subnets), never guessed ones.
    """
    found, current = {}, None
    for line in (text or "").splitlines():
        head = re.match(r'Configuration for interface "(.+)"', line)
        if head:
            current = head.group(1)
            continue
        ip_match = re.match(r"\s*IP Address:\s*(\S+)", line)
        if ip_match and current:
            found.setdefault(current, []).append(
                [ip_match.group(1), None])
        prefix_match = re.match(
            r"\s*Subnet Prefix:\s*(\S+)/(\d+)", line)
        if prefix_match and current and found.get(current):
            try:
                plen = int(prefix_match.group(2))
            except ValueError:
                continue
            found[current][-1][1] = plen
    out = {}
    for iface, pairs in found.items():
        for ip, plen in pairs:
            try:
                if ipaddress.ip_address(ip).is_loopback:
                    continue
                out.setdefault(iface, []).append((ip, plen or 32))
            except ValueError:
                continue
    return out


def local_subnets(timeout_s=15):
    """Observed local subnets as ip_network objects (may be []).

    Empty means "subnets unknown" (segment inference degrades to
    per-host segments + ARP links), never a guessed /24.
    """
    nets = []
    if sys.platform.startswith("win"):
        try:
            proc = subprocess.run(
                ["netsh", "interface", "ipv4", "show", "addresses"],
                capture_output=True, timeout=timeout_s, check=False)
        except (FileNotFoundError, subprocess.SubprocessError,
                OSError):
            return []
        if proc.returncode != 0:
            return []
        addrs = parse_netsh_addresses(
            proc.stdout.decode("utf-8", errors="replace"))
    else:
        try:
            proc = subprocess.run(
                ["ip", "-o", "-f", "inet", "addr", "show"],
                capture_output=True, timeout=timeout_s, check=False)
        except (FileNotFoundError, subprocess.SubprocessError,
                OSError):
            return []
        if proc.returncode != 0:
            return []
        addrs = parse_ip_addr(
            proc.stdout.decode("utf-8", errors="replace"))
    for pairs in addrs.values():
        for ip, plen in pairs:
            try:
                nets.append(ipaddress.ip_network(f"{ip}/{plen}",
                                                 strict=False))
            except ValueError:
                continue
    # Deduplicate, stable order.
    return sorted(set(nets), key=lambda n: (n.network_address.packed,
                                            n.prefixlen))


# --------------------------------------------------------------- declared --

@dataclass
class DeclaredHost:
    hostname: str
    ips: tuple = ()
    macs: tuple = ()
    role: str = ""
    segment: str = ""


class DeclaredInventory:
    """Operator-asserted real machines (adapter-dependent source)."""

    def __init__(self, hosts):
        self.hosts = list(hosts)

    @classmethod
    def from_json_file(cls, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            raise InventoryError(
                f"declared inventory not found: {path}")
        except json.JSONDecodeError as exc:
            raise InventoryError(
                f"declared inventory {path} is not valid JSON: {exc}")
        except OSError as exc:
            raise InventoryError(
                f"declared inventory {path} unreadable: {exc}")
        if not isinstance(raw, dict) or not isinstance(
                raw.get("hosts"), list):
            raise InventoryError(
                f"declared inventory {path} needs "
                f"{{\"hosts\": [...]}}")
        hosts, seen_ids, seen_ips = [], set(), {}
        for pos, item in enumerate(raw["hosts"]):
            where = f"{path} host #{pos}"
            if not isinstance(item, dict):
                raise InventoryError(f"{where} must be an object")
            hostname = str(item.get("hostname", "") or "").strip()
            if not hostname:
                raise InventoryError(f"{where} needs a hostname")
            asset_id = asset_id_for_hostname(hostname)
            if asset_id in seen_ids:
                raise InventoryError(
                    f"{where}: duplicate hostname {hostname!r}")
            seen_ids.add(asset_id)
            ips = item.get("ips", [])
            if (not isinstance(ips, list) or not ips or not all(
                    isinstance(i, str) and i.strip() for i in ips)):
                raise InventoryError(
                    f"{where}: 'ips' must be a non-empty string list")
            for ip in ips:
                try:
                    ipaddress.ip_address(ip.strip())
                except ValueError:
                    raise InventoryError(
                        f"{where}: invalid IP {ip!r}")
                if ip.strip() in seen_ips:
                    raise InventoryError(
                        f"{where}: duplicate IP {ip!r} (also on "
                        f"{seen_ips[ip.strip()]})")
                seen_ips[ip.strip()] = hostname
            macs = tuple(str(m) for m in (item.get("macs", []) or ()))
            hosts.append(DeclaredHost(
                hostname=hostname, ips=tuple(i.strip() for i in ips),
                macs=macs, role=str(item.get("role", "") or ""),
                segment=str(item.get("segment", "") or "")))
        return cls(hosts)

    def to_assets(self):
        return [RealAsset(
            asset_id=asset_id_for_hostname(h.hostname),
            hostname=h.hostname, ips=h.ips, macs=h.macs,
            role=h.role, segment_id=h.segment or "declared",
            coverage="uncovered",
            coverage_detail=("declared by operator; no sensor coverage "
                             "-- state unknown (reads STALE)"))
            for h in self.hosts]


# ---------------------------------------------------------- fleet build --

def build_fleet_topology(local_asset, declared_assets=(),
                         arp_entries=(), subnets=()):
    """Assemble multi-host RealTopology (deterministic, labeled).

    * local_asset .... covered-local (full sensing).
    * declared ..... uncovered (no sensor yet) -- STALE downstream.
    * arp-only ..... observed-arp-only stubs (wire proof, nothing more).
    * segments ..... shared observed subnet -> lan-<cidr>; explicit
      declared segments honored as operator truth; else per-host.
    * links ........ arp / subnet / declared, always labeled.
    """
    assets = [local_asset]
    known_ips = set(local_asset.ips or ())
    links = []
    # Declared assets (skip any that duplicate the local machine).
    local_ids = {local_asset.asset_id}
    for asset in declared_assets:
        if asset.asset_id in local_ids or \
                set(asset.ips or ()) & known_ips:
            continue
        assets.append(asset)
        known_ips.update(asset.ips or ())
    # ARP stubs for wire-seen IPs we cannot attribute. Broadcast,
    # multicast, and unspecified addresses are not peers (never become
    # stub "assets"); neither are broadcast MACs.
    broadcasts = set()
    for net in subnets or ():
        try:
            broadcasts.add(net.broadcast_address)
        except Exception:
            continue
    for entry in arp_entries or ():
        if entry.ip in known_ips:
            continue
        try:
            parsed = ipaddress.ip_address(entry.ip)
        except ValueError:
            continue
        if parsed.is_loopback or parsed.is_multicast \
                or parsed.is_unspecified or parsed in broadcasts:
            continue
        if (entry.mac or "").lower() in ("ff:ff:ff:ff:ff:ff",
                                          "ff-ff-ff-ff-ff-ff"):
            continue
        stub = RealAsset(
            asset_id=f"ip:{entry.ip}", hostname=entry.ip,
            ips=(entry.ip,),
            macs=(entry.mac,) if entry.mac else (),
            segment_id="observed",
            coverage="observed-arp-only",
            coverage_detail=("seen in ARP table"
                             + (f" on {entry.iface}" if entry.iface
                                else "") + "; nothing else known"))
        assets.append(stub)
        known_ips.add(entry.ip)
        links.append(RealLink(asset_id=local_asset.asset_id,
                              peer=stub.asset_id, via="arp"))
    # Segments: shared observed subnet -> one lan segment.
    segments, by_segment = [], {}
    if subnets:
        grouped = {}
        for asset in assets:
            for ip in asset.ips or ():
                for net in subnets:
                    try:
                        if ipaddress.ip_address(ip) in net:
                            grouped.setdefault(str(net), []).append(
                                asset.asset_id)
                    except ValueError:
                        continue
        for cidr, members in sorted(grouped.items()):
            if len(members) >= 2:
                seg_id = f"lan-{cidr.replace('/', '_')}"
                segments.append(RealSegment(
                    segment_id=seg_id, name=f"LAN {cidr}",
                    kind="lan", cidrs=(cidr,)))
                for member in members:
                    by_segment[member] = seg_id
                for first, second in zip(sorted(members),
                                         sorted(members)[1:]):
                    links.append(RealLink(asset_id=first, peer=second,
                                          via="subnet"))
    # Declared segment hints (operator truth, only for declared pairs).
    declared_groups = {}
    for asset in assets:
        if asset.coverage == "uncovered" and asset.segment_id not in (
                "", "declared"):
            declared_groups.setdefault(asset.segment_id, []).append(
                asset.asset_id)
    for seg_id, members in sorted(declared_groups.items()):
        if len(members) >= 2 and seg_id not in by_segment.values():
            segments.append(RealSegment(
                segment_id=seg_id, name=f"Declared {seg_id}",
                kind="declared", cidrs=()))
            for first, second in zip(sorted(members),
                                     sorted(members)[1:]):
                links.append(RealLink(asset_id=first, peer=second,
                                      via="declared"))
    # Per-host fallback segments for anything ungrouped.
    final_assets = []
    for asset in assets:
        seg = by_segment.get(asset.asset_id)
        if seg is None:
            if asset.coverage == "uncovered" and asset.segment_id \
                    not in ("", "declared"):
                seg = asset.segment_id
                if seg not in [s.segment_id for s in segments]:
                    segments.append(RealSegment(
                        segment_id=seg, name=f"Declared {seg}",
                        kind="declared", cidrs=()))
            elif asset.coverage == "observed-arp-only":
                seg = "observed"
                if "observed" not in [s.segment_id for s in segments]:
                    segments.append(RealSegment(
                        segment_id="observed",
                        name="Observed on wire (ARP)",
                        kind="observed", cidrs=()))
            else:
                seg = "host-only"
                if "host-only" not in [s.segment_id for s in segments]:
                    segments.append(RealSegment(
                        segment_id="host-only",
                        name="This machine (host-only)",
                        kind="host-only", cidrs=()))
        final_assets.append(RealAsset(
            asset_id=asset.asset_id, hostname=asset.hostname,
            ips=asset.ips, macs=asset.macs, os_name=asset.os_name,
            os_version=asset.os_version, role=asset.role,
            segment_id=seg, services=asset.services,
            last_seen=asset.last_seen, coverage=asset.coverage,
            coverage_detail=asset.coverage_detail))
    # Deterministic link order.
    links = sorted(links, key=lambda l: (l.asset_id, l.peer, l.via))
    deduped = []
    for link in links:
        if not deduped or (deduped[-1].asset_id != link.asset_id
                           or deduped[-1].peer != link.peer
                           or deduped[-1].via != link.via):
            deduped.append(link)
    return RealTopology(assets=sorted(final_assets,
                                      key=lambda a: a.asset_id),
                        segments=sorted(segments,
                                        key=lambda s: s.segment_id),
                        links=deduped, source="fleet-inventory",
                        discovered_at=time.time())


# ---------------------------------------------------------- collector --

class FleetCollector(TelemetryCollector):
    """Live multi-asset collector: covered assets only, honest gaps.

    * local asset -> EndpointTelemetryBuilder (REAL sensing).
    * agent_collectors {asset_id: collector} -> remote agents
      (adapter-dependent; each exposes
      ``collect(asset_id) -> HostTelemetry | None``).
    * everyone else (declared-but-uncovered, arp stubs) -> OMITTED,
      so the frozen normalizer marks them STALE via the adapter.
      Omission counts ride in ``notes`` + ``partial_errors``.
    ``exhausted`` is always False (live sensor, no end-of-stream).
    """

    name = "fleet"

    def __init__(self, local_builder, local_asset_fn,
                 declared_assets=(), agent_collectors=None,
                 arp_fn=None, subnets_fn=None, arp_ttl_s=60.0,
                 clock=None):
        self.local_builder = local_builder
        self._local_asset_fn = local_asset_fn
        self._declared = list(declared_assets or ())
        self._agents = dict(agent_collectors or {})
        self._arp_fn = arp_fn or read_arp_table
        self._subnets_fn = subnets_fn or local_subnets
        self._arp_ttl = float(arp_ttl_s)
        self._clock = clock or time.time
        self._arp_cache = (0.0, [], True, "")
        self._last_arp_note = ""

    @property
    def topology(self):
        return self._fleet_topology()

    @property
    def exhausted(self):
        return False

    def _fleet_topology(self):
        now = float(self._clock())
        cached_at, entries, available, reason = self._arp_cache
        if now - cached_at > self._arp_ttl:
            try:
                entries, available, reason = self._arp_fn()
            except Exception as exc:
                entries, available, reason = (
                    [], False, f"arp reader raised "
                               f"{type(exc).__name__}: {exc}")
            self._arp_cache = (now, entries, available, reason)
        else:
            entries, available, reason = self._arp_cache[1:]
        try:
            subnets = self._subnets_fn()
        except Exception:
            subnets = []
        local_asset = self._local_asset_fn()
        topology = build_fleet_topology(
            local_asset, declared_assets=self._declared,
            arp_entries=entries if available else (), subnets=subnets)
        self._last_arp_note = "" if available else (
            f"arp unavailable ({reason}); wire-observed peers unknown")
        return topology

    def next_batch(self):
        now = float(self._clock())
        topology = self._fleet_topology()
        by_id = {a.asset_id: a for a in topology.assets}
        hosts, partial, covered = {}, [], []
        # Local asset: genuine sensing.
        local_ids = [a.asset_id for a in topology.assets
                     if a.coverage == "covered-local"]
        for asset_id in local_ids:
            host, errors = self.local_builder.build(now=now)
            host.key = asset_id
            hosts[asset_id] = host
            partial.extend(errors)
            covered.append(asset_id)
        # Agent-backed remotes.
        for asset_id, agent in self._agents.items():
            if asset_id not in by_id:
                partial.append({"key": asset_id,
                                "error": "agent configured for unknown "
                                         "asset (not in topology)"})
                continue
            try:
                host = agent.collect(asset_id)
            except Exception as exc:
                partial.append({"key": asset_id,
                                "error": f"agent raised "
                                         f"{type(exc).__name__}: {exc}"})
                continue
            if host is None:
                partial.append({"key": asset_id,
                                "error": "agent returned no data "
                                         "(reads STALE)"})
                continue
            host.key = asset_id
            hosts[asset_id] = host
            covered.append(asset_id)
        uncovered = sorted(set(by_id) - set(covered))
        if uncovered:
            partial.append({"key": "fleet",
                            "error": f"{len(uncovered)} asset(s) without "
                                     f"sensor coverage (reads STALE): "
                                     f"{','.join(uncovered)}"})
        if self._last_arp_note:
            partial.append({"key": "fleet",
                            "error": self._last_arp_note})
        return TelemetryBatch(
            timestamp=now, hosts=hosts,
            notes=(f"fleet telemetry: {len(covered)} covered, "
                   f"{len(uncovered)} uncovered (stale)"),
            source=self.name, partial_errors=partial)

    def close(self):
        pass


__all__ = [
    "COMPONENT_KIND", "InventoryError", "ArpEntry",
    "parse_arp_windows", "parse_proc_net_arp", "parse_ip_neigh",
    "read_arp_table", "parse_ip_addr", "parse_netsh_addresses",
    "local_subnets", "DeclaredHost", "DeclaredInventory",
    "build_fleet_topology", "FleetCollector",
]
