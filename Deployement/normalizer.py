"""Telemetry normalizer: real telemetry -> CC4-compatible host semantics.

Per bound asset the normalizer derives the same three signals training
used (see ``Marl/mappo/env.py::_compute_host_snapshot``):
  - ``compromised`` : ONLY from explicit high-confidence intrusion
    evidence (``intrusion_confirmed`` events or critical IDS alerts
    naming a red-style session). Benign/admin activity can NEVER set
    this -- the one signal that cannot come from legitimate use.
    Compromise is STICKY: once seen it persists until an explicit
    recovery signal (``recovery_confirmed``/``host_reimaged``/
    ``remediated`` event, the ``recovered`` set, or ``reset()``),
    because an intrusion does not vanish when telemetry goes quiet.
  - ``process_event`` / ``connection_event``: any activity outside the
    configured baselines. NOT intrusion-exclusive (legitimate admin
    work also lands here), exactly like Green activity in simulation.
    A compromised host ALWAYS raises both (promotion): in-sim, red
    sessions unconditionally generate host events (observed AND
    unobserved), so this is what the equivalent training evidence
    looks like -- and it is what lets masks/policy respond without
    changing the trained 92/210-dim representation.

Health semantics (fail-safe, never quiet-by-default):
  - ``quiet``      = fresh telemetry + no evidence
  - ``stale``      = telemetry missing/stale/failed for this host.
    Stale forces BOTH alert bits True (investigative posture) while
    preserving last-known ``compromised``. Unknown is NEVER healthy.
  - ``compromised``= explicit intrusion evidence (sticky, see above).

Baselines are resolved per host with explicit precedence
host > role > zone > global; each dimension (processes/ports/peers)
uses the most-specific scope that defines it. An undefined dimension
means "everything is an event" (noisy but safe).
"""

from dataclasses import dataclass, field


INTRUSION_KINDS = frozenset({"intrusion_confirmed", "red_session"})
RECOVERY_KINDS = frozenset({"recovery_confirmed", "host_reimaged",
                            "reimaged", "remediated"})
HIGH_SEVERITIES = frozenset({"high", "critical"})

HEALTH_QUIET = "quiet"
HEALTH_STALE = "stale"
HEALTH_COMPROMISED = "compromised"

# Baseline schema (inferred from what the normalizer consumes):
#   {"processes": [str...], "ports": [0..65535...],
#    "peers": [str...],
#    "roles": {role: scope}, "zones": {zone: scope},
#    "hosts": {key-or-cc4: scope}}
# where scope = {"processes": [...], "ports": [...], "peers": [...]}.
# Unknown keys are rejected (typo safety); ports accept ints or digit
# strings in range; empty/missing dimensions mean "everything is an
# event" (fail-safe, never silent).
BASELINE_DIMENSIONS = ("processes", "ports", "peers")
BASELINE_SCOPES = ("roles", "zones", "hosts")


def _is_str_list(value):
    return (isinstance(value, (list, tuple, set, frozenset))
            and all(isinstance(v, str) and v.strip() for v in value))


def _check_ports(value, where):
    if isinstance(value, (str, bytes)) or not isinstance(
            value, (list, tuple, set, frozenset)):
        raise ValueError(f"baselines: {where}.ports must be a list of "
                         f"port numbers, got {value!r}")
    for port in value:
        if isinstance(port, bool):
            raise ValueError(f"baselines: {where}.ports has boolean "
                             f"{port!r} (not a port number)")
        if isinstance(port, int):
            number = port
        elif isinstance(port, str) and port.strip().isdigit():
            number = int(port.strip())
        else:
            raise ValueError(f"baselines: {where}.ports has invalid "
                             f"port {port!r} (want 0..65535)")
        if not 0 <= number <= 65535:
            raise ValueError(f"baselines: {where}.ports has out-of-range "
                             f"port {port!r} (want 0..65535)")


def _check_scope(scope, where):
    if not isinstance(scope, dict):
        raise ValueError(f"baselines: {where} must be an object, "
                         f"got {scope!r}")
    for key in scope:
        if key not in BASELINE_DIMENSIONS:
            raise ValueError(f"baselines: {where} has unknown field "
                             f"{key!r}; valid fields: "
                             f"{list(BASELINE_DIMENSIONS)}")
    if "processes" in scope and not _is_str_list(scope["processes"]):
        raise ValueError(f"baselines: {where}.processes must be a list "
                         f"of non-empty strings")
    if "ports" in scope:
        _check_ports(scope["ports"], where)
    if "peers" in scope and not _is_str_list(scope["peers"]):
        raise ValueError(f"baselines: {where}.peers must be a list "
                         f"of non-empty strings")


def validate_baselines(baselines):
    """Validate baseline config BEFORE it can shape detection.

    Returns the input unchanged (None -> {}) so call sites can chain.
    Anything malformed raises ValueError with the exact offending path
    -- a half-loaded baseline must never silently change what counts
    as anomalous.
    """
    if baselines is None:
        return {}
    if not isinstance(baselines, dict):
        raise ValueError("baselines root must be an object, "
                         f"got {type(baselines).__name__}")
    for key in baselines:
        if key not in BASELINE_DIMENSIONS + BASELINE_SCOPES:
            raise ValueError(f"baselines: unknown top-level field {key!r}; "
                             f"valid: "
                             f"{list(BASELINE_DIMENSIONS + BASELINE_SCOPES)}")
    _check_scope({k: baselines[k] for k in BASELINE_DIMENSIONS
                  if k in baselines}, "baselines")
    for scope_name in BASELINE_SCOPES:
        scopes = baselines.get(scope_name, {})
        if not isinstance(scopes, dict):
            raise ValueError(f"baselines: {scope_name!r} must map names "
                             f"to scopes, got {scopes!r}")
        for name, scope in scopes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"baselines: {scope_name!r} has an "
                                 f"invalid scope name {name!r}")
            _check_scope(scope, f"baselines.{scope_name}.{name!r}")
    return baselines


SEVERITIES = ("critical", "high", "medium", "low", "unknown", "none")

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3,
                  "critical": 4}


@dataclass
class NormalizedHost:
    cc4: str
    key: str
    compromised: bool = False
    process_event: bool = False
    connection_event: bool = False
    session_count: int = 0
    up: bool = True
    health: str = HEALTH_QUIET
    last_seen: float = 0.0
    # Worst-case severity of this host's current evidence, for display
    # filtering only -- it never changes flags, masks, or policy input.
    # "unknown" marks stale hosts without compromise evidence.
    severity: str = "none"
    notes: list = field(default_factory=list)


@dataclass
class NormalizedState:
    hosts: dict          # cc4 -> NormalizedHost (bound assets only)
    unseen_keys: list    # telemetry keys with no asset-map slot


class TelemetryNormalizer:
    def __init__(self, baselines=None, stale_after_s=300):
        # Validated up front: malformed baselines fail HERE with the
        # exact path, never half-load and silently reshape detection.
        # Per dimension, the most-specific scope defining it wins
        # (host > role > zone > global); an undefined dimension means
        # "everything is an event" (fail-safe).
        self.baselines = validate_baselines(baselines)
        self.stale_after_s = float(stale_after_s)
        self._last_seen = {}        # cc4 -> timestamp of fresh telemetry
        self._last_compromised = {}  # cc4 -> sticky compromise flag

    def reset(self):
        """Clear freshness/compromise memory (new deployment session)."""
        self._last_seen.clear()
        self._last_compromised.clear()

    # ------------------------------------------------------------- API --
    def normalize(self, batch, asset_map, now=None, recovered=()):
        """Fold one TelemetryBatch into per-CC4-host signals.

        ``now`` defaults to the batch timestamp (deterministic replay);
        live deployments should pass wall-clock time so staleness is
        measured against the real clock. ``recovered`` is an iterable
        of cc4 names with confirmed remediation (clears stickiness).

        Batches are FULL snapshots: any bound host absent from the
        batch reads as STALE immediately (never as quiet), no matter
        how recently it was seen. Differential/incremental feeds must
        be expanded to snapshots by the collector.
        """
        from .asset_map import match_zone
        now = batch.timestamp if now is None else float(now)
        for cc4 in recovered:
            self._last_compromised.pop(cc4, None)
        # Reverse index: telemetry key -> list of (agent, cc4).
        key_to_slots = {}
        slot_meta = {}  # cc4 -> (agent, role, zone)
        for agent in asset_map.agents:
            for cc4, info in asset_map.agents[agent]["hosts"].items():
                try:
                    zone = match_zone(cc4)
                except Exception:
                    zone = ""
                slot_meta[cc4] = (agent, str(info.get("role", "")),
                                  zone)
                for key in (info.get("ip", ""), info.get("hostname", "")):
                    if key:
                        key_to_slots.setdefault(key, []).append(
                            (agent, cc4))
        batch_stale = (now - float(batch.timestamp)) > self.stale_after_s
        failed_keys = {e["key"] for e in (batch.partial_errors or [])
                       if isinstance(e, dict) and "key" in e}
        hosts = {}
        for agent in asset_map.agents:
            for cc4 in asset_map.agents[agent]["hosts"]:
                hosts[cc4] = NormalizedHost(cc4=cc4, key="")
        unseen = []
        seen_cc4 = set()
        for key, tele in batch.hosts.items():
            slots = key_to_slots.get(key, [])
            if not slots:
                unseen.append(key)
                continue
            for _agent, cc4 in slots:
                if key in failed_keys:
                    continue  # handled as stale below, not as fresh data
                norm = hosts[cc4]
                norm.key = key
                seen_cc4.add(cc4)
                self._fold_host(norm, tele, asset_map, cc4, slot_meta)
                self._last_seen[cc4] = now
                if norm.compromised:
                    self._last_compromised[cc4] = True
                elif self._last_compromised.get(cc4, False):
                    # Same promotion as fresh evidence: an unremediated
                    # compromise still generates host events in-sim, so
                    # masks/policy must keep responding to it.
                    norm.compromised = True
                    norm.process_event = True
                    norm.connection_event = True
                    norm.severity = "critical"
                    norm.notes.append("compromise carried (sticky)")
        # Staleness pass: absent, failed, ancient-batch, or never-seen
        # hosts are UNKNOWN -- forced alert posture, never quiet.
        for cc4, norm in hosts.items():
            last = self._last_seen.get(cc4)
            is_stale = (batch_stale or cc4 not in seen_cc4
                        or last is None
                        or (now - last) > self.stale_after_s)
            if is_stale:
                norm.health = HEALTH_STALE
                norm.process_event = True
                norm.connection_event = True
                if self._last_compromised.get(cc4, False):
                    norm.compromised = True
                    norm.severity = "critical"
                    norm.notes.append("compromise carried (sticky, stale)")
                else:
                    # Stale hosts keep no severity claim: unknown, and
                    # graded as unknown (never quiet, never critical).
                    norm.severity = "unknown"
                    norm.notes.append("telemetry stale/missing: "
                                      "fail-safe alert posture")
            elif norm.compromised:
                norm.health = HEALTH_COMPROMISED
        return NormalizedState(hosts=hosts, unseen_keys=sorted(set(unseen)))

    # ---------------------------------------------------------- internals --
    def _scope_for(self, cc4, key, slot_meta):
        """(processes, ports, peers) sets for one host, by precedence."""
        _agent, role, zone = slot_meta.get(cc4, ("", "", ""))
        scopes = [self.baselines]
        if zone and isinstance(self.baselines.get("zones"), dict):
            scopes.append(self.baselines["zones"].get(zone, {}))
        if role and isinstance(self.baselines.get("roles"), dict):
            scopes.append(self.baselines["roles"].get(role, {}))
        hosts_scope = self.baselines.get("hosts", {})
        if isinstance(hosts_scope, dict):
            scopes.append(hosts_scope.get(key, {}))
            if cc4 != key:
                scopes.append(hosts_scope.get(cc4, {}))

        def pick(dimension, coerce=None):
            for scope in reversed(scopes):
                if isinstance(scope, dict) and dimension in scope:
                    values = scope[dimension]
                    if coerce is not None:
                        return {coerce(v) for v in values}
                    return set(values)
            return set()
        return (pick("processes", str),
                pick("ports", int),
                pick("peers", str))

    def _fold_host(self, norm, tele, asset_map, cc4, slot_meta):
        norm.up = bool(tele.up)
        norm.session_count = len(tele.sessions)
        known_processes, known_ports, known_peers = self._scope_for(
            cc4, norm.key or cc4, slot_meta)
        rank = 0
        # Explicit intrusion evidence only (sets sticky compromise).
        for event in tele.events:
            kind = str(event.kind or "").lower()
            sev = str(event.severity or "").lower()
            if kind in RECOVERY_KINDS:
                norm.compromised = False
                self._last_compromised.pop(cc4, None)
                norm.notes.append(f"recovery evidence: {event.kind}")
            elif kind in INTRUSION_KINDS or (
                    kind == "ids_alert" and sev in HIGH_SEVERITIES
                    and "red" in (event.details or "").lower()):
                norm.compromised = True
                rank = max(rank, 4)
                norm.notes.append(f"intrusion evidence: {event.kind}")
            elif sev in HIGH_SEVERITIES:
                rank = max(rank, 3)
            elif sev == "medium" or kind in (
                    "suspicious_process", "suspicious_connection",
                    "ids_alert", "failed_login", "port_scan"):
                rank = max(rank, 2)
        # Baseline-relative activity.
        for proc in tele.processes:
            if proc not in known_processes:
                norm.process_event = True
                rank = max(rank, 1)
                break
        # Supervised services share the process scope: a service is a
        # supervised process, so an unknown one is the same class of
        # evidence (activity, never a verdict). Empty/absent services
        # change nothing (all existing quiet baselines stay quiet).
        for svc in getattr(tele, "services", None) or []:
            if svc not in known_processes:
                norm.process_event = True
                rank = max(rank, 1)
                norm.notes.append(f"unknown service: {svc}")
                break
        for conn in tele.connections:
            port = _port_of(conn)
            if (port is None or port not in known_ports
                    or not _peer_known(conn, known_peers)):
                norm.connection_event = True
                rank = max(rank, 1)
                break
        for event in tele.events:
            kind = str(event.kind or "").lower()
            if kind in ("process", "suspicious_process"):
                norm.process_event = True
                rank = max(rank, 2 if kind.startswith("suspicious") else 1)
            elif kind in ("connection", "suspicious_connection", "ids_alert",
                          "failed_login", "port_scan"):
                norm.connection_event = True
                rank = max(rank, 2 if kind not in ("connection",) else 1)
        # Promotion (item 7): compromise evidence subsumes alert-level
        # visibility -- in-sim, red sessions unconditionally generate
        # host events (observed AND unobserved), so the equivalent
        # training representation has these bits set. Without this, a
        # host with ONLY compromise evidence would look clean to the
        # mask/policy. This never creates compromise from alerts.
        if norm.compromised:
            norm.process_event = True
            norm.connection_event = True
            norm.health = HEALTH_COMPROMISED
            rank = max(rank, 4)
        norm.severity = {4: "critical", 3: "high", 2: "medium",
                         1: "low"}.get(rank, "none")


def _peer_segment(conn):
    """Extract the peer token from telemetry connection text.

    Canonical form is ``"proto:port->peer"`` (see telemetry.py); a bare
    ``"proto:port"`` or bare peer token is also accepted. Returns the
    lowercased, whitespace-stripped peer segment with one trailing
    FQDN dot removed ("" when absent).
    """
    text = str(conn).strip() if isinstance(conn, str) else ""
    if not text:
        return ""
    peer = text.rsplit("->", 1)[-1].strip().lower() if "->" in text else text.lower()  # noqa: E501
    return peer.rstrip(".")


def _strip_single_port_suffix(peer):
    """Remove one trailing ``:digits`` suffix (``host:443`` -> ``host``).

    Only a SINGLE colon may be present (IPv6 literals such as
    ``fe80::1`` are returned unchanged) and the suffix must be all
    digits; otherwise the token is returned unchanged.
    """
    if peer.count(":") == 1:
        head, _, tail = peer.partition(":")
        if head and tail.isdigit():
            return head
    return peer


def _looks_like_ip_prefix(token):
    """Digits-and-dots only (e.g. ``10.0.0``); hostnames excluded."""
    return bool(token) and all(ch.isdigit() or ch == "." for ch in token)


def peer_is_known(conn, known_peers):
    """Structured peer allowlist check (no substring matching).

    A connection's peer is known iff it EQUALS a known peer
    (case-insensitive; one trailing FQDN dot is ignored), or -- for
    dotted-numeric (IP) prefixes only -- it starts with
    ``known + "."``. So a ``10.0.0`` entry covers ``10.0.0.5`` but
    ``host1`` never matches ``host10`` or ``host1-extra``, and
    ``10.0.0.1`` never matches ``10.0.0.10``. Hostname allowlisting is
    deliberately exact: prefix-matching hostnames would treat distinct
    machines as one identity. Only the peer segment is compared --
    never the ``proto:port`` prefix.
    """
    if not known_peers:
        return False
    peer = _peer_segment(conn)
    if not peer:
        return False
    candidates = {peer, _strip_single_port_suffix(peer)}
    for known in known_peers:
        known_norm = str(known).strip().lower().rstrip(".")
        if not known_norm:
            continue
        for candidate in candidates:
            if candidate == known_norm:
                return True
            if (_looks_like_ip_prefix(known_norm)
                    and candidate.startswith(known_norm + ".")):
                return True
    return False


def _peer_known(conn, known_peers):
    """Back-compat alias (strict semantics since the substring fix)."""
    return peer_is_known(conn, known_peers)


def _port_of(conn):
    """Strict ``proto:port`` parse from telemetry connection text.

    Accepts ``"proto:port->peer"`` or bare ``"proto:port"`` where proto
    starts with a letter (letters/digits/+/-/. allowed) and port is
    1-5 digits in 0..65535. Anything else (missing/extra colons,
    non-digit or out-of-range ports, empty text) returns None, which
    the caller treats as an event (fail-safe: malformed records raise
    alerts, never silence).
    """
    if not isinstance(conn, str):
        return None
    text = conn.strip()
    head = text.split("->", 1)[0] if "->" in text else text
    if head.count(":") != 1:
        return None
    proto, _, port_s = head.partition(":")
    if not proto or not proto[0].isalpha():
        return None
    if any(ch not in "abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+.-"
           for ch in proto):
        return None
    if not port_s.isdigit() or len(port_s) > 5:
        return None
    port = int(port_s)
    if not 0 <= port <= 65535:
        return None
    return port
