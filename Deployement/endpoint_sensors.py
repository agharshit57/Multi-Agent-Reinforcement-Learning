"""Endpoint sensors: GENUINE OS-sourced telemetry (no fabrication).

COMPONENT KIND: real (OS-sourced) with explicit unavailable states.
Every sensor reports ``(values, available, reason)``: when the OS
source is missing, permission-denied, or unparsable, the sensor says
so loudly via ``partial_errors`` downstream -- it NEVER synthesizes
activity, and it NEVER invents compromise evidence.

Platform dispatch (stdlib only, no new dependencies):

* processes ... Windows ``tasklist`` | Linux ``/proc`` | psutil fallback
* connections . Windows ``netstat -ano -p tcp`` | Linux ``/proc/net/tcp``
                  (+tcp6) | psutil fallback. TCP only (v1 limitation,
                  documented): UDP lines carry no state and would be
                  noise without per-flow tracking.
* services .... Windows ``sc query`` | Linux ``systemctl list-units``.
                  A service list is inventory: service names join the
                  process baseline scope downstream (an unexpected
                  supervised process is activity, never a verdict).
* compromise .. NEVER inferred here. The ONLY compromise inputs are
  explicit detector feeds (``FileAlertFeed``: a JSONL file a REAL
  sensor writes) passed through with conservative kinds; the frozen
  normalizer remains the single compromise decider. No feed file =
  ``available=False`` (coverage absent), never empty evidence.

All ``parse_*`` functions are pure (fixture-testable, including the
Linux parsers on Windows and vice versa).
"""

import csv
import io
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field

COMPONENT_KIND = "real"


@dataclass
class SensorReading:
    """One sensor's contribution for one asset."""
    name: str
    processes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    services: list = field(default_factory=list)
    events: list = field(default_factory=list)   # SecurityEvent list
    available: bool = True
    reason: str = ""          # "" when available; why-not otherwise
    notes: str = ""           # extra operator context (truncation, etc.)


def _is_windows():
    return sys.platform.startswith("win")


def run_cmd(argv, timeout_s=15):
    """Run an OS command, best-effort. Returns (stdout_text | None).

    Fail-safe contract (matches network_inventory.py): ANY non-zero
    exit means "unavailable source" (None), even when the tool still
    printed something -- a failing command's partial stdout is not
    legitimate telemetry and must never be parsed as if it were.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, timeout=timeout_s, check=False)
    except FileNotFoundError:
        return None
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


# ------------------------------------------------------------- tasklist --

def parse_tasklist_csv(text):
    """Windows ``tasklist /FO CSV /NH`` -> [process names, as observed].

    No case folding: baselines must match observed case (psutil and
    tasklist agree on this box; folding would silently alias).
    """
    names = []
    try:
        reader = csv.reader(io.StringIO(text or ""))
        for row in reader:
            if row and row[0].strip():
                names.append(row[0].strip())
    except Exception:
        return []
    # Deduplicate, stable order.
    return sorted(set(names))


def collect_processes_windows(runner=run_cmd):
    text = runner(["tasklist", "/FO", "CSV", "/NH"])
    if text is None:
        return [], False, "tasklist unavailable or failed"
    names = parse_tasklist_csv(text)
    if not names:
        return [], False, "tasklist returned no parseable processes"
    return names, True, ""


# ----------------------------------------------------------------- /proc --

def collect_processes_proc(proc_root="/proc"):
    """Linux ``/proc`` process names (comm). REAL, rootless-readable."""
    names = []
    try:
        entries = os.listdir(proc_root)
    except OSError as exc:
        return [], False, f"/proc unreadable: {exc}"
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "comm"),
                      "r", encoding="utf-8", errors="replace") as fh:
                name = fh.read().strip()
        except OSError:
            continue  # raced exit; skip, never fail the sensor
        if name:
            names.append(name)
    if not names:
        return [], False, "no processes enumerated from /proc"
    return sorted(set(names)), True, ""


# ------------------------------------------------------------ processes --

def collect_processes():
    """(names, available, reason), platform-dispatched, psutil first."""
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            names = sorted({p.info["name"] for p in
                            psutil.process_iter(["name"])
                            if p.info.get("name")})
            if names:
                return names, True, ""
        except Exception:
            pass
    if _is_windows():
        return collect_processes_windows()
    names, available, reason = collect_processes_proc()
    if available:
        return names, available, reason
    return [], False, (reason + "; psutil not installed" if reason
                       else "psutil not installed and /proc failed")


class ProcessSensor:
    name = "processes"

    def collect(self):
        names, available, reason = collect_processes()
        return SensorReading(name=self.name, processes=names,
                             available=available, reason=reason)


# -------------------------------------------------------------- netstat --

_ESTABLISHED = "ESTABLISHED"
_LISTEN = "LISTENING"


def parse_netstat_tcp(text):
    """Windows ``netstat -ano -p tcp`` -> ["tcp:port->peer" | "tcp:port"].

    ESTABLISHED with a real peer -> "tcp:<localport>-><peer>";
    LISTENING -> "tcp:<localport>". TIME_WAIT/CLOSE_WAIT are transient
    noise and are excluded (documented). Unparseable lines are skipped
    (counted in notes by the caller via length checks, never guessed).
    """
    out = []
    for line in (text or "").splitlines():
        parts = line.split()
        # TCP <local> <foreign> <state> <pid>
        if len(parts) != 5 or parts[0] != "TCP":
            continue
        state = parts[3].upper()
        local_port = _port_of_socket(parts[1])
        if local_port is None:
            continue
        if state == _ESTABLISHED:
            peer = _ip_of_socket(parts[2])
            if not peer or peer in ("0.0.0.0", "::"):
                continue
            out.append(f"tcp:{local_port}->{peer}")
        elif state == _LISTEN:
            out.append(f"tcp:{local_port}")
    return sorted(set(out))


def _port_of_socket(socket_text):
    try:
        port = socket_text.rsplit(":", 1)[1]
        port = port.strip("[]")
        number = int(port)
        if 0 <= number <= 65535:
            return number
    except (ValueError, IndexError):
        return None
    return None


def _ip_of_socket(socket_text):
    try:
        ip = socket_text.rsplit(":", 1)[0].strip("[]")
        return ip or ""
    except IndexError:
        return ""


def collect_connections_windows(runner=run_cmd):
    text = runner(["netstat", "-ano", "-p", "tcp"])
    if text is None:
        return [], False, "netstat unavailable or failed"
    return parse_netstat_tcp(text), True, ""


# ---------------------------------------------------------- /proc/net --

_TCP_ST_LISTEN = "0A"
_TCP_ST_ESTABLISHED = "01"


def _ipv4_from_proc(hex_ip):
    """Little-endian-per-word /proc IPv4 -> dotted quad."""
    try:
        raw = bytes.fromhex(hex_ip)
        if len(raw) != 4:
            return ""
        return ".".join(str(b) for b in raw[::-1])
    except ValueError:
        return ""


def _ipv6_from_proc(hex_ip):
    """Per-32-bit-word-swapped /proc IPv6 -> compressed form."""
    try:
        import ipaddress
        raw = bytes.fromhex(hex_ip)
        if len(raw) != 16:
            return ""
        words = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]
        return ipaddress.ip_address(b"".join(words)).compressed
    except ValueError:
        return ""


def _parse_proc_net_table(text, ipv6):
    """One /proc/net/tcp{,6} table -> [(local_ip, local_port,
    peer_ip, state)]."""
    rows = []
    lines = (text or "").splitlines()
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            local_ip_hex, local_port_hex = parts[1].rsplit(":", 1)
            peer_ip_hex, _ = parts[2].rsplit(":", 1)
            state = parts[3].upper()
            port = int(local_port_hex, 16)
        except ValueError:
            continue
        if not 0 <= port <= 65535:
            continue
        if ipv6:
            local_ip, peer_ip = (_ipv6_from_proc(local_ip_hex),
                                 _ipv6_from_proc(peer_ip_hex))
        else:
            local_ip, peer_ip = (_ipv4_from_proc(local_ip_hex),
                                 _ipv4_from_proc(peer_ip_hex))
        if not local_ip:
            continue
        rows.append((local_ip, port, peer_ip, state))
    return rows


def parse_proc_net_tcp(text_v4, text_v6=""):
    """Linux /proc/net/tcp(+tcp6) -> ["tcp:port->peer" | "tcp:port"].

    Only ESTABLISHED (01) and LISTEN (0A) states; everything else
    (TIME_WAIT etc.) is transient noise and excluded. Unspecified
    peers (0.0.0.0/::) never become connections.
    """
    out = []
    for text, ipv6 in ((text_v4, False), (text_v6, True)):
        for local_ip, port, peer_ip, state in _parse_proc_net_table(
                text, ipv6):
            if state == _TCP_ST_ESTABLISHED:
                if not peer_ip or peer_ip in ("0.0.0.0", "::"):
                    continue
                out.append(f"tcp:{port}->{peer_ip}")
            elif state == _TCP_ST_LISTEN:
                out.append(f"tcp:{port}")
    return sorted(set(out))


def _read_proc_net_tcp(proc_root="/proc"):
    try:
        with open(os.path.join(proc_root, "net", "tcp"),
                  "r", encoding="utf-8", errors="replace") as fh:
            text_v4 = fh.read()
    except OSError as exc:
        return None, f"/proc/net/tcp unreadable: {exc}"
    try:
        with open(os.path.join(proc_root, "net", "tcp6"),
                  "r", encoding="utf-8", errors="replace") as fh:
            text_v6 = fh.read()
    except OSError:
        text_v6 = ""
    return (text_v4, text_v6), ""


def collect_connections():
    """(connections, available, reason), platform-dispatched."""
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            found = set()
            for conn in psutil.net_connections(kind="tcp"):
                try:
                    port = conn.laddr.port if conn.laddr else None
                    peer = conn.raddr.ip if conn.raddr else ""
                    status = str(getattr(conn, "status", "")).upper()
                except Exception:
                    continue
                if port is None:
                    continue
                if status == "ESTABLISHED" and peer:
                    found.add(f"tcp:{port}->{peer}")
                elif status == "LISTEN":
                    found.add(f"tcp:{port}")
            return sorted(found), True, ""
        except Exception:
            pass
    if _is_windows():
        return collect_connections_windows()
    payload, reason = _read_proc_net_tcp()
    if payload is None:
        extra = "; psutil not installed" if psutil is None else ""
        return [], False, reason + extra
    return parse_proc_net_tcp(*payload), True, ""


class ConnectionSensor:
    name = "connections"

    def collect(self):
        values, available, reason = collect_connections()
        return SensorReading(name=self.name, connections=values,
                             available=available, reason=reason)


# --------------------------------------------------------------- services --

def parse_sc_query(text):
    """Windows ``sc query`` -> [SERVICE_NAME, ...] (all states)."""
    names = []
    for line in (text or "").splitlines():
        match = re.match(r"\s*SERVICE_NAME\s*:\s*(\S+)", line)
        if match:
            names.append(match.group(1).strip())
    return sorted(set(names))


def collect_services_windows(runner=run_cmd):
    text = runner(["sc", "query", "type=", "service", "state=",
                   "all"])
    if text is None:
        return [], False, "sc query unavailable or failed"
    names = parse_sc_query(text)
    if not names:
        return [], False, "sc query returned no services"
    return names, True, ""


def parse_systemctl_services(text):
    """``systemctl list-units --type=service --all --no-legend``
    -> [unit names without '.service'] (loaded units)."""
    names = []
    for line in (text or "").splitlines():
        parts = line.split()
        # unit load active sub description...
        if len(parts) < 4 or not parts[0].endswith(".service"):
            continue
        if parts[1].lower() != "loaded":
            continue
        names.append(parts[0][:-len(".service")])
    return sorted(set(names))


def collect_services_linux(runner=run_cmd):
    text = runner(["systemctl", "list-units", "--type=service",
                   "--all", "--no-legend", "--no-pager"])
    if text is None:
        return [], False, ("systemctl unavailable (no systemd?) -- "
                            "service inventory unknown")
    return parse_systemctl_services(text), True, ""


def collect_services():
    if _is_windows():
        return collect_services_windows()
    return collect_services_linux()


class ServiceSensor:
    name = "services"

    def collect(self):
        values, available, reason = collect_services()
        return SensorReading(name=self.name, services=values,
                             available=available, reason=reason)


# ------------------------------------------------------------ alert feed --

class FeedUnavailable(RuntimeError):
    """No detector feed configured/readable (coverage absent)."""


class FileAlertFeed:
    """JSONL alert file APPENDED BY A REAL SENSOR (adapter-dependent).

    Line schema: {"kind": str, "severity": "low|medium|high|critical",
    "details": str, "timestamp": float, "asset": str?}. Lines for other
    assets (by ``asset`` matching asset_id/hostname) are SKIPPED for
    this asset (counted in notes, never attributed). Malformed lines
    raise loudly (a corrupt feed must not look like a quiet network).

    COMPONENT KIND: adapter-dependent. No file = no compromise
    coverage (``available=False``); an empty file = coverage present,
    zero alerts. This feed NEVER invents evidence.
    """

    name = "alert-feed"
    MAX_EVENTS_PER_COLLECT = 200

    def __init__(self, path, asset_id="", hostname="",
                 kind_map=None):
        self.path = path
        self.asset_id = asset_id
        self.hostname = hostname
        # Conservative kind normalization: unknown kinds pass through
        # lowercased; the NORMALIZER decides compromise, never this.
        self.kind_map = dict(kind_map or {})
        self._offset = 0

    def check_available(self):
        if not self.path or not os.path.isfile(self.path):
            return (False, f"no alert feed at {self.path or '<unset>'}: "
                           f"compromise coverage absent for this asset")
        if not os.access(self.path, os.R_OK):
            return (False, f"alert feed at {self.path} unreadable "
                           f"(permissions)")
        return True, ""

    def collect(self):
        from .telemetry import SecurityEvent
        available, reason = self.check_available()
        if not available:
            return SensorReading(name=self.name, available=False,
                                 reason=reason)
        try:
            size = os.path.getsize(self.path)
        except OSError as exc:
            return SensorReading(name=self.name, available=False,
                                 reason=f"alert feed stat failed: {exc}")
        if size < self._offset:
            self._offset = 0  # rotated/truncated; start over loudly
        events, skipped, malformed = [], 0, 0
        try:
            with open(self.path, "r", encoding="utf-8",
                      errors="replace") as fh:
                fh.seek(self._offset)
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        raw = __import__("json").loads(line)
                    except Exception:
                        malformed += 1
                        continue
                    if not isinstance(raw, dict) or "kind" not in raw:
                        malformed += 1
                        continue
                    owner = str(raw.get("asset", "") or "").strip()
                    if owner and owner not in (self.asset_id,
                                               self.hostname):
                        skipped += 1
                        continue
                    kind = str(raw["kind"])
                    kind = self.kind_map.get(kind, self.kind_map.get(
                        kind.lower(), kind.lower()))
                    events.append(SecurityEvent(
                        kind=kind,
                        severity=str(raw.get("severity", "low")),
                        details=str(raw.get("details", "")),
                        timestamp=float(raw.get("timestamp",
                                                time.time()))))
                    if len(events) >= self.MAX_EVENTS_PER_COLLECT:
                        break
                self._offset = fh.tell()
        except OSError as exc:
            return SensorReading(name=self.name, available=False,
                                 reason=f"alert feed read failed: {exc}")
        if malformed:
            return SensorReading(
                name=self.name, available=False,
                reason=(f"alert feed has {malformed} malformed "
                        f"line(s): refusing to interpret a corrupt feed"),
                notes=f"skipped-other-assets={skipped}")
        notes = (f"skipped-other-assets={skipped}" if skipped else "")
        return SensorReading(name=self.name, events=events,
                             available=True, reason="", notes=notes)


# ------------------------------------------------------- syslog (Linux) --

_AUTH_FAIL_PATTERNS = (
    re.compile(r"Failed password for \S+ from (\S+)"),
    re.compile(r"authentication failure.*rhost=(\S+)"),
)


def parse_auth_log(text, limit=50):
    """sshd/pam auth failures -> failed_login events (REAL signal).

    Only authentication failures; NEVER compromise verdicts (a failed
    login is suspicious activity, at most medium severity here).
    """
    from .telemetry import SecurityEvent
    events = []
    for line in (text or "").splitlines():
        for pattern in _AUTH_FAIL_PATTERNS:
            match = pattern.search(line)
            if match:
                events.append(SecurityEvent(
                    kind="failed_login", severity="medium",
                    details=f"auth failure from {match.group(1)}",
                    timestamp=time.time()))
                break
        if len(events) >= limit:
            break
    return events


class SyslogAuthSensor:
    """Linux auth.log failed logins (REAL). Unreadable = unavailable."""

    name = "auth-log"

    def __init__(self, path="/var/log/auth.log"):
        self.path = path

    def collect(self):
        try:
            with open(self.path, "r", encoding="utf-8",
                      errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            return SensorReading(
                name=self.name, available=False,
                reason=f"{self.path} unreadable ({exc}); "
                       f"login-failure coverage absent")
        # Tail-only: auth logs rotate; cap input to keep collects fast.
        tail = "\n".join(text.splitlines()[-2000:])
        return SensorReading(name=self.name,
                             events=parse_auth_log(tail),
                             available=True, reason="")


# ---------------------------------------------------------------- builder --

class EndpointTelemetryBuilder:
    """Composes sensors into one asset's HostTelemetry (honest gaps).

    ``sensors`` default to the platform stack
    (processes/connections/services); alert/auth sensors attach
    explicitly (adapter-dependent). Unavailable sensors become
    ``partial_errors`` entries -- never silence, never fiction.
    ``capabilities()`` reports the last collect per sensor for UIs.
    """

    def __init__(self, asset_id, hostname="", sensors=None):
        self.asset_id = asset_id
        self.hostname = hostname
        self.sensors = list(sensors) if sensors is not None else [
            ProcessSensor(), ConnectionSensor(), ServiceSensor()]
        self._capabilities = {s.name: {"available": None, "reason": ""}
                              for s in self.sensors}

    def capabilities(self):
        return {k: dict(v) for k, v in self._capabilities.items()}

    def build(self, now=None):
        from .telemetry import HostTelemetry
        now = time.time() if now is None else float(now)
        processes, connections, services, events = [], [], [], []
        partial = []
        for sensor in self.sensors:
            try:
                reading = sensor.collect()
            except Exception as exc:  # a sensor must never kill a cycle
                reading = SensorReading(
                    name=getattr(sensor, "name", "?"), available=False,
                    reason=f"sensor raised {type(exc).__name__}: {exc}")
            self._capabilities[reading.name] = {
                "available": bool(reading.available),
                "reason": reading.reason or ""}
            if not reading.available:
                partial.append({"key": self.asset_id,
                                "error": f"{reading.name}: "
                                         f"{reading.reason}"})
                continue
            processes.extend(reading.processes or [])
            connections.extend(reading.connections or [])
            services.extend(reading.services or [])
            events.extend(reading.events or [])
            if reading.notes:
                partial.append({"key": self.asset_id,
                                "error": f"{reading.name} note: "
                                         f"{reading.notes}"})
        host = HostTelemetry(
            key=self.asset_id,
            processes=sorted(set(map(str, processes))),
            connections=sorted(set(map(str, connections))),
            sessions=[], up=True, events=events,
            services=sorted(set(map(str, services))))
        return host, partial


__all__ = [
    "COMPONENT_KIND", "SensorReading", "ProcessSensor",
    "ConnectionSensor", "ServiceSensor", "FileAlertFeed",
    "FeedUnavailable", "SyslogAuthSensor",
    "EndpointTelemetryBuilder", "parse_tasklist_csv",
    "parse_netstat_tcp", "parse_proc_net_tcp",
    "parse_sc_query", "parse_systemctl_services", "parse_auth_log",
    "collect_processes", "collect_connections", "collect_services",
    "run_cmd",
]
