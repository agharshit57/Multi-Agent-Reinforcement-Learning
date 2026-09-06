"""Real network telemetry: collection interfaces and data model.

A deployment installation feeds this layer from EDR/syslog/NetFlow/etc.
Only the *shape* of the data is fixed here; vendor-specific parsing
belongs in a ``TelemetryCollector`` implementation. Ships with:
  - ``MockCollector``  : scripted timelines for demo/tests (no network).
  - ``FileCollector``  : replays JSON-lines telemetry captures.
  - ``LiveCollector``  : base class stub documenting the live contract
    (a real connector is a site-specific blocker, see README).

All timestamps are epoch seconds (float). Hosts are keyed by the real
asset ``ip`` (or ``hostname`` if ``ip`` is empty) matching the asset map.
"""

import json
import time
from dataclasses import dataclass, field


@dataclass
class SecurityEvent:
    kind: str            # e.g. "process", "connection", "ids_alert",
                         # "failed_login", "intrusion_confirmed"
    severity: str = "low"  # low | medium | high | critical
    details: str = ""
    timestamp: float = 0.0


@dataclass
class HostTelemetry:
    key: str             # asset-map lookup key (ip preferred)
    processes: list = field(default_factory=list)    # process names running
    connections: list = field(default_factory=list)  # "proto:port->peer" str
    sessions: list = field(default_factory=list)     # user session names
    up: bool = True
    events: list = field(default_factory=list)       # SecurityEvent list
    # Supervised service names (endpoint inventory). Folded into the
    # process baseline scope by the normalizer (an unexpected service
    # is activity, never a verdict). [] = unknown/absent, never quiet
    # by itself -- presence in the batch still governs freshness.
    services: list = field(default_factory=list)


@dataclass
class TelemetryBatch:
    timestamp: float
    hosts: dict          # key -> HostTelemetry
    notes: str = ""
    source: str = ""     # collector name that produced this batch
    partial_errors: list = field(default_factory=list)
    # Per-host translation failures (strings). A non-empty list means
    # the batch is PARTIAL: present hosts are usable, listed hosts are
    # unknown -- never silently treated as quiet (see normalizer).


class TelemetryCollector:
    """Source of TelemetryBatch objects. Subclass for live connectors.

    Contract:
      - return a TelemetryBatch per call while data exists;
      - return None ONLY at end-of-stream (batch collectors);
      - raise CollectorError (see live.py) on any failure. A live
        collector must NEVER return None to signal failure.
    """

    @property
    def exhausted(self):
        """True when end-of-stream was reached (batch collectors)."""
        return False

    def next_batch(self):
        raise NotImplementedError

    def close(self):
        pass


class MockCollector(TelemetryCollector):
    """Replays a scripted list of batches (demo/tests, deterministic)."""

    def __init__(self, batches):
        self._batches = list(batches)
        self._pos = 0

    def next_batch(self):
        if self._pos >= len(self._batches):
            return None
        batch = self._batches[self._pos]
        self._pos += 1
        return batch

    @property
    def exhausted(self):
        return self._pos >= len(self._batches)


class FileCollector(TelemetryCollector):
    """Replays a JSON-lines file; one TelemetryBatch per line."""

    def __init__(self, path):
        self._fh = open(path, "r", encoding="utf-8")
        self._eof = False

    @property
    def exhausted(self):
        return self._eof

    def next_batch(self):
        # Skip whitespace-only lines (trailing newlines, hand-edited
        # separators): they carry no telemetry and must not raise a
        # spurious stale incident. Genuinely malformed lines still fail
        # loudly below.
        while True:
            line = self._fh.readline()
            if not line:
                self._eof = True
                return None
            if line.strip():
                break
        try:
            raw = json.loads(line)
        except Exception as exc:
            # A corrupt capture line is a LOUD failure, not a quiet gap:
            # callers must not mistake it for a healthy network.
            from .live import CollectorError
            raise CollectorError(f"capture file has malformed line: "
                                 f"{exc}") from exc
        hosts = {}
        for key, h in raw.get("hosts", {}).items():
            if not isinstance(h, dict):
                from .live import CollectorError
                raise CollectorError(
                    f"capture file: host {key!r} entry is not an object")
            try:
                events = [SecurityEvent(**e) for e in h.get("events", [])]
            except Exception as exc:
                from .live import CollectorError
                raise CollectorError(
                    f"capture file: host {key!r} has malformed events: "
                    f"{exc}") from exc
            hosts[key] = HostTelemetry(
                key=key,
                processes=list(h.get("processes", [])),
                connections=list(h.get("connections", [])),
                sessions=list(h.get("sessions", [])),
                up=bool(h.get("up", True)),
                events=events,
                services=list(h.get("services", [])))
        return TelemetryBatch(timestamp=float(raw.get("timestamp",
                                                      time.time())),
                              hosts=hosts,
                              notes=str(raw.get("notes", "")),
                              source=str(raw.get("source", "")),
                              partial_errors=_coerce_partial_errors(
                                  raw.get("partial_errors", [])))

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _coerce_partial_errors(raw):
    """Capture files use [{"key","error"}]; coerce defensively."""
    if not isinstance(raw, list):
        from .live import CollectorError
        raise CollectorError("capture file: 'partial_errors' must be a list")
    out = []
    for item in raw:
        if isinstance(item, dict) and "key" in item:
            out.append({"key": str(item["key"]),
                        "error": str(item.get("error", ""))})
        else:
            from .live import CollectorError
            raise CollectorError(
                "capture file: partial_errors entries must be "
                "{'key','error'} objects")
    return out


class LiveCollector(TelemetryCollector):
    """Contract for a real connector (NOT implemented -- site-specific).

    A production implementation polls the site's EDR/SIEM/firewall APIs on
    each ``next_batch()`` call and translates the result into HostTelemetry
    keyed by asset-map keys. This is intentionally left abstract: API
    shapes, credentials, and polling cadence differ per environment.
    """

    def next_batch(self):
        raise NotImplementedError(
            "LiveCollector has no site connector configured. Use "
            "MockCollector/FileCollector, or subclass LiveCollector.")
