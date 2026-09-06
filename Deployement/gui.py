"""SOC-style deployment console (stdlib + numpy only, no training deps).

Layout: persistent left sidebar (Overview, Network, Threats, Agents,
Actions, Audit, Model, Telemetry, Settings) + header (mode banner,
global search, notifications, theme) + main workspace with one page
visible at a time.

Design rules:
  - All data shaping lives in pure ``*_model``/helper functions below
    (importable and unit-testable without a display); widgets only
    render what those return.
  - The console never invents model behavior: every number comes from
    the pipeline (records, normalized state, enforcement, config).
  - Safety gates are reused, never bypassed: approvals still flow
    through ``pipeline.approve_pending``/deny-pop, live mode still
    needs its confirmation, destructive actions stay queued.
  - Importing this module never touches the network, the model, or a
    display. ``Tk()`` is only constructed inside ``launch()``.

Back-compat names kept for existing callers/tests: ``launch``,
``DeploymentConsole``, ``mode_style``, ``MODE_STYLES``,
``QUICK_START_TEXT``, ``format_message``, ``summarize_decision``.
"""

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    _TK_AVAILABLE = True
except Exception:  # minimal/embedded interpreters
    tk = None
    filedialog = messagebox = ttk = None
    _TK_AVAILABLE = False

import csv
import time
from collections import deque

from .asset_map import match_zone
from .config import (AGENT_NAMES, AGENT_ZONES, COMMUNICATION_DIM,
                     COMMUNICATION_LATENT_DIM, MODES, NUM_AGENTS,
                     NUM_HOST_TARGETS, STABLE_HOST_INDEX)
from .observation import comms_policy_edges
from .validator import ActionValidator

# ---------------------------------------------------------------- helpers --
# Mode presentation (pure data -- safe to test without a display).
MODE_STYLES = {
    "shadow": {"bg": "#334155", "fg": "white",
               "label": "SHADOW",
               "blurb": "observe only -- nothing is enforced"},
    "mock": {"bg": "#92400e", "fg": "white",
             "label": "MOCK",
             "blurb": "simulated effects on internal state only"},
    "supervised": {"bg": "#b45309", "fg": "white",
                   "label": "SUPERVISED",
                   "blurb": "destructive actions need your approval"},
    "live": {"bg": "#b91c1c", "fg": "white",
             "label": "LIVE",
             "blurb": "APPROVED ACTIONS EXECUTE FOR REAL"},
}
_UNKNOWN_MODE_STYLE = {"bg": "#111827", "fg": "white",
                       "label": "?", "blurb": ""}

QUICK_START_TEXT = (
    "Quick start: 1) Rebuild  2) Start (auto) or Step (manual)  "
    "3) review Dashboard + Messages  "
    "4) approve/deny destructive actions in Approvals")


def mode_style(mode):
    """Presentation for a mode banner. Unknown modes get a neutral style
    (never crash the console on a config typo)."""
    return dict(MODE_STYLES.get(mode, _UNKNOWN_MODE_STYLE))


def format_message(message):
    if not message:
        return "-"
    parts = []
    for key in ("event_type", "target_type", "target_id", "threat_level",
                "status", "priority", "confidence"):
        parts.append(f"{key}={message.get(key)}")
    return " ".join(parts)


def summarize_decision(record):
    return (f"cyc={record.get('cycle')} {record.get('agent_id')}: "
            f"{record.get('label')} [{record.get('risk')}] "
            f"-> {record.get('operation')} ({record.get('status')})")


# ------------------------------------------------------------------ themes --
THEMES = {
    "dark": {
        "label": "Dark (SOC)",
        "style_theme": "clam",
        "page_bg": "#0f172a",
        "card_bg": "#1e293b",
        "text_fg": "#e2e8f0",
        "muted_fg": "#94a3b8",
        "accent": "#38bdf8",
        "good": "#4ade80",
        "warn": "#fbbf24",
        "bad": "#f87171",
        "banner_fg": "white",
        "entry_bg": "#0b1220",
        "table_bg": "#1e293b",
        "table_fg": "#e2e8f0",
        "mono_font": ("TkFixedFont", 9),
        "base_font": ("TkDefaultFont", 10),
        "title_font": ("TkDefaultFont", 12, "bold"),
        "big_font": ("TkDefaultFont", 22, "bold"),
    },
    "light": {
        "label": "Light",
        "style_theme": "clam",
        "page_bg": "#f1f5f9",
        "card_bg": "#ffffff",
        "text_fg": "#0f172a",
        "muted_fg": "#64748b",
        "accent": "#0284c7",
        "good": "#15803d",
        "warn": "#b45309",
        "bad": "#b91c1c",
        "banner_fg": "white",
        "entry_bg": "#ffffff",
        "table_bg": "#ffffff",
        "table_fg": "#0f172a",
        "mono_font": ("TkFixedFont", 9),
        "base_font": ("TkDefaultFont", 10),
        "title_font": ("TkDefaultFont", 12, "bold"),
        "big_font": ("TkDefaultFont", 22, "bold"),
    },
}
DEFAULT_THEME = "dark"


def apply_theme(root, name="dark"):
    """Apply a theme to a Tk root. Returns the palette dict.

    Raises ValueError on unknown theme names. Cosmetic failures (a
    minimal Tk without clam) fall back silently -- never crash setup.
    """
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; want one of "
                         f"{sorted(THEMES)}")
    palette = dict(THEMES[name])
    if ttk is None:
        return palette
    try:
        style = ttk.Style(root)
        if palette["style_theme"] in style.theme_names():
            style.theme_use(palette["style_theme"])
        style.configure("TFrame", background=palette["page_bg"])
        style.configure("Card.TFrame", background=palette["card_bg"])
        style.configure("TLabel", background=palette["page_bg"],
                        foreground=palette["text_fg"],
                        font=palette["base_font"])
        style.configure("Card.TLabel", background=palette["card_bg"],
                        foreground=palette["text_fg"])
        style.configure("Title.TLabel", background=palette["card_bg"],
                        foreground=palette["text_fg"],
                        font=palette["title_font"])
        style.configure("Muted.TLabel", background=palette["card_bg"],
                        foreground=palette["muted_fg"])
        style.configure("TButton", font=palette["base_font"])
        style.configure("TNotebook", background=palette["page_bg"])
        style.configure("Treeview", background=palette["table_bg"],
                        fieldbackground=palette["table_bg"],
                        foreground=palette["table_fg"],
                        rowheight=24, font=palette["base_font"])
        style.configure("Treeview.Heading", font=palette["title_font"])
        style.configure("TEntry", fieldbackground=palette["entry_bg"],
                        foreground=palette["text_fg"])
    except Exception:
        pass
    return palette


# ------------------------------------------------------------ model layer --
# Pure functions over a pipeline object (duck-typed: tests use stub
# pipelines). Everything here runs headless.

SEVERITY_ORDER = ("critical", "high", "medium", "low", "unknown", "none")

SEVERITY_FILTERS = ("all",) + SEVERITY_ORDER


def severity_rank(severity):
    """Lower sorts first (most severe). Unknown severities sort last."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return len(SEVERITY_ORDER)


def _normalized_hosts(pipeline):
    normalized = getattr(pipeline, "last_normalized", None)
    hosts = getattr(normalized, "hosts", None)
    return hosts if isinstance(hosts, dict) else {}


def _agent_of_cc4(asset_map, cc4):
    for agent_name in getattr(asset_map, "agents", {}):
        try:
            agent_id = int(str(agent_name).rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            continue
        hosts = asset_map.agents[agent_name].get("hosts", {})
        if cc4 in hosts:
            return agent_id, agent_name
    return None, ""


def _zone_of_cc4(cc4):
    try:
        return match_zone(cc4)
    except Exception:
        return "?"


def _host_display_status(host):
    if host is None:
        return "no data"
    if getattr(host, "compromised", False):
        return "Compromised"
    if getattr(host, "health", "") == "stale":
        return "Stale"
    if getattr(host, "process_event", False) or getattr(
            host, "connection_event", False):
        return "Active"
    return "Nominal"


def host_rows(pipeline):
    """One display row per bound host (112 in the default map)."""
    asset_map = getattr(pipeline, "asset_map", None)
    if asset_map is None:
        return []
    hosts = _normalized_hosts(pipeline)
    rows = []
    for agent_name in sorted(getattr(asset_map, "agents", {})):
        for cc4 in sorted(asset_map.agents[agent_name].get("hosts", {})):
            info = asset_map.agents[agent_name]["hosts"][cc4] or {}
            host = hosts.get(cc4)
            rows.append({
                "observed": host is not None,
                "cc4": cc4,
                "agent": agent_name,
                "agent_id": _agent_of_cc4(asset_map, cc4)[0],
                "zone": _zone_of_cc4(cc4),
                "ip": info.get("ip", ""),
                "hostname": info.get("hostname", ""),
                "role": info.get("role", ""),
                "health": getattr(host, "health", "unknown")
                if host is not None else "no data",
                "severity": getattr(host, "severity", "unknown")
                if host is not None else "unknown",
                "status": _host_display_status(host),
                "compromised": bool(getattr(host, "compromised", False)),
                "up": bool(getattr(host, "up", True)),
                "sessions": getattr(host, "session_count", 0),
                "notes": list(getattr(host, "notes", []) or []),
            })
    return rows


def overall_status(pipeline):
    """Security posture summary: level, headline, reasons.

    Levels: ``critical`` > ``attention`` > ``monitoring`` > ``idle``.
    Never claims "secure" -- the honest ceiling is "monitoring".
    """
    hosts = _normalized_hosts(pipeline)
    history = list(getattr(pipeline, "history", []) or [])
    health = getattr(getattr(pipeline, "health", None), "state", "ok")
    policy_error = getattr(pipeline, "last_policy_error", "") or ""
    compromised = sorted(
        cc4 for cc4, host in hosts.items()
        if getattr(host, "compromised", False))
    stale = sorted(
        cc4 for cc4, host in hosts.items()
        if getattr(host, "health", "") == "stale")
    pending = getattr(getattr(pipeline, "enforcement", None),
                      "pending_approvals", None) or []
    unseen = list(getattr(pipeline, "last_unseen", []) or [])
    reasons = []
    if compromised:
        reasons.append(f"{len(compromised)} compromised host(s): "
                       + ", ".join(compromised[:5])
                       + ("…" if len(compromised) > 5 else ""))
    if policy_error:
        reasons.append(f"policy error: {policy_error}")
    if health == "down":
        reasons.append("telemetry collector is DOWN")
    if stale:
        reasons.append(f"{len(stale)} host(s) with stale telemetry")
    if health == "degraded":
        reasons.append("telemetry collector is degraded")
    if pending:
        reasons.append(f"{len(pending)} action(s) awaiting approval")
    if unseen:
        reasons.append(f"{len(unseen)} unmapped telemetry source(s)")
    if compromised or policy_error or health == "down":
        level, headline = ("critical",
                           "CRITICAL — immediate attention required"
                           if compromised or health == "down"
                           else "CRITICAL — policy failure")
    elif reasons:
        level, headline = "attention", "ATTENTION — items need review"
    elif not hosts and not history:
        level, headline = "idle", "No data yet — press Step or Start"
    else:
        level, headline = "monitoring", "MONITORING — network quiet"
    return {"level": level, "headline": headline, "reasons": reasons,
            "compromised": compromised, "stale": stale}


def threat_list(pipeline, severity="all", query=""):
    """Hosts with threat evidence (severity != none), ranked + filtered."""
    query = (query or "").strip().lower()
    rows = []
    for row in host_rows(pipeline):
        # Unobserved hosts (no normalized data yet) are NOT threats --
        # only graded evidence counts.
        if not row["observed"] or row["severity"] == "none":
            continue
        if severity != "all" and row["severity"] != severity:
            continue
        if query:
            haystack = " ".join((row["cc4"], row["ip"], row["hostname"],
                                 " ".join(row["notes"]))).lower()
            if query not in haystack:
                continue
        status = ("active" if row["compromised"]
                  else ("stale" if row["health"] == "stale" else "watch"))
        rows.append({**row, "status": status})
    rows.sort(key=lambda r: (severity_rank(r["severity"]), r["cc4"]))
    return rows


def _record_mentions(record, cc4):
    label = str(record.get("label") or "")
    if cc4 and cc4 in label:
        return True
    message = record.get("message") or {}
    try:
        target_id = int(message.get("target_id", -1))
    except (TypeError, ValueError):
        target_id = -1
    if str(message.get("target_type", "")).upper() == "HOST":
        try:
            from .config import STABLE_HOST_LIST
            if 0 <= target_id < len(STABLE_HOST_LIST):
                return STABLE_HOST_LIST[target_id] == cc4
        except Exception:
            pass
    return False


def threat_detail(pipeline, cc4):
    """Detail bundle for one threatened host: evidence + timeline."""
    rows = {row["cc4"]: row for row in host_rows(pipeline)}
    host = rows.get(cc4)
    timeline = []
    for record in reversed(list(getattr(pipeline, "history", []) or [])):
        if _record_mentions(record, cc4):
            timeline.append({
                "cycle": record.get("cycle"),
                "timestamp": record.get("timestamp"),
                "agent_id": record.get("agent_id"),
                "label": record.get("label"),
                "status": record.get("status"),
                "operation": record.get("operation"),
                "risk": record.get("risk"),
            })
        if len(timeline) >= 20:
            break
    audit_hits = []
    enforcement = getattr(pipeline, "enforcement", None)
    for entry in list(getattr(enforcement, "audit", []) or []):
        if not isinstance(entry, dict):
            continue
        if entry.get("target") == cc4:
            audit_hits.append(entry)
    return {"host": host, "timeline": timeline, "audit": audit_hits[-20:]}


def _latest_record_for_agent(pipeline, agent_id):
    for record in reversed(list(getattr(pipeline, "history", []) or [])):
        if record.get("agent_id") == agent_id:
            return record
    return None


def confidence_of(message):
    """Human confidence text from a structured message (or em dash)."""
    try:
        value = float((message or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        return "—"
    if value <= 0:
        return "—"
    return f"{value:.0%}"


def _trust_summary(trust_row):
    try:
        values = [float(v) for v in (trust_row or [])]
    except (TypeError, ValueError):
        return "—"
    if not values:
        return "—"
    return f"avg {sum(values) / len(values):.2f} over {len(values)} peers"


def agent_rows(pipeline):
    """One row per blue agent: zones, hosts, health, latest decision."""
    hosts = _normalized_hosts(pipeline)
    rows = []
    for agent_id in range(NUM_AGENTS):
        zones = tuple(AGENT_ZONES.get(agent_id, ()))
        zone_hosts = [h for h in hosts.values()
                      if _zone_of_cc4_host(h) in zones]
        worst = "none"
        for host in zone_hosts:
            if severity_rank(getattr(host, "severity", "none")) < severity_rank(worst):
                worst = getattr(host, "severity", "none")
        latest = _latest_record_for_agent(pipeline, agent_id)
        message = (latest or {}).get("message") or {}
        rows.append({
            "agent_id": agent_id,
            "agent": f"blue_agent_{agent_id}",
            "zones": list(zones),
            "host_count": 16 * len(zones),
            "health": worst if worst != "none" else "nominal",
            "latest_label": (latest or {}).get("label", "—"),
            "latest_status": (latest or {}).get("status", "—"),
            "latest_cycle": (latest or {}).get("cycle", "—"),
            "confidence": confidence_of(message),
            "comm": (_trust_summary((latest or {}).get("trust_row"))
                     + (" | " + format_message(message) if message else "")),
        })
    return rows


def _zone_of_cc4_host(host):
    cc4 = getattr(host, "cc4", "") or ""
    return _zone_of_cc4(cc4)


def _zone_of_cc4(cc4):
    try:
        return match_zone(cc4)
    except Exception:
        return "?"


def _table_entry(pipeline, agent_id, action_index):
    try:
        tables = getattr(pipeline, "tables", None) or {}
        table = tables.get(agent_id, []) if isinstance(tables, dict) \
            else tables[agent_id]
        if 0 <= int(action_index) < len(table):
            return table[int(action_index)]
    except Exception:
        pass
    return None


def evidence_for(pipeline, agent_id, action_index):
    """Evidence notes behind one decision (empty when routine)."""
    entry = _table_entry(pipeline, agent_id, action_index)
    if not entry:
        return []
    target = entry.get("target")
    hosts = _normalized_hosts(pipeline)
    host = hosts.get(target)
    if host is None:
        return []
    return list(getattr(host, "notes", []) or [])


_HUMAN_VERBS = {
    "Sleep": "Hold position",
    "Monitor": "Monitor",
    "Analyse": "Investigate host",
    "Remove": "Remove malicious sessions",
    "Restore": "Restore host from backup",
    "DeployDecoy": "Deploy decoy",
    "AllowTrafficZone": "Allow zone traffic",
    "BlockTrafficZone": "Block zone traffic",
}


def action_rows(pipeline):
    """Latest cycle's decisions in human-readable form (no raw ids)."""
    history = list(getattr(pipeline, "history", []) or [])
    if not history:
        return []
    latest_cycle = max(r.get("cycle", 0) for r in history)
    rows = []
    for record in history:
        if record.get("cycle") != latest_cycle:
            continue
        agent_id = record.get("agent_id")
        entry = _table_entry(pipeline, agent_id, record.get("action_index"))
        target = (entry or {}).get("target", "")
        asset = (entry or {}).get("asset") or {}
        rows.append({
            "agent": f"blue_agent_{agent_id}",
            "action": _HUMAN_VERBS.get(record.get("command"),
                                       record.get("command") or "?"),
            "command": record.get("command"),
            "target": target or "—",
            "asset_ip": asset.get("ip", "") if isinstance(asset, dict)
            else "",
            "reason": "; ".join(evidence_for(
                pipeline, agent_id, record.get("action_index"))) or
            "routine monitoring",
            "confidence": confidence_of(record.get("message")),
            "risk": record.get("risk", "?"),
            "status": record.get("status", "?"),
            "needs_approval": bool(
                (record.get("approval") or {}).get("needed", False)),
            "record": record,
        })
    return rows


def pending_rows(pipeline):
    """Queued approvals in human-readable form (with table context)."""
    enforcement = getattr(pipeline, "enforcement", None)
    pending = list(getattr(enforcement, "pending_approvals", []) or [])
    rows = []
    for index, decision in enumerate(pending):
        if not isinstance(decision, dict):
            continue
        agent_id = decision.get("agent_id")
        entry = _table_entry(pipeline, agent_id,
                             decision.get("action_index"))
        rows.append({
            "index": index,
            "agent": f"blue_agent_{agent_id}",
            "action": _HUMAN_VERBS.get(decision.get("command"),
                                       decision.get("command") or "?"),
            "command": decision.get("command"),
            "target": (entry or {}).get("target")
            or decision.get("target", ""),
            "operation": decision.get("operation"),
            "label": (entry or {}).get("label")
            or decision.get("label", ""),
            "attempts": decision.get("attempts", 0),
            "decision": decision,
        })
    return rows


def notification_list(pipeline):
    """Operator notifications, most severe first (capped at 50)."""
    status = overall_status(pipeline)
    items = []
    for cc4 in status["compromised"]:
        items.append({"level": "critical", "kind": "threat",
                      "text": f"Host compromised: {cc4}", "ref": cc4})
    if getattr(pipeline, "last_policy_error", ""):
        items.append({"level": "critical", "kind": "policy",
                      "text": f"Policy error: {pipeline.last_policy_error}",
                      "ref": ""})
    health = getattr(getattr(pipeline, "health", None), "state", "ok")
    if health == "down":
        items.append({"level": "critical", "kind": "telemetry",
                      "text": "Telemetry collector is DOWN", "ref": ""})
    pending = getattr(getattr(pipeline, "enforcement", None),
                      "pending_approvals", None) or []
    if pending:
        items.append({"level": "warning", "kind": "approvals",
                      "text": f"{len(pending)} action(s) awaiting approval",
                      "ref": ""})
    if status["stale"]:
        items.append({"level": "warning", "kind": "telemetry",
                      "text": f"{len(status['stale'])} host(s) with stale "
                              f"telemetry", "ref": ""})
    if health == "degraded":
        items.append({"level": "warning", "kind": "telemetry",
                      "text": "Telemetry collector is degraded", "ref": ""})
    unseen = list(getattr(pipeline, "last_unseen", []) or [])
    if unseen:
        items.append({"level": "info", "kind": "telemetry",
                      "text": f"{len(unseen)} unmapped telemetry source(s)",
                      "ref": ""})
    order = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda item: order.get(item["level"], 3))
    return items[:50]


def search(pipeline, query):
    """Global search across hosts, agents, threats, audit, and pages."""
    query = (query or "").strip().lower()
    if not query:
        return []
    results = []
    for row in host_rows(pipeline):
        haystack = " ".join((row["cc4"], row["ip"], row["hostname"])).lower()
        if query in haystack:
            results.append({"kind": "host",
                            "title": row["hostname"] or row["cc4"],
                            "subtitle": f"{row['cc4']} · {row['status']}",
                            "ref": row["cc4"]})
    for row in agent_rows(pipeline):
        haystack = " ".join([row["agent"], *row["zones"]]).lower()
        if query in haystack:
            results.append({"kind": "agent", "title": row["agent"],
                            "subtitle": ", ".join(row["zones"]),
                            "ref": row["agent_id"]})
    for row in threat_list(pipeline, query=query):
        results.append({"kind": "threat", "title": row["cc4"],
                        "subtitle": f"{row['severity']} · {row['status']}",
                        "ref": row["cc4"]})
    enforcement = getattr(pipeline, "enforcement", None)
    for entry in list(getattr(enforcement, "audit", []) or []):
        if not isinstance(entry, dict):
            continue
        haystack = " ".join(str(entry.get(key, "")) for key in
                            ("operation", "target", "details", "error",
                             "mode", "event")).lower()
        if query in haystack:
            results.append(
                {"kind": "audit",
                 "title": f"{entry.get('event')}: {entry.get('operation')}",
                 "subtitle": str(entry.get("target", "")),
                 "ref": entry})
    for page in ("Overview", "Network", "Threats", "Agents", "Actions",
                 "Audit", "Model", "Telemetry", "Settings"):
        if query in page.lower():
            results.append({"kind": "tab", "title": f"Go to {page}",
                            "subtitle": "page", "ref": page})
    return results[:50]


def telemetry_model(pipeline, event_log=None, now=None):
    """Collector + freshness snapshot for the Telemetry page."""
    import time as _time
    health = getattr(pipeline, "health", None)
    batch = getattr(pipeline, "last_batch", None)
    hosts = _normalized_hosts(pipeline)
    stale = sorted(cc4 for cc4, host in hosts.items()
                   if getattr(host, "health", "") == "stale")
    events_last_batch = 0
    partial_errors = []
    source, batch_ts = "", None
    if batch is not None:
        source = getattr(batch, "source", "") or ""
        batch_ts = getattr(batch, "timestamp", None)
        partial_errors = list(getattr(batch, "partial_errors", []) or [])
        for tele in (getattr(batch, "hosts", {}) or {}).values():
            events_last_batch += len(getattr(tele, "processes", []) or [])
            events_last_batch += len(getattr(tele, "connections", []) or [])
            events_last_batch += len(getattr(tele, "events", []) or [])
    rate = None
    if event_log:
        moment = now if now is not None else _time.time()
        window = [(ts, count) for ts, count in event_log
                  if moment - ts <= 300]
        span = (max(ts for ts, _ in window)
                - min(ts for ts, _ in window)) if len(window) > 1 else 0
        total = sum(count for _, count in window)
        rate = (total / span) if span > 0 else None
    degraded = []
    if getattr(health, "state", "ok") != "ok":
        degraded.append("collector "
                        f"({getattr(health, 'state', '?')})")
    if stale:
        degraded.append(f"{len(stale)} stale host(s)")
    if partial_errors:
        degraded.append(f"{len(partial_errors)} partial error(s)")
    return {
        "state": getattr(health, "state", "ok"),
        "consecutive_failures": getattr(health, "consecutive_failures", 0),
        "last_error": getattr(health, "last_error", ""),
        "last_success_ts": getattr(health, "last_success_ts", 0.0),
        "total_batches": getattr(health, "total_batches", 0),
        "total_failures": getattr(health, "total_failures", 0),
        "last_update_ts": batch_ts,
        "source": source,
        "events_last_batch": events_last_batch,
        "events_per_sec": rate,
        "partial_errors": partial_errors,
        "stale_hosts": stale,
        "unseen_keys": list(getattr(pipeline, "last_unseen", []) or []),
        "degraded": degraded,
    }


def model_page_model(pipeline):
    """Model/inference status for the Model page."""
    from .config import NUM_HOST_TARGETS
    policy = getattr(pipeline, "policy", None)
    info = dict(getattr(pipeline, "policy_info", None) or {})
    engine = getattr(policy, "name", None) or type(policy).__name__ \
        if policy is not None else "none"
    checkpoint = info.get("checkpoint")
    tables = getattr(pipeline, "tables", None) or {}
    action_sizes = {}
    for agent_id in range(NUM_AGENTS):
        table = tables.get(agent_id) if isinstance(tables, dict) \
            else tables[agent_id]
        action_sizes[agent_id] = len(table or [])
    obs_row = None
    last_obs = getattr(pipeline, "last_obs", None)
    if last_obs is not None:
        try:
            obs_row = [int(len(row)) for row in last_obs]
        except TypeError:
            obs_row = None
    history = list(getattr(pipeline, "history", []) or [])
    latencies = [r.get("latency_ms") for r in history
                 if isinstance(r.get("latency_ms"), (int, float))]
    errors = []
    if getattr(pipeline, "last_policy_error", ""):
        errors.append(str(pipeline.last_policy_error))
    for record in reversed(history):
        status = str(record.get("status", ""))
        if status.startswith("rejected") or status.startswith("failed"):
            errors.append(f"cycle {record.get('cycle')} agent "
                          f"{record.get('agent_id')}: {status}")
        if len(errors) >= 5:
            break
    contract = int(info.get("num_host_targets") or NUM_HOST_TARGETS)
    return {
        "engine": engine,
        "checkpoint": checkpoint or "mock policy (no weights)",
        "checkpoint_loaded": bool(checkpoint),
        "vocab_host": contract,
        "vocab_match": contract == NUM_HOST_TARGETS,
        "obs_dims": "92 (agents 0-3 native), 210 (agent 4 / model rows)",
        "obs_rows": obs_row,
        "action_sizes": action_sizes,
        "latency_avg_ms": (sum(latencies) / len(latencies)
                           if latencies else None),
        "latency_max_ms": (max(latencies) if latencies else None),
        "errors": errors,
        "comm_dim": COMMUNICATION_DIM,
        "comm_latent_dim": COMMUNICATION_LATENT_DIM,
        "comm_fields": 7,
    }


def audit_rows(entries, query="", event="all"):
    """Filter audit entries (newest first). Tolerant of odd entries."""
    query = (query or "").strip().lower()
    rows = []
    for entry in reversed(list(entries or [])):
        if not isinstance(entry, dict):
            continue
        if event != "all" and entry.get("event") != event:
            continue
        if query:
            haystack = " ".join(str(entry.get(key, "")) for key in
                                ("event", "operation", "target", "details",
                                 "error", "mode", "approver",
                                 "backend")).lower()
            if query not in haystack:
                continue
        rows.append(entry)
    return rows


def export_audit(entries, path, format="csv"):
    """Write filtered audit entries to ``path`` (csv or json).

    Returns the path. Raises ValueError on unknown formats and lets
    OSError propagate (callers surface it; never silently drop audit).
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    columns = ("audit_ts", "event", "operation", "target", "mode",
               "applied", "approver", "attempts", "backend", "details",
               "error")
    if format == "json":
        import json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, default=str)
        return path
    if format == "csv":
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns),
                                    extrasaction="ignore")
            writer.writeheader()
            for entry in entries:
                writer.writerow({key: entry.get(key, "") for key in columns})
        return path
    raise ValueError(f"unknown export format {format!r}; want csv or json")


# ------------------------------------------------------- topology model ----
SUBNET_POSITIONS = {
    # 3x3 alphabetical grid + agent badge strip; fixed canvas 620x380.
    "admin_network_subnet": (110, 70),
    "contractor_network_subnet": (310, 70),
    "internet_subnet": (510, 70),
    "office_network_subnet": (110, 170),
    "operational_zone_a_subnet": (310, 170),
    "operational_zone_b_subnet": (510, 170),
    "public_access_zone_subnet": (110, 270),
    "restricted_zone_a_subnet": (310, 270),
    "restricted_zone_b_subnet": (510, 270),
}
SUBNET_NODE_RADIUS = 30
AGENT_BADGE_Y = 345
AGENT_BADGE_RADIUS = 14


def topology_layout(width=620, height=380):
    """Static node geometry; data overlays are drawn per refresh."""
    from .config import AGENT_ZONES
    agents = {}
    for agent_id, zones in AGENT_ZONES.items():
        agents[agent_id] = (70 + agent_id * 120, AGENT_BADGE_Y)
    return {"width": width, "height": height,
            "subnets": dict(SUBNET_POSITIONS), "agents": agents}


def topology_hit(layout, x, y):
    """Hit test: agent badges first, then subnets. Returns
    ("agent", id) | ("subnet", name) | None. Pure + headless-safe."""
    for agent_id, (ax, ay) in layout.get("agents", {}).items():
        if (x - ax) ** 2 + (y - ay) ** 2 <= AGENT_BADGE_RADIUS ** 2:
            return ("agent", agent_id)
    for name, (sx, sy) in layout.get("subnets", {}).items():
        if (x - sx) ** 2 + (y - sy) ** 2 <= SUBNET_NODE_RADIUS ** 2:
            return ("subnet", name)
    return None


# ------------------------------------------------------- console shell --
PAGES = ("Overview", "Network", "Threats", "Agents", "Actions", "Audit",
         "Model", "Telemetry", "Settings")


class DeploymentConsole:
    """SOC console bound to a pipeline factory.

    ``build_pipeline`` is a zero-arg callable returning
    ``(pipeline, describe_dict)`` so asset/checkpoint/mode changes take
    effect on (re)start without restarting the app.
    """

    def __init__(self, root, build_pipeline):
        if not _TK_AVAILABLE:
            raise RuntimeError("tkinter is not available in this Python")
        self.root = root
        self.build_pipeline = build_pipeline
        self.pipeline = None
        self.pipeline_info = {}
        self.running = False
        self.last_records = []
        self._active_mode = "shadow"
        self._live_confirmed = False
        self._theme = DEFAULT_THEME
        self._palette = dict(THEMES[DEFAULT_THEME])
        self._pages = {}
        self._nav_buttons = {}
        self._modals = []
        self._event_log = deque(maxlen=120)
        self._build_widgets()
        self._update_banner()
        self.log("console ready (pipeline starts on demand)")

    # ------------------------------------------------------------ chrome --
    def _build_widgets(self):
        self.root.title("Cyber MARL — SOC Console")
        self.root.geometry("1280x800")
        self._palette = apply_theme(self.root, self._theme)

        # Header: mode banner + search + notifications + theme.
        header = ttk.Frame(self.root, padding=(8, 6))
        header.pack(side="top", fill="x")
        self.banner_mode_var = tk.StringVar(value="")
        self.banner_health_var = tk.StringVar(value="")
        self._banner_mode_label = tk.Label(
            header, textvariable=self.banner_mode_var,
            font=("TkDefaultFont", 14, "bold"),
            padx=12, pady=6, anchor="w")
        self._banner_mode_label.pack(side="left", fill="x", expand=True)
        self._banner_health_label = tk.Label(
            header, textvariable=self.banner_health_var,
            font=("TkDefaultFont", 10),
            padx=12, pady=6, anchor="e")
        self._banner_health_label.pack(side="left")

        search_frame = ttk.Frame(header)
        search_frame.pack(side="left", padx=8)
        ttk.Label(search_frame, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var,
                                 width=28)
        search_entry.pack(side="left", padx=(4, 0))
        search_entry.bind("<Return>", lambda _e: self._open_search())
        self._tooltip(search_entry,
                      "Search hosts, agents, threats, audit events, pages.\n"
                      "Enter opens results; double-click jumps to an item.")
        ttk.Button(search_frame, text="Go",
                   command=self._open_search).pack(side="left", padx=(4, 0))

        self._bell_var = tk.StringVar(value="Alerts: 0")
        bell = ttk.Button(header, textvariable=self._bell_var,
                          width=7, command=self._open_notifications)
        bell.pack(side="left", padx=4)
        self._tooltip(bell, "Notifications: critical threats, pending\n"
                            "approvals, telemetry problems.")
        theme_btn = ttk.Button(header, text="◐ Theme",
                               command=self._toggle_theme, width=8)
        theme_btn.pack(side="left", padx=4)
        self._tooltip(theme_btn, "Switch dark / light theme")

        ttk.Label(self.root, text=QUICK_START_TEXT,
                  padding=(8, 2)).pack(side="top", fill="x")

        # Bound here (not in __init__): StringVar needs a live Tk root.
        self.mode_var = tk.StringVar(value="shadow")

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # Sidebar: one button per page + footer.
        sidebar = ttk.Frame(body, padding=6, width=150)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="SOC CONSOLE",
                  font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 8))
        for index, page in enumerate(PAGES):
            btn = ttk.Button(sidebar, text=f"{index + 1}. {page}", width=16,
                             command=lambda p=page: self.show_page(p))
            btn.pack(pady=2, fill="x")
            self._nav_buttons[page] = btn
            self._tooltip(btn, f"Go to {page} (Alt+{index + 1})")
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(sidebar, text="Start",
                   command=self.start).pack(pady=2, fill="x")
        ttk.Button(sidebar, text="Pause",
                   command=self.pause).pack(pady=2, fill="x")
        ttk.Button(sidebar, text="Step",
                   command=self.step_once).pack(pady=2, fill="x")
        ttk.Button(sidebar, text="Rebuild",
                   command=self.rebuild).pack(pady=2, fill="x")
        self.version_var = tk.StringVar(value="deployment v0.1 · headless OK")
        ttk.Label(sidebar, textvariable=self.version_var,
                  wraplength=140).pack(side="bottom", pady=8)

        # Workspace: one frame per page, exactly one visible.
        workspace = ttk.Frame(body, padding=6)
        workspace.pack(side="left", fill="both", expand=True)
        for page in PAGES:
            frame = ttk.Frame(workspace, padding=6)
            self._pages[page] = frame
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w").pack(side="bottom", fill="x")
        self._build_pages()
        self.show_page("Overview")

        # Keyboard: Alt+1..9 switch pages, Escape closes popups.
        for index in range(min(9, len(PAGES))):
            self.root.bind(f"<Alt-KeyPress-{index + 1}>",
                           lambda _e, p=PAGES[index]: self.show_page(p))
        self.root.bind("<Escape>", lambda _e: self._close_top_modal())

    # ---------------------------------------------------------- navigation --
    def show_page(self, name):
        """Show one workspace page (predictable 1-click navigation)."""
        if name not in self._pages:
            return
        for page_name, frame in self._pages.items():
            if page_name == name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        for page_name, btn in self._nav_buttons.items():
            try:
                btn.state(["pressed" if page_name == name else "!pressed"])
            except Exception:
                pass
        refresher = getattr(self, f"_refresh_{name.lower()}_page", None)
        if callable(refresher):
            try:
                refresher()
            except Exception as exc:
                self.log(f"[ERROR] refresh {name}: {exc}")
        self._current_page = name

    # ------------------------------------------------------------- control --
    def _guarded(self, label, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # never crash the console
            self.log(f"[ERROR] {label}: {exc}")
            self.status_var.set(f"error: {label}: {exc}")
            return None

    def rebuild(self):
        self._guarded("rebuild", self._rebuild)

    def _rebuild(self):
        if self.pipeline is not None:
            self.pipeline.close()
        self.pipeline, self.pipeline_info = self.build_pipeline()
        self.pipeline.set_mode(self.mode_var.get())
        self._active_mode = self.mode_var.get()
        self._update_banner()
        self.last_records = []
        self._event_log.clear()
        self._refresh_static()
        self.status_var.set(
            f"rebuilt: {self.pipeline_info.get('policy')} "
            f"| mode={self.mode_var.get()} "
            f"| assets={self.pipeline_info.get('assets')}")
        self.log(f"pipeline rebuilt: {self.pipeline_info}")
        self._refresh_all()

    def start(self):
        if self.pipeline is None:
            self.rebuild()
            if self.pipeline is None:
                return
        self.running = True
        self.status_var.set("running")
        self._tick()

    def pause(self):
        self.running = False
        self.status_var.set("paused")

    def step_once(self):
        if self.pipeline is None:
            self.rebuild()
            if self.pipeline is None:
                return
        self._guarded("step", self._do_step)

    def _tick(self):
        if not self.running:
            return
        records = self._guarded("cycle", self._do_step)
        if records is not None and not records:
            self.log("collector exhausted -- stopping")
            self.pause()
            return
        self.root.after(800, self._tick)

    def _do_step(self):
        records = self.pipeline.step()
        if records:
            self.last_records = records
            self._refresh_dynamic(records)
            health = self.pipeline.health.state
            stale = sum(1 for r in records if r.get("stale_hosts"))
            self.status_var.set(
                f"cycle {self.pipeline.cycle} "
                f"({len(records)} decisions) "
                f"health={health} stale_hosts_seen={stale}")
            self._update_banner(
                f"health: {health}  |  cycle {self.pipeline.cycle}")
        return records

    def _on_mode_change(self):
        new_mode = self.mode_var.get()
        if new_mode == "live" and not self._live_confirmed:
            proceed = self._guarded(
                "live confirm",
                lambda: messagebox.askyesno(
                    "Confirm LIVE mode",
                    "LIVE mode executes approved destructive actions "
                    "FOR REAL via the configured backend.\n\n"
                    "Continue in LIVE mode?"))
            if not proceed:
                self.mode_var.set(self._active_mode)
                return
            self._live_confirmed = True
            self.log("live mode explicitly confirmed by operator")
        if new_mode != "live":
            self._live_confirmed = False
        self._active_mode = new_mode
        self._update_banner()
        if self.pipeline is not None:
            self._guarded("mode change",
                          self.pipeline.set_mode, new_mode)

    def _toggle_theme(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self._palette = apply_theme(self.root, self._theme)
        self._recolor_chrome()
        self.show_page(getattr(self, "_current_page", "Overview"))

    def _recolor_chrome(self):
        style = mode_style(self.mode_var.get()
                           if hasattr(self, "mode_var") else "?")
        for widget in (getattr(self, "_banner_mode_label", None),
                       getattr(self, "_banner_health_label", None)):
            if widget is None:
                continue
            try:
                widget.configure(bg=style["bg"], fg=style["fg"])
            except Exception:
                pass

    def _update_banner(self, health=""):
        """Refresh the mode banner (safe to call before any pipeline)."""
        style = mode_style(self.mode_var.get()
                           if hasattr(self, "mode_var") else "?")
        self.banner_mode_var.set(f"MODE: {style['label']} — {style['blurb']}")
        self._recolor_chrome()
        if health:
            self.banner_health_var.set(str(health))
        self._refresh_bell()

    def _refresh_bell(self):
        try:
            count = len(notification_list(self.pipeline)) \
                if self.pipeline is not None else 0
        except Exception:
            count = 0
        self._bell_var.set(f"Alerts: {count}")

    # ------------------------------------------------------ widget helpers --
    def _tooltip(self, widget, text):
        """Minimal hover tooltip (display-only; safe headless import)."""
        tip = {"window": None}

        def show(_event=None):
            try:
                if tip["window"] is not None:
                    return
                window = tk.Toplevel(widget)
                window.wm_overrideredirect(True)
                window.wm_geometry(
                    f"+{widget.winfo_rootx() + 16}"
                    f"+{widget.winfo_rooty() + 16}")
                tk.Label(window, text=text, justify="left",
                         background="#111827", foreground="white",
                         relief="solid", borderwidth=1,
                         padx=6, pady=4).pack()
                tip["window"] = window
            except Exception:
                pass

        def hide(_event=None):
            try:
                if tip["window"] is not None:
                    tip["window"].destroy()
            except Exception:
                pass
            tip["window"] = None

        try:
            widget.bind("<Enter>", show)
            widget.bind("<Leave>", hide)
        except Exception:
            pass

    def _close_top_modal(self):
        modals = getattr(self, "_modals", None) or []
        while modals:
            window = modals.pop()
            try:
                if window.winfo_exists():
                    window.destroy()
                    return
            except Exception:
                continue

    def _track_modal(self, window):
        if not hasattr(self, "_modals") or self._modals is None:
            self._modals = []
        self._modals.append(window)
        try:
            window.protocol(
                "WM_DELETE_WINDOW",
                lambda: (self._untrack_modal(window), window.destroy()))
        except Exception:
            pass
        return window

    def _untrack_modal(self, window):
        try:
            self._modals.remove(window)
        except (AttributeError, ValueError):
            pass

    def _modal(self, title, width=640, height=480):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(f"{width}x{height}")
        window.transient(self.root)
        self._track_modal(window)
        window.bind("<Escape>", lambda _e: (
            self._untrack_modal(window), window.destroy()))
        return window

    def _card(self, parent, title):
        frame = ttk.LabelFrame(parent, text=f" {title} ", padding=8)
        frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        return frame

    def _make_table_tab(self, parent, columns):
        """Table widget pack; returns the frame (frame.tree = widget)."""
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="w")
        tree.pack(fill="both", expand=True)
        frame.tree = tree
        frame.pack(fill="both", expand=True)
        return frame

    def _make_text_tab(self, parent):
        frame = ttk.Frame(parent)
        text = tk.Text(frame, wrap="word", height=20)
        text.pack(fill="both", expand=True)
        frame.text = text
        frame.pack(fill="both", expand=True)
        return frame

    def _fill_table(self, tree, columns, rows, empty_text="(no data yet)"):
        try:
            for row in tree.get_children():
                tree.delete(row)
        except Exception:
            return
        if not rows:
            tree.insert("", "end",
                        values=(empty_text,) + ("",) * (len(columns) - 1))
            return
        for row in rows:
            try:
                tree.insert("", "end", values=tuple(row))
            except Exception:
                continue

    def _set_text(self, tab, content):
        try:
            tab.text.delete("1.0", "end")
            tab.text.insert("end", content)
        except Exception:
            pass

    def log(self, message):
        try:
            self._tab_log.text.insert("end", message + "\n")
            self._tab_log.text.see("end")
        except Exception:
            pass

    def close(self):
        self.running = False
        if self.pipeline is not None:
            try:
                self.pipeline.close()
            except Exception:
                pass

    # ------------------------------------------------------- page: common --
    def _refresh_all(self):
        self._refresh_bell()
        current = getattr(self, "_current_page", "Overview")
        self.show_page(current)

    def _refresh_dynamic(self, records):
        # Event-rate sample, then refresh whatever is visible.
        try:
            batch = getattr(self.pipeline, "last_batch", None)
            count = 0
            if batch is not None:
                for tele in (getattr(batch, "hosts", {}) or {}).values():
                    count += len(getattr(tele, "processes", []) or [])
                    count += len(getattr(tele, "connections", []) or [])
                    count += len(getattr(tele, "events", []) or [])
            self._event_log.append((time.time(), count))
        except Exception:
            pass
        self._refresh_all()

    def _refresh_static(self):
        # Assets live on the Network page now; kept as a harmless alias
        # for any external caller of the old tabbed layout.
        try:
            self._refresh_network_page()
        except Exception:
            pass

    def _telemetry_snapshot(self):
        model = telemetry_model(self.pipeline)
        lines = [
            f"collector state : {model['state']}",
            f"batches/failures: {model['total_batches']} / "
            f"{model['total_failures']}",
            f"last error      : {model['last_error'] or '—'}",
            f"stale hosts     : {len(model['stale_hosts'])}",
            f"unseen sources  : {len(model['unseen_keys'])}",
        ]
        return "\n".join(lines)

    def _obs_snapshot(self):
        model = model_page_model(self.pipeline)
        lines = [
            f"engine          : {model['engine']}",
            f"checkpoint      : {model['checkpoint']}",
            f"obs rows        : {model['obs_rows'] or 'no cycle run yet'}",
            f"latency avg/max : {model['latency_avg_ms']} / "
            f"{model['latency_max_ms']} ms",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------- Overview --
    def _build_overview_page(self):
        page = self._pages["Overview"]
        self._ov_headline = tk.StringVar(value="No data yet")
        headline = tk.Label(page, textvariable=self._ov_headline,
                            font=("TkDefaultFont", 18, "bold"), anchor="w")
        headline.pack(fill="x", padx=4, pady=(0, 4))
        self._ov_reasons = tk.Text(page, wrap="word", height=4)
        self._ov_reasons.pack(fill="x", padx=4, pady=(0, 4))
        cards = ttk.Frame(page)
        cards.pack(fill="x", padx=0, pady=4)
        self._ov_cards = {}
        for key, title in (("hosts", "Hosts"), ("threats", "Threats"),
                           ("agents", "Agents"), ("approvals", "Approvals")):
            frame = self._card(cards, title)
            var = tk.StringVar(value="—")
            tk.Label(frame, textvariable=var,
                     font=("TkDefaultFont", 16, "bold"),
                     anchor="w").pack(fill="x")
            sub = tk.StringVar(value="")
            tk.Label(frame, textvariable=sub, anchor="w",
                     wraplength=220, justify="left").pack(fill="x")
            self._ov_cards[key] = (var, sub)
        mid = ttk.Frame(page)
        mid.pack(fill="both", expand=True)
        net_card = self._card(mid, "Network snapshot")
        self._ov_chips = ttk.Frame(net_card)
        self._ov_chips.pack(fill="x")
        self._ov_chip_buttons = {}
        act_card = self._card(mid, "Recent activity")
        self._ov_recent = self._make_table_tab(
            act_card, ("cycle", "agent", "action", "status"))
        attn_card = self._card(mid, "Needs attention")
        self._ov_attention = tk.Text(attn_card, wrap="word", height=8)
        self._ov_attention.pack(fill="both", expand=True)

    def _refresh_overview_page(self):
        status = overall_status(self.pipeline)
        level_colors = {"critical": "#b91c1c", "attention": "#b45309",
                        "monitoring": "#15803d", "idle": "#475569"}
        self._ov_headline.set(status["headline"])
        try:
            for widget in self._pages["Overview"].winfo_children():
                pass
        except Exception:
            pass
        self._ov_reasons.delete("1.0", "end")
        self._ov_reasons.insert(
            "end", "\n".join(status["reasons"]) or "Nothing to report.")
        try:
            self._ov_headline_master = getattr(self, "_ov_headline_master",
                                               None)
        except Exception:
            pass
        hosts = host_rows(self.pipeline)
        nominal = sum(1 for h in hosts if h["status"] == "Nominal")
        by_sev = {}
        for h in hosts:
            if h["severity"] != "none":
                by_sev[h["severity"]] = by_sev.get(h["severity"], 0) + 1
        agents = agent_rows(self.pipeline)
        worst_agent = "—"
        if agents:
            worst_agent = sorted(
                agents,
                key=lambda a: severity_rank(
                    "nominal" if a["health"] == "nominal" else a["health"])
            )[0]["health"]
        pending = len(getattr(getattr(self.pipeline, "enforcement", None),
                              "pending_approvals", None) or [])
        cards = {
            "hosts": (f"{nominal}/{len(hosts)} nominal",
                      f"{len(hosts)} bound hosts"),
            "threats": (f"{sum(by_sev.values())} active",
                        ", ".join(f"{k}: {v}"
                                  for k, v in sorted(by_sev.items())) or
                        "no active threats"),
            "agents": (f"{len(agents)} agents",
                       f"worst health: {worst_agent}"),
            "approvals": (f"{pending} pending",
                          "see Actions page" if pending else "queue empty"),
        }
        for key, (big, small) in cards.items():
            var, sub = self._ov_cards[key]
            var.set(big)
            sub.set(small)
        for widget in list(self._ov_chips.winfo_children()):
            widget.destroy()
        self._ov_chip_buttons = {}
        subnet_health = {}
        for h in hosts:
            subnet_health[h["zone"]] = min(
                subnet_health.get(h["zone"], 5),
                severity_rank(h["severity"]))
        for zone in sorted(subnet_health):
            short = zone.replace("_subnet", "").replace("_", " ")
            btn = ttk.Button(
                self._ov_chips, text=f"{short}",
                command=lambda z=zone: self._goto_subnet(z))
            btn.pack(side="left", padx=2, pady=2)
            self._ov_chip_buttons[zone] = btn
        history = list(getattr(self.pipeline, "history", []) or [])
        self._fill_table(
            self._ov_recent.tree, ("cycle", "agent", "action", "status"),
            [(r.get("cycle"), r.get("agent_id"), r.get("label"),
              r.get("status")) for r in history[-8:]],
            empty_text="(no cycles yet — press Step or Start)")
        notes = notification_list(self.pipeline)[:6]
        self._ov_attention.delete("1.0", "end")
        if notes:
            self._ov_attention.insert(
                "end", "\n".join(
                    f"[{n['level'].upper()}] {n['text']}" for n in notes))
        else:
            self._ov_attention.insert("end", "Nothing needs attention.")
        try:
            self._banner_health_var.set(
                f"health: {getattr(getattr(self.pipeline, 'health', None), 'state', '?')}  |  "
                f"cycle {getattr(self.pipeline, 'cycle', 0)}")
        except Exception:
            pass
        _ = level_colors  # reserved for per-level accents

    def _goto_subnet(self, zone):
        self._selected = ("subnet", zone)
        self.show_page("Network")

    # ------------------------------------------------------------- Network --
    def _build_network_page(self):
        from .config import SUBNETS
        page = self._pages["Network"]
        left = ttk.Frame(page)
        left.pack(side="left", fill="both", expand=True)
        self._topo_layout = topology_layout()
        self._topo_canvas = tk.Canvas(
            left, width=self._topo_layout["width"],
            height=self._topo_layout["height"], highlightthickness=1,
            highlightbackground="#475569")
        self._topo_canvas.pack(padx=4, pady=4)
        self._topo_canvas.bind("<Button-1>", self._on_topo_click)
        self._tooltip(self._topo_canvas,
                      "Click a subnet or agent badge for details.\n"
                      "Edges show the live comms-policy graph.")
        right = ttk.Frame(page, width=300)
        right.pack(side="left", fill="y", padx=(4, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Details",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self._topo_details = tk.Text(right, wrap="word", height=12)
        self._topo_details.pack(fill="x")
        ttk.Label(right, text="Hosts in selection").pack(anchor="w",
                                                         pady=(6, 0))
        self._topo_hosts = tk.Listbox(right, height=12)
        self._topo_hosts.pack(fill="both", expand=True)
        self._topo_hosts.bind("<Double-Button-1>",
                              lambda _e: self._open_selected_topo_host())
        ttk.Button(right, text="Open host",
                   command=self._open_selected_topo_host).pack(pady=4)
        self._topo_selection = None
        self._topo_host_rows = []

    def _topo_health_maps(self):
        hosts = host_rows(self.pipeline)
        subnet_rank, agent_rank = {}, {}
        for row in hosts:
            rank = severity_rank(row["severity"])
            zone = row["zone"]
            if rank < subnet_rank.get(zone, 5):
                subnet_rank[zone] = rank
            agent = row["agent_id"]
            if agent is not None and rank < agent_rank.get(agent, 5):
                agent_rank[agent] = rank
        return subnet_rank, agent_rank

    def _draw_topology(self):
        canvas = self._topo_canvas
        layout = self._topo_layout
        canvas.delete("all")
        palette = self._palette
        phase = 0
        if self.pipeline is not None:
            try:
                phase = int(getattr(self.pipeline.state_builder,
                                    "mission_phase", 0))
            except Exception:
                phase = 0
        try:
            edges = comms_policy_edges(phase)
        except Exception:
            edges = set()
        names = list(layout.get("subnets", {}))
        for edge in edges:
            pair = tuple(edge)
            if len(pair) != 2:
                continue
            a, b = pair
            if a in layout["subnets"] and b in layout["subnets"]:
                x1, y1 = layout["subnets"][a]
                x2, y2 = layout["subnets"][b]
                canvas.create_line(x1, y1, x2, y2,
                                   fill=palette.get("muted_fg", "#64748b"))
        subnet_rank, agent_rank = self._topo_health_maps()
        short = {name: name.replace("_subnet", "").replace("_", " ")
                 for name in names}
        selected = getattr(self, "_topo_selection", None)
        for name, (x, y) in layout["subnets"].items():
            rank = subnet_rank.get(name, 5)
            color = {0: palette.get("bad", "red"),
                     1: "#f97316", 2: palette.get("warn", "orange"),
                     3: palette.get("warn", "orange"),
                     4: palette.get("warn", "orange")}.get(
                         rank, palette.get("good", "green"))
            canvas.create_oval(x - 28, y - 22, x + 28, y + 22, fill=color,
                               outline="white" if selected == ("subnet", name)
                               else color, width=3 if selected == ("subnet", name)
                               else 1)
            canvas.create_text(x, y, text=short[name], fill="white",
                               font=("TkDefaultFont", 9, "bold"))
        for agent_id, (x, y) in layout.get("agents", {}).items():
            rank = agent_rank.get(agent_id, 5)
            color = palette.get("good", "green") if rank >= 5 else \
                palette.get("bad", "red") if rank == 0 else \
                palette.get("warn", "orange")
            canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill=color,
                               outline="white"
                               if selected == ("agent", agent_id)
                               else color,
                               width=3 if selected == ("agent", agent_id)
                               else 1)
            canvas.create_text(x, y, text=f"A{agent_id}", fill="white",
                               font=("TkDefaultFont", 9, "bold"))

    def _on_topo_click(self, event):
        hit = topology_hit(self._topo_layout, event.x, event.y)
        if hit is None:
            return
        self._topo_selection = hit
        self._draw_topology()
        self._refresh_network_details()

    def _refresh_network_details(self):
        lines = []
        hosts = []
        selection = getattr(self, "_topo_selection", None)
        if selection is None:
            lines.append("Click a subnet node or agent badge.")
            lines.append("")
            lines.append("Edges show the live comms-policy graph for the "
                         "current mission phase.")
        else:
            kind, ref = selection
            if kind == "subnet":
                rows = [r for r in host_rows(self.pipeline)
                        if r["zone"] == ref]
                compromised = sum(1 for r in rows if r["compromised"])
                stale = sum(1 for r in rows if r["health"] == "stale")
                lines.append(f"Subnet: {ref}")
                lines.append(f"hosts: {len(rows)}  compromised: "
                             f"{compromised}  stale: {stale}")
                lines.append("")
                hosts = [(f"{r['cc4']} — {r['status']}", r["cc4"])
                         for r in rows]
            else:
                for row in agent_rows(self.pipeline):
                    if row["agent_id"] == ref:
                        lines.append(f"Agent: {row['agent']}")
                        lines.append(f"zones: {', '.join(row['zones'])}")
                        lines.append(f"hosts: {row['host_count']}  "
                                     f"health: {row['health']}")
                        lines.append(f"latest: {row['latest_label']} "
                                     f"({row['latest_status']})")
                        lines.append(f"confidence: {row['confidence']}")
                        lines.append(f"comm: {row['comm']}")
                        break
                else:
                    lines.append("No data yet — press Step or Start.")
        try:
            details = self._topo_details
            details.delete("1.0", "end")
            details.insert("end", "\n".join(lines))
            box = self._topo_hosts
            box.delete(0, "end")
            self._topo_host_rows = hosts
            for text, _cc4 in hosts:
                box.insert("end", text)
        except Exception as exc:
            self.log(f"[ERROR] network details: {exc}")

    def _open_selected_topo_host(self):
        try:
            box = self._topo_hosts
            selection = box.curselection()
            if not selection:
                return
            cc4 = self._topo_host_rows[selection[0]][1]
        except Exception:
            return
        self._open_host(cc4)

    def _refresh_network_page(self):
        try:
            self._draw_topology()
        except Exception as exc:
            self.log(f"[ERROR] draw topology: {exc}")
        self._refresh_network_details()
        blocks = []
        if self.pipeline is not None:
            try:
                blocks = sorted(
                    self.pipeline.enforcement.as_enforcement()
                    .get("blocks", {}).items())
            except Exception:
                blocks = []
        if blocks and hasattr(self, "_topo_details"):
            try:
                self._topo_details.insert(
                    "end", "\n\nActive blocks:\n" + "\n".join(
                        f"  {dst} <- {sorted(src)}" for dst, src in blocks))
            except Exception:
                pass

    # ------------------------------------------------------------- Threats --
    def _build_threats_page(self):
        page = self._pages["Threats"]
        bar = ttk.Frame(page)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Severity:").pack(side="left")
        self._threat_sev = tk.StringVar(value="all")
        sev_menu = ttk.OptionMenu(bar, self._threat_sev, "all",
                                  *SEVERITY_FILTERS,
                                  command=lambda _v: self._refresh_threats_page())  # noqa: E501
        sev_menu.pack(side="left", padx=4)
        ttk.Label(bar, text="Search:").pack(side="left", padx=(8, 0))
        self._threat_query = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self._threat_query, width=30)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: self._refresh_threats_page())
        ttk.Button(bar, text="Apply",
                   command=self._refresh_threats_page).pack(side="left")
        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self._threats_table = self._make_table_tab(
            left, ("host", "agent", "severity", "status", "evidence"))
        right = ttk.Frame(body, width=340)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Threat detail",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self._threat_detail = tk.Text(right, wrap="word", height=18)
        self._threat_detail.pack(fill="both", expand=True)
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Open host",
                   command=self._open_selected_threat_host).pack(
                       side="left", padx=2)
        ttk.Button(btns, text="View in Actions",
                   command=self._threat_to_actions).pack(side="left", padx=2)
        self._threat_selected = None
        self._threat_table_rows = []
        self._threats_table.tree.bind(
            "<<TreeviewSelect>>", lambda _e: self._on_threat_select())

    def _on_threat_select(self):
        try:
            selection = self._threats_table.tree.selection()
            if not selection:
                return
            values = self._threats_table.tree.item(selection[0])["values"]
        except Exception:
            return
        for row in self._threat_table_rows:
            if row["cc4"] == values[0]:
                self._threat_selected = row["cc4"]
                break
        self._refresh_threat_detail()

    def _open_selected_threat_host(self):
        if getattr(self, "_threat_selected", None):
            self._open_host(self._threat_selected)

    def _threat_to_actions(self):
        if getattr(self, "_threat_selected", None):
            self._actions_host_filter.set(self._threat_selected)
            self.show_page("Actions")

    def _refresh_threats_page(self):
        rows = threat_list(
            self.pipeline,
            severity=self._threat_sev.get(),
            query=self._threat_query.get())
        self._threat_table_rows = rows
        self._fill_table(
            self._threats_table.tree,
            ("host", "agent", "severity", "status", "evidence"),
            [(r["cc4"], r["agent"], r["severity"], r["status"],
              "; ".join(r["notes"][:2]) or "—") for r in rows],
            empty_text="(no matching threats)")
        self._refresh_threat_detail()

    def _refresh_threat_detail(self):
        cc4 = getattr(self, "_threat_selected", None)
        detail = threat_detail(self.pipeline, cc4) if cc4 else None
        lines = []
        if not detail or detail.get("host") is None:
            lines.append("Select a threat for evidence and timeline.")
        else:
            host = detail["host"]
            lines.append(f"Host: {cc4}")
            lines.append(f"status: {host['status']}  "
                         f"severity: {host['severity']}")
            lines.append(f"agent: {host['agent']}  zone: {host['zone']}")
            lines.append("")
            lines.append("Evidence:")
            if host["notes"]:
                lines.extend(f"  - {note}" for note in host["notes"])
            else:
                lines.append("  (none recorded)")
            lines.append("")
            lines.append("Timeline (latest first):")
            if detail["timeline"]:
                for item in detail["timeline"]:
                    lines.append(f"  cyc {item.get('cycle')}: "
                                 f"{item.get('label')} "
                                 f"({item.get('status')})")
            else:
                lines.append("  (no decisions reference this host yet)")
        self._set_text(self._threat_detail, "\n".join(lines))

    # ------------------------------------------------------- page factory --
    def _build_pages(self):
        self._build_overview_page()
        self._build_network_page()
        self._build_threats_page()
        self._build_agents_page()
        self._build_actions_page()
        self._build_audit_page()
        self._build_model_page()
        self._build_telemetry_page()
        self._build_settings_page()
        # Session log lives on Settings (attribute kept for log()).
        self._tab_log = self._make_text_tab(self._pages["Settings"])
        ttk.Label(self._pages["Settings"],
                  text="Session log (newest at bottom)").pack()

    # -------------------------------------------------------------- Agents --
    def _build_agents_page(self):
        page = self._pages["Agents"]
        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self._agents_table = self._make_table_tab(
            left, ("agent", "zones", "hosts", "health", "latest",
                   "confidence"))
        right = ttk.Frame(body, width=340)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Agent detail",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self._agent_detail = tk.Text(right, wrap="word", height=20)
        self._agent_detail.pack(fill="both", expand=True)
        self._agents_table.tree.bind(
            "<<TreeviewSelect>>", lambda _e: self._refresh_agent_detail())
        self._agent_selected = None

    def _refresh_agents_page(self):
        rows = agent_rows(self.pipeline)
        self._agent_table_rows = rows
        self._fill_table(
            self._agents_table.tree,
            ("agent", "zones", "hosts", "health", "latest", "confidence"),
            [(r["agent"], ", ".join(z.replace("_subnet", "")
                                    for z in r["zones"]),
              r["host_count"], r["health"], r["latest_label"],
              r["confidence"]) for r in rows],
            empty_text="(no data yet — press Step or Start)")
        self._refresh_agent_detail()

    def _refresh_agent_detail(self):
        selected = None
        try:
            selection = self._agents_table.tree.selection()
            if selection:
                values = self._agents_table.tree.item(selection[0])["values"]
                for row in getattr(self, "_agent_table_rows", []):
                    if row["agent"] == values[0]:
                        selected = row
                        break
        except Exception:
            selected = None
        lines = []
        if selected is None:
            lines.append("Select an agent for zones, health, latest "
                         "decision and communication.")
        else:
            lines.append(f"Agent: {selected['agent']}")
            lines.append(f"zones: {', '.join(selected['zones'])}")
            lines.append(f"hosts: {selected['host_count']}  "
                         f"health: {selected['health']}")
            lines.append(f"latest: {selected['latest_label']} "
                         f"({selected['latest_status']}, "
                         f"cycle {selected['latest_cycle']})")
            lines.append(f"confidence: {selected['confidence']}")
            lines.append(f"comm: {selected['comm']}")
        self._set_text(self._agent_detail, "\n".join(lines))

    # -------------------------------------------------------------- Actions --
    def _build_actions_page(self):
        page = self._pages["Actions"]
        pend_card = ttk.LabelFrame(page, text=" Needs approval ", padding=6)
        pend_card.pack(fill="x", pady=(0, 6))
        # Kept attribute: the approvals table (tree columns start with
        # "#" and include "attempts"; selection index stays stable).
        self._tab_approvals = self._make_table_tab(
            pend_card, ("#", "agent", "operation", "target", "label",
                        "attempts"))
        btns = ttk.Frame(pend_card)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Approve selected",
                   command=lambda: self._guarded(
                       "approve", self._approve_selected)).pack(side="left",
                                                                padx=2)
        ttk.Button(btns, text="Reject selected",
                   command=lambda: self._guarded(
                       "deny", self._deny_selected)).pack(side="left",
                                                          padx=2)
        ttk.Button(btns, text="Details...",
                   command=lambda: self._guarded(
                       "action details", self._open_selected_action)
                   ).pack(side="left", padx=2)
        recent_card = ttk.LabelFrame(page, text=" Recent decisions ",
                                     padding=6)
        recent_card.pack(fill="both", expand=True)
        filt = ttk.Frame(recent_card)
        filt.pack(fill="x", pady=(0, 4))
        ttk.Label(filt, text="Host filter:").pack(side="left")
        self._actions_host_filter = tk.StringVar()
        entry = ttk.Entry(filt, textvariable=self._actions_host_filter,
                          width=32)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: self._refresh_actions_page())
        ttk.Button(filt, text="Apply",
                   command=self._refresh_actions_page).pack(side="left")
        self._actions_table = self._make_table_tab(
            recent_card, ("agent", "action", "target", "confidence",
                          "risk", "status"))

    def _refresh_actions_page(self):
        rows = pending_rows(self.pipeline)
        filt = (getattr(self, "_actions_host_filter", None).get()
                if hasattr(self, "_actions_host_filter") else "")
        filt = (filt or "").strip().lower()
        if filt:
            rows = [r for r in rows if filt in str(r.get("target", ""))
                    .lower() or filt in str(r.get("label", "")).lower()]
        self._pending_table_rows = rows
        self._fill_table(
            self._tab_approvals.tree,
            ("#", "agent", "operation", "target", "label", "attempts"),
            [(r["index"], r["agent"], r["operation"], r["target"],
              r["label"], r["attempts"]) for r in rows],
            empty_text="(approval queue empty)")
        recent = action_rows(self.pipeline)
        if filt:
            recent = [r for r in recent
                      if filt in str(r.get("target", "")).lower()]
        self._recent_action_rows = recent
        self._fill_table(
            self._actions_table.tree,
            ("agent", "action", "target", "confidence", "risk", "status"),
            [(r["agent"], r["action"], r["target"], r["confidence"],
              r["risk"], r["status"]) for r in recent],
            empty_text="(no decisions yet — press Step or Start)")

    def _refresh_approvals(self):
        # Legacy entry point kept working: refresh the approvals table
        # wherever the Actions page currently is.
        try:
            self._refresh_actions_page()
        except Exception as exc:
            self.log(f"[ERROR] refresh approvals: {exc}")

    def _selected_approval(self):
        tree = self._tab_approvals.tree
        selection = tree.selection()
        if not selection:
            return None
        try:
            return int(tree.item(selection[0])["values"][0])
        except (TypeError, ValueError, IndexError):
            # Empty-state placeholder row (or anything unparsable)
            # is never a real approval.
            return None

    def _approve_bar(self):
        # Legacy hook kept (buttons now live on the Actions page).
        return None

    def _approve_selected(self):
        index = self._selected_approval()
        if index is None:
            return
        result = self.pipeline.approve_pending(index)
        self.log(f"approved -> {result.operation} on {result.target}: "
                 f"{result.details}")
        self._refresh_all()

    def _deny_selected(self):
        index = self._selected_approval()
        if index is None:
            return
        pending = self.pipeline.enforcement.pending_approvals
        denied = pending.pop(index)
        self.log(f"denied {denied.get('operation')} on "
                 f"{denied.get('target')}")
        self._refresh_all()

    def _open_selected_action(self):
        try:
            tree = self._actions_table.tree
            selection = tree.selection()
            if not selection:
                return
            values = tree.item(selection[0])["values"]
        except Exception:
            return
        for row in getattr(self, "_recent_action_rows", []):
            if (row["agent"], row["action"], row["target"]) == tuple(
                    values[:3]):
                self._open_action(row["record"])
                return

    # --------------------------------------------------------------- Audit --
    def _build_audit_page(self):
        page = self._pages["Audit"]
        bar = ttk.Frame(page)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Search:").pack(side="left")
        self._audit_query = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self._audit_query, width=30)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: self._refresh_audit_page())
        ttk.Label(bar, text="Event:").pack(side="left", padx=(8, 0))
        self._audit_event = tk.StringVar(value="all")
        events = ["all", "approved", "queued", "executed", "failed",
                  "shadowed"]
        ttk.OptionMenu(bar, self._audit_event, "all", *events,
                       command=lambda _v: self._refresh_audit_page()
                       ).pack(side="left", padx=4)
        ttk.Button(bar, text="Apply",
                   command=self._refresh_audit_page).pack(side="left")
        ttk.Button(bar, text="Export CSV",
                   command=lambda: self._guarded(
                       "export csv", self._export_audit, "csv")
                   ).pack(side="right", padx=2)
        ttk.Button(bar, text="Export JSON",
                   command=lambda: self._guarded(
                       "export json", self._export_audit, "json")
                   ).pack(side="right", padx=2)
        self._audit_table = self._make_table_tab(
            page, ("time", "event", "operation", "target", "mode",
                   "applied", "approver", "attempts", "error"))
        self._audit_rows_cache = []

    def _refresh_audit_page(self):
        entries = []
        if self.pipeline is not None:
            try:
                entries = list(
                    getattr(self.pipeline.enforcement, "audit", []) or [])
            except Exception:
                entries = []
        rows = audit_rows(
            entries,
            query=self._audit_query.get(),
            event=self._audit_event.get())
        self._audit_rows_cache = rows
        import time as _time

        def _ts(entry):
            try:
                return _time.strftime(
                    "%H:%M:%S", _time.localtime(float(entry.get("audit_ts",
                                                               0))))
            except (TypeError, ValueError):
                return "—"

        self._fill_table(
            self._audit_table.tree,
            ("time", "event", "operation", "target", "mode", "applied",
             "approver", "attempts", "error"),
            [(_ts(e), e.get("event"), e.get("operation"), e.get("target"),
              e.get("mode"), e.get("applied"), e.get("approver", ""),
              e.get("attempts", ""), str(e.get("error", ""))[:60])
             for e in rows],
            empty_text="(no audit events yet)")

    def _export_audit(self, format):
        if filedialog is None:
            self.log("[ERROR] export: no file dialog available")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{format}",
            filetypes=[(f"{format.upper()} files", f"*.{format}"),
                       ("All files", "*.*")])
        if not path:
            return
        export_audit(self._audit_rows_cache, path, format=format)
        self.log(f"audit exported to {path} "
                 f"({len(self._audit_rows_cache)} events)")

    # --------------------------------------------------------------- Model --
    def _build_model_page(self):
        page = self._pages["Model"]
        self._model_cards = ttk.Frame(page)
        self._model_cards.pack(fill="x")
        self._model_cards_vars = {}
        for key, title in (("engine", "Policy engine"),
                           ("checkpoint", "Checkpoint"),
                           ("vocab", "Vocabulary"),
                           ("geometry", "Geometry"),
                           ("latency", "Inference latency")):
            frame = self._card(self._model_cards, title)
            var = tk.StringVar(value="—")
            tk.Label(frame, textvariable=var, anchor="w",
                     justify="left", wraplength=300).pack(fill="x")
            self._model_cards_vars[key] = var
        tech_bar = ttk.Frame(page)
        tech_bar.pack(fill="x", pady=(6, 0))
        self._model_tech_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tech_bar, text="Show technical details (CC4 state, "
                           "observations, masks, inference)",
            variable=self._model_tech_var,
            command=self._refresh_model_page).pack(anchor="w")
        self._model_tech = self._make_text_tab(page)
        self._model_errors = tk.StringVar(value="")
        ttk.Label(page, text="Model errors:").pack(anchor="w", pady=(6, 0))
        tk.Label(page, textvariable=self._model_errors, anchor="w",
                 justify="left", wraplength=900).pack(fill="x")

    def _refresh_model_page(self):
        model = model_page_model(self.pipeline)
        cards = self._model_cards_vars
        cards["engine"].set(model["engine"])
        cards["checkpoint"].set(str(model["checkpoint"]))
        vocab_ok = "match ✓" if model["vocab_match"] else "MISMATCH ✗"
        cards["vocab"].set(f"HOST {model['vocab_host']} ({vocab_ok})")
        sizes = ", ".join(f"a{i}:{n}" for i, n in sorted(
            model["action_sizes"].items())) or "—"
        cards["geometry"].set(
            f"obs {model['obs_dims']}\nactions [{sizes}]\n"
            f"rows {model['obs_rows'] or 'no cycle yet'}")
        if model["latency_avg_ms"] is None:
            cards["latency"].set("no inference yet")
        else:
            cards["latency"].set(
                f"avg {model['latency_avg_ms']:.1f} ms / "
                f"max {model['latency_max_ms']:.1f} ms")
        errors = model["errors"]
        self._model_errors.set("\n".join(errors) if errors else "none")
        if self._model_tech_var.get():
            lines = self._model_technical_lines()
            self._set_text(self._model_tech, "\n".join(lines))
            self._model_tech.pack(fill="both", expand=True)
        else:
            self._model_tech.pack_forget()

    def _model_technical_lines(self):
        lines = []
        pipeline = self.pipeline
        if pipeline is None:
            return ["(no pipeline — press Rebuild)"]
        try:
            phase = getattr(pipeline.state_builder, "mission_phase", "?")
        except Exception:
            phase = "?"
        lines.append(f"mission_phase: {phase}")
        try:
            blocks = pipeline.enforcement.as_enforcement().get("blocks", {})
            lines.append(f"blocks: {blocks or '{}'}")
        except Exception:
            pass
        obs = getattr(pipeline, "last_obs", None)
        if obs is not None:
            try:
                import numpy as _np
                arr = _np.asarray(obs)
                lines.append(f"obs batch shape: {tuple(arr.shape)}")
                for i, row in enumerate(arr):
                    nz = int((row != 0).sum())
                    lines.append(f"  agent {i}: {nz} nonzero features")
            except Exception as exc:
                lines.append(f"obs: unreadable ({exc})")
        else:
            lines.append("obs: no cycle run yet")
        masks = getattr(pipeline, "last_masks", None)
        if masks is not None:
            try:
                for i, mask in enumerate(masks):
                    allowed = int((mask != 0).sum())
                    lines.append(f"  agent {i} mask: {allowed} allowed "
                                 f"actions")
            except Exception as exc:
                lines.append(f"masks: unreadable ({exc})")
        else:
            lines.append("masks: no cycle run yet")
        if self.pipeline is not None:
            try:
                for agent in range(5):
                    latest = _latest_record_for_agent(self.pipeline, agent)
                    if latest is None:
                        continue
                    message = latest.get("message") or {}
                    lines.append(
                        f"agent {agent} last message: "
                        + format_message(message))
                    lines.append(
                        f"agent {agent} trust: {latest.get('trust_row')}")
            except Exception as exc:
                lines.append(f"comm: unreadable ({exc})")
        return lines

    # ----------------------------------------------------------- Telemetry --
    def _build_telemetry_page(self):
        page = self._pages["Telemetry"]
        self._tele_cards = ttk.Frame(page)
        self._tele_cards.pack(fill="x")
        self._tele_cards_vars = {}
        for key, title in (("collector", "Collector"),
                           ("freshness", "Freshness"),
                           ("volume", "Volume"),
                           ("problems", "Problems")):
            frame = self._card(self._tele_cards, title)
            var = tk.StringVar(value="—")
            tk.Label(frame, textvariable=var, anchor="w",
                     justify="left", wraplength=300).pack(fill="x")
            self._tele_cards_vars[key] = var
        bottom = ttk.Frame(page)
        bottom.pack(fill="both", expand=True)
        partial = self._card(bottom, "Partial errors")
        self._tele_partial = tk.Text(partial, wrap="word", height=8)
        self._tele_partial.pack(fill="both", expand=True)
        degraded = self._card(bottom, "Degraded components")
        self._tele_degraded = tk.Text(degraded, wrap="word", height=8)
        self._tele_degraded.pack(fill="both", expand=True)

    def _refresh_telemetry_page(self):
        model = telemetry_model(self.pipeline, event_log=list(
            getattr(self, "_event_log", [])))
        cards = self._tele_cards_vars
        cards["collector"].set(
            f"state: {model['state']}\n"
            f"source: {model['source'] or '—'}\n"
            f"batches: {model['total_batches']}  "
            f"failures: {model['total_failures']}")
        if model["last_update_ts"]:
            import time as _time
            age = _time.time() - model["last_update_ts"]
            cards["freshness"].set(
                f"last update: {_time.strftime('%H:%M:%S', _time.localtime(model['last_update_ts']))} "  # noqa: E501
                f"({age:.0f}s ago)")
        else:
            cards["freshness"].set("last update: never")
        rate = model["events_per_sec"]
        cards["volume"].set(
            f"events last batch: {model['events_last_batch']}\n"
            f"events/sec (5 min): "
            f"{f'{rate:.1f}' if rate is not None else '—'}")
        cards["problems"].set(
            f"consecutive failures: {model['consecutive_failures']}\n"
            f"last error: {model['last_error'] or '—'}\n"
            f"stale hosts: {len(model['stale_hosts'])}  "
            f"unseen: {len(model['unseen_keys'])}")
        self._set_text(self._tele_partial, "\n".join(
            f"{e.get('key')}: {e.get('error')}"
            for e in model["partial_errors"]) or "(none)")
        self._set_text(self._tele_degraded, "\n".join(
            model["degraded"]) or "(none — all components healthy)")

    # ------------------------------------------------------------ Settings --
    def _build_settings_page(self):
        page = self._pages["Settings"]
        top = ttk.Frame(page)
        top.pack(fill="x")
        theme_card = self._card(top, "Appearance")
        self._theme_var = tk.StringVar(value=self._theme)
        for name in ("dark", "light"):
            ttk.Radiobutton(
                theme_card, text=name.capitalize(), value=name,
                variable=self._theme_var,
                command=self._on_theme_radio).pack(anchor="w")
        session_card = self._card(top, "Session")
        self._session_info = tk.StringVar(value="")
        tk.Label(session_card, textvariable=self._session_info,
                 anchor="w", justify="left", wraplength=300).pack(fill="x")
        ttk.Button(session_card, text="Reset session",
                   command=lambda: self._guarded(
                       "reset", self._reset_session)).pack(pady=4,
                                                           anchor="w")
        ops_card = self._card(top, "Operations")
        ttk.Label(ops_card, text="Mission phase:").pack(anchor="w")
        self._mission_var = tk.StringVar(value="Preplanning")
        ttk.OptionMenu(
            ops_card, self._mission_var, "Preplanning", "Preplanning",
            "MissionA", "MissionB",
            command=lambda _v: self._guarded(
                "mission phase", self._apply_mission_phase)).pack(
                    anchor="w", pady=2)
        ttk.Label(ops_card, text="Shortcuts: Alt+1..9 switch pages, "
                                 "Esc closes popups.",
                  wraplength=300, justify="left").pack(anchor="w", pady=4)
        log_card = ttk.LabelFrame(page, text=" Session log ", padding=8)
        log_card.pack(fill="both", expand=True, padx=4, pady=4)
        self._tab_log = self._make_text_tab(log_card)

    def _on_theme_radio(self):
        self._theme = self._theme_var.get()
        self._palette = apply_theme(self.root, self._theme)
        self._recolor_chrome()
        self.show_page(getattr(self, "_current_page", "Overview"))

    def _reset_session(self):
        if self.pipeline is not None:
            self.pipeline.reset()
            self.last_records = []
            self._event_log.clear()
            self.log("session reset by operator")
        self._refresh_all()

    def _apply_mission_phase(self):
        mapping = {"Preplanning": 0, "MissionA": 1, "MissionB": 2}
        phase = mapping.get(self._mission_var.get(), 0)
        if self.pipeline is not None:
            self.pipeline.state_builder.set_mission_phase(phase)
            self.log(f"mission phase set to {phase}")
        self._refresh_all()

    def _refresh_settings_page(self):
        info = ["session: —", "cycle: —", "mode: —", "collector: —",
                "checkpoint: —"]
        if self.pipeline is not None:
            try:
                model = model_page_model(self.pipeline)
                tele = telemetry_model(self.pipeline)
                info = [
                    f"session: {self.pipeline.session_id}",
                    f"cycle: {self.pipeline.cycle}",
                    f"mode: {self.pipeline.validator.mode}",
                    f"collector: {tele['state']}",
                    f"checkpoint: {model['checkpoint']}",
                ]
            except Exception as exc:
                info = [f"(unreadable: {exc})"]
        self._session_info.set("\n".join(info))

    # --------------------------------------------------------------- modals --
    def _open_host(self, cc4):
        rows = {row["cc4"]: row for row in host_rows(self.pipeline)}
        row = rows.get(cc4)
        if row is None:
            self.log(f"[ERROR] unknown host {cc4!r}")
            return
        window = self._modal(f"Host — {cc4}", width=560, height=520)
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=row["hostname"] or cc4,
                  font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            body,
            text=f"{row['zone']}  ·  {row['agent']}  ·  {row['status']} "
                 f"({row['severity']})").pack(anchor="w", pady=(0, 6))
        grid = ttk.Frame(body)
        grid.pack(fill="x")
        details = (
            ("IP", row["ip"] or "—"),
            ("Hostname", row["hostname"] or "—"),
            ("Role", row["role"] or "—"),
            ("Agent", row["agent"] or "—"),
            ("Health", row["health"]),
            ("Up", "yes" if row["up"] else "no"),
            ("Sessions", str(row["sessions"])),
            ("Compromised", "yes" if row["compromised"] else "no"),
        )
        for index, (key, value) in enumerate(details):
            ttk.Label(grid, text=f"{key}:",
                      font=("TkDefaultFont", 9, "bold")).grid(
                          row=index, column=0, sticky="w", padx=(0, 8))
            ttk.Label(grid, text=value).grid(row=index, column=1,
                                             sticky="w")
        tele = self._raw_telemetry_for(row)
        ttk.Label(body, text="Telemetry (latest batch)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w",
                                                           pady=(8, 0))
        tele_text = tk.Text(body, wrap="word", height=8)
        tele_text.pack(fill="both", expand=True)
        if tele is None:
            lines = ["(no fresh telemetry for this host)"]
        else:
            lines = [
                f"processes ({len(tele.get('processes', []) or [])}): "
                f"{', '.join((tele.get('processes', []) or [])[:12]) or '—'}",  # noqa: E501
                f"connections ({len(tele.get('connections', []) or [])}):",
            ]
            lines.extend(f"  {c}"
                         for c in (tele.get("connections", []) or [])[:12])
            lines.append(
                f"sessions: {', '.join((tele.get('sessions', []) or [])) or '—'}")  # noqa: E501
            events = tele.get("events", []) or []
            lines.append(f"recent events ({len(events)}):")
            for event in events[-8:]:
                lines.append(
                    f"  [{event.get('severity', '?')}] "
                    f"{event.get('kind', '?')}: {event.get('details', '')}")
        tele_text.insert("end", "\n".join(lines))
        ttk.Label(body, text="Current recommendation",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w",
                                                           pady=(8, 0))
        ttk.Label(body, text=self._recommendation_for(cc4),
                  wraplength=520, justify="left").pack(anchor="w")
        show_tech = tk.BooleanVar(value=False)
        tech_frame = ttk.Frame(body)
        tech_text = tk.Text(tech_frame, wrap="word", height=8)
        tech_text.pack(fill="both", expand=True)

        def _toggle_tech():
            if show_tech.get():
                tech_text.delete("1.0", "end")
                tech_text.insert("end", "\n".join(
                    self._host_technical_lines(row)))
                tech_frame.pack(fill="both", expand=True)
            else:
                tech_frame.pack_forget()

        ttk.Checkbutton(body, text="Show technical details (CC4 state)",
                        variable=show_tech,
                        command=_toggle_tech).pack(anchor="w", pady=(6, 0))
        ttk.Button(body, text="Close",
                   command=lambda: (self._untrack_modal(window),
                                    window.destroy())).pack(pady=6)

    def _raw_telemetry_for(self, row):
        batch = getattr(self.pipeline, "last_batch", None)
        hosts = getattr(batch, "hosts", None) or {}
        for key in (row.get("ip", ""), row.get("hostname", "")):
            if key and key in hosts:
                tele = hosts[key]
                if isinstance(tele, dict):
                    return tele
                return {"processes": list(getattr(tele, "processes", [])
                                         or []),
                        "connections": list(getattr(tele, "connections", [])
                                            or []),
                        "sessions": list(getattr(tele, "sessions", [])
                                         or []),
                        "events": [self._event_dict(e) for e in
                                   (getattr(tele, "events", []) or [])]}
        return None

    @staticmethod
    def _event_dict(event):
        if isinstance(event, dict):
            return event
        return {"kind": getattr(event, "kind", "?"),
                "severity": getattr(event, "severity", "?"),
                "details": getattr(event, "details", "")}

    def _recommendation_for(self, cc4):
        for record in reversed(list(getattr(self.pipeline, "history", [])
                                    or [])):
            if cc4 and cc4 in str(record.get("label") or ""):
                return (f"{record.get('label')} "
                        f"({record.get('status')}, cycle "
                        f"{record.get('cycle')})")
        return "Monitor (routine — no targeted action proposed)"

    def _host_technical_lines(self, row):
        from .config import STABLE_HOST_INDEX
        lines = []
        try:
            slot = self.pipeline.asset_map.slot_of(row["agent"], row["cc4"])
            lines.append(f"slot (subnet, host): {slot}")
        except Exception as exc:
            lines.append(f"slot: unavailable ({exc})")
        lines.append(f"stable target id: "
                     f"{STABLE_HOST_INDEX.get(row['cc4'], 'n/a')}")
        lines.append(f"alerts: process={row['process_event']} "
                     f"connection={row['connection_event']}")
        tables = getattr(self.pipeline, "tables", None) or {}
        masks = getattr(self.pipeline, "last_masks", None)
        agent_id = row["agent_id"]
        table = tables.get(agent_id, []) if isinstance(tables, dict) \
            else []
        try:
            mask = list(masks[agent_id]) if masks is not None else []
        except Exception:
            mask = []
        for command in ("Analyse", "Remove", "Restore", "DeployDecoy"):
            indices = [e["index"] for e in table
                       if e.get("command") == command
                       and e.get("target") == row["cc4"]]
            states = []
            for index in indices:
                states.append("allowed" if index < len(mask) and mask[index]
                              else "masked")
            lines.append(f"{command}: "
                         f"{', '.join(f'#{i} {s}' for i, s in zip(indices, states)) or '—'}")  # noqa: E501
        return lines

    def _open_action(self, record):
        window = self._modal("Action details", width=600, height=520)
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=str(record.get("label")),
                  font=("TkDefaultFont", 13, "bold"),
                  wraplength=560).pack(anchor="w")
        grid = ttk.Frame(body)
        grid.pack(fill="x", pady=6)
        items = (
            ("Agent", str(record.get("agent_id"))),
            ("Command", str(record.get("command"))),
            ("Operation", str(record.get("operation"))),
            ("Risk", str(record.get("risk"))),
            ("Status", str(record.get("status"))),
            ("Confidence",
             confidence_of((record.get("message") or {}))),
            ("Validation", ("ok" if (record.get("validation") or {}).get("ok")
                            else (record.get("validation") or {}).get(
                                "error", "failed"))),
            ("Approval", str((record.get("approval") or {}))),
            ("Execution", str((record.get("exec") or {}))),
        )
        for index, (key, value) in enumerate(items):
            ttk.Label(grid, text=f"{key}:",
                      font=("TkDefaultFont", 9, "bold")).grid(
                          row=index, column=0, sticky="w", padx=(0, 8))
            ttk.Label(grid, text=value, wraplength=420,
                      justify="left").grid(row=index, column=1, sticky="w")
        ttk.Label(body, text="Message",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w",
                                                           pady=(6, 0))
        msg = tk.Text(body, wrap="word", height=6)
        msg.pack(fill="both", expand=True)
        msg.insert("end", format_message(record.get("message")))
        ttk.Button(body, text="Close",
                   command=lambda: (self._untrack_modal(window),
                                    window.destroy())).pack(pady=6)

    def _open_search(self):
        query = self.search_var.get() if hasattr(self, "search_var") else ""
        results = search(self.pipeline, query)
        window = self._modal(f"Search: {query!r}", width=560, height=380)
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        box = tk.Listbox(body, height=16)
        box.pack(fill="both", expand=True)
        idents = []
        for result in results:
            box.insert("end",
                       f"[{result['kind']}] {result['title']} — "
                       f"{result['subtitle']}")
            idents.append(result)
        if not results:
            box.insert("end", "(no matches — try a hostname, IP, "
                              "agent, zone, or operation)")
        box.results = idents

        def _activate(_event=None):
            try:
                selection = box.curselection()
                if not selection:
                    return
                self._activate_search_result(box.results[selection[0]])
                self._untrack_modal(window)
                window.destroy()
            except Exception as exc:
                self.log(f"[ERROR] search open: {exc}")

        box.bind("<Double-Button-1>", _activate)
        box.bind("<Return>", _activate)
        ttk.Button(body, text="Close",
                   command=lambda: (self._untrack_modal(window),
                                    window.destroy())).pack(pady=6)

    def _activate_search_result(self, result):
        kind, ref = result.get("kind"), result.get("ref")
        if kind == "host":
            self._open_host(ref)
        elif kind == "agent":
            self.show_page("Agents")
        elif kind == "threat":
            self._threat_selected = ref
            self.show_page("Threats")
        elif kind == "audit":
            if isinstance(ref, dict):
                self._audit_query.set(ref.get("operation", ""))
            self.show_page("Audit")
        elif kind == "tab":
            self.show_page(ref)

    def _open_notifications(self):
        items = notification_list(self.pipeline)
        window = self._modal("Notifications", width=560, height=380)
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        text = tk.Text(body, wrap="word")
        text.pack(fill="both", expand=True)
        colors = {"critical": "#f87171", "warning": "#fbbf24",
                  "info": "#38bdf8"}
        for level in ("critical", "warning", "info"):
            try:
                text.tag_configure(level, foreground=colors[level])
            except Exception:
                pass
        if not items:
            text.insert("end", "All clear — nothing needs attention.")
        for item in items:
            try:
                text.insert("end", "● ", (item["level"],))
            except Exception:
                text.insert("end", "- ")
            text.insert("end", f"[{item['level'].upper()}] {item['text']}\n")
        text.configure(state="disabled")
        ttk.Button(body, text="Close",
                   command=lambda: (self._untrack_modal(window),
                                    window.destroy())).pack(pady=6)


def launch(build_pipeline):
    """Create the root window and run the console (blocks on mainloop)."""
    if not _TK_AVAILABLE:
        raise RuntimeError("tkinter is not available in this Python")
    root = tk.Tk()
    console = DeploymentConsole(root, build_pipeline)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (console.close(), root.destroy()))
    root.mainloop()
    return console
# __END__