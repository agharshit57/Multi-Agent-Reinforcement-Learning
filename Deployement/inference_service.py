"""Policy-inference sidecar: clean boundary for non-Python frontends.

The trained PyTorch MAPPO stack stays in Python (no model port). Native
frontends (the C# SOC app in ``CyberMarl.Deployment/``) talk to this
loopback HTTP service instead::

    GET  /health    -> service/engine/mode status (auth required)
    GET  /contract  -> frozen dims + vocab + checkpoint meta
    POST /decide    -> {obs:[5,210], masks:[5,242], host_masks?, host_valid?}
                       -> per-agent {action, label, message, trust_row}
    POST /cycle     -> {real_topology, real_batch} -> one real-world
                       cycle -> {records, real_actions, mapping}
    POST /approve   -> {index, approver?} -> approve one queued
                       supervised/live decision exactly once
                       (single-ownership pop-first, same semantics as
                       DeploymentPipeline.approve_pending)

Design rules: stdlib only (``http.server``); 127.0.0.1 default;
shared-secret Bearer auth (``INFERENCE_TOKEN`` env or generated);
8 MiB body cap; every error is a JSON ``{error}`` envelope (never a
traceback); torch is imported ONLY when a trained engine is attached
(the mock path serves without the training venv); modes/masks/gates
are enforced by the underlying pipeline, never reimplemented here.
Schema version: ``"real-first/v1"`` (``/contract`` announces it).
"""

import hashlib
import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCHEMA_VERSION = "real-first/v1"
MAX_BODY_BYTES = 8 * 1024 * 1024


class InferenceServiceError(RuntimeError):
    """Sidecar misuse (bad schema, unknown route, auth failure)."""


def _contract_dict(policy_info=None):
    from .config import (ACTION_DIM, LARGE_ACTION_DIM, LARGE_OBS_DIM,
                         NUM_AGENTS, NUM_HOST_TARGETS, OBS_DIM,
                         SMALL_ACTION_DIM, SMALL_OBS_DIM, SUBNETS)
    return {"schema": SCHEMA_VERSION, "obs_dim": OBS_DIM,
            "small_obs_dim": SMALL_OBS_DIM,
            "large_obs_dim": LARGE_OBS_DIM,
            "action_dim": ACTION_DIM,
            "small_action_dim": SMALL_ACTION_DIM,
            "large_action_dim": LARGE_ACTION_DIM,
            "num_agents": NUM_AGENTS,
            "num_host_targets": NUM_HOST_TARGETS,
            "subnets": list(SUBNETS),
            "policy": dict(policy_info or {})}


class InferenceService:
    """Loopback inference + real-world-cycle service (see module docs)."""

    def __init__(self, host="127.0.0.1", port=0, token=None,
                 real_pipeline=None, policy=None, tables=None,
                 policy_info=None):
        self.host = host
        self.port = int(port)
        self.token = token or os.environ.get("INFERENCE_TOKEN") or \
            secrets.token_hex(32)
        self.generated_token = token is None and \
            "INFERENCE_TOKEN" not in os.environ
        self.real_pipeline = real_pipeline
        self.policy = policy
        self.tables = tables
        self.policy_info = dict(policy_info or {})
        self._server = None
        self._thread = None

    # ------------------------------------------------------------ serve --
    def start(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CyberMarlInference/1.0"

            def log_message(self, *args):
                pass  # never log tokens/bodies to stdout

            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self):
                want = service.token
                got = self.headers.get("Authorization", "")
                if not got.startswith("Bearer "):
                    return False
                return hmac.compare_digest(got[len("Bearer "):], want)

            def do_GET(self):
                if self.path not in ("/health", "/contract"):
                    self._send(404, {"error": "unknown route"})
                    return
                if not self._authorized():
                    self._send(401, {"error": "unauthorized"})
                    return
                if self.path == "/health":
                    self._send(200, service.health())
                else:
                    self._send(200, service.contract())

            def do_POST(self):
                if self.path not in ("/decide", "/cycle", "/approve"):
                    self._send(404, {"error": "unknown route"})
                    return
                if not self._authorized():
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > MAX_BODY_BYTES:
                    self._send(400, {"error": "bad content length"})
                    return
                try:
                    payload = json.loads(
                        self.rfile.read(length).decode("utf-8"))
                except Exception as exc:
                    self._send(400, {
                        "error": f"malformed JSON: {exc}"})
                    return
                try:
                    if self.path == "/decide":
                        self._send(200, service.decide(payload))
                    elif self.path == "/cycle":
                        self._send(200, service.cycle(payload))
                    else:
                        self._send(200, service.approve(payload))
                except InferenceServiceError as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:  # never leak tracebacks
                    self._send(500, {
                        "error": f"{type(exc).__name__}: {exc}"})

        self._server = ThreadingHTTPServer((self.host, self.port),
                                           Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05},
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            try:
                self._server.server_close()
            except Exception:
                pass
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    # --------------------------------------------------------------- API --
    def health(self):
        inner = getattr(self.real_pipeline, "inner", None)
        engine = self.policy_info.get("engine")
        if not engine and self.policy is not None:
            engine = getattr(self.policy, "name", "standalone-policy")
        if not engine and inner is not None and \
                getattr(inner, "policy", None) is not None:
            engine = getattr(inner.policy, "name", "pipeline-owned")
        return {"schema": SCHEMA_VERSION, "ok": True,
                "engine": engine or "mock",
                "mode": (getattr(getattr(inner, "validator", None),
                                 "mode", None)),
                "cycle": getattr(inner, "cycle", None),
                "mapping": (self.real_pipeline.mapping_summary()
                            if self.real_pipeline is not None else None)}

    def contract(self):
        return _contract_dict(self.policy_info)

    def decide(self, payload):
        """Pure inference: validated tensors in, decisions out."""
        import numpy as np
        from .config import NUM_AGENTS
        if self.policy is None or self.tables is None:
            raise InferenceServiceError(
                "this sidecar serves /cycle only (no standalone "
                "policy attached)")
        try:
            obs = np.asarray(payload["obs"], dtype=np.float32)
            masks = np.asarray(payload["masks"], dtype=bool)
        except KeyError as exc:
            raise InferenceServiceError(
                f"missing field {exc} (want obs, masks)") from exc
        if obs.shape != (NUM_AGENTS, 210):
            raise InferenceServiceError(
                f"obs must be [5,210], got {list(obs.shape)}")
        if masks.shape != (NUM_AGENTS, 242):
            raise InferenceServiceError(
                f"masks must be [5,242], got {list(masks.shape)}")
        host_masks = payload.get("host_masks")
        host_valid = payload.get("host_valid")
        try:
            decisions = self.policy.decide(
                obs, masks, host_masks=host_masks, host_valid=host_valid)
        except Exception as exc:
            raise InferenceServiceError(
                f"policy failure: {type(exc).__name__}: {exc}") from exc
        out = []
        for agent, decision in enumerate(decisions):
            table = self.tables[agent]
            action = int(decision.get("action", -1))
            label = (table[action]["label"]
                     if 0 <= action < len(table) else "<invalid>")
            message = decision.get("message") or {}
            trust = [float(v) for v in
                     (decision.get("trust_row") or [])]
            out.append({"agent_id": agent, "action": action,
                        "label": label, "message": message,
                        "trust_row": trust})
        return {"schema": SCHEMA_VERSION, "decisions": out}

    def cycle(self, payload):
        """One real-world cycle from posted real telemetry (shadow-safe).

        Payload: {"topology": {...}, "batch": {...}} in the DTO shapes
        documented in ``CyberMarl.Deployment/README.md``. The attached
        pipeline's own collector is NOT consumed here; posted data runs
        exactly one adapted cycle through the frozen layers.
        """
        if self.real_pipeline is None:
            raise InferenceServiceError(
                "this sidecar serves /decide only (no pipeline "
                "attached)")
        if not isinstance(payload, dict) or "batch" not in payload:
            raise InferenceServiceError(
                "want {'topology': {...}, 'batch': {...}}")
        from .real_world import RealAsset, RealSegment, RealTopology
        from .telemetry import HostTelemetry, SecurityEvent, TelemetryBatch
        topology = _dto_topology(payload.get("topology") or {},
                                 RealAsset, RealSegment, RealTopology)
        batch = _dto_batch(payload["batch"], HostTelemetry,
                           SecurityEvent, TelemetryBatch)
        pipeline = self.real_pipeline
        previous_topology = pipeline.adapter.topology
        try:
            pipeline.adapter.topology = topology
            new_map, new_report = pipeline.adapter.build_policy_map(
                previous=pipeline.report)
            adapted = pipeline.adapter.adapt_batch(batch, new_map,
                                                   new_report)
            # Run ONE adapted batch through the frozen inner pipeline
            # without touching its collector stream.
            inner = pipeline.inner
            real_next = inner.collector.next_batch
            seen = {"used": False}

            def one_shot():
                if seen["used"]:
                    return None
                seen["used"] = True
                return adapted

            inner.collector.next_batch = one_shot
            try:
                records = inner.step()
            finally:
                inner.collector.next_batch = real_next
            plans = []
            for record in records:
                from .real_actions import translate_decision
                plans.append(translate_decision(
                    record["agent_id"], record["action_index"],
                    pipeline.tables, new_map, new_report))
            return {"schema": SCHEMA_VERSION,
                    "mapping": new_report.to_dict(),
                    "real_topology": topology_echo(topology, batch),
                    "records": [_jsonable(r) for r in records],
                    "real_actions": [_plan_json(p) for p in plans]}
        finally:
            pipeline.adapter.topology = previous_topology

    def approve(self, payload):
        """Approve one queued decision exactly once (delegated)."""
        if self.real_pipeline is None:
            raise InferenceServiceError(
                "this sidecar serves /decide only (no pipeline "
                "attached)")
        if not isinstance(payload, dict) or "index" not in payload:
            raise InferenceServiceError(
                "want {'index': int, 'approver': str?}")
        try:
            index = int(payload["index"])
        except (TypeError, ValueError):
            raise InferenceServiceError(
                f"approval index must be an int, got "
                f"{payload.get('index')!r}")
        approver = str(payload.get("approver", "csharp-operator"))
        try:
            result = self.real_pipeline.approve_pending(
                index, approver=approver)
        except IndexError as exc:
            raise InferenceServiceError(str(exc)) from exc
        except Exception as exc:
            raise InferenceServiceError(
                f"approval failure: {type(exc).__name__}: "
                f"{exc}") from exc
        return {"schema": SCHEMA_VERSION,
                "operation": result.operation,
                "target": result.target,
                "applied": bool(result.applied),
                "queued": bool(result.queued),
                "duplicate": bool(result.duplicate),
                "backend": result.backend,
                "error": result.error,
                "details": result.details}


def _dto_topology(dto, RealAsset, RealSegment, RealTopology):
    from .real_world import RealLink
    assets = []
    for item in dto.get("assets", []) or []:
        if not isinstance(item, dict) or not item.get("asset_id"):
            raise InferenceServiceError(
                "topology.assets[] need {asset_id, hostname}")
        assets.append(RealAsset(
            asset_id=str(item["asset_id"]),
            hostname=str(item.get("hostname") or item["asset_id"]),
            ips=tuple(item.get("ips", []) or ()),
            macs=tuple(item.get("macs", []) or ()),
            os_name=str(item.get("os_name", "")),
            os_version=str(item.get("os_version", "")),
            role=str(item.get("role", "")),
            segment_id=str(item.get("segment_id", "host-only")),
            services=tuple(item.get("services", []) or ()),
            coverage=str(item.get("coverage", "unknown")),
            coverage_detail=str(item.get("coverage_detail", ""))))
    segments = []
    for item in dto.get("segments", []) or []:
        if not isinstance(item, dict) or not item.get("segment_id"):
            raise InferenceServiceError(
                "topology.segments[] need {segment_id}")
        segments.append(RealSegment(
            segment_id=str(item["segment_id"]),
            name=str(item.get("name", item["segment_id"])),
            kind=str(item.get("kind", "unknown")),
            cidrs=tuple(item.get("cidrs", []) or ())))
    links = []
    for item in dto.get("links", []) or []:
        if not isinstance(item, dict) or not item.get("asset_id") \
                or not item.get("peer"):
            raise InferenceServiceError(
                "topology.links[] need {asset_id, peer}")
        via = str(item.get("via", "unknown"))
        if via not in ("arp", "subnet", "declared", "unknown"):
            raise InferenceServiceError(
                f"topology link via={via!r} is not a labeled "
                f"relationship (want arp|subnet|declared)")
        links.append(RealLink(asset_id=str(item["asset_id"]),
                              peer=str(item["peer"]), via=via))
    if not assets:
        raise InferenceServiceError("topology has no assets")
    return RealTopology(assets=assets, segments=segments,
                        links=links,
                        source=str(dto.get("source", "api")),
                        discovered_at=float(
                            dto.get("discovered_at", 0.0)))


def topology_echo(topology, batch):
    """Server-side real-topology view for one cycle (UI source).

    Services merge from the observed batch (what the cycle SAW);
    coverage/links/segments come from topology. CC4 slots never
    appear here -- see the separate "mapping" object.
    """
    batch_hosts = getattr(batch, "hosts", {}) or {}
    assets = []
    for asset in topology.assets or []:
        tele = batch_hosts.get(asset.asset_id)
        if tele is not None:
            services = sorted(set(map(str, list(
                getattr(tele, "services", []) or []))))
            observed = True
        else:
            services = sorted(set(map(str, list(
                getattr(asset, "services", []) or ()))))
            observed = False
        assets.append({
            "asset_id": asset.asset_id, "hostname": asset.hostname,
            "ips": list(asset.ips or ()), "macs": list(
                asset.macs or ()),
            "os_name": asset.os_name, "role": asset.role,
            "segment_id": asset.segment_id,
            "coverage": getattr(asset, "coverage", "unknown"),
            "coverage_detail": getattr(asset, "coverage_detail", ""),
            "services": services, "observed_this_cycle": observed})
    return {
        "assets": assets,
        "segments": [{"segment_id": s.segment_id, "name": s.name,
                      "kind": s.kind, "cidrs": list(s.cidrs or ())}
                     for s in topology.segments or []],
        "links": [{"asset_id": l.asset_id, "peer": l.peer,
                   "via": l.via}
                  for l in getattr(topology, "links", []) or []],
        "source": topology.source,
        "discovered_at": topology.discovered_at}


def _dto_batch(dto, HostTelemetry, SecurityEvent, TelemetryBatch):
    if not isinstance(dto, dict) or not isinstance(
            dto.get("hosts"), dict):
        raise InferenceServiceError("batch needs {hosts: {...}}")
    hosts = {}
    for key, item in dto["hosts"].items():
        if not isinstance(item, dict):
            raise InferenceServiceError(
                f"batch host {key!r} must be an object")
        events = []
        for event in item.get("events", []) or []:
            if not isinstance(event, dict) or "kind" not in event:
                raise InferenceServiceError(
                    f"batch host {key!r} has a malformed event")
            events.append(SecurityEvent(
                kind=str(event["kind"]),
                severity=str(event.get("severity", "low")),
                details=str(event.get("details", "")),
                timestamp=float(event.get("timestamp", 0.0))))
        hosts[str(key)] = HostTelemetry(
            key=str(key),
            processes=list(item.get("processes", []) or []),
            connections=list(item.get("connections", []) or []),
            sessions=list(item.get("sessions", []) or []),
            up=bool(item.get("up", True)), events=events,
            services=list(item.get("services", []) or []))
    partial = dto.get("partial_errors", []) or []
    if not isinstance(partial, list):
        raise InferenceServiceError("partial_errors must be a list")
    return TelemetryBatch(
        timestamp=float(dto.get("timestamp", 0.0)),
        hosts=hosts, notes=str(dto.get("notes", "")),
        source=str(dto.get("source", "api")),
        partial_errors=partial)


def _jsonable(record):
    try:
        return {k: (_scalar(v)) for k, v in dict(record).items()}
    except Exception:
        return {"unserializable": True}


def _scalar(value):
    import numpy as _np
    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, (_np.integer,)):
        return int(value)
    if isinstance(value, (_np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar(v) for v in value]
    return value


def _plan_json(plan):
    return {"asset_id": plan.asset_id, "hostname": plan.hostname,
            "ips": list(plan.ips or ()),
            "operation": plan.operation,
            "params": dict(plan.params or {}), "risk": plan.risk,
            "requires_approval": bool(plan.requires_approval),
            "blocked": bool(plan.blocked),
            "block_reason": plan.block_reason,
            "policy_provenance": dict(plan.policy_provenance or {}),
            "description": plan.describe()}


def token_fingerprint(token):
    """Log-safe token identifier (first 8 hex of sha256)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


__all__ = ["InferenceService", "InferenceServiceError",
           "SCHEMA_VERSION", "MAX_BODY_BYTES", "token_fingerprint",
           "topology_echo"]
