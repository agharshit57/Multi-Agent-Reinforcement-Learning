"""Deployment pipeline: one closed loop over the full architecture.

Per cycle::
    telemetry batch -> normalize -> apply mock remediation effects
    -> CC4 state (+ enforcement blocks) -> observation batch [5, 210]
    -> action masks -> policy.decide -> validate -> execute
    -> record DecisionRecord (JSONL)

Communication timing mirrors training: messages generated this cycle
are delivered next cycle (the engines own that memory; ``reset()``
clears it at session start). Trust stays frozen (production has no
ground truth) -- same rule as evaluation.
"""

import json
import os
import threading
import time
import uuid

import numpy as np

from .action_mask import compute_mask, pad_mask
from .action_table import build_all_tables
from .config import (ACTION_DIM, AGENT_ZONES, NUM_AGENTS,
                     NUM_HOST_TARGETS, STABLE_HOST_INDEX)
from .executor import (EnforcementState, ExecutionResult, MockBackend,
                       load_backend, make_executor)
from .live import CollectorError, CollectorHealth
from .normalizer import TelemetryNormalizer
from .observation import ObservationBuilder
from .state_builder import CC4StateBuilder
from .validator import ActionValidator


class DecisionRecord(dict):
    pass


# In-memory history cap: the durable record is the JSONL log (+ the
# executor audit trail), so a multi-day console cannot grow RAM
# without bound. Sized for ~4000 cycles x 5 agents.
HISTORY_MAX_RECORDS = 20000

# JSONL rotation threshold: decisions.jsonl rolls to decisions.jsonl.1
# (single backup, oldest data dropped) instead of growing a disk
# without bound. 0/None disables rotation (tests, tiny runs).
DEFAULT_MAX_LOG_BYTES = 50 * 1024 * 1024


class DeploymentPipeline:
    def __init__(self, asset_map, collector, policy_engine,
                 mode="shadow", baselines=None, mission_phase=0,
                 cooldown_s=300, session_dir=None, stale_after_s=300,
                 live_backend="null", max_log_bytes=DEFAULT_MAX_LOG_BYTES,
                 history_max=HISTORY_MAX_RECORDS, policy_info=None):
        self.asset_map = asset_map
        self.collector = collector
        self.policy = policy_engine
        # Optional describe dict (checkpoint path, vocab, engine name)
        # for display surfaces; never affects decisions.
        self.policy_info = dict(policy_info or {})
        self.tables = build_all_tables(asset_map)
        self.normalizer = TelemetryNormalizer(
            baselines, stale_after_s=stale_after_s)
        self.state_builder = CC4StateBuilder(mission_phase)
        self.validator = ActionValidator(mode=mode, cooldown_s=cooldown_s)
        self.enforcement = EnforcementState()
        self.live_backend = live_backend
        # Validate the backend spec SHAPE now (all modes): typos like
        # "moc" fail fast here; adapter imports resolve when live mode
        # constructs (loud BackendError either way, never silent).
        from .executor import EnforcementBackend as _BackendABC
        _spec_ok = (
            isinstance(live_backend, _BackendABC)
            or (isinstance(live_backend, str)
                and (live_backend.strip().lower() in ("null", "mock")
                     or ":" in live_backend)))
        if not _spec_ok:
            from .executor import BackendError
            raise BackendError(
                f"invalid enforcement backend {live_backend!r}: want "
                f"'null', 'mock', an EnforcementBackend instance, or "
                f"'module:Class'")
        # Concurrency model: single-owner event loop (GUI main thread /
        # headless loop) PLUS this re-entrant lock as a backstop, so a
        # background poller or a second operator console cannot corrupt
        # approvals, cooldowns, idempotency keys, or enforcement state.
        # All public mutating entry points serialize on it.
        self._lock = threading.RLock()
        self._reverse_index = self._build_reverse_index()
        self.executor = self._make_executor(mode)
        self.observer = ObservationBuilder(asset_map)
        self.session_dir = session_dir
        self.session_id = uuid.uuid4().hex[:12]
        self.health = CollectorHealth()
        self.last_unseen = []
        self.cycle = 0
        self.history = []
        self.history_max = history_max
        self.max_log_bytes = max_log_bytes
        # Latest-cycle snapshots for display surfaces (host details,
        # technical expandables). Read-only views; never fed back into
        # decisions.
        self.last_batch = None
        self.last_normalized = None
        self.last_obs = None
        self.last_masks = None
        self.last_policy_error = ""
        self._log_path = None
        self._log_fh = None
        if session_dir:
            os.makedirs(session_dir, exist_ok=True)
            self._log_path = os.path.join(session_dir, "decisions.jsonl")
            self._log_fh = open(self._log_path, "a", encoding="utf-8")

    def _build_reverse_index(self):
        """Real identity -> CC4 slot (ips AND hostnames, 1:1 by load)."""
        index = {}
        for agent in self.asset_map.agents:
            for cc4, info in self.asset_map.agents[agent]["hosts"].items():
                for key in (info.get("ip", ""), info.get("hostname", "")):
                    if key and key not in index:
                        index[key] = cc4
        return index

    def _resolve_asset(self, identity):
        """Map a real asset identity back to its CC4 slot (or None)."""
        if not isinstance(identity, str) or not identity:
            return None
        return self._reverse_index.get(identity.strip())

    def _make_executor(self, mode):
        # live_backend accepts "null" | "mock" | "module:Class" |
        # EnforcementBackend instance. Anything else fails loudly here
        # (construction time), never as a silent mid-run fallback.
        # NOTE: a caller-supplied backend INSTANCE is reused across
        # reset() (it is caller-owned, including its idempotency
        # memory); "null"/"mock"/spec strings always get fresh objects.
        if mode == "live":
            spec = self.live_backend
            if isinstance(spec, str) and spec.strip().lower() == "mock":
                return make_executor(
                    mode, self.enforcement,
                    backend=MockBackend(
                        self.enforcement, resolve=self._resolve_asset))
            return make_executor(mode, self.enforcement,
                                 backend=load_backend(spec))
        return make_executor(mode, self.enforcement)

    # ------------------------------------------------------------- API --
    def set_mode(self, mode):
        with self._lock:
            self.validator.set_mode(mode)
            self.executor = self._make_executor(mode)

    def reset(self):
        """Start a genuinely new session: nothing leaks across.

        New session id; fresh enforcement state (+rebound executor),
        cleared validator cooldowns, cleared normalizer freshness/
        compromise memory, cleared approvals, reset policy memory,
        fresh collector health. The JSONL log (if any) stays open and
        keeps appending -- records are separated by session_id.
        """
        with self._lock:
            self.session_id = uuid.uuid4().hex[:12]
            self.enforcement = EnforcementState()
            self.executor = self._make_executor(self.validator.mode)
            self.validator.reset()
            self.normalizer.reset()
            self.health = CollectorHealth()
            self.last_unseen = []
            self.cycle = 0
            self.history = []
            self.last_batch = None
            self.last_normalized = None
            self.last_obs = None
            self.last_masks = None
            self.last_policy_error = ""
            self.policy.reset()

    def health_summary(self):
        return {"health": self.health.as_dict(),
                "unseen_keys": list(self.last_unseen),
                "session_id": self.session_id,
                "cycle": self.cycle}

    def _next_batch(self):
        """Fetch telemetry, tracking collector health (never quiet-fail).

        Returns (batch, end_of_stream). End-of-stream (exhausted batch
        collectors) returns (None, True) and the caller stops. Any
        CollectorError -- or a None from a NON-exhausted (live)
        collector -- synthesizes an all-stale batch so the cycle sees
        UNKNOWN, never healthy/quiet.
        """
        from .telemetry import TelemetryBatch
        try:
            batch = self.collector.next_batch()
        except CollectorError as exc:
            self.health.record_failure(exc)
            return self._stale_batch(f"collector failure: {exc}"), False
        except Exception as exc:  # never let a collector crash the loop
            self.health.record_failure(
                f"collector raised {type(exc).__name__}: {exc}")
            return self._stale_batch(f"collector raised: {exc}"), False
        if batch is None:
            if bool(getattr(self.collector, "exhausted", False)):
                return None, True
            self.health.record_failure(
                "live collector returned None (contract violation)")
            return self._stale_batch("live collector returned None"), False
        self.health.record_success(time.time())
        return batch, False

    def _stale_batch(self, reason):
        from .telemetry import TelemetryBatch
        return TelemetryBatch(timestamp=time.time(), hosts={},
                              notes=reason, source="synthesis-stale")

    def step(self):
        """Run one decision cycle. Returns list of DecisionRecords.

        Serialized on the pipeline lock (see __init__): concurrent
        step/approve/reset/set_mode calls cannot interleave.
        """
        with self._lock:
            return self._step_locked()

    def _step_locked(self):
        batch, end = self._next_batch()
        if batch is None and end:
            return []
        t0 = time.time()
        # Hosts with mock-confirmed remediation re-enter with a clean
        # compromise slate (a later intrusion event re-sets stickiness).
        # Everything else keeps last-known compromise until an explicit
        # recovery signal -- an intrusion never vanishes silently.
        normalized = self.normalizer.normalize(
            batch, self.asset_map,
            recovered=self.enforcement.remediated_hosts)
        self.last_unseen = list(normalized.unseen_keys)
        # Mock remediation effects: cleared hosts stop raising alerts.
        for cc4 in self.enforcement.cleared_flags:
            host = normalized.hosts.get(cc4)
            if host is not None and not host.compromised:
                host.process_event = False
                host.connection_event = False
        stale_hosts = sorted(
            cc4 for cc4, host in normalized.hosts.items()
            if host.health == "stale")
        state = self.state_builder.build(
            normalized, self.enforcement.as_enforcement())
        obs = self.observer.build_batch(state)
        masks, host_masks, valid_masks = self._masks(normalized)
        try:
            decisions = self.policy.decide(obs, masks,
                                           host_masks=host_masks,
                                           host_valid=valid_masks)
            policy_error = ""
        except Exception as exc:  # policy failure must not stop the cycle
            decisions = [{"action": -1, "probs_top": [], "message": None,
                          "trust_row": []} for _ in range(NUM_AGENTS)]
            policy_error = f"{type(exc).__name__}: {exc}"
        # Display snapshots (read-only; the next cycle rebuilds them).
        self.last_batch = batch
        self.last_normalized = normalized
        self.last_obs = obs
        self.last_masks = masks
        self.last_policy_error = policy_error
        # Rotate BEFORE writing: the live file always ends with the
        # latest records (never an empty just-rotated stub), while the
        # total on disk stays bounded at ~2x the threshold.
        if self._log_fh is not None:
            self._maybe_rotate_log()
        records = []
        for agent in range(NUM_AGENTS):
            # Per-agent handler latency (validate+execute for THIS
            # agent's record), not cumulative time since cycle start.
            agent_t0 = time.time()
            table = self.tables[agent]
            action = int(decisions[agent]["action"])
            validation = {"ok": True, "error": ""}
            approval = {"needed": False, "by": None}
            exec_info = {"applied": False, "error": "",
                         "backend": "", "duplicate": False}
            try:
                if policy_error:
                    raise RuntimeError(f"policy failure: {policy_error}")
                approved = self.validator.validate(
                    agent, action, table, masks[agent],
                    key_context=f"{self.session_id}:{self.cycle}")
                approval["needed"] = bool(approved.get("needs_approval"))
                if isinstance(approved.get("approver"), str):
                    approval["by"] = approved["approver"]
                result = self.executor.execute(approved)
                exec_info.update(
                    applied=bool(result.applied),
                    error=result.error or "",
                    backend=result.backend
                    or getattr(self.executor, "name", ""),
                    duplicate=bool(result.duplicate))
                status = ("applied" if result.applied
                          else ("queued" if result.queued
                                else "logged"))
                if result.error and not result.applied:
                    status = f"failed: {result.error}"
            except Exception as exc:  # never let one agent stop a cycle
                approved = None
                result = None
                validation = {"ok": False, "error": str(exc)}
                status = f"rejected: {exc}"
            record = DecisionRecord(
                session_id=self.session_id,
                timestamp=round(t0, 3),
                cycle=self.cycle, agent_id=agent,
                action_index=action,
                label=(table[action]["label"]
                       if 0 <= action < len(table) else "<invalid>"),
                command=(table[action]["command"]
                         if 0 <= action < len(table) else "?"),
                operation=(approved["operation"] if approved else None),
                risk=(self.validator.risk_of(table[action]["command"])
                      if 0 <= action < len(table) else "?"),
                status=status,
                validation=validation,
                approval=approval,
                exec=exec_info,
                health=self.health.state,
                stale_hosts=stale_hosts,
                unseen_keys=list(normalized.unseen_keys),
                message=decisions[agent].get("message"),
                latency_ms=round(
                    (time.time() - agent_t0) * 1000.0, 2),
                notes=getattr(batch, "notes", ""))
            records.append(record)
            self.history.append(record)
            if len(self.history) > self.history_max:
                del self.history[:-self.history_max]
            if self._log_fh is not None:
                from .live import scrub_secrets
                self._log_fh.write(
                    json.dumps(scrub_secrets(_jsonable(record))) + "\n")
        if self._log_fh is not None:
            self._log_fh.flush()
        self.cycle += 1
        return records

    def _maybe_rotate_log(self):
        """Roll decisions.jsonl -> decisions.jsonl.1 past the threshold."""
        if not self.max_log_bytes or self._log_path is None:
            return
        try:
            if os.path.getsize(self._log_path) < self.max_log_bytes:
                return
        except OSError:
            return  # file vanished/renamed externally; keep appending
        try:
            self._log_fh.close()
        except Exception:
            pass
        try:
            backup = self._log_path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self._log_path, backup)
        except OSError:
            pass  # rotation best-effort; logging continues below
        try:
            self._log_fh = open(self._log_path, "a", encoding="utf-8")
        except OSError:
            self._log_fh = None

    def run(self, max_cycles=None):
        """Run until the collector is exhausted or max_cycles reached."""
        out = []
        while True:
            if max_cycles is not None and self.cycle >= max_cycles:
                break
            records = self.step()
            if not records:
                break
            out.extend(records)
        return out

    def approve_pending(self, index=None, approver="human",
                        approval_id=None):
        """Approve one queued decision and execute it exactly once.

        Reference the entry EITHER by stable ``approval_id`` (preferred:
        queue positions shift as later cycles queue more entries, so a
        position captured by a UI in a previous cycle can alias a
        different, older decision) OR by current ``index`` (legacy;
        valid only against a freshly-read queue snapshot). Exactly one
        of the two must be given; unknown ids fail loudly instead of
        popping whatever happens to sit at a stale position.
        Single-ownership protocol (under the pipeline lock): the entry
        is popped first, so two concurrent approvers cannot both
        execute it. Outcomes:
          - approved + applied      -> returned, done (attempt audited
            by the executor);
          - approved + FAILED       -> the approved decision is
            RE-QUEUED at its index with an incremented ``attempts``
            counter plus a failure audit entry, and the failed result
            is returned. Nothing is lost; nothing auto-retries: the
            operator explicitly retries (re-approve) or denies. The
            unchanged idempotency key lets the backend dedupe if the
            first attempt actually applied server-side;
          - approve() itself errors (wrong mode, malformed entry) ->
            the entry is restored and the error propagates (loud,
            operator-visible; deny to clear it).
        """
        with self._lock:
            pending = self.enforcement.pending_approvals
            if approval_id is not None:
                # Resolve the STABLE id to its CURRENT position: the
                # queue persists across cycles, positions do not.
                matches = [i for i, entry in enumerate(pending)
                           if isinstance(entry, dict)
                           and entry.get("approval_id") == approval_id]
                if not matches:
                    raise IndexError(
                        f"no pending approval {approval_id!r} "
                        f"({len(pending)} queued; stale id or already "
                        f"handled -- refusing to pop by position)")
                index = matches[0]
            elif index is None:
                raise IndexError(
                    "approve_pending needs index= or approval_id=")
            if not (0 <= index < len(pending)):
                raise IndexError(
                    f"no pending approval at index {index}")
            decision = pending.pop(index)
            try:
                approved = self.validator.approve(decision, approver)
            except Exception:
                pending.insert(min(index, len(pending)), decision)
                raise
            approved["attempts"] = int(approved.get("attempts", 0)) + 1
            self.enforcement.audit_record(
                {"event": "approved",
                 "operation": approved.get("operation"),
                 "target": str(approved.get("target")),
                 "mode": self.validator.mode,
                 "approver": approver, "applied": False,
                 "attempts": approved["attempts"]})
            try:
                result = self.executor.execute(approved)
            except Exception as exc:
                pending.insert(min(index, len(pending)), approved)
                self.enforcement.audit_record(
                    {"event": "failed",
                     "operation": approved.get("operation"),
                     "target": str(approved.get("target")),
                     "mode": self.validator.mode, "applied": False,
                     "error": f"{type(exc).__name__}: {exc}",
                     "attempts": approved["attempts"]})
                return ExecutionResult(
                    operation=approved.get("operation"),
                    target=str(approved.get("target")),
                    mode=self.validator.mode, applied=False,
                    details="execution raised (see error); re-queued",
                    error=f"{type(exc).__name__}: {exc}")
            if not result.applied:
                pending.insert(min(index, len(pending)), approved)
                self.enforcement.audit_record(
                    {"event": "failed",
                     "operation": approved.get("operation"),
                     "target": str(approved.get("target")),
                     "mode": self.validator.mode, "applied": False,
                     "error": result.error or result.details,
                     "attempts": approved["attempts"]})
            return result

    def close(self):
        try:
            self.collector.close()
        except Exception:
            pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    # ---------------------------------------------------------- internals --
    def _masks(self, normalized):
        import numpy as np

        from .asset_map import match_zone
        masks, host_masks, valid_masks = [], [], []
        for agent in range(NUM_AGENTS):
            zones = AGENT_ZONES[agent]
            mask = compute_mask(self.tables[agent], normalized, zones)
            masks.append(pad_mask(mask, ACTION_DIM))
            hm = np.zeros((3, 16), dtype=bool)
            for slot, zone in enumerate(zones):
                for cc4 in sorted(
                        self.asset_map.agents[f"blue_agent_{agent}"]
                        ["hosts"]):
                    # Strict attribution (shared match_zone helper):
                    # never substring-match, never silently pick.
                    if match_zone(cc4) == zone and "router" not in cc4:
                        s, h = self.asset_map.slot_of(
                            f"blue_agent_{agent}", cc4)
                        hm[s, h] = True
            host_masks.append(hm)
            # Positions MUST follow STABLE_HOST_LIST (the decoder head
            # order), not a per-agent subset order.
            vm = np.zeros(NUM_HOST_TARGETS, dtype=bool)
            for c in self.asset_map.bound_hosts(f"blue_agent_{agent}"):
                if "router" not in c and c in STABLE_HOST_INDEX:
                    vm[STABLE_HOST_INDEX[c]] = True
            valid_masks.append(vm)
        return masks, np.stack(host_masks), np.stack(valid_masks)


def _jsonable(record):
    out = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        elif isinstance(value, (np.ndarray,)):
            value = value.tolist()
        out[key] = value
    return out
