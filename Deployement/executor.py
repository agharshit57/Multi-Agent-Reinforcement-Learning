"""Executors: what happens AFTER validation (shadow/mock/supervised).

No executor touches the real network. Real enforcement connectors
(EDR isolate API, firewall ACL pushes, ticketing) are explicit future
work (see README "Blockers"); each op documents its live equivalent.

Enforcement state (blocks, isolations, remediations, decoys) feeds back
into the next cycle's observation via ``as_enforcement()``, so mock
mode behaves like a closed loop:
  - isolate_zone_traffic  -> adds (from_zone -> to_zone) block
  - allow_zone_traffic    -> removes matching blocks
  - terminate_suspicious  -> clears that host's alert flags (mock)
  - reimage_host          -> clears flags + records remediation
  - deploy_honeypot       -> records decoy (no flag change)
  - collect_*/noop        -> no state change
"""

from dataclasses import dataclass, field
import uuid

from .config import OP_TIMEOUT_S, DEFAULT_OP_TIMEOUT_S


def stamp_approval_id(decision):
    """Stable identity for one queued approval (position-independent).

    The server-side queue persists across cycles while UI list positions
    do not: approving by queue POSITION can pop the wrong decision after
    another cycle queues more entries. Every queued entry therefore
    carries an ``approval_id`` minted once here (preserved across
    approve/re-queue retries via setdefault) that approval calls must
    reference instead of positions.
    """
    entry = dict(decision)
    entry.setdefault("approval_id", uuid.uuid4().hex[:12])
    return entry


@dataclass
class ExecutionResult:
    operation: str
    target: str
    mode: str
    applied: bool
    details: str = ""
    error: str = ""        # non-empty iff the operation FAILED
    duplicate: bool = False  # True when deduped via idempotency key
    backend: str = ""      # backend name that handled it (""/mock/live)
    queued: bool = False   # True when parked for human approval. Status
    # logic must use this flag, never substring-match details text.


@dataclass
class EnforcementState:
    blocks: dict = field(default_factory=dict)  # to_zone -> set(from_zone)
    isolated_hosts: set = field(default_factory=set)
    remediated_hosts: set = field(default_factory=set)
    decoys: set = field(default_factory=set)
    cleared_flags: set = field(default_factory=set)  # mock: cc4 hosts
    log: list = field(default_factory=list)  # legacy tuple log (compat)
    pending_approvals: list = field(default_factory=list)
    audit: list = field(default_factory=list)  # structured dicts (item 15)

    def as_enforcement(self):
        return {"blocks": {k: sorted(v) for k, v in self.blocks.items()}}

    def audit_record(self, record):
        """Append a structured audit entry (never raises)."""
        try:
            entry = dict(record)
            entry.setdefault("audit_ts", __import__("time").time())
            self.audit.append(entry)
        except Exception:
            pass


class BackendError(RuntimeError):
    """A real enforcement operation failed (never silent success)."""


class BackendUnavailable(BackendError):
    """No functioning backend wired (e.g. NullBackend in live mode)."""


class RollbackUnsupported(BackendError):
    """This operation has no safe automatic inverse; manual required."""


@dataclass
class BackendResult:
    applied: bool
    details: str = ""
    error: str = ""


class EnforcementBackend:
    """Interface for REAL enforcement (vendor adapters implement this).

    Method contract (all of them):
      - accept ``timeout_s`` (must be enforced by real backends) and a
        stable ``idempotency_key`` (same key twice => single effect);
      - return BackendResult, or raise BackendError on ANY failure;
      - never report success unless the effect was confirmed;
      - record every attempt in ``self.audit`` (structured dicts).
    Rollback: ``rollback(audit_record)`` best-effort inverts a recorded
    effect where technically possible, else raises RollbackUnsupported.
    """

    name = "base"

    def __init__(self):
        self.audit = []

    # -- enforcing operations (override in adapters) --
    def noop(self, timeout_s, idempotency_key):
        return BackendResult(applied=False, details="noop")

    def collect_status(self, asset, timeout_s, idempotency_key):
        raise NotImplementedError

    def collect_forensics(self, asset, timeout_s, idempotency_key):
        raise NotImplementedError

    def deploy_honeypot(self, asset, timeout_s, idempotency_key):
        raise NotImplementedError

    def terminate_suspicious(self, asset, timeout_s, idempotency_key):
        raise NotImplementedError

    def reimage_host(self, asset, timeout_s, idempotency_key):
        raise NotImplementedError

    def set_zone_block(self, from_zone, to_zone, timeout_s,
                       idempotency_key):
        raise NotImplementedError

    def clear_zone_block(self, from_zone, to_zone, timeout_s,
                         idempotency_key):
        raise NotImplementedError

    def rollback(self, audit_record):
        raise RollbackUnsupported(
            f"{self.name}: no automatic rollback; manual required")


class NullBackend(EnforcementBackend):
    """Refuses everything, loudly. The safe default for live mode."""

    name = "null"

    def _refuse(self, op):
        raise BackendUnavailable(
            f"live mode has no enforcement backend wired for {op}; "
            f"supply one (see README 'Blockers') -- nothing was executed")

    def collect_status(self, asset, timeout_s, idempotency_key):
        self._refuse("collect_status")

    def collect_forensics(self, asset, timeout_s, idempotency_key):
        self._refuse("collect_forensics")

    def deploy_honeypot(self, asset, timeout_s, idempotency_key):
        self._refuse("deploy_honeypot")

    def terminate_suspicious(self, asset, timeout_s, idempotency_key):
        self._refuse("terminate_suspicious")

    def reimage_host(self, asset, timeout_s, idempotency_key):
        self._refuse("reimage_host")

    def set_zone_block(self, from_zone, to_zone, timeout_s,
                       idempotency_key):
        self._refuse("set_zone_block")

    def clear_zone_block(self, from_zone, to_zone, timeout_s,
                         idempotency_key):
        self._refuse("clear_zone_block")


class BaseExecutor:
    name = "base"

    def __init__(self, state=None):
        self.state = state or EnforcementState()

    def execute(self, decision):
        raise NotImplementedError


class ShadowExecutor(BaseExecutor):
    """Log-only. Never mutates enforcement state."""

    name = "shadow"

    def execute(self, decision):
        self.state.log.append(("shadow", decision["operation"],
                               str(decision.get("target"))))
        self.state.audit_record({"event": "shadowed",
                                 "operation": decision["operation"],
                                 "target": str(decision.get("target")),
                                 "mode": "shadow", "applied": False})
        return ExecutionResult(operation=decision["operation"],
                               target=str(decision.get("target")),
                               mode="shadow", applied=False,
                               details="recorded, no effect")


class MockExecutor(BaseExecutor):
    """Simulated effects on internal enforcement state."""

    name = "mock"

    def execute(self, decision):
        op = decision["operation"]
        target = decision.get("target")
        zone = decision.get("zone")
        applied = True
        if op == "isolate_zone_traffic":
            self.state.blocks.setdefault(zone, set()).add(
                str(target).lower())
            details = f"mock block {target} -> {zone}"
        elif op == "allow_zone_traffic":
            if zone in self.state.blocks:
                self.state.blocks[zone].discard(str(target).lower())
                if not self.state.blocks[zone]:
                    del self.state.blocks[zone]
            details = f"mock allow {target} -> {zone}"
        elif op == "terminate_suspicious":
            self.state.cleared_flags.add(target)
            self.state.isolated_hosts.add(target)
            details = f"mock session/process termination on {target}"
        elif op == "reimage_host":
            self.state.cleared_flags.add(target)
            self.state.remediated_hosts.add(target)
            details = f"mock reimage of {target}"
        elif op == "deploy_honeypot":
            self.state.decoys.add(target)
            details = f"mock honeypot on {target}"
        elif op in ("noop", "collect_status", "collect_forensics"):
            applied = False
            details = "read-only, no state change"
        else:
            self.state.audit_record({"event": "failed",
                                     "operation": op,
                                     "target": str(target), "mode": "mock",
                                     "applied": False,
                                     "error": f"unknown operation {op!r}"})
            raise ValueError(f"unknown operation {op!r}")
        self.state.log.append(("mock", op, str(target)))
        self.state.audit_record({"event": "executed", "operation": op,
                                 "target": str(target), "mode": "mock",
                                 "applied": applied, "details": details})
        return ExecutionResult(operation=op, target=str(target),
                               mode="mock", applied=applied, details=details)


class SupervisedExecutor(MockExecutor):
    """Supervised mode: only approved (or inherently safe) decisions apply.

    Unapproved destructive decisions are queued in
    ``state.pending_approvals`` for the GUI/human operator.
    """

    name = "supervised"

    def __init__(self, state=None):
        super().__init__(state)
        if not hasattr(self.state, "pending_approvals"):
            self.state.pending_approvals = []

    def execute(self, decision):
        if decision.get("needs_approval") and not decision.get("approved"):
            entry = stamp_approval_id(decision)
            self.state.pending_approvals.append(entry)
            self.state.log.append(("queued", decision["operation"],
                                   str(decision.get("target"))))
            self.state.audit_record({"event": "queued",
                                     "operation": decision["operation"],
                                     "target": str(decision.get("target")),
                                     "approval_id": entry["approval_id"],
                                     "mode": "supervised",
                                     "applied": False})
            return ExecutionResult(operation=decision["operation"],
                                   target=str(decision.get("target")),
                                   mode="supervised", applied=False,
                                   details="queued for human approval",
                                   queued=True)
        return super().execute(decision)


class MockBackend(EnforcementBackend):
    """In-memory backend for integration tests and mock-mode parity.

    Applies the same effects as MockExecutor to an EnforcementState,
    with real idempotency-key dedupe, an audit trail, and rollback for
    zone blocks (the only safely invertible effect; host remediation
    has no automatic inverse and raises RollbackUnsupported).
    Timeouts are accepted and recorded (instant backend -- real
    backends must enforce them for real).
    """

    name = "mock"

    def __init__(self, state=None, resolve=None):
        """``resolve`` maps a real asset identity (ip/hostname) back to
        its CC4 slot name for pipeline-facing state (``cleared_flags``,
        ``isolated_hosts``, ... are keyed by CC4 name, exactly what the
        observation/mask path reads). Pass None to key state by the raw
        identity (legacy unit-test behavior)."""
        super().__init__()
        self.state = state or EnforcementState()
        self._resolve = resolve
        self._applied_keys = {}

    def _state_key(self, identity):
        if self._resolve is not None:
            try:
                mapped = self._resolve(identity)
            except Exception:
                mapped = None
            if mapped:
                return mapped
        return identity

    def _once(self, key, work, undo=None):
        # Dedupe ONLY keys whose first attempt verifiably applied. A
        # failed attempt must never poison the key: otherwise a retry
        # would report success ("duplicate suppressed") for an effect
        # that never happened.
        if key in self._applied_keys:
            record = dict(self._applied_keys[key])
            record["duplicate"] = True
            self.audit.append(record)
            return BackendResult(applied=True, details=record["details"]
                                 + " [duplicate suppressed]"), True
        result = work()
        record = {"key": key, "applied": bool(result.applied),
                  "details": result.details, "error": result.error,
                  "undo": undo, "duplicate": False}
        if result.applied:
            self._applied_keys[key] = record
        self.audit.append(record)
        return result, False

    def _ok(self, details):
        return BackendResult(applied=True, details=details)

    def collect_status(self, asset, timeout_s, idempotency_key):
        result, _dup = self._once(
            idempotency_key,
            lambda: self._ok(f"mock status collected for {asset}"))
        return result

    def collect_forensics(self, asset, timeout_s, idempotency_key):
        result, _dup = self._once(
            idempotency_key,
            lambda: self._ok(f"mock forensics collected for {asset}"))
        return result

    def deploy_honeypot(self, asset, timeout_s, idempotency_key):
        def work():
            self.state.decoys.add(self._state_key(asset))
            return self._ok(f"mock honeypot on {asset}")
        result, _dup = self._once(idempotency_key, work)
        return result

    def terminate_suspicious(self, asset, timeout_s, idempotency_key):
        def work():
            key = self._state_key(asset)
            self.state.cleared_flags.add(key)
            self.state.isolated_hosts.add(key)
            return self._ok(f"mock termination on {asset}")
        result, _dup = self._once(idempotency_key, work)
        return result

    def reimage_host(self, asset, timeout_s, idempotency_key):
        def work():
            key = self._state_key(asset)
            self.state.cleared_flags.add(key)
            self.state.remediated_hosts.add(key)
            return self._ok(f"mock reimage of {asset}")
        result, _dup = self._once(idempotency_key, work)
        return result

    def set_zone_block(self, from_zone, to_zone, timeout_s,
                       idempotency_key):
        def work():
            self.state.blocks.setdefault(to_zone, set()).add(from_zone)
            return self._ok(f"mock block {from_zone} -> {to_zone}")
        result, _dup = self._once(idempotency_key, work,
                                  undo=("clear_zone_block", from_zone,
                                        to_zone))
        return result

    def clear_zone_block(self, from_zone, to_zone, timeout_s,
                         idempotency_key):
        def work():
            if to_zone in self.state.blocks:
                self.state.blocks[to_zone].discard(from_zone)
                if not self.state.blocks[to_zone]:
                    del self.state.blocks[to_zone]
            return self._ok(f"mock allow {from_zone} -> {to_zone}")
        result, _dup = self._once(idempotency_key, work,
                                  undo=("set_zone_block", from_zone,
                                        to_zone))
        return result

    def rollback(self, audit_record):
        undo = (audit_record or {}).get("undo")
        if not undo:
            raise RollbackUnsupported(
                "mock: no automatic inverse recorded; manual required")
        kind, from_zone, to_zone = undo
        if kind == "clear_zone_block":
            self.clear_zone_block(from_zone, to_zone, DEFAULT_OP_TIMEOUT_S,
                                  f"rollback:{audit_record.get('key')}")
        elif kind == "set_zone_block":
            self.set_zone_block(from_zone, to_zone, DEFAULT_OP_TIMEOUT_S,
                                f"rollback:{audit_record.get('key')}")
        else:  # pragma: no cover - defensive (undo only ever block ops)
            raise RollbackUnsupported(f"mock: unknown undo {kind!r}")
        return True


class LiveExecutor(BaseExecutor):
    """Live mode: approved decisions execute via an EnforcementBackend.

    Safety rules (hard):
      - unapproved approval-gated decisions are QUEUED, never executed;
      - backend failures surface as applied=False + error (never
        silent success);
      - destructive ops are attempted ONCE (no blind retries; the
        idempotency key lets the backend dedupe, the client never
        re-fires);
      - read-only ops may retry twice on BackendError.
    Without an explicit backend (default NullBackend) every enforcing
    op fails loudly -- live mode can never silently do nothing.
    """

    name = "live"

    def __init__(self, state=None, backend=None):
        super().__init__(state)
        self.backend = backend or NullBackend()
        if not hasattr(self.state, "pending_approvals"):
            self.state.pending_approvals = []

    def _timeout(self, operation):
        return float(OP_TIMEOUT_S.get(operation, DEFAULT_OP_TIMEOUT_S))

    @staticmethod
    def _real_asset(decision):
        """Resolved real-world asset identity for enforcement.

        Prefers the bound asset's IP, then its hostname, and only then
        the CC4 slot name. Host-enforcement backend methods MUST receive
        this (never the simulated CC4 hostname) -- read-only ops already
        did; see issue: simulated target leaking into live enforcement.
        """
        asset = decision.get("asset") or {}
        for field in ("ip", "hostname"):
            value = asset.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return decision.get("target")

    def _dispatch(self, decision):
        op = decision["operation"]
        asset = self._real_asset(decision)
        key = decision.get("idempotency_key") or (
            f"{decision.get('agent_id')}:{op}:{decision.get('zone')}:"
            f"{decision.get('target')}:{decision.get('timestamp')}")
        timeout_s = self._timeout(op)
        backend = self.backend
        if op == "noop":
            return BackendResult(applied=False, details="noop")
        if op == "collect_status":
            return backend.collect_status(asset, timeout_s, key)
        if op == "collect_forensics":
            return backend.collect_forensics(asset, timeout_s, key)
        if op == "deploy_honeypot":
            return backend.deploy_honeypot(asset, timeout_s, key)
        if op == "terminate_suspicious":
            return backend.terminate_suspicious(asset, timeout_s, key)
        if op == "reimage_host":
            return backend.reimage_host(asset, timeout_s, key)
        if op == "isolate_zone_traffic":
            return backend.set_zone_block(str(decision.get("target")).lower(),
                                          decision.get("zone"), timeout_s,
                                          key)
        if op == "allow_zone_traffic":
            return backend.clear_zone_block(
                str(decision.get("target")).lower(), decision.get("zone"),
                timeout_s, key)
        raise BackendError(f"unknown operation {op!r}")

    def execute(self, decision):
        asset = self._real_asset(decision)
        if decision.get("needs_approval") and not decision.get("approved"):
            entry = stamp_approval_id(decision)
            self.state.pending_approvals.append(entry)
            self.state.log.append(("queued", decision["operation"],
                                   str(decision.get("target"))))
            self.state.audit_record({"event": "queued",
                                     "operation": decision["operation"],
                                     "target": str(decision.get("target")),
                                     "asset": asset,
                                     "approval_id": entry["approval_id"],
                                     "mode": "live", "applied": False})
            return ExecutionResult(operation=decision["operation"],
                                   target=str(decision.get("target")),
                                   mode="live", applied=False,
                                   details="queued for human approval",
                                   backend=self.backend.name,
                                   queued=True)
        read_only = decision["operation"] in (
            "noop", "collect_status", "collect_forensics")
        attempts = 3 if read_only else 1
        last_error = ""
        for _ in range(attempts):
            try:
                result = self._dispatch(decision)
                self.state.audit_record(
                    {"event": "executed", "operation": decision["operation"],
                     "target": str(decision.get("target")), "asset": asset,
                     "mode": "live",
                     "applied": bool(result.applied),
                     "details": result.details, "error": result.error,
                     "backend": self.backend.name})
                return ExecutionResult(
                    operation=decision["operation"],
                    target=str(decision.get("target")), mode="live",
                    applied=bool(result.applied), details=result.details,
                    error=result.error, backend=self.backend.name)
            except BackendError as exc:
                last_error = str(exc)
                if not read_only:
                    break
        self.state.audit_record(
            {"event": "failed", "operation": decision["operation"],
             "target": str(decision.get("target")), "asset": asset,
             "mode": "live",
             "applied": False, "error": last_error,
             "backend": self.backend.name})
        return ExecutionResult(operation=decision["operation"],
                               target=str(decision.get("target")),
                               mode="live", applied=False,
                               details="backend failure (see error)",
                               error=last_error,
                               backend=self.backend.name)


def make_executor(mode, state=None, backend=None):
    if mode == "shadow":
        return ShadowExecutor(state)
    if mode == "mock":
        return MockExecutor(state)
    if mode == "supervised":
        return SupervisedExecutor(state)
    if mode == "live":
        return LiveExecutor(state, backend=backend)
    raise ValueError(f"unknown mode {mode!r}")


def load_backend(spec, state=None):
    """Load an EnforcementBackend from configuration (no code changes).

    ``spec`` may be:
      - ``"null"`` (or None/blank) -> NullBackend (refuses everything);
      - ``"mock"`` -> MockBackend (in-memory simulation);
      - an ``EnforcementBackend`` instance -> used as-is;
      - ``"module:Class"`` -> import ``module``, instantiate ``Class``
        with NO arguments (adapters read their own credentials from
        the environment), and require an EnforcementBackend instance.

    Anything else -- unimportable modules, missing attributes,
    non-backend objects, constructors needing arguments -- raises
    BackendError with an actionable message. Callers must surface that
    error instead of silently falling back to mock/null.
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return NullBackend()
    if isinstance(spec, EnforcementBackend):
        return spec
    if isinstance(spec, str):
        lowered = spec.strip().lower()
        if lowered == "null":
            return NullBackend()
        if lowered == "mock":
            return MockBackend(state)
        if ":" in spec:
            module_name, _, attr = spec.partition(":")
            if not module_name or not attr:
                raise BackendError(
                    f"invalid backend spec {spec!r}: want 'module:Class'")
            try:
                import importlib
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise BackendError(
                    f"cannot import backend module {module_name!r} "
                    f"from spec {spec!r}: {exc}") from exc
            try:
                factory = getattr(module, attr)
            except AttributeError:
                raise BackendError(
                    f"backend module {module_name!r} has no attribute "
                    f"{attr!r} (spec {spec!r})") from None
            if isinstance(factory, EnforcementBackend):
                return factory
            if not callable(factory):
                raise BackendError(
                    f"backend {spec!r} is not callable and not an "
                    f"EnforcementBackend instance")
            try:
                instance = factory()
            except TypeError as exc:
                raise BackendError(
                    f"backend {spec!r} must be constructible with no "
                    f"arguments (adapters read config from the "
                    f"environment): {exc}") from exc
            except Exception as exc:
                raise BackendError(
                    f"backend {spec!r} failed to construct: "
                    f"{exc}") from exc
            if not isinstance(instance, EnforcementBackend):
                raise BackendError(
                    f"backend {spec!r} produced "
                    f"{type(instance).__name__}, not an "
                    f"EnforcementBackend")
            return instance
    raise BackendError(
        f"invalid backend spec {spec!r}: want 'null', 'mock', an "
        f"EnforcementBackend instance, or 'module:Class'")
