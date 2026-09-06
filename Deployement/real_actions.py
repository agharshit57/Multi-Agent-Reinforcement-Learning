"""Real-world actions: policy decisions -> operations on REAL assets.

A MAPPO decision (agent, action_index, CC4 label) is ADVISORY INPUT.
Nothing executes until it becomes a ``RealActionPlan`` against physical
assets AND passes the unchanged validator/executor gates (modes,
masks, cooldowns, approvals, idempotency, backends).

Refusal rules (loud, never silent):

* policy target is an UNPOPULATED slot -> ``blocked`` plan ("phantom
  target"): no physical asset exists there, so there is nothing to
  reimage/isolate/forensicate. Previously such decisions addressed
  placeholder bindings as if they were infrastructure.
* zone operation affecting zero bound real assets -> ``blocked`` plan
  ("no physical footprint in this policy zone").
* unknown command / missing table entry -> ``blocked`` plan.

``PhantomGuardBackend`` wraps any ``EnforcementBackend`` as a second
net: host operations resolving to an unpopulated placeholder identity
(TEST-NET-3 or the ``unpopulated-policy-slot-`` hostname) raise
``BackendError`` instead of reaching vendor APIs. Defence in depth --
translation should already have blocked them.
"""

from dataclasses import dataclass, field

from .executor import BackendError, EnforcementBackend
from .policy_adapter import UNPOPULATED_HOSTNAME_PREFIX, UNPOPULATED_IP_PREFIX
from .validator import (DESTRUCTIVE_COMMANDS, INVESTIGATIVE_COMMANDS,
                        TRANSLATION, ActionValidator)


@dataclass
class RealActionPlan:
    """An executable-or-refused operation on the real network."""
    asset_id: str = ""            # "" = machine-local / no single target
    hostname: str = ""
    ips: tuple = ()
    operation: str = ""           # validator TRANSLATION value
    params: dict = field(default_factory=dict)
    risk: str = ""                # safe|elevated|destructive
    requires_approval: bool = False
    blocked: bool = False
    block_reason: str = ""
    policy_provenance: dict = field(default_factory=dict)
    # policy_provenance: {agent_id, action_index, command, kind,
    #                     cc4_target, policy_label, mapping_digest}

    def describe(self):
        if self.blocked:
            return f"BLOCKED: {self.block_reason}"
        target = self.hostname or self.asset_id or "local"
        return (f"{self.operation} on {target} "
                f"[{self.risk}"
                f"{', needs-approval' if self.requires_approval else ''}]")


def translate_decision(agent_id, action_index, tables, policy_map,
                       report):
    """Build the RealActionPlan for one policy decision.

    ``tables`` are the frozen action tables (``build_all_tables``);
    ``policy_map``/``report`` come from ``PolicyAdapter``. Never raises
    for policy content: untranslatable decisions become blocked plans.
    """
    digest = getattr(report, "digest", "")
    try:
        agent = int(agent_id)
        index = int(action_index)
        table = tables[agent]
        if not 0 <= index < len(table):
            raise IndexError(
                f"action {index} out of range [0, {len(table)})")
        entry = table[index]
    except Exception as exc:
        return RealActionPlan(
            blocked=True, block_reason=f"unknown policy action: {exc}",
            policy_provenance={"agent_id": agent_id,
                               "action_index": action_index,
                               "mapping_digest": digest})
    command = entry.get("command", "")
    kind = entry.get("kind", "none")
    operation = TRANSLATION.get(command, "")
    if not operation:
        return RealActionPlan(
            blocked=True,
            block_reason=f"no real-world translation for {command!r}",
            policy_provenance=_provenance(
                agent_id, action_index, entry, digest))
    risk = ActionValidator.risk_of(command)
    requires_approval = (command in DESTRUCTIVE_COMMANDS
                         or command in INVESTIGATIVE_COMMANDS)
    provenance = _provenance(agent_id, action_index, entry, digest)

    if kind == "none":
        # Sleep/Monitor: machine-local, no physical target.
        return RealActionPlan(
            operation=operation, params={}, risk=risk,
            requires_approval=requires_approval,
            policy_provenance=provenance)
    if kind == "host":
        return _translate_host(entry, operation, risk, requires_approval,
                               provenance, policy_map, report)
    if kind == "zone":
        return _translate_zone(entry, operation, risk, requires_approval,
                               provenance, report)
    return RealActionPlan(
        blocked=True, block_reason=f"unknown target kind {kind!r}",
        policy_provenance=provenance)


def _provenance(agent_id, action_index, entry, digest):
    return {"agent_id": int(agent_id), "action_index": int(action_index),
            "command": entry.get("command", ""),
            "kind": entry.get("kind", "none"),
            "cc4_target": entry.get("target", ""),
            "policy_label": entry.get("label", ""),
            "mapping_digest": digest}


def _translate_host(entry, operation, risk, requires_approval,
                    provenance, policy_map, report):
    cc4 = entry.get("target", "")
    asset_id = report.asset_of_slot(cc4) if report else None
    if not asset_id:
        return RealActionPlan(
            operation=operation, risk=risk,
            requires_approval=requires_approval, blocked=True,
            block_reason=(f"phantom target: policy slot {cc4!r} has no "
                          f"physical asset bound; refusing to act on "
                          f"placeholder infrastructure"),
            policy_provenance=provenance)
    try:
        info = policy_map.real_asset(
            f"blue_agent_{int(provenance['agent_id'])}", cc4)
    except Exception:
        info = {"ip": "", "hostname": ""}
    ips = tuple(ip for ip in (info.get("ip", ""),) if ip)
    return RealActionPlan(
        asset_id=asset_id,
        hostname=info.get("hostname", "") or asset_id,
        ips=ips, operation=operation,
        params={"cc4_slot": cc4}, risk=risk,
        requires_approval=requires_approval,
        policy_provenance=provenance)


def _translate_zone(entry, operation, risk, requires_approval,
                    provenance, report):
    # Zone entries carry structured zone fields; fall back to parsing
    # nothing -- unknown shape is a refusal, not a guess.
    zones = [entry.get("zone") or entry.get("target") or ""]
    zones = [z for z in zones if z]
    affected = []
    for zone in zones:
        try:
            affected.extend(report.bound_assets_in_policy_zone(zone))
        except Exception:
            continue
    affected = sorted(set(affected))
    if not affected:
        return RealActionPlan(
            operation=operation, risk=risk,
            requires_approval=requires_approval, blocked=True,
            block_reason=(f"no physical footprint: policy zone(s) "
                          f"{zones} contain no bound real assets"),
            params={"policy_zones": zones},
            policy_provenance=provenance)
    return RealActionPlan(
        operation=operation,
        params={"policy_zones": zones, "affected_assets": affected},
        risk=risk, requires_approval=requires_approval,
        policy_provenance=provenance)


def is_phantom_asset(asset):
    """True for unpopulated placeholder identities (never real)."""
    hostname = str((asset or {}).get("hostname", ""))
    ip = str((asset or {}).get("ip", ""))
    return (hostname.startswith(UNPOPULATED_HOSTNAME_PREFIX)
            or ip.startswith(UNPOPULATED_IP_PREFIX))


class PhantomGuardBackend(EnforcementBackend):
    """EnforcementBackend refusing phantom host targets (defence in depth).

    Subclasses the frozen ``EnforcementBackend`` so it passes through
    the UNCHANGED ``load_backend``/``DeploymentPipeline`` path
    (``isinstance``-checked). Every method delegates to the wrapped
    backend; the four host-mutating operations first refuse
    unpopulated placeholder identities (TEST-NET-3 or the
    ``unpopulated-policy-slot-`` hostname) with ``BackendError``.
    Translation (``translate_decision``) should already have blocked
    them -- this is the second net before vendor APIs.
    """

    def __init__(self, inner):
        super().__init__()
        if not isinstance(inner, EnforcementBackend):
            raise BackendError(
                f"phantom guard needs an EnforcementBackend, got "
                f"{type(inner).__name__}")
        self._inner = inner
        self.name = f"phantom-guarded({getattr(inner, 'name', '?')})"
        # Share the audit trail: attempts on either object are visible
        # in both places, matching caller-owned-backend semantics.
        self.audit = getattr(inner, "audit", self.audit)

    @staticmethod
    def _guard(asset):
        if is_phantom_asset(asset if isinstance(asset, dict) else {}):
            raise BackendError(
                f"refusing phantom target {asset!r}: policy slot has "
                f"no physical asset (unpopulated padding)")

    # -- delegated interface (guards on host-mutating ops only) --
    def noop(self, timeout_s, idempotency_key):
        return self._inner.noop(timeout_s, idempotency_key)

    def collect_status(self, asset, timeout_s, idempotency_key):
        return self._inner.collect_status(asset, timeout_s,
                                          idempotency_key)

    def collect_forensics(self, asset, timeout_s, idempotency_key):
        self._guard(asset)
        return self._inner.collect_forensics(asset, timeout_s,
                                             idempotency_key)

    def deploy_honeypot(self, asset, timeout_s, idempotency_key):
        self._guard(asset)
        return self._inner.deploy_honeypot(asset, timeout_s,
                                           idempotency_key)

    def terminate_suspicious(self, asset, timeout_s, idempotency_key):
        self._guard(asset)
        return self._inner.terminate_suspicious(asset, timeout_s,
                                                idempotency_key)

    def reimage_host(self, asset, timeout_s, idempotency_key):
        self._guard(asset)
        return self._inner.reimage_host(asset, timeout_s,
                                        idempotency_key)

    def set_zone_block(self, from_zone, to_zone, timeout_s,
                       idempotency_key):
        return self._inner.set_zone_block(from_zone, to_zone, timeout_s,
                                          idempotency_key)

    def clear_zone_block(self, from_zone, to_zone, timeout_s,
                         idempotency_key):
        return self._inner.clear_zone_block(from_zone, to_zone,
                                            timeout_s, idempotency_key)

    def rollback(self, audit_record):
        return self._inner.rollback(audit_record)


__all__ = [
    "RealActionPlan", "translate_decision", "is_phantom_asset",
    "PhantomGuardBackend",
]
