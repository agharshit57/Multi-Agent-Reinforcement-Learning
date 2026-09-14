"""Action validation/translation: policy index -> safe real operation.

Nothing the policy emits reaches the network without passing this
layer. Checks, in order:
  1. index in range + entry structurally valid (mask was applied)
  2. target asset actually bound in the asset map
  3. mode policy:
       shadow     -> every decision is log-only, no state change
       mock       -> simulated effect on internal enforcement state
       supervised -> SAFE/INVESTIGATIVE auto-apply (mock-apply, still
                     no real enforcement); DESTRUCTIVE queue for human
                     approval, never auto-applied
       live       -> like supervised, but approved decisions carry
                     enforce=True for a real EnforcementBackend;
                     explicitly opt-in only (--enable-live)
  4. cooldowns: same (command, target) refused within ``cooldown_s``,
     stamped at approval/execution time (see approve())

Translation table (CC4 command -> real defensive operation):
  Sleep               -> noop
  Monitor             -> collect_status        (read-only)
  Analyse <host>      -> collect_forensics     (read-only)
  DeployDecoy <host>  -> deploy_honeypot       (needs approval*)
  Remove <host>       -> terminate_suspicious  (needs approval*)
  Restore <host>      -> reimage_host          (needs approval*)w
  AllowTrafficZone    -> allow_zone_traffic    (restorative, safe)
  BlockTrafficZone    -> isolate_zone_traffic  (needs approval*)

(*) approval only in supervised mode; in mock mode the effect is
simulated on internal state; in shadow mode everything is log-only.
"""

import time

from .config import (DESTRUCTIVE_COMMANDS, INVESTIGATIVE_COMMANDS,
                     MODES, SAFE_COMMANDS)

TRANSLATION = {
    "Sleep": "noop",
    "Monitor": "collect_status",
    "Analyse": "collect_forensics",
    "DeployDecoy": "deploy_honeypot",
    "Remove": "terminate_suspicious",
    "Restore": "reimage_host",
    "AllowTrafficZone": "allow_zone_traffic",
    "BlockTrafficZone": "isolate_zone_traffic",
}

READ_ONLY_OPS = frozenset({"noop", "collect_status", "collect_forensics"})


class ValidationError(ValueError):
    pass


class ActionValidator:
    def __init__(self, mode="shadow", cooldown_s=300):
        if mode not in MODES:
            raise ValidationError(f"unknown mode {mode!r}, want one of {MODES}")
        self.mode = mode
        self.cooldown_s = float(cooldown_s)
        self._last_fired = {}

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValidationError(f"unknown mode {mode!r}")
        self.mode = mode

    def reset(self):
        """Clear cooldown memory (new deployment session)."""
        self._last_fired = {}

    @property
    def live_enforcement(self):
        """True only in live mode: approved ops really execute."""
        return self.mode == "live"

    # ------------------------------------------------------------- API --
    def validate(self, agent_id, action_index, table, mask, now=None,
                 key_context=""):
        """Validate + translate. Returns an ApprovedAction dict.

        Raises ValidationError on any failure. Never raises for mode
        handling: shadow/mock decisions are returned with
        ``enforce=False`` / simulated effect markers instead.

        ``key_context`` (e.g. ``f"{session_id}:{cycle}"``) scopes the
        idempotency key to one decision cycle. Without it, two distinct
        intents validated within the same wall-clock second would share
        a key and the backend would wrongly suppress the second as a
        duplicate.
        """
        now = time.time() if now is None else float(now)
        if not (0 <= action_index < len(table)):
            raise ValidationError(f"action {action_index} out of range "
                                  f"[0, {len(table)})")
        if not bool(mask[action_index]):
            raise ValidationError(
                f"action {action_index} is masked out this cycle")
        entry = table[action_index]
        command = entry["command"]
        if command not in TRANSLATION:
            raise ValidationError(f"unknown command {command!r}")
        if entry["kind"] == "host" and not entry.get("asset"):
            raise ValidationError(
                f"host target {entry['target']} has no bound real asset")
        op = TRANSLATION[command]
        needs_approval = (command in DESTRUCTIVE_COMMANDS
                          or command in INVESTIGATIVE_COMMANDS)
        key = (command, entry.get("zone"), entry.get("target"))
        last = self._last_fired.get(key, 0.0)
        if now - last < self.cooldown_s and op not in READ_ONLY_OPS:
            raise ValidationError(
                f"cooldown: {command} on {entry.get('target')} "
                f"fired {now - last:.0f}s ago")
        key_scope = f"{key_context}:" if key_context else ""
        decision = {"agent_id": agent_id, "action_index": action_index,
                    "command": command, "operation": op,
                    "zone": entry.get("zone"), "target": entry.get("target"),
                    "asset": entry.get("asset"), "label": entry.get("label"),
                    "needs_approval": needs_approval, "enforce": False,
                    "approved": False, "simulated": False,
                    "timestamp": now,
                    "idempotency_key": (
                        f"{key_scope}{agent_id}:{command}:"
                        f"{entry.get('zone')}:"
                        f"{entry.get('target')}:{action_index}:{now:.0f}")}
        if self.mode == "shadow":
            return decision  # log-only by construction
        if self.mode == "mock":
            decision["simulated"] = True
            self._last_fired[key] = now
            return decision
        # supervised / live: destructive waits for a human; safe ops
        # apply now. Only live marks enforce=True (real backend);
        # supervised stays simulated (no live connectors wired).
        if needs_approval:
            return decision  # caller queues for human approval
        decision["approved"] = True
        if self.live_enforcement:
            decision["enforce"] = True
        else:
            decision["simulated"] = True
        self._last_fired[key] = now
        return decision

    def approve(self, decision, approver="human", now=None):
        """Human approval for a queued destructive decision.

        Cooldowns stamp the APPROVAL/execution time (``now``, default
        wall-clock), NOT the original policy-decision timestamp: a
        decision made at 10:00, approved at 10:10 and executed at 10:11
        cools down from ~10:10, so an identical re-decision at 10:11 is
        still gated while a genuinely later one is not.
        """
        if self.mode not in ("supervised", "live"):
            raise ValidationError(
                "approvals only exist in supervised/live mode")
        if not decision.get("needs_approval"):
            raise ValidationError("decision does not need approval")
        now = time.time() if now is None else float(now)
        decision = dict(decision)
        decision["approved"] = True
        if self.live_enforcement:
            decision["enforce"] = True
        else:
            decision["simulated"] = True
        decision["approver"] = approver
        decision["approved_at"] = now
        key = (decision["command"], decision.get("zone"),
               decision.get("target"))
        self._last_fired[key] = now
        return decision

    @staticmethod
    def risk_of(command):
        if command in SAFE_COMMANDS:
            return "safe"
        if command in INVESTIGATIVE_COMMANDS:
            return "elevated"
        return "destructive"
