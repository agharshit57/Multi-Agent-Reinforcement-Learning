"""Real-world deployment pipeline: REAL topology in, REAL actions out.

Composition (no logic duplicated, frozen layers untouched)::

    real collector (asset_id-keyed) ──> PolicyAdapter ──> slot batch
        ──> DeploymentPipeline (frozen: normalize/state/obs/masks/
            policy/validate/execute) ──> DecisionRecords
        ──> real_actions.translate_decision ──> RealActionPlans

The inner ``DeploymentPipeline`` operates on the adapter-built policy
map, whose bound slots carry REAL ``ip``/``hostname`` values -- so the
unchanged validator/executor path already resolves real targets, and
the unchanged GUI keeps working as a fallback/reference console.

Safety preserved 1:1: modes (shadow/mock/supervised/live + live
opt-in is the caller's, via mode="live"), action masking, validation,
cooldowns, approvals (``approve_pending`` delegates), idempotency,
audit trail, stale-fail-safe telemetry. Live backends passed as
instances are wrapped in ``PhantomGuardBackend`` (second net against
phantom targets); string specs ("null"/"mock"/"module:Class") pass
through to the inner pipeline unchanged.
"""

from .action_table import build_all_tables
from .executor import EnforcementBackend
from .pipeline import DeploymentPipeline
from .policy_adapter import PolicyAdapter
from .real_actions import PhantomGuardBackend, translate_decision
from .telemetry import TelemetryCollector


class AdapterCollector(TelemetryCollector):
    """Inner-pipeline collector: real batches adapted to slot batches.

    Rebinds STICKILY when the real asset SET changes (assets keyed by
    ``asset_id``; arrival/departure only reallocates the delta).
    Rebind events are recorded in ``rebind_log`` for audit.
    """

    def __init__(self, real_collector, adapter, policy_map, report):
        self._real = real_collector
        self._adapter = adapter
        self.policy_map = policy_map
        self.report = report
        self.rebind_log = []

    @property
    def exhausted(self):
        return bool(getattr(self._real, "exhausted", False))

    def _maybe_rebind(self):
        topology = getattr(self._real, "topology", None)
        assets = getattr(topology, "assets", None) if topology else None
        if assets is None:
            return
        current = sorted(a.asset_id for a in assets)
        bound = sorted(b["asset_id"] for b in self.report.bindings)
        overflow = sorted(self.report.overflow)
        if current == sorted(bound + overflow):
            return
        from .real_world import RealTopology
        fresh_topology = RealTopology(
            assets=list(assets),
            segments=list(getattr(topology, "segments", [])),
            source=getattr(topology, "source", ""),
            discovered_at=getattr(topology, "discovered_at", 0.0))
        self._adapter.topology = fresh_topology
        new_map, new_report = self._adapter.build_policy_map(
            previous=self.report)
        self.policy_map = new_map
        self.report = new_report
        self.rebind_log.append(
            {"event": "rebind", "digest": new_report.digest,
             "bound": len(new_report.bindings),
             "unpopulated": len(new_report.unpopulated),
             "overflow": list(new_report.overflow)})

    def next_batch(self):
        batch = self._real.next_batch()
        if batch is None:
            return None
        self._maybe_rebind()
        return self._adapter.adapt_batch(batch, self.policy_map,
                                         self.report)

    def close(self):
        try:
            self._real.close()
        except Exception:
            pass


class RealWorldPipeline:
    """Deployment pipeline grounded in a real topology.

    ``real_collector`` yields asset_id-keyed ``TelemetryBatch`` objects
    (e.g. ``LocalMachineCollector``). ``real_topology`` is the matching
    ``RealTopology`` (``collector.topology`` when omitted and
    available). ``policy`` is any frozen-contract engine
    (``MockPolicyEngine`` / ``TrainedPolicyEngine``).
    """

    def __init__(self, real_collector, policy, real_topology=None,
                 mode="shadow", baselines=None, mission_phase=0,
                 cooldown_s=300, stale_after_s=300, session_dir=None,
                 live_backend="null", max_log_bytes=None,
                 history_max=None, policy_info=None,
                 guard_phantoms=True):
        from .pipeline import (DEFAULT_MAX_LOG_BYTES,
                               HISTORY_MAX_RECORDS)
        if real_topology is None:
            real_topology = getattr(real_collector, "topology", None)
        if real_topology is None:
            raise ValueError(
                "RealWorldPipeline needs a real topology: pass "
                "real_topology or a collector exposing .topology")
        self.real_collector = real_collector
        self.real_topology = real_topology
        self.adapter = PolicyAdapter(real_topology)
        self.policy_map, self.report = (
            self.adapter.build_policy_map())
        self.tables = build_all_tables(self.policy_map)
        if guard_phantoms and isinstance(live_backend,
                                         EnforcementBackend):
            live_backend = PhantomGuardBackend(live_backend)
        self._adapter_collector = AdapterCollector(
            real_collector, self.adapter, self.policy_map, self.report)
        kwargs = dict(mode=mode, baselines=baselines,
                      mission_phase=mission_phase, cooldown_s=cooldown_s,
                      session_dir=session_dir,
                      stale_after_s=stale_after_s,
                      live_backend=live_backend,
                      policy_info=dict(policy_info or {}))
        if max_log_bytes is not None:
            kwargs["max_log_bytes"] = max_log_bytes
        else:
            kwargs["max_log_bytes"] = DEFAULT_MAX_LOG_BYTES
        if history_max is not None:
            kwargs["history_max"] = history_max
        else:
            kwargs["history_max"] = HISTORY_MAX_RECORDS
        self.inner = DeploymentPipeline(
            self.policy_map, self._adapter_collector, policy, **kwargs)
        # Inner pipeline built its own tables from the same map; keep
        # one reference so translation and execution cannot disagree.
        self.tables = self.inner.tables

    # ------------------------------------------------------- stepping --
    def step(self):
        """One cycle -> (records, real-action plans)."""
        records = self.inner.step()
        # AdapterCollector may have rebound (topology change): adopt the
        # fresh map/report so translation matches execution.
        self.policy_map = self._adapter_collector.policy_map
        self.report = self._adapter_collector.report
        plans = [translate_decision(r["agent_id"], r["action_index"],
                                    self.tables, self.policy_map,
                                    self.report)
                 for r in records]
        return records, plans

    def run(self, max_cycles=None):
        out_records, out_plans = [], []
        while True:
            records, plans = self.step()
            if not records and self._adapter_collector.exhausted:
                break
            out_records.extend(records)
            out_plans.extend(plans)
            if (max_cycles is not None
                    and self.inner.cycle >= max_cycles):
                break
        return out_records, out_plans

    # ------------------------------------------------------ delegation --
    def approve_pending(self, approval_id, approver="human"):
        return self.inner.approve_pending(approval_id, approver=approver)

    def set_mode(self, mode):
        self.inner.set_mode(mode)

    def reset(self):
        self.inner.reset()

    def close(self):
        self.inner.close()

    @property
    def cycle(self):
        return self.inner.cycle

    @property
    def session_id(self):
        return self.inner.session_id

    def mapping_summary(self):
        return self.report.summary()

    def real_inventory(self):
        """Operator-facing real assets (never CC4 names)."""
        return [{"asset_id": a.asset_id, "hostname": a.hostname,
                 "ips": list(a.ips or ()), "os": a.os_name,
                 "segment": a.segment_id}
                for a in self.real_topology.assets]


__all__ = ["RealWorldPipeline", "AdapterCollector"]
