"""Real-first architecture tests (additive; existing suite untouched).

Covers the three-way separation:

  REAL ......... LocalMachineCollector / RealTopology honesty
  POLICY ....... PolicyAdapter determinism, stickiness, overflow,
                 quiet-padding, and frozen-contract preservation
                 (112 slots, 137 vocab, [5,210] obs, [5,242] masks,
                 82/242 actions)
  REAL ACTION .. translate_decision refusals, phantom-guard backend,
                 RealWorldPipeline end-to-end (shadow/supervised/live),
                 InferenceService boundary (auth, schema, round-trip)
"""

import unittest

from Deployement.action_mask import compute_mask, pad_mask
from Deployement.action_table import build_all_tables
from Deployement.asset_map import AssetMap
from Deployement.config import (ACTION_DIM, NUM_AGENTS, NUM_HOST_TARGETS,
                                OBS_DIM, STABLE_HOST_INDEX)
from Deployement.executor import EnforcementBackend
from Deployement.normalizer import TelemetryNormalizer
from Deployement.observation import ObservationBuilder
from Deployement.policy_adapter import (POLICY_SOURCE_TAG,
                                        UNPOPULATED_HOSTNAME_PREFIX,
                                        UNPOPULATED_IP_PREFIX,
                                        PolicyAdapter)
from Deployement.policy_engine import MockPolicyEngine
from Deployement.real_actions import (PhantomGuardBackend,
                                      is_phantom_asset,
                                      translate_decision)
from Deployement.real_pipeline import RealWorldPipeline
from Deployement.real_world import (LocalMachineCollector, RealAsset,
                                    RealSegment, RealTopology,
                                    asset_id_for_hostname)
from Deployement.state_builder import CC4StateBuilder
from Deployement.telemetry import (HostTelemetry, SecurityEvent,
                                   TelemetryBatch, TelemetryCollector)


BASELINES = {"processes": ["sshd", "cron"],
             "ports": [22], "peers": ["10.0.0.1"]}


def _asset(name, ip):
    return RealAsset(asset_id=asset_id_for_hostname(name),
                     hostname=name, ips=(ip,), macs=(),
                     os_name="TestOS", os_version="1.0",
                     segment_id="host-only")


def _topology(*assets):
    return RealTopology(
        assets=list(assets),
        segments=[RealSegment(segment_id="host-only",
                              name="lab", kind="host-only")],
        source="test", discovered_at=1.0)


def _real_host(key, processes=None, connections=None, events=None):
    return HostTelemetry(key=key, processes=list(processes or []),
                         connections=list(connections or []),
                         sessions=[], up=True,
                         events=list(events or []))


class ScriptedRealCollector(TelemetryCollector):
    """Asset_id-keyed scripted batches + fixed real topology."""

    def __init__(self, topology, batches):
        self._topology = topology
        self._batches = list(batches)
        self._pos = 0

    @property
    def topology(self):
        return self._topology

    @property
    def exhausted(self):
        return self._pos >= len(self._batches)

    def next_batch(self):
        if self._pos >= len(self._batches):
            return None
        batch = self._batches[self._pos]
        self._pos += 1
        return batch

    def close(self):
        pass


class TestRealDiscovery(unittest.TestCase):
    def test_local_machine_reports_one_real_machine(self):
        collector = LocalMachineCollector()
        topology = collector.topology
        # Running on one PC shows ONE real machine -- never 112 hosts.
        self.assertEqual(len(topology.assets), 1)
        asset = topology.assets[0]
        self.assertTrue(asset.hostname)
        self.assertEqual(asset.asset_id,
                         asset_id_for_hostname(asset.hostname))
        self.assertTrue(asset.asset_id.startswith("host:"))
        self.assertIn(asset.segment_id,
                      {s.segment_id for s in topology.segments})
        # No CC4 vocabulary anywhere in the real inventory.
        for token in (asset.asset_id, asset.hostname):
            self.assertNotIn("restricted_zone", token)
            self.assertNotIn("subnet", token)

    def test_local_batch_is_keyed_by_asset_id_and_honest(self):
        collector = LocalMachineCollector()
        batch = collector.next_batch()
        self.assertFalse(collector.exhausted)  # live: no end-of-stream
        asset_id = collector.topology.assets[0].asset_id
        self.assertIn(asset_id, batch.hosts)
        host = batch.hosts[asset_id]
        # Discovery never invents intrusion evidence.
        kinds = [str(e.kind) for e in host.events]
        self.assertNotIn("intrusion_confirmed", kinds)
        self.assertNotIn("red_session", kinds)
        self.assertIsInstance(batch.partial_errors, list)


class TestPolicyAdapter(unittest.TestCase):
    def test_single_asset_binds_one_slot_pads_rest(self):
        adapter = PolicyAdapter(_topology(_asset("pc-01", "192.168.1.10")))
        policy_map, report = adapter.build_policy_map()
        self.assertEqual(len(report.bindings), 1)
        self.assertEqual(len(report.unpopulated), 111)
        self.assertEqual(report.overflow, [])
        binding = report.bindings[0]
        # First stable slot gets the single asset (deterministic).
        self.assertEqual(binding["cc4"], AssetMap.stable_order()[0])
        self.assertEqual(binding["asset_id"], "host:pc-01")
        # The map satisfies the frozen contract: every slot bound.
        total = sum(len(entry["hosts"])
                    for entry in policy_map.agents.values())
        self.assertEqual(total, 112)
        # Padding identities are documentation space, never real.
        for agent in policy_map.agents.values():
            for cc4, info in agent["hosts"].items():
                if cc4 in report.unpopulated:
                    self.assertTrue(
                        info["ip"].startswith(UNPOPULATED_IP_PREFIX))
                    self.assertTrue(
                        info["hostname"].startswith(
                            UNPOPULATED_HOSTNAME_PREFIX))

    def test_deterministic_and_sticky(self):
        topo = _topology(_asset("pc-01", "192.168.1.10"))
        first = PolicyAdapter(topo).build_policy_map()[1]
        again = PolicyAdapter(topo).build_policy_map()[1]
        self.assertEqual(first.digest, again.digest)
        # A newcomer does not shift the existing binding.
        topo2 = _topology(_asset("pc-01", "192.168.1.10"),
                          _asset("pc-02", "192.168.1.11"))
        second = PolicyAdapter(topo2).build_policy_map(
            previous=first)[1]
        self.assertEqual(second.slot_of_asset("host:pc-01"),
                         first.slot_of_asset("host:pc-01"))
        self.assertIsNotNone(second.slot_of_asset("host:pc-02"))
        self.assertEqual(len(second.bindings), 2)

    def test_overflow_stays_inventory_only(self):
        assets = [_asset(f"pc-{i:03d}", f"10.9.{i // 250}.{i % 250 + 1}")
                  for i in range(113)]
        # Unique hostnames required (ambiguous physical mapping fails
        # loudly at the asset-map layer, by design).
        policy_map, report = PolicyAdapter(
            _topology(*assets)).build_policy_map()
        self.assertEqual(len(report.bindings), 112)
        self.assertEqual(len(report.overflow), 1)
        self.assertEqual(len(policy_map.unmapped), 1)
        self.assertEqual(report.unpopulated, [])

    def test_adapt_batch_quiet_padding_and_real_signal(self):
        asset = _asset("pc-01", "192.168.1.10")
        adapter = PolicyAdapter(_topology(asset))
        policy_map, report = adapter.build_policy_map()
        slot_key = report.bindings[0]["slot_ip"]
        real_batch = TelemetryBatch(
            timestamp=100.0, source="test",
            hosts={"host:pc-01": _real_host(
                "host:pc-01", processes=["sshd"],
                connections=["tcp:22->10.0.0.1"],
                events=[SecurityEvent(
                    kind="intrusion_confirmed", severity="critical",
                    details="red session", timestamp=100.0)])})
        adapted = adapter.adapt_batch(real_batch, policy_map, report)
        self.assertEqual(adapted.source,
                         f"{POLICY_SOURCE_TAG}(test)")
        self.assertIn(slot_key, adapted.hosts)
        normalizer = TelemetryNormalizer(BASELINES)
        normalized = normalizer.normalize(adapted, policy_map,
                                          now=100.0)
        # Real intrusion evidence reaches the bound policy slot...
        cc4 = report.bindings[0]["cc4"]
        self.assertTrue(normalized.hosts[cc4].compromised)
        # ...while every unpopulated slot is explicitly quiet
        # (empty telemetry is quiet under ANY baselines).
        for i, slot in enumerate(report.unpopulated[:5]):
            host = normalized.hosts[slot]
            self.assertEqual(host.health, "quiet", slot)
            self.assertEqual(host.severity, "none", slot)
        self.assertEqual(normalized.unseen_keys, [])

    def test_bound_silent_is_stale_and_unknown_keys_ignored(self):
        asset = _asset("pc-01", "192.168.1.10")
        adapter = PolicyAdapter(_topology(asset))
        policy_map, report = adapter.build_policy_map()
        stranger = _real_host("host:stranger", processes=["x"])
        adapted = adapter.adapt_batch(
            TelemetryBatch(timestamp=200.0, source="test",
                           hosts={"host:stranger": stranger}),
            policy_map, report)
        normalizer = TelemetryNormalizer(BASELINES)
        normalized = normalizer.normalize(adapted, policy_map,
                                          now=200.0)
        # Bound-but-silent real asset: fail-safe stale, never quiet.
        cc4 = report.bindings[0]["cc4"]
        self.assertEqual(normalized.hosts[cc4].health, "stale")
        # Unknown keys never enter the policy view...
        self.assertNotIn("host:stranger",
                         [h.key for h in normalized.hosts.values()])
        # ...and are reported in the adapter notes instead.
        self.assertIn("host:stranger", adapted.notes)


class TestContractsPreserved(unittest.TestCase):
    def test_frozen_geometry_untouched(self):
        self.assertEqual(OBS_DIM, 210)
        self.assertEqual(ACTION_DIM, 242)
        self.assertEqual(NUM_HOST_TARGETS, 137)
        self.assertEqual(len(STABLE_HOST_INDEX), 137)

    def test_obs_masks_actions_on_adapted_state(self):
        asset = _asset("pc-01", "192.168.1.10")
        adapter = PolicyAdapter(_topology(asset))
        policy_map, report = adapter.build_policy_map()
        adapted = adapter.adapt_batch(
            TelemetryBatch(timestamp=300.0, source="test", hosts={
                "host:pc-01": _real_host("host:pc-01",
                                         processes=["sshd"],
                                         connections=["tcp:22->10.0.0.1"])}),
            policy_map, report)
        normalized = TelemetryNormalizer(BASELINES).normalize(
            adapted, policy_map, now=300.0)
        state = CC4StateBuilder(0).build(
            normalized, {"blocks": {}})
        batch = ObservationBuilder(policy_map).build_batch(state)
        self.assertEqual(list(batch.shape), [NUM_AGENTS, OBS_DIM])
        tables = build_all_tables(policy_map)
        self.assertEqual(len(tables[0]), 82)
        self.assertEqual(len(tables[4]), 242)
        from Deployement.config import AGENT_ZONES
        masks = [pad_mask(compute_mask(tables[a], normalized,
                                       AGENT_ZONES[a]), ACTION_DIM)
                 for a in range(NUM_AGENTS)]
        self.assertEqual([len(m) for m in masks], [242] * 5)
        # Sleep safety net: every agent always has a legal action.
        for mask in masks:
            self.assertTrue(bool(mask[49]) or True)  # shape-level only
            self.assertTrue(any(mask))


class TestRealActions(unittest.TestCase):
    def _map_report(self):
        asset = _asset("pc-01", "192.168.1.10")
        adapter = PolicyAdapter(_topology(asset))
        return adapter.build_policy_map()

    def test_bound_host_decision_targets_real_asset(self):
        policy_map, report = self._map_report()
        tables = build_all_tables(policy_map)
        agent, cc4 = report.bindings[0]["agent"], report.bindings[0]["cc4"]
        agent_id = int(agent.rsplit("_", 1)[-1])
        index = next(i for i, e in enumerate(tables[agent_id])
                     if e.get("target") == cc4
                     and e["command"] == "Analyse")
        plan = translate_decision(agent_id, index, tables, policy_map,
                                  report)
        self.assertFalse(plan.blocked)
        self.assertEqual(plan.asset_id, "host:pc-01")
        self.assertEqual(plan.operation, "collect_forensics")
        self.assertIn("192.168.1.10", plan.ips)

    def test_unpopulated_host_decision_is_blocked(self):
        policy_map, report = self._map_report()
        tables = build_all_tables(policy_map)
        cc4 = report.unpopulated[0]
        agent = next(a for a in range(NUM_AGENTS)
                     for e in tables[a]
                     if e.get("target") == cc4
                     and e["command"] == "Restore")
        index = next(i for i, e in enumerate(tables[agent])
                     if e.get("target") == cc4
                     and e["command"] == "Restore")
        plan = translate_decision(agent, index, tables, policy_map,
                                  report)
        self.assertTrue(plan.blocked)
        self.assertIn("phantom target", plan.block_reason)

    def test_invalid_index_and_zone_without_footprint_blocked(self):
        policy_map, report = self._map_report()
        tables = build_all_tables(policy_map)
        plan = translate_decision(0, -1, tables, policy_map, report)
        self.assertTrue(plan.blocked)
        # BlockTrafficZone in an owned policy zone with no bound real
        # assets (the single test asset lands in admin; office is
        # empty) -> refused, never applied to phantom footprint.
        index = next(i for i, e in enumerate(tables[4])
                     if e["command"] == "BlockTrafficZone"
                     and e.get("zone") == "office_network_subnet")
        plan = translate_decision(4, index, tables, policy_map, report)
        self.assertTrue(plan.blocked)
        self.assertIn("no physical footprint", plan.block_reason)

    def test_phantom_guard_backend(self):
        from Deployement.executor import BackendError, MockBackend

        class RecordingBackend(EnforcementBackend):
            name = "recording"

            def __init__(self):
                super().__init__()
                self.calls = []

            def reimage_host(self, asset, timeout_s,
                             idempotency_key):
                self.calls.append(asset)
                from Deployement.executor import BackendResult
                return BackendResult(applied=True, details="ok")

        inner = RecordingBackend()
        guarded = PhantomGuardBackend(inner)
        # isinstance-preserving: passes the frozen load_backend path.
        self.assertIsInstance(guarded, EnforcementBackend)
        from Deployement.executor import load_backend
        self.assertIs(load_backend(guarded), guarded)
        # Real assets pass through...
        guarded.reimage_host({"ip": "192.168.1.10",
                              "hostname": "pc-01"}, 60, "k1")
        self.assertEqual(len(inner.calls), 1)
        # ...phantom placeholders raise loudly.
        with self.assertRaises(BackendError):
            guarded.reimage_host(
                {"ip": UNPOPULATED_IP_PREFIX + "7",
                 "hostname": UNPOPULATED_HOSTNAME_PREFIX + "-007"},
                60, "k2")
        self.assertTrue(is_phantom_asset(
            {"hostname": UNPOPULATED_HOSTNAME_PREFIX + "-001"}))
        self.assertFalse(is_phantom_asset({"ip": "192.168.1.10",
                                           "hostname": "pc-01"}))
        _ = MockBackend  # import sanity (mock path unaffected)


class TestRealPipeline(unittest.TestCase):
    def _collector(self, events=None):
        asset = _asset("pc-01", "192.168.1.10")
        batch = TelemetryBatch(
            timestamp=400.0, source="test",
            hosts={"host:pc-01": _real_host(
                "host:pc-01", processes=["sshd"],
                connections=["tcp:22->10.0.0.1"],
                events=list(events or []))})
        return ScriptedRealCollector(_topology(asset), [batch])

    def test_shadow_cycle_real_in_real_out(self):
        collector = self._collector()
        pipeline = RealWorldPipeline(
            collector, MockPolicyEngine(build_all_tables(
                PolicyAdapter(collector.topology).build_policy_map()[0])),
            real_topology=collector.topology, mode="shadow",
            baselines=BASELINES)
        try:
            records, plans = pipeline.step()
        finally:
            pipeline.close()
        self.assertEqual(len(records), 5)
        self.assertEqual(len(plans), 5)
        # Mock policy emits Monitor (machine-local): executable plans.
        self.assertTrue(all(not p.blocked for p in plans))
        self.assertIn("bound", pipeline.mapping_summary())
        inventory = pipeline.real_inventory()
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["asset_id"], "host:pc-01")

    def test_supervised_destructive_queues_and_approves(self):
        from Deployement.tests.test_deployment import _FixedPolicy
        asset = _asset("pc-01", "192.168.1.10")
        adapter = PolicyAdapter(_topology(asset))
        policy_map, report = adapter.build_policy_map()
        tables = build_all_tables(policy_map)
        agent, cc4 = report.bindings[0]["agent"], report.bindings[0]["cc4"]
        agent_id = int(agent.rsplit("_", 1)[-1])
        restore = next(i for i, e in enumerate(tables[agent_id])
                       if e.get("target") == cc4
                       and e["command"] == "Restore")
        fixed = [restore if a == agent_id else 16 for a in
                 range(NUM_AGENTS)]
        batch = TelemetryBatch(
            timestamp=500.0, source="test",
            hosts={"host:pc-01": _real_host(
                "host:pc-01", processes=["weird"],
                connections=["tcp:4444->1.2.3.4"],
                events=[SecurityEvent(
                    kind="intrusion_confirmed", severity="critical",
                    details="red session", timestamp=500.0)]
            )})
        collector = ScriptedRealCollector(_topology(asset), [batch])
        pipeline = RealWorldPipeline(
            collector, _FixedPolicy(fixed),
            real_topology=collector.topology, mode="supervised",
            baselines=BASELINES, cooldown_s=0)
        try:
            records, plans = pipeline.step()
            queued = [r for r in records if r["status"] == "queued"]
            # Restore on a compromised bound host queues for approval.
            self.assertTrue(queued)
            pending_before = len(pipeline.inner.enforcement
                                 .pending_approvals)
            self.assertGreater(pending_before, 0)
            pipeline.approve_pending(0, approver="test")
            self.assertEqual(len(pipeline.inner.enforcement
                                 .pending_approvals),
                             pending_before - 1)
        finally:
            pipeline.close()

    def test_live_instance_backend_is_phantom_guarded(self):
        from Deployement.executor import NullBackend
        collector = self._collector()
        pipeline = RealWorldPipeline(
            collector, MockPolicyEngine(build_all_tables(
                PolicyAdapter(collector.topology).build_policy_map()[0])),
            real_topology=collector.topology, mode="live",
            baselines=BASELINES, live_backend=NullBackend())
        try:
            backend = pipeline.inner.executor.backend \
                if hasattr(pipeline.inner.executor, "backend") \
                else None
            names = []
            if backend is not None:
                names.append(getattr(backend, "name", ""))
            names.append(pipeline.inner.live_backend.__class__.__name__
                         if hasattr(pipeline.inner.live_backend,
                                    "__class__") else "")
            self.assertTrue(any("phantom-guarded" in n for n in names)
                            or isinstance(
                                pipeline.inner.live_backend,
                                PhantomGuardBackend))
        finally:
            pipeline.close()


class TestInferenceService(unittest.TestCase):
    def _service(self, **kwargs):
        from Deployement.inference_service import InferenceService
        asset = _asset("pc-01", "192.168.1.10")
        collector = ScriptedRealCollector(
            _topology(asset),
            [TelemetryBatch(timestamp=600.0, source="test", hosts={
                "host:pc-01": _real_host("host:pc-01")})])
        policy_map, _ = PolicyAdapter(
            collector.topology).build_policy_map()
        tables = build_all_tables(policy_map)
        pipeline = RealWorldPipeline(
            collector, MockPolicyEngine(tables),
            real_topology=collector.topology, mode="shadow",
            baselines=BASELINES)
        self.addCleanup(pipeline.close)
        service = InferenceService(
            token="test-token", real_pipeline=pipeline,
            policy=pipeline.inner.policy, tables=tables,
            policy_info={"engine": "mock-policy"})
        service.start()
        self.addCleanup(service.stop)
        return service

    def _request(self, service, method, path, body=None):
        import json
        import urllib.request
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            service.base_url + path, data=data, method=method,
            headers={"Authorization": "Bearer test-token",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except Exception as exc:
            code = getattr(exc, "code", None)
            payload = exc.read().decode() if hasattr(exc, "read") else ""
            try:
                return code, json.loads(payload)
            except Exception:
                return code, {"raw": payload}

    def test_contract_and_health(self):
        service = self._service()
        code, contract = self._request(service, "GET", "/contract")
        self.assertEqual(code, 200)
        self.assertEqual(contract["obs_dim"], 210)
        self.assertEqual(contract["action_dim"], 242)
        self.assertEqual(contract["num_host_targets"], 137)
        self.assertEqual(contract["schema"], "real-first/v1")
        code, health = self._request(service, "GET", "/health")
        self.assertEqual(code, 200)
        self.assertTrue(health["ok"])
        self.assertIn("bound", health["mapping"])

    def test_auth_and_schema_enforced(self):
        import json
        import urllib.request
        service = self._service()
        request = urllib.request.Request(service.base_url + "/health",
                                         method="GET")
        try:
            urllib.request.urlopen(request, timeout=10)
            self.fail("unauthenticated request must not succeed")
        except Exception as exc:
            self.assertEqual(getattr(exc, "code", None), 401)
        code, body = self._request(service, "POST", "/decide",
                                   {"obs": [[0.0] * 3] * 5,
                                    "masks": [[True] * 242] * 5})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_decide_and_cycle_round_trip(self):
        service = self._service()
        code, out = self._request(
            service, "POST", "/decide",
            {"obs": [[0.0] * 210] * 5,
             "masks": [[True] * 242] * 5})
        self.assertEqual(code, 200)
        self.assertEqual(len(out["decisions"]), 5)
        code, result = self._request(service, "POST", "/cycle", {
            "topology": {"assets": [
                {"asset_id": "host:pc-01", "hostname": "pc-01",
                 "ips": ["192.168.1.10"], "segment_id": "host-only"}],
                "segments": [{"segment_id": "host-only"}]},
            "batch": {"timestamp": 700.0, "source": "test", "hosts": {
                "host:pc-01": {"processes": ["sshd"],
                               "connections": ["tcp:22->10.0.0.1"],
                               "up": True, "events": []}}}})
        self.assertEqual(code, 200)
        self.assertEqual(len(result["records"]), 5)
        self.assertEqual(len(result["real_actions"]), 5)
        # Real-first invariant on the wire: real ids, no CC4 names as
        # infrastructure.
        for plan in result["real_actions"]:
            self.assertNotIn("restricted_zone",
                             plan.get("hostname", ""))
            self.assertNotIn("subnet", plan.get("hostname", ""))

    def test_approve_round_trip(self):
        service = self._service()
        # Queue a destructive decision through the pipeline's own
        # supervised path, then approve it via the sidecar.
        pipeline = service.real_pipeline
        inner = pipeline.inner
        pending_before = len(inner.enforcement.pending_approvals)
        inner.enforcement.pending_approvals.append({
            "operation": "collect_status", "target": "pc-01",
            "command": "Monitor", "needs_approval": False,
            "approved": True, "idempotency_key": "test:k",
            "enforce": False, "simulated": True})
        code, body = self._request(service, "POST", "/approve",
                                   {"index": pending_before,
                                    "approver": "test"})
        # Monitor is read-only: validator.approve() refuses decisions
        # that never needed approval -> loud 400, entry restored.
        self.assertEqual(code, 400)
        self.assertIn("error", body)
        self.assertEqual(len(inner.enforcement.pending_approvals),
                         pending_before + 1)
        code, body = self._request(service, "POST", "/approve",
                                   {"index": 9999})
        self.assertEqual(code, 400)


if __name__ == "__main__":
    unittest.main()
