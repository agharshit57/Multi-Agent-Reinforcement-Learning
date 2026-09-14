"""Live-path safety regressions (reported failure modes, pinned).

1. PhantomGuardBackend MUST fire on the real LiveExecutor path, where
   the asset arrives as a collapsed STRING (not the dict the old unit
   test used -- that test passed while the guard was dead in prod).
2. Approvals MUST be addressable by stable approval_id: the server
   queue persists across cycles while UI positions shift, so approving
   by position can pop the wrong decision.
3. run_cmd MUST treat ANY non-zero exit as unavailable, even when the
   tool still printed something (the documented fail-safe contract).
"""

import sys
import unittest

from Deployement.action_table import build_all_tables
from Deployement.endpoint_sensors import run_cmd
from Deployement.executor import (BackendError, BackendResult,
                                  EnforcementBackend, EnforcementState,
                                  LiveExecutor)
from Deployement.policy_adapter import PolicyAdapter
from Deployement.real_actions import PhantomGuardBackend, is_phantom_asset
from Deployement.real_pipeline import RealWorldPipeline
from Deployement.real_world import RealTopology, asset_id_for_hostname
from Deployement.real_world import RealAsset
from Deployement.telemetry import (HostTelemetry, SecurityEvent,
                                   TelemetryBatch, TelemetryCollector)


BASELINES = {"processes": ["sshd"], "ports": [22],
             "peers": ["10.0.0.1"]}


def _asset(name, ip):
    return RealAsset(asset_id=asset_id_for_hostname(name),
                     hostname=name, ips=(ip,),
                     coverage="covered-local")


class _RecordingBackend(EnforcementBackend):
    """Backend that would really act: counts host-mutating calls."""

    name = "recording"

    def __init__(self):
        super().__init__()
        self.calls = []

    def reimage_host(self, asset, timeout_s, idempotency_key):
        self.calls.append(("reimage_host", asset, idempotency_key))
        return BackendResult(applied=True, details="reimaged")


class _ScriptedRealCollector(TelemetryCollector):
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


def _host(key, events=()):
    return HostTelemetry(key=key, processes=["sshd"],
                         connections=["tcp:22->10.0.0.1"], sessions=[],
                         up=True, events=list(events))


class TestPhantomGuardLivePath(unittest.TestCase):
    """The guard must fire on STRING identities (LiveExecutor path)."""

    def _executor(self):
        inner = _RecordingBackend()
        guarded = PhantomGuardBackend(inner)
        state = EnforcementState()
        return LiveExecutor(state, backend=guarded), inner

    def _decision(self, asset):
        return {"operation": "reimage_host", "command": "Restore",
                "target": "some_slot", "zone": "some_zone",
                "agent_id": 0, "timestamp": 1.0,
                "asset": asset, "approved": True,
                "needs_approval": True,
                "idempotency_key": "test:k1"}

    def test_string_phantom_refused_before_backend(self):
        executor, inner = self._executor()
        result = executor.execute(self._decision(
            {"ip": "203.0.113.7",
             "hostname": "unpopulated-policy-slot-007"}))
        # LiveExecutor collapses the asset to the "203.0.113.7" string
        # before the backend sees it: the guard must catch THAT shape.
        self.assertFalse(result.applied)
        self.assertIn("phantom", result.error)
        self.assertEqual(inner.calls, [])  # backend never touched

    def test_string_real_passes_through(self):
        executor, inner = self._executor()
        result = executor.execute(self._decision(
            {"ip": "192.168.29.10", "hostname": "pc-01"}))
        self.assertTrue(result.applied)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(inner.calls[0][1], "192.168.29.10")

    def test_dict_shape_still_guarded(self):
        guarded = PhantomGuardBackend(_RecordingBackend())
        with self.assertRaises(BackendError):
            guarded.reimage_host(
                {"ip": "203.0.113.7",
                 "hostname": "unpopulated-policy-slot-007"},
                60, "k")
        self.assertTrue(is_phantom_asset("203.0.113.7"))
        self.assertTrue(
            is_phantom_asset("unpopulated-policy-slot-007"))
        self.assertFalse(is_phantom_asset("192.168.29.10"))
        self.assertFalse(is_phantom_asset("pc-01"))
        self.assertFalse(is_phantom_asset(None))
        self.assertFalse(is_phantom_asset({}))


class TestStableApprovalIds(unittest.TestCase):
    """Queue positions shift; approval_ids must not."""

    def _pipeline(self):
        from Deployement.policy_engine import MockPolicyEngine
        from Deployement.tests.test_deployment import _FixedPolicy
        asset = _asset("pc-01", "192.168.29.10")
        topology = RealTopology(
            assets=[asset], segments=[], source="test",
            discovered_at=1.0)
        policy_map, report = PolicyAdapter(topology).build_policy_map()
        tables = build_all_tables(policy_map)
        agent = int(report.bindings[0]["agent"].rsplit("_", 1)[-1])
        cc4 = report.bindings[0]["cc4"]
        restore = next(i for i, e in enumerate(tables[agent])
                       if e.get("target") == cc4
                       and e["command"] == "Restore")
        fixed = [restore if a == agent else 16 for a in range(5)]
        batch = TelemetryBatch(
            timestamp=500.0, source="test", hosts={
                "host:pc-01": _host("host:pc-01", events=[
                    SecurityEvent(
                        kind="intrusion_confirmed",
                        severity="critical", details="red session",
                        timestamp=500.0)]),
            })
        collector = _ScriptedRealCollector(
            topology, [batch, TelemetryBatch(
                timestamp=501.0, source="test",
                hosts=dict(batch.hosts))])
        pipeline = RealWorldPipeline(
            collector, _FixedPolicy(fixed),
            real_topology=topology, mode="supervised",
            baselines=BASELINES, cooldown_s=0)
        self.addCleanup(pipeline.close)
        return pipeline

    def test_ids_stable_and_position_independent(self):
        pipeline = self._pipeline()
        pipeline.step()
        pipeline.step()
        pending = pipeline.inner.enforcement.pending_approvals
        self.assertEqual(len(pending), 2)
        first_id = pending[0]["approval_id"]
        second_id = pending[1]["approval_id"]
        self.assertTrue(first_id and second_id)
        self.assertNotEqual(first_id, second_id)
        # Approving the SECOND entry by id pops exactly that entry --
        # the old positional UI would have popped the first.
        pipeline.approve_pending(approval_id=second_id,
                                 approver="test")
        remaining = pipeline.inner.enforcement.pending_approvals
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["approval_id"], first_id)
        # Re-approving a handled id fails loudly (never pops position).
        with self.assertRaises(IndexError):
            pipeline.approve_pending(approval_id=second_id)
        self.assertEqual(
            len(pipeline.inner.enforcement.pending_approvals), 1)
        # Legacy positional path still works against a fresh snapshot.
        pipeline.approve_pending(0, approver="test")
        self.assertEqual(
            pipeline.inner.enforcement.pending_approvals, [])

    def test_legacy_index_without_id_means_current_position(self):
        pipeline = self._pipeline()
        pipeline.step()
        # Positional call without approval_id == current position 0.
        pipeline.approve_pending(0, approver="test")
        self.assertEqual(
            pipeline.inner.enforcement.pending_approvals, [])


class TestApprovalEndpointIds(unittest.TestCase):
    def _service(self):
        from Deployement.inference_service import InferenceService
        from Deployement.policy_engine import MockPolicyEngine
        asset = _asset("pc-01", "192.168.29.10")
        topology = RealTopology(
            assets=[asset], segments=[], source="test",
            discovered_at=1.0)
        batch = TelemetryBatch(
            timestamp=600.0, source="test",
            hosts={"host:pc-01": _host("host:pc-01")})
        collector = _ScriptedRealCollector(topology, [batch])
        tables = build_all_tables(
            PolicyAdapter(topology).build_policy_map()[0])
        pipeline = RealWorldPipeline(
            collector, MockPolicyEngine(tables),
            real_topology=topology, mode="supervised",
            baselines=BASELINES, cooldown_s=0)
        self.addCleanup(pipeline.close)
        # One queued entry with a stable id (as the executor stamps).
        pipeline.inner.enforcement.pending_approvals.append({
            "operation": "reimage_host", "command": "Restore",
            "target": "slot-0", "asset": {"ip": "192.168.29.10",
                                          "hostname": "pc-01"},
            "needs_approval": True, "approved": False,
            "approval_id": "abc123def456",
            "idempotency_key": "test:k2"})
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

    def test_approvals_snapshot_and_id_approve(self):
        service = self._service()
        code, snapshot = self._request(service, "GET", "/approvals")
        self.assertEqual(code, 200)
        self.assertEqual(len(snapshot["approvals"]), 1)
        entry = snapshot["approvals"][0]
        self.assertEqual(entry["approval_id"], "abc123def456")
        self.assertEqual(entry["operation"], "reimage_host")
        self.assertEqual(entry["asset"], "192.168.29.10")
        self.assertEqual(entry["risk"], "destructive")
        # Approve by stable id (mock backend applies simulated).
        code, body = self._request(service, "POST", "/approve", {
            "approval_id": "abc123def456", "approver": "test"})
        self.assertEqual(code, 200)
        code, snapshot = self._request(service, "GET", "/approvals")
        self.assertEqual(snapshot["approvals"], [])
        # Same id twice: loud refusal, nothing popped.
        code, body = self._request(service, "POST", "/approve", {
            "approval_id": "abc123def456"})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_neither_id_nor_index_rejected(self):
        service = self._service()
        code, body = self._request(service, "POST", "/approve",
                                   {"approver": "test"})
        self.assertEqual(code, 400)
        self.assertIn("error", body)


class TestRunCmdFailSafe(unittest.TestCase):
    def test_nonzero_with_stdout_is_unavailable(self):
        if sys.platform.startswith("win"):
            failing = ["cmd", "/c", "echo hi & exit 1"]
            passing = ["cmd", "/c", "echo hi"]
        else:
            failing = ["sh", "-c", "echo hi; exit 1"]
            passing = ["sh", "-c", "echo hi"]
        # A command that FAILS but still prints must NOT have its
        # output parsed as telemetry (old code parsed it).
        self.assertIsNone(run_cmd(failing))
        # Sanity: the same output with exit 0 is legitimate.
        self.assertIn("hi", run_cmd(passing))

    def test_missing_binary_is_unavailable(self):
        self.assertIsNone(run_cmd(["definitely-not-a-real-binary-xyz"]))


if __name__ == "__main__":
    unittest.main()
