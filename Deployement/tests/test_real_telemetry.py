"""Real telemetry + fleet tests (parsers, honesty, failure cases).

Every ``parse_*`` test uses fixtures (Windows AND Linux samples run on
any platform). Live-sensor tests assert SHAPE + honesty (available flag
+ reason), never exact OS contents. Failure cases assert loud refusal,
never silent fiction and never fabricated compromise evidence.
"""

import json
import os
import tempfile
import unittest

from Deployement.action_table import build_all_tables
from Deployement.endpoint_sensors import (
    COMPONENT_KIND as SENSORS_KIND)
from Deployement.endpoint_sensors import (
    EndpointTelemetryBuilder, FileAlertFeed, SensorReading,
    SyslogAuthSensor, parse_auth_log, parse_netstat_tcp,
    parse_proc_net_tcp, parse_sc_query, parse_systemctl_services,
    parse_tasklist_csv)
from Deployement.network_inventory import (
    COMPONENT_KIND as INVENTORY_KIND)
from Deployement.network_inventory import (
    ArpEntry, DeclaredInventory, FleetCollector, InventoryError,
    build_fleet_topology, parse_arp_windows, parse_ip_addr,
    parse_ip_neigh, parse_netsh_addresses, parse_proc_net_arp)
from Deployement.normalizer import TelemetryNormalizer
from Deployement.policy_adapter import PolicyAdapter
from Deployement.policy_engine import MockPolicyEngine
from Deployement.real_actions import translate_decision
from Deployement.real_pipeline import RealWorldPipeline
from Deployement.real_world import (RealAsset, RealSegment, RealTopology,
                                    asset_id_for_hostname)
from Deployement.telemetry import (HostTelemetry, SecurityEvent,
                                   TelemetryBatch)


BASELINES = {"processes": ["sshd"], "ports": [22],
             "peers": ["10.0.0.1"]}


def _asset(name, ip, coverage="unknown"):
    return RealAsset(asset_id=asset_id_for_hostname(name),
                     hostname=name, ips=(ip,), macs=(),
                     os_name="TestOS", segment_id="host-only",
                     coverage=coverage)


class _StubSensor:
    def __init__(self, name="stub", procs=(), conns=(), svcs=(),
                 events=(), available=True, reason=""):
        self.name = name
        self._reading = SensorReading(
            name=name, processes=list(procs), connections=list(conns),
            services=list(svcs), events=list(events),
            available=available, reason=reason)

    def collect(self):
        return self._reading


TASKLIST_SAMPLE = (
    '"System Idle Process","0","Services","0","8 K"\r\n'
    '"svchost.exe","1234","Services","0","10,240 K"\r\n'
    '"notepad.exe","5678","Console","1","12,000 K"\r\n')

NETSTAT_SAMPLE = (
    "Active Connections\r\n"
    "  Proto  Local Address          Foreign Address        State           PID\r\n"  # noqa: E501
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1234\r\n"  # noqa: E501
    "  TCP    192.168.1.10:49712     93.184.216.34:443      ESTABLISHED     5678\r\n"  # noqa: E501
    "  TCP    192.168.1.10:49713     93.184.216.34:443      TIME_WAIT       5678\r\n"  # noqa: E501
    "  UDP    0.0.0.0:500            *:*                                    1111\r\n")  # noqa: E501

PROC_TCP_SAMPLE = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"  # noqa: E501
    "   0: 0100007F:0035 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1234 1 0000000000000000 100 0 0 10 0\n"  # noqa: E501
    "   1: 0A00020F:01BB 22D8B85D:01BB 01 00000000:00000000 00:00000000 00000000     0        0 1235 1 0000000000000000 100 0 0 10 0\n"  # noqa: E501
    "   2: 0A00020F:01BC 22D8B85D:01BB 06 00000000:00000000 00:00000000 00000000     0        0 1236 1 0000000000000000 100 0 0 10 0\n")  # noqa: E501

PROC_TCP6_SAMPLE = (
    "  sl  local_address                         rem_address                          st\n"  # noqa: E501
    "   0: 00000000000000000000000001000000:0035 00000000000000000000000000000000:0000 0A\n")  # noqa: E501

SC_SAMPLE = (
    "SERVICE_NAME: wuauserv\r\n"
    "DISPLAY_NAME: Windows Update\r\n"
    "        TYPE               : 20  WIN32_SHARE_PROCESS\r\n"
    "        STATE              : 4  RUNNING\r\n"
    "SERVICE_NAME: Spooler\r\n"
    "        STATE              : 1  STOPPED\r\n")

SYSTEMCTL_SAMPLE = (
    "cron.service     loaded active running Regular background program processing daemon\n"  # noqa: E501
    "ssh.service      loaded active running OpenBSD Secure Shell server\n"
    "sockets.target   loaded active active  Sockets\n")

ARP_WIN_SAMPLE = (
    "Interface: 192.168.1.10 --- 0x4\r\n"
    "  Internet Address      Physical Address      Type\r\n"
    "  192.168.1.1           aa-bb-cc-dd-ee-ff     dynamic\r\n"
    "  192.168.1.255         ff-ff-ff-ff-ff-ff     static\r\n"
    "  224.0.0.22            01-00-5e-00-00-16     static\r\n")

PROC_ARP_SAMPLE = (
    "IP address       HW type     Flags       HW address            Mask     Device\n"  # noqa: E501
    "192.168.1.1      0x1         0x2         aa:bb:cc:dd:ee:ff     *        eth0\n"  # noqa: E501
    "192.168.1.99     0x1         0x0         00:00:00:00:00:00     *        eth0\n")  # noqa: E501

IP_NEIGH_SAMPLE = (
    "192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE\n"
    "192.168.1.2 dev eth0 lladdr bb:cc:dd:ee:ff:00 STALE\n"
    "192.168.1.3 dev eth0  FAILED\n")

IP_ADDR_SAMPLE = (
    "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever\n"  # noqa: E501
    "2: eth0    inet 172.24.16.5/20 brd 172.24.31.255 scope global eth0\\       valid_lft forever preferred_lft forever\n")  # noqa: E501

NETSH_SAMPLE = (
    'Configuration for interface "Ethernet"\r\n'
    "    DHCP enabled:                         Yes\r\n"
    "    IP Address:                           192.168.29.10\r\n"
    "    Subnet Prefix:                        192.168.29.0/24 (mask 255.255.255.0)\r\n")  # noqa: E501

AUTH_LOG_SAMPLE = (
    "Sep  6 10:00:01 host sshd[123]: Failed password for root from 1.2.3.4 port 22 ssh2\n"  # noqa: E501
    "Sep  6 10:00:02 host CRON[124]: pam_unix(cron:session): session opened\n")


class TestComponentLabels(unittest.TestCase):
    def test_kinds_declared(self):
        self.assertEqual(SENSORS_KIND, "real")
        self.assertEqual(INVENTORY_KIND, "real")


class TestWindowsParsers(unittest.TestCase):
    def test_tasklist(self):
        names = parse_tasklist_csv(TASKLIST_SAMPLE)
        self.assertEqual(names, ["System Idle Process", "notepad.exe",
                                 "svchost.exe"])

    def test_netstat_tcp_states(self):
        conns = parse_netstat_tcp(NETSTAT_SAMPLE)
        # LISTEN + ESTABLISHED kept; TIME_WAIT + UDP excluded.
        self.assertEqual(conns, ["tcp:135",
                                 "tcp:49712->93.184.216.34"])

    def test_sc_query(self):
        self.assertEqual(parse_sc_query(SC_SAMPLE),
                         ["Spooler", "wuauserv"])

    def test_arp_windows(self):
        entries = parse_arp_windows(ARP_WIN_SAMPLE)
        by_ip = {e.ip: e for e in entries}
        self.assertIn("192.168.1.1", by_ip)
        self.assertEqual(by_ip["192.168.1.1"].mac, "aa-bb-cc-dd-ee-ff")
        # Multicast is not a peer but still parses (filtered later).
        self.assertIn("224.0.0.22", by_ip)

    def test_netsh(self):
        addrs = parse_netsh_addresses(NETSH_SAMPLE)
        self.assertEqual(addrs, {"Ethernet": [("192.168.29.10", 24)]})


class TestLinuxParsers(unittest.TestCase):
    def test_proc_net_tcp(self):
        conns = parse_proc_net_tcp(PROC_TCP_SAMPLE)
        self.assertEqual(conns, ["tcp:443->93.184.216.34", "tcp:53"])

    def test_proc_net_tcp6_loopback(self):
        conns = parse_proc_net_tcp("", PROC_TCP6_SAMPLE)
        self.assertEqual(conns, ["tcp:53"])

    def test_systemctl(self):
        self.assertEqual(parse_systemctl_services(SYSTEMCTL_SAMPLE),
                         ["cron", "ssh"])

    def test_proc_net_arp_skips_incomplete(self):
        entries = parse_proc_net_arp(PROC_ARP_SAMPLE)
        self.assertEqual([e.ip for e in entries], ["192.168.1.1"])
        self.assertEqual(entries[0].iface, "eth0")

    def test_ip_neigh_skips_failed(self):
        entries = parse_ip_neigh(IP_NEIGH_SAMPLE)
        self.assertEqual([e.ip for e in entries],
                         ["192.168.1.1", "192.168.1.2"])

    def test_ip_addr_skips_loopback(self):
        addrs = parse_ip_addr(IP_ADDR_SAMPLE)
        self.assertEqual(addrs, {"eth0": [("172.24.16.5", 20)]})

    def test_auth_log_never_compromise(self):
        events = parse_auth_log(AUTH_LOG_SAMPLE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "failed_login")
        self.assertNotIn(events[0].kind,
                         ("intrusion_confirmed", "red_session"))
        self.assertNotEqual(events[0].severity, "critical")


class TestSensorHonesty(unittest.TestCase):
    def test_missing_feed_is_unavailable_not_empty(self):
        feed = FileAlertFeed("/nonexistent/path/alerts.jsonl",
                             asset_id="host:pc-01")
        available, reason = feed.check_available()
        self.assertFalse(available)
        self.assertIn("coverage", reason)
        reading = feed.collect()
        self.assertFalse(reading.available)
        self.assertEqual(reading.events, [])

    def test_feed_intrusion_passes_through_verbatim(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False,
                encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "intrusion_confirmed", "severity": "critical",
                "details": "red session", "timestamp": 5.0,
                "asset": "host:pc-01"}) + "\n")
            fh.write(json.dumps({
                "kind": "ids_alert", "severity": "low",
                "details": "scan", "timestamp": 6.0,
                "asset": "host:other"}) + "\n")
            path = fh.name
        self.addCleanup(os.remove, path)
        feed = FileAlertFeed(path, asset_id="host:pc-01",
                             hostname="pc-01")
        reading = feed.collect()
        self.assertTrue(reading.available)
        self.assertEqual(len(reading.events), 1)
        self.assertEqual(reading.events[0].kind,
                         "intrusion_confirmed")
        self.assertIn("skipped-other-assets=1", reading.notes)

    def test_corrupt_feed_is_refused(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False,
                encoding="utf-8") as fh:
            fh.write("this is not json\n")
            path = fh.name
        self.addCleanup(os.remove, path)
        reading = FileAlertFeed(path).collect()
        self.assertFalse(reading.available)
        self.assertIn("malformed", reading.reason)

    def test_syslog_missing_path_unavailable(self):
        reading = SyslogAuthSensor("/nonexistent/auth.log").collect()
        self.assertFalse(reading.available)

    def test_builder_merges_and_reports_gaps(self):
        builder = EndpointTelemetryBuilder(
            "host:pc-01", hostname="pc-01", sensors=[
                _StubSensor("ok", procs=["sshd"], conns=["tcp:22"],
                            svcs=["cron"]),
                _StubSensor("broken", available=False,
                            reason="no such source")])
        host, partial = builder.build(now=100.0)
        self.assertEqual(host.key, "host:pc-01")
        self.assertEqual(host.processes, ["sshd"])
        self.assertEqual(host.services, ["cron"])
        self.assertTrue(any("broken" in e["error"] for e in partial))
        capabilities = builder.capabilities()
        self.assertTrue(capabilities["ok"]["available"])
        self.assertFalse(capabilities["broken"]["available"])

    def test_sensor_exception_never_kills_cycle(self):
        class _Boom:
            name = "boom"

            def collect(self):
                raise RuntimeError("driver exploded")

        host, partial = EndpointTelemetryBuilder(
            "host:pc-01", sensors=[_Boom()]).build(now=1.0)
        self.assertEqual(host.processes, [])
        self.assertTrue(any("boom" in e["error"] for e in partial))


class TestDeclaredInventory(unittest.TestCase):
    def _write(self, obj):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False,
                encoding="utf-8") as fh:
            json.dump(obj, fh)
            return fh.name

    def test_valid_file(self):
        path = self._write({"hosts": [
            {"hostname": "srv-01", "ips": ["192.168.29.11"],
             "role": "server", "segment": "srv"}]})
        self.addCleanup(os.remove, path)
        inventory = DeclaredInventory.from_json_file(path)
        assets = inventory.to_assets()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].asset_id, "host:srv-01")
        self.assertEqual(assets[0].coverage, "uncovered")

    def test_missing_file(self):
        with self.assertRaises(InventoryError):
            DeclaredInventory.from_json_file("/nonexistent/x.json")

    def test_bad_ip_and_duplicates(self):
        bad_ip = self._write({"hosts": [
            {"hostname": "a", "ips": ["999.1.1.1"]}]})
        self.addCleanup(os.remove, bad_ip)
        with self.assertRaises(InventoryError):
            DeclaredInventory.from_json_file(bad_ip)
        dup = self._write({"hosts": [
            {"hostname": "a", "ips": ["10.0.0.1"]},
            {"hostname": "a", "ips": ["10.0.0.2"]}]})
        self.addCleanup(os.remove, dup)
        with self.assertRaises(InventoryError):
            DeclaredInventory.from_json_file(dup)
        no_name = self._write({"hosts": [{"ips": ["10.0.0.1"]}]})
        self.addCleanup(os.remove, no_name)
        with self.assertRaises(InventoryError):
            DeclaredInventory.from_json_file(no_name)


class TestFleetTopology(unittest.TestCase):
    def test_local_declared_arp_with_segments(self):
        import ipaddress
        local = _asset("pc-01", "192.168.29.10",
                       coverage="covered-local")
        declared = [RealAsset(
            asset_id="host:srv-01", hostname="srv-01",
            ips=("192.168.29.11",), segment_id="srv",
            coverage="uncovered",
            coverage_detail="declared by operator; no sensor coverage "
                            "-- state unknown (reads STALE)")]
        arp = [ArpEntry(ip="192.168.29.12", mac="aa:bb:cc:dd:ee:01"),
               ArpEntry(ip="192.168.29.255",
                        mac="ff:ff:ff:ff:ff:ff"),  # broadcast: no stub
               ArpEntry(ip="224.0.0.22",
                        mac="01:00:5e:00:00:16")]  # multicast: no stub
        subnets = [ipaddress.ip_network("192.168.29.0/24")]
        topology = build_fleet_topology(
            local, declared_assets=declared, arp_entries=arp,
            subnets=subnets)
        by_id = {a.asset_id: a for a in topology.assets}
        # Local + declared + exactly one ARP stub (broadcast/multicast
        # are not peers and never become assets).
        self.assertEqual(
            sorted(by_id),
            ["host:pc-01", "host:srv-01", "ip:192.168.29.12"])
        self.assertEqual(by_id["ip:192.168.29.12"].coverage,
                         "observed-arp-only")
        # Shared observed subnet -> one lan segment + subnet link.
        seg_ids = {s.segment_id for s in topology.segments}
        self.assertIn("lan-192.168.29.0_24", seg_ids)
        vias = {(l.asset_id, l.peer, l.via) for l in topology.links}
        self.assertIn(("host:pc-01", "ip:192.168.29.12", "arp"), vias)
        self.assertTrue(any(v == "subnet" for _, _, v in vias))
        # Deterministic rebuild.
        again = build_fleet_topology(
            local, declared_assets=declared, arp_entries=arp,
            subnets=subnets)
        self.assertEqual(
            [(a.asset_id, a.segment_id) for a in topology.assets],
            [(a.asset_id, a.segment_id) for a in again.assets])

    def test_no_subnets_no_guessing(self):
        local = _asset("pc-01", "10.9.9.9", coverage="covered-local")
        topology = build_fleet_topology(
            local, arp_entries=[ArpEntry(ip="10.9.9.10")], subnets=[])
        by_id = {a.asset_id: a for a in topology.assets}
        # No mask known -> per-host segments, ARP link only, no lan-*.
        self.assertEqual(by_id["host:pc-01"].segment_id, "host-only")
        self.assertEqual(by_id["ip:10.9.9.10"].segment_id, "observed")
        self.assertFalse(any(s.segment_id.startswith("lan-")
                             for s in topology.segments))

    def test_fleet_collector_covers_only_sensed(self):
        local = _asset("pc-01", "192.168.29.10",
                       coverage="covered-local")
        builder = EndpointTelemetryBuilder(
            "host:pc-01", sensors=[_StubSensor("ok", procs=["sshd"])])
        declared = [RealAsset(
            asset_id="host:srv-01", hostname="srv-01",
            ips=("192.168.29.11",), coverage="uncovered",
            coverage_detail="declared")]
        collector = FleetCollector(
            builder, lambda: local, declared_assets=declared,
            arp_fn=lambda: ([ArpEntry(ip="192.168.29.12")], True, ""),
            subnets_fn=lambda: [])
        try:
            batch = collector.next_batch()
        finally:
            collector.close()
        # Only the sensed local asset is present; everyone else is
        # omitted (reads STALE downstream) and counted honestly.
        self.assertEqual(sorted(batch.hosts), ["host:pc-01"])
        self.assertIn("uncovered", batch.notes)
        self.assertTrue(batch.partial_errors)
        topology = collector.topology
        self.assertEqual(len(topology.assets), 3)
        self.assertFalse(collector.exhausted)  # live: no end-of-stream


class TestServicesNormalization(unittest.TestCase):
    def _normalize(self, services, baselines=BASELINES):
        from Deployement.policy_adapter import PolicyAdapter
        from Deployement.real_world import RealTopology
        asset = _asset("pc-01", "192.168.29.10")
        topology = RealTopology(
            assets=[asset], segments=[], source="test",
            discovered_at=1.0)
        policy_map, report = PolicyAdapter(topology).build_policy_map()
        slot_key = report.bindings[0]["slot_ip"]
        batch = TelemetryBatch(
            timestamp=10.0, source="test", hosts={
                slot_key: HostTelemetry(
                    key=slot_key, processes=["sshd"],
                    connections=["tcp:22->10.0.0.1"], up=True,
                    events=[], services=list(services))})
        normalized = TelemetryNormalizer(baselines).normalize(
            batch, policy_map, now=10.0)
        return normalized.hosts[report.bindings[0]["cc4"]]

    def test_unknown_service_is_activity_not_verdict(self):
        host = self._normalize(["weird_svc"])
        self.assertTrue(host.process_event)
        self.assertFalse(host.compromised)
        self.assertEqual(host.severity, "low")

    def test_baselined_service_stays_quiet(self):
        host = self._normalize(["sshd"])
        self.assertFalse(host.process_event)
        self.assertEqual(host.severity, "none")


class TestFleetEndToEnd(unittest.TestCase):
    def test_uncovered_reads_stale_covered_compromised(self):
        local = _asset("pc-01", "192.168.29.10",
                       coverage="covered-local")
        builder = EndpointTelemetryBuilder(
            "host:pc-01", sensors=[_StubSensor(
                "ok", procs=["sshd"], conns=["tcp:4444->1.2.3.4"],
                events=[SecurityEvent(
                    kind="intrusion_confirmed", severity="critical",
                    details="red session", timestamp=1.0)])])
        declared = [RealAsset(
            asset_id="host:srv-01", hostname="srv-01",
            ips=("192.168.29.11",), coverage="uncovered",
            coverage_detail="declared")]
        collector = FleetCollector(
            builder, lambda: local, declared_assets=declared,
            arp_fn=lambda: ([], True, ""), subnets_fn=lambda: [])
        pipeline = RealWorldPipeline(
            collector, MockPolicyEngine(build_all_tables(
                PolicyAdapter(collector.topology).build_policy_map()[0])),
            real_topology=collector.topology, mode="shadow",
            baselines=BASELINES)
        try:
            records, plans = pipeline.step()
            normalized = pipeline.inner.last_normalized
            report = pipeline.report
        finally:
            pipeline.close()
        self.assertEqual(len(records), 5)
        self.assertEqual(len(plans), 5)
        by_slot = {b["asset_id"]: b["cc4"] for b in report.bindings}
        # Genuine intrusion evidence reaches the covered slot...
        self.assertTrue(
            normalized.hosts[by_slot["host:pc-01"]].compromised)
        # ...while the uncovered declared asset reads STALE (unknown,
        # investigate) -- never quiet, never healthy.
        stale = normalized.hosts[by_slot["host:srv-01"]]
        self.assertEqual(stale.health, "stale")
        self.assertFalse(stale.compromised)
        # Translation still resolves the covered asset really.
        agent = int(next(
            b["agent"] for b in report.bindings
            if b["asset_id"] == "host:pc-01").rsplit("_", 1)[-1])
        tables = pipeline.tables
        cc4 = by_slot["host:pc-01"]
        index = next(i for i, e in enumerate(tables[agent])
                     if e.get("target") == cc4
                     and e["command"] == "Analyse")
        plan = translate_decision(agent, index, tables,
                                  pipeline.policy_map, report)
        self.assertFalse(plan.blocked)
        self.assertEqual(plan.asset_id, "host:pc-01")

    def test_multi_asset_mapping_stable(self):
        assets = [_asset(f"pc-{i:02d}", f"192.168.29.{10 + i}")
                  for i in range(3)]
        from Deployement.real_world import RealTopology
        first = PolicyAdapter(RealTopology(
            assets=assets, segments=[], source="test",
            discovered_at=1.0)).build_policy_map()[1]
        self.assertEqual(len(first.bindings), 3)
        more = assets + [_asset("pc-04", "192.168.29.14")]
        second = PolicyAdapter(RealTopology(
            assets=more, segments=[], source="test",
            discovered_at=2.0)).build_policy_map(previous=first)[1]
        for binding in first.bindings:
            self.assertEqual(
                second.slot_of_asset(binding["asset_id"]),
                (binding["agent"], binding["cc4"]))
        self.assertEqual(len(second.bindings), 4)


class TestCycleTopologyEcho(unittest.TestCase):
    def test_cycle_returns_real_topology_with_links(self):
        from Deployement.inference_service import InferenceService
        asset = _asset("pc-01", "192.168.29.10")
        from Deployement.real_world import RealTopology
        from Deployement.telemetry import TelemetryCollector
        from Deployement.real_pipeline import RealWorldPipeline as RWP

        class _OneShot(TelemetryCollector):
            def __init__(self, topology, batch):
                self._topology = topology
                self._batch = batch
                self._done = False

            @property
            def topology(self):
                return self._topology

            @property
            def exhausted(self):
                return self._done

            def next_batch(self):
                if self._done:
                    return None
                self._done = True
                return self._batch

            def close(self):
                pass

        topology = RealTopology(
            assets=[asset], segments=[], source="test",
            discovered_at=1.0)
        batch = TelemetryBatch(
            timestamp=20.0, source="test", hosts={
                "host:pc-01": HostTelemetry(
                    key="host:pc-01", processes=["sshd"],
                    connections=[], up=True, events=[],
                    services=["cron"])})
        collector = _OneShot(topology, batch)
        pipeline = RWP(
            collector, MockPolicyEngine(build_all_tables(
                PolicyAdapter(topology).build_policy_map()[0])),
            real_topology=topology, mode="shadow",
            baselines=BASELINES)
        self.addCleanup(pipeline.close)
        service = InferenceService(
            token="test-token", real_pipeline=pipeline,
            policy=pipeline.inner.policy, tables=pipeline.tables,
            policy_info={"engine": "mock-policy"})
        service.start()
        self.addCleanup(service.stop)
        import json
        import urllib.request
        payload = {
            "topology": {
                "assets": [{"asset_id": "host:pc-01",
                            "hostname": "pc-01",
                            "ips": ["192.168.29.10"],
                            "macs": ["aa:bb:cc:dd:ee:ff"],
                            "coverage": "covered-local",
                            "coverage_detail": "sensed",
                            "services": ["cron"],
                            "segment_id": "host-only"}],
                "segments": [{"segment_id": "host-only"}],
                "links": [{"asset_id": "host:pc-01",
                           "peer": "ip:192.168.29.12",
                           "via": "arp"}]},
            "batch": {"timestamp": 20.0, "source": "test", "hosts": {
                "host:pc-01": {"processes": ["sshd"],
                               "connections": [], "up": True,
                               "events": [],
                               "services": ["cron"]}}}}
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            service.base_url + "/cycle", data=data, method="POST",
            headers={"Authorization": "Bearer test-token",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        echo = result["real_topology"]
        self.assertEqual(len(echo["assets"]), 1)
        self.assertEqual(echo["assets"][0]["coverage"],
                         "covered-local")
        self.assertEqual(echo["assets"][0]["services"], ["cron"])
        self.assertEqual(echo["assets"][0]["macs"],
                         ["aa:bb:cc:dd:ee:ff"])
        self.assertEqual(echo["links"],
                         [{"asset_id": "host:pc-01",
                           "peer": "ip:192.168.29.12", "via": "arp"}])
        # No CC4 vocabulary in the real-topology echo.
        blob = json.dumps(echo)
        self.assertNotIn("restricted_zone", blob)
        self.assertNotIn("subnet", blob)


if __name__ == "__main__":
    unittest.main()
