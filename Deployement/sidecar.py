"""Policy-inference sidecar launcher (loopback HTTP, stdlib only).

Mock policy (deploy venv is enough)::

    python -m Deployement.sidecar --port 8410 --policy mock --mode shadow

Trained policy (training venv + checkpoint)::

    python -m Deployement.sidecar --port 8410 --policy trained \\
        --checkpoint checkpoints/fixedMaybe/mappo_final.pt \\
        --mode supervised

The C# SOC console (Deployement/CyberMarl.Deployment/) connects with
INFERENCE_URL/INFERENCE_TOKEN. The token prints ONCE at startup (or set
INFERENCE_TOKEN yourself); only its fingerprint is ever logged again.

Safety: binds 127.0.0.1 unless --allow-remote is given (logged LOUDLY).
Modes/masks/approvals/backends are enforced by the owned
RealWorldPipeline -- this launcher only wires and serves it.
"""

import argparse
import os
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cyber MARL policy-inference sidecar")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default loopback)")
    parser.add_argument("--port", type=int, default=8410)
    parser.add_argument("--policy", choices=("mock", "trained"),
                        default="mock")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode",
                        choices=("shadow", "mock", "supervised", "live"),
                        default="shadow")
    parser.add_argument("--enable-live", action="store_true",
                        help="explicit opt-in for live enforcement mode")
    parser.add_argument("--backend", default="null",
                        help="live-mode backend spec (default refuses)")
    parser.add_argument("--baselines", default=None,
                        help="baselines JSON path (default: {} = "
                             "everything is an event; noisy but safe)")
    parser.add_argument("--fleet", action="store_true",
                        help="multi-host fleet mode (local sensing + "
                             "ARP-observed peers + --inventory hosts; "
                             "uncovered assets read STALE, never quiet)")
    parser.add_argument("--inventory", default=None,
                        help="declared real-host fleet JSON "
                             "(implies --fleet)")
    parser.add_argument("--alerts", default=None,
                        help="detector alert feed JSONL (a REAL sensor "
                             "appends intrusion-class alerts; missing "
                             "file = no compromise coverage, loud)")
    parser.add_argument("--no-arp", action="store_true",
                        help="skip ARP wire observation in fleet mode")
    parser.add_argument("--allow-remote", action="store_true",
                        help="allow binding a non-loopback address "
                             "(the Bearer token then crosses the "
                             "network: use only behind TLS/auth)")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1") \
            and not args.allow_remote:
        parser.error("refusing non-loopback --host without "
                     "--allow-remote (token would cross the network)")
    if args.mode == "live" and not args.enable_live:
        parser.error("--mode live requires --enable-live (explicit "
                     "opt-in: approved ops execute for real)")
    if args.policy == "trained" and not args.checkpoint:
        parser.error("--policy trained needs --checkpoint")
    return args


def build_local_collector(alerts_path=None):
    """Local collector with genuine OS sensors (processes, TCP
    connections, services) plus an optional detector alert feed."""
    from .endpoint_sensors import (ConnectionSensor,
                                   EndpointTelemetryBuilder,
                                   FileAlertFeed, ProcessSensor,
                                   ServiceSensor)
    from .real_world import LocalMachineCollector
    collector = LocalMachineCollector()
    asset = collector.topology.assets[0]
    sensors = [ProcessSensor(), ConnectionSensor(), ServiceSensor()]
    if alerts_path:
        sensors.append(FileAlertFeed(alerts_path,
                                     asset_id=asset.asset_id,
                                     hostname=asset.hostname))
    collector.builder = EndpointTelemetryBuilder(
        asset.asset_id, hostname=asset.hostname, sensors=sensors)
    return collector


def main(argv=None):
    args = parse_args(argv)
    from .action_table import build_all_tables
    from .inference_service import InferenceService, token_fingerprint
    from .model_loader import DeploymentError
    from .policy_engine import MockPolicyEngine
    from .real_pipeline import RealWorldPipeline

    baselines = {}
    if args.baselines:
        from .site_config import load_baselines
        try:
            baselines = load_baselines(args.baselines)
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                f"cannot load baselines {args.baselines!r}: "
                f"{exc}") from exc
    collector = build_local_collector(args.alerts)
    fleet_note = "local-machine"
    if args.fleet or args.inventory:
        from .network_inventory import (DeclaredInventory,
                                        FleetCollector, local_subnets,
                                        read_arp_table)
        declared = ()
        if args.inventory:
            try:
                declared = DeclaredInventory.from_json_file(
                    args.inventory).to_assets()
            except Exception as exc:
                raise DeploymentError(
                    f"cannot load declared inventory: {exc}") from exc
        local_asset_fn = lambda: collector.topology.assets[0]  # noqa: E731
        fleet = FleetCollector(
            collector.builder, local_asset_fn,
            declared_assets=declared,
            arp_fn=(lambda: ([], False, "disabled by --no-arp")
                    if args.no_arp else read_arp_table()),
            subnets_fn=local_subnets)
        real_collector, real_topology = fleet, fleet.topology
        fleet_note = (f"fleet: {len(real_topology.assets)} asset(s), "
                      f"{len(real_topology.links)} link(s)")
    else:
        real_collector, real_topology = collector, collector.topology
    policy_info = {"engine": "mock-policy", "checkpoint": None}
    from .policy_adapter import PolicyAdapter
    probe_map, _ = PolicyAdapter(
        real_topology).build_policy_map()
    if args.policy == "trained":
        from .model_loader import load_trained_mappo
        from .policy_engine import TrainedPolicyEngine
        mappo, meta = load_trained_mappo(args.checkpoint)
        policy = TrainedPolicyEngine(
            mappo, build_all_tables(probe_map),
            meta["num_host_targets"])
        policy_info = {"engine": "trained-mappo",
                       "checkpoint": args.checkpoint,
                       "num_host_targets": meta.get("num_host_targets"),
                       "num_subnet_targets": meta.get(
                           "num_subnet_targets"),
                       "device": meta.get("device")}
    else:
        policy = MockPolicyEngine(build_all_tables(probe_map))
    pipeline = RealWorldPipeline(
        real_collector, policy,
        real_topology=real_topology, mode=args.mode,
        baselines=baselines, live_backend=args.backend,
        policy_info=policy_info)
    # The pipeline owns canonical tables built from an equal
    # deterministic map; re-point the engine so decide/execute agree.
    policy.tables = pipeline.tables
    pipeline.inner.policy = policy
    token = os.environ.get("INFERENCE_TOKEN")
    service = InferenceService(host=args.host, port=args.port,
                               token=token, real_pipeline=pipeline,
                               policy=pipeline.inner.policy,
                               tables=pipeline.tables,
                               policy_info=policy_info)
    service.start()
    try:
        print(f"sidecar on {service.base_url} "
              f"engine={policy_info['engine']} mode={args.mode} "
              f"topology={fleet_note}")
        print(f"contract: obs 210 / act 242 / vocab 137 / "
              f"real-first/v1")
        print(f"mapping: {pipeline.mapping_summary()}")
        if service.generated_token:
            print(f"INFERENCE_TOKEN={service.token} "
                  f"(generated; store it now, shown once)")
        else:
            print(f"token fingerprint: "
                  f"{token_fingerprint(service.token)}")
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print("WARNING: bound to a non-loopback address with "
                  "--allow-remote; protect the token in transit")
        print("serving (Ctrl-C to stop)…")
        import threading
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
