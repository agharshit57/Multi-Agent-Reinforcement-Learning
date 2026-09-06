"""Deployment app entry point (installable launcher).

Headless validator / demo runner::

    python -m Deployement.app --demo --headless --cycles 5 --mode shadow

Desktop console::

    python -m Deployement.app --demo --gui --mode mock

Production wiring (same venv as training, real checkpoint)::

    python -m Deployement.app --assets site-assets.json \\
        --checkpoint checkpoints/fixedMaybe/mappo_final.pt \\
        --mode supervised --gui
"""

import argparse
import os
import sys


def build_pipeline(args, asset_map, site=None):
    from .action_table import build_all_tables
    from .demo import demo_baselines, demo_collector
    from .live import (EndpointConfig, HttpJsonCollector,
                       ResilientCollector, resolve_secret)
    # Hoisted (not inside try bodies): except-clauses below reference
    # DeploymentError at try-setup time, before any body import runs.
    from .model_loader import DeploymentError
    from .pipeline import DeploymentPipeline
    from .policy_engine import MockPolicyEngine
    from .telemetry import FileCollector

    site = site or {}
    collector_cfg = dict(site.get("collector", {}))
    collector_type = collector_cfg.get("type", "demo")
    want_http = (args.collector or "").lower().startswith(
        ("http://", "https://"))
    if args.demo or (args.collector == "demo"
                     and collector_type == "demo"):
        collector = demo_collector(asset_map)
    elif want_http or (args.collector == "demo"
                       and collector_type in ("http", "https")):
        endpoint = (args.collector if want_http
                    else collector_cfg.get("endpoint", ""))
        token_env = collector_cfg.get("token_env", "")
        translate_name = collector_cfg.get("translate", "")
        if not endpoint:
            raise SystemExit(
                "error: HTTP collector needs an endpoint "
                "(--collector URL or site config collector.endpoint)")
        max_response_bytes = collector_cfg.get("max_response_bytes")
        if os.environ.get("DEPLOY_MAX_RESPONSE_BYTES", "") != "":
            max_response_bytes = os.environ["DEPLOY_MAX_RESPONSE_BYTES"]
        try:
            max_response_bytes = (
                int(max_response_bytes)
                if max_response_bytes is not None
                else 8 * 1024 * 1024)
        except (TypeError, ValueError):
            raise SystemExit(
                "error: max_response_bytes must be a positive byte "
                "count (site collector.max_response_bytes or "
                "DEPLOY_MAX_RESPONSE_BYTES)")
        cfg = EndpointConfig(
            name=collector_cfg.get("name", "live"),
            endpoint=endpoint,
            timeout_s=float(collector_cfg.get("timeout_s", 10.0)),
            max_response_bytes=max_response_bytes,
            token_env=token_env)
        if token_env:
            resolve_secret(token_env, cfg.name)  # fail fast, name only
        translate = _load_translate(translate_name)
        from .live import CollectorError
        try:
            collector = HttpJsonCollector(cfg, translate)
            collector = ResilientCollector(
                collector,
                retries=int(collector_cfg.get("retries", 2)),
                backoff_s=float(collector_cfg.get("backoff_s", 1.0)))
        except (CollectorError, ValueError) as exc:
            raise SystemExit(f"error: invalid live collector config: "
                             f"{exc}")
    else:
        try:
            collector = FileCollector(args.collector)
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                f"cannot open telemetry capture {args.collector!r}: "
                f"{exc}") from exc
    baselines = demo_baselines() if (args.demo or args.collector == "demo") \
        else None
    baselines_path = site.get("baselines_path")
    if baselines_path:
        from .site_config import load_baselines
        try:
            baselines = load_baselines(baselines_path)
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                f"cannot load baselines {baselines_path!r}: {exc}") from exc
    policy = None
    policy_name = "mock-policy"
    policy_info = {"engine": "mock-policy", "checkpoint": None}
    if args.policy == "trained":
        from .model_loader import load_trained_mappo
        from .policy_engine import TrainedPolicyEngine
        mappo, meta = load_trained_mappo(args.checkpoint)
        tables = build_all_tables(asset_map)
        policy = TrainedPolicyEngine(
            mappo, tables, meta["num_host_targets"])
        policy_name = (f"trained-mappo "
                       f"({os.path.basename(args.checkpoint)})")
        policy_info = {"engine": "trained-mappo",
                       "checkpoint": args.checkpoint,
                       "num_host_targets": meta.get("num_host_targets"),
                       "num_subnet_targets": meta.get("num_subnet_targets"),
                       "device": meta.get("device")}
    if policy is None:
        policy = MockPolicyEngine(build_all_tables(asset_map))
    # Backend selection: explicit CLI > site config > safe default.
    # Invalid specs fail HERE (clean CLI error), never as a silent
    # mid-run fallback to mock/null.
    backend_spec = (getattr(args, "backend", None)
                    or site.get("backend") or "null")
    from .executor import BackendError
    try:
        pipeline = DeploymentPipeline(
            asset_map, collector, policy, mode=args.mode,
            baselines=baselines, session_dir=args.session_dir,
            mission_phase=int(site.get("mission_phase", 0)),
            cooldown_s=float(site.get("cooldown_s", 300)),
            stale_after_s=float(site.get("stale_after_s", 300)),
            live_backend=backend_spec,
            policy_info=policy_info)
    except BackendError as exc:
        raise SystemExit(f"error: invalid enforcement backend: {exc}")

    def describe():
        return {"policy": policy_name, "assets": args.assets or "default",
                "mode": args.mode,
                "checkpoint": args.checkpoint or "-"}
    return pipeline, describe


def _load_translate(name):
    """Resolve the vendor translate adapter by dotted path.

    The translate function is site/vendor-specific code that lives
    OUTSIDE this repo (e.g. ``site_adapters.siem:translate_hosts``).
    Reviewed here only by contract: ``dict -> dict[key, HostTelemetry]``.
    """
    if not name:
        raise SystemExit(
            "error: HTTP collector needs a vendor translate adapter "
            "(site config collector.translate as 'module:object'). "
            "No vendor API is bundled -- see Deployement/README.md.")
    module_name, _, attr = name.partition(":")
    if not module_name or not attr:
        raise SystemExit(
            f"error: bad translate reference {name!r}; want "
            f"'module:object'")
    try:
        import importlib
        module = importlib.import_module(module_name)
        translate = getattr(module, attr)
    except Exception as exc:
        raise SystemExit(
            f"error: cannot load translate adapter {name!r}: {exc}")
    if not callable(translate):
        raise SystemExit(
            f"error: translate adapter {name!r} is not callable")
    return translate


def load_assets(path):
    # Operator-facing file errors (missing path, malformed JSON,
    # invalid map) surface as DeploymentError -> clean CLI exit 2,
    # never a raw traceback for a typo'd path.
    from .asset_map import AssetMap, generate_default_map
    from .model_loader import DeploymentError
    if path:
        try:
            return AssetMap.from_json_file(path)
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                f"cannot load asset map {path!r}: {exc}") from exc
    return generate_default_map()


def _resolve_policy(args, site):
    # Explicit CLI intent wins; site config only fills gaps. "auto"
    # means trained iff any checkpoint is available.
    if not args.checkpoint and site.get("checkpoint"):
        args.checkpoint = site["checkpoint"]
    if args.policy == "auto":
        args.policy = "trained" if args.checkpoint else "mock"
    if args.policy == "trained" and not args.checkpoint:
        raise SystemExit(
            "error: --policy trained needs --checkpoint (CLI or site "
            "config)")


def run_headless(args):
    from .site_config import load_site_config
    site = load_site_config(getattr(args, "site_config", None))
    _resolve_policy(args, site)
    asset_map = load_assets(args.assets or site.get("assets"))
    pipeline, describe = build_pipeline(args, asset_map, site)
    print(f"policy={describe()['policy']} mode={args.mode}")
    try:
        records = pipeline.run(max_cycles=args.cycles)
    finally:
        pipeline.close()
    by_status = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    print(f"cycles={pipeline.cycle} decisions={len(records)} "
          f"status={by_status}")
    for record in records:
        print(f"  cyc={record['cycle']} agent={record['agent_id']} "
              f"{record['label']} [{record['risk']}] -> "
              f"{record['operation']} ({record['status']})")
    return 0 if records else 1


def run_gui(args):
    from . import gui
    from .site_config import load_site_config
    site = load_site_config(getattr(args, "site_config", None))
    _resolve_policy(args, site)
    asset_map = load_assets(args.assets or site.get("assets"))

    def factory():
        return build_pipeline(args, asset_map, site)

    gui.launch(factory)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cyber MARL deployment console")
    parser.add_argument("--assets", default=None,
                        help="asset map JSON (default: built-in demo map)")
    parser.add_argument("--checkpoint", default=None,
                        help="trained .pt weights (needs --policy trained)")
    parser.add_argument("--policy", choices=("auto", "mock", "trained"),
                        default="auto",
                        help="auto=mock unless --checkpoint given with "
                             "'trained'")
    parser.add_argument("--collector", default="demo",
                        help="'demo' or a JSON-lines telemetry file")
    parser.add_argument("--demo", action="store_true",
                        help="scripted intrusion timeline")
    parser.add_argument("--mode", choices=("shadow", "mock", "supervised",
                                              "live"),
                        default="shadow")
    parser.add_argument("--enable-live", action="store_true",
                        help="explicit opt-in for live enforcement mode "
                             "(required with --mode live; destructive ops "
                             "execute for real via a configured backend)")
    parser.add_argument("--backend", default=None,
                        help="live-mode enforcement backend: 'null' refuses "
                             "everything loudly (default, safe); 'mock' "
                             "simulates against internal state; "
                             "'module:Class' loads a real site adapter "
                             "(zero-arg constructor, env credentials). "
                             "Site config 'backend' fills the gap when "
                             "omitted.")
    parser.add_argument("--site-config", default=None,
                        help="site JSON config (non-secret settings only; "
                             "secrets come from environment variables)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="headless max cycles (0 = run to exhaustion)")
    parser.add_argument("--session-dir", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true")
    group.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.policy == "trained" and not args.checkpoint \
            and not getattr(args, "site_config", None):
        # (A site file may still supply the checkpoint; resolved later.
        # Explicit --policy mock is always honored as-is.)
        parser.error("--policy trained needs --checkpoint "
                     "(or a site config providing one)")
    if args.mode == "live" and not args.enable_live:
        parser.error("--mode live requires --enable-live (explicit "
                     "opt-in: approved destructive ops will execute "
                     "for real via the configured backend)")
    if not args.gui and not args.headless:
        args.headless = True
    if args.cycles == 0:
        args.cycles = None
    return args


def _is_display_error(exc):
    """True only for Tk display failures (headless box, no $DISPLAY)."""
    try:
        from tkinter import TclError
    except Exception:
        return False
    return isinstance(exc, TclError)


def main(argv=None):
    # DeploymentError -> clean one-line CLI failure (exit 2), never a
    # raw traceback/ModuleNotFoundError for missing venv/checkpoints.
    # A display failure at GUI launch -> clean message (exit 3).
    from .model_loader import DeploymentError
    args = parse_args(argv)
    try:
        if args.gui:
            return run_gui(args)
        return run_headless(args)
    except DeploymentError as exc:
        print(f"deployement error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if args.gui and _is_display_error(exc):
            print(f"deployement error: cannot open a display for the "
                  f"console ({exc}); use --headless on servers",
                  file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
