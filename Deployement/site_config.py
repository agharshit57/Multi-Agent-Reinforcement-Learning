"""Site configuration: files for settings, environment for secrets.

Rule: anything sensitive (tokens, passwords, API keys) is referenced
BY ENVIRONMENT VARIABLE NAME ONLY (``*_env`` keys) and resolved at
runtime via ``live.resolve_secret``. Config files must never contain
secret values -- ``validate_no_embedded_secrets()`` rejects them loudly
(checks key names AND common accidents like a token pasted into an
endpoint URL).

Example ``site.json``::

    {"assets": "site-assets.json",
     "checkpoint": "checkpoints/fixedMaybe/mappo_final.pt",
     "policy": "trained", "mode": "supervised",
     "mission_phase": 0, "cooldown_s": 300,
     "session_dir": "sessions/site-a",
     "stale_after_s": 300,
     "collector": {"type": "http",
                   "endpoint": "https://siem.local/api/hosts",
                   "timeout_s": 10.0, "token_env": "DEPLOY_SIEM_TOKEN",
                   "retries": 2, "backoff_s": 1.0,
                   "max_response_bytes": 8388608},
     "baselines_path": "site-baselines.json",
     "backend": "null"}

Environment overrides (take precedence over the file):
    DEPLOY_ASSETS, DEPLOY_CHECKPOINT, DEPLOY_POLICY, DEPLOY_MODE,
    DEPLOY_MISSION_PHASE, DEPLOY_COOLDOWN_S, DEPLOY_SESSION_DIR,
    DEPLOY_COLLECTOR_ENDPOINT, DEPLOY_MAX_RESPONSE_BYTES,
    DEPLOY_STALE_AFTER_S, DEPLOY_BACKEND
"""

import json
import os

SUSPICIOUS_VALUE_HINTS = ("bearer ", "token=", "api_key=", "passwd")


def validate_no_embedded_secrets(obj, where="site config"):
    """Reject configs that appear to embed secret VALUES."""
    problems = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(n in lowered for n in
                       ("token", "secret", "password", "api_key", "apikey",
                        "cookie", "authorization")):
                    if isinstance(value, str) and value and not value.startswith("$"):  # noqa: E501
                        # A non-empty value on a secret-NAMED key: only
                        # acceptable for "*_env" reference keys.
                        if not str(key).lower().endswith("_env"):
                            problems.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, str):
            lowered = node.lower()
            if ("://" in node and "@" in node.split("://", 1)[1]):
                problems.append(f"{path} (credentials embedded in URL)")
            elif any(h in lowered for h in SUSPICIOUS_VALUE_HINTS):
                problems.append(f"{path} (looks like an embedded secret)")

    walk(obj, where)
    if problems:
        raise ValueError(
            "refusing to load config with possible embedded secrets at: "
            + ", ".join(problems) + ". Move secrets to environment "
            "variables and reference them with '*_env' keys.")
    return True


def _as_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"site config: {name} must be a number, "
                         f"got {value!r}")


def load_site_config(path=None):
    """Load + merge site config. Returns a plain validated dict."""
    cfg = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            try:
                cfg = json.load(fh)
            except Exception as exc:
                raise ValueError(
                    f"site config {path!r} is not valid JSON: {exc}")
        if not isinstance(cfg, dict):
            raise ValueError("site config root must be an object")
        validate_no_embedded_secrets(cfg, where=path)
    env = os.environ

    def pick(key, *names, default=None, coerce=None):
        for name in names:
            if name in env and env[name] != "":
                value = env[name]
                return coerce(value) if coerce else value
        value = cfg.get(key, default)
        return value

    try:
        mission_phase = pick("mission_phase", "DEPLOY_MISSION_PHASE",
                             default=0, coerce=int)
    except ValueError:
        raise ValueError("site config: mission_phase must be an integer "
                         "0, 1, or 2")
    if mission_phase not in (0, 1, 2):
        # Fail HERE with a clean config error -- not downstream where an
        # out-of-range phase would silently select the wrong comms-policy
        # subgraph (or crash the observation builder).
        raise ValueError(f"site config: mission_phase must be 0 "
                         f"(Preplanning), 1 (MissionA), or 2 (MissionB); "
                         f"got {mission_phase}")
    merged = {
        "assets": pick("assets", "DEPLOY_ASSETS", default=None),
        "checkpoint": pick("checkpoint", "DEPLOY_CHECKPOINT", default=None),
        "policy": pick("policy", "DEPLOY_POLICY", default="mock"),
        "mode": pick("mode", "DEPLOY_MODE", default="shadow"),
        "mission_phase": mission_phase,
        "cooldown_s": pick("cooldown_s", "DEPLOY_COOLDOWN_S",
                           default=300, coerce=_as_float_safe),
        "session_dir": pick("session_dir", "DEPLOY_SESSION_DIR",
                            default=None),
        "stale_after_s": pick("stale_after_s", "DEPLOY_STALE_AFTER_S",
                              default=300, coerce=_as_float_safe),
        "collector": dict(cfg.get("collector", {})),
        "baselines_path": pick("baselines_path", "DEPLOY_BASELINES",
                               default=None),
        "backend": pick("backend", "DEPLOY_BACKEND", default="null"),
    }
    endpoint = env.get("DEPLOY_COLLECTOR_ENDPOINT", "")
    if endpoint:
        merged["collector"]["endpoint"] = endpoint
    if merged["mode"] not in ("shadow", "mock", "supervised", "live"):
        raise ValueError(
            f"site config: unknown mode {merged['mode']!r}")
    if merged["policy"] not in ("auto", "mock", "trained"):
        raise ValueError(
            f"site config: unknown policy {merged['policy']!r}")
    return merged


def _as_float_safe(value):
    return _as_float(value, "numeric option")


def load_baselines(path_or_dict=None):
    """Load baselines from a JSON path or pass a dict through.

    Validated with normalizer.validate_baselines (same rules the
    normalizer enforces at construction): malformed files fail here,
    never half-load into detection.
    """
    from .normalizer import validate_baselines
    if path_or_dict is None:
        return None
    if isinstance(path_or_dict, dict):
        return validate_baselines(path_or_dict)
    with open(path_or_dict, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except Exception as exc:
            raise ValueError(
                f"baselines file {path_or_dict!r} is not valid JSON: "
                f"{exc}")
    if not isinstance(data, dict):
        raise ValueError("baselines file root must be an object")
    return validate_baselines(data)
