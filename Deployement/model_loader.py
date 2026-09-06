"""Trained-weight loader: reuses the REAL training architecture.

No model code is duplicated here. Loading imports the actual classes
(``Marl.mappo.mappo.MAPPO`` over ``Marl/mappo/gnn_attention.py``) and
restores a ``checkpoints/.../*.pt`` file with the same validation the
training evaluator uses (host/subnet vocabulary check inside
``MAPPO.load``). Imports are LAZY (inside functions) so the rest of
the deployment package -- telemetry, observations, GUI scaffolding --
runs without torch or the training venv.

Requires: the training venv (``Requirements.txt``) + a checkpoint
produced by current training (stable 137-host vocabulary).
"""


class DeploymentError(RuntimeError):
    """Raised when the trained policy cannot be loaded/used."""


def try_training_imports():
    """Return (ok, reason). Never raises: probes training deps."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch unavailable: {exc}"
    try:
        from Marl.mappo.mappo import MAPPO  # noqa: F401
        from Marl.mappo.config import (  # noqa: F401
            NUM_AGENTS, OBS_DIM, ACTION_DIM)
    except Exception as exc:
        return False, f"training modules unavailable: {exc}"
    return True, "ok"


def load_trained_mappo(checkpoint_path, device=None):
    """Load a frozen MAPPO policy from ``checkpoint_path``.

    Returns (mappo, meta) with ``meta`` = {"num_host_targets",
    "num_subnet_targets", "trust" (frozen matrix list), "device"}.
    Trust is restored from the checkpoint and left FROZEN (production
    has no ground truth to update it -- same rule as evaluation).

    Fails ONLY with DeploymentError (never a raw ModuleNotFoundError):
    training-dependency availability is probed FIRST via
    try_training_imports(), before any direct torch/Marl import and
    before touching the checkpoint file.

    ``device`` is informational only: training MAPPO pins its own
    device from training config at construction/load time, and every
    deployment tensor is explicitly moved to ``mappo.device`` at
    inference. Passing anything other than None is rejected loudly
    rather than silently ignored, so a caller asking for cuda can
    never believe it got cuda while serving on cpu.
    """
    ok, reason = try_training_imports()
    if not ok:
        raise DeploymentError(
            f"Cannot load trained policy: {reason}. "
            "Run deployment from the training venv "
            "(see Deployement/README.md: install Requirements.txt, "
            "then Deployement/requirements-deploy.txt).")

    import os

    import torch

    if not os.path.isfile(checkpoint_path):
        raise DeploymentError(
            f"checkpoint not found: {checkpoint_path}")
    from Marl.mappo.mappo import MAPPO
    probe = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(probe, dict) or "model" not in probe:
        raise DeploymentError(
            f"{checkpoint_path} is not a MAPPO checkpoint "
            "(missing 'model' state_dict)")
    num_host = probe.get("num_host_targets")
    num_subnet = probe.get("num_subnet_targets")
    if num_host is None:
        raise DeploymentError(
            "Checkpoint predates the stable host vocabulary "
            "(no num_host_targets); retrain with current code.")
    try:
        mappo = MAPPO(num_host_targets=num_host,
                      num_subnet_targets=num_subnet)
        mappo.load(checkpoint_path)  # validates vocab sizes internally
    except DeploymentError:
        raise
    except Exception as exc:
        # Incompatible weights (vocabulary/shape/architecture drift)
        # must surface as an actionable deployment error, not a raw
        # torch/MAPPO traceback mid-incident.
        raise DeploymentError(
            f"Checkpoint {checkpoint_path} is incompatible with the "
            f"current deployment contract "
            f"(host_targets={num_host}, subnet_targets={num_subnet}): "
            f"{exc}") from exc
    mappo.eval()
    try:
        trust = mappo.get_trust_matrix().detach().cpu().tolist()
    except Exception:
        trust = None
    if device is not None and str(device) != str(mappo.device):
        raise DeploymentError(
            f"requested device {device!r} but the trained model serves "
            f"on {mappo.device} (training config pins the device; "
            f"deployment moves every tensor there explicitly). Pass "
            f"device=None to accept the model's device.")
    meta = {"num_host_targets": num_host,
            "num_subnet_targets": num_subnet,
            "trust": trust,
            "device": str(mappo.device),
            "checkpoint": checkpoint_path}
    return mappo, meta
