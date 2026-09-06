"""Policy engines: trained MAPPO or mock, behind one interface.

Cycle semantics mirror training/evaluation rollouts (one-step
communication delay, frozen trust):
  1. select greedy actions from current obs + previous messages/trust
  2. generate outgoing structured messages for the NEXT cycle

``TrainedPolicyEngine`` wraps the real ``MAPPO`` object (lazy import,
training venv required). ``MockPolicyEngine`` implements the same
``decide()`` contract with transparent safe defaults so the pipeline,
GUI, and tests run without weights. Both return per-agent records::

    {"action": int, "probs_top": [(label, p), ...],
     "message": {7 structured fields}, "trust_row": [...]}

Message decoding reuses the real schema.attrs when available; the mock
path emits schema-shaped dicts directly (documented in config).
"""

import numpy as np

from .config import MESSAGE_FIELD_NAMES, NUM_AGENTS


class MockPolicyEngine:
    """Weights-free stand-in: Monitor by default, Sleep if masked out."""

    name = "mock-policy"
    needs_weights = False

    def __init__(self, tables):
        self.tables = tables
        self._prev_messages = None

    def _find(self, table, command):
        for entry in table:
            if entry["command"] == command:
                return entry["index"]
        return 0

    def decide(self, obs_batch, masks, host_masks=None,
               host_valid=None):
        decisions = []
        for agent in range(NUM_AGENTS):
            table = self.tables[agent]
            mask = masks[agent]
            prefer = self._find(table, "Monitor")
            allowed = np.flatnonzero(mask)
            if mask[prefer]:
                action = prefer
            elif len(allowed):
                action = int(allowed[0])
            else:
                # Unreachable via compute_mask (Sleep safety net), but a
                # hand-fed all-False mask must not raise IndexError
                # mid-cycle: fall back to Sleep, or index 0 if the
                # table has no Sleep entry at all.
                action = self._find(table, "Sleep")
            message = {name: (0 if name != "confidence" else 0.0)
                       for name in MESSAGE_FIELD_NAMES}
            decisions.append({"action": int(action),
                              "probs_top": [(table[int(action)]["label"],
                                             1.0)],
                              "message": message,
                              "trust_row": [0.5] * NUM_AGENTS})
        return decisions

    def reset(self):
        self._prev_messages = None


class TrainedPolicyEngine:
    """Frozen trained MAPPO policy (greedy, evaluation semantics)."""

    name = "trained-mappo"
    needs_weights = True

    def __init__(self, mappo, tables, host_valid_width):
        from .config import NUM_HOST_TARGETS
        self.mappo = mappo
        self.tables = tables
        self.host_valid_width = host_valid_width
        # Deployment contract: the 137-token vocabulary the tables,
        # masks, and ground-truth mapping were built against. Fail
        # here -- not mid-episode -- on any mismatch.
        if getattr(mappo, "num_host_targets", None) != NUM_HOST_TARGETS:
            raise ValueError(
                f"checkpoint vocabulary {getattr(mappo, 'num_host_targets', '?')} "  # noqa: E501
                f"!= deployment contract {NUM_HOST_TARGETS}; refusing to "
                f"serve mismatched weights")
        if host_valid_width != NUM_HOST_TARGETS:
            raise ValueError(
                f"host_valid_width={host_valid_width} != deployment "
                f"contract {NUM_HOST_TARGETS}")
        # Evaluation semantics: frozen weights, no dropout/batch updates.
        # model_loader.load_trained_mappo already calls eval(); repeat
        # defensively so a hand-built engine cannot serve in train mode.
        if hasattr(mappo, "eval"):
            mappo.eval()
        self._prev_messages = None

    def decide(self, obs_batch, masks, host_masks=None,
               host_valid=None):
        import torch

        # The ENTIRE inference path runs under no_grad (mirrors
        # Marl/mappo/evaluate.py): production must never build autograd
        # graphs -- that would leak memory every cycle and slow serving.
        with torch.no_grad():
            return self._decide_inner(obs_batch, masks, host_masks,
                                      host_valid)

    def _decide_inner(self, obs_batch, masks, host_masks, host_valid):
        import torch

        mappo = self.mappo
        device = mappo.device
        actions = []
        for agent in range(NUM_AGENTS):
            obs = torch.as_tensor(obs_batch[agent], dtype=torch.float32,
                                  device=device)
            mask_t = torch.as_tensor(masks[agent], dtype=torch.bool,
                                     device=device)
            if self._prev_messages is None:
                received, trust = None, None
            else:
                received = mappo.get_messages_for_agent(
                    receiver_id=agent, messages=self._prev_messages)
                trust = mappo.get_trust_for_agent(receiver_id=agent)
            logits = mappo.actor_forward(
                observations=obs, action_masks=mask_t,
                received_messages=received, trust_weights=trust,
                host_active_mask=_prep(mappo, host_masks, agent),
            )
            if bool((~torch.isfinite(logits)).any()):
                raise RuntimeError(
                    f"agent {agent}: non-finite policy logits -- refusing "
                    f"to act on a corrupt forward pass")
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(5, probs.shape[-1]))
            action = int(torch.argmax(probs).item())
            if not (0 <= action < len(self.tables[agent])):
                raise RuntimeError(
                    f"agent {agent}: policy returned out-of-range action "
                    f"{action} -- refusing to translate")
            labels = [self.tables[agent][i]["label"]
                      if i < len(self.tables[agent])
                      else f"<pad {i}>" for i in top.indices.tolist()]
            actions.append((action, list(zip(labels,
                                             top.values.tolist())),
                            received, trust))
        # Outgoing messages for the NEXT cycle (same observations the
        # actions used -- mirrors train.py / evaluate.py timing).
        # NOTE: routed through a compatibility shim (older checkpoints
        # predate the host_valid_mask kwarg).
        obs_t = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32),
                                device=device)
        host_valid_t = None
        if host_valid is not None:
            host_valid_t = torch.as_tensor(np.asarray(host_valid,
                                                      dtype=bool),
                                           device=device)
        outgoing, decoded = self._outgoing(obs_t, host_masks, host_valid_t)
        self._prev_messages = outgoing.detach()
        decisions = []
        for agent in range(NUM_AGENTS):
            action, top, _r, trust = actions[agent]
            msg = decoded[agent].as_dict() if hasattr(
                decoded[agent], "as_dict") else dict(decoded[agent])
            trow = trust.detach().cpu().tolist() if trust is not None \
                else [0.5] * NUM_AGENTS
            decisions.append({"action": action, "probs_top": top,
                              "message": msg, "trust_row": trow})
        return decisions

    def _outgoing(self, obs_t, host_masks, host_valid_t):
        import inspect

        mappo = self.mappo
        kwargs = {"return_decoded": True, "host_active_mask": host_masks}
        params = inspect.signature(
            mappo.get_outgoing_messages).parameters
        if "host_valid_mask" in params and host_valid_t is not None:
            kwargs["host_valid_mask"] = host_valid_t
        with _no_grad():
            return mappo.get_outgoing_messages(obs_t, **kwargs)[:2]

    def reset(self):
        self._prev_messages = None


class _no_grad:
    def __enter__(self):
        import torch
        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def _prep(mappo, host_masks, agent):
    # decide() always forwards a single observation, so batch width 1.
    if host_masks is None:
        return None
    return mappo._prepare_host_active_mask(host_masks[agent], 1)


def describe_engine(engine):
    return {"name": engine.name,
            "needs_weights": engine.needs_weights}
