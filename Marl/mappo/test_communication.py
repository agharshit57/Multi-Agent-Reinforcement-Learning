"""
manual_input_receiver_prediction_test.py

MANUAL INPUT -> MODEL PREDICTION communication test.

You choose the sender, the receivers, and all seven StructuredMessage
fields by hand. Nothing about the message content is predicted or
decoded -- it is exactly what you typed. What IS a genuine model
output is what happens after that: the message is encoded with the
project's real trained MessageEncoder, and each receiver's ACTUAL
trained SharedActor is run forward on it, with the frozen trust from
your checkpoint applied exactly where the trained architecture applies
it.

Traced receiver-side path (read this before trusting the output)
------------------------------------------------------------------
There is NO component anywhere in this codebase that takes a received
message and decodes/classifies it back into a semantic judgement, e.g.
"receiver believes this means COMPROMISE at Host_X". MessageEvaluator
(communication/evaluator.py) sounds like it might do that, but it does
NOT run through any trained weights -- it is a training-time, rule-
based Python comparator that grades a message against environment
ground truth purely to drive the trust UPDATE. It has nothing to do
with the receiver's neural pathway. Using it here would be exactly the
invented rule-based prediction this file was asked not to build, so
this file never imports or calls it.

The actual, real, trained receiver-side path -- traced directly from
gnn_attention.py's SharedActor -- is:

    SharedActor.forward(observation, received_messages, trust_weights)
        -> local_hidden = self._get_local_hidden(observation)   [receiver's OWN observation]
        -> self._apply_received_communication(...):
               keys   = self.communication_key(messages)
               values = self.communication_value(messages)
               values = values * trust.clamp(1e-4, 1.0)         <-- frozen trust scales here
               query  = self.communication_query(local_hidden)
               context, communication_weights = self.communication_attention(
                   query, keys, values, need_weights=True
               )
               self.last_communication_attention = communication_weights.detach()  <-- inspectable
        -> policy_representation = concat([local_hidden, context])
        -> logits = self.policy_head(policy_representation)     <-- ACTION logits, nothing else

The trained model only ever uses a received message to modulate the
receiver's ACTION policy, through exactly one trust-scaled cross-
attention layer. There is no other receiver-side output to read. So
"model-derived result" in this file means, honestly and exactly:

    1. The attention weight the receiver's OWN trained attention head
       assigns to this exact message (read from
       actor.last_communication_attention after a real forward pass).
       Trust is already baked into this number, since trust scales the
       attention VALUES before the weights are computed.

    2. The measurable shift in the receiver's action distribution
       caused by this message, versus a baseline forward pass with no
       message at all -- computed via ppo.actor_forward(), the exact
       function evaluate.py's select_action_greedy() calls. Reported
       as: whether the greedy action changes, the top-k action
       probabilities with the message present, and the KL divergence
       between the with-message and without-message policies.

Nothing here decodes the message back into StructuredMessage fields,
and nothing here is a hand-written scoring rule. If what you actually
want is a literal "receiver interprets this as X" output, that would
require a NEW, trained, receiver-side auxiliary head added to the
model (e.g. one trained to predict environment ground truth from the
post-attention receiver hidden state) -- that component does not exist
in this codebase today, and this file does not fake one.

Flow printed by this script:

    MANUAL INPUT
        -> ENCODED 128-D MESSAGE      (real MessageEncoder, no copy)
        -> RECEIVER MODEL PREDICTION  (real SharedActor forward: attention weight + action-distribution shift)
        -> FROZEN TRUST FOR EACH RECEIVER

IMPORTANT:
    Does not modify schema.py, encoder.py, decoder.py,
    structured_communication.py, trust.py, gnn_attention.py, mappo.py,
    env.py, train.py, action_mask.py, or evaluate.py -- it only
    imports pad_observation()/episode_is_done() from train.py and
    compute_padded_mask() from action_mask.py, which are the project's
    own, already-used entry points for exactly this purpose. Trust is
    loaded from the checkpoint and never updated; no PPO update is
    performed anywhere in this file.

Run from the project root:

    python -m Marl.mappo.manual_input_receiver_prediction_test
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from .env import CC4Env
from .mappo import MAPPO
from .train import pad_observation, episode_is_done
from .action_mask import compute_padded_mask
from .config import NUM_AGENTS, OBS_DIM, ACTION_DIM
from .gnn_attention import MESSAGE_DIM
from .communication.schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
    confidence_level_to_value,
)


# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_PATH = (
    "/mnt/c/cyber/cage-challenge-4/"
    "checkpoints/gnn_attention_AAM_newMappo/mappo_final.pt"
)

# How many steps to advance the environment (random Blue actions --
# only receiver OBSERVATIONS are needed from this, not any particular
# policy behavior) before reading the real observations used as each
# receiver's "own state" during the forward pass below.
WARMUP_STEPS = 5

TOP_K_ACTIONS = 5


# ============================================================================
# Checkpoint helpers
# ============================================================================

def get_checkpoint_num_targets(checkpoint_path: str):
    """
    Read the communication target vocabulary sizes directly from the
    checkpoint.

    HOST and SUBNET are separate, independently-sized vocabularies
    (see mappo.py's module docstring) -- there is no longer a single
    combined "num_targets". MAPPO.save() persists both directly on the
    checkpoint dict as "num_host_targets" / "num_subnet_targets", so
    this reads those keys rather than inferring a shape from a single
    target_head weight, which no longer exists as one tensor (the
    decoder now has separate host_target_head/subnet_target_head).

    Returns
    -------
    (int, int or None)
        (num_host_targets, num_subnet_targets).
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. Expected a dictionary, "
            f"got {type(checkpoint).__name__}."
        )

    if "model" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain a 'model' state_dict.")

    num_host_targets = checkpoint.get("num_host_targets")
    num_subnet_targets = checkpoint.get("num_subnet_targets")

    if num_host_targets is None:
        raise RuntimeError(
            "Checkpoint does not contain 'num_host_targets'. This "
            "checkpoint predates the host/subnet target split and "
            "cannot be loaded by this script."
        )

    if num_host_targets <= 0:
        raise RuntimeError(
            f"Invalid checkpoint num_host_targets: {num_host_targets}"
        )

    if num_subnet_targets is not None and num_subnet_targets <= 0:
        raise RuntimeError(
            f"Invalid checkpoint num_subnet_targets: {num_subnet_targets}"
        )

    return num_host_targets, num_subnet_targets


def load_trained_mappo(checkpoint_path: str):
    """
    Construct the same MAPPO architecture used by the checkpoint and
    load it. Trust state is restored, then left completely frozen --
    this file never calls ppo.update_trust() or performs a PPO update.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    num_host_targets, num_subnet_targets = get_checkpoint_num_targets(
        checkpoint_path
    )

    print()
    print("=" * 72)
    print("LOADING TRAINED MODEL")
    print("=" * 72)
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Host targets  : {num_host_targets}")
    print(f"Subnet targets: {num_subnet_targets}")

    ppo = MAPPO(
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )

    checkpoint = torch.load(checkpoint_path, map_location=ppo.device)

    model_state = checkpoint["model"]

    # Obsolete GNN buffers from the checkpoint version this model was
    # trained with -- current gnn_attention.py builds its own
    # structural_adjacency / real_topology buffers instead.
    obsolete_keys = {"actor.gnn1.adjacency", "actor.gnn2.adjacency"}

    model_state = {
        key: value
        for key, value in model_state.items()
        if key not in obsolete_keys
    }

    missing, unexpected = ppo.model.load_state_dict(model_state, strict=False)

    expected_missing = {
        "actor.structural_adjacency",
        "actor.real_topology",
    }

    unexpected_set = set(unexpected)
    missing_set = set(missing)

    if unexpected_set:
        raise RuntimeError(
            "Unexpected checkpoint keys after compatibility filtering:\n"
            + "\n".join(sorted(unexpected_set))
        )

    unexpected_missing = missing_set - expected_missing

    if unexpected_missing:
        raise RuntimeError(
            "Unexpected missing model keys:\n"
            + "\n".join(sorted(unexpected_missing))
        )

    trust_state = checkpoint.get("trust_state")

    if trust_state is not None and hasattr(
        ppo.actor.communication, "load_trust_state"
    ):
        ppo.actor.communication.load_trust_state(trust_state)

    ppo.eval()

    print("Model      : loaded successfully")
    print("Trust      : loaded from checkpoint")
    print("Trust      : frozen for this test")
    print("=" * 72)

    return ppo, num_host_targets, num_subnet_targets


# ============================================================================
# Environment helpers (real receiver observations)
# ============================================================================

def build_observation_batch(agent_names, obs_dims, obs_dict) -> np.ndarray:
    """Pad every agent's raw CC4 observation, exactly as train.py does."""

    obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

    for i, name in enumerate(agent_names):
        obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

    return obs_array


def advance_environment(env, warmup_steps: int):
    """
    Reset, then step forward with random Blue actions so receiver
    observations aren't a trivial freshly-reset state. Only the
    COMMUNICATION path is under test here, so the specific actions
    taken don't matter.
    """

    obs_dict, info = env.reset()

    for _ in range(warmup_steps):

        actions = env.sample_actions()
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        if episode_is_done(terminated, truncated):
            obs_dict, info = env.reset()

    return obs_dict


# ============================================================================
# Input helpers
# ============================================================================

def choose_from_enum(enum_cls, title: str):
    """Interactively choose one IntEnum member."""

    members = list(enum_cls)

    print()
    print(title)

    for i, member in enumerate(members):
        print(f"  {i}. {member.name}")

    while True:
        raw = input("Select number: ").strip()

        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= index < len(members):
            return members[index]

        print(f"Choose a number from 0 to {len(members) - 1}.")


def choose_agent(num_agents: int, title: str) -> int:
    """Interactively choose one agent ID."""

    print()
    print(title)

    for agent_id in range(num_agents):
        print(f"  {agent_id}. Agent_{agent_id + 1}")

    while True:
        raw = input("Select number: ").strip()

        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= index < num_agents:
            return index

        print(f"Choose a number from 0 to {num_agents - 1}.")


def choose_receivers(sender: int, num_agents: int) -> List[int]:
    """Select receiving agents."""

    available = [a for a in range(num_agents) if a != sender]

    print()
    print("Receivers")
    print("  A. All other agents")

    for agent_id in available:
        print(f"  {agent_id + 1}. Agent_{agent_id + 1}")

    while True:
        raw = input(
            "Select 'A' for all, or comma-separated agent numbers: "
        ).strip().lower()

        if raw == "a":
            return available

        try:
            selected = [int(item.strip()) - 1 for item in raw.split(",")]
        except ValueError:
            print("Invalid selection.")
            continue

        if not selected:
            print("Select at least one receiver.")
            continue

        if len(set(selected)) != len(selected):
            print("Do not select the same receiver twice.")
            continue

        invalid = [a for a in selected if a not in available]

        if invalid:
            print(
                "Invalid receiver(s): "
                + ", ".join(f"Agent_{a + 1}" for a in invalid)
            )
            continue

        return selected


def choose_target_id(num_targets: int) -> int:
    """
    Select a target ID within the given vocabulary. HOST and SUBNET
    are separate, independently-sized vocabularies (see schema.py) --
    the caller picks which count to pass based on target_type.
    """

    print()
    print("Target ID")
    print(f"  Valid range: 0 - {num_targets - 1}")

    while True:
        raw = input("Enter target ID: ").strip()

        try:
            target_id = int(raw)
        except ValueError:
            print("Target ID must be an integer.")
            continue

        if 0 <= target_id < num_targets:
            return target_id

        print(f"Target ID must be between 0 and {num_targets - 1}.")


def build_manual_message(
    num_host_targets: int,
    num_subnet_targets: int,
) -> StructuredMessage:
    """Ask the user for all seven schema fields -- this IS the input."""

    print()
    print("=" * 72)
    print("MANUAL INPUT")
    print("=" * 72)

    event_type = choose_from_enum(EventType, "Event Type")
    target_type = choose_from_enum(TargetType, "Target Type")

    if target_type == TargetType.NONE:
        target_id = 0
    elif target_type == TargetType.SUBNET:
        if num_subnet_targets is None:
            raise RuntimeError(
                "This checkpoint was not trained with a SUBNET "
                "target vocabulary (num_subnet_targets is None), "
                "so a SUBNET target cannot be selected."
            )
        target_id = choose_target_id(num_subnet_targets)
    else:
        target_id = choose_target_id(num_host_targets)

    threat_level = choose_from_enum(ThreatLevel, "Threat Level")
    confidence_level = choose_from_enum(ConfidenceLevel, "Confidence")
    status = choose_from_enum(HostStatus, "Status")
    priority = choose_from_enum(Priority, "Priority")

    confidence = confidence_level_to_value(confidence_level)

    return StructuredMessage(
        event_type=event_type,
        target_type=target_type,
        target_id=target_id,
        threat_level=threat_level,
        confidence=confidence,
        status=status,
        priority=priority,
    )


# ============================================================================
# Real encoder integration (no reconstructed copy)
# ============================================================================

def message_to_field_ids(
        message: StructuredMessage,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Convert the manually chosen StructuredMessage into batch-of-1 ids."""

        confidence_id = min(
            range(len(ConfidenceLevel)),
            key=lambda level: abs(
                confidence_level_to_value(level)
                - message.confidence
            ),
        )

        return {
            "event_type": torch.tensor(
                [int(message.event_type)],
                dtype=torch.long,
                device=device,
            ),
            "target_type": torch.tensor(
                [int(message.target_type)],
                dtype=torch.long,
                device=device,
            ),
            "target_id": torch.tensor(
                [int(message.target_id)],
                dtype=torch.long,
                device=device,
            ),
            "threat_level": torch.tensor(
                [int(message.threat_level)],
                dtype=torch.long,
                device=device,
            ),
            "confidence": torch.tensor(
                [int(confidence_id)],
                dtype=torch.long,
                device=device,
            ),
            "status": torch.tensor(
                [int(message.status)],
                dtype=torch.long,
                device=device,
            ),
            "priority": torch.tensor(
                [int(message.priority)],
                dtype=torch.long,
                device=device,
            ),
        }


def encode_manual_message(
        ppo: MAPPO,
        message: StructuredMessage,
    ) -> torch.Tensor:
        """
        Encode using the project's REAL trained MessageEncoder.

        The manually created field IDs are placed on the same device
        as the trained encoder parameters.
        """

        encoder = ppo.actor.communication.encoder

        device = next(
            encoder.parameters()
        ).device

        field_ids = message_to_field_ids(
            message,
            device,
        )

        with torch.no_grad():
            vector = encoder.encode_from_ids(
                field_ids
            )

        return vector.squeeze(0)

# ============================================================================
# Real receiver-side forward pass
# ============================================================================

def run_receiver_model(
    ppo: MAPPO,
    env: CC4Env,
    agent_names,
    obs_array: np.ndarray,
    sender: int,
    receiver: int,
    encoded_vector: torch.Tensor,
):
    """
    Run the receiver's ACTUAL trained SharedActor forward pass, once
    with the manually-built message present and once without, using
    the exact function evaluate.py's select_action_greedy() calls
    (ppo.actor_forward -> self.actor(...) -> real policy_head logits).

    Returns
    -------
    dict with:
        trust_score               frozen trust(sender -> receiver)
        attention_weight          receiver's trained attention weight
                                   on sender's message slot (trust
                                   already baked in -- see module
                                   docstring)
        baseline_action           greedy action with NO message
        message_action             greedy action WITH the message
        action_changed             bool
        top_actions                list of (label, probability) with
                                    the message present
        kl_divergence              KL(with_message || without_message)
    """

    device = next(
        ppo.actor.parameters()
    ).device

    receiver_obs = torch.as_tensor(
        obs_array[receiver],
        dtype=torch.float32,
        device=device,
    )

    # mask = compute_padded_mask(env, agent_names[receiver])
    # mask_tensor = torch.as_tensor(mask, dtype=torch.bool)

    mask = compute_padded_mask(
        env,
        agent_names[receiver],
    )

    mask_tensor = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=device,
    )

    trust_row = ppo.get_trust_for_agent(receiver_id=receiver)  # [NUM_AGENTS], frozen
    trust_score = float(trust_row[sender].item())

    messages_matrix = torch.zeros(
        NUM_AGENTS,
        encoded_vector.numel(),
        dtype=encoded_vector.dtype,
        device=encoded_vector.device,
    )

    messages_matrix[sender] = encoded_vector

    with torch.no_grad():

        baseline_logits = ppo.actor_forward(
            observations=receiver_obs,
            action_masks=mask_tensor,
            received_messages=None,
            trust_weights=None,
        )

        message_logits = ppo.actor_forward(
            observations=receiver_obs,
            action_masks=mask_tensor,
            received_messages=messages_matrix,
            trust_weights=trust_row,
        )

    # actor.last_communication_attention is set as a side effect of the
    # attention branch inside _apply_received_communication() -- only
    # populated on the WITH-message call above, since the baseline call
    # (received_messages=None) skips that branch entirely.
    attention = ppo.actor.last_communication_attention

    # Shape is [batch=1, target_len=1, source_len=NUM_AGENTS] with
    # need_weights=True's default head-averaging.
    attention_weight = float(attention[0, 0, sender].item())

    baseline_dist = Categorical(logits=baseline_logits)
    message_dist = Categorical(logits=message_logits)

    baseline_action = int(torch.argmax(baseline_dist.probs).item())
    message_action = int(torch.argmax(message_dist.probs).item())

    top_probs, top_indices = torch.topk(
        message_dist.probs, k=min(TOP_K_ACTIONS, message_dist.probs.shape[-1])
    )

    labels = env.action_labels(agent_names[receiver])

    top_actions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = labels[idx] if idx < len(labels) else f"<pad_index_{idx}>"
        top_actions.append((label, prob))

    divergence = float(
        kl_divergence(message_dist, baseline_dist).item()
    )

    return {
        "trust_score": trust_score,
        "attention_weight": attention_weight,
        "baseline_action": baseline_action,
        "message_action": message_action,
        "action_changed": baseline_action != message_action,
        "top_actions": top_actions,
        "kl_divergence": divergence,
    }


# ============================================================================
# Display
# ============================================================================

def format_target(message: StructuredMessage) -> str:

    if message.target_type == TargetType.HOST:
        return f"Host_id={message.target_id}"

    if message.target_type == TargetType.SUBNET:
        return f"Subnet_id={message.target_id} (no subnet vocabulary exists -- see schema.py)"

    return "None"


def print_manual_input(sender: int, receivers: List[int], message: StructuredMessage) -> None:

    print()
    print("=" * 72)
    print("MANUAL INPUT")
    print("=" * 72)
    print(f"Sender:       Agent_{sender + 1}")
    print(
        "Receivers:    "
        + ", ".join(f"Agent_{r + 1}" for r in receivers)
    )
    print(f"Event:        {message.event_type.name}")
    print(f"Target Type:  {message.target_type.name}")
    print(f"Target:       {format_target(message)}")
    print(f"Threat Level: {message.threat_level.name}")
    print(f"Confidence:   {message.confidence:.3f}")
    print(f"Status:       {message.status.name}")
    print(f"Priority:     {message.priority.name}")


def print_encoded_vector(vector: torch.Tensor) -> None:

    print()
    print("=" * 72)
    print("ENCODED 128-D MESSAGE")
    print("=" * 72)
    print(f"Shape: {tuple(vector.shape)}")
    print(f"Norm:  {float(vector.norm()):.4f}")
    print(f"First 8 dims: {[round(v, 4) for v in vector[:8].tolist()]}")


def print_receiver_prediction(sender: int, receiver: int, result: dict) -> None:

    print()
    print(f"Agent_{receiver + 1} RECEIVER MODEL PREDICTION (from Agent_{sender + 1})")
    print("-" * 60)
    print(
        f"Frozen trust(sender->receiver):        "
        f"{result['trust_score']:.3f}"
    )
    print(
        f"Trained attention weight on this msg:  "
        f"{result['attention_weight']:.4f}  "
        f"(trust already scales this)"
    )
    print(
        f"Greedy action WITHOUT this message:    "
        f"{result['baseline_action']}"
    )
    print(
        f"Greedy action WITH this message:       "
        f"{result['message_action']}"
    )
    print(
        f"Action changed by this message:        "
        f"{result['action_changed']}"
    )
    print(
        f"KL(with_message || without_message):   "
        f"{result['kl_divergence']:.5f}"
    )
    print("Top action probabilities WITH this message:")
    for label, prob in result["top_actions"]:
        print(f"    {prob:.4f}  {label}")


# ============================================================================
# Main test
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print("CYBER MARL - MANUAL INPUT -> MODEL PREDICTION TEST")
    print("=" * 72)
    print("Message fields are YOUR input. Everything after encoding is")
    print("computed by the real trained model -- see this file's module")
    print("docstring for the exact traced path and its limits.")
    print("=" * 72)

    ppo, num_host_targets, num_subnet_targets = load_trained_mappo(
        CHECKPOINT_PATH
    )

    print()
    print(f"Agents           : {NUM_AGENTS}")
    print(f"Host target IDs  : 0 - {num_host_targets - 1}")
    print(
        "Subnet target IDs: "
        + (
            f"0 - {num_subnet_targets - 1}"
            if num_subnet_targets is not None
            else "not available in this checkpoint"
        )
    )

    # ------------------------------------------------------------------
    # Real receiver observations
    # ------------------------------------------------------------------

    env = CC4Env()
    agent_names = sorted(env.possible_agents)

    assert len(agent_names) == NUM_AGENTS, (
        f"Expected {NUM_AGENTS} blue agents, found {len(agent_names)}: "
        f"{agent_names}"
    )

    obs_dims = env.get_observation_dims()

    print()
    print(
        f"Advancing the environment {WARMUP_STEPS} steps so receiver "
        f"observations aren't a trivial freshly-reset state..."
    )

    obs_dict = advance_environment(env, WARMUP_STEPS)
    obs_array = build_observation_batch(agent_names, obs_dims, obs_dict)

    # ------------------------------------------------------------------
    # Manual input
    # ------------------------------------------------------------------

    sender = choose_agent(NUM_AGENTS, "Sender")
    receivers = choose_receivers(sender, NUM_AGENTS)
    message = build_manual_message(num_host_targets, num_subnet_targets)

    print_manual_input(sender, receivers, message)

    # ------------------------------------------------------------------
    # Encode with the real trained encoder
    # ------------------------------------------------------------------

    encoded_vector = encode_manual_message(ppo, message)
    print_encoded_vector(encoded_vector)

    # ------------------------------------------------------------------
    # Real receiver-side model forward, per receiver
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("RECEIVER MODEL PREDICTION")
    print("=" * 72)

    results = {}

    for receiver in receivers:

        result = run_receiver_model(
            ppo=ppo,
            env=env,
            agent_names=agent_names,
            obs_array=obs_array,
            sender=sender,
            receiver=receiver,
            encoded_vector=encoded_vector,
        )

        results[receiver] = result

        print_receiver_prediction(sender, receiver, result)

    # ------------------------------------------------------------------
    # Frozen trust summary
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FROZEN TRUST FOR EACH RECEIVER")
    print("=" * 72)

    for receiver in receivers:

        trust_score = results[receiver]["trust_score"]

        if trust_score <= 0.25:
            level = "VERY LOW"
        elif trust_score <= 0.50:
            level = "LOW"
        elif trust_score <= 0.75:
            level = "HIGH"
        else:
            level = "VERY HIGH"

        print(
            f"Agent_{receiver + 1} trusts Agent_{sender + 1}: "
            f"{trust_score:.3f} ({level})"
        )

    print()
    print("=" * 72)
    print("TEST COMPLETE")
    print("=" * 72)
    print()
    print(
        "Note: trust was loaded from the checkpoint and never updated. "
        "No PPO update was performed. The message content above is "
        "exactly what you typed; everything under RECEIVER MODEL "
        "PREDICTION was computed by a real forward pass through the "
        "checkpoint's trained SharedActor."
    )


if __name__ == "__main__":
    main()