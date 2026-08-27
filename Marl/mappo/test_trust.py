"""
show_trust_matrix.py

Load a trained MAPPO checkpoint and display its complete trust matrix.

Does NOT:
- create MAPPO
- run the environment
- update trust
- perform PPO
- modify any existing project files

It only reads trust_state from the checkpoint and displays it.
"""

from __future__ import annotations 

import torch


CHECKPOINT_PATH = (
    "/mnt/c/cyber/cage-challenge-4/"
    "checkpoints/groundTruth/mappo_final.pt"
)


def main() -> None:

    print()
    print("=" * 80)
    print("CYBER MARL - CHECKPOINT TRUST MATRIX")
    print("=" * 80)

    print()
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    if "trust_state" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'trust_state'."
        )

    trust_state = checkpoint["trust_state"]

    print()
    print(
        f"trust_state type: {type(trust_state).__name__}"
    )

    print()
    print("=" * 80)
    print("RAW TRUST STATE")
    print("=" * 80)

    if isinstance(trust_state, dict):

        for key, value in trust_state.items():

            print()
            print(f"{key}:")

            if torch.is_tensor(value):
                print(
                    f"  shape: {tuple(value.shape)}"
                )
                print(value)

            else:
                print(f"  {value}")

    elif torch.is_tensor(trust_state):

        print(
            f"shape: {tuple(trust_state.shape)}"
        )
        print(trust_state)

    else:

        print(trust_state)

    # ---------------------------------------------------------------
    # Try to identify the actual trust matrix
    # ---------------------------------------------------------------

    trust_matrix = None

    if torch.is_tensor(trust_state):

        if trust_state.ndim == 2:
            trust_matrix = trust_state

    elif isinstance(trust_state, dict):

        for key, value in trust_state.items():

            if not torch.is_tensor(value):
                continue

            if value.ndim == 2:
                trust_matrix = value
                break

    if trust_matrix is None:

        print()
        print("=" * 80)
        print("NO 2-D TRUST MATRIX FOUND")
        print("=" * 80)

        return

    trust_matrix = trust_matrix.detach().cpu()

    num_agents = trust_matrix.shape[0]

    print()
    print("=" * 80)
    print("TRUST MATRIX")
    print("=" * 80)

    print()
    print(
        "Rows    = receiving/trusting agent"
    )
    print(
        "Columns = sender/trusted agent"
    )

    print()

    # Header
    print(
        f"{'Receiver':<12}",
        end="",
    )

    for sender in range(num_agents):

        print(
            f"{'Agent_' + str(sender + 1):>12}",
            end="",
        )

    print()

    print("-" * (12 + 12 * num_agents))

    # Matrix
    for receiver in range(num_agents):

        print(
            f"{'Agent_' + str(receiver + 1):<12}",
            end="",
        )

        for sender in range(num_agents):

            value = float(
                trust_matrix[
                    receiver,
                    sender,
                ].item()
            )

            print(
                f"{value:>12.4f}",
                end="",
            )

        print()

    print()
    print("=" * 80)
    print("AGENT-TO-AGENT TRUST")
    print("=" * 80)

    for receiver in range(num_agents):

        for sender in range(num_agents):

            if receiver == sender:
                continue

            value = float(
                trust_matrix[
                    receiver,
                    sender,
                ].item()
            )

            print(
                f"Agent_{receiver + 1} "
                f"trusts Agent_{sender + 1}: "
                f"{value:.4f}"
            )

    print()
    print("=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()