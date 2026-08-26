import numpy as np

from Marl.mappo.env import CC4Env
from Marl.mappo.action_mask import explain_mask


env = CC4Env()

obs, info = env.reset()

for agent in env.agents:

    mask, reasons = explain_mask(env, agent)

    labels = env.action_labels(agent)

    print("\n" + "=" * 70)
    print(f"AGENT: {agent}")
    print("=" * 70)

    print(f"Total actions: {len(mask)}")
    print(f"Enabled actions: {mask.sum()}")
    print(f"Masked actions: {(~mask).sum()}")

    print("\nENABLED ACTIONS:")
    for i, enabled in enumerate(mask):
        if enabled:
            print(f"  [{i:3}] {labels[i]}")

    print("\nAAM-SUPPRESSED ACTIONS:")
    for i, reason in reasons.items():
        print(f"  [{i:3}] {reason}")