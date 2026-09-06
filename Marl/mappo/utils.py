"""
utils.py

Utility functions for MAPPO.

Currently contains:

    - Generalized Advantage Estimation (GAE)

Later this file can also contain

    - checkpoint helpers
    - observation padding
    - evaluation helpers
"""

import numpy as np

from .config import (
    GAMMA,
    GAE_LAMBDA,
)


# Generalized Advantage Estimation

def compute_gae(
    rewards,
    values,
    dones,
    last_value,
):
    """
    Compute GAE advantages and returns.

    Parameters
    ----------
    rewards : np.ndarray
        Shape:
            [T, NUM_AGENTS]

    values : np.ndarray
        Shape:
            [T, NUM_AGENTS]

    dones : np.ndarray
        Shape:
            [T]

        Episode termination flags.

    last_value : np.ndarray
        Shape:
            [NUM_AGENTS]

        Critic prediction for the state
        after the final collected timestep.

    Returns
    -------
    advantages : np.ndarray
        Shape:
            [T, NUM_AGENTS]

    returns : np.ndarray
        Shape:
            [T, NUM_AGENTS]
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    last_value = np.asarray(last_value, dtype=np.float32)

    T = rewards.shape[0]
    N = rewards.shape[1]

    advantages = np.zeros_like(rewards, dtype=np.float32)

    gae = np.zeros(N, dtype=np.float32)


    for t in reversed(range(T)):

        if t == T - 1:

            next_value = last_value

            not_done = 1.0 - dones[t]

        else:

            next_value = values[t + 1]

            not_done = 1.0 - dones[t]
        delta = (
            rewards[t]
            + GAMMA * next_value * not_done
            - values[t]
        )

        gae = (
            delta
            + GAMMA
            * GAE_LAMBDA
            * not_done
            * gae
        )

        advantages[t] = gae
    returns = advantages + values

    return advantages, returns