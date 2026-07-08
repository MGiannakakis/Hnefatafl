"""
Evaluation utilities: win-rate measurement, Elo tracking, adaptation curves.
"""

import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent


def win_rate(
    policy_fn: Callable,
    side: int,
    opponent_fn: Callable,
    n_episodes: int = 200,
    obs_mode: str = "canonical",
) -> dict:
    """
    Measure win/draw/loss rate for `policy_fn` playing as `side` against `opponent_fn`.

    policy_fn: (obs: np.ndarray, action_mask: np.ndarray) -> int
    opponent_fn: (board: np.ndarray, valid_actions: list) -> int
    """
    from env.tafl_wrapper import TaflEnv

    os.chdir(_PROJECT_ROOT)
    env = TaflEnv(side=side, opponent_fn=opponent_fn, obs_mode=obs_mode)

    wins = draws = losses = 0
    episode_lengths = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy_fn(obs, info.get("action_mask", None))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        episode_lengths.append(steps)
        winner = info.get("winner")
        from gym_tafl.envs.configs import DRAW
        if winner == side:
            wins += 1
        elif winner == DRAW or winner == -1:
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "n_episodes": total,
    }


def model_policy_fn(model):
    """Wrap a MaskablePPO model as a policy_fn for win_rate()."""
    model.policy.set_training_mode(False)

    def fn(obs, action_mask):
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)

    return fn


def random_policy_fn(obs, action_mask):
    valid = np.where(action_mask)[0]
    return int(np.random.choice(valid))


class EloTracker:
    """Simple Elo rating tracker for comparing agents over time."""

    def __init__(self, k: float = 32.0, initial_rating: float = 1200.0):
        self.k = k
        self.ratings: dict[str, float] = {}
        self._default = initial_rating

    def get(self, name: str) -> float:
        return self.ratings.get(name, self._default)

    def update(self, winner: str, loser: str):
        ra = self.get(winner)
        rb = self.get(loser)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - (1 - ea))

    def __repr__(self):
        ranked = sorted(self.ratings.items(), key=lambda x: -x[1])
        return "\n".join(f"  {n}: {r:.1f}" for n, r in ranked)
