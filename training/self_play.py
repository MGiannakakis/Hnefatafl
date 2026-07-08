"""
Phase 1 self-play training.

Trains a single side (ATK or DEF) using MaskablePPO with a self-play opponent.
Every `opponent_update_freq` timesteps, the current policy is snapshotted and
used as the new opponent, replacing the previous one.
"""

import os
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

_PROJECT_ROOT = Path(__file__).parent.parent


def _get_action_mask(env):
    return env.action_masks()


class SelfPlayCallback(BaseCallback):
    """
    Snapshots the current policy as the new opponent at regular intervals.
    """

    def __init__(self, train_env, opponent_update_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.opponent_update_freq = opponent_update_freq
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.opponent_update_freq:
            self._snapshot_opponent()
            self._last_update = self.num_timesteps
        return True

    def _snapshot_opponent(self):
        policy = copy.deepcopy(self.model.policy)
        policy.set_training_mode(False)
        env_side = self.train_env.env.side  # unwrap Monitor
        env_obs_mode = self.train_env.env.obs_mode

        def opponent_fn(board, valid_actions):
            from env.observations import encode_observation
            obs = encode_observation(board, env_side, env_obs_mode)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.model.device)
            mask = np.zeros(self.model.action_space.n, dtype=bool)
            for a in valid_actions:
                mask[a] = True
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                action, _, _ = policy(obs_t, action_masks=mask_t)
            action = int(action.item())
            if action not in valid_actions:
                action = valid_actions[np.random.randint(len(valid_actions))]
            return action

        self.train_env.env.opponent_fn = opponent_fn
        if self.verbose:
            print(f"[SelfPlay] Opponent updated at step {self.num_timesteps}")


def make_env(side: int, obs_mode: str, opponent_fn=None):
    from env.tafl_wrapper import TaflEnv
    env = TaflEnv(side=side, opponent_fn=opponent_fn, obs_mode=obs_mode)
    env = ActionMasker(env, _get_action_mask)
    env = Monitor(env)
    return env


def train(
    side: int,
    total_timesteps: int = 1_000_000,
    opponent_update_freq: int = 50_000,
    obs_mode: str = "canonical",
    features_dim: int = 256,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    save_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    use_wandb: bool = False,
    verbose: int = 1,
):
    from gym_tafl.envs.configs import ATK
    from agents.networks import TaflCNN

    os.chdir(_PROJECT_ROOT)
    side_name = "atk" if side == ATK else "def"
    run_name = run_name or f"selfplay_{side_name}_{obs_mode}"
    save_dir = Path(save_dir) if save_dir else _PROJECT_ROOT / "checkpoints" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(side=side, obs_mode=obs_mode)

    policy_kwargs = {
        "features_extractor_class": TaflCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
    }

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        verbose=verbose,
        tensorboard_log=str(save_dir / "tb_logs") if verbose else None,
    )

    callbacks = [SelfPlayCallback(env, opponent_update_freq=opponent_update_freq, verbose=verbose)]

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(project="hnefatafl", name=run_name, sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(save_dir / "final_model"))
    print(f"[train] Saved to {save_dir / 'final_model'}")
    return model


def load(path: str):
    return MaskablePPO.load(path)
