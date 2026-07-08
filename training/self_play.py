"""
Phase 1 self-play training.

Trains a single side (ATK or DEF) using MaskablePPO with a self-play opponent.
Every `opponent_update_freq` timesteps, the current policy is snapshotted and
used as the new opponent, replacing the previous one.

Rollouts are collected from `n_envs` parallel environments (SubprocVecEnv by
default). The opponent snapshot runs on CPU inside each env: batch-1 inference
of a net this small is faster on CPU than the GPU round trip, it parallelizes
across worker processes, and a CPU policy is picklable into subprocess workers.
"""

import copy
import os
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

_PROJECT_ROOT = Path(__file__).parent.parent


def _get_action_mask(env):
    return env.action_masks()


class PolicyOpponent:
    """
    Frozen policy snapshot acting as the in-env opponent, on CPU.

    Defined at module level so VecEnv.env_method can pickle instances into
    SubprocVecEnv workers (a closure cannot be pickled through the pipe).
    """

    def __init__(self, policy, side: int, obs_mode: str):
        policy = policy.to("cpu")
        policy.set_training_mode(False)
        self.policy = policy
        self.side = side
        self.obs_mode = obs_mode
        self.n_actions = int(policy.action_space.n)

    def __call__(self, board: np.ndarray, valid_actions: list) -> int:
        from env.observations import encode_observation
        obs = encode_observation(board, self.side, self.obs_mode)
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[valid_actions] = True
        mask_t = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy(obs_t, action_masks=mask_t)
        action = int(action.item())
        if action not in valid_actions:
            action = valid_actions[np.random.randint(len(valid_actions))]
        return action


class SelfPlayCallback(BaseCallback):
    """
    Snapshots the current policy as the new opponent at regular intervals,
    installing it into every env of the training VecEnv.
    """

    def __init__(self, opponent_update_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.opponent_update_freq = opponent_update_freq
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.opponent_update_freq:
            self._snapshot_opponent()
            self._last_update = self.num_timesteps
        return True

    def _snapshot_opponent(self):
        vec_env = self.model.get_env()
        side = vec_env.get_attr("side")[0]
        obs_mode = vec_env.get_attr("obs_mode")[0]
        opponent = PolicyOpponent(copy.deepcopy(self.model.policy), side=side, obs_mode=obs_mode)
        vec_env.env_method("set_opponent", opponent)
        if self.verbose:
            print(f"[SelfPlay] Opponent updated at step {self.num_timesteps}")


def make_env(side: int, obs_mode: str, opponent_fn=None, torch_threads: Optional[int] = None):
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    from env.tafl_wrapper import TaflEnv
    env = TaflEnv(side=side, opponent_fn=opponent_fn, obs_mode=obs_mode)
    env = ActionMasker(env, _get_action_mask)
    env = Monitor(env)
    return env


def make_vec_env(side: int, obs_mode: str, n_envs: int = 8, vec_env: str = "subproc"):
    """
    Vectorize `n_envs` TaflEnvs. "subproc" steps envs (and their CPU opponents)
    in parallel across processes; "dummy" steps them serially in-process but
    still batches the agent's GPU forward across envs.
    """
    if vec_env == "subproc" and n_envs > 1:
        # One torch thread per worker: n_envs single-board inferences in
        # parallel beat n_envs processes each fighting over every core.
        fns = [partial(make_env, side=side, obs_mode=obs_mode, torch_threads=1)
               for _ in range(n_envs)]
        return SubprocVecEnv(fns)
    fns = [partial(make_env, side=side, obs_mode=obs_mode) for _ in range(n_envs)]
    return DummyVecEnv(fns)


def train(
    side: int,
    total_timesteps: int = 1_000_000,
    opponent_update_freq: int = 50_000,
    obs_mode: str = "canonical",
    features_dim: int = 256,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 512,
    n_epochs: int = 10,
    n_envs: int = 8,
    vec_env: str = "subproc",
    save_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    use_wandb: bool = False,
    verbose: int = 1,
    plot_freq: int = 10_000,
    dashboard_port: int = 8787,
    checkpoint_freq: int = 25_000,
):
    from gym_tafl.envs.configs import ATK
    from agents.networks import TaflCNN

    os.chdir(_PROJECT_ROOT)
    side_name = "atk" if side == ATK else "def"
    run_name = run_name or f"selfplay_{side_name}_{obs_mode}"
    save_dir = Path(save_dir) if save_dir else _PROJECT_ROOT / "checkpoints" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(side=side, obs_mode=obs_mode, n_envs=n_envs, vec_env=vec_env)

    policy_kwargs = {
        "features_extractor_class": TaflCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
    }

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,  # per env: buffer per update = n_steps * n_envs
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        verbose=verbose,
        tensorboard_log=str(save_dir / "tb_logs") if verbose else None,
    )

    callbacks = [SelfPlayCallback(opponent_update_freq=opponent_update_freq, verbose=verbose)]

    if checkpoint_freq and checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(checkpoint_freq // n_envs, 1),  # save_freq counts vec-env steps, not timesteps
            save_path=str(save_dir),
            name_prefix="model",
            verbose=verbose,
        ))

    if plot_freq and plot_freq > 0:
        from stable_baselines3.common.logger import configure
        from training.diagnostics import DiagnosticsCallback
        log_dir = save_dir / "logs"
        model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
        callbacks.append(DiagnosticsCallback(
            log_dir=log_dir,
            out_dir=save_dir / "diagnostics",
            plot_freq=plot_freq,
            dashboard_port=dashboard_port,
            run_info={
                "run_name": run_name,
                "side": side_name.upper(),
                "obs_mode": obs_mode,
                "total_timesteps": total_timesteps,
            },
            verbose=verbose,
        ))

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(project="hnefatafl", name=run_name, sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(save_dir / "final_model"))
    print(f"[train] Saved to {save_dir / 'final_model'}")
    env.close()
    return model


def load(path: str):
    return MaskablePPO.load(path)
