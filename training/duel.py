"""
Phase 1b adversarial co-training ("duel" mode).

Two side-dedicated MaskablePPO networks — one permanently ATK, one
permanently DEF — train alternately in one process. While one side trains,
a frozen CPU snapshot of the other plays inside its envs (installed via the
existing PolicyOpponent machinery). Each snapshot encodes observations with
its OWN side, so every network only ever sees inputs consistent with its
training. There is no random-opponent code path: phase 0's opponent is the
untrained (or warm-started) DEF network.

Spec: docs/superpowers/specs/2026-07-08-duel-co-training-design.md
"""

import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO

_PROJECT_ROOT = Path(__file__).parent.parent


class PoolOpponent:
    """
    Latest-biased pool of frozen rival snapshots.

    Playing only the newest snapshot invites cycling: each phase the learner
    finds that one rival's blind spot, wins every game with a short exploit
    line, and forgets everything else. Sampling each episode's opponent from
    the snapshot history (latest with probability `p_latest`, else uniform
    over the older members) forces strategies that beat the whole population.

    TaflEnv.reset() calls new_episode() to re-sample; TaflEnv.
    add_opponent_snapshot() calls add_member(). Module-level class so it
    pickles into SubprocVecEnv workers.
    """

    def __init__(self, side: int, obs_mode: str, max_members: int = 20,
                 p_latest: float = 0.8):
        self.side = side
        self.obs_mode = obs_mode
        self.max_members = max_members
        self.p_latest = p_latest
        self.members: list = []
        self._active = None

    def add_member(self, snapshot) -> None:
        self.members.append(snapshot)
        if len(self.members) > self.max_members:
            self.members.pop(0)  # FIFO: bounded worker RAM beats deep history

    def new_episode(self) -> None:
        if len(self.members) == 1 or np.random.rand() < self.p_latest:
            self._active = self.members[-1]
        else:
            self._active = self.members[np.random.randint(len(self.members) - 1)]

    def __call__(self, board: np.ndarray, valid_actions: list) -> int:
        if self._active is None:
            self.new_episode()
        return self._active(board, valid_actions)


def _make_snapshot(model, side: int, obs_mode: str):
    """Freeze `model`'s current policy into a PolicyOpponent encoded with the
    snapshot's own `side`."""
    from training.self_play import PolicyOpponent
    # Right after learn(), policy.action_dist still caches the last gradient
    # step's MaskableCategorical, whose logits are non-leaf tensors deepcopy
    # refuses. The cache is transient (rebuilt on every forward) — drop it.
    dist = getattr(model.policy, "action_dist", None)
    if dist is not None and getattr(dist, "distribution", None) is not None:
        dist.distribution = None
    return PolicyOpponent(copy.deepcopy(model.policy), side=side, obs_mode=obs_mode)


def _snapshot_into(env, model, side: int, obs_mode: str) -> None:
    """Freeze `model`'s current policy and feed it to every sub-env of `env`:
    appended if the opponent is a pool, installed outright otherwise."""
    env.env_method("add_opponent_snapshot", _make_snapshot(model, side, obs_mode))


def _resolve_ckpt(label: str, ckpt: Optional[str]) -> Optional[str]:
    if not ckpt:
        return None
    if Path(ckpt).exists() or Path(f"{ckpt}.zip").exists():
        return ckpt
    raise FileNotFoundError(f"duel.{label} not found: {ckpt}")


def duel_train(
    total_timesteps_per_side: int = 1_000_000,
    steps_per_phase: int = 50_000,
    obs_mode: str = "canonical",
    features_dim: int = 256,
    learning_rate: float = 3e-4,
    n_envs: int = 8,
    vec_env: str = "subproc",
    n_steps: int = 256,
    batch_size: int = 512,
    n_epochs: int = 10,
    ent_coef: float = 0.01,
    pool_size: int = 20,
    pool_p_latest: float = 0.8,
    atk_ckpt: Optional[str] = None,
    def_ckpt: Optional[str] = None,
    run_name: Optional[str] = None,
    checkpoint_freq: int = 25_000,
    plot_freq: int = 10_000,
    dashboard_port: int = 8787,
    verbose: int = 1,
):
    from gym_tafl.envs.configs import ATK, DEF
    from agents.networks import TaflCNN
    from training.self_play import make_vec_env

    os.chdir(_PROJECT_ROOT)
    warm = {ATK: _resolve_ckpt("atk_ckpt", atk_ckpt),
            DEF: _resolve_ckpt("def_ckpt", def_ckpt)}  # fail fast before env spawn

    run_name = run_name or f"duel_{obs_mode}"
    run_dir = _PROJECT_ROOT / "checkpoints" / run_name
    side_names = {ATK: "atk", DEF: "def"}

    policy_kwargs = {
        "features_extractor_class": TaflCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
    }

    envs, models, callbacks, budgets = {}, {}, {}, {}
    for i, side in enumerate((ATK, DEF)):
        name = side_names[side]
        side_dir = run_dir / name
        side_dir.mkdir(parents=True, exist_ok=True)

        envs[side] = make_vec_env(side=side, obs_mode=obs_mode,
                                  n_envs=n_envs, vec_env=vec_env)

        if warm[side]:
            models[side] = MaskablePPO.load(
                warm[side], env=envs[side], n_steps=n_steps, batch_size=batch_size,
                ent_coef=ent_coef)
        else:
            models[side] = MaskablePPO(
                "CnnPolicy",
                envs[side],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                n_steps=n_steps,  # per env: buffer per update = n_steps * n_envs
                batch_size=batch_size,
                n_epochs=n_epochs,
                ent_coef=ent_coef,  # keep policies stochastic: harder to exploit
                gamma=0.99,
                verbose=verbose,
            )
        # budget is ADDITIONAL steps this run (warm starts keep their counter)
        budgets[side] = models[side].num_timesteps + total_timesteps_per_side
        models[side].set_logger(configure(str(side_dir / "logs"),
                                          ["stdout", "csv", "tensorboard"]))

        cbs = []
        if checkpoint_freq and checkpoint_freq > 0:
            cbs.append(CheckpointCallback(
                save_freq=max(checkpoint_freq // n_envs, 1),  # counts vec-env steps
                save_path=str(side_dir),
                name_prefix="model",
                verbose=verbose,
            ))
        if plot_freq and plot_freq > 0:
            from training.diagnostics import DiagnosticsCallback
            cbs.append(DiagnosticsCallback(
                log_dir=side_dir / "logs",
                out_dir=side_dir / "diagnostics",
                plot_freq=plot_freq,
                dashboard_port=(dashboard_port + i) if dashboard_port else 0,
                run_info={
                    "run_name": f"{run_name}/{name}",
                    "side": name.upper(),
                    "obs_mode": obs_mode,
                    "total_timesteps": budgets[side],
                },
                verbose=verbose,
            ))
        callbacks[side] = cbs

    # Each side faces a pool of the rival's snapshot history, seeded with the
    # rival's starting policy (untrained or warm-started) so phase 0 has an
    # opponent. Every _snapshot_into afterwards appends to these pools.
    # One pool INSTANCE per sub-env: with DummyVecEnv a broadcast-shared pool
    # would receive every add_opponent_snapshot n_envs times, filling itself
    # with duplicates and evicting real history early. (Sharing the seed
    # snapshot object is fine — snapshots are stateless per call.)
    for side, other in ((ATK, DEF), (DEF, ATK)):
        seed = _make_snapshot(models[other], other, obs_mode)
        for idx in range(n_envs):
            pool = PoolOpponent(side=other, obs_mode=obs_mode,
                                max_members=pool_size, p_latest=pool_p_latest)
            pool.add_member(seed)
            envs[side].env_method("set_opponent", pool, indices=[idx])

    phase = 0
    try:
        while any(models[s].num_timesteps < budgets[s] for s in (ATK, DEF)):
            for side, other in ((ATK, DEF), (DEF, ATK)):
                model = models[side]
                if model.num_timesteps >= budgets[side]:
                    continue
                target = min(model.num_timesteps + steps_per_phase, budgets[side])
                if verbose:
                    print(f"[duel] Phase {phase}: {side_names[side].upper()} "
                          f"{model.num_timesteps:,} -> {target:,} "
                          f"(vs {side_names[other].upper()} @ "
                          f"{models[other].num_timesteps:,})")
                model.learn(total_timesteps=target,
                            callback=callbacks[side],
                            reset_num_timesteps=False)
                _snapshot_into(envs[other], model, side, obs_mode)
                phase += 1
    finally:
        for side in (ATK, DEF):
            models[side].save(str(run_dir / side_names[side] / "final_model"))
            envs[side].close()
        print(f"[duel] Saved both sides under {run_dir}")

    return {side_names[s]: models[s] for s in (ATK, DEF)}
