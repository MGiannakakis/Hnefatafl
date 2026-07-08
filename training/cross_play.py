"""
Phase 2 cross-perspective experiments.

Takes a checkpoint trained as one side (e.g. ATK) and fine-tunes it to play
the opposing side (DEF), measuring how quickly the transferred policy adapts.

Usage:
    python -m training.cross_play \
        --source_ckpt checkpoints/selfplay_atk_canonical/final_model \
        --target_side def \
        --total_timesteps 200000 \
        --run_name cross_atk2def
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent


def transfer(
    source_ckpt: str,
    target_side: int,
    total_timesteps: int = 200_000,
    opponent_update_freq: int = 25_000,
    obs_mode: str = "canonical",
    save_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    use_wandb: bool = False,
    verbose: int = 1,
):
    """
    Load a trained model from `source_ckpt`, switch it to play `target_side`,
    and fine-tune with self-play. The initial opponent is random (baseline),
    then transitions to self-play.
    """
    from gym_tafl.envs.configs import ATK, DEF
    from agents.networks import TaflCNN
    from training.self_play import make_env, SelfPlayCallback, load
    from sb3_contrib import MaskablePPO

    os.chdir(_PROJECT_ROOT)
    side_name = "atk" if target_side == ATK else "def"
    run_name = run_name or f"cross_to_{side_name}"
    save_dir = Path(save_dir) if save_dir else _PROJECT_ROOT / "checkpoints" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(side=target_side, obs_mode=obs_mode)

    # Load the source model; the policy weights transfer — only the side label changes
    model = MaskablePPO.load(source_ckpt, env=env)
    print(f"[cross_play] Loaded {source_ckpt} → now playing as {'ATK' if target_side == ATK else 'DEF'}")

    callbacks = [SelfPlayCallback(env, opponent_update_freq=opponent_update_freq, verbose=verbose)]

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(project="hnefatafl", name=run_name, sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=True)
    model.save(str(save_dir / "final_model"))
    print(f"[cross_play] Saved to {save_dir / 'final_model'}")
    return model


def zero_shot_eval(
    ckpt: str,
    target_side: int,
    n_episodes: int = 200,
    obs_mode: str = "canonical",
) -> dict:
    """
    Evaluate a checkpoint with NO fine-tuning on the opposite side.
    Returns win/draw/loss rates.
    """
    from gym_tafl.envs.configs import ATK, DEF
    from env.tafl_wrapper import TaflEnv, random_opponent
    from env.observations import encode_observation
    from sb3_contrib import MaskablePPO
    import torch

    os.chdir(_PROJECT_ROOT)
    model = MaskablePPO.load(ckpt)
    model.policy.set_training_mode(False)

    env = TaflEnv(side=target_side, opponent_fn=random_opponent, obs_mode=obs_mode)

    wins = draws = losses = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        winner = info.get("winner")
        if winner == target_side:
            wins += 1
        elif winner == -1:  # DRAW
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    return {"wins": wins / total, "draws": draws / total, "losses": losses / total, "n": total}
