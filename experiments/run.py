"""
Unified entry point for all experiment phases.

  Phase 1 – self-play training:
    python experiments/run.py mode=train training.side=atk

  Phase 2 – cross-perspective transfer:
    python experiments/run.py mode=cross_play \
        cross_play.source_ckpt=checkpoints/selfplay_atk_canonical/final_model \
        cross_play.target_side=def

  Evaluation only:
    python experiments/run.py mode=eval \
        eval.ckpt=checkpoints/selfplay_atk_canonical/final_model \
        eval.side=atk
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    mode = cfg.get("mode", "train")

    if mode == "train":
        from gym_tafl.envs.configs import ATK, DEF
        from training.self_play import train

        side_str = cfg.training.side.lower()
        side = ATK if side_str == "atk" else DEF

        train(
            side=side,
            total_timesteps=cfg.training.total_timesteps,
            opponent_update_freq=cfg.training.opponent_update_freq,
            obs_mode=cfg.obs_mode,
            features_dim=cfg.training.features_dim,
            learning_rate=cfg.training.learning_rate,
            n_steps=cfg.training.n_steps,
            batch_size=cfg.training.batch_size,
            n_epochs=cfg.training.n_epochs,
            save_dir=cfg.training.save_dir,
            run_name=cfg.training.run_name,
            use_wandb=cfg.training.use_wandb,
            verbose=cfg.training.verbose,
            plot_freq=cfg.training.plot_freq,
        )

    elif mode == "cross_play":
        from gym_tafl.envs.configs import ATK, DEF
        from training.cross_play import transfer

        target_str = cfg.cross_play.target_side.lower()
        target_side = ATK if target_str == "atk" else DEF

        transfer(
            source_ckpt=cfg.cross_play.source_ckpt,
            target_side=target_side,
            total_timesteps=cfg.cross_play.total_timesteps,
            opponent_update_freq=cfg.cross_play.opponent_update_freq,
            obs_mode=cfg.obs_mode,
            run_name=cfg.cross_play.run_name,
            use_wandb=cfg.cross_play.use_wandb,
        )

    elif mode == "eval":
        from gym_tafl.envs.configs import ATK, DEF
        from training.self_play import load
        from eval.metrics import win_rate, model_policy_fn, random_policy_fn
        from env.tafl_wrapper import random_opponent

        side_str = cfg.get("eval", {}).get("side", "atk").lower()
        side = ATK if side_str == "atk" else DEF
        ckpt = cfg.eval.ckpt

        model = load(ckpt)
        results = win_rate(
            policy_fn=model_policy_fn(model),
            side=side,
            opponent_fn=random_opponent,
            n_episodes=cfg.eval.n_episodes,
            obs_mode=cfg.obs_mode,
        )
        print(f"\n=== Eval results ({side_str} vs random, {cfg.eval.n_episodes} eps) ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    else:
        print(f"Unknown mode: {mode}. Use mode=train|cross_play|eval")


if __name__ == "__main__":
    main()
