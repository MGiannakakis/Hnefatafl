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
            n_envs=cfg.training.n_envs,
            vec_env=cfg.training.vec_env,
            save_dir=cfg.training.save_dir,
            run_name=cfg.training.run_name,
            use_wandb=cfg.training.use_wandb,
            verbose=cfg.training.verbose,
            plot_freq=cfg.training.plot_freq,
            dashboard_port=cfg.training.dashboard_port,
            checkpoint_freq=cfg.training.checkpoint_freq,
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
            n_steps=cfg.cross_play.n_steps,
            batch_size=cfg.cross_play.batch_size,
            n_envs=cfg.cross_play.n_envs,
            vec_env=cfg.cross_play.vec_env,
            run_name=cfg.cross_play.run_name,
            use_wandb=cfg.cross_play.use_wandb,
            checkpoint_freq=cfg.cross_play.checkpoint_freq,
        )

    elif mode == "duel":
        from training.duel import duel_train

        duel_train(
            total_timesteps_per_side=cfg.duel.total_timesteps_per_side,
            steps_per_phase=cfg.duel.steps_per_phase,
            obs_mode=cfg.obs_mode,
            features_dim=cfg.duel.features_dim,
            learning_rate=cfg.duel.learning_rate,
            n_envs=cfg.duel.n_envs,
            vec_env=cfg.duel.vec_env,
            n_steps=cfg.duel.n_steps,
            batch_size=cfg.duel.batch_size,
            n_epochs=cfg.duel.n_epochs,
            ent_coef=cfg.duel.ent_coef,
            pool_size=cfg.duel.pool_size,
            pool_p_latest=cfg.duel.pool_p_latest,
            atk_ckpt=cfg.duel.atk_ckpt,
            def_ckpt=cfg.duel.def_ckpt,
            run_name=cfg.duel.run_name,
            checkpoint_freq=cfg.duel.checkpoint_freq,
            plot_freq=cfg.duel.plot_freq,
            dashboard_port=cfg.duel.dashboard_port,
            verbose=cfg.duel.verbose,
        )

    elif mode == "eval":
        from gym_tafl.envs.configs import ATK, DEF
        from training.self_play import load
        from eval.metrics import win_rate, model_policy_fn, random_policy_fn
        from env.tafl_wrapper import random_opponent

        side_str = cfg.get("eval", {}).get("side", "atk").lower()
        side = ATK if side_str == "atk" else DEF
        ckpt = cfg.eval.ckpt

        if cfg.eval.opponent_ckpt:
            from training.self_play import PolicyOpponent
            opp_side = DEF if side == ATK else ATK
            opp_model = load(cfg.eval.opponent_ckpt)
            opponent_fn = PolicyOpponent(opp_model.policy, side=opp_side,
                                         obs_mode=cfg.obs_mode)
            opp_name = str(cfg.eval.opponent_ckpt)
        else:
            opponent_fn = random_opponent
            opp_name = "random"

        model = load(ckpt)
        results = win_rate(
            policy_fn=model_policy_fn(model),
            side=side,
            opponent_fn=opponent_fn,
            n_episodes=cfg.eval.n_episodes,
            obs_mode=cfg.obs_mode,
        )
        print(f"\n=== Eval results ({ckpt} as {side_str} vs {opp_name}, "
              f"{cfg.eval.n_episodes} eps) ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    else:
        print(f"Unknown mode: {mode}. Use mode=train|duel|cross_play|eval")


if __name__ == "__main__":
    main()
