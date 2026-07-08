# Running experiments: the commands and the config

**Date:** 2026-07-08
**Topic:** How to actually run training/eval, how Hydra config works, where outputs land.

← [Self-play training](2026-07-08-self-play-training.md) · back to [overview](2026-07-08-codebase-overview.md)

## Summary

Everything runs through one script: [experiments/run.py](../experiments/run.py). A `mode=` switch picks the phase; every other setting comes from [experiments/configs/default.yaml](../experiments/configs/default.yaml) and can be overridden on the command line (that's **Hydra**, the config library).

## The three modes

```bash
# Phase 1 — train an attacker from scratch via self-play (~1M moves)
python experiments/run.py mode=train training.side=atk

# Phase 2 — take the trained attacker, fine-tune it to play defender
python experiments/run.py mode=cross_play \
    cross_play.source_ckpt=checkpoints/selfplay_atk_canonical/final_model \
    cross_play.target_side=def

# Evaluate a checkpoint: win rate over 200 games vs a random opponent
python experiments/run.py mode=eval eval.ckpt=<path> eval.side=atk
```

Any config value is overridable in the same `dotted.path=value` style, e.g.:

```bash
python experiments/run.py training.side=def obs_mode=perspective training.use_wandb=true
```

Use the project venv: `.venv\Scripts\python.exe` (Python 3.10; GPU torch install steps are in [CLAUDE.md](../CLAUDE.md)).

## What happens when you hit enter (mode=train)

1. `run.py` chdirs to the project root (the [cwd gotcha](2026-07-08-codebase-overview.md#one-gotcha-worth-knowing-from-day-one)) and Hydra merges `default.yaml` with your overrides, printing the final config.
2. `main()` ([experiments/run.py:31](../experiments/run.py#L31)) dispatches to `train()` in [training/self_play.py](../training/self_play.py).
3. 8 game processes spin up, the PPO model is built, and the [collect → learn → snapshot loop](2026-07-08-self-play-training.md) runs to `total_timesteps`.
4. Progress: console log lines, a CSV/TensorBoard log, and the live dashboard at `http://127.0.0.1:8787`.

## Settings you'll actually touch

From [default.yaml](../experiments/configs/default.yaml):

| Key | Default | Meaning |
|---|---|---|
| `training.side` | `atk` | Which side the agent learns to play |
| `obs_mode` | `canonical` | Board encoding — a real research variable, see [environment note](2026-07-08-game-environment.md) |
| `training.total_timesteps` | 1M | Training length (agent moves) |
| `training.opponent_update_freq` | 50k | How often the self-play opponent refreshes |
| `training.n_envs` | 8 | Parallel game processes |
| `training.vec_env` | `subproc` | Set to `dummy` for single-process debugging (breakpoints work) |
| `training.plot_freq` | 10k | Dashboard/plot refresh; `0` disables diagnostics |
| `training.use_wandb` | false | Mirror logs to Weights & Biases |

## Where outputs land

```
checkpoints/<run_name>/          e.g. selfplay_atk_canonical/
├── final_model.zip              the trained agent (policy + weights)
├── logs/progress.csv            metrics feed for diagnostics
├── tb_logs/                     TensorBoard logs
└── diagnostics/
    ├── dashboard.html           open directly, or served live on port 8787
    ├── data.json                metrics + recent final boards
    └── *.png                    static plot exports
```

Run names default to `selfplay_<side>_<obs_mode>`. Hydra also writes an `outputs/` directory with per-run config snapshots — useful for reconstructing what settings a run used.

## Odds and ends

- **No tests or linter** are configured in this repo.
- Game rules live in [../variants/tablut.ini](../variants/tablut.ini); a new variant = new `variants/<name>.ini` + `TaflEnv(variant=...)`.
- The dashboard template ([training/dashboard_template.html](../training/dashboard_template.html)) is UTF-8; editing it with PowerShell 5.1 defaults will corrupt it — use tools that read/write UTF-8 explicitly.
- Evaluation utilities beyond win-rate (an `EloTracker` for comparing snapshots over time) live in [eval/metrics.py:78](../eval/metrics.py#L78) but aren't wired into the CLI yet.
