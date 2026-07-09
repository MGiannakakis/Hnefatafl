# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL research project training agents to play Hnefatafl (Tablut variant, 9x9) with MaskablePPO.
The research is phased: Phase 1 = single-side self-play training (baseline), Phase 1b =
adversarial co-training of two side-dedicated networks (`mode=duel`), Phase 2 =
cross-perspective transfer (fine-tune an ATK-trained policy to play DEF, or vice
versa), Phase 3 = shared representation (side-conditioned network, see `SharedTaflCNN`).

The game engine itself is NOT in this repo — it comes from the external `gym_tafl` package
(`tafl-gym`, installed from GitHub via requirements.txt). This repo wraps it for SB3.

## Commands

Uses the project-local venv at `.venv` (Python 3.10; the requirements.txt header
mentioning a conda env is historical):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
# GPU torch (RTX 3080): the plain PyPI wheel is CPU-only — replace it with
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# (cu128 is the newest CUDA wheel line for Python 3.10; torch >= 2.12 needs Python 3.11+)
```

All experiments go through the Hydra entry point (config: `experiments/configs/default.yaml`):

```bash
# Phase 1 — self-play training
python experiments/run.py mode=train training.side=atk

# Phase 1b — adversarial co-training (two networks battle and both train)
python experiments/run.py mode=duel duel.run_name=duel_v1

# Head-to-head evaluation between two checkpoints (opponent plays the other side)
python experiments/run.py mode=eval eval.ckpt=checkpoints/duel_v1/atk/final_model \
    eval.side=atk eval.opponent_ckpt=checkpoints/duel_v1/def/final_model

# Phase 2 — cross-perspective transfer
python experiments/run.py mode=cross_play \
    cross_play.source_ckpt=checkpoints/selfplay_atk_canonical/final_model \
    cross_play.target_side=def

# Evaluation vs random opponent
python experiments/run.py mode=eval eval.ckpt=<path> eval.side=atk

# Any config value is overridable Hydra-style, e.g.:
python experiments/run.py training.side=def obs_mode=perspective training.use_wandb=true
```

Models save to `checkpoints/<run_name>/`: `model_<steps>_steps.zip` every
`training.checkpoint_freq` timesteps (default 25k, 0 disables — so interrupted runs keep
their latest weights) plus `final_model.zip` on completion. There are no tests or linter
configured.

## Critical gotcha: working directory

`gym_tafl` loads `configs.ini` and `variants/<variant>.ini` **relative to the current working
directory** at import/instantiation time. That is why:

- `configs.ini` and `variants/tablut.ini` live at the repo root — they are consumed by
  `gym_tafl`, not read directly by this repo's code.
- `env/tafl_wrapper.py` and `env/observations.py` call `os.chdir(_PROJECT_ROOT)` at module
  import time, and training/eval functions re-chdir defensively. Importing `env.*` changes
  your process cwd as a side effect — preserve this pattern in new modules that touch `gym_tafl`.
- Modules use lazy (function-level) imports of `gym_tafl` and `env.*` deliberately, so the
  chdir happens before the ini files load. Don't "clean up" these imports to top-level.

## Architecture

Layered data flow:

```
gym_tafl (external engine)  →  env/  →  agents/  →  training/  →  experiments/run.py
                                          ↘  eval/metrics.py
```

- `env/tafl_wrapper.py` — `TaflEnv`, a **single-agent** Gymnasium wrapper: the opponent lives
  *inside* the env as `opponent_fn(board, valid_actions) -> action`, called during `step()`
  until it's the agent's turn again. Rewards are terminal-only and side-aware (+1 win /
  -1 loss / 0 draw, mapped from `info["winner"]` vs the agent's side); the engine's shaped
  per-move reward is deliberately discarded — it is defender-centric material scoring paid
  every move, which trained the attacker to stall for draws (don't reintroduce it).
  Action space is `Discrete(1296)`: 81 squares x 16 rook-style destinations (8 per row +
  8 per column), indexed via `gym_tafl`'s `IDX_TO_POS`. Exposes `action_masks()` for
  MaskablePPO (sb3-contrib `ActionMasker`).
- `env/observations.py` — two 9x9x6 encodings selected by `obs_mode`: `canonical`
  (absolute attacker/defender/king channels + side flag in channel 5) and `perspective`
  (own/enemy relative). Constants `ATK`, `DEF`, `KING`, etc. come from `gym_tafl.envs.configs`
  (values defined in `configs.ini`).
- `agents/networks.py` — SB3 `BaseFeaturesExtractor`s. `TaflCNN` is the default.
  `SharedTaflCNN` routes through ATK/DEF-specific heads by reading the side flag from
  observation channel 5 — it only works with `obs_mode=canonical`.
- `training/self_play.py` — rollouts come from `training.n_envs` parallel envs
  (`SubprocVecEnv` by default, `vec_env=dummy` for in-process debugging; `n_steps` is
  per env). `SelfPlayCallback` snapshots the current policy every `opponent_update_freq`
  timesteps into a `PolicyOpponent` (module-level class so it pickles into workers;
  runs on CPU — batch-1 inference is faster there than the GPU round trip) and installs
  it via `vec_env.env_method("set_opponent", ...)`. Until the first snapshot the opponent
  is uniform random. Gymnasium 1.x does not forward attribute access through wrappers
  (TaflEnv → ActionMasker → Monitor): use `env_method`/`get_attr` (they resolve via
  `get_wrapper_attr`), never `VecEnv.set_attr`, which only touches the outermost wrapper.
- `training/duel.py` — Phase 1b: two side-dedicated MaskablePPO models train
  alternately (`duel.steps_per_phase` each) in one process, each side playing against
  a `PoolOpponent` of the rival's frozen CPU snapshots (per episode: newest with
  `duel.pool_p_latest`, else uniform over the last `duel.pool_size`; pools grow via
  `env_method("add_opponent_snapshot", ...)`, one pool instance per sub-env;
  snapshots encode observations with their own side). The pool + short phases +
  `duel.ent_coef` exist to prevent self-play cycling — don't set `pool_size=1` /
  `ent_coef=0` except to reproduce that pathology deliberately. No random
  opponents: phase 0's opponent is the untrained DEF network.
  Outputs per side under `checkpoints/<run>/{atk,def}/`; ATK dashboard on
  `duel.dashboard_port`, DEF on the next port. `duel.total_timesteps_per_side` is a
  budget of ADDITIONAL steps (warm starts via `duel.atk_ckpt`/`duel.def_ckpt` keep
  their step counters). Repeated `learn()` calls require callbacks to be
  reused/idempotent — see `DiagnosticsCallback._training_started`.
- `training/diagnostics.py` — `DiagnosticsCallback` regenerates
  `checkpoints/<run>/diagnostics/` every `training.plot_freq` timesteps (0 disables):
  PNG figures plus a live HTML dashboard (`dashboard.html` + `data.json`, template at
  `training/dashboard_template.html`) served on `http://127.0.0.1:<training.dashboard_port>`
  by a daemon thread. Metrics come from the CSV logger set up in `train()`; final boards
  from the `final_board` key `TaflEnv` puts in `info` at episode end. The template is
  UTF-8 — regenerate/edit it only with tools that read and write UTF-8 explicitly
  (PowerShell 5.1 defaults will mojibake it).
- `training/cross_play.py` — loads a checkpoint, swaps the env side, fine-tunes with
  self-play; `zero_shot_eval()` measures transfer without fine-tuning.
- `experiments/run.py` — dispatches `mode=train|cross_play|eval` to the above; the only CLI.

Game rules (board layout, capture rules, draw conditions) are configured in
`variants/tablut.ini`, parsed by the external engine — new variants are added as
`variants/<name>.ini` and selected via `TaflEnv(variant=...)`.

`notes/` holds plain-English explainer notes for the codebase (linked markdown with
line-level code references) — when a change invalidates something they describe, update
the affected note.
