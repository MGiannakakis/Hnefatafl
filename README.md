# Hnefatafl RL

Reinforcement learning experiments on **Hnefatafl** (the Tablut variant, 9x9), the asymmetric
Norse board game: the **attackers** (ATK) try to capture the king; the **defenders** (DEF)
try to escort the king to an edge of the board.

The asymmetry is the point. This project trains agents with self-play PPO and studies how
well a policy learned as one side transfers to the other:

- **Phase 1 — Self-play:** train a single side (ATK or DEF) with [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html),
  periodically snapshotting the policy as its own opponent.
- **Phase 2 — Cross-perspective transfer:** take a checkpoint trained as one side and
  evaluate/fine-tune it playing the *other* side, measuring zero-shot performance and
  adaptation speed.
- **Phase 3 — Shared representation:** a single CNN trunk with side-specific heads
  (`SharedTaflCNN`) to test whether the two roles can share features.

The game engine comes from [tafl-gym](https://github.com/gallorob/tafl-gym); this repo wraps
it in a Gymnasium single-agent env with action masking and adds the training/eval stack.

## Setup

Create a Python 3.10 virtual environment and install the requirements:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default PyPI torch wheel is CPU-only. For an NVIDIA GPU, swap in the CUDA build:

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(cu128 is the newest CUDA wheel line published for Python 3.10; torch ≥ 2.12 requires
Python 3.11+.)

## Usage

Everything runs through one Hydra entry point. Defaults live in
[experiments/configs/default.yaml](experiments/configs/default.yaml); any value can be
overridden on the command line.

```bash
# Phase 1: self-play training (attacker side)
python experiments/run.py mode=train training.side=atk

# ...or the defender, with the perspective observation encoding and W&B logging
python experiments/run.py mode=train training.side=def obs_mode=perspective training.use_wandb=true

# Phase 2: transfer an attacker checkpoint to the defender side
python experiments/run.py mode=cross_play \
    cross_play.source_ckpt=checkpoints/selfplay_atk_canonical/final_model \
    cross_play.target_side=def

# Evaluate a checkpoint against a random opponent
python experiments/run.py mode=eval \
    eval.ckpt=checkpoints/selfplay_atk_canonical/final_model \
    eval.side=atk
```

Models are written to `checkpoints/<run_name>/`: interim `model_<steps>_steps.zip` every
25k timesteps (`training.checkpoint_freq`, `0` disables) so interrupted runs keep their
latest weights, plus `final_model.zip` on completion. TensorBoard and CSV metric logs go
to `checkpoints/<run_name>/logs/`.

### Live diagnostics

Training serves an interactive dashboard at **http://127.0.0.1:8787/dashboard.html**
(port via `training.dashboard_port`, `0` disables the server). It auto-refreshes every
5 s and shows loss curves, PPO health metrics (approx KL, clip fraction, explained
variance), episode stats, and the final boards of recent self-play games — with a
"how to read" explainer on every panel, an EMA smoothing slider, hover tooltips, and
a light/dark theme toggle.

Everything is refreshed every PPO update (tune with `training.plot_freq=<timesteps>`,
`0` disables) under `checkpoints/<run_name>/diagnostics/`:

- `dashboard.html` + `data.json` — the live dashboard; the latest data is also embedded
  into the HTML, so opening the file directly after training shows the final snapshot
- `dashboard.png` / `recent_games.png` — static exports of the same content

## Project layout

| Path | Purpose |
|---|---|
| [env/tafl_wrapper.py](env/tafl_wrapper.py) | `TaflEnv` — Gymnasium wrapper around tafl-gym; the opponent is a pluggable `opponent_fn` inside the env; provides `action_masks()` for MaskablePPO |
| [env/observations.py](env/observations.py) | Board encodings: `canonical` (absolute pieces + side flag) and `perspective` (own vs. enemy) — both 9x9x6 |
| [agents/networks.py](agents/networks.py) | CNN feature extractors for SB3 (`TaflCNN`, `SharedTaflCNN`) |
| [training/self_play.py](training/self_play.py) | Phase 1 training loop and the self-play opponent snapshot callback |
| [training/cross_play.py](training/cross_play.py) | Phase 2 transfer fine-tuning and zero-shot evaluation |
| [eval/metrics.py](eval/metrics.py) | Win-rate measurement and a simple Elo tracker |
| [experiments/run.py](experiments/run.py) | Hydra CLI dispatching `mode=train\|cross_play\|eval` |
| [variants/tablut.ini](variants/tablut.ini) | Game rules (board layout, king capture, draw conditions) read by tafl-gym |
| [configs.ini](configs.ini) | Piece/player constants read by tafl-gym |

Note: tafl-gym resolves `configs.ini` and `variants/*.ini` relative to the current working
directory, so the env modules `os.chdir()` to the repo root on import. Run commands from the
repo root.

## Environment details

- **Action space:** `Discrete(1296)` — 81 squares x 16 rook-style destinations (8 in the
  row, 8 in the column). Illegal moves are masked out via sb3-contrib's `ActionMasker`.
- **Rewards:** engine-shaped move rewards, +1-style terminal bonus on wins, -1.0 when the
  opponent's move ends the game or on an unmasked invalid action.
- **Draws:** 50 turns without capture, threefold repetition, or the 100-move cap
  (see [variants/tablut.ini](variants/tablut.ini)).
