# Duel mode: adversarial co-training of two side-dedicated networks

**Date:** 2026-07-08
**Status:** Approved design, pending implementation
**Mode name:** `mode=duel`

## Motivation

Phase 1 self-play trains one side against frozen snapshots of *itself* forced to play
the other side. Those snapshots never learned the opponent's moves (their logits for
that side's actions are untrained), so the "self-play opponent" is effectively a
quasi-random player and no adversarial arms race occurs. Duel mode fixes this by
training two separate networks — one permanently ATK, one permanently DEF — that battle
each other and improve together in a single run.

## Goals

- One command trains both networks in one process, each side always facing the other's
  latest frozen weights.
- Each network only ever sees observations encoded for its own side (kills the
  side-flag mismatch by construction).
- Outputs slot into the existing pipeline: per-side checkpoints usable by `mode=eval`
  and Phase 2 `cross_play` unchanged.
- Head-to-head evaluation between any two saved checkpoints (`eval.opponent_ckpt`).

## Non-goals (deliberate, revisit later)

- Opponent pools / league play (sampling older snapshots to prevent strategy cycling).
- Elo tracking across phases (`eval/metrics.py:EloTracker` exists but stays unwired).
- True concurrent (same-timestep) multi-agent learning; alternating phases chosen for
  stability and reuse of the SB3 single-agent pipeline.
- Shared weights between the two networks (that is Phase 3's experiment).
- `use_wandb` in duel mode (two learners in one process complicate SB3's wandb
  integration; not supported in v1).

## Research plan update

Duel mode becomes **Phase 1b**. The phased plan in CLAUDE.md/README becomes:

| Phase | What | Status |
|---|---|---|
| 1 | Single-side self-play vs frozen self-snapshots (`mode=train`) | exists; kept as controlled baseline |
| **1b** | **Adversarial co-training: two networks battle and train alternately (`mode=duel`)** | **this spec** |
| 2 | Cross-side transfer fine-tuning (`mode=cross_play`) | exists; can consume duel checkpoints |
| 3 | Shared side-conditioned representation (`SharedTaflCNN`) | future |

## Design

### Training loop (new file `training/duel.py`)

Two independent `MaskablePPO` models and two persistent vec envs, built once:

```
atk_env  = make_vec_env(side=ATK, ...)      # opponent slot filled with DEF snapshots
def_env  = make_vec_env(side=DEF, ...)      # opponent slot filled with ATK snapshots
atk_model = fresh MaskablePPO or load(duel.atk_ckpt)
def_model = fresh MaskablePPO or load(duel.def_ckpt)

install PolicyOpponent(def_model.policy, side=DEF) into atk_env    # phase 0 opponent:
                                                                   # untrained DEF net
while either model < duel.total_timesteps_per_side:
    atk_model.learn(to atk_model.num_timesteps + steps_per_phase)  # DEF frozen
    install PolicyOpponent(atk_model.policy, side=ATK) into def_env
    def_model.learn(to def_model.num_timesteps + steps_per_phase)  # ATK frozen
    install PolicyOpponent(def_model.policy, side=DEF) into atk_env
```

- Alternation guarantees each learner faces a **stationary** opponent within a phase
  (PPO's assumption), while both improve across the run.
- Phase 0: ATK trains against the untrained DEF network — through a masked softmax
  that plays approximately randomly. No `random_opponent` code path exists in duel
  mode; the opponent is always the other network.
- `SelfPlayCallback` is not used; opponent swaps happen only at phase boundaries.
- Snapshots reuse `training/self_play.py:PolicyOpponent` unchanged (deep-copied
  policy, moved to CPU, installed via `vec_env.env_method("set_opponent", ...)`).
  **Each snapshot encodes observations with its own side** (`side=DEF` for the DEF
  network playing inside ATK's env), so every network always sees inputs consistent
  with its training.

### SB3 mechanics (implementation constraints)

- Repeated `learn()` calls use `reset_num_timesteps=False` and an **absolute** target:
  `model.learn(total_timesteps=model.num_timesteps + steps_per_phase, ...)`.
- Both networks use the top-level `obs_mode` (same convention as every other mode).
- Each model gets its own logger via `set_logger(configure(...))` once, before the
  loop, writing to its side's directory (`atk/logs/`, `def/logs/`).
- Callback instances (checkpoint + diagnostics) are created once per side and reused
  across every `learn()` call so their state (recent-games deque, save counters)
  persists. `DiagnosticsCallback._on_training_start` fires on every `learn()`; it must
  become idempotent (start the HTTP server and mkdir only on first call).
- `CheckpointCallback(save_freq=checkpoint_freq // n_envs)` per side, saving into that
  side's directory.

### Configuration (`experiments/configs/default.yaml`)

```yaml
duel:
  total_timesteps_per_side: 1_000_000
  steps_per_phase: 50_000    # alternation granularity per side
  features_dim: 256
  learning_rate: 3e-4
  n_envs: 8                  # per vec env (two vec envs exist; only one trains at a time)
  vec_env: subproc
  n_steps: 256
  batch_size: 512
  n_epochs: 10
  atk_ckpt: null             # optional warm starts; fresh networks when null
  def_ckpt: null
  run_name: null             # default: duel_<obs_mode>
  checkpoint_freq: 25_000
  plot_freq: 10_000
  dashboard_port: 8787       # ATK dashboard; DEF serves on dashboard_port + 1
  verbose: 1
```

`experiments/run.py` gains a `mode=duel` branch dispatching to
`training.duel.duel_train(...)`.

### Outputs

```
checkpoints/<run_name>/
  atk/  model_<steps>_steps.zip ..., final_model.zip, logs/, diagnostics/
  def/  model_<steps>_steps.zip ..., final_model.zip, logs/, diagnostics/
```

Both `final_model.zip`s are ordinary MaskablePPO checkpoints — `mode=eval` and
`mode=cross_play` consume them with no changes.

### Head-to-head evaluation (`eval.opponent_ckpt`)

New optional config key:

```yaml
eval:
  ckpt: null
  side: atk
  opponent_ckpt: null   # checkpoint that plays the OTHER side; random opponent when null
  n_episodes: 200
```

In `run.py`'s eval branch: when `opponent_ckpt` is set, load it and build
`PolicyOpponent(loaded.policy, side=<opposite of eval.side>, obs_mode=cfg.obs_mode)`
as the `opponent_fn` passed to `eval/metrics.py:win_rate` (which already accepts any
opponent_fn — no changes there). The evaluated agent plays deterministically (as
today); the opponent samples stochastically — intentional, otherwise every episode of
a deterministic-vs-deterministic pairing is the identical game. Output labels both
checkpoints.

Eval-vs-random (opponent_ckpt null) remains the **fixed yardstick** for absolute
progress: measured baselines are ATK 16% / DEF 41% / draws 43% for random-vs-random.
It is never used inside duel training.

### Error handling

- The phase loop runs in `try/finally`: on interrupt or crash, both models save
  `final_model.zip`, both vec envs close. Interim checkpoints already persist every
  `checkpoint_freq` steps.
- Loading `atk_ckpt`/`def_ckpt` validates the file exists before building envs
  (fail fast). Warm-started models take duel's `n_steps`/`batch_size` as load
  overrides, same pattern as `cross_play.transfer`.

### Resource notes

- Two models on GPU is trivial (~12 MB weights each).
- Two subproc vec envs = 2 × n_envs worker processes. Idle workers cost no CPU
  (blocked on pipe reads) but each holds a torch import in RAM. During smoke testing,
  measure total working set; if RAM-constrained, drop `duel.n_envs` or use
  `vec_env=dummy`. Do not rebuild envs per phase (respawn cost ~10-20 s × many phases).

## Verification plan

1. **Smoke duel** (dummy vec, small `steps_per_phase`, few phases): assert both
   models' `num_timesteps` advance to target; opponents are installed at every phase
   boundary; the opponent installed in ATK's env has `side == DEF` (and vice versa);
   per-side checkpoints and logs appear in the right directories.
2. **Subproc duel smoke**: same, through SubprocVecEnv, plus dashboard on two ports.
3. **Interrupt test**: kill a smoke run mid-phase; both sides' latest weights exist.
4. **Head-to-head eval smoke**: `mode=eval` with `opponent_ckpt` between the two smoke
   checkpoints runs and reports win/draw/loss.
5. **Real run success criteria**: duel-ATK's win rate vs a random defender exceeds the
   16% random baseline and rises across phases; duel-DEF exceeds its 41% baseline;
   head-to-head stays broadly competitive (neither side pinned at ~100% for many
   consecutive phases, which would indicate broken alternation) with decisive games
   becoming more common than the random-vs-random 57%.

## Risks and monitoring

- **Strategy cycling** (rock-paper-scissors across phases): visible as oscillating
  head-to-head evals. Mitigation if observed: opponent pools (non-goal for v1).
- **Draw-heavy equilibrium**: with draw reward 0 for both sides, mutual stalling
  yields no gradient. If both sides converge to draws, consider a small
  attacker-only draw penalty (config knob, future work).
- **One-sided collapse**: if one side wins ~everything for several phases, the loser's
  reward signal is all-losses (sparse). Shorter `steps_per_phase` gives the weaker
  side more frequent catch-up opportunities; tune before reaching for anything fancier.

## Files to change

| File | Change |
|---|---|
| `training/duel.py` | new — phase loop, model/env setup, save/close handling |
| `training/diagnostics.py` | make `_on_training_start` idempotent across repeated `learn()` calls |
| `experiments/configs/default.yaml` | add `duel:` block; add `eval.opponent_ckpt` |
| `experiments/run.py` | `mode=duel` dispatch; head-to-head branch in eval |
| `CLAUDE.md` | phase table (add 1b), architecture bullet for `training/duel.py` |
| `README.md` | phase list, duel usage example, head-to-head eval example |
| `notes/` | short duel-mode note following the existing notes convention |
