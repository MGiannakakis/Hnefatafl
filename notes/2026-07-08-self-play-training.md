# The self-play learning loop, step by step

**Date:** 2026-07-08
**Topic:** The exact sequence of events when training runs — the centerpiece of how learning works here.

← [Neural network](2026-07-08-neural-network.md) · [overview](2026-07-08-codebase-overview.md) · Next: [running experiments](2026-07-08-running-experiments.md)

## Summary

`train()` in [training/self_play.py:119](../training/self_play.py#L119) wires everything together: 8 parallel game environments, a MaskablePPO learner on the GPU, and a callback that periodically freezes the current agent and installs it as its own opponent. Numbers below are the defaults from [experiments/configs/default.yaml](../experiments/configs/default.yaml).

## Setup (happens once)

1. **8 environments are created** ([make_vec_env, line 103](../training/self_play.py#L103)), each in its **own OS process** (`SubprocVecEnv`), so 8 games run simultaneously on different CPU cores. Each env is wrapped twice: `ActionMasker` (exposes legal-move masks to PPO) and `Monitor` (records episode rewards/lengths for the logs).
2. **The opponent in every env starts as pure random** — the agent's first sparring partner is a coin-flipper.
3. **The MaskablePPO model is built** ([line 154](../training/self_play.py#L154)) with the `TaflCNN` extractor. Fresh random weights: at this point the agent is also essentially a coin-flipper.

## The core loop (repeats until 1,000,000 timesteps)

### Step 1 — Collect a rollout (gameplay)

Each of the 8 envs plays **256 agent moves** (`n_steps`) → a buffer of **2048 recorded transitions**: (board seen, move chosen, probability it had, reward, critic's value estimate). The 8 boards are batched into single GPU forward passes; opponent moves happen invisibly inside each env's `step()` on CPU.

### Step 2 — Learn from the rollout (the PPO update)

1. **Compute advantages**: for each recorded move, combine subsequent rewards (discounted by `gamma=0.99`) with the critic's predictions to score "did this move work out better or worse than expected?"
2. **Make 10 passes** (`n_epochs`) over the 2048 transitions in shuffled mini-batches of **512** (`batch_size`). For each batch, gradient descent adjusts *all* weights (conv layers + both heads) to:
   - raise the probability of better-than-expected moves and lower worse-than-expected ones — **clipped** so no single update moves the policy far (the "Proximal" in PPO);
   - make the critic's value predictions more accurate;
   - keep a bit of randomness in move choice (entropy bonus) so exploration doesn't die too early.
3. **Discard the rollout** — it described the old policy — and return to Step 1.

One cycle ≈ 2048 timesteps, so training is roughly **490 collect→learn cycles**.

### Step 3 — Every 50,000 timesteps: the opponent levels up

`SelfPlayCallback` ([line 66](../training/self_play.py#L66)) fires between cycles:

1. **Deep-copies the current policy** — a frozen snapshot, unaffected by further learning.
2. Wraps it as a `PolicyOpponent` ([line 35](../training/self_play.py#L35)): a function that encodes the board *from the opponent side's point of view*, runs the frozen net, returns its move.
3. **Installs it into all 8 envs**, replacing the previous opponent.

So the agent spends 50k steps learning to beat "itself from one snapshot ago," then the bar rises. Random → snapshot-1 → snapshot-2 → ... — a self-generating curriculum. About 20 opponent generations happen over a full run.

Two engineering choices worth noting:

- **The opponent runs on CPU inside each worker process** — batch-of-1 inference on a tiny net is faster than a GPU round trip, it parallelizes across the 8 workers, and a CPU policy can be pickled into subprocesses.
- **Installing uses `vec_env.env_method("set_opponent", ...)`**, which reaches through the wrapper layers to the base `TaflEnv` — Gymnasium wrappers don't forward attribute writes, a known trap in this codebase.

## Watching it learn

If `plot_freq > 0`, a `DiagnosticsCallback` ([training/diagnostics.py:79](../training/diagnostics.py#L79)) regenerates plots and a **live dashboard** at `http://127.0.0.1:8787` every 10k steps — reward curves, loss curves, and final boards of recent games. See the [jargon table in RL basics](2026-07-08-rl-basics.md#jargon-cheat-sheet) for what each curve means.

One subtlety when reading the reward curve: because the opponent keeps improving too, **`ep_rew_mean` hovering near 0 does not mean "not learning"** — it can mean the agent and its recent snapshot stay evenly matched while both get stronger. Expect a sawtooth: reward climbs as the agent outgrows the frozen snapshot, then drops when the opponent updates. Absolute strength is measured separately, by evaluating against a *fixed* opponent ([eval/metrics.py:13](../eval/metrics.py#L13)).

## The finish line

After 1M timesteps the model is saved to `checkpoints/<run_name>/final_model.zip`. That checkpoint is the input to Phase 2: [training/cross_play.py:24](../training/cross_play.py#L24) loads it, points it at the *opposite* side, and fine-tunes — measuring how much attacker knowledge transfers to playing defender (or vice versa). `zero_shot_eval` ([line 78](../training/cross_play.py#L78)) measures transfer with *no* fine-tuning at all.
