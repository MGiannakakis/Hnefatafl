# Hnefatafl RL — codebase overview (start here)

**Date:** 2026-07-08
**Topic:** Plain-English map of the whole project: what it does, how the pieces fit, and where to read next.

## Summary

This project teaches a computer program to play **Hnefatafl** (specifically the 9x9 *Tablut* variant — an old Norse board game where attackers try to capture a king and defenders try to help him escape). It learns purely by **playing against itself millions of times** and gradually adjusting a neural network so that moves that led to wins become more likely. No human games, no hand-written strategy — just trial, error, and a reward signal.

The research question behind it: if a network learns to play one side (say, attacker), how much of that knowledge **transfers** to playing the other side?

## The one-paragraph version of how learning works

The agent starts out playing essentially random moves. Every game ends in a reward: roughly **+1 for a win, -1 for a loss**. After each batch of games, a learning algorithm called **PPO** nudges the network's millions of internal dials so that the moves which preceded wins become slightly more probable, and moves which preceded losses become slightly less probable. Its opponent is a frozen copy of *itself* from a little while ago, refreshed every 50,000 moves — so as the agent improves, its opposition improves too, like climbing a ladder it builds under itself. Repeat for a million moves and a genuine playing style emerges.

## The big picture: how data flows

```
gym_tafl            env/                agents/            training/           experiments/
(external game  →   wraps the game  →   the neural     →   the learning   →   run.py
 rules engine)      for the RL          network (the        loop (self-        (the command
                    library             "brain")            play + PPO)        you actually run)
```

- **`gym_tafl`** — the game engine (board, legal moves, captures, win conditions). *Not in this repo*; installed as a package. Rules are configured by [../variants/tablut.ini](../variants/tablut.ini).
- **[env/](../env/)** — turns the two-player board game into a "video game for one player" that the RL library understands. See [Game environment note](2026-07-08-game-environment.md).
- **[agents/](../agents/)** — the neural network that looks at a board and decides a move. See [Neural network note](2026-07-08-neural-network.md).
- **[training/](../training/)** — the learning loop: self-play, PPO updates, live diagnostics dashboard, cross-side transfer. See [Self-play training note](2026-07-08-self-play-training.md).
- **[eval/](../eval/)** — measuring how good a trained agent is (win rate vs a random player, Elo).
- **[experiments/run.py](../experiments/run.py)** — the single command-line entry point for everything. See [Running experiments note](2026-07-08-running-experiments.md).

## The three research phases

1. **Phase 1 — self-play**: train one side (attacker *or* defender) from scratch against copies of itself.
2. **Phase 2 — cross-perspective transfer**: take a network trained as attacker, make it play defender (or vice versa), and measure how quickly it adapts ([training/cross_play.py](../training/cross_play.py)).
3. **Phase 3 — shared representation**: one network with a shared "trunk" and two side-specific "heads", so both sides share most of their knowledge (`SharedTaflCNN` in [agents/networks.py](../agents/networks.py)).

## Reading order

1. [RL basics, using this project's vocabulary](2026-07-08-rl-basics.md) — agent, environment, reward, policy, PPO. Read this first if RL is new.
2. [The game environment](2026-07-08-game-environment.md) — how a board game becomes something an RL algorithm can play.
3. [The neural network](2026-07-08-neural-network.md) — the "brain" that picks moves.
4. [The self-play learning loop](2026-07-08-self-play-training.md) — the step-by-step mechanics of how it actually learns. **The centerpiece.**
5. [Running experiments](2026-07-08-running-experiments.md) — the commands, the config file, where results end up.

## One gotcha worth knowing from day one

The external `gym_tafl` engine loads its config files (`configs.ini`, `variants/tablut.ini`) **relative to whatever directory the process happens to be in**. That's why several modules call `os.chdir(...)` to the project root at import time, and why imports of `gym_tafl` are often buried inside functions instead of at the top of files. It looks untidy; it is deliberate. Don't "clean it up."
