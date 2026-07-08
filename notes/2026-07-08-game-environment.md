# The game environment: turning Hnefatafl into an RL problem

**Date:** 2026-07-08
**Topic:** How `env/` wraps the external game engine so the RL library can play it.

← [RL basics](2026-07-08-rl-basics.md) · [overview](2026-07-08-codebase-overview.md) · Next: [the neural network](2026-07-08-neural-network.md)

## Summary

RL libraries expect a "world" with a standard interface (the **Gymnasium** API): `reset()` starts an episode, `step(action)` advances it and returns what happened. Hnefatafl is a *two-player* game, but the RL setup here is *single-agent* — so [env/tafl_wrapper.py](../env/tafl_wrapper.py) hides the second player **inside** the environment. The agent only ever experiences: "I see a board, I pick a move, the world responds (opponent moved), I see the next board."

## Two layers: engine vs wrapper

- **The engine (`gym_tafl`, external package)** knows the rules: legal moves, captures, king escape, draw conditions. Rules are read from [../variants/tablut.ini](../variants/tablut.ini) at startup — board layout, capture rules, and so on are *config*, not code.
- **The wrapper (`TaflEnv`)** adapts the engine to the RL interface and embeds the opponent.

## What one `step()` looks like

From [env/tafl_wrapper.py:99](../env/tafl_wrapper.py#L99), when the agent submits a move:

1. **Safety check** — if the move is illegal (shouldn't happen thanks to masking), end the game with reward -1.0.
2. **Apply the agent's move** via the engine. If it ends the game (king captured / king escaped / draw), return the final reward and stop.
3. **Let the opponent play** — `_play_opponent()` ([line 206](../env/tafl_wrapper.py#L206)) calls `opponent_fn(board, valid_actions)` until it's the agent's turn again. If the game ends on the opponent's move, the agent receives **-1.0** ([line 127](../env/tafl_wrapper.py#L127)) — losing hurts no matter whose move sealed it.
4. **Return** the new board encoding, the reward, and a fresh action mask.

The opponent is just a plugged-in function. It starts as `random_opponent` ([line 216](../env/tafl_wrapper.py#L216)) — pick any legal move uniformly — and during training gets swapped for a frozen copy of the agent itself (see the [self-play note](2026-07-08-self-play-training.md)). `set_opponent()` ([line 149](../env/tafl_wrapper.py#L149)) does the swap.

## Actions: why exactly 1296?

Every move is "from square A to square B" where pieces slide like chess rooks. Encoding: **81 starting squares × 16 destinations** (8 possible squares along the row + 8 along the column) = 1296. Each integer 0–1295 maps to a concrete (from, to) pair via the engine's lookup table. Most are illegal at any moment — that's what the **action mask** ([line 140](../env/tafl_wrapper.py#L140)) is for: a 1296-long true/false array marking the currently legal ones.

## Observations: what the network actually sees

The raw board is a 9×9 grid of piece codes. [env/observations.py](../env/observations.py) re-encodes it as a **9×9×6 stack of layers** (all 0s and 1s), because neural networks digest "one concept per layer" far better than arbitrary code numbers. Two encodings, chosen by `obs_mode`:

- **`canonical`** ([line 13](../env/observations.py#L13)) — absolute labels: layer 0 = attackers, 1 = defenders, 2 = king, 3 = throne, 4 = corners, 5 = *entire layer filled with 1 if you're playing attacker, 0 if defender*. Same board looks the same to both sides; the flag layer says which side you're on.
- **`perspective`** ([line 33](../env/observations.py#L33)) — relative labels: layer 0 = *my* pieces, 2 = *enemy* pieces, etc. The same board looks *different* depending on your side, but "my pieces" always means the same thing.

This choice is a real research variable: canonical vs perspective changes how well knowledge transfers between sides (Phase 2/3 of the project).

## Rewards in one place

| Event | Reward |
|---|---|
| Agent's move wins the game | ~ +1 (from the engine) |
| Game ends in a loss on the opponent's move | -1.0 |
| Opponent left with no legal moves | +1.0 |
| Illegal action (masking failed) | -1.0, game over |
| Ordinary mid-game move | ~ 0 |

## The working-directory gotcha

The engine loads its `.ini` files **relative to the current working directory**. So [env/tafl_wrapper.py:10](../env/tafl_wrapper.py#L10) does `os.chdir(project_root)` *at import time*, and other modules import `gym_tafl` lazily inside functions so the chdir happens first. Side effect: importing `env.*` changes your process's cwd. Preserve this pattern in any new module touching `gym_tafl`.
