# The neural network: the agent's "brain"

**Date:** 2026-07-08
**Topic:** What the CNN in `agents/networks.py` does and how SB3 builds a full policy around it.

← [Game environment](2026-07-08-game-environment.md) · [overview](2026-07-08-codebase-overview.md) · Next: [the self-play learning loop](2026-07-08-self-play-training.md)

## Summary

The network's job: take the 9×9×6 board encoding and produce (a) a probability for each of the 1296 moves and (b) an estimate of "am I winning?". This repo only defines the *front half* — a convolutional feature extractor. The SB3 library automatically bolts the decision-making heads onto it.

## Why a CNN for a board game?

A **convolutional neural network (CNN)** scans small 3×3 windows across the board, the same little pattern-detector applied at every position. That matches board games well: "a piece flanked by two enemies" means the same thing in the corner as in the center, so it's wasteful to learn it separately for each square. Stacking conv layers lets later layers see combinations of earlier patterns — local tactics compose into board-wide structure.

## TaflCNN — the default

[agents/networks.py:7](../agents/networks.py#L7):

```
9×9×6 board → conv (64 filters) → conv (128) → conv (128) → flatten → linear → 256 numbers
```

Those **256 numbers are a learned summary of the position** — not human-interpretable, but containing whatever the network found useful for predicting wins. One mechanical detail: SB3 delivers boards as (height, width, channels), PyTorch convs want (channels, height, width), hence the `permute` in `forward()` ([line 40](../agents/networks.py#L40)).

## What SB3 adds on top

When `MaskablePPO` is constructed with this extractor ([training/self_play.py:155](../training/self_play.py#L155)), it appends two small heads reading those 256 features:

- **Policy head (the "actor")** → 1296 scores, masked to legal moves, softmaxed into move probabilities. *This chooses moves.*
- **Value head (the "critic")** → 1 number: expected final reward from here. *This judges positions* — its predictions are what make the training signal informative (see [RL basics](2026-07-08-rl-basics.md)).

Both heads share the same 256-feature summary, so improving position understanding helps both choosing and judging. Everything — conv filters and both heads — trains together by gradient descent during PPO updates.

## SharedTaflCNN — the Phase 3 variant

[agents/networks.py:46](../agents/networks.py#L46) is built for the shared-representation experiment: **one shared conv trunk, but two separate final layers** — one for playing attacker, one for defender. Each sample is routed by reading the side flag from observation channel 5 ([line 75](../agents/networks.py#L75)): flag = 1 → attacker head, 0 → defender head.

The hypothesis it tests: most Hnefatafl understanding (threats, captures, king safety) is side-agnostic and can live in the shared trunk, with only the "so what do *I* do about it" part being side-specific.

**Constraint:** it reads the side flag from channel 5, which only exists in `canonical` mode — it cannot work with `obs_mode=perspective` (channel 5 is corners there).

## Scale check

Three conv layers and two small heads — on the order of a few million parameters. Tiny by modern standards, which is why the frozen opponent copies run comfortably on CPU inside each environment worker (a deliberate choice explained in the [self-play note](2026-07-08-self-play-training.md)).
