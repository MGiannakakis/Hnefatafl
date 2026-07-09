"""Phase budgeting regression.

With reset_num_timesteps=False, SB3 treats learn(total_timesteps=N) as N
ADDITIONAL steps (it adds num_timesteps internally). Passing an absolute
target instead made every phase run ~2x all prior steps, so a run did only
~log2(budget/phase) phases and blew past the per-side budget. This asserts
the loop runs ~budget/steps_per_phase phases and stops near budget.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import shutil

RUN = "_smoke_phase"
RUN_DIR = REPO / "checkpoints" / RUN


def main():
    from sb3_contrib import MaskablePPO
    from training.duel import duel_train

    shutil.rmtree(RUN_DIR, ignore_errors=True)

    P = 64            # steps_per_phase
    B = 512           # per-side budget -> 8 phases/side when correct, ~4 when doubling
    ROLLOUT = 32 * 2  # n_steps * n_envs

    calls = {"n": 0}
    orig_learn = MaskablePPO.learn

    def counting_learn(self, *a, **k):
        calls["n"] += 1
        return orig_learn(self, *a, **k)

    MaskablePPO.learn = counting_learn
    try:
        models = duel_train(
            total_timesteps_per_side=B, steps_per_phase=P,
            obs_mode="canonical", n_envs=2, vec_env="dummy",
            n_steps=32, batch_size=64, checkpoint_freq=0, plot_freq=0,
            run_name=RUN, verbose=0, ent_coef=0.0, pool_size=20, pool_p_latest=0.8,
        )
    finally:
        MaskablePPO.learn = orig_learn

    atk = models["atk"].num_timesteps
    dfn = models["def"].num_timesteps
    # each side stops within one rollout of budget, NOT exponentially over
    assert B <= atk <= B + ROLLOUT, f"ATK overshot budget: {atk} (budget {B})"
    assert B <= dfn <= B + ROLLOUT, f"DEF overshot budget: {dfn} (budget {B})"
    # ~B/P phases per side (16 total), not ~log2(B/P) (~8)
    expected = 2 * (B // P)
    assert calls["n"] >= expected - 2, (
        f"too few phases: {calls['n']} learn() calls, expected ~{expected}")
    print(f"OK: {calls['n']} phases total (~{expected}), "
          f"ATK={atk} DEF={dfn}, budget {B}+/-{ROLLOUT}")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
