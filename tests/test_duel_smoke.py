"""Duel-mode smoke test (dummy vec envs, tiny budgets).

Checks: both models reach their step budget; snapshots are installed with
the opponent's OWN side; per-side final_model.zip files are written; the
try/finally saves both sides even when a phase is interrupted.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import shutil

RUN = "_smoke_duel"
RUN_DIR = REPO / "checkpoints" / RUN


def test_snapshot_side_encoding():
    from gym_tafl.envs.configs import ATK, DEF
    from sb3_contrib import MaskablePPO
    from training.self_play import make_vec_env, PolicyOpponent
    from training.duel import _snapshot_into

    atk_env = make_vec_env(side=ATK, obs_mode="canonical", n_envs=2, vec_env="dummy")
    # MlpPolicy: the extractor is irrelevant here and plain CnnPolicy rejects
    # float observations (the real pipeline always passes TaflCNN)
    def_model = MaskablePPO("MlpPolicy",
                            make_vec_env(side=DEF, obs_mode="canonical", n_envs=1, vec_env="dummy"),
                            n_steps=64, batch_size=64, device="cpu")

    _snapshot_into(atk_env, def_model, DEF, "canonical")
    opponents = atk_env.get_attr("opponent_fn")
    assert len(opponents) == 2
    for opp in opponents:
        assert isinstance(opp, PolicyOpponent), type(opp)
        assert opp.side == DEF, "DEF snapshot must encode obs with side=DEF"
        assert opp.obs_mode == "canonical"
    atk_env.close()
    def_model.env.close()
    print("OK: snapshots encode with the opponent's own side")


def test_duel_train_completes():
    from training.duel import duel_train

    shutil.rmtree(RUN_DIR, ignore_errors=True)
    models = duel_train(
        total_timesteps_per_side=512,
        steps_per_phase=256,
        obs_mode="canonical",
        n_envs=2,
        vec_env="dummy",
        n_steps=64,
        batch_size=128,
        checkpoint_freq=0,
        plot_freq=0,
        run_name=RUN,
        verbose=0,
    )
    assert models["atk"].num_timesteps >= 512, models["atk"].num_timesteps
    assert models["def"].num_timesteps >= 512, models["def"].num_timesteps
    assert (RUN_DIR / "atk" / "final_model.zip").exists()
    assert (RUN_DIR / "def" / "final_model.zip").exists()
    print("OK: duel_train trains both sides and saves both final models")


def test_interrupt_saves_both_sides():
    from sb3_contrib import MaskablePPO
    from training.duel import duel_train

    shutil.rmtree(RUN_DIR, ignore_errors=True)
    orig_learn = MaskablePPO.learn
    calls = {"n": 0}

    def interrupting_learn(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # die at the start of the first DEF phase
            raise KeyboardInterrupt
        return orig_learn(self, *args, **kwargs)

    MaskablePPO.learn = interrupting_learn
    try:
        duel_train(
            total_timesteps_per_side=100_000,  # would run long if not interrupted
            steps_per_phase=256,
            obs_mode="canonical",
            n_envs=2, vec_env="dummy", n_steps=64, batch_size=128,
            checkpoint_freq=0, plot_freq=0, run_name=RUN, verbose=0,
        )
        raise AssertionError("KeyboardInterrupt should have propagated")
    except KeyboardInterrupt:
        pass
    finally:
        MaskablePPO.learn = orig_learn

    assert (RUN_DIR / "atk" / "final_model.zip").exists(), "ATK not saved on interrupt"
    assert (RUN_DIR / "def" / "final_model.zip").exists(), "DEF not saved on interrupt"
    print("OK: interrupt mid-run still saves both final models")


if __name__ == "__main__":
    try:
        test_snapshot_side_encoding()
        test_duel_train_completes()
        test_interrupt_saves_both_sides()
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    print("ALL OK")
