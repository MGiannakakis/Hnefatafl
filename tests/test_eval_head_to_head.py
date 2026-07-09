"""Head-to-head eval: mode=eval with eval.opponent_ckpt pits two
checkpoints against each other instead of a random opponent."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
RUN = "_smoke_h2h"
RUN_DIR = REPO / "checkpoints" / RUN
sys.path.insert(0, str(REPO))


def make_checkpoints():
    from training.duel import duel_train
    duel_train(
        total_timesteps_per_side=256, steps_per_phase=128,
        obs_mode="canonical", n_envs=2, vec_env="dummy",
        n_steps=64, batch_size=128, checkpoint_freq=0, plot_freq=0,
        run_name=RUN, verbose=0,
    )


def main():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    make_checkpoints()
    cmd = [
        str(PY), str(REPO / "experiments" / "run.py"),
        "mode=eval",
        f"eval.ckpt=checkpoints/{RUN}/atk/final_model",
        "eval.side=atk",
        f"eval.opponent_ckpt=checkpoints/{RUN}/def/final_model",
        "eval.n_episodes=10",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise AssertionError(f"head-to-head eval failed with code {proc.returncode}")
    assert "win_rate" in proc.stdout, "results table missing"
    assert f"checkpoints/{RUN}/def/final_model" in proc.stdout, (
        "output must label the opponent checkpoint")
    print("OK: head-to-head eval runs and labels both checkpoints")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
