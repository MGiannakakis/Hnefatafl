"""End-to-end duel run through the Hydra CLI with SubprocVecEnv.
Asserts exit code 0 and the per-side artifact layout from the spec."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
RUN = "_smoke_duel_cli"
RUN_DIR = REPO / "checkpoints" / RUN


def main():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    cmd = [
        str(PY), str(REPO / "experiments" / "run.py"),
        "mode=duel",
        f"duel.run_name={RUN}",
        "duel.total_timesteps_per_side=2048",
        "duel.steps_per_phase=1024",
        "duel.n_envs=4",
        "duel.vec_env=subproc",
        "duel.n_steps=64",
        "duel.batch_size=256",
        "duel.checkpoint_freq=1024",
        "duel.plot_freq=1024",
        "duel.dashboard_port=18790",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          timeout=900)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise AssertionError(f"CLI duel run failed with code {proc.returncode}")

    for name in ("atk", "def"):
        side = RUN_DIR / name
        assert (side / "final_model.zip").exists(), f"missing {name} final model"
        assert any(side.glob("model_*_steps.zip")), f"missing {name} interim checkpoints"
        assert (side / "logs" / "progress.csv").exists(), f"missing {name} csv log"
        assert (side / "diagnostics" / "dashboard.html").exists(), f"missing {name} dashboard"
    assert "[duel] Phase 0: ATK" in proc.stdout, "phase log line missing"
    assert "[duel] Phase 1: DEF" in proc.stdout, "alternation to DEF missing"
    print("OK: CLI duel run produced the full per-side artifact layout")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
