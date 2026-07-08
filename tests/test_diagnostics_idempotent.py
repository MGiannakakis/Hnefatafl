"""DiagnosticsCallback must tolerate repeated learn() calls (duel mode):
_on_training_start fires once per learn(), but the server/template setup
must happen only once."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import shutil
import tempfile


def main():
    from sb3_contrib import MaskablePPO
    from gym_tafl.envs.configs import ATK
    from training.self_play import make_vec_env
    from training.diagnostics import DiagnosticsCallback

    tmp = Path(tempfile.mkdtemp(prefix="diag_idem_"))
    env = make_vec_env(side=ATK, obs_mode="canonical", n_envs=1, vec_env="dummy")
    # MlpPolicy: the extractor is irrelevant here and plain CnnPolicy rejects
    # float observations (the real pipeline always passes TaflCNN)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu")

    cb = DiagnosticsCallback(
        log_dir=tmp / "logs", out_dir=tmp / "diag",
        plot_freq=1000, dashboard_port=18787, verbose=0,
    )
    cb.init_callback(model)

    cb.on_training_start(locals_={}, globals_={})
    first_server = cb._httpd
    assert first_server is not None, "server should start on first call"

    cb.on_training_start(locals_={}, globals_={})
    assert cb._httpd is first_server, (
        "second _on_training_start must NOT start a new server"
    )

    first_server.shutdown()
    env.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK: DiagnosticsCallback._on_training_start is idempotent")


if __name__ == "__main__":
    main()
