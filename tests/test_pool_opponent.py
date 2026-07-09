"""Opponent-pool tests: per-episode sampling, TaflEnv hooks, duel wiring.

The pool breaks self-play cycling by making each episode's opponent a random
draw from the rival's snapshot history (latest-biased) instead of always the
newest snapshot.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import shutil

import numpy as np

RUN = "_smoke_pool"
RUN_DIR = REPO / "checkpoints" / RUN


def _tiny_policy_opponent(side):
    from sb3_contrib import MaskablePPO
    from training.self_play import make_vec_env, PolicyOpponent
    env = make_vec_env(side=side, obs_mode="canonical", n_envs=1, vec_env="dummy")
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu")
    opp = PolicyOpponent(model.policy, side=side, obs_mode="canonical")
    env.close()
    return opp


def test_pool_sampling_is_latest_biased():
    from gym_tafl.envs.configs import DEF
    from training.duel import PoolOpponent

    pool = PoolOpponent(side=DEF, obs_mode="canonical", max_members=10, p_latest=0.8)
    base = _tiny_policy_opponent(DEF)
    for _ in range(3):  # 3 distinct member objects (shared weights are fine)
        pool.add_member(base.__class__(base.policy, side=DEF, obs_mode="canonical"))

    np.random.seed(0)
    picks = []
    for _ in range(2000):
        pool.new_episode()
        picks.append(pool._active)
    latest_share = sum(p is pool.members[-1] for p in picks) / len(picks)
    assert 0.75 < latest_share < 0.85, f"latest sampled {latest_share:.2f}, want ~0.8"
    older = set(id(p) for p in picks if p is not pool.members[-1])
    assert len(older) == 2, "both older members should be sampled sometimes"
    print("OK: pool samples latest ~80%, older members otherwise")


def test_pool_eviction_and_single_member():
    from gym_tafl.envs.configs import DEF
    from training.duel import PoolOpponent

    pool = PoolOpponent(side=DEF, obs_mode="canonical", max_members=2, p_latest=0.8)
    base = _tiny_policy_opponent(DEF)
    members = [base.__class__(base.policy, side=DEF, obs_mode="canonical") for _ in range(3)]
    for m in members:
        pool.add_member(m)
    assert len(pool.members) == 2, "max_members must cap the pool (FIFO)"
    assert pool.members[0] is members[1] and pool.members[1] is members[2]

    solo = PoolOpponent(side=DEF, obs_mode="canonical", max_members=5, p_latest=0.8)
    solo.add_member(members[0])
    for _ in range(20):
        solo.new_episode()
        assert solo._active is members[0]
    print("OK: FIFO eviction and single-member pool behave")


def test_env_reset_triggers_new_episode():
    from gym_tafl.envs.configs import ATK
    from env.tafl_wrapper import TaflEnv, random_opponent

    class CountingOpponent:
        def __init__(self):
            self.episodes = 0

        def new_episode(self):
            self.episodes += 1

        def __call__(self, board, valid_actions):
            return random_opponent(board, valid_actions)

    opp = CountingOpponent()
    env = TaflEnv(side=ATK, opponent_fn=opp)
    for _ in range(5):
        env.reset()
    assert opp.episodes == 5, f"reset must call new_episode (got {opp.episodes})"
    print("OK: TaflEnv.reset() re-samples pool opponents")


def test_add_opponent_snapshot_ducktypes():
    from gym_tafl.envs.configs import ATK, DEF
    from env.tafl_wrapper import TaflEnv
    from training.duel import PoolOpponent

    snap1 = _tiny_policy_opponent(DEF)
    snap2 = _tiny_policy_opponent(DEF)

    # plain opponent -> replaced outright
    env = TaflEnv(side=ATK, opponent_fn=snap1)
    env.add_opponent_snapshot(snap2)
    assert env.opponent_fn is snap2

    # pool opponent -> appended
    pool = PoolOpponent(side=DEF, obs_mode="canonical", max_members=5, p_latest=0.8)
    pool.add_member(snap1)
    env.set_opponent(pool)
    env.add_opponent_snapshot(snap2)
    assert env.opponent_fn is pool and len(pool.members) == 2
    print("OK: add_opponent_snapshot appends to pools, replaces plain opponents")


def test_duel_wires_pool_and_ent_coef():
    from gym_tafl.envs.configs import ATK, DEF
    from training.duel import duel_train, PoolOpponent
    from training.self_play import make_vec_env

    shutil.rmtree(RUN_DIR, ignore_errors=True)
    models = duel_train(
        total_timesteps_per_side=256, steps_per_phase=128,
        obs_mode="canonical", n_envs=2, vec_env="dummy",
        n_steps=64, batch_size=128, checkpoint_freq=0, plot_freq=0,
        run_name=RUN, verbose=0, ent_coef=0.02, pool_size=4, pool_p_latest=0.7,
    )
    assert models["atk"].ent_coef == 0.02, models["atk"].ent_coef
    assert models["def"].ent_coef == 0.02

    # duel installs one pool PER SUB-ENV and grows them per phase: verify the
    # machinery directly on a dummy vec env (duel's own envs are closed by
    # now). Per-env instances matter: a broadcast-shared pool would receive
    # each snapshot n_envs times in dummy mode.
    env = make_vec_env(side=ATK, obs_mode="canonical", n_envs=2, vec_env="dummy")
    seed = _tiny_policy_opponent(DEF)
    for idx in range(2):
        pool = PoolOpponent(side=DEF, obs_mode="canonical", max_members=4, p_latest=0.7)
        pool.add_member(seed)
        env.env_method("set_opponent", pool, indices=[idx])
    env.env_method("add_opponent_snapshot", _tiny_policy_opponent(DEF))
    live = env.get_attr("opponent_fn")
    assert all(isinstance(p, PoolOpponent) for p in live)
    assert live[0] is not live[1], "each sub-env must own its own pool"
    assert all(len(p.members) == 2 for p in live), [len(p.members) for p in live]
    env.close()
    print("OK: duel plumbs ent_coef and pools grow via env_method")


if __name__ == "__main__":
    try:
        test_pool_sampling_is_latest_biased()
        test_pool_eviction_and_single_member()
        test_env_reset_triggers_new_episode()
        test_add_opponent_snapshot_ducktypes()
        test_duel_wires_pool_and_ent_coef()
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    print("ALL OK")
