import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Must set cwd before tafl-gym loads its ini files
_PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(_PROJECT_ROOT)

from gym_tafl.envs._game_engine import GameEngine
from gym_tafl.envs._utils import make_dictionaries, IDX_TO_POS
from gym_tafl.envs.configs import ATK, DEF, KING

from env.observations import encode_observation, BOARD_SIZE, N_CHANNELS

N_ACTIONS = 1296  # all possible (from, to) moves on a 9x9 board
make_dictionaries(BOARD_SIZE, BOARD_SIZE)  # populate IDX_TO_POS / POS_TO_IDX

OpponentFn = Callable[[np.ndarray, list], int]


class TaflEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper for Hnefatafl (Tablut variant).

    The learning agent controls `side` (ATK or DEF). The other side is driven
    by `opponent_fn(board, valid_actions) -> action_index`. Defaults to a
    uniformly random opponent.

    Supports action masking via `action_masks()` for use with MaskablePPO.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        side: int = ATK,
        opponent_fn: Optional[OpponentFn] = None,
        obs_mode: str = "canonical",
        variant: str = "tablut",
    ):
        super().__init__()
        self.side = side
        self.opponent_fn: OpponentFn = opponent_fn if opponent_fn is not None else random_opponent
        self.obs_mode = obs_mode
        self.variant = variant

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BOARD_SIZE, BOARD_SIZE, N_CHANNELS),
            dtype=np.float32,
        )

        # Build the engine once and reuse it across episodes: construction
        # re-reads variants/<variant>.ini from disk, and fill_board also
        # accumulates into engine.MAX_REWARD, so both must run exactly once.
        # The only per-episode engine state is no_capture_turns_counter,
        # which reset() clears.
        os.chdir(_PROJECT_ROOT)
        self._engine = GameEngine(self.variant)
        self._start_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self._engine.fill_board(self._start_board)

        self._board: Optional[np.ndarray] = None
        self._current_player: int = ATK
        self._valid_actions: list = []
        self._last_moves: list = []
        self._n_moves: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._board = self._start_board.copy()
        self._engine.no_capture_turns_counter = 0
        self._current_player = self._engine.STARTING_PLAYER
        self._valid_actions = self._engine.legal_moves(self._board, self._current_player)
        self._last_moves = []
        self._n_moves = 0
        self._done = False

        # If opponent moves first, run their turn(s)
        if self._current_player != self.side:
            done, _ = self._play_opponent()
            if done:
                obs = encode_observation(self._board, self.side, self.obs_mode)
                return obs, {"action_mask": self.action_masks()}

        obs = encode_observation(self._board, self.side, self.obs_mode)
        return obs, {"action_mask": self.action_masks()}

    def step(self, action: int):
        assert not self._done, "Episode is over — call reset() first"

        # Hard penalty for an invalid action (MaskablePPO should prevent this)
        if action not in self._valid_actions:
            self._done = True
            obs = encode_observation(self._board, self.side, self.obs_mode)
            return obs, -1.0, True, False, {
                "invalid_action": True,
                "action_mask": self.action_masks(),
                "final_board": self._board.copy(),
            }

        # Apply our move
        reward, terminated, info = self._apply_move(action, self._current_player)
        if terminated:
            self._done = True
            obs = encode_observation(self._board, self.side, self.obs_mode)
            info["action_mask"] = self.action_masks()
            info["final_board"] = self._board.copy()
            return obs, reward, True, False, info

        # Opponent plays until it's our turn again or game ends
        opp_done, opp_info = self._play_opponent()
        if opp_done:
            self._done = True
            obs = encode_observation(self._board, self.side, self.obs_mode)
            merged = {**info, **opp_info, "action_mask": self.action_masks(), "final_board": self._board.copy()}
            return obs, -1.0, True, False, merged

        obs = encode_observation(self._board, self.side, self.obs_mode)
        info["action_mask"] = self.action_masks()
        return obs, reward, False, False, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Action masking (for MaskablePPO / sb3-contrib)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[self._valid_actions] = True
        return mask

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def set_opponent(self, opponent_fn: OpponentFn) -> None:
        """Install a new opponent. Reachable through the wrapper stack via
        VecEnv.env_method("set_opponent", ...) — do not use VecEnv.set_attr,
        which only touches the outermost wrapper."""
        self.opponent_fn = opponent_fn

    def get_board(self) -> np.ndarray:
        return self._board.copy()

    def get_valid_actions(self) -> list:
        return list(self._valid_actions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_move(self, action: int, player: int):
        """Apply `action` for `player`. Returns (reward, terminated, info)."""
        move = IDX_TO_POS[action]
        res = self._engine.apply_move(self._board, move)
        reward = res["reward"]
        info = {"move": res["move"]}

        if res["game_over"]:
            # The player who just moved is the winner
            king_on_board = np.any(self._board == KING)
            if not king_on_board:
                info.update({"winner": ATK, "reason": "King captured"})
            else:
                info.update({"winner": DEF, "reason": "King escaped"})
            return reward, True, info

        end = self._engine.check_endgame(
            last_moves=self._last_moves,
            last_move=move,
            player=player,
            n_moves=self._n_moves,
        )
        reward += end["reward"]
        if end["game_over"]:
            info.update({"reason": end["reason"], "winner": end["winner"]})
            return reward, True, info

        # Advance state
        if len(self._last_moves) == 8:
            self._last_moves.pop(0)
        self._last_moves.append(move)
        self._n_moves += 1
        self._current_player = DEF if player == ATK else ATK
        self._valid_actions = self._engine.legal_moves(self._board, self._current_player)

        if not self._valid_actions:
            info["reason"] = "No moves available"
            return reward + 1.0, True, info

        return reward, False, info

    def _play_opponent(self):
        """Drive the opponent until it's our turn or game ends. Returns (done, info)."""
        while self._current_player != self.side:
            action = self.opponent_fn(self._board, self._valid_actions)
            _, terminated, info = self._apply_move(action, self._current_player)
            if terminated:
                return True, info
        return False, {}


def random_opponent(board: np.ndarray, valid_actions: list) -> int:
    return valid_actions[np.random.randint(len(valid_actions))]
