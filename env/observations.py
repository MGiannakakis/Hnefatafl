import os
from pathlib import Path
import numpy as np

# Ensure cwd before tafl-gym constants load
os.chdir(Path(__file__).parent.parent)
from gym_tafl.envs.configs import ATK, KING, ATTACKER, DEFENDER, THRONE, CORNER

BOARD_SIZE = 9
N_CHANNELS = 6  # atk, def, king, throne, corner, side


def encode_canonical(board: np.ndarray, side: int) -> np.ndarray:
    """
    6-channel encoding with absolute piece labels.
    ch0: attacker positions
    ch1: defender positions
    ch2: king position
    ch3: throne squares
    ch4: corner squares
    ch5: 1.0 if side is ATK, 0.0 if DEF
    """
    obs = np.zeros((BOARD_SIZE, BOARD_SIZE, N_CHANNELS), dtype=np.float32)
    obs[:, :, 0] = board == ATTACKER
    obs[:, :, 1] = board == DEFENDER
    obs[:, :, 2] = board == KING
    obs[:, :, 3] = board == THRONE
    obs[:, :, 4] = board == CORNER
    obs[:, :, 5] = float(side == ATK)
    return obs


def encode_perspective(board: np.ndarray, side: int) -> np.ndarray:
    """
    6-channel perspective-aware encoding: always "own" vs "enemy".
    ch0: own pawns  (ATK pieces if side=ATK; DEF pieces if side=DEF)
    ch1: own king   (0 if side=ATK; KING positions if side=DEF)
    ch2: enemy pawns
    ch3: enemy king (KING positions if side=ATK; 0 if side=DEF)
    ch4: throne squares
    ch5: corner squares
    """
    obs = np.zeros((BOARD_SIZE, BOARD_SIZE, N_CHANNELS), dtype=np.float32)
    if side == ATK:
        obs[:, :, 0] = board == ATTACKER
        obs[:, :, 1] = 0.0
        obs[:, :, 2] = board == DEFENDER
        obs[:, :, 3] = board == KING
    else:
        obs[:, :, 0] = board == DEFENDER
        obs[:, :, 1] = board == KING
        obs[:, :, 2] = board == ATTACKER
        obs[:, :, 3] = 0.0
    obs[:, :, 4] = board == THRONE
    obs[:, :, 5] = board == CORNER
    return obs


def encode_observation(board: np.ndarray, side: int, mode: str = "canonical") -> np.ndarray:
    if mode == "perspective":
        return encode_perspective(board, side)
    return encode_canonical(board, side)
