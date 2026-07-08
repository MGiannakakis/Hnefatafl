import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TaflCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for the (9, 9, 6) Hnefatafl observation.

    Two conv layers with residual connections, followed by a linear projection.
    SB3 passes observations as (batch, H, W, C); we permute to (batch, C, H, W).
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[2]  # 6

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 9, 9)
            conv_out_dim = self.conv(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(conv_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (batch, H, W, C) -> (batch, C, H, W)
        x = obs.permute(0, 3, 1, 2)
        return self.proj(self.conv(x))


# Shared-trunk variant: one CNN trunk, two separate linear heads for ATK and DEF sides.
# Useful for Phase 3 (shared representation) experiments.
class SharedTaflCNN(BaseFeaturesExtractor):
    """
    Shared CNN trunk with side-specific projection heads.
    The side is encoded in channel 5 of the observation (canonical mode).
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[2]

        self.trunk = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 9, 9)
            trunk_out = self.trunk(dummy).shape[1]

        self.atk_head = nn.Linear(trunk_out, features_dim)
        self.def_head = nn.Linear(trunk_out, features_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.permute(0, 3, 1, 2)
        features = self.trunk(x)
        side = obs[:, 0, 0, 5]  # channel 5, any pixel — broadcast scalar per sample
        atk_mask = side.unsqueeze(1)
        out = atk_mask * self.relu(self.atk_head(features)) + \
              (1 - atk_mask) * self.relu(self.def_head(features))
        return out
