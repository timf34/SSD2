from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym as gym

import os
import datetime

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray import shutdown, tune
from ray.tune.registry import register_env
from torch import nn, Tensor


# TODO: not sure if this is working, and not sure what the best way is to test it, but going to move on to implementing
# the envs for now.


# Use this with lambda wrapper returning observations only
class CustomCNN(TorchModelV2, nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        num_outputs: int,
        model_config: dict,
        name: str = None,
        framework: str = None,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        super(CustomCNN, self).__init__(obs_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations, state, seq_lens) -> Tuple[Tensor, List[Any]]:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features, state