import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, List, Any, Dict

import gym as gym
from gym.spaces import Discrete
from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from models.ray_lstm import ActorCriticLSTM
from models.model_utils import print_model_params


class CustomCNN_visionnet_hardcoded(TorchModelV2, torch.nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 ):
        # num_outputs = 9
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        torch.nn.Module.__init__(self)

        in_channels = 3
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=[3, 3],
            stride=1,
            padding='valid',
        )

        # Initialize two FCN layers here
        self.fc1 = nn.Linear(in_features=1014, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self._features = None

        self.logit_layer = nn.Linear(in_features=32, out_features=num_outputs)

        # Not certain whether this in_features here should be 1014 or 32... I will come back to this later.
        self.value_function_fc = nn.Linear(in_features=32, out_features=1)

        self.LSTM = ActorCriticLSTM(32, action_space, num_outputs, model_config, name)


    def forward(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType,
                ) -> (TensorType, List[TensorType]):
        # Feed through CNN here
        features = input_dict["obs"]["curr_obs"].permute(0, 3, 1, 2)
        features = self.conv(features)
        # flatten
        features = torch.flatten(features, start_dim=1)

        # feed through linear layers
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        # print("features shape and length", features.shape, len(features))  # >>> torch.Size([1, 32]) 1 (Check if this is the same in `forward_with_asserts`!)
        self._features = features
        logits = self.logit_layer(features)

        return logits, state

    def forward_rnn(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType,
                ) -> (TensorType, List[TensorType]):
        x = self.lstm.fowward_rnn(input_dict, state, seq_lens)
        return x


    def forward_with_asserts(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType] = None,
                ) -> (TensorType, List[TensorType]):
        # Feed through CNN here
        # I just needed to permute the input first
        features = input_dict["obs"]["curr_obs"].permute(0, 3, 1, 2)
        print("features shape", features.shape)
        features = self.conv(features)

        # print shape
        print("here is the shape of features after being put through the cnn", features.shape)

        # flatten
        features = torch.flatten(features, start_dim=1)
        print("flattened shape ", features.shape)

        # feed through linear layers
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        print("linear layer features shape (this is what is stored as self._features ", features.shape)

        # feed through logit layer
        logits = self.logit_layer(features)
        print("logits shape ", logits.shape) # >>> torch.Size([1, 9])
        assert len(logits) == 9, "ensure outputs is 9 (for Cleanup!)"

        # We don't resize here as Ray expects a List
        # logits = torch.squeeze(logits, dim=0)
        # print("squeezed features", logits.shape)  # >>> torch.Size([9])

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Returns the value function estimate for the current observations.
        :return:
        x (TensorType): A single value estimate for the current observations. Shape is [1]
        """
        x = self.value_function_fc(self._features)
       #  print("shape of x right now: ", x.shape, x)
        x = torch.reshape(x, [-1])
        # print("shape of x right nwo tho: ", x.shape, x)
        return x


def tes_cnn():

    # sample inputs...
    obs_space = {"obs":
                    {
                        "curr_obs": Box(0, 255, (15, 15, 3), np.uint8),
                        "other_agent_actions": Box(0, 9, (4,), np.uint8),
                        "prev_visible_agents": Box(0, 1, (4,), np.uint8),
                        "visible_agents": Box(0, 1, (4,), np.uint8)
                    }
                 }
    obs_space["obs"]["curr_obs"] = torch.zeros(1, 15, 15, 3)

    # This is also generally equal to the number of outputs I think
    act_space = Discrete(9)

    x = CustomCNN_visionnet_hardcoded(obs_space=obs_space, action_space=act_space, num_outputs=9, model_config={}, name="yo")

    # TODO: the code below needs to be cleaned up. Might be no harm to have a look at other testing code that is reasonably descriptive.

    # Testing `def forward()`
    # test_output = x(obs_space)
    test_output = x(obs_space)
    print("test output: ", test_output)
    print("output shape (needs to be [9])", len(test_output), len(test_output[0][0]))
    print("the above is len, this is shape ", test_output[0].shape)
    # The [0] is just when a Tuple is returned, the first element is the output
    assert len(test_output[0][0]) == 9, "ensure outputs is 9 (for Cleanup!)"
    # assert len(test_output) == 9, "ensure outputs is 9 (for Cleanup!). This is for forward_with_asserts()"

    # Testing `def value_function()`
    test_value = x.value_function()
    print("value shape (needs to be [1])", test_value.shape)
    assert test_value.shape == torch.Size([1]), "ensure value is 1 (for Cleanup!)"


if __name__ == '__main__':
    tes_cnn()
