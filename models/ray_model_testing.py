import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, List, Any, Dict

import gym as gym
from gym.spaces import Discrete
from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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

from models.model_utils import print_model_params


# This is basically a clone of the Torch VisionNet from RLLIB
class CustomCNN_rllib(TorchModelV2, torch.nn.Module):
    """
    Configuring this basic model... hopefully this will work on the GPU .
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = [[6, [3, 3], 1]]  # just hardcoding for now
        # filters = self.model_config["conv_filters"]
        print("filters", len(filters), filters)
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        # Note: This doesn't seem to be in the standard A3C Config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        (w, h, in_channels) = obs_space["curr_obs"].shape

        in_size = [w, h]

        # "conv_filters": [[6, [3, 3], 1]], -> (out_channels, kernels, stride)
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    SlimFC(
                        in_size=out_channels,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
                padding, _ = same_padding(in_size, [1, 1], [1, 1])
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    in_size = out_channels
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            SlimFC(
                                in_size=in_size,
                                out_size=out_size,
                                activation_fn=post_fcnet_activation
                                if i < len(post_fcnet_hiddens) - 1
                                else None,
                                initializer=normc_initializer(1.0),
                            )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv2d(
                        out_channels,
                        num_outputs,
                        [1, 1],
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

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

    def forward(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType,
                ) -> (TensorType, List[TensorType]):
        # New logic
        # Feed through CNN here
        # I just needed to permute the input first
        features = input_dict["obs"]["curr_obs"].permute(0, 3, 1, 2)
        # print("features shape", features.shape)
        features = self.conv(features)

        # print shape
        # print("shape of x right now: ", x)
        # print("here is the shape of features after being put through the cnn", features.shape)

        # flatten
        features = torch.flatten(features, start_dim=1)
        # print("flattened shape ", features.shape)

        # feed through linear layers
        features = F.relu(self.fc1(features))
        # TODO: note that this seems kinda wrong almost
        features = F.relu(self.fc2(features))
        self._features = features

        # print("linear layer features shape ", features.shape) # >>> torch.Size([1, 9])
        # print shape and reshape if neccessary

        # feed through logit layer
        logits = self.logit_layer(features)
        # print("logits shape ", logits.shape) # >>> torch.Size([1, 9])

        logits = torch.squeeze(logits, dim=0)
        # print("squeezed features", logits.shape)  # >>> torch.Size([9])

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
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

    # Testing `def forward()`
    test_output = x(obs_space)
    print("output shape (needs to be [9])", len(test_output))
    print("the above is len, this is shape ", test_output[0].shape)
    assert len(test_output[0]) == 9, "ensure outputs is 9 (for Cleanup!)"

    # Testing `def value_function()`
    test_value = x.value_function()
    print("value shape (needs to be [1])", test_value.shape)
    assert test_value.shape == torch.Size([1]), "ensure value is 1 (for Cleanup!)"


if __name__ == '__main__':
    tes_cnn()
