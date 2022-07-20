import argparse

import gym
import supersuit as ss
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import time
from torch import nn
import wandb
from wandb.integration.sb3 import WandbCallback

from social_dilemmas.envs.pettingzoo_env import parallel_env
from config.configuration import Config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Available device is: ", DEVICE)


WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
wandb.login(key=WANDB_API_KEY)
EXPERIMENT_NAME = f"PPO_{time.strftime('%d_%m_%Y_%H%M%S')}"

# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
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

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main(args):

    wandb.init(project="sb3_train",
               name=EXPERIMENT_NAME,
               config=args,
               sync_tensorboard=True,
               save_code=True
    )

    env = parallel_env(
        max_cycles=args.rollout_len,
        env=args.env_name,
        num_agents=args.num_agents,
        use_collective_reward=args.use_collective_reward,
        inequity_averse_reward=args.inequity_averse_reward,
        alpha=args.alpha,
        beta=args.beta,
    )

    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    print("We made it")
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = "./results/sb3/cleanup_ppo_paramsharing"

    model = PPO(
        "CnnPolicy",
        env=env,
        learning_rate=lr,
        n_steps=args.rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=DEVICE
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=1000,
            model_save_path=f"models/{EXPERIMENT_NAME}",
            verbose=2
        )
    )

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    # args = parse_args()
    conf = Config()
    main(conf)
