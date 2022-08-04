import torch
import os
import datetime
print("Torch version:", torch.__version__)

from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import time

import wandb
from wandb.integration.sb3 import WandbCallback


WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
wandb.login(key=WANDB_API_KEY)

# This is just a simple basic file to help ensure whether pettingzoo + supersuit + sb3 is working


config = {
    "total_timesteps": 20000,
    "env_name": "pistonball_v6"
}

# Experiment name is time and date
experiment_name = f"PPO_{time.strftime('%d_%m_%Y_%H%M%S')}"

wandb.init(
    project="sb3-testing",
    name=experiment_name,
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True
)

env = pistonball_v6.parallel_env()
# env = Monitor(env)
print("1. Env type is: ", type(env))
env = ss.color_reduction_v0(env, mode='B')
print("2. Env type is: ", type(env))
env = ss.resize_v1(env, x_size=84, y_size=84)
print("3. Env type is: ", type(env))
env = ss.frame_stack_v1(env, 3)
print("4. Env type is: ", type(env))
env = ss.pettingzoo_env_to_vec_env_v1(env)
print("5. Env type is: ", type(env))
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
print("6. Env type is: ", type(env))


# Add VecMonitor to env to record stats
LOG_DIR = "./vec_monitor_logs/"
datetime_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, f"{datetime_filename}")

env = VecMonitor(env, filename=log_file_path)


model = PPO('CnnPolicy', env, verbose=3, n_steps=16, tensorboard_log=f"runs/{experiment_name}")
print("We made it")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=100,
        model_save_path=f"models/{experiment_name}",
        verbose=2
        )
)
print("We made it here")
print("Why is this code not finishing exectuion")