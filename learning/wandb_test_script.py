import time
import datetime
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
wandb.login(key=WANDB_API_KEY)

config = {"policy_type": "MlpPolicy", "total_timesteps": 10000}
experiment_name = f"PPO_{int(time.time())}"

wandb.init(
    name=experiment_name,
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    return gym.make("CartPole-v1")

env = DummyVecEnv([make_env])

LOG_DIR = "./vec_monitor_logs/"
datetime_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, f"{datetime_filename}")

env = VecMonitor(env, filename=log_file_path)

# This doesn't seem to work when using WSL2. I should try again to see if I can get it to work (recording rollout vids would be helpful to ensure things are working)
# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)  # record videos


model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=100,
        model_save_path=f"models/{experiment_name}",
    ),
)