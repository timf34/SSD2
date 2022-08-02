import numpy as np
import os
import sys

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from social_dilemmas.envs.cleanup import CleanupEnv
from utils import get_supersuit_parallelized_environment, get_parallelized_env

# Note to run this file by itself (when in learning dir) from the terminal, run `python -m learning.sb3_video_recorder`


class TestingVideoRecorder:
    def __init__(self):
        self.env = get_supersuit_parallelized_environment("cleanup")
        self.env.reset()
        self.env = VecVideoRecorder(self.env, video_folder='./videos/', record_video_trigger=lambda x: x == 0)

    def print_attributes(self):
        print("Attributes of self.env:")
        for attr in dir(self.env):
            print(attr)

    def rollout(self) -> None:
        MAX_CYCLES = 100
        self.env.reset()
        n_act = self.env.action_space
        # For some reason, the Markov env doesn't inherit the num_agents attribute from the parallelized env
        for _ in range(MAX_CYCLES * self.env.num_envs):
            actions = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
            _, _, _, _ = self.env.step(actions)
            # self.env.render(mode='human')  # Uncomment this to display to the screen
        self.env.close()
        print("ss_env rollout complete")


if __name__ == "__main__":
    print("we have entered")
    x = TestingVideoRecorder()
    # x.print_attributes()
    x.rollout()

    print("we are going to try and exit... it doesnt seem to exit")
    sys.exit(0)
    print("we have exited")
