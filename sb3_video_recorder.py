import numpy as np
import os

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from social_dilemmas.envs.cleanup import CleanupEnv
from utils import get_supersuit_parallelized_environment, get_parallelized_env

# Note to run this file by itself (when in learning dir) from the terminal, run `python -m learning.sb3_video_recorder`


class TestingVideoRecorder:
    def __init__(self):
        self.env = get_parallelized_env("cleanup")
        self.env.reset()
        print("env type before is: ", type(self.env))
        # self.env = VecVideoRecorder(self.env,
        #                             video_folder='./videos/',
        #                             record_video_trigger=lambda x: x == 0)
        print("env type after is: ", type(self.env))

    def print_attributes(self):
        """
        Agents are 100% an attribute of the environment, I might need to use another '.' in order to access them?

        I should print all the attributes of the environment to help me get a better idea
        """
        # Print the attributes of self.env
        print("Attributes of self.env:")
        for attr in dir(self.env):
            print(attr)

    def test_rollout(self):
        """
                I just need to test the rollout for a vectorized env with multiple agents first!

                Agents are 100% an attribute of the environment, I might need to use another '.' in order to access them?

                I should print all the attributes of the environment to help me get a better idea
        """
        # agent_ids = ["agent-" + str(agent_number) for agent_number in range(5)]
        # actions = {}
        print(self.env.action_space("agent-0"))
        n_act = self.env.action_space("agent-0").n
        actions = {agent: np.random.randint(n_act) for agent in self.env.agents}

        print("agents", self.env.agents)
        print("we made it dawg")


if __name__ == "__main__":
    print("we have entered")
    x = TestingVideoRecorder()
    x.print_attributes()
    x.test_rollout()
    # x.test_video_recorder()
