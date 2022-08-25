# I will move this file to the learning directory afterwards.

# There seem to be some fundamental differences between the two environment wrappers, they behave differently when
# put on top of one another

import numpy as np
import threading
import os
import psutil
from signal import SIGINT, signal

from utils.env_getter_utils import get_supersuit_parallelized_environment, get_parallelized_env


class CompareSupersuitAndPettingzoo:
    def __init__(self):
        self.pz_env = get_parallelized_env()
        self.ss_env = get_supersuit_parallelized_environment()
        print("ss_env.metadata: ", self.ss_env.metadata)
        # Clost the envs
        self.pz_env.close()
        self.ss_env.close()

    def print_types(self):
        print("pz_env type: ", type(self.pz_env))
        print("ss_env type: ", type(self.ss_env))

    def print_env_information(self):
        # These are just useful for debugging
        print("Num agents and possible agents", self.pz_env.num_agents, self.pz_env.possible_agents)

    def print_attributes(self):
        # Print the attributes of both envs
        print("Attributes of pz_env:")
        for attr in dir(self.pz_env):
            print(attr)
        print("\nAttributes of ss_env:")
        for attr in dir(self.ss_env):
            print(attr)

    def rollout_pz_env(self):
        MAX_CYCLES = 10
        self.pz_env.reset()
        n_act = self.pz_env.action_space("agent-0").n
        print("pz_env.agents: ", self.pz_env.__dict__)
        for _ in range(MAX_CYCLES * self.pz_env.num_agents):
            actions = {agent: np.random.randint(n_act) for agent in self.pz_env.agents}
            obs, rewards, dones, infos = self.pz_env.step(actions)
            self.pz_env.render()
            if not self.pz_env.agents:
                print("pz_env is empty")
                self.pz_env.reset()


    def step_pz(self, num_steps: int = 1):
        n_act = self.pz_env.action_space("agent-0").n
        print("pz_env.agents: ", self.pz_env.__dict__)
        for _ in range(num_steps * len(self.pz_env.possible_agents)):
            actions = {agent: np.random.randint(n_act) for agent in self.pz_env.possible_agents}
            obs, rewards, dones, infos = self.pz_env.step(actions)
            return obs, rewards, dones, infos

    def step_ss(self, num_steps: int = 1):
        for _ in range(num_steps * self.ss_env.num_envs):
            actions = [self.ss_env.action_space.sample() for _ in range(self.ss_env.num_envs)]
            obs, rewards, dones, infos = self.ss_env.step(actions)
            return obs, rewards, dones, infos

    def rollout_ss_env(self):
        # This page is useful: https://github.com/Farama-Foundation/SuperSuit/blob/master/test/test_vector/test_pettingzoo_to_vec.py
        MAX_CYCLES = 3
        self.ss_env.reset()
        n_act = self.ss_env.action_space
        # For some reason, the Markov env doesn't inherit the num_agents attribute from the parallelized env
        for _ in range(MAX_CYCLES * self.ss_env.num_envs):
            actions = [self.ss_env.action_space.sample() for _ in range(self.ss_env.num_envs)]
            obs, rewards, dones, infos = self.ss_env.step(actions)
            self.ss_env.render(mode='human')
            # Pause
            input("Press Enter to continue...")

            if not self.ss_env.num_envs:
                print("ss_env is empty")
                self.ss_env.reset()
        print("ss_env rollout complete")


def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


if __name__ == '__main__':
    signal(SIGINT, shutdown_handler)
    x = CompareSupersuitAndPettingzoo()
    # x.print_types()
    # x.print_attributes()
    # obs, rewards, dones, infos = x.step_pz(num_steps=1)
    # print("obs shape: ", obs.shape)
    # print("obs type: ", type(obs))
    # print("rewards shape: ", rewards.shape)
    # print("rewards type: ", type(rewards))
    # print("dones shape: ", dones.shape)
    # print("infos shape: ", infos.shape)
    #
    # obs, rewards, dones, infos = x.step_ss(num_steps=1)
    # print("obs shape: ", obs.shape)
    # print("obs type: ", type(obs))
    # print("rewards shape: ", rewards.shape)
    # print("rewards type: ", type(rewards))
    # print("dones shape: ", dones.shape)
    # print("infos shape: ", infos.shape)
    # x.rollout_ss_env()
    x.rollout_pz_env()


