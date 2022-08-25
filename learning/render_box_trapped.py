import supersuit as ss
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

from social_dilemmas.envs.box_trapped import BoxTrapped


def main():
    env = BoxTrapped()
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class="stable_baselines3")

    for _ in range (100):
        actions = {agent: np.random.randint(9) for agent in env.agents}
        print(actions)
        obs, rewards, dones, infos = env.step(actions)
        env.render()
    print("done")


if __name__ == '__main__':
    main()

