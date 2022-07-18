import torch
print("Torch version:", torch.__version__)

from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss


if __name__ == "__main__":

    env = pistonball_v6.parallel_env()
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
    model = PPO('CnnPolicy', env, verbose=3, n_steps=16)
    print("We made it")
    model.learn(total_timesteps=2000000)