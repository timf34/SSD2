import supersuit as ss
import time

from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from config.configuration import Config as Config
from social_dilemmas.envs.pettingzoo_env import parallel_env


class RenderRolloutVecMonitor(VecMonitor):
    """
    This class inherits from VecMonitor, simply for the purpose of saving rollouts of episodes during training.
    """
    def __init__(self, env, filename):
        super(RenderRolloutVecMonitor, self).__init__(env, filename)


def get_parallelized_env(args: Config) -> parallel_env:
    if args is None:
        args = Config()
    return parallel_env(
        max_cycles=args.rollout_len,
        env=args.env_name,
        num_agents=args.num_agents,
        use_collective_reward=args.use_collective_reward,
        inequity_averse_reward=args.inequity_averse_reward,
        alpha=args.alpha,
        beta=args.beta,
    )


def get_supersuit_parallelized_environment(args: Config) -> SB3VecEnvWrapper:
    if args is None:
        args = Config()
    env = get_parallelized_env(args)
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, args.num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=args.num_envs, num_cpus=args.num_cpus, base_class="stable_baselines3"
    )
    return env


class Timer:
    def __init__(self):
        self.start = 0

    def begin(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start