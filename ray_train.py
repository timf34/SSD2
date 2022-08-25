from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.a2c import A2CConfig

from social_dilemmas.config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator

from config.configuration import Config


def build_config(args: Config):

    env_creator = get_env_creator(env=args.env_name,
                                  num_agents=args.num_agents,
                                  use_collective_reward=args.use_collective_reward)

    register_env(args.env_name, env_creator)

    # Get the observation and action space sizes - this could be cleaner and clearer. Shouldn't have to initialize an
    # environment just to get the observation and action space sizes.
    single_env = env_creator(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    config = A2CConfig()
    print(config.to_dict())


def main():
    args = Config()
    build_config(args)


if __name__ == "__main__":
    main()