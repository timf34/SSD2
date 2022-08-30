import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.a3c import A3CConfig, A3C
from ray.tune.logger import pretty_print
from ray.tune.integration.wandb import WandbLoggerCallback

from typing import Dict
import wandb

from social_dilemmas.envs.env_creator import get_env_creator

from config.configuration import Config
from custom_callback import CustomCallback

USE_TUNE = True

WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'


# wandb.login(key=WANDB_API_KEY)
# wandb.init(mode='online', project='Ray Testing')


def configure_model(config: Dict) -> Dict:
    """
    configures the model - this function is just to remove config code from the main training script
    :param config:
    :return:
    """
    config["model"] = {}
    # The final layer must be chosen specifically so that its output is [B, 1, 1, X]. See the explanation in
    # https://docs.ray.io/en/latest/rllib-models.html#built-in-models.
    # It is because rllib is unable to flatten to a vector otherwise.
    # TODO: check if this might be a source of error for running on the GPU
    config["model"] = {"conv_filters": [[6, [3, 3], 1]],
                       "conv_activation": "relu",
                       "fcnet_hiddens": [32, 32],
                       "use_lstm": True,
                       "lstm_use_prev_action": True,
                       "lstm_use_prev_reward": False,
                       "lstm_cell_size": 128}
    return config


def configure_config(env_creator, args, policy_graphs, policy_mapping_fn) -> Dict:
    config = A3CConfig().to_dict()

    # TODO: I should move this config to a specific config file/ class... can I inherit from A3CConfig as a @dataclass?

    # misc
    config["framework"] = "torch"
    config["env"] = args.env_name
    config["callbacks"] = CustomCallback
    config[
        "keep_per_episode_custom_metrics"] = False  # keep custom metrics + log pure metric, not mean, max, mean, etc.

    # hardware
    config["num_workers"] = 1  # args.num_cpus
    config["num_gpus"] = 0
    config["num_cpus_per_worker"] = 1
    config["num_gpus_per_worker"] = 0

    # information for replay - not sure why we need this
    config["env_config"]["func_create"] = env_creator
    config["env_config"]["env_name"] = args.env_name

    # configure model
    config = configure_model(config)

    # Multi-agents
    config["multiagent"] = {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn}

    # hyperparams
    config["horizon"] = 1000
    return config


def training_script(args: Config):
    env_creator = get_env_creator(env=args.env_name,
                                  num_agents=args.num_agents,
                                  use_collective_reward=args.use_collective_reward)

    register_env(args.env_name, env_creator)

    # Get the observation and action space sizes
    # Note: this could be tidier... do I really have to initialize the environment here?
    single_env = env_creator(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    # deleting (necessary?)
    del single_env

    # Agent policies initialized
    def policy_mapping_fn(agent_id):
        return agent_id

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": args.algo_name}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(args.num_agents):
        policy_graphs[f"agent-{str(i)}"] = gen_policy()

    config = configure_config(env_creator, args, policy_graphs, policy_mapping_fn)

    stop = {
        "training_iteration": 10000,
        "timesteps_total": 2e8
    }

    algo = A3C(env=args.env_name, config=config)

    print(config)

    ray.init(ignore_reinit_error=True, num_cpus=10)

    # ray.init(local_mode=True)
    # ray.init()
    tuner = ray.tune.Tuner \
            (
            trainable="A3C",
            param_space=config,
            run_config=air.RunConfig(name="testingA3C",
                                     stop=stop,
                                     callbacks=[
                                         WandbLoggerCallback(api_key=WANDB_API_KEY,
                                                             project="Ray Testing",
                                                             log_config=False,
                                                             save_checkpoints=True)
                                     ],
                                     local_dir="ray_results/cleanup",
                                     checkpoint_config=air.CheckpointConfig(num_to_keep=100,
                                                                            checkpoint_frequency=200,
                                                                            checkpoint_at_end=True)
                                     )
            )
    tuner.fit()


def main():
    args = Config()
    training_script(args)


if __name__ == "__main__":
    main()
