from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.a2c import A2CConfig, A2C
from ray.tune.logger import pretty_print

from social_dilemmas.config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator

from config.configuration import Config


def build_config(args: Config):

    env_creator = get_env_creator(env=args.env_name,
                                  num_agents=args.num_agents,
                                  use_collective_reward=args.use_collective_reward)

    register_env(args.env_name, env_creator)

    # Get the observation and action space sizes
    single_env = env_creator(args.num_agents)
    obs_space = single_env.observation_space
    # print("obs spcaes: ", obs_space)
    act_space = single_env.action_space
    # print("act spcaes: ", act_space)

    # Agent policies initialized
    def policy_mapping_fn(agent_id):
        return agent_id

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": args.algo_name}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(args.num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()

    config = A2CConfig()


    config.num_workers = 1  # args.num_cpus
    config.framework_str = "torch"
    print(config.to_dict())
    config = config.to_dict()

    # information for replay - not sure why we need this
    config["env_config"]["func_create"] = env_creator
    config["env_config"]["env_name"] = args.env_name

    # Set the num_gpus to 1
    config["num_gpus"] = 1

    # TODO :see why the conifg for the CNN from MeltingPot didn't work.

    # Setup for the neural network.
    config["model"] = {}
    # The strides of the first convolutional layer were chosen to perfectly line
    # up with the sprites, which are 8x8.
    # The final layer must be chosen specifically so that its output is
    # [B, 1, 1, X]. See the explanation in
    # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
    # because rllib is unable to flatten to a vector otherwise.
    # The a3c models used as baselines in the meltingpot paper were not run using
    # rllib, so they used a different configuration for the second convolutional
    # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
    config["model"]["conv_filters"] = [[6, [3, 3], 1]]
    config["model"]["conv_activation"] = "relu"
    config["model"]["fcnet_hiddens"] = [32, 32]
    config["model"]["use_lstm"] = True
    config["model"]["lstm_use_prev_action"] = True
    config["model"]["lstm_use_prev_reward"] = False
    config["model"]["lstm_cell_size"] = 128

    # hyperparams
    config["multiagent"] = {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn}
    config["horizon"] = 1000

    # Updated config file passed to the trainer.
    # This is equivalen to `get_trainer`
    algo = A2C(env=args.env_name, config=config)

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = algo.train()
        print(pretty_print(result))


def main():
    args = Config()
    build_config(args)


if __name__ == "__main__":
    main()