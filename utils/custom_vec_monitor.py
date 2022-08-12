import time
import sys
import wandb
import pandas as pd

from typing import Optional, Tuple, Dict, List
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO

from utils.sb3_custom_cnn import CustomCNN
from utils.env_getter_utils import get_supersuit_parallelized_environment
from config.configuration import Config

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
wandb.login(key=WANDB_API_KEY)
wandb.init(
        project="sb3_train_test",
        name="testing adding new variables",
        mode="online",
)


# TODO: note down this crazy silly error. This specific individual dict was being shared! I need to use objects here!
agent_dict = {"individual_rewards": [], "beam_fired": [], "beam_hit": [], "apples_consumed": []}


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.individual_rewards: List[float] = []
        self.beam_fired: List[int] = []
        self.beam_hit: List[int] = []
        self.apples_consumed: List[int] = []

    # The below two methods are for item assignment (calling and setting things like with a dict -> dict["key"] = value)
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def reset(self):
        self.individual_rewards = []
        self.beam_fired = []
        self.beam_hit = []
        self.apples_consumed = []


class CustomVecMonitor(VecMonitor):
    def __init__(self,
                 venv: VecEnv,
                 number_agents: int,
                 filename: Optional[str] = None,
                 info_keywords: Tuple[str, ...] = (),
                 use_wandb: bool = True,
                 ):
        super().__init__(venv, filename, info_keywords=info_keywords)
        self.per_agent_returns: Dict[int, List[int]] = {}
        self.number_agents = number_agents
        self.agents: Dict[str, Dict[str, List[int]]] = {}
        self.use_wandb = use_wandb

        # These "agents" are solely for storing information about the agents... not used at all!
        if self.number_agents is not None:
            for i in range(self.number_agents):
                agent_id = f"agent-{str(i)}"
                # TODO: We might want to use a class here, then I could also use a .reset() method
                # TODO: do I want to create a list of agents for every num_envs? Check what the current VecMonitor does
                self.agents[agent_id] = Agent(agent_id)

        print("self.agents: ", self.agents)

        print("Venv: ", self.venv)

    def print_venv_attributes(self):
        print("\n Here are the attributes of the venv: ")
        # Pretty print the __dict__ of the venv
        print(self.venv)
        for key, value in self.venv.__dict__.items():
            print(f"{key}: {value}")

        print("\n Here are the attributes of the venv.venv: ")
        print(self.venv.venv)
        for key, value in self.venv.venv.__dict__.items():
            # Skipping observations buffers keys because they are huge
            if key != "observations_buffers":
                print(f"{key}: {value}")


    def step_wait(self) -> VecEnvStepReturn:
        # TODO: note that everything done here is done every step!
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        # episode_info = {}
        for i in range(self.number_agents):
            for j in range(i, len(dones), self.number_agents):
                if dones[j]:
                    # TODO: note that everything done here is done only at the end of an episode!
                    info = infos[j].copy()
                    episode_return = self.episode_returns[j]

                    # TODO: note that I might want to leave some of these print statements here somehow to help
                    #  with debugging in the future.
                    # print(f"episode return no.{j}: {episode_return}")
                    # if j == 0:
                    #     print("episode_return: ", self.episode_returns)
                    episode_length = self.episode_lengths[j]
                    episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}

                    info["episode"] = episode_info
                    self.episode_count += 1
                    self.episode_returns[j] = 0
                    self.episode_lengths[j] = 0
                    if self.results_writer:
                        self.results_writer.write_row(episode_info)
                    new_infos[j] = info

                    agent_id = f"agent-{str(i)}"

                    if self.use_wandb is True and self.agents[agent_id]["individual_rewards"] != []:
                        # print("ehre now self.agents[agent_id][individual_rewards]: ", self.agents[agent_id]["individual_rewards"])

                        wandb.log({"x dude": 5})
                        # Note that num_envs is an attribute of one of the inherited classes (VecEnvWrapper)
                        wandb.log({f"{agent_id}_individual_rewards": sum(self.agents[agent_id]["individual_rewards"])/self.num_envs})
                        # TODO: this is where things are getting printed... sometimes with all 0's
                        # print(f"{agent_id}_individual_rewards: {sum(self.agents[agent_id]['individual_rewards'])}")
                        wandb.log({f"{agent_id}_beam_fired": sum(self.agents[agent_id]["beam_fired"])/self.num_envs})
                        wandb.log({f"{agent_id}_beam_hit": sum(self.agents[agent_id]["beam_hit"])/self.num_envs})
                        wandb.log({f"{agent_id}_apples_consumed": sum(self.agents[agent_id]["apples_consumed"])/self.num_envs})

                    self.agents[agent_id]["individual_rewards"] = []
                    self.agents[agent_id]["beam_fired"] = []
                    self.agents[agent_id]["apples_consumed"] = []
                    self.agents[f"{str(agent_id)}"]["beam_hit"] = []


                else:
                    # Individual agent metrics.
                    # TODO: Note that this wouldn't include metrics from the last step!
                    agent_id = f"agent-{str(i)}"
                    self.agents[agent_id]["individual_rewards"] += [rewards[j]]
                    if rewards[j] == -1:
                        self.agents[agent_id]["beam_fired"] += [1]
                    elif rewards[j] == 1:
                        self.agents[agent_id]["apples_consumed"] += [1]
                    elif rewards[j] == -50:
                        self.agents[f"{str(agent_id)}"]["beam_hit"] += [1]

        return obs, rewards, dones, new_infos


def pseudo_step(rewards, dones, step_size=5) -> None:
    agents = {}
    print(rewards)  # TODO: why don't the rewards match the individual agent rewards.
    num_agents = 5
    for i in range(num_agents):
        agent_id = f"agent-{str(i)}"
        agents[agent_id] = Agent(agent_id)

    for i in range(step_size):
        for j in range(i, len(dones), step_size):
            print(f"{i} {j}")
            print(f"rewards[i] and dones[j] {rewards[j]} {dones[j]}")
            agent_id = f"agent-{str(i)}"
            if dones[j]:
                print(agent_id)

                # Individual agent metrics.
                # Note that this wouldn't include metrics from the last step!
                # agent_id = j % 5
                print(rewards)
                print("agents[agent_id]", agents[agent_id])
                agents[f"{agent_id}"]["individual_rewards"] += [rewards[j]]
                if rewards[j] == -1:
                    print(agents[agent_id])
                    agents[f"{str(agent_id)}"]["beam_fired"] += [1]
                elif rewards[j] == 1:
                    agents[f"{str(agent_id)}"]["apples_consumed"] += [1]
                elif rewards[j] == -50:
                    agents[f"{str(agent_id)}"]["beam_hit"] += [1]


    # print the agents dict
    for agent_id, agent_object in agents.items():
        print(f"\nAgent {agent_object.__dict__}")

    print(agents["agent-0"]["individual_rewards"])

    if num_agents == 5:
        for agent_id, agent_object in agents.items():
            assert agents["agent-0"]["individual_rewards"] == [0.0, 0.0]
            assert agents["agent-1"]["individual_rewards"] == [-1.0, -50.0]
            assert agents["agent-2"]["individual_rewards"] == [1.0, -1.0]
            assert agents["agent-3"]["individual_rewards"] == [0, 0]
            assert agents["agent-4"]["individual_rewards"] == [-1.0, 0]

            assert agents["agent-0"]["beam_fired"] == []
            assert agents["agent-1"]["beam_fired"] == [1]
            assert agents["agent-2"]["beam_fired"] == [1]

            assert agents["agent-0"]["beam_hit"] == []
            assert agents["agent-1"]["beam_hit"] == [1]

            assert agents["agent-0"]["apples_consumed"] == []
            assert agents["agent-1"]["apples_consumed"] == []
            assert agents["agent-2"]["apples_consumed"] == [1]

    print(agents)


def lets_tests_taking_metrics() -> None:
    """
    We will just use this to test that we are taking metrics, etc. properly.
    Using inputs such as we have above.
    :return:
    """
    sample_rewards = [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, -50.0, -1.0, 0.0, 0.0]
    dones = [1 for _ in range(10)]
    pseudo_step(sample_rewards, dones)


class CustomCallback(WandbCallback):
    def __init__( self,
        num_agents: int,
        num_envs: int,
        log_dir: str = "./logs/vec_monitor_logs",
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        super(CustomCallback, self).__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)
        self.log_dir = log_dir
        self.num_rows = num_agents * num_envs

    def _on_step(self) -> bool:
        # My code:
        value: int = 5
        self.logger.record('random_value', value)

        # TODO: do note that this gets galled every single step, and not just at the end of each episode.
        results = load_results(self.log_dir)
        if not results.empty:
            # print("Here are the last num agents x num envs results:")
            # print(results.x.values[-(self.num_rows):])
            self.logger.record('last_results', results.x.values[-(self.num_rows):])

        # Wandb code: https://github.com/wandb/wandb/blob/584e2efeeaf9f894b4f0984a40c61efa9b6e3104/wandb/integration/sb3/sb3.py#L134
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True


def lets_tests_vec_monitor(train=True):
    """
    x.
    :return:
    """
    args = Config()
    env = get_supersuit_parallelized_environment(args)
    print("env:", env)
    print("env.env:", env.venv)
    env = CustomVecMonitor(env, number_agents=args.num_agents, filename=args.log_file_path, info_keywords=("x",))
    # env = CustomVecMonitor(env, filename=args.log_file_path)
    # env.print_venv_attributes()

    if train:
        tensorboard_log = f"./logs/tb_results/sb3/{args.env_name}_ppo_paramsharing"

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=args.features_dim, num_frames=args.num_frames, fcnet_hiddens=args.fcnet_hiddens),
            net_arch=[args.features_dim])

        model = PPO(
            "CnnPolicy", env=env,learning_rate=args.lr, n_steps=args.rollout_len, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
            max_grad_norm=args.grad_clip, target_kl=args.target_kl, policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log, verbose=args.verbose, device='cuda'
        ).learn(
            total_timesteps=100000,
            # callback=CustomCallback(
            #     num_agents=args.num_agents,
            #     num_envs=args.num_envs,
            #     verbose=2,
            #     model_save_path=f"logs/saved_model_logs/testing",
            #     model_save_freq=1000,
            #     gradient_save_freq=1000,  # TODO: I can probs get rid of this!
            #     ))
        )


if __name__ == '__main__':
    # test_vec_monitor()
    # lets_tests_taking_metrics()
    lets_tests_vec_monitor()
