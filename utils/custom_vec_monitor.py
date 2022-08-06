import time
from typing import Optional, Tuple, Dict, List
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnv
from stable_baselines3 import PPO

from utils.sb3_custom_cnn import CustomCNN
from utils.env_getter_utils import get_supersuit_parallelized_environment
from config.configuration import Config


agent_dict = {"indivudal_rewards": [], "beam_fired": [], "beam_hit": [], "apples_consumed": []}


class CustomVecMonitor(VecMonitor):
    def __init__(self,
                 venv: VecEnv,
                 filename: Optional[str] = None,
                 info_keywords: Tuple[str, ...] = (),
                 number_agents: int = 5,
                 ):
        super().__init__(venv, filename, info_keywords)
        self.per_agent_returns: Dict[int, List[int]] = {}
        self.number_agents = number_agents
        self.agents: Dict[str, Dict[str, List[int]]] = {}

        if self.number_agents is not None:
            for i in range(self.number_agents):
                agent_id = f"agent-{str(i)}"
                # TODO: We might want to use a class here, then I could also use a .reset() method
                self.agents[agent_id] = agent_dict

    def print_venv_attributes(self):
        print("\n Here are the attributes of the venv: ")
        # Pretty print the __dict__ of the venv
        for key, value in self.venv.__dict__.items():
            print(f"{key}: {value}")

        print("\n Here are the attributes of the venv.env: ")
        for key, value in self.venv.env.__dict__.items():
            print(f"{key}: {value}")

        print("\n Here are the attributes of the venv.env.venv: ")
        for key, value in self.venv.env.venv.__dict__.items():
            print(f"{key}: {value}")

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        episode_info = {}
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
            else:
                # Individual agent metrics.
                # TODO: Note that this wouldn't include metrics from the last step!
                agent_id = i % self.number_agents
                self.agents[f"agent-{str(agent_id)}"]["indivudal_rewards"] += [rewards[i]]
                if rewards[i] == -1:
                    self.agents[f"agent-{str(agent_id)}"]["beam_fired"] += 1
                elif rewards[i] == 1:
                    self.agents[f"agent-{str(agent_id)}"]["apples_consumed"] += 1
                elif rewards[i] == -50:
                    self.agents[f"agent-{str(agent_id)}"]["beam_hit"] += 1

        return obs, rewards, dones, new_infos


def pseudo_step(rewards, dones, step_size=5) -> None:
    agents = {}
    print(rewards)  # TODO: why don't the rewards match the individual agent rewards.
    for i in range(5):
        agent_id = f"agent-{str(i)}"
        agents[agent_id] = agent_dict

    for i in range(step_size):
        for j in range(i, len(dones), step_size):
            print(f"{i} {j}")
            print(f"{rewards[j]} {dones[j]}")
            agent_id = f"agent-{str(i)}"
            if dones[j]:
                print(agent_id)
                print(agents[agent_id])

                # Individual agent metrics.
                # Note that this wouldn't include metrics from the last step!
                # agent_id = j % 5
                print(rewards)
                print("agents[agent_id]", agents[agent_id])
                agents[f"{agent_id}"]["indivudal_rewards"] += [rewards[i]]
                if rewards[i] == -1:
                    agents[f"{str(agent_id)}"]["beam_fired"] += [1]
                elif rewards[i] == 1:
                    agents[f"{str(agent_id)}"]["apples_consumed"] += [1]
                elif rewards[i] == -50:
                    agents[f"agent-{str(agent_id)}"]["beam_hit"] += [1]

    # print the agents dict
    for agent_id, _agent_dict in agents.items():
        print(f"\nAgent {agent_id}")
        for key, value in _agent_dict.items():
            print(f"{key}: {value}")

    print(agents)

def test_taking_metrics():
    """
    We will just use this to test that we are taking metrics, etc. properly.
    Using inputs such as we have above.
    :return:
    """
    sample_rewards = [0., -1.,  1.,  0., -1.,  0.,  -50., -1., 0, 0]
    dones = [1 for i in range(10)]

    pseudo_step(sample_rewards, dones)


def test_vec_monitor(train=True):
    """
    x.
    :return:
    """
    args = Config()
    env = get_supersuit_parallelized_environment(args)
    env = CustomVecMonitor(env, filename=args.log_file_path)
    env.print_venv_attributes()

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
        ).learn(total_timesteps=args.total_timesteps)


if __name__ == '__main__':
    # test_vec_monitor()
    test_taking_metrics()