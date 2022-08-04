from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnv

from typing import Optional, Tuple, Dict, List
import time


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

        if self.number_agents is not None:
            for i in range(self.number_agents):
                agent_id = f"agent-{str(i)}"
                pass
                # self.agents



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



        # Shape: (number of agents (=number of agents in env x number of environments), 15, 15, 18)
        # The latter dimension is 6 (the number of frames being stacked before input to the network) x 3 (the number of channels)
        # I'll test this in a sec.
        print("Here are the obs shape: ")
        print(obs.shape)

        # Sample: [ -28. -327. -378. -442. -175. -381. -327.  -34.  -68. -318.] -> List[int]
        print("self.episode_returns: ")
        print(self.episode_returns)

        # [  0.  -1.  -1.   0.   0.   0. -50.   1.   0.  -1.] -> List[int]
        # -50 means the agent got hit, -1 means they zapped a beam, +1 means they ate an apple.
        print("\n Here are the rewards: ")
        print(rewards)

        # Essentially a bool for when the episode is done for each agent
        # [0 0 0 0 0 0 0 0 0 0] -> List[int] (0 means not done, 1 means done)
        print("\n Here are the dones: ")
        print(dones)

        # An empty tuple it seems.
        print("\n info key words:")
        print(self.info_keywords)


        # [{'terminal_observation': array([[[  0, 255,   0, ...,   0,   0,   0],
        #         [113,  75,  24, ..., 113,  75,  24],
        print("\n Here are the infos: ")
        print(infos)

        # During an episode
        # {}

        # At the end of an episode
        # {'r': -292.0, 'l': 1000, 't': 59.348197}
        print("episode info:")
        print(episode_info)


        # When done is False (during an episode
        #  [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

        # At the end of an episode (ie when done is True)
        # [{'terminal_observation': array([[[0, 255, 0, ..., 0, 0, 0],
        #                                  [113, 75, 24, ..., 113, 75, 24],
        #                                  [0, 255, 0, ..., 0, 0, 0],
        #                                  ...,
        # [  0,   0,   0, ...,   0,   0,   0]]], dtype=uint8), 'episode': {'r': -91.0, 'l': 1000, 't': 59.345771}}, {'terminal_observation': array([[[  0,   0,   0, ...,   0,   0,   0],
        print("\n Here are the new_infos: ")
        print(new_infos)

        return obs, rewards, dones, new_infos


def test_taking_metrics():
    """
    We will just use this to test that we are taking metrics, etc. properly.
    Using inputs such as we have above.
    :return:
    """