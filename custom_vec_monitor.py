from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnv

from typing import Optional, Tuple
import time


class CustomVecMonitor(VecMonitor):
    def __init__(self,
                 venv: VecEnv,
                 filename: Optional[str] = None,
                 info_keywords: Tuple[str, ...] = (),
                 ):
        super().__init__(venv, filename, info_keywords)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
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
        print("Here are the obs shape: ")
        print(obs.shape)

        print("\n Here are the rewards: ")
        print(rewards)

        print("\n Here are the dones: ")
        print(dones)

        print("\n Here are the new_infos: ")
        print(new_infos)

        print("\n Here are the infos: ")
        return obs, rewards, dones, new_infos