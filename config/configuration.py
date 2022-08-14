from dataclasses import dataclass, field, asdict
from typing import Tuple
import os
import time



@dataclass
class Config:

    # TODO: I should comment the figures I typically use for diff things here!

    # Parser args
    env_name: str = "cleanup" # Choices: ["harvest", "cleanup"]
    algo_name: str = "A2C"
    num_agents: int = 2
    rollout_len: int = 1000
    total_timesteps: int = 2e8
    use_collective_reward: bool = False # Give each agent the collective reward across all agents
    inequity_averse_reward: bool = False # Use inequity averse rewards from 'Inequity aversion...' paper
    alpha: float = 5
    beta: float = 0.05

    num_cpus: int = 12  # 12 for colab with 5 agents;
    num_envs: int = 2  # Number of parallel multi-agent environments; 10 for colab with 5 agents;
    num_frames: int = 5  # Number of frames to stack together for input to the network; use >4 to avoid automatic VecTransposeImage

    features_dim: int= 128 # output layer of cnn extractor AND shared layer for policy and value functions
    fcnet_hiddens: Tuple[int, int] = (1024, 128)  # Two hidden layers for cnn extractor
    ent_coef: int = 0.001  # entropy coefficient in loss
    batch_size: int = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr: float = 0.0001
    n_epochs: int = 30
    gae_lambda: float = 1.0
    gamma: float = 0.99
    target_kl: float = 0.01
    grad_clip:int = 40
    verbose: int = 3

    # Wandb args
    wandb_mode = 'online' # Can be 'online', 'offline', or 'disabled'
    use_wandb: bool = True
    save_vid_every_n_steps: int = 50000
    vec_video_rollout_legnth: int = 1000 # How many steps to save to the video (default is 200, and total episode would be 1000)

    # Paths
    # TODO: note that this has to be run by a file from the home directory for it to work properly (otherwise it will
    #  create logs in a random dir.
    #  And I can't use absolute paths as we have to consider Colab... not sure if this is the most robust design choice.
    #  We will move on for now but I'll come back to this.
    log_dir: str = "./logs/vec_monitor_logs/"
    vid_dir: str = "./logs/vec_videos/"

    def __post_init__(self):
        self.wandb_experiment_name: str = f"{self.algo_name}_{self.env_name}_{self.num_agents}_AGENT(S)_{time.strftime('%d_%m_%Y_%H_%M_%S')}"
        self.datetime_filename = time.strftime('%d_%m_%Y_%H_%M_%S')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, f"{self.datetime_filename}")
        self.video_file_path = os.path.join(self.vid_dir, f"{self.datetime_filename}")


def test_config():
    print(Config)
    conf = Config()
    # print(conf)
    for i in asdict(conf).keys():
        print(f"{i}: {getattr(conf, i)}")


if __name__ == '__main__':
    test_config()
