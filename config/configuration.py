from dataclasses import dataclass, field, asdict
from typing import Tuple
import time



@dataclass
class Config:
    # Parser args
    env_name: str = "harvest" # Choices: ["harvest", "cleanup"]
    num_agents: int = 1
    rollout_len: int = 1000
    total_timesteps: int = 2e8
    use_collective_reward: bool = False # Give each agent the collective reward across all agents
    inequity_averse_reward: bool = False # Use inequity averse rewards from 'Inequity aversion...' paper
    alpha: float = 5
    beta: float = 0.05

    # Args from def main():
    num_cpus: int = 12
    num_envs: int = 8  # Number of parallel multi-agent environments
    num_frames: int = 8  # Number of frames to stack together for input to the network; use >4 to avoid automatic VecTransposeImage
    features_dim: int= 128  # output layer of cnn extractor AND shared layer for policy and value functions
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
    wandb_mode = 'disabled' # Can be 'online', 'offline', or 'disabled'
    use_wandb: bool = True
    save_vid_every_n_steps: int = 10000

    vec_video_rollout_legnth: int = 1000 # How many steps to save to the video (default is 200, and total episode would be 1000)

    def __post_init__(self):
        self.wandb_experiment_name: str = f"PPO_ONE_AGENT_HARVEST_{time.strftime('%d_%m_%Y_%H%M%S')}"


def test_config():
    print(Config)
    conf = Config()
    print(conf)


if __name__ == '__main__':
    test_config()
