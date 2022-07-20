from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class Config:
    # Parser args
    env_name: str = "harvest" # Choices: ["harvest", "cleanup"]
    num_agents: int = 5
    rollout_len: int = 1000
    total_timesteps: int = 5e8
    use_collective_reward: bool = False # Give each agent the collective reward across all agents
    inequity_averse_reward: bool = False # Use inequity averse rewards from 'Inequity aversion...' paper
    alpha: float = 5
    beta: float = 0.05

    # Args from def main():
    num_cpus: int = 4
    num_envs: int = 12  # Number of parallel multi-agent environments
    num_frames: int = 6  # Number of frames to stack together for input to the network; use >4 to avoid automatic VecTransposeImage
    feature_dim: int= 128  # output layer of cnn extractor AND shared layer for policy and value functions
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


def test_config():
    print(Config)
    conf = Config()
    print(conf)


if __name__ == '__main__':
    test_config()
