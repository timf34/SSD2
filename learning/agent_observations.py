import numpy as np

from utils.env_getter_utils import get_supersuit_parallelized_environment
from utils.sb3_custom_cnn import CustomCNN
from config.configuration import Config

def adjust_and_return_config(config: Config = Config()) -> Config:
    """
    This is just to change some params of the config to help suit this file and the tests we want to run,
    rather than changing the actual thing
    :param config:
    :return:
    """
    config.num_envs = 1
    config.num_cpus = 1
    config.num_frames = 5  # TODO: look into this but for now use >4 to avoid automatic VecTransposeImage

    return config


def inspect_observations():
    """
    This function is a test to ensure that the apples show up on the agents observations, and are clearly defined.

    Ensuring that they don't get wiped away during the supersuit processing.
    :return:
    """
    config = adjust_and_return_config()
    env = get_supersuit_parallelized_environment()
    env.reset()


def learning_numpy():
    # Create a random numpy array of shape (10, 15, 15, 30)
    x = np.random.rand(10, 15, 15, 30)

    # Now slice a (1, 15, 15, 3) array out of it
    print(x[0, :, :, :3].shape)
    print(x[0][:][:][:3].shape)

    print(x.shape)

learning_numpy()