from functools import lru_cache

from gym.utils import EzPickle
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.env import ParallelEnv

from social_dilemmas.envs.env_creator import get_env_creator

MAX_CYCLES = 1000


def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)

# Note that this doesn't seem to get used in the training script!
def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    return parallel_to_aec_wrapper(parallel_env(max_cycles, **ssd_args))

# TODO: note that this is not used in the training script! I can probably remove it to clean up the code and any confusion
#  I think this was also the bulk of the errors I was previously getting to when running pytests
def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


# TODO: Following PEP, this should be in camelCase
class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        """
        From `gym`: https://github.com/openai/gym/blob/8e812e1de501ae359f16ce5bcd9a6f40048b342f/gym/core.py#L169
        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)
        """
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        return self.ssd_env.reset()

    def seed(self, seed=None):
        return self.ssd_env.reset(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        obss, rews, self.dones, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            self.dones = {agent: True for agent in self.agents}
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, infos

    def __str__(self):
        print("Printing methods for env:", env)
        print(self.__dict__)


class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"],
                "video.frames_per_second": 5}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)
