from typing import Dict, List, Optional, Union

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.sample_batch import SampleBatch


class CustomCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: MultiAgentEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ) -> None:
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print(f"episode {episode.episode_id} (env-idx={env_index}) started.")
        # agent_ids = episode.get_agents() # Ok so the agents aren't initialized here yet... this returns []
        agent_ids = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"] # hardcoding for now...
        print("here are our agent ids", agent_ids)
        episode.user_data["agent_actions"] = {}
        for i in agent_ids:
            print("agent id ", i)
            episode.user_data["agent_actions"][i] = []
            print("ep user data", episode.user_data["agent_actions"])

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: MultiAgentEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        agent_ids = episode.get_agents()

        for agent_id in agent_ids:
            print(agent_id, episode.last_action_for(agent_id))
            print("episode.user data", episode.user_data)
            # if agent_id not in episode.user_data["agent_actions"]:
            #     # I shouldn't have to do this but I seem to be getting a key error despite `on_episode_start` above
            #     episode.user_data["agent_actions"] = {agent_id: []}
            episode.user_data["agent_actions"][agent_id].append(episode.last_action_for(agent_id))

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: MultiAgentEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        agent_ids = episode.get_agents()
        for agent_id in agent_ids:
            # implement logic here to log agent actions (which are in agent.py)
            # Note that I think permanent data gets stored in `episode.custom_metrics`
            print("here we are", agent_id, episode.user_data["agent_actions"][agent_id])

