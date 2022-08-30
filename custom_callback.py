from typing import Dict, List, Optional, Union
import wandb
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
        # agent_ids = episode.get_agents() # Ok so the agents aren't initialized here yet... this returns []
        agent_ids = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"]  # hardcoding for now...
        episode.user_data["agent_actions"] = {}
        episode.user_data["agent_rewards"] = {}

        for i in agent_ids:
            print("agent id ", i)
            episode.user_data["agent_actions"][i] = []
            episode.user_data["agent_rewards"][i] = []

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

        # Note: `user_data` is temp. `custom_metrics` is permanent + gets logged

        for agent_id in agent_ids:
            episode.user_data["agent_actions"][agent_id].append(episode.last_action_for(agent_id))
            episode.user_data["agent_rewards"][agent_id].append(episode.last_reward_for(agent_id))


    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: MultiAgentEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
    ) -> None:

        # agent_ids = episode.get_agents()
        # harcoding for now, the above function isn't working.
        agent_ids = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"]

        for agent_id in agent_ids:
            # action specific metrics
            episode.custom_metrics[f"{agent_id}-beam_fired"] = float(
                episode.user_data["agent_actions"][agent_id].count(7))
            episode.custom_metrics[f"{agent_id}-cleaning"] = float(
                episode.user_data["agent_actions"][agent_id].count(8))

            # reward specific metrics
            episode.custom_metrics[f"{agent_id}-firing_beam_fired"] = episode.user_data["agent_rewards"][agent_id].count(-1)
            episode.custom_metrics[f"{agent_id}-beam_hit"] = episode.user_data["agent_rewards"][agent_id].count(-50)
            episode.custom_metrics[f"{agent_id}-apples_consumed"] = episode.user_data["agent_rewards"][agent_id].count(1)

