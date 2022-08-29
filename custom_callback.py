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


# WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
# wandb.init(name="Ray Testing")


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

        agent_ids = episode.get_agents()
        print("check agent ids", agent_ids)
        agent_ids = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"]  # hardcoding for now...
        print("checking again", agent_ids)  # TODO: not sure why but the get_agents() function isn't working.

        for agent_id in agent_ids:
            # implement logic here to log agent actions (which are in agent.py)
            # Note that I think permanent data gets stored in `episode.custom_metrics`
            # if not episode.custom_metrics[agent_id]: # Its per episode! So this doesn't matter. We can reinit it each time!
            # episode.custom_metrics[agent_id] = {"beam_fired" : [], "cleaning" : []}
            print("ptinting type", type(episode.user_data["agent_actions"][agent_id][0]))
            print("type dawg", type(episode.user_data["agent_actions"][agent_id].count(7)))

            # action specific metrics
            episode.custom_metrics[f"{agent_id}-beam_fired"] = float(
                episode.user_data["agent_actions"][agent_id].count(7))
            episode.custom_metrics[f"{agent_id}-cleaning"] = float(
                episode.user_data["agent_actions"][agent_id].count(8))

            # reward specific metrics
            episode.custom_metrics[f"{agent_id}-beam_fired"] = episode.user_data["agent_rewards"][agent_id].count(-1)
            episode.custom_metrics[f"{agent_id}-beam_hit"] = episode.user_data["agent_rewards"][agent_id].count(-50)
            episode.custom_metrics[f"{agent_id}-apples_consumed"] = episode.user_data["agent_rewards"][agent_id].count(1)

            # try:
                # This won't be accepted as I believe this callback is called before the WANDB callback which calls
                # wandb.init(). I have tried running wandb.init at the start of the `ray_train.py` script but it doenst
                # work. This is a decent workaround for now. I still don't know why the custom metrics won't log
                # when I set keep_per_episode_custom_metrics to True... I think it might be something in the way that
                # the logger output is nested perhaps
                # Yep so this doesn't work either:')
                # I might just use the mean, max, mins for now... or edit the source file direclty (I was hesitant to
                # do that as I'm not sure how easy it'll be to do that in AWS but we can give it a go. tmrw though, its
                # too late now.
                # wandb.log({"hio": 423})
                # TODO: this doesn't work either. Note that the notes/ state of the codebase is from an all nighter.
            # except Exception:
            #     pass

        print("here we are", agent_id, episode.user_data["agent_actions"][agent_id])
        print("and here", episode.custom_metrics)
