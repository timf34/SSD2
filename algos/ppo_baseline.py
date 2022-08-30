from __future__ import absolute_import, division, print_function

from ray.rllib.algorithms.ppo.ppo import (
    # choose_policy_optimizer,
    # update_kl, # replaced by KLCoeffMixin https://github.com/ray-project/ray/blob/c01bb831d4fcf6066c8bd60f73999115b315148a/rllib/policy/tf_mixins.py#L153
    # There is actually also a UpdateKL class in the ppo file https://github.com/ray-project/ray/blob/436c89ba1a337fffcd6f46c78c4d5a5a3434d0a8/rllib/algorithms/ppo/ppo.py
    # which we have below
    warn_about_bad_reward_scales,
)

from ray.rllib.policy.tf_mixins import KLCoeffMixin

from ray.rllib.algorithms.dqn.dqn_tf_policy import clip_gradients

from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    ValueNetworkMixin,
    validate_config,
    # ppo_surrogate_loss,
    # setup_config,
    # setup_mixins,
    # vf_preds_fetches,
)

# TODO: check the below once I get back. 
# from ray.rllib.algorithms.trainer_template import build_trainer
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.tf_mixins import EntropyCoeffSchedule, LearningRateSchedule

from algos.common_funcs_baseline import BaselineResetConfigMixin, postprocess_ppo_gae, kl_and_loss_stats


def build_ppo_baseline_trainer(config):
    """
    Creates a PPO policy class, then creates a trainer with this policy.
    :param config: The configuration dictionary.
    :return: A new PPO trainer.
    """
    policy = build_tf_policy(
        name="PPOTFPolicy",
        get_default_config=lambda: config,
        # loss_fn=ppo_surrogate_loss,
        stats_fn=kl_and_loss_stats,
        # extra_action_fetches_fn=vf_preds_fetches,
        postprocess_fn=postprocess_ppo_gae,
        gradients_fn=clip_gradients,
        # before_init=setup_config,
        # before_loss_init=setup_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
    )

    ppo_trainer = Algorithm(
        name="BaselinePPOTrainer",
        # make_policy_optimizer=choose_policy_optimizer,
        default_policy=policy,
        default_config=config,
        validate_config=validate_config,
        # after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[BaselineResetConfigMixin],
    )
    return ppo_trainer