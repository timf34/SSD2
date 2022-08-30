from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing

import tensorflow as tf


class BaselineResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config):
        for policy in policies:
            policy.entropy_coeff_schedule.value = lambda _: new_config["entropy_coeff"]
            policy.config["entropy_coeff"] = new_config["entropy_coeff"]
            policy.lr_schedule.value = lambda _: new_config["lr"]
            policy.config["lr"] = new_config["lr"]

    def reset_config(self, new_config):
        self.reset_policies(self.optimizer.policies.values(), new_config)
        self.config = new_config
        return True


def kl_and_loss_stats(policy, train_batch):
    policy.explained_variance = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function())

    stats_fetches = {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": policy.explained_variance,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
    }

    return stats_fetches

# TODO: highly uncertain if this is correct. From here https://github.com/ray-project/ray/issues/8011
def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch