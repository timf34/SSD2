import os
import datetime

import supersuit as ss
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback

from social_dilemmas.envs.pettingzoo_env import parallel_env
from config.configuration import Config
from utils.custom_vec_monitor import CustomVecMonitor
from utils.env_getter_utils import get_supersuit_parallelized_environment
from utils.sb3_custom_cnn import CustomCNN
from utils.wandb_vec_vid_recorder import WandbVecVideoRecorder
from utils.custom_vec_monitor import CustomVecMonitor, CustomCallback

from marl_baselines3 import IndependentPPO

SEED = 42

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Available device is: ", DEVICE)


WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
wandb.login(key=WANDB_API_KEY)


def get_algo(env, policy_kwargs, tensorboard_log, args, policy_model="CnnPolicy"):
    if args.algo_name == 'PPO':
        return PPO(
        policy=policy_model,
        env=env,
        learning_rate=args.lr,
        n_steps=args.rollout_len,
        batch_size=args.batch_size, # not this
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        max_grad_norm=args.grad_clip,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=args.verbose,
        seed=SEED,
        device=DEVICE,
    )
    elif args.algo_name == 'A2C':
        return A2C(
        policy=policy_model,
        env=env,
        learning_rate=args.lr,
        n_steps=args.rollout_len,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        max_grad_norm=args.grad_clip,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=args.verbose,
        seed=SEED,
        device=DEVICE
    )
    elif args.algo_name == 'IndependentPPO':
        return IndependentPPO(
        policy=policy_model,
        num_agents=args.num_agents,
        env=env,
        learning_rate=args.lr,
        n_steps=args.rollout_len,
        batch_size=args.batch_size, # not this
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        max_grad_norm=args.grad_clip,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=args.verbose,
        device=DEVICE,
    )
    else:
        raise ValueError(f"Unknown algo name: {args.algo_name}")


def main(args):

    wandb.init(
        project="sb3_train",
        name=args.wandb_experiment_name,
        config=args,
        mode=args.wandb_mode,
        sync_tensorboard=True,
        save_code=True,
        # monitor_gym=True,
    )

    env = get_supersuit_parallelized_environment(args)
    env = WandbVecVideoRecorder(env,
                                args.video_file_path,
                                record_video_trigger=lambda x: x % args.save_vid_every_n_steps == 0,
                                video_length=args.vec_video_rollout_legnth,
                                use_wandb=args.use_wandb,
                                )
    # This monitors/ logs the tb_results of our vectorized environment; we need to pass a filename/ directory to save to
    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_monitor.py
    env = CustomVecMonitor(env, number_agents=args.num_agents, filename=args.log_file_path, info_keywords=("x",))

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim, num_frames=args.num_frames, fcnet_hiddens=args.fcnet_hiddens
        ),
        net_arch=[args.features_dim],
    )

    tensorboard_log = f"./logs/tb_results/sb3/{args.env_name}_{args.algo_name}_paramsharing"

    model = get_algo(env=env, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, args=args)
    if args.algo_name != 'IndependentPPO':
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=WandbCallback(
                gradient_save_freq=1000, # TODO: I can probs get rid of this!
                model_save_freq=1000,
                model_save_path=f"logs/saved_model_logs/{args.wandb_experiment_name}",
                verbose=2
            )
        )
    else:
        model.learn(
            total_timesteps=args.total_timesteps,
    )

    logdir = model.logger.dir
    # TODO: This is too late to be saving the model if I want to do any sort of checkpointing; add a proper callback!
    model.save(f"{logdir}/model")
    del model
    model = PPO.load(f"{logdir}/model")

    # TODO: it doesn't seem that we are working. Chcek the printes


if __name__ == "__main__":
    # args = parse_args()
    conf = Config()
    main(conf)