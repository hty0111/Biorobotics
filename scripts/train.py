"""
Author: HTY
Email: 1044213317@qq.com
Date: 2024-04-08 17:03
Description:
"""

import time
import setproctitle

import stable_baselines3 as sb3
import torch
from stable_baselines3.common.utils import set_random_seed

from config.config import get_train_config
from runner.runner import Runner


def train() -> None:
    # set configs
    args = get_train_config()
    set_random_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    if args.track:
        import wandb
        run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        tags = [*args.wandb_tags, f"v{sb3.__version__}"]
        run = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=tags,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs/{run_name}"

    # set process name
    setproctitle.setproctitle(str(args.env) + "-" + str(args.algo))

    # run experiments
    exp_manager = Runner(args)

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            # we need to save the loaded hyperparameters
            args.saved_hyperparams = saved_hyperparams
            assert run is not None  # make mypy happy
            run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


if __name__ == "__main__":
    train()
