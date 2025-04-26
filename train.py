# -----------------------------------------------------------
# Rllib PPO training script for FlagFrenzy
#
# Author : Sanjna Ravichandar
# Created: April 2025
# -----------------------------------------------------------
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import FlagFrenzyModel
from models.hybrid_action_dist import HybridActionDistribution

import os
import json

# Register model
ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

register_env("FlagFrenzyEnv-v0", env_creator)
ray.init(ignore_reinit_error=True)


from ray.rllib.algorithms.callbacks import DefaultCallbacks


class FlagFrenzyCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result["training_iteration"]
        if it <= 200:  # linear decay 0.01 -> 0.003
            new_coeff = 0.01 - 0.007 * (it / 200)
            algorithm.config["entropy_coeff"] = new_coeff

# Config for PPO
config = (
    PPOConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .rollouts(
        num_rollout_workers=1,
        rollout_fragment_length=900,
    )
    .training(
        model={
            "custom_model": "flag_frenzy_model",
            "custom_action_dist": "hybrid_action_dist",
        },
        gamma=0.995,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.01,
        vf_clip_param=10.0,
        grad_clip=0.5,
        lr=3e-5,
        train_batch_size=43200,
        sgd_minibatch_size=2048,
        num_sgd_iter=20,
    )
    .resources(num_gpus=0)
    .callbacks(FlagFrenzyCallbacks)
    .to_dict()
)

config["horizon"] = 900

results = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"training_iteration": 450},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1,
            num_to_keep=5),
        name="flag_frenzy_ppo",
        log_to_file=True),
    param_space=config,

).fit()

