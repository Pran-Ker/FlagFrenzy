# -----------------------------------------------------------
# Simplified Rllib PPO training script for FlagFrenzy
# -----------------------------------------------------------
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import FlagFrenzyModel
from models.hybrid_action_dist import HybridActionDistribution

# Register model
ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

# Initialize Ray
ray.init()

# Register environment
register_env("FlagFrenzyEnv-v0", env_creator)

# Create PPO config
config = {
    "env": "FlagFrenzyEnv-v0",
    "framework": "torch",
    "num_workers": 1,
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 3e-5,
    "gamma": 0.995,
    "lambda": 0.95,
    "model": {
        "custom_model": "flag_frenzy_model",
        "custom_action_dist": "hybrid_action_dist",
    },
    # Disable config validation
    "_validate_config": False
}

# Create PPO trainer
trainer = PPO(config=config)

# Train the model
for i in range(5):
    print(f"Training iteration {i}")
    result = trainer.train()
    print(f"Episode reward mean: {result['episode_reward_mean']}")

# Save the final checkpoint
checkpoint_path = trainer.save()
print(f"Checkpoint saved at: {checkpoint_path}")