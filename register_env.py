# Register custom env
from ray.tune.registry import register_env
from env.flag_frenzy_env import FlagFrenzyEnv

def env_creator(env_config):
    return FlagFrenzyEnv()

register_env("FlagFrenzyEnv-v0", env_creator)
