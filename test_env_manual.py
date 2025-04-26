# -----------------------------------------------------------
# Manual testing script for FlagFrenzy environment
# -----------------------------------------------------------
from env.flag_frenzy_env import FlagFrenzyEnv
import numpy as np

# Create the environment
env = FlagFrenzyEnv()

# Get initial observation
obs, info = env.reset()

# Run for 100 steps
total_reward = 0
for i in range(100):
    print(f"Step {i}")
    
    # Sample a random action
    action = env.action_space.sample()
    
    # Take a step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"  Reward: {reward:.4f}, Total reward: {total_reward:.4f}")
    
    if terminated or truncated:
        print("Episode finished!")
        break

print(f"Final total reward: {total_reward:.4f}")
env.close()