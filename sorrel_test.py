# Step 8: (Make sure that you did the previous steps in the sorrel.py file)
# cd D:\Coding\Sorrel\Code
#  poetry run python -m sorrel.examples.treasurehunt.main

import gym
from sorrel.envs.treasure_hunt.treasure_hunt import TreasureHuntEnv
from gym.envs.registration import register

# Register the custom environment
register(
    id='TreasureHunt-v0',
    entry_point='sorrel.envs.treasure_hunt.treasure_hunt:TreasureHuntEnv',
)

# Create and test the environment
env = gym.make("TreasureHunt-v0")
obs = env.reset()
print("Initial observation:", obs)

action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"Action: {action}, Reward: {reward}, Done: {done}")
