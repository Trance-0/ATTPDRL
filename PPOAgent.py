"""
The Proximal Policy Optimization

References:

- [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
"""


import numpy as np

from envs.truckParkingEnv import TruckParkingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from utils import get_agent_save_path

env = TruckParkingEnv(render_mode='human')
# TODO: refine action spaces

# AssertionError: The algorithm only supports (<class 'gymnasium.spaces.box.Box'>, <class 'gymnasium.spaces.discrete.Discrete'>, <class 'gymnasium.spaces.multi_discrete.MultiDiscrete'>, <class 'gymnasium.spaces.multi_binary.MultiBinary'>) as action spaces but Dict('move_direction': Discrete(2), 'steer_angle': Discrete(7)) was provided

episode_over = False
total_reward = 0

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save(get_agent_save_path('DDPG_truck_agent'))

# print learned rewards
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# visualize performance
episode_over = False
total_reward = 0
observation,_ = env.reset()
while not episode_over:
    action,_ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished; Total reward: {total_reward}")
env.close()    
