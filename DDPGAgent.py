"""
Deterministic Deep Deterministic Policy Gradient (DDPG) Agent
Reference:

- [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)
"""

import numpy as np

from envs.truckParkingEnv import TruckParkingEnv

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from utils import get_agent_save_path

# refine the environment for DDPG
from gymnasium import spaces
class TruckParkingEnvBoxSpace(TruckParkingEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # parse the action space [moving direction, steering angle]
        self.action_space = spaces.Box(low=np.array([0.0,0.0]), high=np.array([1.0,2*self.numSteerAngle+1]), shape=(2,), dtype=np.float32)

    def step(self, action):
        print(action,type(action))
        _action_dict = {}
        # use hard coded 0.5 for binary action
        _action_dict ['move_direction'] = int(action[0])
        # use hard coded separation points for steering angle, map [0,1] to {0,1,2,...,2*numSteerAngle+1}
        _action_dict ['steer_angle'] = int(action[1])
        return super().step(_action_dict)

env = TruckParkingEnvBoxSpace(render_mode='human')
episode_over = False
total_reward = 0

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
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