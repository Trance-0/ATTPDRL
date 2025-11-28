"""
The Proximal Policy Optimization

References:

- [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
"""

import numpy as np

from envs.truckSteeringEnv import *
from envs.truckParkingEnv import TruckParkingEnvDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import joblib

from utils import get_agent_save_path, set_logger

from collections import defaultdict

# parameter used to store the trained agent
TASK_NAME = "PPO_truck_agent_default_param_env_new_1M"
TOTAL_TIMESTEPS = 10**6

logger = set_logger(TASK_NAME)

if __name__ == "__main__":
    # env = TruckParkingEnvBoxSpace(render_mode='human')
    # env = TruckParkingEnvBoxSpace(render_mode='rgb_array',mode='simple')
    # env = TruckParkingEnvDiscrete(render_mode='rgb_array')
    env = TruckSteeringForwardEnv(render_mode='rgb_array')
    # env.setParams(reward_weights = np.array([0.01,1.5,2,0.5]),
    #             time_penalty=0.01,
    #             collisionReward=-100,
    #             successReward=100,
    #             maxSteps=200)
    episode_over = False
    total_reward = 0

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10, progress_bar=True)
    model.save(get_agent_save_path(TASK_NAME))
    joblib.dump(env,get_agent_save_path(f'{TASK_NAME}_env.pkl'))

    # print learned rewards
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    logger.info(f"Mean reward: {mean_reward} +/- {std_reward}")

    # visualize performance
    episode_over = False
    total_reward = 0
    observation,_ = env.reset()
    while not episode_over:
        action,_ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    logger.info(f"Episode finished; Total reward: {total_reward}")
    env.close()
