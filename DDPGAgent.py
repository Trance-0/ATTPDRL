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
import joblib

from utils import get_agent_save_path, set_logger

TASK_NAME = "DDPG_truck_agent"

logger = set_logger(TASK_NAME)

# refine the environment for DDPG
from PPOAgent import TruckParkingEnvBoxSpace

if __name__ == "__main__":
    env = TruckParkingEnvBoxSpace(render_mode='rgb_array',mode='standard')
    # env = TruckParkingEnvBoxSpace(render_mode='human')
    episode_over = False
    total_reward = 0

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, device='cuda')
    model.learn(total_timesteps=30000, log_interval=200, progress_bar=True)
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