import numpy as np
from envs.truckSteeringEnv import *
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import joblib

from utils import get_agent_save_path, set_logger

# parameter used to store the trained agent
TASK_NAME = "DQN_truck_agent_default_param_1M"
TOTAL_TIMESTEPS = 10**6


# initialize environment
env = TruckSteeringForwardEnv(render_mode='rgb_array')
# env.setParams(reward_weights = np.array([2,0.5,0.5,0.5]),
#               time_penalty=0.01,
#               collisionReward=-100,
#               successReward=100,
#               maxSteps=200)
# train agent
model = DQN("MlpPolicy", env, verbose=1,exploration_fraction=0.5,tensorboard_log='./tensorboard_logs',device='cuda')
model.learn(total_timesteps=TOTAL_TIMESTEPS,log_interval=4,progress_bar=True)
model.save(get_agent_save_path(TASK_NAME)) # save the trained agent as 'DQN_truck_agent'
joblib.dump(env,get_agent_save_path(f'{TASK_NAME}_env.pkl'))

# print learned rewards
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# visualize performance
episode_over = False
total_reward = 0

env.render_mode = "human"
observation,_ = env.reset()
while not episode_over:
    action,_ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished; Total reward: {total_reward}")
env.close()    