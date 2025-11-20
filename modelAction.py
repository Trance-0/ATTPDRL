from stable_baselines3 import *
from stable_baselines3.common.evaluation import evaluate_policy
import joblib
from PPOAgent import TruckParkingEnvBoxSpace

env = joblib.load('PPO_truck_agent_1M_env.pkl') # for unknowm reasons, the environment must be saved after training and reloaded this way
model = PPO.load("PPO_truck_agent_1M.zip",env=env)

# print learned rewards
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")

episode_over = False
total_reward = 0

env.render_mode = "human"
observation,_ = env.reset()
while not episode_over:
    action,_ = model.predict(observation,deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished; Total reward: {total_reward}")
env.close()  