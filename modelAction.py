from stable_baselines3 import *
from stable_baselines3.common.evaluation import evaluate_policy
import joblib

env = joblib.load('./agents/DQN_truck_agen_1M_env.pkl') # for unknowm reasons, the environment must be saved after training and reloaded this way
model = DQN.load("./agents/DQN_truck_agen_1M",env=env)

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