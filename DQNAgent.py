import numpy as np
from envs.truckParkingEnv import VeryVerySimpleTruckParkingEnvForDQN
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import joblib


# initialize environment
env = VeryVerySimpleTruckParkingEnvForDQN(render_mode='rgb_array')
env.setParams(reward_weights = np.array([0.01,1.5,2,0.5]),
              time_penalty=0.01,
              collisionReward=-100,
              successReward=100,
              maxSteps=200)
# train agent
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=3000,log_interval=200)
model.save('DQN_truck_agent') # save the trained agent as 'DQN_truck_agent'
joblib.dump(env,'DQN_simple_env.pkl')

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