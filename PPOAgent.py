"""
The Proximal Policy Optimization

References:

- [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
"""

import numpy as np

from envs.parkingLot import EmptyParkingLot, ParkingLot
from envs.truckParkingEnv import TruckParkingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import joblib

from utils import get_agent_save_path, set_logger

from collections import defaultdict

# parameter used to store the trained agent
TASK_NAME = "PPO_truck_agent"

logger = set_logger(TASK_NAME)

# refine the environment for DDPG
from gymnasium import spaces
class TruckParkingEnvBoxSpace(TruckParkingEnv):
    
    # hyper parameters for environment
    mode='standard'
    action_distribution = defaultdict(int)
    log_counter = 0
    log_interval = 100

    def __init__(self, render_mode=None, simulation_speed=100, mode='standard'):
        super().__init__(render_mode=render_mode)
        # parse the action space [moving direction, steering angle]
        self.action_space = spaces.Box(low=np.array([-1.0,-self.numSteerAngle]), high=np.array([1.0,self.numSteerAngle]), shape=(2,), dtype=np.float32)
        logger.debug(f"refined action space: {self.action_space}")
        # accelerate game speed
        # self.set_simulation_speed(simulation_speed)
        if mode == 'simple':
            logger.debug("simple mode activated")
            # load simple case
            self.timePerStep = 0.1
            self.mode='simple'
            # target location and facing    
            self.c_star = np.array([20,6])
            self.theta_star = -np.pi/2
            self.c_tol = 0.5 # tolerance in meters of manhattan distance
            self.theta_tol = np.pi/30 # tolerance in radial
            self.parkingLot:ParkingLot = EmptyParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    # def set_simulation_speed(self, speed):
    #     self.speed = self.speed*speed
    #     self.timePerStep = self.timePerStep/speed
    #     self.h = self.h/speed
    #     if speed != 1:
    #         logger.debug(f"Simulation speed set to {speed}, h: {self.h}, timePerStep: {self.timePerStep}")
    #     # unlock fps
    #     self.metadata["render_fps"] = 1/self.h

    def step(self, action):
        _action_dict = {
            'move_direction' : 1 if action[0]>=0 else 0,
            'steer_angle':int(action[1]+self.numSteerAngle)
        }
        self.action_distribution[tuple(action)] += 1
        self.log_counter += 1
        # if self.log_counter%self.log_interval == 0:
        #     logger.debug(f"output action: {_action_dict}, original action: {action}, reward: {self._reward(_action_dict)}")
        return super().step(_action_dict)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)
        if self.mode == 'simple':
            self.c = np.array([20,7])
            self.theta = -np.pi/2
            self.prev_dx = 1 # default to start parking with forward direction
            self.prev_alpha = 0 # default to start parking with straight angle
            self.truck.reset()
            self.steps = 0
        obs = self._get_obs()

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}

if __name__ == "__main__":
    # env = TruckParkingEnvBoxSpace(render_mode='human')
    # env = TruckParkingEnvBoxSpace(render_mode='rgb_array',mode='simple')
    env = TruckParkingEnvBoxSpace(render_mode='rgb_array',mode='standard')
    env.setParams(reward_weights = np.array([0.01,1.5,2,0.5]),
                time_penalty=0.01,
                collisionReward=-100,
                successReward=100,
                maxSteps=200)
    episode_over = False
    total_reward = 0

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=30000, log_interval=10, progress_bar=True)
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
