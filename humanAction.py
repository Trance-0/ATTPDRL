from envs.truckParkingEnv import TruckParkingEnv
from utils import set_logger
import pygame

logger = set_logger("humanAction")

env = TruckParkingEnv(render_mode='human')
episode_over = False
total_reward = 0

# key binding section
left = pygame.K_a
right = pygame.K_d
forward = pygame.K_w
backward = pygame.K_s

# initial states
action = {'move_direction':1,'steer_angle':env.numSteerAngle}
logger.info(f"Initial action: {action}")
def update_action(action, key):
    if key[left]:
        action['steer_angle'] = min(2*env.numSteerAngle,action['steer_angle']+1)
        logger.debug(f"Steer angle left: {action['steer_angle']}")
    if key[right]:
        action['steer_angle'] = max(0,action['steer_angle']-1)
        logger.debug(f"Steer angle right: {action['steer_angle']}")
    if key[forward]:
        action['move_direction'] = min(1,action['move_direction']+1)
        logger.debug(f"Move direction forward: {action['move_direction']}")
    if key[backward]:
        action['move_direction'] = max(0,action['move_direction']-1)
        logger.debug(f"Move direction backward: {action['move_direction']}")

pygame.init()
running = True

while running and not episode_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys=pygame.key.get_pressed()
    if keys: logger.debug(f"Key pressed: W:{keys[forward]}, A:{keys[left]}, S:{keys[backward]}, D:{keys[right]}")
    update_action(action, keys)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

logger.info(f"Episode finished; Total reward: {total_reward}")
env.close()