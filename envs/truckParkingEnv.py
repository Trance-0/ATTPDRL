import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from envs.parkingLot import *
from envs.truck import *

class TruckParkingEnv(gym.Env):
    
    # dimensions of the parking lot, meter
    xmax = 40
    ymax = 40
    parkingLotParams = dict()

    # truck control
    numSteerAngle = 3 # the actual number of steering angles will be 2*this+1
    maxSteerAngle = np.pi/6
    truckParams = {'maxTrailer':np.pi*2/3}# the maximum angle between trailer and driver

    # running parameters
    speed = 1 # constant speed, meter/s
    timePerStep = 1 # truck run for this time (seconds) before next state
    h = 0.01 # time step to run the truck (differentiated path)
    maxSteps = 200 # max number of steps per trajectory
    collisionReward = -100
    successReward = 100

    # target location and facing 
    c_star = np.array([20,11])
    theta_star = np.pi/2
    c_tol = 0.1 # tolerance in meters of manhattan distance
    theta_tol = np.pi/60 # tolerance in radial

    # reward function parameter
    time_penalty = 0.01
    reward_weights = np.array([1,0.5,0.25,0])

    # for plotting
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": max(1/h,1000)}
    window_size = 1000


    def __init__(self, render_mode=None):
        super().__init__()

        # define the parking lot
        self.parkingLot:ParkingLot = VerticalParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

        # define the truck
        self.truck:Truck = TrailerTruck(self.maxSteerAngle,**self.truckParams)

        # dx, alpha
        # dx==0 -> backing; dx==1 -> forwarding
        _action_dict = {
            'move_direction':spaces.Discrete(2),
            'steer_angle':spaces.Discrete(2*self.numSteerAngle+1)
        }
        self.action_space = spaces.Dict(_action_dict)

        # c, theta, prev_dx, prev_alpha, (possible) extra observation from the truck
        _observe_dict = {
            'location':spaces.Box(np.array([0,0]),np.array([self.xmax,self.ymax]),dtype=float),
            'facing':spaces.Box(-np.pi,np.pi,dtype=float),
            'prev_direction':spaces.Discrete(2),
            'prev_steer_angle':spaces.Discrete(2*self.numSteerAngle+1)
        }
        _observe_dict.update(self.truck.getObsSpace())
        self.observation_space = spaces.Dict(_observe_dict)

        obs,_ = self.reset()
        print('Environment initialized with state:\n',obs)

        # for plotting
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def __str__(self):
        return super().__str__()+' truckParkingEnv'
    
    def setParams(self,reward_weights:np.ndarray=None,time_penalty:float=None,collisionReward:float=None,successReward:float=None,maxSteps:int=None):
        if reward_weights is not None:
            self.reward_weights = reward_weights
        if time_penalty is not None:
            self.time_penalty = time_penalty
        if collisionReward is not None:
            self.collisionReward = collisionReward
        if successReward is not None:
            self.successReward = successReward
        if maxSteps is not None:
            self.maxSteps = maxSteps

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        # reset the env to a valid random state
        valid = False
        while not valid:
            self.c = np.array([self.np_random.uniform(0,self.xmax),self.np_random.uniform(0,self.ymax)])
            self.theta = self.np_random.uniform(-np.pi, np.pi)
            self.prev_dx = 1 # default to start parking with forward direction
            self.prev_alpha = 0 # default to start parking with straight angle
            self.truck.reset()

            # ensure target position is valid
            x0s,x1s,x2s = self.truck.getShapes(self.c_star,self.theta_star)
            assert not self.parkingLot.isCollision(x0s,x1s,x2s)

            self.steps = 0
            valid = not self._isTerminal()[0]
        obs = self._get_obs()
        self.prev_obs = obs
        self.d_c0 = np.linalg.norm(self.c-self.c_star)

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
    
    def _get_obs(self) -> dict:
        obs = {
            'location':np.array(self.c,dtype=np.float32),# already np array
            'facing':np.array([self.theta],dtype=np.float32),
            'prev_direction':int((self.prev_dx+1)/2),
            'prev_steer_angle':int((self.prev_alpha+self.maxSteerAngle)/(self.maxSteerAngle/self.numSteerAngle))
        }
        obs.update(self.truck.getObs())
        return obs

    # Generat step function, action ==  array([dx,alpha])
    def step(self, action:dict) -> tuple[dict,float,bool,bool,dict]:
        # verify action validity
        # assert action['move_direction'] in [0,1] and action['steer_angle'] in range(2*self.numSteerAngle+1)

        # reward is called on s,a and previous obs
        reward = self._reward(action)
        # update self to s' from s,a
        self.prev_obs = self._get_obs() # update previous obs to be the current obs
        halfway_done,halfway_reward = self._transition(action)
        obs = self._get_obs()
        # isTerminal is called on s'
        done,extraReward = self._isTerminal()
        reward += extraReward
        if halfway_done:
            done = True
            reward += halfway_reward
        self.steps += 1

        # for plotting
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, False, {}
    
    # R(s,a) with s being self and a == array([dx,alpha])
    def _reward(self,action:dict) -> float:
        d_c = np.linalg.norm(self.c-self.c_star)
        d_theta = abs(self.theta-self.theta_star)
        rew_dx = -int(self.prev_dx != action['move_direction'])
        rew_alpha = -abs(self.prev_alpha-action['steer_angle'])
        rew_c = -d_c/self.d_c0
        rew_theta = -d_theta/np.pi
        return -self.time_penalty+np.dot([rew_c,rew_theta,rew_dx,rew_alpha],self.reward_weights)
    
    # T(s,a) with s being self and a == array([dx,alpha]), modify self's variables and return nothing
    def _transition(self,action:dict):
        # extract actions
        dx = [-1,1][action['move_direction']] # driving direction
        alpha = -self.maxSteerAngle+action['steer_angle']*self.maxSteerAngle/self.numSteerAngle

        # update c and theta, truck updates its own params
        t = 0
        truck_trajectory = [[],[],[]]
        halfway_done = False
        while t<self.timePerStep:
            self.c,self.theta = self.truck.transition(self.c,self.theta,dx,alpha,self.speed,self.h)
            t += self.h

            # detect halfway collisions. May be modified to detect halfway success
            x0s,x1s,x2s = self.truck.getShapes(self.c,self.theta)
            for i in range(x0s.shape[0]):
                truck_trajectory[0].append(x0s[i])
                truck_trajectory[1].append(x1s[i])
                truck_trajectory[2].append(x2s[i])
            halfway_done = self.truck.isCollision()

            # for plotting
            if self.render_mode == "human":
                self._render_frame()

        # update prev_dx and prev alpha
        self.prev_dx = dx
        self.prev_alpha = alpha

        halfway_done = self.parkingLot.isCollision(np.array(truck_trajectory[0]),np.array(truck_trajectory[1]),np.array(truck_trajectory[2]))
        halfway_reward = self.collisionReward
        return halfway_done, halfway_reward
    
    def _isTerminal(self) -> tuple[bool,float]:
        # return true if success or collision or too many steps
        x0s,x1s,x2s = self.truck.getShapes(self.c,self.theta)
        collision = self.truck.isCollision() or self.parkingLot.isCollision(x0s,x1s,x2s)
        truncated = self.steps > self.maxSteps
        success = np.sum(np.abs(self.c-self.c_star)) < self.c_tol and abs(self.theta-self.theta_star) < self.theta_tol

        # extra reward is given if collision or success
        extraReward = 0
        if collision:
            extraReward = self.collisionReward
        if success:
            extraReward = self.successReward
        return collision or success or truncated, extraReward
    
    def render(self): 
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self): 
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 128, 128))

        pixPerUnit = self.window_size/max(self.xmax,self.ymax)
        self.parkingLot.draw(canvas,pixPerUnit)
        self.truck.draw(canvas,self.c,self.theta,pixPerUnit)

        # flip at last, so that drawing can be done in typical coordinates
        canvas = pygame.transform.flip(canvas,False,True)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Because stable_baseline3's DQN only supports Discrete type action space:
class TruckParkingEnvDiscrete(TruckParkingEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.action_space = spaces.Discrete(2*(2*self.numSteerAngle+1))

    def step(self, action):
        _action_dict = {
            'move_direction':int(action/(2*self.numSteerAngle+1)),
            'steer_angle':action%(2*self.numSteerAngle+1)
        }
        return super().step(_action_dict)
    
# with continuous action space
class TruckParkingEnvContinuous(TruckParkingEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.action_space = spaces.Box(low=np.array([-1.0,-self.numSteerAngle]), high=np.array([1.0,self.numSteerAngle]), shape=(2,), dtype=np.float32)
        _observe_dict = {
            'location':spaces.Box(np.array([0,0]),np.array([self.xmax,self.ymax]),dtype=float),
            'facing':spaces.Box(-np.pi,np.pi,dtype=float),
            'prev_direction':spaces.Discrete(2),
            'prev_steer_angle':spaces.Box(-self.numSteerAngle,self.numSteerAngle,dtype=float)
        }
        _observe_dict.update(self.truck.getObsSpace())
        self.observation_space = spaces.Dict(_observe_dict)

    def _get_obs(self) -> dict:
        obs = {
            'location':np.array(self.c,dtype=np.float64),# already np array
            'facing':np.array([self.theta],dtype=np.float64),
            'prev_direction':int((self.prev_dx+1)/2),
            'prev_steer_angle':np.array([(self.prev_alpha+self.maxSteerAngle)/(self.maxSteerAngle/self.numSteerAngle)],dtype=np.float64)
        }
        obs.update(self.truck.getObs())
        return obs

    def step(self, action):
        _action_dict = {
            'move_direction':1 if action[0]>=0 else 0,
            'steer_angle':action[1]
        }
        return super().step(_action_dict)
    
    
    
class VeryVerySimpleTruckParkingEnvDiscrete(TruckParkingEnvDiscrete):
    '''
    In this environment, the truck simply needs to drive straightly forward for 1 meter to success
    it is used to test the validity of the naive training case
    '''

    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([20,11])
    theta_star = np.pi/2
    c_tol = 0.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/30 # tolerance in radial

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.parkingLot:ParkingLot = EmptyParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([20,33])
        self.theta = np.pi/2*1.1
        self.prev_dx = 1 # default to start parking with forward direction
        self.prev_alpha = 0 # default to start parking with straight angle
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
    
class VeryVerySimpleTruckParkingEnvContinuous(TruckParkingEnvContinuous):
    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([20,11])
    theta_star = np.pi/2
    c_tol = 0.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/30 # tolerance in radial

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.parkingLot:ParkingLot = EmptyParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([20,33])
        self.theta = np.pi/2*1.1
        self.prev_dx = 1 # default to start parking with forward direction
        self.prev_alpha = 0 # default to start parking with straight angle
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
