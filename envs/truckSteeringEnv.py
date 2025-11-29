import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from envs.parkingLot import *
from envs.truck import *

class TruckSteeringEnv(gym.Env):
    
    # dimensions of the parking lot, meter
    xmax = 40
    ymax = 40
    parkingLotParams = dict()

    # truck control
    deltaSteerAngle = np.pi/60 # the step-wise changes in steering angle
    maxSteerAngle = np.pi/6
    truckParams = {'maxTrailer':np.pi*2/3}# the maximum angle between trailer and driver
    direction = 1 # 1 for forward, -1 for back

    # running parameters
    speed = 1 # constant speed, meter/s
    timePerStep = 0.1 # truck run for this time (seconds) before next state
    h = 0.01 # time step to run the truck (differentiated path)
    maxSteps = 200 # max number of steps per trajectory
    
    # target location and facing 
    c_star = np.array([6,35])
    theta_star = np.pi
    c_tol = 1.5 # tolerance in meters of distance
    theta_tol = np.pi/6 # tolerance in radial

    # reward function parameter
    time_penalty = 0.01
    reward_weights = np.array([1,0.5,0.5,1])
    wallPenaltyThreshold = 1.5
    collisionReward = -100
    successReward = 100

    # for plotting
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": max(1/h,1000)}
    window_size = 1000


    def __init__(self, render_mode=None,direction:int=None,reward_weights:np.ndarray=None,time_penalty:float=None,collisionReward:float=None,successReward:float=None,maxSteps:int=None):
        super().__init__()

        # define the parking lot
        self.parkingLot:ParkingLot = EmptyParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

        # define the truck
        self.truck:Truck = TrailerTruck(self.maxSteerAngle,**self.truckParams)

        # binary action space, 0 for left, 1 for right
        self.action_space = spaces.Discrete(2)

        # c, theta, (possible) extra observation from the truck (must be Box)
        truck_space = self.truck.getObsSpace()
        low = np.array([0,0,-np.pi])
        high = np.array([self.xmax,self.ymax,np.pi])
        if truck_space is not None:
            assert isinstance(truck_space,gym.spaces.Box)
            low = np.concat((low,truck_space.low))
            high = np.concat((high,truck_space.high))

        self.observation_space = spaces.Box(low,high)

        obs,_ = self.reset()
        print('Environment initialized with state:\n',obs)

        # for plotting
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        if direction is not None:
            self.direction = direction
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
            self.alpha = 0
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
    
    def _get_obs(self) -> np.ndarray:
        obs = np.concat((self.c,[self.theta],self.truck.getObs()))
        return obs

    # Generat step function, action ==  array([dx,alpha])
    def step(self, action:float) -> tuple[dict,float,bool,bool,dict]:
        # verify action validity
        # assert action in [0,1] 

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
    def _reward(self,action:float) -> float:
        d_c = np.linalg.norm(self.c-self.c_star)
        d_theta = abs(self.theta-self.theta_star)
        d_c_prev = np.linalg.norm(self.prev_obs[0:2]-self.c_star)
        x0s,x1s,x2s = self.truck.getShapes(self.c,self.theta)
        d_wall = self.parkingLot.minDistance(x0s,x1s,x2s)
        rew_c = -(d_c)/self.d_c0
        rew_theta = -(d_theta/np.pi)
        rew_truck = self.truck.getReward(action)
        rew_wall = 0
        if d_wall <= self.wallPenaltyThreshold:
            rew_wall = -(1+self.wallPenaltyThreshold)/(1+d_wall)#penalize when too close to wall
        return -self.time_penalty+np.dot([rew_c,rew_theta,rew_truck,rew_wall],self.reward_weights)
    
    # T(s,a) with s being self and a == array([dx,alpha]), modify self's variables and return nothing
    def _transition(self,action:float) -> tuple[bool,float]:
        # extract actions
        self.alpha += (2*action-1)*self.deltaSteerAngle
        self.alpha = max(-self.maxSteerAngle,min(self.maxSteerAngle,self.alpha))

        # update c and theta, truck updates its own params
        t = 0
        truck_trajectory = [[],[],[]]
        halfway_done = False
        while t<self.timePerStep:
            self.c,self.theta = self.truck.transition(self.c,self.theta,self.direction,self.alpha,self.speed,self.h)
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

        halfway_done = self.parkingLot.isCollision(np.array(truck_trajectory[0]),np.array(truck_trajectory[1]),np.array(truck_trajectory[2]))
        halfway_reward = self.collisionReward
        return halfway_done, halfway_reward
    
    def _isTerminal(self) -> tuple[bool,float]:
        # return true if success or collision or too many steps
        x0s,x1s,x2s = self.truck.getShapes(self.c,self.theta)
        collision = self.truck.isCollision() or self.parkingLot.isCollision(x0s,x1s,x2s)
        truncated = self.steps > self.maxSteps
        success = np.sum(np.abs(self.c-self.c_star)) < self.c_tol and min(abs(self.theta-self.theta_star),np.pi*2-abs(self.theta-self.theta_star)) < self.theta_tol

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

class TruckSteeringEnvCts(TruckSteeringEnv):
    def __init__(self, render_mode=None, direction = None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, direction, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.action_space = spaces.Box(-self.maxSteerAngle,self.maxSteerAngle)

    def _transition(self, action):
        # extract actions
        self.alpha += action[0]
        self.alpha = max(-self.maxSteerAngle,min(self.maxSteerAngle,self.alpha))

        # update c and theta, truck updates its own params
        t = 0
        truck_trajectory = [[],[],[]]
        halfway_done = False
        while t<self.timePerStep:
            self.c,self.theta = self.truck.transition(self.c,self.theta,self.direction,self.alpha,self.speed,self.h)
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

        halfway_done = self.parkingLot.isCollision(np.array(truck_trajectory[0]),np.array(truck_trajectory[1]),np.array(truck_trajectory[2]))
        halfway_reward = self.collisionReward
        return halfway_done, halfway_reward

class TruckSteeringForwardEnv(TruckSteeringEnv):
    '''
    In this environment, the truck simply needs to drive straightly forward for to success
    it is used to test the validity of the naive training case
    '''

    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([34,35])
    theta_star = 0
    c_tol = 1.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/6 # tolerance in radial

    def __init__(self, render_mode=None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, 1, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.parkingLot:ParkingLot = LongParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([11,35])
        self.theta = 0
        self.alpha = 0
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs
        self.d_c0 = np.linalg.norm(self.c-self.c_star)

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
    
class TruckSteeringForwardTurnEnv(TruckSteeringEnv):
    '''
    In this environment, the truck simply needs to drive straightly forward for to success
    it is used to test the validity of the naive training case
    '''

    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([35,6])
    theta_star = -np.pi/2
    c_tol = 1.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/6 # tolerance in radial

    def __init__(self, render_mode=None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, 1, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.parkingLot:ParkingLot = TurnParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([11,35])
        self.theta = 0
        self.alpha = 0
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs
        self.d_c0 = np.linalg.norm(self.c-self.c_star)

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}

class TruckSteeringBackwardEnv(TruckSteeringEnv):
    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([29,35])
    theta_star = np.pi
    c_tol = 1.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/6 # tolerance in radial

    def __init__(self, render_mode=None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, -1, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.parkingLot:ParkingLot = LongParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([6,35])
        self.theta = np.pi
        self.alpha = 0
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs
        self.d_c0 = np.linalg.norm(self.c-self.c_star)

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
    
class TruckSteeringBackwardEnv(TruckSteeringEnv):
    timePerStep = 0.1

    # target location and facing 
    c_star = np.array([35,11])
    theta_star = np.pi
    c_tol = 1.5 # tolerance in meters of manhattan distance
    theta_tol = np.pi/6 # tolerance in radial

    def __init__(self, render_mode=None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, -1, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.parkingLot:ParkingLot = TurnParkingLot(self.xmax,self.ymax,self.c_star,self.theta_star)

    def reset(self, seed=None, options=None) -> tuple[dict,dict]:
        super().reset(seed=seed)

        self.c = np.array([6,35])
        self.theta = np.pi
        self.alpha = 0
        self.truck.reset()
        self.steps = 0
        obs = self._get_obs()
        self.prev_obs = obs
        self.d_c0 = np.linalg.norm(self.c-self.c_star)

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}

class TruckSteeringForwardEnvCts(TruckSteeringForwardEnv):
    def __init__(self, render_mode=None, direction = None, reward_weights = None, time_penalty = None, collisionReward = None, successReward = None, maxSteps = None):
        super().__init__(render_mode, direction, reward_weights, time_penalty, collisionReward, successReward, maxSteps)
        self.action_space = spaces.Box(-self.maxSteerAngle,self.maxSteerAngle)

    def _transition(self, action):
        # extract actions
        self.alpha += action[0]
        self.alpha = max(-self.maxSteerAngle,min(self.maxSteerAngle,self.alpha))

        # update c and theta, truck updates its own params
        t = 0
        truck_trajectory = [[],[],[]]
        halfway_done = False
        while t<self.timePerStep:
            self.c,self.theta = self.truck.transition(self.c,self.theta,self.direction,self.alpha,self.speed,self.h)
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

        halfway_done = self.parkingLot.isCollision(np.array(truck_trajectory[0]),np.array(truck_trajectory[1]),np.array(truck_trajectory[2]))
        halfway_reward = self.collisionReward
        return halfway_done, halfway_reward
    
