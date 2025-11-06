import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from envs.parkingLot import ParkingLot, VerticalParkingLot
from envs.truck import Truck, TrailerTruck

class TruckParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    window_size = 100
    
    # dimensions of the parking lot, meter
    xmax = 40
    ymax = 40
    parkingLotParams = dict()

    # truck control
    numSteerAngle = 3 # the actual number of steering angles will be 2*this+1
    maxSteerAngle = np.pi/6
    truckParams = {'maxTrailer':np.pi*2/3}# the maximum angle between trailer and driver

    # running parameters
    speed = 0.5 # constant speed, meter/s
    timePerStep = 1 # truck run for this time before next state
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
    reward_weights = np.array([0.1,0.01,1,0.5])

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

            self.steps = 0
            valid = not self._isTerminal()[0]
        obs = self._get_obs()

        # for plotting
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, {}
    
    def _get_obs(self) -> dict:
        obs = {
            'location':self.c,
            'facing':self.theta,
            'prev_direction':self.prev_dx,
            'prev_steer_angle':self.prev_alpha
        }
        obs.update(self.truck.getObs())
        return obs

    # Generat step function, action ==  array([dx,alpha])
    def step(self, action):
        # verify action validity
        assert action[0] in [0,1] and action[1] in range(2*self.numSteerAngle+1)

        # reward is called on s,a
        reward = self._reward(action)
        # update self to s' from s,a
        self._transition(action)
        obs = self._get_obs()
        # isTerminal is called on s'
        done,extraReward = self._isTerminal()
        reward += extraReward
        self.steps += 1

        # for plotting
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, False, {}
    
    # R(s,a) with s being self and a == array([dx,alpha])
    def _reward(self,action) -> float:
        d_c = np.sum(np.abs(self.c-self.c_star))
        d_theta = abs(self.theta-self.theta_star)
        d_dx = int(self.prev_dx != action[0])
        d_alpha = self.prev_alpha-action[1]
        return -self.time_penalty-np.dot([d_c,(1-np.cos(d_theta))/(1+d_c),d_dx,d_alpha],self.reward_weights)
    
    # T(s,a) with s being self and a == array([dx,alpha]), modify self's variables and return nothing
    def _transition(self,action):
        # extract actions
        dx = [-1,1][action[0]] # driving direction
        alpha = -self.maxSteerAngle+action[1]*self.maxSteerAngle/self.numSteerAngle

        # update c and theta, truck updates its own params
        self.c,self.theta = self.truck.transition(self.c,self.theta,dx,alpha,self.speed,self.timePerStep)
        # update prev_dx and prev alpha
        self.prev_dx = dx
        self.prev_alpha = alpha
    
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
        self.parkingLot.draw(pixPerUnit)
        self.truck.draw(self.c,self.theta,pixPerUnit)

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
