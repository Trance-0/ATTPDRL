import numpy as np
import pygame
from gymnasium import spaces

class Truck():

    def __init__(self,maxSteer:float):
        assert 0<maxSteer and maxSteer<np.pi/2
        self.maxSteerAngle = maxSteer
        
    def reset(self):
        self.alpha = 0
    
    def getObsSpace(self) -> spaces.Box:
        return None
    
    def getObs(self) -> dict:
        return dict()
    
    # get the rectangles representing the truck
    def getShapes(self,c0:np.ndarray,theta0:float) -> tuple[np.ndarray]:
        return np.array([[0,0]]),np.array([[0,0]]),np.array([[0,0]])
    
    # monitored by the environment, thus self's method shoudn't use this
    def isCollision(self) -> bool:
        return False
    
    def getReward(self,action:int) -> float:
        return 0
    
    # drive the truck in direction dx, with steering angle alpha, with speed, for time
    # return d_c, d_theta; d_c is array([dxc,dyc])
    def transition(self,c0:np.ndarray,theta0:float,dx:int,alpha:float,speed:float,h:float) -> tuple[np.ndarray,float]:
        return np.array([0,0]),0
    
    def draw(self,c0:np.ndarray,theta0:float,canvas:pygame.Surface,pixPerUnit:int):
        return

    # points has shape (n,2). translate by c: array (2) and rotate by theta
    def shape_transform(points:np.ndarray,c:np.ndarray,theta:float) -> np.ndarray:
        resl = np.zeros_like(points)
        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            thetaNew = theta+np.arctan2(y,x)
            resl[i,0] = c[0]+np.sqrt(x*x+y*y)*np.cos(thetaNew)
            resl[i,1] = c[1]+np.sqrt(x*x+y*y)*np.sin(thetaNew)
        return resl

class TrailerTruck(Truck):

    # dimensions of the truck
    la = 4.5
    lb = 0.5
    lc = 10
    l0 = 1.2
    l1 = 0.5
    l2 = 0.4
    l3 = 0.7
    l4 = 1

    trailerAnglePenaltyThreshold = 0.5

    def __init__(self,maxSteer,maxTrailer):
        super().__init__(maxSteer)
        assert 0<maxTrailer and maxTrailer<np.pi
        self.maxTrailerAngle = maxTrailer

    def __str__(self):
        return 'TrailerTruck: ['+str(self.la)+','+str(self.lb)+','+str(self.lc)+']'

    def reset(self):
        super().reset()
        self.beta = 0

    def getObsSpace(self):
        return spaces.Box(-self.maxTrailerAngle,self.maxTrailerAngle,dtype=float)
    
    def getObs(self):
        return np.array([self.beta],dtype=float)
    
    def getShapes(self,c0,theta0):
        shapes0 = np.array([[-self.lb-self.l2,-self.l0],[-(self.lc+self.l4)*np.cos(self.beta)-self.l0*np.sin(self.beta), (self.lc+self.l4)*np.sin(self.beta)-self.l0*np.cos(self.beta)]])
        shapes1 = np.array([[-self.lb-self.l2,self.l0],[-(self.lc+self.l4)*np.cos(self.beta)+self.l0*np.sin(self.beta), (self.lc+self.l4)*np.sin(self.beta)+self.l0*np.cos(self.beta)]])
        shapes2 = np.array([[self.la+self.l1,-self.l0],[self.l3*np.cos(self.beta)-self.l0*np.sin(self.beta), -self.l3*np.sin(self.beta)-self.l0*np.cos(self.beta)]])
        shapes0 = Truck.shape_transform(shapes0,c0,theta0)
        shapes1 = Truck.shape_transform(shapes1,c0,theta0)
        shapes2 = Truck.shape_transform(shapes2,c0,theta0)
        return shapes0,shapes1,shapes2
    
    
    def isCollision(self):
        return abs(self.beta) >= self.maxTrailerAngle
    
    def getReward(self,action:int):
        if abs(self.beta)<self.maxTrailerAngle*self.trailerAnglePenaltyThreshold:
            return 0
        return (1+self.maxTrailerAngle*(1-self.trailerAnglePenaltyThreshold))/abs(1+self.maxTrailerAngle-abs(self.beta))
    
    def transition(self,c0,theta0,dx,alpha,speed,h):
        self.alpha = np.sign(alpha)*min(abs(alpha),self.maxSteerAngle)
        # update beta for itself
        # return c and theta for the environment
        theta = theta0
        xc = c0[0]
        yc = c0[1]
        beta = self.beta
        
        # first compute time derivatives as buffer
        d_theta = dx*speed*np.tan(self.alpha)/(self.la+self.lb)
        d_xc = dx*speed*np.cos(theta)-self.lb*np.sin(theta)*d_theta
        d_yc = dx*speed*np.sin(theta)+self.lb*np.cos(theta)*d_theta
        d_beta = d_theta-dx*speed*np.sin(self.beta)/self.lc

        theta += h*d_theta
        xc += h*d_xc
        yc += h*d_yc
        beta += h*d_beta

        self.beta = beta
        return np.array([xc,yc]),theta
    
    def draw(self,canvas,c0,theta0,pixPerUnit):
        x0s,x1s,x2s = self.getShapes(c0,theta0)
        x3s = x1s+x2s-x0s
        x0s = (x0s*pixPerUnit)
        x2s = (x2s*pixPerUnit)
        x3s = (x3s*pixPerUnit)
        x1s = (x1s*pixPerUnit)
        for i in range(2):
            pygame.draw.polygon(canvas,[64,64,64],[x0s[i,:],x2s[i,:],x3s[i,:],x1s[i,:]],width=2)
        # also draw steering direction
        _steer = np.array([[self.la,0],[self.la+self.l0*np.cos(self.alpha),self.l0*np.sin(self.alpha)]])
        _steer = (Truck.shape_transform(_steer,c0,theta0)*pixPerUnit).astype(int)
        pygame.draw.line(canvas,[96,96,96],_steer[0,:],_steer[1,:],width=2)
        # also draw center
        _center = (c0*pixPerUnit)
        pygame.draw.circle(canvas,[32,32,32],_center,2)

    