import numpy as np
import pygame
from gymnasium import spaces

class Truck():
    
    h = 0.01 # time step to run the truck (differentiated path)

    def __init__(self,maxSteer):
        assert 0<maxSteer and maxSteer<np.pi/2
        self.maxSteerAngle = maxSteer
        
    def reset(self):
        pass
    
    def getObsSpace(self):
        return dict()
    
    def getObs(self):
        return dict()
    
    # get the rectangles representing the truck
    def getShapes(self,c0,theta0) -> tuple[np.ndarray]:
        return np.array([[0,0]]),np.array([[0,0]]),np.array([[0,0]])
    
    # monitored by the environment, thus self's method shoudn't use this
    def isCollision(self):
        return False
    
    # drive the truck in direction dx, with steering angle alpha, with speed, for time
    # return d_c, d_theta; d_c is array([dxc,dyc])
    def transition(self,c0,theta0,dx,alpha,speed,time):
        return np.array([0,0]),0
    
    def draw(self,c0,theta0,canvas:pygame.Surface,pixPerUnit:int):
        pass

    # points has shape (n,2). translate by c: array (2) and rotate by theta
    def shape_transform(points:np.ndarray,c,theta) -> np.ndarray:
        resl = np.zeros_like(points)
        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            thetaNew = theta+np.arctan2(x,y)
            resl[i,0] = c[0]+np.sqrt(x*x+y*y)*np.cos(thetaNew)
            resl[i,1] = c[1]+np.sqrt(x*x+y*y)*np.sin(thetaNew)
        return resl

class TrailerTruck(Truck):

    # dimensions of the truck
    la = 4
    lb = 1
    lc = 8
    l0 = 1.5
    l1 = 0.5
    l2 = 0.5
    l3 = 0.5
    l4 = 2

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
        return {'trailer_angle':spaces.Box(-self.maxTrailerAngle,self.maxTrailerAngle,dtype=float)}
    
    def getObs(self):
        return {'beta':self.beta}
    
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
    
    def transition(self,c0,theta0,dx,alpha,speed,time):
        # update beta for itself
        # return c and theta for the environment
        theta = theta0
        xc = c0[0]
        yc = c0[1]
        beta = self.beta
        t = 0
        while t < time:
            # first compute time derivatives as buffer
            d_theta = speed*np.tan(alpha)/(self.la+self.lb)
            d_xc = speed*np.cos(theta)-self.lb*np.sin(theta)*d_theta
            d_yc = speed*np.sin(theta)+self.lb*np.cos(theta)*d_theta
            d_beta = (np.cos(theta-beta)*d_yc-np.sin(theta-beta)*d_xc)/self.lc

            theta += dx*self.h*d_theta
            xc += dx*self.h*d_xc
            yc += dx*self.h*d_yc
            beta += dx*self.h*d_beta

            t += self.h
        self.beta = beta
        return np.array([xc,yc]),theta
    
    def draw(self,c0,theta0,canvas:pygame.Surface,pixPerUnit:int):
        x0s,x1s,x2s = self.getShapes(c0,theta0)
        x3s = x1s+x2s-x0s
        x0s *= pixPerUnit
        x2s *= pixPerUnit
        x3s *= pixPerUnit
        x1s *= pixPerUnit
        for i in range(2):
            pygame.draw.polygon(canvas,[216,216,216],[x0s[i,:],x2s[i,:],x3s[i,:],x1s[i,:]],width=1)

    