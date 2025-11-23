import numpy as np
import pygame
import shapely
from shapely import Polygon

from envs.dcel import parking_adapter

class ParkingLot():

    USE_THIRD_PARTY_MODULE = True

    def __init__(self,xm:float,ym:float,c_s:np.ndarray,theta_s:float):
        # parking lot dimensions
        assert xm >0 and ym >0
        self.xmax = xm
        self.ymax = ym

        # target parking position
        self.c_star = c_s
        self.theta_star = theta_s

    # use three points to represent a rectangle (parallelogram)
    # p1
    # |\
    # | \
    # |  \
    # |   \
    # |    \
    # |_____\
    # p0     p2
    # p3 = p1+p2-p0

    # determine whether a list of rectangles (parallelograms) will collide with the environment
    # each array is shape (n,2)
    def isCollision(self,p0s:np.ndarray,p1s:np.ndarray,p2s:np.ndarray) -> bool:
        shape0 = p0s.shape
        shape1 = p1s.shape
        shape2 = p2s.shape
        assert len(shape0)==2 and len(shape1)==2 and len(shape2)==2 and shape0[0] == shape1[0] and shape1[0] == shape2[0]
        return False
    
    def minDistance(self,p0s:np.ndarray,p1s:np.ndarray,p2s:np.ndarray) -> bool:
        shape0 = p0s.shape
        shape1 = p1s.shape
        shape2 = p2s.shape
        assert len(shape0)==2 and len(shape1)==2 and len(shape2)==2 and shape0[0] == shape1[0] and shape1[0] == shape2[0]
        return 0
    
    def getShapes(self) -> tuple[np.ndarray]:
        return np.array([[0,0]]),np.array([[0,0]]),np.array([[0,0]])

    def draw(self,canvas:pygame.Surface,pixPerUnit:int):
        return

class VerticalParkingLot(ParkingLot):

    # dimensions of the parking lot
    x0 = 16
    x1 = 24
    x2 = 40
    y0 = 15
    y1 = 40

    def __init__(self, xm, ym, c_s, theta_s):
        super().__init__(xm, ym, c_s, theta_s)
        # in this case, we have y1==ymax and x1==xmax
        self.x2 = xm
        self.y1 = ym

    def __str__(self):
        return 'VerticalParkingLot: [',str(self.x2)+','+str(self.y1)+']'

    def isCollision(self, r0s, r1s, r2s):
        if ParkingLot.USE_THIRD_PARTY_MODULE:
            r3s = r1s+r2s-r0s
            parking_poly = Polygon([(0,self.y0),(self.x0,self.y0),(self.x0,0),(self.x1,0),(self.x1,self.y0),(self.x2,self.y0),(self.x2,self.y1),(0,self.y1)])
            truck_polys = [Polygon([r0s[i],r2s[i],r3s[i],r1s[i]]) for i in range(r0s.shape[0])]
            for truck_poly in truck_polys:
                if not parking_poly.contains(truck_poly):
                    return True
            return False
        # Return False iff all rectangles identified by p0s,p1s,p2s are contained in x0s,x1s,x2s
        r3s = r1s+r2s-r0s
        parking_points = [(0,self.y0),(self.x0,self.y0),(self.x0,0),(self.x1,0),(self.x1,self.y0),(self.x2,self.y0),(self.x2,self.y1),(0,self.y1)]
        parking_segments = [(parking_points[i],parking_points[(i+1)%len(parking_points)]) for i in range(len(parking_points))]
        # avoid overlapping edges by moving truck detection points slightly
        dy=0.00001
        truck_points = [(r0s[i],r2s[i],r3s[i],r1s[i]) for i in range(r0s.shape[0])]
        truck_segments=[]
        def nparray_to_tuple(nparray):
            return (float(nparray[0]),float(nparray[1])+dy)
        for i in range(r0s.shape[0]):
            for j in range(4):
                truck_segments.append((nparray_to_tuple(truck_points[i][j]),nparray_to_tuple(truck_points[i][(j+1)%4])))
        return parking_adapter(parking_segments, truck_segments)
    
    def minDistance(self, r0s, r1s, r2s):
        resl = np.inf
        r3s = r1s+r2s-r0s
        parking_poly = Polygon([(0,self.y0),(self.x0,self.y0),(self.x0,0),(self.x1,0),(self.x1,self.y0),(self.x2,self.y0),(self.x2,self.y1),(0,self.y1)])
        parking_poly_bounds = parking_poly.boundary
        truck_polys = [Polygon([r0s[i],r2s[i],r3s[i],r1s[i]]) for i in range(r0s.shape[0])]
        for truck_poly in truck_polys:
            resl = min(resl,shapely.distance(truck_poly,parking_poly_bounds))

        return resl
    
    def getShapes(self):
        shapes0 = np.array([[self.x0,0],[0,self.y0]])
        shapes1 = np.array([[self.x0,self.y0],[0,self.y1]])
        shapes2 = np.array([[self.x1,0],[self.x2,self.y0]])
        return shapes0,shapes1,shapes2
    
    def draw(self,canvas,pixPerUnit):
        x0s,x1s,x2s = self.getShapes()
        x3s = x1s+x2s-x0s
        x0s = (x0s*pixPerUnit)
        x2s = (x2s*pixPerUnit)
        x3s = (x3s*pixPerUnit)
        x1s = (x1s*pixPerUnit)
        for i in range(2):
            pygame.draw.polygon(canvas,[255,255,255],[x0s[i,:],x2s[i,:],x3s[i,:],x1s[i,:]],width=0)
        _center = (self.c_star*pixPerUnit)
        pygame.draw.circle(canvas,[32,32,32],_center,2)

class EmptyParkingLot(VerticalParkingLot):
    x0 = 0
    x1 = 40
    x2 = 40
    y0 = 0
    y1 = 40

    def __init__(self, xm, ym, c_s, theta_s):
        super().__init__(xm, ym, c_s, theta_s)
        # in this case, we have empty parking lot
        self.x1 = xm

class LongParkingLot(VerticalParkingLot):
    x0 = 40
    x1 = 40
    x2 = 40
    y0 = 30
    y1 = 40

    def __init__(self, xm, ym, c_s, theta_s):
        super().__init__(xm, ym, c_s, theta_s)
        # in this case, we have empty parking lot
        self.x1 = xm