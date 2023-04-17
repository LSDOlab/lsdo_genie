import numpy as np

class Rectangle:
    '''
    Rectangle
    '''

    def __init__(self,w,h,rotation=0):
        '''
        __init__
        '''
        self.h = h
        self.w = w
        self.range = 2*w + 2*h
        self.b1 = w
        self.b2 = w+h
        self.b3 = 2*w+h
        theta = np.deg2rad(rotation)
        c, s = np.cos(theta), np.sin(theta)
        self.rotmat = np.array(((c, -s), (s, c)))

    def surface_points(self,num_pts):
        '''
        surface_points
        '''
        theta = np.linspace(0, self.range, num_pts, endpoint=False)
        pts = np.zeros((len(theta),2))
        for i,t in enumerate(theta):
            if t<self.b1:
                pts[i,0] = t - self.b1/2
                pts[i,1] = -self.h/2
            elif t>=self.b1 and t<self.b2:
                pts[i,0] = self.w/2
                pts[i,1] = (t-self.b1-self.h/2)
            elif t>=self.b2 and t<self.b3:
                pts[i,0] = (self.b3-t-self.w/2)
                pts[i,1] = self.h/2
            elif t<=self.range:
                pts[i,0] = -self.w/2
                pts[i,1] = (self.range-t-self.h/2)
        return pts @ self.rotmat

    def unit_normals(self,num_pts):
        '''
        unit_normals
        '''
        theta = np.linspace(0, self.range, num_pts, endpoint=False)
        norm_vec = np.zeros((len(theta),2))
        for i,t in enumerate(theta):
            if t==0:
                norm_vec[i] = np.array([-1,-1])/np.sqrt(2)
            elif t>0 and t<self.b1:
                norm_vec[i] = np.array([0,-1])
            elif t==self.b1:
                norm_vec[i] = np.array([1,-1])/np.sqrt(2)
            elif t>self.b1 and t<self.b2:
                norm_vec[i] = np.array([1,0])
            elif t==self.b2:
                norm_vec[i] = np.array([1,1])/np.sqrt(2)
            elif t>self.b2 and t<self.b3:
                norm_vec[i] = np.array([0,1])
            elif t==self.b3:
                norm_vec[i] = np.array([-1,1])/np.sqrt(2)
            elif t>self.b3 and t<self.range:
                norm_vec[i] = np.array([-1,0])
        return norm_vec @ self.rotmat