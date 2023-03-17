from lsdo_genie import Genie2D
import numpy as np

class rectangle:
    
    def __init__(self,w,h):
        self.h = h
        self.w = w
        self.range = 2*w + 2*h
        self.b1 = w
        self.b2 = w+h
        self.b3 = 2*w+h

    def points(self,num_pts):
        theta = np.linspace(0,self.range,2*(num_pts)+1)[1::2]
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
        return pts

    def unit_pt_normals(self,num_pts):
        theta = np.linspace(0,self.range,2*(num_pts)+1)[1::2]
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
        return norm_vec
    
num_points = 76
geom_shape = rectangle(5,7)
surface_points = geom_shape.points(num_points)
surface_normals = geom_shape.unit_pt_normals(num_points)

custom_dimensions = np.array(
    [[-4.,4.],[-5.6,5.6]]
)

genie = Genie2D()
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    dimensions=custom_dimensions,
    max_control_points=200,
)
genie.solve_energy_minimization(
    Lp=1e0,
    Ln=1e0,
    Lr=1e-4,
)
genie.visualize()