from lsdo_genie import Genie3D
import numpy as np

class Ellipsoid:

    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def points(self,num_pts):
        indices = np.arange(0, num_pts, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices

        x = self.a*np.cos(theta)*np.sin(phi)
        y = self.b*np.sin(theta)*np.sin(phi)
        z = self.c*np.cos(phi)
        return np.stack((x.flatten(),y.flatten(),z.flatten()),axis=1)

    def unit_pt_normals(self,num_pts):
        pts = self.points(num_pts)
        dx = 2*pts[:,0]/self.a
        dy = 2*pts[:,1]/self.b
        dz = 2*pts[:,2]/self.c
        vec = np.stack((dx,dy,dz),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec / np.tile(norm,(3,1)).T

num_pts = 100

e = Ellipsoid(5,5,5)

surface_points = e.points(num_pts)
surface_normals = e.unit_pt_normals(num_pts)

x,y,z = surface_points[:,0],surface_points[:,1],surface_points[:,2]
border = 5
custom_dimensions = np.array([
    [x.min()-border, x.max()+border],
    [y.min()-border, y.max()+border],
    [z.min()-border, z.max()+border],
])

genie = Genie3D()
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    dimensions=custom_dimensions,
    max_control_points=40,
)
genie.solve_energy_minimization(
    Lp=1e0,
    Ln=1e0,
    Lr=1e-3,
)
genie.visualize()