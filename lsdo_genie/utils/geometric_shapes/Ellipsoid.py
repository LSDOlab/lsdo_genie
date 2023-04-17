import numpy as np
from matplotlib.patches import FancyArrowPatch

class Ellipsoid:
    '''
    Ellipsoid
    '''

    def __init__(self,a,b,c):
        '''
        __init__
        '''
        self.a = a
        self.b = b
        self.c = c

    def surface_points(self,num_pts):
        '''
        surface_points
        '''
        indices = np.arange(0, num_pts, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices

        x = self.a*np.cos(theta)*np.sin(phi)
        y = self.b*np.sin(theta)*np.sin(phi)
        z = self.c*np.cos(phi)
        return np.stack((x.flatten(),y.flatten(),z.flatten()),axis=1)

    def unit_normals(self,num_pts):
        '''
        unit_normals
        '''
        pts = self.surface_points(num_pts)
        dx = 2*pts[:,0]/self.a
        dy = 2*pts[:,1]/self.b
        dz = 2*pts[:,2]/self.c
        vec = np.stack((dx,dy,dz),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec / np.tile(norm,(3,1)).T