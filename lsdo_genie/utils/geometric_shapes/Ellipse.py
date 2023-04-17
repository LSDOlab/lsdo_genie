import numpy as np

class Ellipse:
    '''
    Ellipse
    '''

    def __init__(self,a,b):
        '''
        __init__
        '''
        self.a = a
        self.b = b

    def surface_points(self,num_pts):
        '''
        surface_points
        '''
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = np.stack((self.a*np.cos(theta),self.b*np.sin(theta)),axis=1)
        return pts

    def unit_normals(self,num_pts):
        '''
        unit_normals
        '''
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = np.stack((self.a*np.cos(theta),self.b*np.sin(theta)),axis=1)
        nx = pts[:,0]*self.b/self.a
        ny = pts[:,1]*self.a/self.b
        vec = np.stack((nx,ny),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec/np.stack((norm,norm),axis=1)