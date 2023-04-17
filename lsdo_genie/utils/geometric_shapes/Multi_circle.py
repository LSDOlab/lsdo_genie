import numpy as np

class Multi_circle:
    '''
    Multi_circle
    '''

    def __init__(self,centers,radii):
        '''
        __init__
        '''
        self.center = []
        self.radius = []
        for (c,r) in zip(centers,radii):
            self.center.append(c)
            self.radius.append(r)

    def surface_points(self,num_pts):
        '''
        surface_points
        '''
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        pts = np.empty((0,2))
        for (cent,r) in zip(self.center,self.radius):
            px = cent[0]+r*np.cos(theta)
            py = cent[1]+r*np.sin(theta)
            pts = np.vstack((pts,np.stack((px,py),axis=1)))
        return pts

    def unit_normals(self,num_pts):
        '''
        unit_normals
        '''
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        norms = np.empty((0,2))
        for i in range(len(self.radius)):
            nx = np.cos(theta)
            ny = np.sin(theta)
            norms = np.vstack((norms,np.stack((nx,ny),axis=1)))
        return norms