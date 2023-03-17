import numpy as np

class Ellipse(object):
    
    def __init__(self,a,b):
        self.a = a
        self.b = b


    def surface_points(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = self.get_pts(theta)
        return pts

    def unit_normals(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = self.get_pts(theta)
        nx = pts[:,0]*self.b/self.a
        ny = pts[:,1]*self.a/self.b
        vec = np.stack((nx,ny),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec/np.stack((norm,norm),axis=1)

    def get_pts(self,theta):
        return np.stack((self.a*np.cos(theta),self.b*np.sin(theta)),axis=1)

 
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    a = 18
    b = 5
    num_pts = 25

    e = Ellipse(a,b)
    
    pts = e.surface_points(num_pts)
    plt.plot(pts[:,0],pts[:,1],'b-',label='points')

    # pts = e.midpts(num_pts)
    # plt.plot(pts[:,0],pts[:,1],'r-',label='midpoints')

    pts = e.surface_points(num_pts)
    normals = e.unit_normals(num_pts)
    for i in range(num_pts):
        if i == 0:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k',label='normals')
        else:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k')

    exact = e.surface_points(1000)
    plt.plot(exact[:,0],exact[:,1],'k-',label='exact')

    plt.legend(loc='upper right')
    plt.title('Ellipse a={} b={}'.format(a,b))
    plt.axis('equal')

    plt.show()