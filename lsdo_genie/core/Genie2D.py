from lsdo_genie.core.Bsplines.Bspline_Surface import BSplineSurface
from lsdo_genie.core.Bsplines.knot_vectors import standard_uniform_knot_vector
from lsdo_genie.utils.Hicken_Kaur import explicit_lsf
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.sparse as sps
import seaborn as sns
import numpy as np

class Genie2D(BSplineSurface):
    def __init__(self):
        self.u = {}
        self.v = {}

    def input_point_cloud(self, surface_points:np.ndarray=None, surface_normals:np.ndarray=None):
        self.surface_points = surface_points
        self.surface_normals = surface_normals

        # Number of points
        self.num_surf_pts = int(len(surface_points))        
        # Minimum bounding box properties
        lower = np.min(self.surface_points,axis=0)
        upper = np.max(self.surface_points,axis=0)
        diff = upper-lower
        self.min_bbox = np.stack((lower, upper),axis=1)
        self.Bbox_diag = np.linalg.norm(diff)
        # Printing
        print('Minimum bbox: \n',self.min_bbox)
        print('Minimum bbox diagonal: ',self.Bbox_diag)
        print('num_surf_pts: ', self.num_surf_pts,'\n')
    
    def config(self, dimensions:np.ndarray, max_control_points:int, order:int=4, min_ratio:float=0.5):
        self.order = order
        if (self.surface_points[:,0] < dimensions[0,0]).any() \
            or (self.surface_points[:,0] > dimensions[0,1]).any() \
            or (self.surface_points[:,1] < dimensions[1,0]).any() \
            or (self.surface_points[:,1] > dimensions[1,1]).any():
            raise ValueError("surface points lie outside of the defined dimensions")
        self.dimensions = dimensions

        # Bounding box properties
        dxy = np.diff(dimensions).flatten()
        self.scaling = 1/dxy
        # Number of control points in each direction
        frac = dxy / np.max(dxy)
        num_cps = np.zeros(2,dtype=int)
        for i,ratio in enumerate(frac):
            if ratio < min_ratio:
                ratio = min_ratio
            # num_cps[i] = int(frac[i]*max_control_points)
            # num_cps[i] = int((frac[i]*max_cps)+order-1)
            num_cps[i] = 3*int((ratio*max_control_points)/3)
        self.num_cps = num_cps
        self.num_cps_pts  = int(np.product(self.num_cps))

        # Get initial control points
        self.control_points = self.initialize_control_points(k=6,rho=10)
        # Standard uniform knot vectors
        kv_u = standard_uniform_knot_vector(num_cps[0], order)
        kv_v = standard_uniform_knot_vector(num_cps[1], order)
        # Define Bspline Volume object
        super().__init__('Bspline',order,order,kv_u,kv_v,num_cps)
        
        # Surface points for data terms (Ep, En)
        self.u['surf'], self.v['surf'] = self.spatial_to_parametric(self.surface_points)
        # Quadrature points for regulation term (Er)
        temp_u, temp_v = self.spatial_to_parametric(self.control_points[:,0:2])
        mask = np.argwhere(
            (temp_u>=0)*(temp_u<=1)*\
            (temp_v>=0)*(temp_v<=1)
        )
        self.u['hess'], self.v['hess'] = temp_u[mask].flatten(), temp_v[mask].flatten()
        self.num_hess_pts = int(len(self.u['hess']))

        # Printing
        print('Bspline box: \n',self.dimensions)
        print('Control point grid: ', self.num_cps, '=', self.num_cps_pts)
        print('Number of quadrature points: ', self.num_hess_pts)
        print('Initial min distance: ',np.min(self.control_points[:,2]))
        print('Initial max distance: ',np.max(self.control_points[:,2]))
        print('Bspline order: ',self.order,'\n')

    def solve_energy_minimization(self, Lp=1., Ln=1., Lr=1e-3):
        A0  = self.get_basis(loc='surf',du=0,dv=0)
        Ax  = self.scaling[0]*self.get_basis(loc='surf',du=1,dv=0)
        Ay  = self.scaling[1]*self.get_basis(loc='surf',du=0,dv=1)
        Axx = self.scaling[0]*self.scaling[0]*self.get_basis(loc='hess',du=2,dv=0)
        Axy = self.scaling[0]*self.scaling[1]*self.get_basis(loc='hess',du=1,dv=1)
        Ayy = self.scaling[1]*self.scaling[1]*self.get_basis(loc='hess',du=0,dv=2)

        nx = self.surface_normals[np.newaxis, :, 0]
        ny = self.surface_normals[np.newaxis, :, 1]

        Ap = A0.transpose()@A0
        An = Ax.transpose()@Ax + Ay.transpose()@Ay
        Ar = Axx.transpose()@Axx + 2*Axy.transpose()@Axy + Ayy.transpose()@Ayy

        A  = Lp/self.num_surf_pts * Ap
        A += Ln/self.num_surf_pts * An
        A += Lr/self.num_hess_pts * Ar

        b  = Ln/self.num_surf_pts * (nx@Ax + ny@Ay)
        phi_QP, info = sps.linalg.cg(A,-b.flatten(),x0=self.control_points[:,2])
        print('conjugate gradient solver info: ',info,'\n')
        self.control_points[:,2] = phi_QP

        self.compute_errors()
        return

    def compute_errors(self):
        phi = self.compute_phi(self.surface_points)
        phi = phi/self.Bbox_diag
        print('Relative surface error: \n',
                'Max: ',np.max(abs(phi)),'\n',
                'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
        dx,dy = self.gradient_phi(self.surface_points)
        nx,ny = self.surface_normals[:,0], self.surface_normals[:,1]
        error = ((dx+nx)**2 + (dy+ny)**2)**(1/2)
        print('Gradient error: \n',
                'Max: ',np.max(error),'\n',
                'RMS: ',np.sqrt(np.sum(error**2)/len(error)))

    def visualize(self, phi_cps=None, res=500):
        if phi_cps is None:
            phi_cps = self.control_points[:,2]
        x = self.dimensions[0]
        y = self.dimensions[1]
        dataset = KDTree(self.surface_points)

        sns.set()
        plt.figure()
        ax = plt.axes()
        u = np.einsum('i,j->ij', np.linspace(0,1,res), np.ones(res)).flatten()
        v = np.einsum('i,j->ij', np.ones(res), np.linspace(0,1,res)).flatten()
        b = self.get_basis_matrix(u, v, 0, 0)
        xx = b.dot(self.control_points[:,0]).reshape(res,res)
        yy = b.dot(self.control_points[:,1]).reshape(res,res)
        phi = b.dot(phi_cps).reshape(res,res)
        ax.contour(xx,yy,phi,levels=[-2,-1,0,1,2],colors=['red','orange','green','blue','purple'])
        ax.plot(self.surface_points[:,0],self.surface_points[:,1],'k.',label='surface points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Contour Plot')
        ax.legend(loc='upper right')
        ax.set_xticks([x[0],np.sum(x)/2,x[1]])
        ax.set_yticks([y[0],np.sum(y)/2,y[1]])
        ax.axis('equal')

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(self.control_points[:,0],self.control_points[:,1],phi_cps,'k.')
        uu,vv = np.meshgrid(np.linspace(0,1,res),
                            np.linspace(0,1,res))
        b = self.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
        xx = b.dot(self.control_points[:,0]).reshape(res,res)
        yy = b.dot(self.control_points[:,1]).reshape(res,res)
        phi = b.dot(phi_cps).reshape(res,res)
        ax.contour(xx, yy, phi, levels=0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('$\Phi$')
        ax.set_title('Control Points')
        ax.set_xticks([x[0],np.sum(x)/2,x[1]])
        ax.set_yticks([y[0],np.sum(y)/2,y[1]])
        dx = np.diff(x)
        dy = np.diff(y)
        if dx > dy:
            lim = (x[0],x[1])
        else:
            lim = (y[0],y[1])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(-5,5)


        k=10
        rho=10
        plt.figure()
        ax = plt.axes()
        ones = np.ones(res)
        diag = np.linspace(0,1,res)
        basis = self.get_basis_matrix(diag, 0.5*ones, 0, 0)
        phi = basis.dot(phi_cps)
        xy = basis.dot(self.control_points[:,0:2])
        sdf = explicit_lsf(xy,dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C1', label='X-axis')
        ax.plot(diag, sdf, '--', color='C1')
        basis = self.get_basis_matrix(0.5*ones, diag, 0, 0)
        phi = basis.dot(phi_cps)
        xy = basis.dot(self.control_points[:,0:2])
        sdf = explicit_lsf(xy,dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C2', label='Y-axis')
        ax.plot(diag, sdf, '--', color='C2')
        # ax.axis([0,1,-8,8])
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([-5,0,5])
        ax.set_xlabel('Normalized Location')
        ax.set_ylabel('Phi')
        ax.set_title('Phi along 1D slices')
        ax.legend()
        plt.show()
        return

    def spatial_to_parametric(self,pts):
        param = np.empty((2,len(pts)))
        for i in range(2):
            param[i] = (pts[:,i] - self.dimensions[i,0]) / np.diff(self.dimensions[i,:])[0]
        return param[0], param[1]

    def initialize_control_points(self,k=6,rho=10):
        rangex = self.dimensions[0]
        rangey = self.dimensions[1]
        # Order 4, index 1.0: basis = [1/6, 4/6, 1/6]
        # Order 5, index 1.5: basis = [1/24, 11/24, 11/24, 1/24]
        # Order 6, index 2.0: basis = [1/120, 26/120, 66/120, 26/120, 1/120]
        Q = (self.order-2)/2
        A = np.array([[self.num_cps[0]-1-Q, Q],
                    [Q, self.num_cps[0]-1-Q]])
        b = np.array([rangex[0]*(self.num_cps[0]-1), rangex[1]*(self.num_cps[0]-1)])
        xn = np.linalg.solve(A,b)
        A = np.array([[self.num_cps[1]-1-Q, Q],
                    [Q, self.num_cps[1]-1-Q]])
        b = np.array([rangey[0]*(self.num_cps[1]-1), rangey[1]*(self.num_cps[1]-1)])
        yn = np.linalg.solve(A,b)

        cps = np.zeros((self.num_cps_pts, 3))
        cps[:, 0] = np.einsum('i,j->ij', np.linspace(xn[0],xn[1],self.num_cps[0]), np.ones(self.num_cps[1])).flatten()
        cps[:, 1] = np.einsum('i,j->ij', np.ones(self.num_cps[0]), np.linspace(yn[0],yn[1],self.num_cps[1])).flatten()

        dataset = KDTree(self.surface_points)
        phi = explicit_lsf(cps[:,0:2], dataset, self.surface_normals, k, rho)
        np.random.seed(1)
        phi += 1e-3*self.Bbox_diag*(2*np.random.rand(np.product(self.num_cps))-1)
        cps[:,2] = phi
        return cps

    def get_basis(self,loc='surf',du=0,dv=0):
        basis = self.get_basis_matrix(self.u[loc],self.v[loc],du,dv)
        return basis

    def compute_phi(self,pts):
        u,v = self.spatial_to_parametric(pts)

        b = self.get_basis_matrix(u,v,0,0)
        return b.dot(self.control_points[:,2])
    
    def gradient_phi(self,pts):
        dxy = np.diff(self.dimensions).flatten()
        scaling = 1/dxy
        u,v = self.spatial_to_parametric(pts)
        bdx = self.get_basis_matrix(u,v,1,0)
        dpdx = bdx.dot(self.control_points[:,2])*scaling[0]
        bdy = self.get_basis_matrix(u,v,0,1)
        dpdy = bdy.dot(self.control_points[:,2])*scaling[1]
        return dpdx, dpdy