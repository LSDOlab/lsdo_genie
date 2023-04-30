from lsdo_genie.bsplines.knot_vectors import standard_uniform_knot_vector
from lsdo_genie.bsplines.BsplineVolume import BsplineVolume
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lsdo_genie.utils.Hicken_Kaur import explicit_lsf
from skimage.measure import marching_cubes
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.sparse as sps
import seaborn as sns
import numpy as np
import time

class Genie3D(BsplineVolume):
    '''
    Base class for 3D geometric shape representation

    Attributes
    ----------
    verbose : bool
        Prints out information for troubleshooting
    '''

    def __init__(self, verbose=False):
        '''
        Initialize Genie2D instance
        '''
        self.u = dict()
        self.v = dict()
        self.w = dict()
        self.verbose = verbose

    def input_point_cloud(self, surface_points:np.ndarray=None, surface_normals:np.ndarray=None):
        '''
        Input the point cloud data

        Parameters
        ----------
        surface_points : np.ndarray(N,3)
            Positional data of the points in a point cloud
        surface_normals : np.ndarray(N,3)
            Normal vectors of the points in a point cloud
        '''
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.KDTree_dataset = KDTree(surface_points)

        # Number of points
        self.num_surface_points = int(len(surface_points))        
        # Minimum bounding box properties
        lower = np.min(self.surface_points,axis=0)
        upper = np.max(self.surface_points,axis=0)
        diff = upper-lower
        self.min_bbox = np.stack((lower, upper),axis=1)
        self.Bbox_diag = np.linalg.norm(diff)
        if self.verbose:
            # Printing
            print('Minimum bbox: \n',self.min_bbox)
            print('Minimum bbox diagonal: ',self.Bbox_diag)
            print('num_surface_points: ', self.num_surface_points,'\n')

    def config(self, domain:np.ndarray, max_control_points:int, order:int=4, min_ratio:float=0.5):
        '''
        Set up the Bspline for non-interference constraints

        Parameters
        ----------
        domain : np.ndarray(3,2)
            Lower and upper bounds in each dimension for the domain of interest
        max_control_points : int
            Maximum number of control points along the longest dimension
        min_ratio : float
            Minimum ratio from the shortest dimension to the longest dimension to maintain uniform control point spacing
        order : int
            Polynomial order of the Bspline        
        '''
        self.order = order
        if (self.surface_points[:,0] < domain[0,0]).any() \
            or (self.surface_points[:,0] > domain[0,1]).any() \
            or (self.surface_points[:,1] < domain[1,0]).any() \
            or (self.surface_points[:,1] > domain[1,1]).any() \
            or (self.surface_points[:,2] < domain[2,0]).any() \
            or (self.surface_points[:,2] > domain[2,1]).any():
            raise ValueError("surface points lie outside of the defined domain")
        self.domain = domain

        # Bounding box properties
        dxyz = np.diff(domain).flatten()/self.Bbox_diag
        self.scaling = 1/dxyz
        # Number of control points in each direction
        frac = dxyz / np.max(dxyz)
        num_cps = np.zeros(3,dtype=int)
        for i,ratio in enumerate(frac):
            if ratio < min_ratio:
                ratio = min_ratio
            num_cps[i] = int(ratio*max_control_points)
            # num_cps[i] = int((ratio*max_cps)+order-1)
            # num_cps[i] = 3*int((ratio*max_control_points)/3)
        self.num_cps = num_cps
        self.num_cps_pts  = int(np.product(self.num_cps))

        # Get initial control points
        self.control_points = self.initialize_control_points(k=6,rho=10)
        # Standard uniform knot vectors
        kv_u = standard_uniform_knot_vector(num_cps[0], order)
        kv_v = standard_uniform_knot_vector(num_cps[1], order)
        kv_w = standard_uniform_knot_vector(num_cps[2], order)
        # Define Bspline Volume object
        super().__init__('Bspline',order,order,order,kv_u,kv_v,kv_w,num_cps)

        # Surface points for data terms (Ep, En)
        # self.u['surf'], self.v['surf'], self.w['surf'] = self.spatial_to_parametric(self.surface_points)
        # Quadrature points for regulation term (Er)
        temp_u, temp_v, temp_w = self.spatial_to_parametric(self.control_points[:,0:3])
        mask = np.argwhere(
            (temp_u>=0)*(temp_u<=1)*\
            (temp_v>=0)*(temp_v<=1)*\
            (temp_w>=0)*(temp_w<=1)
        )
        self.u['hess'], self.v['hess'], self.w['hess'] = temp_u[mask].flatten(), temp_v[mask].flatten(), temp_w[mask].flatten()
        self.num_hess_pts = int(len(self.u['hess']))

        if self.verbose:
            # Printing
            print('Bspline box: \n',self.domain)
            print('Control point grid: ', self.num_cps, '=', self.num_cps_pts)
            print('Number of quadrature points: ', self.num_hess_pts)
            print('Initial min distance: ',np.min(self.control_points[:,3]))
            print('Initial max distance: ',np.max(self.control_points[:,3]))
            print('Bspline order: ',self.order,'\n')

    def solve_energy_minimization(self, Lp:float=1., Ln:float=1., Lr:float=1e-3, maxiter=None):
        '''
        Solve the energy minimization problem

        Parameters
        ----------
        Lp : float
            Depricated weighting parameter for zero level set point energy term
        Ln : float
            Weighting parameter for the normal vectors energy term
        Lr : float
            Weighting parameter for the regularization energy term
        '''
        A0  = self.get_basis(loc='surf',du=0,dv=0,dw=0)
        Ax  = self.scaling[0]*self.get_basis(loc='surf',du=1,dv=0,dw=0)
        Ay  = self.scaling[1]*self.get_basis(loc='surf',du=0,dv=1,dw=0)
        Az  = self.scaling[2]*self.get_basis(loc='surf',du=0,dv=0,dw=1)
        Axx = self.scaling[0]*self.scaling[0]*self.get_basis(loc='hess',du=2,dv=0,dw=0)
        Axy = self.scaling[0]*self.scaling[1]*self.get_basis(loc='hess',du=1,dv=1,dw=0)
        Ayy = self.scaling[1]*self.scaling[1]*self.get_basis(loc='hess',du=0,dv=2,dw=0)
        Axz = self.scaling[0]*self.scaling[2]*self.get_basis(loc='hess',du=1,dv=0,dw=1)
        Ayz = self.scaling[1]*self.scaling[2]*self.get_basis(loc='hess',du=0,dv=1,dw=1)
        Azz = self.scaling[2]*self.scaling[2]*self.get_basis(loc='hess',du=0,dv=0,dw=2)

        nx = self.surface_normals[np.newaxis, :, 0]
        ny = self.surface_normals[np.newaxis, :, 1]
        nz = self.surface_normals[np.newaxis, :, 2]

        Ap = A0.T@A0
        An = Ax.T@Ax + Ay.T@Ay + Az.T@Az
        Ar = Axx.T@Axx + 2*Axy.T@Axy + Ayy.T@Ayy + 2*Axz.T@Axz + 2*Ayz.T@Ayz + Azz.T@Azz

        A  = Lp/self.num_surface_points * Ap
        A += Ln/self.num_surface_points * An
        A += Lr/self.num_hess_pts * Ar

        b  = Ln/self.num_surface_points * (nx@Ax + ny@Ay + nz@Az)

        t1 = time.perf_counter()
        phi_solved, info = sps.linalg.bicgstab(
            A,
            -b.flatten(),
            x0=self.control_points[:,3]/self.Bbox_diag,
            maxiter=maxiter,
            atol=1e-6
        )
        self.timetosolve = time.perf_counter() - t1
        phi_solved = phi_solved*self.Bbox_diag
        if info != 0:
            raise Exception(f"Conjugate gradient solver terminated with bad exit code: {info}")

        if self.verbose:
            print(f'Time to solve: {self.timetosolve:1.3f}',' seconds\n')
            print('Final min distance: ',np.min(phi_solved))
            print('Final max distance: ',np.max(phi_solved))
        self.control_points[:,3] = phi_solved

        if self.verbose:
            self.compute_errors()
        return

    def compute_errors(self):
        '''
        Compute error values and print to terminal
        '''
        phi = self.compute_phi(self.surface_points)
        phi = phi/self.Bbox_diag
        print('Relative surface error: \n',
                f'  Max: {np.max(abs(phi)):1.3e}\n',
                f'  RMS: {np.sqrt(np.sum(phi**2)/len(phi)):1.3e}')
        dx,dy,dz = self.gradient_phi(self.surface_points)
        nx,ny,nz = self.surface_normals[:,0], self.surface_normals[:,1], self.surface_normals[:,2]
        error = ((dx+nx)**2 + (dy+ny)**2 + (dz+nz)**2)**(1/2)
        print('Gradient error: \n',
                f'  Max: {np.max(error):1.3e}','\n',
                f'  RMS: {np.sqrt(np.sum(error**2)/len(error)):1.3e}')

    def spatial_to_parametric(self,pts:np.ndarray):
        '''
        Convert (x,y,z) coordinates to (u,v,w) coordinates

        Parameters
        ----------
        pts : np.ndarray(N,3)
            The points to convert to parametric coordinates
        
        Returns
        ----------
        u : np.ndarray(N,)
            The 'u' parametric coordinates
        v : np.ndarray(N,)
            The 'v' parametric coordinates
        w : np.ndarray(N,)
            The 'w' parametric coordinates
        '''
        param = np.empty((3,len(pts)))
        for i in range(3):
            param[i] = (pts[:,i] - self.domain[i,0]) / np.diff(self.domain[i,:])[0]
        return param[0], param[1], param[2]

    def initialize_control_points(self,k:int=10,rho:float=10.):
        '''
        Initialize the control points using Hicken and Kaur's explicit method

        Parameters
        ----------
        k : int
            Numer of nearest neighbors to include in the function
        rho : float
            Smoothing parameter

        Returns
        ----------
        cps : np.ndarray(Ncp,4)
            The initial control points with (x,y,z,phi) values at each point 
        '''
        rangex = self.domain[0]
        rangey = self.domain[1]
        rangez = self.domain[2]
        # Order 3, index 0.5: basis = [1/2, 1/2]
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
        A = np.array([[self.num_cps[2]-1-Q, Q],
                    [Q, self.num_cps[2]-1-Q]])
        b = np.array([rangez[0]*(self.num_cps[2]-1), rangez[1]*(self.num_cps[2]-1)])
        zn = np.linalg.solve(A,b)

        cps = np.zeros((np.product(self.num_cps), 4))
        cps[:, 0] = np.einsum('i,j,k->ijk', np.linspace(xn[0],xn[1],self.num_cps[0]), np.ones(self.num_cps[1]),np.ones(self.num_cps[2])).flatten()
        cps[:, 1] = np.einsum('i,j,k->ijk', np.ones(self.num_cps[0]), np.linspace(yn[0],yn[1],self.num_cps[1]),np.ones(self.num_cps[2])).flatten()
        cps[:, 2] = np.einsum('i,j,k->ijk', np.ones(self.num_cps[0]), np.ones(self.num_cps[1]),np.linspace(zn[0],zn[1],self.num_cps[2])).flatten()

        phi = explicit_lsf(cps[:,0:3],self.KDTree_dataset,self.surface_normals,k,rho)
        np.random.seed(1)
        phi += 1e-3*self.Bbox_diag*(2*np.random.rand(np.product(self.num_cps))-1)
        cps[:, 3] = phi
        return cps

    def get_basis(self,loc='surf',du=0,dv=0,dw=0):
        '''
        Convenience function for building the basis matrix for different points

        Parameters
        ----------
        loc : str
            'surf' for surface points or 'hess' for quadrature points to evaluate the hessian
        du : int
            Derivative in the 'u' direction
        dv : int
            Derivative in the 'v' direction
        dw : int
            Derivative in the 'w' direction
        
        Returns
        ----------
        basis : sps.csc_matrix(N,Ncp)
            The basis matrix that can be multiplied with the control points to get (N,) output values
        '''
        if loc=='surf':
            u, v, w = self.spatial_to_parametric(self.surface_points)
        else:
            u, v, w = self.u[loc], self.v[loc], self.w[loc]
        basis = self.get_basis_matrix(u,v,w,du,dv,dw)
        return basis

    def compute_phi(self,pts):
        '''
        Compute the phi value at a set of points

        Parameters
        ----------
        pts : np.ndarray(N,3)
            Set of points to be evaluated
        
        Returns
        ----------
        phi : np.ndarray(N,)
            The phi values at each input point
        '''
        u,v,w = self.spatial_to_parametric(pts)
        if (u.min()<0) or (v.min()<0) or (w.min()<0):
            raise ValueError(f"A point lies below the bounds of the Bspline: {u.min()},{v.min()},{w.min()}")
        if (u.max()>1) or (v.max()>1) or (w.max()>1):
            raise ValueError(f"A point lies above the bounds of the Bspline: {u.max()},{v.max()},{w.max()}")
        b = self.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.control_points[:,3])

    def gradient_phi(self,pts:np.ndarray):
        '''
        Compute the phi value at a set of points

        Parameters
        ----------
        pts : np.ndarray(N,3)
            Set of points to be evaluated

        Returns
        ----------
        dpdx : np.ndarray(N,)
            The derivative of phi with respect to the x-coordinate
        dpdy : np.ndarray(N,)
            The derivative of phi with respect to the y-coordinate
        dpdz : np.ndarray(N,)
            The derivative of phi with respect to the z-coordinate
        '''
        u,v,w = self.spatial_to_parametric(pts)
        bdx = self.get_basis_matrix(u,v,w,1,0,0)
        dpdx = bdx.dot(self.control_points[:,3])*self.scaling[0]/self.Bbox_diag
        bdy = self.get_basis_matrix(u,v,w,0,1,0)
        dpdy = bdy.dot(self.control_points[:,3])*self.scaling[1]/self.Bbox_diag        
        bdz = self.get_basis_matrix(u,v,w,0,0,1)
        dpdz = bdz.dot(self.control_points[:,3])*self.scaling[2]/self.Bbox_diag
        return dpdx, dpdy, dpdz

    def visualize(self, phi_cps=None, res=30):
        '''
        Visualize the isocontours, control points in 3D, and phi values along the x-y axes

        Parameters
        ----------
        res : int
            The resolution in each dimension to evaluate the function for visualization
        '''
        if phi_cps is None:
            phi_cps = self.control_points[:,3]
        gold = (198/255, 146/255, 20/255)
        x = self.domain[0]
        y = self.domain[1]
        z = self.domain[2]

        sns.set()
        plt.figure()
        ax = plt.axes(projection='3d')
        u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
        v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
        w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
        basis = self.get_basis_matrix(u, v, w, 0, 0, 0)
        phi = basis.dot(phi_cps).reshape((res,res,res))
        verts, faces,_,_ = marching_cubes(phi, 0)
        verts = verts*np.diff(self.domain).flatten()/(res-1) + self.domain[:,0]
        level_set = Poly3DCollection(verts[faces],linewidth=0.25,alpha=1,facecolor=gold,edgecolor='k')
        ax.add_collection3d(level_set)
        # ax.plot(self.surface_points[:,0],self.surface_points[:,1],self.surface_points[:,2],
        #         'k.',label='surface points')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title('Zero level set')
        ax.set_xticks([x[0],(x[1]+x[0])/2,x[1]])
        ax.set_yticks([y[0],(y[1]+y[0])/2,y[1]])
        ax.set_zticks([z[0],(z[1]+z[0])/2,z[1]])
        center = np.mean(self.domain,axis=1)
        d = np.max(np.diff(self.domain,axis=1))
        ax.set_xlim(center[0]-d/2, center[0]+d/2)
        ax.set_ylim(center[1]-d/2, center[1]+d/2)
        ax.set_zlim(center[2]-d/2, center[2]+d/2)
        # ax.axis('equal')

        k=10
        rho=10
        plt.figure()
        ax = plt.axes()
        res = 200
        ones = np.ones(res)
        diag = np.linspace(0,1,res)
        basis = self.get_basis_matrix(diag, 0.5*ones, 0.5*ones, 0, 0, 0)
        phi = basis.dot(phi_cps)
        xyz = basis.dot(self.control_points[:,0:3])
        sdf = explicit_lsf(xyz,self.KDTree_dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C1', label='X-axis')
        ax.plot(diag, sdf, '--', color='C1', label='exact')
        basis = self.get_basis_matrix(0.5*ones, diag, 0.5*ones, 0, 0, 0)
        phi = basis.dot(phi_cps)
        xyz = basis.dot(self.control_points[:,0:3])
        sdf = explicit_lsf(xyz,self.KDTree_dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C2', label='Y-axis')
        ax.plot(diag, sdf, '--', color='C2', label='exact')
        basis = self.get_basis_matrix(0.5*ones, 0.5*ones, diag, 0, 0, 0)
        phi = basis.dot(phi_cps)
        xyz = basis.dot(self.control_points[:,0:3])
        sdf = explicit_lsf(xyz,self.KDTree_dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C3', label='Z-axis')
        ax.plot(diag, sdf, '--', color='C3', label='exact')
        # ax.axis([0,1,phi_cps.min(),phi_cps.max()])
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([phi_cps.min(),0,phi_cps.max()])
        ax.set_xlabel('Normalized Location')
        ax.set_ylabel('Phi')
        ax.set_title('Phi along 1D slices')
        ax.legend()
        plt.show()
        return
