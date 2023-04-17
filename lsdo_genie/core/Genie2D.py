from lsdo_genie.bsplines.BsplineSurface import BsplineSurface
from lsdo_genie.bsplines.knot_vectors import standard_uniform_knot_vector
from lsdo_genie.utils.Hicken_Kaur import explicit_lsf
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.sparse as sps
import seaborn as sns
import numpy as np
import time

class Genie2D(BsplineSurface):
    '''
    Base class for 2D geometric shape representation.
    
    Attributes
    ----------
    verbose : bool
        Prints out information for troubleshooting
    '''

    def __init__(self, verbose=False):
        self.u = dict()
        self.v = dict()
        self.verbose = verbose

    def input_point_cloud(self, surface_points:np.ndarray=None, surface_normals:np.ndarray=None):
        '''
        Input the point cloud data

        Parameters
        ----------
        surface_points : np.ndarray(N,2)
            Positional data of the points in a point cloud
        surface_normals : np.ndarray(N,2)
            Normal vectors of the points in a point cloud
        '''
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
        if self.verbose:
            # Printing
            print('Minimum bbox: \n',self.min_bbox)
            print('Minimum bbox diagonal: ',self.Bbox_diag)
            print('num_surface_points: ', self.num_surf_pts,'\n')
    
    def config(self, domain:np.ndarray, max_control_points:int, min_ratio:float=0.5, order:int=4):
        '''
        Set up the Bspline for non-interference constraints

        Parameters
        ----------
        domain : np.ndarray(2,2)
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
            or (self.surface_points[:,1] > domain[1,1]).any():
            raise ValueError("surface points lie outside of the defined domain")
        self.domain = domain

        # Bounding box properties
        dxy = np.diff(domain).flatten()
        self.scaling = 1/dxy
        # Number of control points in each direction
        frac = dxy / np.max(dxy)
        num_cps = np.zeros(2,dtype=int)
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

        if self.verbose:
            # Printing
            print('Bspline box: \n',self.domain)
            print('Control point grid: ', self.num_cps, '=', self.num_cps_pts)
            print('Number of quadrature points: ', self.num_hess_pts)
            print('Initial min distance: ',np.min(self.control_points[:,2]))
            print('Initial max distance: ',np.max(self.control_points[:,2]))
            print('Bspline order: ',self.order,'\n')

    def solve_energy_minimization(self, Lp:float=1., Ln:float=1., Lr:float=1e-3):
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

        t1 = time.perf_counter()
        phi_solved, info = sps.linalg.cg(A,-b.flatten(),x0=self.control_points[:,2])
        self.timetosolve = time.perf_counter() - t1
        if info != 0:
            raise Exception(f"Conjugate gradient solver terminated with bad exit code: {info}")

        if self.verbose:
            print(f'Time to solve: {self.timetosolve:1.3f}',' seconds\n')
            print('Final min distance: ',np.min(phi_solved))
            print('Final max distance: ',np.max(phi_solved))
        self.control_points[:,2] = phi_solved

        if self.verbose:
            self.compute_errors()

    def compute_errors(self):
        '''
        Compute error values and print to terminal
        '''
        phi = self.compute_phi(self.surface_points)
        phi = phi/self.Bbox_diag
        print('Relative surface error: \n',
                f'  Max: {np.max(abs(phi)):1.3e}\n',
                f'  RMS: {np.sqrt(np.sum(phi**2)/len(phi)):1.3e}')
        dx,dy = self.gradient_phi(self.surface_points)
        nx,ny = self.surface_normals[:,0], self.surface_normals[:,1]
        error = ((dx+nx)**2 + (dy+ny)**2)**(1/2)
        print('Gradient error: \n',
                f'  Max: {np.max(error):1.3e}','\n',
                f'  RMS: {np.sqrt(np.sum(error**2)/len(error)):1.3e}')

    def visualize(self, res:int=300):
        '''
        Visualize the isocontours, control points in 3D, and phi values along the x-y axes

        Parameters
        ----------
        res : int
            The resolution in each dimension to evaluate the function for visualization
        '''
        phi_cps = self.control_points[:,2]
        x = self.domain[0]
        y = self.domain[1]
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
        ax.contour(xx,yy,phi,levels=[-0.1*self.Bbox_diag,-0.05*self.Bbox_diag,0,0.05*self.Bbox_diag,0.1*self.Bbox_diag],colors=['red','orange','green','blue','purple'])
        ax.plot(self.surface_points[:,0],self.surface_points[:,1],'k.',label='surface points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Isocontours [0,+/-0.05,+/-0.1] BBox Diagonal: {self.Bbox_diag:.3f}')
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
        ax.set_zlim(phi_cps.min(),phi_cps.max())


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
        ax.plot(diag, sdf, '--', color='C1', label='exact')
        basis = self.get_basis_matrix(0.5*ones, diag, 0, 0)
        phi = basis.dot(phi_cps)
        xy = basis.dot(self.control_points[:,0:2])
        sdf = explicit_lsf(xy,dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C2', label='Y-axis')
        ax.plot(diag, sdf, '--', color='C2', label='exact')
        # ax.axis([0,1,-8,8])
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([-5,0,5])
        ax.set_xlabel('Normalized Location')
        ax.set_ylabel('Phi')
        ax.set_title('Phi along 1D slices')
        ax.legend()
        plt.show()

    def spatial_to_parametric(self,pts:np.ndarray):
        '''
        Convert (x,y) coordinates to (u,v) coordinates

        Parameters
        ----------
        pts : np.ndarray(N,2)
            The points to convert to parametric coordinates
        
        Returns
        ----------
        u : np.ndarray(N,)
            The 'u' parametric coordinates
        v : np.ndarray(N,)
            The 'v' parametric coordinates
        '''
        param = np.empty((2,len(pts)))
        for i in range(2):
            param[i] = (pts[:,i] - self.domain[i,0]) / np.diff(self.domain[i,:])[0]
        return param[0], param[1]

    def initialize_control_points(self,k:int=6,rho:float=10):
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
        cps : np.ndarray(Ncp,3)
            The initial control points with (x,y,phi) values at each point 
        '''
        rangex = self.domain[0]
        rangey = self.domain[1]
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
        
        Returns
        ----------
        basis : sps.csc_matrix(N,Ncp)
            The basis matrix that can be multiplied with the control points to get (N,) output values
        '''
        basis = self.get_basis_matrix(self.u[loc],self.v[loc],du,dv)
        return basis

    def compute_phi(self,pts:np.ndarray):
        '''
        Compute the phi value at a set of points

        Parameters
        ----------
        pts : np.ndarray(N,2)
            Set of points to be evaluated
        
        Returns
        ----------
        phi : np.ndarray(N,)
            The phi values at each input point
        '''
        u,v = self.spatial_to_parametric(pts)
        if (u.min()<0) or (v.min()<0):
            raise ValueError(f"Points are below the bounds of the Bspline. Parameteric coordinates: {u.min()},{v.min()}")
        if (u.max()>1) or (v.max()>1):
            raise ValueError(f"Points are above the bounds of the Bspline. Parameteric coordinates: {u.max()},{v.max()}")
        b = self.get_basis_matrix(u,v,0,0)
        return b.dot(self.control_points[:,2])
    
    def gradient_phi(self,pts:np.ndarray):
        '''
        Compute the phi value at a set of points

        Parameters
        ----------
        pts : np.ndarray(N,2)
            Set of points to be evaluated

        Returns
        ----------
        dpdx : np.ndarray(N,)
            The derivative of phi with respect to the x-coordinate
        dpdy : np.ndarray(N,)
            The derivative of phi with respect to the y-coordinate
        '''
        u,v = self.spatial_to_parametric(pts)
        bdx = self.get_basis_matrix(u,v,1,0)
        dpdx = bdx.dot(self.control_points[:,2])*self.scaling[0]
        bdy = self.get_basis_matrix(u,v,0,1)
        dpdy = bdy.dot(self.control_points[:,2])*self.scaling[1]
        return dpdx, dpdy
