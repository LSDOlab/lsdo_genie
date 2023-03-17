from lsdo_genie.core.Bsplines.knot_vectors import standard_uniform_knot_vector
from lsdo_genie.core.Bsplines.Bspline_Volume import BSplineVolume
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lsdo_genie.utils.Hicken_Kaur import explicit_lsf
from skimage.measure import marching_cubes
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.sparse as sps
import seaborn as sns
import numpy as np
import time

class Genie3D(BSplineVolume):
    def __init__(self):
        self.u = dict()
        self.v = dict()
        self.w = dict()

    def input_point_cloud(self, surface_points:np.ndarray=None, surface_normals:np.ndarray=None):
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
        # Printing
        print('Minimum bbox: \n',self.min_bbox)
        print('Minimum bbox diagonal: ',self.Bbox_diag)
        print('num_surface_points: ', self.num_surface_points,'\n')
    
    def config(self, dimensions:np.ndarray, max_control_points:int, order:int=4, min_ratio:float=0.5):
        self.order = order
        if (self.surface_points[:,0] < dimensions[0,0]).any() \
            or (self.surface_points[:,0] > dimensions[0,1]).any() \
            or (self.surface_points[:,1] < dimensions[1,0]).any() \
            or (self.surface_points[:,1] > dimensions[1,1]).any() \
            or (self.surface_points[:,2] < dimensions[2,0]).any() \
            or (self.surface_points[:,2] > dimensions[2,1]).any():
            raise ValueError("surface points lie outside of the defined dimensions")
        self.dimensions = dimensions

        # Bounding box properties
        dxyz = np.diff(dimensions).flatten()
        self.scaling = 1/dxyz
        # Number of control points in each direction
        frac = dxyz / np.max(dxyz)
        num_cps = np.zeros(3,dtype=int)
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
        kv_w = standard_uniform_knot_vector(num_cps[2], order)
        # Define Bspline Volume object
        super().__init__('Bspline',order,order,order,kv_u,kv_v,kv_w,num_cps)
        
        # Surface points for data terms (Ep, En)
        self.u['surf'], self.v['surf'], self.w['surf'] = self.spatial_to_parametric(self.surface_points)
        # Quadrature points for regulation term (Er)
        temp_u, temp_v, temp_w = self.spatial_to_parametric(self.control_points[:,0:3])
        mask = np.argwhere(
            (temp_u>=0)*(temp_u<=1)*\
            (temp_v>=0)*(temp_v<=1)*\
            (temp_w>=0)*(temp_w<=1)
        )
        self.u['hess'], self.v['hess'], self.w['hess'] = temp_u[mask].flatten(), temp_v[mask].flatten(), temp_w[mask].flatten()
        self.num_hess_pts = int(len(self.u['hess']))

        # Printing
        print('Bspline box: \n',self.dimensions)
        print('Control point grid: ', self.num_cps, '=', self.num_cps_pts)
        print('Number of quadrature points: ', self.num_hess_pts)
        print('Initial min distance: ',np.min(self.control_points[:,3]))
        print('Initial max distance: ',np.max(self.control_points[:,3]))
        print('Bspline order: ',self.order,'\n')

    def solve_energy_minimization(self, Lp=1., Ln=1., Lr=1e-3):
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
        phi_QP, info = sps.linalg.cg(A,-b.flatten())
        print('conjugate gradient solver info: ',info,'\n')
        self.control_points[:,3] = phi_QP
        self.compute_errors()
        return

    def compute_errors(self):
        phi = self.compute_phi(self.surface_points)
        phi = phi/self.Bbox_diag
        print('Relative surface error: \n',
                'Max: ',np.max(abs(phi)),'\n',
                'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
        dx,dy,dz = self.gradient_phi(self.surface_points)
        nx,ny,nz = self.surface_normals[:,0], self.surface_normals[:,1], self.surface_normals[:,2]
        error = ((dx+nx)**2 + (dy+ny)**2 + (dz+nz)**2)**(1/2)
        print('Gradient error: \n',
                'Max: ',np.max(error),'\n',
                'RMS: ',np.sqrt(np.sum(error**2)/len(error)))

    def spatial_to_parametric(self,pts):
        param = np.empty((3,len(pts)))
        for i in range(3):
            param[i] = (pts[:,i] - self.dimensions[i,0]) / np.diff(self.dimensions[i,:])[0]
        return param[0], param[1], param[2]

    def initialize_control_points(self,k=10,rho=10):
        rangex = self.dimensions[0]
        rangey = self.dimensions[1]
        rangez = self.dimensions[2]
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
        basis = self.get_basis_matrix(self.u[loc],self.v[loc],self.w[loc],du,dv,dw)
        return basis

    def compute_phi(self,pts):
        u,v,w = self.spatial_to_parametric(pts)
        b = self.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.control_points[:,3])
    
    def gradient_phi(self,pts):
        u,v,w = self.spatial_to_parametric(pts)
        bdx = self.get_basis_matrix(u,v,w,1,0,0)
        dpdx = bdx.dot(self.control_points[:,3])*self.scaling[0]
        bdy = self.get_basis_matrix(u,v,w,0,1,0)
        dpdy = bdy.dot(self.control_points[:,3])*self.scaling[1]        
        bdz = self.get_basis_matrix(u,v,w,0,0,1)
        dpdz = bdz.dot(self.control_points[:,3])*self.scaling[2]
        return dpdx, dpdy, dpdz

    def visualize(self, phi_cps=None, res=30):
        if phi_cps is None:
            phi_cps = self.control_points[:,3]
        gold = (198/255, 146/255, 20/255)
        x = self.dimensions[0]
        y = self.dimensions[1]
        z = self.dimensions[2]

        sns.set()
        plt.figure()
        ax = plt.axes(projection='3d')
        u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
        v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
        w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
        basis = self.get_basis_matrix(u, v, w, 0, 0, 0)
        phi = basis.dot(phi_cps).reshape((res,res,res))
        verts, faces,_,_ = marching_cubes(phi, 0)
        verts = verts*np.diff(self.dimensions).flatten()/(res-1) + self.dimensions[:,0]
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
        center = np.mean(self.dimensions,axis=1)
        d = np.max(np.diff(self.dimensions,axis=1))
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
        ax.plot(diag, sdf, '--', color='C1')
        basis = self.get_basis_matrix(0.5*ones, diag, 0.5*ones, 0, 0, 0)
        phi = basis.dot(phi_cps)
        xyz = basis.dot(self.control_points[:,0:3])
        sdf = explicit_lsf(xyz,self.KDTree_dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C2', label='Y-axis')
        ax.plot(diag, sdf, '--', color='C2')
        basis = self.get_basis_matrix(0.5*ones, 0.5*ones, diag, 0, 0, 0)
        phi = basis.dot(phi_cps)
        xyz = basis.dot(self.control_points[:,0:3])
        sdf = explicit_lsf(xyz,self.KDTree_dataset,self.surface_normals,k,rho)
        ax.plot(diag, phi, '-', color='C3', label='Z-axis')
        ax.plot(diag, sdf, '--', color='C3')
        # ax.axis([0,1,phi_cps.min(),phi_cps.max()])
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([phi_cps.min(),0,phi_cps.max()])
        ax.set_xlabel('Normalized Location')
        ax.set_ylabel('Phi')
        ax.set_title('Phi along 1D slices')
        ax.legend()
        plt.show()
        return
