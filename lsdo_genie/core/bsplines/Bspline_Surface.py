import numpy as np
import scipy.sparse as sps

from lsdo_genie.core.bsplines.cython.basis_matrix_surface_py import get_basis_surface_matrix

class BSplineSurface:
    '''
    Base class for Bspline Surfaces
    '''
    def __init__(self, name, order_u, order_v, knots_u, knots_v, shape):
        self.name = name
        self.order_u = order_u
        self.order_v = order_v
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.shape_u = int(shape[0])
        self.shape_v = int(shape[1])
        self.num_control_points = int(np.product(shape))

    def get_basis_matrix(self, u_vec, v_vec, du, dv):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_surface_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            self.order_v, self.shape_v, dv, v_vec, self.knots_v,
            len(u_vec), data, row_indices, col_indices
            )

        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis


def plot_support():
    import matplotlib.pyplot as plt

    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector

    ### test support
    order = 4
    num_cps = [99,30]
    
    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    
    cps = np.zeros((np.product(num_cps), 3))
    cps[:, 0] = np.einsum('i,j->ij', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1])).flatten()
    cps[:, 1] = np.einsum('i,j->ij', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1])).flatten()
    cps[:, 2] = np.random.rand(np.product(num_cps))
    
    Surface = BSplineSurface('name',order,order,kv_u,kv_v,num_cps)

    u_vec = kv_u[(order-1):(num_cps[0] + 1)]
    u_vec = u_vec[::(order-1)]
    v_vec = kv_v[(order-1):(num_cps[1] + 1)]
    v_vec = v_vec[::(order-1)]

    uu,vv = np.meshgrid(
        u_vec, v_vec
    )

    basis = Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,2)
    basis += Surface.get_basis_matrix(uu.flatten(),vv.flatten(),2,0)
    basis += Surface.get_basis_matrix(uu.flatten(),vv.flatten(),1,1)
    x = basis.toarray().transpose()
    print(x)
    cond = True
    for ii,i in enumerate(x):
        ind = np.argwhere(i!=0)
        if len(ind)!=1:
            print(ii)
            cond = False
    print(cond)
    plt.spy(basis.toarray().transpose())
    plt.show()

def comp_time():
    import time
    import matplotlib.pyplot as plt
    
    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector

    ### plot timing performance
    order = 4
    num_cps = [100,30]
    
    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    
    cps = np.zeros((np.product(num_cps), 3))
    cps[:, 0] = np.einsum('i,j->ij', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1])).flatten()
    cps[:, 1] = np.einsum('i,j->ij', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1])).flatten()
    cps[:, 2] = np.random.rand(np.product(num_cps))
    
    Surface = BSplineSurface('name',order,order,kv_u,kv_v,num_cps)
    
    res = 15
    time_set = np.zeros(res)
    time_data = np.logspace(1,5,res,dtype=int)
    for i,num_pts in enumerate(time_data):
        u_vec = np.linspace(0,1,num_pts)
        v_vec = np.linspace(0,1,num_pts)
        t1 = time.perf_counter()
        basis = Surface.get_basis_matrix(u_vec,v_vec,0,0)
        p = basis.dot(cps[:,2])
        t2 = time.perf_counter()
        time_set[i] = t2-t1
        print(num_pts,t2-t1)
    
    plt.loglog(time_data,time_set/time_data)
    plt.title("eval time per point")
    plt.show()

if __name__ == '__main__':
    comp_time()
    # plot_support()
    