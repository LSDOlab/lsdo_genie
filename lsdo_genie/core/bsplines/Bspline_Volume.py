import numpy as np
import scipy.sparse as sps

from lsdo_genie.core.bsplines.cython.basis_matrix_volume_py import get_basis_volume_matrix

class BSplineVolume:
    '''
    Base class for Bspline Volumes
    '''
    def __init__(self, name, order_u, order_v, order_w, knots_u, knots_v, knots_w, shape):
        self.name = name
        self.order_u = order_u
        self.order_v = order_v
        self.order_w = order_w
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.knots_w = knots_w
        self.shape_u = int(shape[0])
        self.shape_v = int(shape[1])
        self.shape_w = int(shape[2])
        self.num_control_points = int(np.product(shape))
        
    def get_basis_matrix(self, u_vec, v_vec, w_vec, du, dv, dw):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            self.order_v, self.shape_v, dv, v_vec, self.knots_v,
            self.order_w, self.shape_w, dw, w_vec, self.knots_w,
            len(u_vec), data, row_indices, col_indices
            )
            
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    
    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector
    
    ### plot timing performance
    res = 160
    order = 4
    num_cps = [42,42,42]

    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    kv_w = std_uniform_knot_vec(num_cps[2],order)

    cps = np.zeros((np.product(num_cps), 4))
    cps[:, 0] = np.einsum('i,j,k->ijk', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1]), np.ones(num_cps[2])).flatten()
    cps[:, 1] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1]), np.ones(num_cps[2])).flatten()
    cps[:, 2] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.ones(num_cps[1]), np.linspace(0,1,num_cps[2])).flatten()
    cps[:, 3] = np.random.rand(np.product(num_cps))
    
    sns.set()
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=12)
    plt.rc('axes', labelsize=16)

    plt.figure(figsize=(6,5),dpi=140)
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Evaluation Time (sec)')
    num_pts = np.logspace(np.log10(6),np.log10(3500),res,dtype=int)
    for order in range(4,7):
        kv_u = std_uniform_knot_vec(num_cps[0],order)
        kv_v = std_uniform_knot_vec(num_cps[1],order)
        kv_w = std_uniform_knot_vec(num_cps[2],order)
        BSpline = BSplineVolume('name',order,order,order,kv_u,kv_v,kv_w,num_cps)
        
        time_set = np.zeros(res)
        for i,i_num_pts in enumerate(num_pts):
            u_vec = np.linspace(0,1,i_num_pts)
            v_vec = np.linspace(0,1,i_num_pts)
            w_vec = np.linspace(0,1,i_num_pts)

            t1 = time.perf_counter()
            basis = BSpline.get_basis_matrix(u_vec,v_vec,w_vec,0,0,0)
            p = basis.dot(cps[:,2])
            t2 = time.perf_counter()
            
            time_set[i] = t2-t1
        plt.loglog(num_pts,time_set,label='Order '+str(order))
    plt.legend()

    plt.tight_layout()
    plt.show()
    