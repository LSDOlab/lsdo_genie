import numpy as np
import scipy.sparse as sps

from lsdo_genie.bsplines.cython.basis_matrix_volume_py import get_basis_volume_matrix

class BsplineVolume:
    '''
    Base class for B-spline Volumes

    Attributes
    ----------
    name : str
        A nickname for the B-spline Curve
    order_u : int
        B-spline polynomial order in the 'u' direction
    knots_u : int
        Knot vector in the 'u' direction
    order_v : int
        B-spline polynomial order in the 'v' direction
    knots_v : int
        Knot vector in the 'v' direction
    order_w : int
        B-spline polynomial order in the 'w' direction
    knots_w : int
        Knot vector in the 'w' direction
    shape : tuple
        Shape for the B-spline control points (num_u,)
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
        '''
        Builds the basis matrix for a given set of points

        Parameters
        ----------
        u_vec : np.ndarray(N,)
            Vector storing the 'u' cooridinates of a set of input points
        du : int
            Derivative in the 'u' direction
        v_vec : np.ndarray(N,)
            Vector storing the 'v' cooridinates of a set of input points
        dv : int
            Derivative in the 'v' direction
        w_vec : np.ndarray(N,)
            Vector storing the 'w' cooridinates of a set of input points
        dw : int
            Derivative in the 'w' direction

        Returns
        ----------
        basis : sps.csc_matrix(N,Ncp)
            The basis matrix that can be multiplied with the control points to get (N,) output values
        '''
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        u_i_starts = np.floor(u_vec*(self.shape_u - self.order_u + 1)).astype(int)
        v_i_starts = np.floor(v_vec*(self.shape_v - self.order_v + 1)).astype(int)
        w_i_starts = np.floor(w_vec*(self.shape_w - self.order_w + 1)).astype(int)

        get_basis_volume_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            u_i_starts,
            self.order_v, self.shape_v, dv, v_vec, self.knots_v,
            v_i_starts,
            self.order_w, self.shape_w, dw, w_vec, self.knots_w,
            w_i_starts,
            len(u_vec), data, row_indices, col_indices
            )
            
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis