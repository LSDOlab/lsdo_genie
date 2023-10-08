import numpy as np
import scipy.sparse as sps

from lsdo_genie.bsplines.cython.basis_matrix_curve_py import get_basis_curve_matrix

class BsplineCurve:
    '''
    Base class for B-spline Curves

    Attributes
    ----------
    name : str
        A nickname for the B-spline Curve
    order_u : int
        B-spline polynomial order in the 'u' direction
    knots_u : int
        Knot vector in the 'u' direction
    shape : tuple
        Shape for the B-spline control points (num_u,)
    '''

    def __init__(self, name, order_u, knots_u, shape):
        self.name = name
        self.order_u = order_u
        self.knots_u = knots_u
        self.shape_u = int(shape[0])
        self.num_control_points = int(np.product(shape))

    def get_basis_matrix(self, u_vec, du):
        '''
        Builds the basis matrix for a given set of points

        Parameters
        ----------
        u_vec : np.ndarray(N,)
            Vector storing the 'u' cooridinates of a set of input points
        du : int
            Derivative in the 'u' direction
        
        Returns
        ----------
        basis : sps.csc_matrix(N,Ncp)
            The basis matrix that can be multiplied with the control points to get (N,) output values
        '''
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        u_i_starts = np.floor(u_vec*(self.shape_u - self.order_u + 1)).astype(int)

        get_basis_curve_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            u_i_starts,
            len(u_vec), data, row_indices, col_indices
            )
            
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis