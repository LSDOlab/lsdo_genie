from libc.stdlib cimport malloc, free

from lsdo_genie.bsplines.cython.basis0 cimport get_basis0
from lsdo_genie.bsplines.cython.basis1 cimport get_basis1
from lsdo_genie.bsplines.cython.basis2 cimport get_basis2


cdef get_basis_surface_matrix(
    int order_u, int num_control_points_u, int u_der, double* u_vec, double* knot_vector_u,
    int* u_i_start, 
    int order_v, int num_control_points_v, int v_der, double* v_vec, double* knot_vector_v,
    int* v_i_start, 
    int num_points, double* data, int* row_indices, int* col_indices,
)