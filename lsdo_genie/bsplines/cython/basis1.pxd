from libc.stdlib cimport malloc, free


cdef int get_basis1(int order, int num_control_points, double u, int i_start, double* knot_vector, double* basis1)