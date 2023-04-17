import numpy as np

def standard_uniform_knot_vector(num_cps:int,order:int):
    '''
    Generates a standard uniform knot vector

    Parameters
    ----------
    num_cps : int
        Number of control points
    order : int
        B-spline polynomial order

    Returns
    ----------
    knot_vector : np.ndarray(num_cps+order,)
        The standard uniform knot vector for unit parametric coordinates
    '''
    knot_vector = (np.arange(num_cps+order) - order + 1) / (num_cps - order + 1)
    return knot_vector

def open_uniform_knot_vector(num_cps:int,order:int):
    '''
    Generates an open uniform knot vector

    Parameters
    ----------
    num_cps : int
        Number of control points
    order : int
        B-spline polynomial order

    Returns
    ----------
    knot_vector : np.ndarray(num_cps+order,)
        The open uniform knot vector for unit parametric coordinates
    '''

    knot_vector = np.zeros(num_cps + order)
    knot_vector[order:num_cps] = (np.arange(order, num_cps) - order + 1) / (num_cps - order + 1)
    knot_vector[num_cps:] = 1.
    return knot_vector