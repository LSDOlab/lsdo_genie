import numpy as np

def standard_uniform_knot_vector(num_cps,order):
    '''
    Generates a standard uniform knot vector
    '''

    knot_vector = (np.arange(num_cps+order) - order + 1) / (num_cps - order + 1)
    return knot_vector

def open_uniform_knot_vector(num_cps,order):
    '''
    Generates an open uniform knot vector
    '''

    knot_vector = np.zeros(num_cps + order)
    knot_vector[order:num_cps] = (np.arange(order, num_cps) - order + 1) / (num_cps - order + 1)
    knot_vector[num_cps:] = 1.
    return knot_vector

if __name__ == '__main__':
    num_cps = 5000
    order = 3
    standard_uniform_knot_vector(num_cps,order)
    open_uniform_knot_vector(num_cps,order)