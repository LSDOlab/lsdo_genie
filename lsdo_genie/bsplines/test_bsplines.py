import numpy as np
import lsdo_test as lt

##test
def standard_uniform_check():
    from lsdo_genie.bsplines.knot_vectors import standard_uniform_knot_vector
    num_cps = 5
    order = 4
    vec = standard_uniform_knot_vector(num_cps,order)

    expected_vector = np.array([
        -1.5, -1., -0.5, 0., 0.5, 1.0, 1.5, 2.0, 2.5,
    ])

    assert lt.elementwise_equal(vec, expected_vector)
