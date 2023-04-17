import numpy as np
from numpy import newaxis as na

'''
"Gradient-based Wind Farm Layout Optimization With Inclusion And Exclusion Zones"
Criado Risco, J., Valotta Rodrigues, R., Friis-Møller, M., Quick, J., Mølgaard Pedersen, M., & Réthoré, P. E. 
Wind Energy Science Discussions, 1-24. (2023).

For source code, see: 
https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2
topfarm/constraint_components/boundary.py 
'''

def get_boundary_properties(xy_boundary, inclusion_zone=True):
    '''
    :no-index:
    '''
    vertices = np.array(xy_boundary)
    def get_edges(vertices, counter_clockwise):
        if np.any(vertices[0] != vertices[-1]):
            vertices = np.r_[vertices, vertices[:1]]
        x1, y1 = A = vertices[:-1].T
        x2, y2 = B = vertices[1:].T
        double_area = np.sum((x1 - x2) * (y1 + y2))  # 2 x Area (+: counterclockwise
        assert double_area != 0, "Area must be non-zero"
        if (counter_clockwise and double_area < 0) or (not counter_clockwise and double_area > 0):  #
            return get_edges(vertices[::-1], counter_clockwise)
        else:
            return vertices[:-1], A, B
    # inclusion zones are defined counter clockwise (unit-normal vector pointing in) while
    # exclusion zones are defined clockwise (unit-normal vector pointing out)
    xy_boundary, A, B = get_edges(vertices, inclusion_zone)
    dx, dy = AB = B - A
    AB_len = np.linalg.norm(AB, axis=0)
    edge_unit_normal = (np.array([-dy, dx]) / AB_len)
    # A_normal and B_normal are the normal vectors at the nodes A,B (the mean of the adjacent edge normal vectors
    A_normal = (edge_unit_normal + np.roll(edge_unit_normal, 1, 1)) / 2
    B_normal = np.roll(A_normal, -1, 1)

    return (xy_boundary, A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal)

# File: topfarm/constraint_components/boundary.py 
# Function: PolygonBoundaryComp._calc_distance_and_gradients()
# Line: 397
def _calc_distance_and_gradients(x, y, boundary_properties):
    '''
    :no-index:
    '''
    def vec_len(vec):
        return np.linalg.norm(vec, axis=0)
    A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal = boundary_properties
    P = np.array([x, y])[:, :, na]
    A, B, AB = A[:, na], B[:, na], AB[:, na]
    edge_unit_normal, A_normal, B_normal = edge_unit_normal[:, na], A_normal[:, na], B_normal[:, na]
    AB_len = AB_len[na]
    AP = P - A
    BP = P - B
    a_tilde = np.sum(AP * AB, axis=0) / AB_len
    use_A = 0 > a_tilde
    use_B = a_tilde > AB_len
    distance = np.sum((AP) * edge_unit_normal, 0)
    good_side_of_A = (np.sum((AP * A_normal)[:, use_A], 0) > 0)
    sign_use_A = np.where(good_side_of_A, 1, -1)
    distance[use_A] = (vec_len(AP[:, use_A]) * sign_use_A)
    good_side_of_B = np.sum((BP * B_normal)[:, use_B], 0) > 0
    sign_use_B = np.where(good_side_of_B, 1, -1)
    distance[use_B] = (vec_len(BP[:, use_B]) * sign_use_B)
    ddist_dxy = np.tile(edge_unit_normal, (1, len(x), 1))
    ddist_dxy[:, use_A] = sign_use_A * (AP[:, use_A] / vec_len(AP[:, use_A]))
    ddist_dxy[:, use_B] = sign_use_B * (BP[:, use_B] / vec_len(BP[:, use_B]))
    ddist_dX, ddist_dY = ddist_dxy
    return distance, ddist_dX, ddist_dY

def calc_distance_and_gradients(x, y, boundary_properties):
    '''
    :no-index:
    '''
    Dist_ij, ddist_dX, ddist_dY = _calc_distance_and_gradients(x, y, boundary_properties)
    dDdk_ijk = np.moveaxis([ddist_dX, ddist_dY], 0, -1)
    sign_i = np.sign(Dist_ij[np.arange(Dist_ij.shape[0]), np.argmin(abs(Dist_ij), axis=1)])
    output = [Dist_ij, dDdk_ijk, sign_i]
    return output

def risco_eval(x, y, boundary_properties):
    '''
    :no-index:
    '''
    Dist_ij, _, _ = calc_distance_and_gradients(x, y, boundary_properties)
    Dist_i = Dist_ij[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1)]
    return Dist_i

def risco_deriv_eval(x, y, boundary_properties):
    '''
    :no-index:
    '''
    Dist_ij, dDdk_ijk, _ = calc_distance_and_gradients(x, y, boundary_properties)
    dSdkx_i, dSdky_i = dDdk_ijk[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1), :].T
    gradients = np.diagflat(dSdkx_i), np.diagflat(dSdky_i)
    return gradients