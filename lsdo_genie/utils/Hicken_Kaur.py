from scipy.spatial import KDTree
import numpy as np

def explicit_lsf(pts,data_points:KDTree,data_normals,k,rho):
    '''
    Explicit level set function derived by Hicken and Kaur which interpolates
    data points via piecewise linear signed distance functions defined by local hyperplanes.
    The function is defined by these piecewise functions with KS-aggregation, but is
    non-differentiable if only using the k-nearest neighbors.

    "An Explicit Level-Set Formula to Approximate Geometries"
    Jason E. Hicken and Sharanjeet Kaur
    doi:10.2514/6.2022-1862

    Parameters
    ----------
    points : np.ndarray(N,d)
        Points to evaluate the level set funcion
    data_points : scipy.spatial.KDTree
        KDTree structure of the point cloud
    data_normals : np.ndarray(Ngamma,d)
        Normal vectors of the point cloud
    k : int
        Numer of nearest neighbors to include in the function
    rho : float
        Smoothing parameter

    Returns
    ----------
    phi : np.ndarray(N,)
        Approximate signed distance value of the points
    '''
    distances,indices = data_points.query(pts,k=k)
    if k==1:
        phi = (data_points.data[indices] - pts)*data_normals[indices]
        return phi
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = data_points.data[indices,:] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ij->i',Dx*data_normals[indices],exp)/np.sum(exp,axis=1)
    return phi
