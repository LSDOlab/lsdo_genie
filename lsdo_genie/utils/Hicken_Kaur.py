from scipy.spatial import KDTree
import numpy as np

'''
Equations based on the explicit level set function derived by Hicken and Kaur which interpolate
data points via piecewise linear signed distance functions defined by local hyperplanes.
The function is defined by these piecewise functions with KS-aggregation, but is only continuous and
nondifferentiable if only using the k-nearest neighbors.

"An Explicit Level-Set Formula to Approximate Geometries"
Jason E. Hicken and Sharanjeet Kaur
doi:10.2514/6.2022-1862
'''

def explicit_lsf(pts,dataset:KDTree,norm_vec,k,rho):
    distances,indices = dataset.query(pts,k=k)
    if k==1:
        phi = (dataset.data[indices] - pts)*norm_vec[indices]
        return phi
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = dataset.data[indices,:] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ij->i',Dx*norm_vec[indices],exp)/np.sum(exp,axis=1)
    return phi
