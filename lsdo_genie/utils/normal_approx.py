import numpy as np
'''
A quick way to approximate the normal vectors from an arbitrary closed polygon.
Normals are an average between the 2 edge normals adjacent to a point.

inputs: np.ndarray shape=(num_pts,2)

Clockwise: normals point outwards
Counter-Clockwise: normals point inwards
'''

def quick_normal_approx(surf_pts):
    edges_i = surf_pts - np.roll(surf_pts, shift=1, axis=0)
    normals_i = np.stack((-edges_i[:,1],edges_i[:,0]),axis=1) / np.linalg.norm(edges_i,axis=1)[:,None]
    edges_j = np.roll(surf_pts, shift=-1, axis=0) - surf_pts
    normals_j = np.stack((-edges_j[:,1],edges_j[:,0]),axis=1) / np.linalg.norm(edges_j,axis=1)[:,None]
    normals = (normals_i + normals_j)/2
    normals = normals/np.linalg.norm(normals,axis=1)[:,None]
    return normals