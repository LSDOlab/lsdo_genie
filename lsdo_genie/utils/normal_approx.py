import numpy as np

def vertex_normal_approx(surf_pts):
    '''
    A quick way to approximate the normal vectors from an arbitrary closed polygon.
    Normals are an average between the 2 edge normals adjacent to a point and follow a covention:
    Clockwise : normals point outwards
    Counter-Clockwise : normals point inwards

    inputs: 
    surf_pts : np.ndarray shape=(num_pts,2)

    output:
    normals : np.ndarray shape=(num_pts,2)
    '''
    edges_i = surf_pts - np.roll(surf_pts, shift=1, axis=0)
    normals_i = np.stack((-edges_i[:,1],edges_i[:,0]),axis=1) / np.linalg.norm(edges_i,axis=1)[:,None]
    edges_j = np.roll(surf_pts, shift=-1, axis=0) - surf_pts
    normals_j = np.stack((-edges_j[:,1],edges_j[:,0]),axis=1) / np.linalg.norm(edges_j,axis=1)[:,None]
    normals = (normals_i + normals_j)/2
    normals = normals/np.linalg.norm(normals,axis=1)[:,None]
    return normals

def edge_midpoints(vertices):
    '''
    Approximate an arbitrary closed polygon by using the midpoints of the edges formed by an input set of vertices
    Normal direction is determined by the direction of the points given in the input:
    Clockwise : normals point outwards
    Counter-Clockwise : normals point inwards

    inputs: 
    vertices : np.ndarray shape=(num_pts,2)

    output:
    midpoints: np.ndarray shape=(num_pts,2)
    normals : np.ndarray shape=(num_pts,2)
    '''
    pts_i = vertices
    pts_j = np.roll(vertices, shift=1, axis=0)
    edges = pts_i - pts_j
    normals = np.stack((-edges[:,1],edges[:,0]),axis=1) / np.linalg.norm(edges,axis=1)[:,None]

    midpoints = np.mean(np.stack((pts_i,pts_j),axis=2),axis=2)
    print(np.shape(midpoints))
    return midpoints, normals