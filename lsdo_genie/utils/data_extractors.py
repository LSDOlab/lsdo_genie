import numpy as np
from stl.mesh import Mesh

def extract_stl_data(filename:str, verbose:bool=False):
    '''
    Extract the point cloud data from a .stl file

    Parameters
    ----------
    filename : str
        The location of your file relative to your working directory when running
    verbose : bool
        Prints the number of points loaded to the terminal

    Returns
    ----------
    centroids : np.ndarray(N,3)
        The centroids of the triangulation
    normals : np.ndarray(N,3)
        The normal vectors of the associated centroids
    '''
    mesh_import = Mesh.from_file(filename)
    all_points = mesh_import.points

    uniq_pts = all_points.reshape(3*len(mesh_import.points),3)
    vertices,_ = np.unique(uniq_pts,axis=0,return_index=True)
    vertices = np.float64(vertices)

    if verbose:
        print('Number of points loaded: ',len(all_points))

    faces = all_points.reshape(len(all_points),3,3)
    centroids = np.mean(faces,axis=1)
    normals = mesh_import.get_unit_normals()

    return centroids, normals