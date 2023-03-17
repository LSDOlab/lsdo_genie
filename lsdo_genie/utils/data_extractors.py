import numpy as np
from stl.mesh import Mesh

def extract_stl_data(filename):
    mesh_import = Mesh.from_file(filename)
    all_points = mesh_import.points

    uniq_pts = all_points.reshape(3*len(mesh_import.points),3)
    vertices,_ = np.unique(uniq_pts,axis=0,return_index=True)
    vertices = np.float64(vertices)

    print('Number of triangles: ',len(all_points))

    faces = all_points.reshape(len(all_points),3,3)
    face_centroids = np.mean(faces,axis=1)
    normals = mesh_import.get_unit_normals()

    return face_centroids, normals