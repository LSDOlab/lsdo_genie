from lsdo_genie import Genie3D
from lsdo_genie.utils import Ellipsoid
import numpy as np

num_pts = 100

e = Ellipsoid(5,5,5)

surface_points = e.surface_points(num_pts)
surface_normals = e.unit_normals(num_pts)

x,y,z = surface_points[:,0],surface_points[:,1],surface_points[:,2]
border = 5
custom_domain = np.array([
    [x.min()-border, x.max()+border],
    [y.min()-border, y.max()+border],
    [z.min()-border, z.max()+border],
])

genie = Genie3D(verbose=True)
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    domain=custom_domain,
    max_control_points=40,
    min_ratio=0.75,
)
genie.solve_energy_minimization(
    Ln=1e0,
    Lr=1e-1,
)
genie.visualize()