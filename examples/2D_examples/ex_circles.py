from lsdo_genie import Genie2D
from lsdo_genie.utils.geometric_shapes import Multi_circle
import numpy as np

num_surface_pts = 25

centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
radii = [2.,2.,4.,3.]
geom_shape = Multi_circle(centers,radii)

surface_points = geom_shape.surface_points(num_surface_pts)
surface_normals = geom_shape.unit_normals(num_surface_pts)

custom_domain = np.array([
    [-18.,18.],
    [-8.,8.]
])

genie = Genie2D(verbose=True)
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    domain=custom_domain,
    max_control_points=30,
    min_ratio=0.75,
)
genie.solve_energy_minimization(
    Ln=1e0,
    Lr=1e-4,
)
genie.visualize()