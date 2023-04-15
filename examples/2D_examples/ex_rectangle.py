from lsdo_genie import Genie2D
from lsdo_genie.utils import Rectangle
import numpy as np

num_points = 76
geom_shape = Rectangle(5,7)
surface_points = geom_shape.surface_points(num_points)
surface_normals = geom_shape.unit_normals(num_points)

custom_domain = np.array([
    [-4.0, 4.0],
    [-5.6, 5.6],
])

genie = Genie2D(verbose=True)
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    domain=custom_domain,
    max_control_points=70,
    min_ratio=0.75,
)
genie.solve_energy_minimization(
    Ln=1e0,
    Lr=1e-4,
)
genie.visualize()