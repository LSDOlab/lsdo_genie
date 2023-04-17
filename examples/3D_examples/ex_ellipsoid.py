'''
3D Ellipsoid Problem
'''
from lsdo_genie import Genie3D
from lsdo_genie.utils.geometric_shapes import Ellipsoid
import numpy as np
from lsdo_genie.utils import enlarged_bbox

num_pts = 100

e = Ellipsoid(5,5,5)

surface_points = e.surface_points(num_pts)
surface_normals = e.unit_normals(num_pts)

custom_domain = enlarged_bbox(surface_points,percent=15.)

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