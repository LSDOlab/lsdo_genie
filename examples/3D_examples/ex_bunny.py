'''
3D Stanford Bunny Problem
'''
from lsdo_genie import Genie3D
from lsdo_genie.utils import extract_stl_data, enlarged_bbox
from lsdo_genie.utils.geometric_shapes import geometry_path

surface_points, surface_normals = extract_stl_data(geometry_path+"Bunny.stl")

custom_domain = enlarged_bbox(surface_points,percent=15.)

genie = Genie3D(verbose=True)
genie.input_point_cloud(
    surface_points=surface_points,
    surface_normals=surface_normals,
)
genie.config(
    domain=custom_domain,
    max_control_points=28,
    min_ratio=0.75,
)
genie.solve_energy_minimization(
    Ln=1e0,
    Lr=1e-6,
)
genie.visualize()