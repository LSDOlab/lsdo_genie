from lsdo_genie import Genie3D
from lsdo_genie.utils import Sphere
from lsdo_genie.utils import Genie3DCSDLModel
import numpy as np
import csdl
from python_csdl_backend import Simulator

num_pts = 100
e = Sphere(5)
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
genie.input_point_cloud(surface_points=surface_points,surface_normals=surface_normals)
genie.config(domain=custom_domain,max_control_points=40,min_ratio=0.75)
genie.solve_energy_minimization(Ln=1e0,Lr=1e-1)

num_samples = 10
x_name = "x_positions"
y_name = "y_positions"
z_name = "z_positions"
out_name = "phi"
xs = np.linspace(x.min(),x.max(),num_samples)
ys = np.linspace(y.min(),y.max(),num_samples)
zs = np.linspace(z.min(),z.max(),num_samples)

model = csdl.Model()
csdl_x = model.create_input(x_name,val=xs)
csdl_y = model.create_input(y_name,val=ys)
csdl_z = model.create_input(z_name,val=zs)
phi = csdl.custom(csdl_x, csdl_y, csdl_z, op=Genie3DCSDLModel(
    num_pts=num_samples,
    x_name=x_name,
    y_name=y_name,
    z_name=z_name,
    out_name=out_name,
    genie_object=genie,
))
model.register_output(out_name,phi)
sim = Simulator(model)
sim.run()
sim.check_partials(compact_print=True)
for key in [x_name,y_name,z_name,out_name]:
    print(key,": ",sim[key])