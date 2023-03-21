from lsdo_genie.utils import vertex_normal_approx, edge_midpoints
import numpy as np
import matplotlib.pyplot as plt

vertex_points = np.array([
    [10.        , 4.13619728],
    [ 8.34605869, 4.21052632],
    [ 6.84210526, 3.70084407],
    [ 6.31578947, 2.58397925],
    [ 4.73684211, 2.92381402],
    [ 4.9240031 , 4.21052632],
    [ 4.21052632, 5.43612266],
    [ 4.73684211, 6.65612795],
    [ 4.72569161, 7.89473684],
    [ 3.98321809, 9.47368421],
    [ 5.26315789, 9.8343276 ],
    [ 6.84210526, 9.2020613 ],
    [ 7.36842105, 7.76377389],
    [ 7.94206498, 6.31578947],
    [ 8.94736842, 4.91640485],
])

vertex_normals = vertex_normal_approx(vertex_points)
midpoints, midpt_normals = edge_midpoints(vertex_points)

### The rest of this file is plotting:

plt.figure()
plt.plot(vertex_points[:,0],vertex_points[:,1],'k.-', label="vertices")
closure = np.roll(vertex_points,shift=1,axis=0)
plt.plot(closure[:,0],closure[:,1],'k.-')
show_label = True
for (x,y),(nx,ny) in zip(vertex_points, vertex_normals):
    if show_label:
        plt.arrow(x, y, nx, ny, color='k', head_width=.2, label="vertex normals")
        show_label = False
    else: 
        plt.arrow(x, y, nx, ny, color='k', head_width=.2)
    
plt.title("Vertex approximation")
plt.legend()
plt.axis('equal')

plt.figure()
plt.plot(midpoints[:,0],midpoints[:,1],'k.-', label="midpoints")
closure = np.roll(midpoints,shift=1,axis=0)
plt.plot(closure[:,0],closure[:,1],'k.-')
show_label = True
for (x,y),(nx,ny) in zip(midpoints, midpt_normals):
    if show_label:
        plt.arrow(x, y, nx, ny, color='k', head_width=.2, label="midpoint normals")
        show_label = False
    else: 
        plt.arrow(x, y, nx, ny, color='k', head_width=.2)
plt.title("Midpoints")
plt.legend()
plt.axis('equal')

plt.figure()
plt.plot(midpoints[:,0],midpoints[:,1],'r.-', label="midpoints")
closure = np.roll(midpoints,shift=1,axis=0)
plt.plot(closure[:,0],closure[:,1],'r.-')
show_label = True
for (x,y),(nx,ny) in zip(midpoints, midpt_normals):
    if show_label:
        plt.arrow(x, y, nx, ny, color='r', head_width=.2, label="midpoint normals")
        show_label = False
    else: 
        plt.arrow(x, y, nx, ny, color='r', head_width=.2)
plt.plot(vertex_points[:,0],vertex_points[:,1],'k.-', label="vertices")
closure = np.roll(vertex_points,shift=1,axis=0)
plt.plot(closure[:,0],closure[:,1],'k.-')
show_label = True
for (x,y),(nx,ny) in zip(vertex_points, vertex_normals):
    if show_label:
        plt.arrow(x, y, nx, ny, color='k', head_width=.2, label="vertex normals")
        show_label = False
    else: 
        plt.arrow(x, y, nx, ny, color='k', head_width=.2)
plt.title("Both overlayed")
plt.legend()
plt.axis('equal')

plt.show()