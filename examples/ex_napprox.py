from lsdo_genie.utils import quick_normal_approx
import numpy as np
import matplotlib.pyplot as plt

surface_points = np.array([
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

surface_normals = quick_normal_approx(surface_points)

plt.plot(surface_points[:,0],surface_points[:,1],'k.-')
closure = np.roll(surface_points,shift=1,axis=0)
plt.plot(closure[:,0],closure[:,1],'k.-')
for (x,y),(nx,ny) in zip(surface_points, surface_normals):
    plt.arrow(x, y, nx, ny, color='k', head_width=.2, label='AB')
plt.show()