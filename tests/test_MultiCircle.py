import matplotlib.pyplot as plt
import seaborn as sns
from lsdo_genie.utils.geometric_shapes import Multi_circle

centers = [[-13.,-0.5],[-7.,0.5],[2.,0.],[10.,-4.]]
radii = [2.,2.,4.,3.]

m = Multi_circle(centers,radii)

surf_pts = m.surface_points(8)
normals = m.unit_normals(8)

# sns.set()
# plt.plot(surf_pts[:,0],surf_pts[:,1],'b.',markersize=20,label='points')
# for i,(i_pt,i_norm) in enumerate(zip(surf_pts,normals)):
#     if i == 0:
#         plt.arrow(i_pt[0],i_pt[1],i_norm[0],i_norm[1],color='k',label='normals')
#     else:
#         plt.arrow(i_pt[0],i_pt[1],i_norm[0],i_norm[1],color='k')

# exact = m.surface_points(1000)
# plt.plot(exact[:,0],exact[:,1],'k.',markersize=1,label='exact')

# plt.legend(loc='upper right')
# plt.title('Mutli-Circles')
# plt.axis('equal')
# plt.show()