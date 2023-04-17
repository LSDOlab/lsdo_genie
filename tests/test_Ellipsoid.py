import matplotlib.pyplot as plt
import seaborn as sns
from lsdo_genie.utils.geometric_shapes import Ellipsoid

a = 16
b = 6
c = 8
num_pts = 100

e = Ellipsoid(a,b,c)

pts = e.surface_points(num_pts)
normals = e.unit_normals(num_pts)

num_pts = len(pts)

# sns.set()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i,(i_pt,i_norm) in enumerate(zip(pts,normals)):
#     i_norm *= 1
#     if i == 0:
#         pt1 = i_pt + i_norm
#         pt = np.vstack((i_pt,pt1))
#         ax.plot(pt[:,0],pt[:,1],pt[:,2],'k.-',label='Normals')
#     else:
#         pt1 = i_pt + i_norm
#         pt = np.vstack((i_pt,pt1))
#         ax.plot(pt[:,0],pt[:,1],pt[:,2],'k.-')
# ax.plot(pts[:,0],pts[:,1],pts[:,2],'b.',label='Surface Points')
# ax.set_title('$N_{\Gamma}$ = %i' %num_pts)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(-a,a)
# ax.set_ylim(-a,a)
# ax.set_zlim(-a,a)
# ax.legend()
# plt.show()