import matplotlib.pyplot as plt
import seaborn as sns
from lsdo_genie.utils.geometric_shapes import Rectangle

sns.set()

w = 5
h = 7
num_pts = 78

r = Rectangle(w,h,rotation=45)

pts = r.surface_points(num_pts)
normals = r.unit_normals(num_pts)
# plt.plot(pts[:,0],pts[:,1],'k.',label='points')
# for i in range(num_pts):
#     if i == 0:
#         plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k',label='normals')
#     else:
#         plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k')

# exact = r.surface_points(1000)
# plt.plot(exact[:,0],exact[:,1],'b-',label='exact')

# plt.legend(loc='upper right')
# plt.title('Rectangle w={} h={}'.format(w,h))
# plt.axis('equal')

# plt.show()