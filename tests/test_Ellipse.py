import matplotlib.pyplot as plt
import seaborn as sns
from lsdo_genie.utils.geometric_shapes import Ellipse

sns.set()

a = 18
b = 5
num_pts = 25

e = Ellipse(a,b)

pts = e.surface_points(num_pts)
normals = e.unit_normals(num_pts)
# plt.plot(pts[:,0],pts[:,1],'k-',label='points')
# for i in range(num_pts):
#     if i == 0:
#         plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='b',label='normals')
#     else:
#         plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='b')

# exact = e.surface_points(1000)
# plt.plot(exact[:,0],exact[:,1],'k-',label='exact')

# plt.legend(loc='upper right')
# plt.title('Ellipse a={} b={}'.format(a,b))
# plt.axis('equal')

# plt.show()