# import matplotlib.pyplot as plt
import numpy as np
from lsdo_genie.bsplines import BsplineCurve

def std_uniform_knot_vec(num_cps,order):
    knot_vector = np.zeros(num_cps + order)
    for i in range(num_cps + order):
        knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
    return knot_vector

np.random.seed(1)

order = 4
num_cps = [25,]

kv_u = std_uniform_knot_vec(num_cps[0],order)

x = np.linspace(0,4*np.pi,np.product(num_cps))
domain_size_x = x.max()-x.min()
phi = np.cos(x) 

cps = np.zeros((np.product(num_cps), 2))
cps[:, 0] = np.linspace(0,1,num_cps[0])
cps[:, 1] = phi
    
Surface = BsplineCurve('name',order,kv_u,num_cps)

num_pts = 1000
u_vec = np.linspace(0,1,num_pts)
basis = Surface.get_basis_matrix(u_vec,0)
p = basis.dot(cps[:,1])
basis = Surface.get_basis_matrix(u_vec,1)
dx = basis.dot(cps[:,1])/domain_size_x
basis = Surface.get_basis_matrix(u_vec,2)
dxx = basis.dot(cps[:,1])/domain_size_x/domain_size_x

# plt.plot(cps[:,0],cps[:,1],'r.',label='control points')
# plt.plot(u_vec,p,label='y')
# plt.plot(u_vec,dx,label='dydx')
# plt.plot(u_vec,dxx,label='d2ydx2')
# plt.legend()

np.random.seed(1)

order = 4
num_cps = [99,]

kv_u = std_uniform_knot_vec(num_cps[0],order)

x = np.linspace(0,4*np.pi,np.product(num_cps))
phi = np.cos(x) 

cps = np.zeros((np.product(num_cps), 2))
cps[:, 0] = np.linspace(0,1,num_cps[0])
cps[:, 1] = phi
    
Surface = BsplineCurve('name',order,kv_u,num_cps)

# print(kv_u)

# u_vec = []
# for i in range(int(num_cps[0]/(order-1))):
#     i = int(i)
#     u_vec.append(kv_u[(i+1)*(order-1)])
# u_vec = np.array(u_vec)
# print(u_vec)
# k = kv_u[(order-1):(len(kv_u)-(order-1))]
# print(k)
# print(k[::(order-1)])
u_vec = kv_u[(order-1):(num_cps[0] + 1)]
u_vec = u_vec[::(order-1)]
# print(u_vec)
basis = Surface.get_basis_matrix(u_vec,2)
x = basis.toarray().transpose()
# print(x)
cond = True
for ii,i in enumerate(x):
    ind = np.argwhere(i!=0)
    if len(ind)!=1:
        # print(ii)
        cond = False
# print(cond)
# plt.show()