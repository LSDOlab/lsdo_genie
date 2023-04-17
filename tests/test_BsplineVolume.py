import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from lsdo_genie.bsplines import BsplineVolume

def std_uniform_knot_vec(num_cps,order):
    knot_vector = np.zeros(num_cps + order)
    for i in range(num_cps + order):
        knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
    return knot_vector

### plot timing performance
res = 160
order = 4
num_cps = [42,42,42]

kv_u = std_uniform_knot_vec(num_cps[0],order)
kv_v = std_uniform_knot_vec(num_cps[1],order)
kv_w = std_uniform_knot_vec(num_cps[2],order)

cps = np.zeros((np.product(num_cps), 4))
cps[:, 0] = np.einsum('i,j,k->ijk', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1]), np.ones(num_cps[2])).flatten()
cps[:, 1] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1]), np.ones(num_cps[2])).flatten()
cps[:, 2] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.ones(num_cps[1]), np.linspace(0,1,num_cps[2])).flatten()
cps[:, 3] = np.random.rand(np.product(num_cps))

# plt.figure(figsize=(6,5),dpi=140)
# plt.xlabel('Number of Evaluations')
# plt.ylabel('Evaluation Time (sec)')
num_pts = np.logspace(np.log10(6),np.log10(3500),res,dtype=int)
for order in range(4,7):
    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    kv_w = std_uniform_knot_vec(num_cps[2],order)
    BSpline = BsplineVolume('name',order,order,order,kv_u,kv_v,kv_w,num_cps)
    
    time_set = np.zeros(res)
    for i,i_num_pts in enumerate(num_pts):
        u_vec = np.linspace(0,1,i_num_pts)
        v_vec = np.linspace(0,1,i_num_pts)
        w_vec = np.linspace(0,1,i_num_pts)

        t1 = time.perf_counter()
        basis = BSpline.get_basis_matrix(u_vec,v_vec,w_vec,0,0,0)
        p = basis.dot(cps[:,2])
        t2 = time.perf_counter()
        
        time_set[i] = t2-t1
    # plt.loglog(num_pts,time_set,label='Order '+str(order))
# plt.legend()

# plt.tight_layout()
# plt.show()
