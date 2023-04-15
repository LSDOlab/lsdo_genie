import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_2Dptcloud(points, normals, show_normals=True):
    sns.set_style('ticks')
    plt.figure()
    plt.plot(points[:,0],points[:,1],'k.', label="points")
    if show_normals:
        show_label = True
        for (x,y),(nx,ny) in zip(points, normals):
            if show_label:
                plt.arrow(x, y, nx, ny, color='b', head_width=.2, label="normals")
                show_label = False
            else: 
                plt.arrow(x, y, nx, ny, color='b', head_width=.2)
    plt.title("Point cloud with normals")
    plt.legend()
    plt.axis('equal')
    sns.despine()
    plt.show()

def visualize_3Dptcloud(points, normals, show_normals=True):
    sns.set_style('ticks')
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(points[:,0],points[:,1],points[:,2],'k.', label="points")
    if show_normals:
        show_label = True
        for (x,y,z),(nx,ny,nz) in zip(points, normals):
            if show_label:
                xpts = np.array([x,x+nx])
                ypts = np.array([y,y+ny])
                zpts = np.array([z,z+nz])
                ax.plot(xpts,ypts,zpts, 'b.-', label="normals")
                show_label = False
            else: 
                xpts = np.array([x,x+nx])
                ypts = np.array([y,y+ny])
                zpts = np.array([z,z+nz])
                ax.plot(xpts,ypts,zpts, 'b-')
    ax.set_title("Point cloud with normals")
    ax.legend()
    lower = np.min(points,axis=0)
    upper = np.max(points,axis=0)
    centers = (upper+lower)/2
    width = np.max(abs(upper-lower))

    ax.set_xlim(centers[0]-width/2, centers[0]+width/2)
    ax.set_ylim(centers[1]-width/2, centers[1]+width/2)
    ax.set_zlim(centers[2]-width/2, centers[2]+width/2)
    sns.despine()
    plt.show()