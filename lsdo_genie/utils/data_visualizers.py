import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_2Dptcloud(points:np.ndarray, normals:np.ndarray, show_normals:bool=True, title:str="2D Point cloud with normals", perc_normal_length:float=5.):
    '''
    Plots and shows a 2D point cloud with normal vectors

    Parameters
    ----------
    points : np.ndarray(N,2)
        The points in the poind cloud
    normals : np.ndarray(N,2)
        The normal vectors of the point cloud
    show_normals : bool
        Show the normal vectors with a blue arrow
    title : str
        Title of the plot
    '''
    lower = np.min(points,axis=0)
    upper = np.max(points,axis=0)
    domain = np.stack((lower, upper),axis=1)
    bbox_diag = np.linalg.norm(np.diff(domain,axis=1).flatten())
    normal_length = perc_normal_length/100*bbox_diag
    sns.set_style('ticks')
    plt.figure()
    plt.plot(points[:,0],points[:,1],'k.', label="points")
    if show_normals:
        show_label = True
        for (x,y),(nx,ny) in zip(points, normals):
            if show_label:
                plt.arrow(x, y, nx*normal_length, ny*normal_length, color='b', head_width=.2, label="normals")
                show_label = False
            else: 
                plt.arrow(x, y, nx*normal_length, ny*normal_length, color='b', head_width=.2)
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    sns.despine()
    plt.show()

def visualize_3Dptcloud(points:np.ndarray, normals:np.ndarray, show_normals:bool=True, title:str="3D Point cloud with normals", perc_normal_length:float=5.):
    '''
    Plots and shows a 3D point cloud with normal vectors

    Parameters
    ----------
    points : np.ndarray(N,3)
        The points in the poind cloud
    normals : np.ndarray(N,3)
        The normal vectors of the point cloud
    show_normals : bool
        Show the normal vectors with a blue arrow
    title : str
        Title of the plot
    '''
    lower = np.min(points,axis=0)
    upper = np.max(points,axis=0)
    domain = np.stack((lower, upper),axis=1)
    bbox_diag = np.linalg.norm(np.diff(domain,axis=1).flatten())
    normal_length = perc_normal_length/100*bbox_diag
    sns.set_style('ticks')
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(points[:,0],points[:,1],points[:,2],'k.', label="points")
    if show_normals:
        show_label = True
        for (x,y,z),(nx,ny,nz) in zip(points, normals):
            if show_label:
                xpts = np.array([x,x+nx*normal_length])
                ypts = np.array([y,y+ny*normal_length])
                zpts = np.array([z,z+nz*normal_length])
                ax.plot(xpts,ypts,zpts, 'b.-', label="normals")
                show_label = False
            else: 
                xpts = np.array([x,x+nx*normal_length])
                ypts = np.array([y,y+ny*normal_length])
                zpts = np.array([z,z+nz*normal_length])
                ax.plot(xpts,ypts,zpts, 'b-')
    ax.set_title(title)
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