import numpy as np

def minimum_bbox(surface_points:np.ndarray):
    '''
    The minimum bounding box of a given point cloud 

    Parameters
    ----------
    surface_points : np.ndarray(Ngamma,d)
        The positions of points in a point cloud in 'd' dimensions

    Returns
    ----------
    domain : np.ndarray(d,2)
        The lower and upper bounds along each dimension
    '''
    lower = np.min(surface_points,axis=0)
    upper = np.max(surface_points,axis=0)
    domain = np.stack((lower, upper),axis=1)
    return domain

def enlarged_bbox(surface_points:np.ndarray, percent:float=10.):
    '''
    Calcuate an enlarged bounding box of a given point cloud

    Parameters
    ----------
    surface_points : np.ndarray(Ngamma,d)
        The positions of points in a point cloud in 'd' dimensions
    percent : float
        The percentage to increase the bounding box from the minimum bounding box

    Returns
    ----------
    domain : np.ndarray(d,2)
        The lower and upper bounds along each dimension
    '''
    lower = np.min(surface_points,axis=0)
    upper = np.max(surface_points,axis=0)
    border = percent/100*(upper-lower)/2
    domain = np.stack((lower-border, upper+border),axis=1)
    return domain