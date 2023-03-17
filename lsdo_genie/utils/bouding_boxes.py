import numpy as np

def minimum_bbox(surface_points):
    lower = np.min(surface_points,axis=0)
    upper = np.max(surface_points,axis=0)
    dimensions = np.stack((lower, upper),axis=1)
    return dimensions

def enlarged_bbox(surface_points, percent=10):
    lower = np.min(surface_points,axis=0)
    upper = np.max(surface_points,axis=0)
    border = percent/100*(upper-lower)/2
    dimensions = np.stack((lower-border, upper+border),axis=1)
    return dimensions