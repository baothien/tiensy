# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:15:36 2014

@author: bao
The number of streamline going through the ROIs
Input: tract as an array of streamlines 
       ROIs: array of roi (M,3) Each ROI is defined by the center coordinate
       Rs: array of fload (M) - radius of each ROI
"""
import numpy as np

def my_inside_sphere(xyz,center,radius):
	tmp = xyz-center
	return (np.sum((tmp * tmp), axis=1)<=(radius * radius)).any()==True
 

def compute_intersecting(tracks, voxel, R, coords, si):
        x_idx = np.where((coords[0] >= voxel[0] - R) & (coords[0] <= (voxel[0] + R)))[0]
        y_idx = x_idx[np.where((coords[1][x_idx] >=  voxel[1] - R) & (coords[1][x_idx] <= (voxel[1] + R)))[0]]
        z_idx = y_idx[np.where((coords[2][y_idx] >=  voxel[2] - R) & (coords[2][y_idx] <= (voxel[2] + R)))[0]]
    
        s_idx = np.unique(si[z_idx]).astype(np.int)
        #print len(s_idx)
        #temp = np.array([my_inside_sphere(st, voxel, R) for st in tracks[s_idx]])
        temp = np.array([my_inside_sphere(si, voxel, R) for si in tracks[s_idx]])
        #print 'temp : ', temp
        
        #return s_idx[np.array([my_inside_sphere(st, voxel, R) for st in tracks[s_idx]])]
        if len(temp)!=0:        
            return s_idx[temp]
        return []

def intersec_ROIs(tracks, ROIs, Rs, vis = False):
    
    tracks = np.array([s for s in tracks], dtype=np.object)
        
    si = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(tracks)]).astype(np.int)
    
    coords = np.vstack(tracks).T  
            
    intersecting = []
    for i in np.arange(len(ROIs)):
        voxel = ROIs[i]
        R = Rs[i]       
        intersecting.append(compute_intersecting(tracks, voxel, R, coords, si))
        if vis == True:
            print '\t\t voxel:', voxel, '\t R:', R, '\t streamlines cross: ', len(intersecting[-1])
        

    if len(intersecting[0]) == 0:
        return []
    
    common = set(intersecting[0].tolist())
    if vis == True:
        print '\t\t 0  common streamlines:', len(common)
    #common = set(intersecting[0].tolist()).intersection(set(intersecting[1].tolist()))
    
    k = 1
    while (k <len(ROIs)):        
        if len(intersecting[k])==0:            
            return []
        common = set(common).intersection(set(intersecting[k].tolist()))        
        if vis == True:
            print '\t\t',k, " common streamlines:", len(common)
        k = k + 1
    
    return common
 
def compute_intersecting_dpy(tracks, voxel, R):
    
    from dipy.tracking.metrics import intersect_sphere
   
    s_idx = []
    for idx in np.arange(len(tracks)):
        if intersect_sphere(tracks[idx],voxel,R)==True:
            s_idx.append(idx)          
    return np.array(s_idx,dtype=int)

def intersec_ROIs_dpy(tracks, ROIs, Rs, vis = False):
    
    tracks = np.array([s for s in tracks], dtype=np.object)       
            
    intersecting = []
    for i in np.arange(len(ROIs)):
        voxel = ROIs[i]
        R = Rs[i]       
        intersecting.append(compute_intersecting_dpy(tracks, voxel, R))
        if vis == True:
            print '\t\t voxel:', voxel, '\t R:', R, '\t streamlines cross: ', len(intersecting[-1])
        

    if len(intersecting[0]) == 0:
        return []
    
    common = set(intersecting[0].tolist())
    if vis == True:
        print '\t\t 0  common streamlines:', len(common)
    #common = set(intersecting[0].tolist()).intersection(set(intersecting[1].tolist()))
    
    k = 1
    while (k <len(ROIs)):        
        if len(intersecting[k])==0:            
            return []
        common = set(common).intersection(set(intersecting[k].tolist()))        
        if vis == True:
            print '\t\t',k, " common streamlines:", len(common)
        k = k + 1
    
    return common    
