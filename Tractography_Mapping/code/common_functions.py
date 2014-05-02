# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:37:20 2014

@author: bao
Some common functions for working with tractography
"""
import numpy as np
from dipy.io.dpy import Dpy
from dipy.viz import fvtk
from dipy.tracking.metrics import length
from dipy.io.pickles import load_pickle, save_pickle
from dissimilarity_common_20130925 import subset_furthest_first as sff
   
def mklink():
    import os
    ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212, 213]
    for i in np.arange(len(ids)):
        sub = str(ids[i])
        arg1 = 'ln -s '
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M_linear.dpy '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M_linear.dpy'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M_linear.trk '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M_linear.trk'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M.dpy'
        
        arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.trk '
        arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M.trk'
        
        full_cmd = arg1 + arg2 + arg3        
        os.system(full_cmd)
    
def spheres_intersection(point1, radius1, point2, radius2):
    '''
    calculate the volume of two spheres' intersection
    http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    '''
    #distance between two center points
    
    import math 
    d = math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]) + (point1[2]-point2[2])*(point1[2]-point2[2]))
    
    if (d>=(radius1 + radius2)):
        return d, 0.

    volume = math.pi * ((radius1 + radius2 - d)*(radius1 + radius2 - d)) * (d*d + 2*d*radius1 - 3*(radius1*radius1) + 2*d*radius2 +6*radius1*radius2 - 3*(radius2*radius2)) / (12*d)
    
    return d, volume

def nativevoxel2MNImm(point, anatomy, flirt_mat ):
    import os
    cmd = 'echo "' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '" | img2stdcoord '
    arg1 = '-img ' + anatomy + ' '#/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz '
    #arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm -xfm '
    arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -xfm '
    arg3 = flirt_mat + ' -vox'#'/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox'
    full_cmd = cmd + arg1 + arg2 + arg3    
    #os.system('echo "156 111 145" | img2stdcoord -img /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz -std $FSLDIR/data/standard/MNI152_T1_1mm -xfm /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox')
    #print full_cmd    
    #print point    
    os.system(full_cmd)

def MNImm2MNIvoxel(point):
    import os
    #echo "-36.18 -21.52 41.28" | std2imgcoord -img $FSLDIR/data/standard/MNI152_T1_1mm -std $FSLDIR/data/standard/MNI152_T1_1mm -vox -
    cmd = 'echo "' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '" | std2imgcoord '
    #arg1 = '-img $FSLDIR/data/standard/MNI152_T1_1mm '
    #arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm -vox -'  
    arg1 = '-img $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz '
    arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -vox -'  
    
    full_cmd = cmd + arg1 + arg2    
    #print point    
    os.system(full_cmd)    

def concat(tract1, tract2):
    
    res = []
    for i in np.arange(len(tract1)):
        res.append(tract1[i])
    
    for j in np.arange(len(tract2)):
        res.append(tract2[j])
    
    return res
    
def overlap(A1, A2):
    '''
    Return the overlap array between two arrays: A1 and A2
    Note that if in A1, one element appears many times, then these appearance are different
    So, the return is an array not a set.
    '''
    overlap = []
    for i in np.arange(len(A1)):
        if (A1[i] in A2):
            overlap.append(A1[i])
    return overlap
    
def minus(A1, A2):
    '''
    Return the A1 - A2
    Note that if in A1, one element appears many times, then these appearance are different
    So, the return is an array not a set.
    '''
    minus = []
    for i in np.arange(len(A1)):
        if (A1[i] not in A2):
            minus.append(A1[i])
    return minus
            
def load_whole_tract(tracks_filename):
    
    from dipy.io.dpy import Dpy
    from dipy.io.pickles import load_pickle
    dpr_tracks = Dpy(tracks_filename, 'r')
    all_tracks=dpr_tracks.read_tracks()
    dpr_tracks.close()
    
    all_tracks = np.array(all_tracks,dtype=np.object)
    return all_tracks
    
def load_tract(tracks_filename, id_file):
    
    from dipy.io.dpy import Dpy
    from dipy.io.pickles import load_pickle
    dpr_tracks = Dpy(tracks_filename, 'r')
    all_tracks=dpr_tracks.read_tracks()
    dpr_tracks.close()
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]
    
    tract = np.array(tract,dtype=np.object)
    return tract
   
def load_whole_tract_trk(tracks_filename):
    '''
    load tract from trackvis format
    '''
    import nibabel as nib
    streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
    all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    return all_tracks
    
def load_tract_trk(tracks_filename, id_file):
    '''
    load tract from trackvis format
    '''
    import nibabel as nib
    streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
    all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    from dipy.io.pickles import load_pickle
    tracks_id = load_pickle(id_file)
    tract = [all_tracks[i] for i  in tracks_id]
    
    tract = np.array(tract,dtype=np.object)  
    
    return tract
    
def load_trk_file(tracks_filename):
    '''
    load tract from trackvis format
    '''
    import nibabel as nib
    streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
    all_tracks = np.array([s[0] for s in streams], dtype=np.object)
     
    return all_tracks
    
def save_id_tract_plus_sff(tracks_filename, id_file, num_proto, distance, out_fname):
   
    dpr_tracks = Dpy(tracks_filename, 'r')
    all_tracks=dpr_tracks.read_tracks()
    dpr_tracks.close()
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]    
    
    not_tract_fil = []
    id_not_tract_fil = []
    min_len = min(len(i) for i in tract)
    #print 'min_len of cst', min_len
    min_len = min_len*2.2/3#2./3.2# - 20
    for i in np.arange(len(all_tracks)):
        if (i not in tracks_id) and (length(all_tracks[i]) > min_len):
            not_tract_fil.append(all_tracks[i])
            id_not_tract_fil.append(i)
    
    not_tract_fil = np.array(not_tract_fil,dtype=np.object)        
    sff_pro_id = sff(not_tract_fil, num_proto, distance)        
    
    tract_sff_id = []
    for i in tracks_id:
        tract_sff_id.append(i)
        
    for idx in sff_pro_id:        
        tract_sff_id.append(id_not_tract_fil[idx])
        
    #tract_sff_id.append(id_not_tract_fil[i] for i in sff_pro_id)
    print len(tract), len(tract_sff_id)
    save_pickle(out_fname, tract_sff_id)
    return tract_sff_id
   
def save_id_tract_ext(tracks_filename, id_file,  distance, out_fname):
    
    dpr_tracks = Dpy(tracks_filename, 'r')
    all_tracks=dpr_tracks.read_tracks()
    dpr_tracks.close()
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]    
    
    not_tract_fil = []
    id_not_tract_fil = []
    min_len = min(len(i) for i in tract)
    #print 'min_len of cst', min_len
    min_len = min_len*2.2/3#2./3.2# - 20
    for i in np.arange(len(all_tracks)):
        if (i not in tracks_id) and (length(all_tracks[i]) > min_len):
            not_tract_fil.append(all_tracks[i])
            id_not_tract_fil.append(i)
       
   
    k = np.round(len(tract)*1.2)   
            
    from dipy.segment.quickbundles import QuickBundles
    
    qb = QuickBundles(tract,200,18)
    
    medoid_tract = qb.centroids[0]
    
    med_nottract_dm =  distance([medoid_tract], not_tract_fil)
    med_tract_dm =  distance([medoid_tract], tract)
    
    tract_rad = med_tract_dm[0][np.argmax(med_tract_dm[0])]
    len_dis = tract_rad * 2.8/2.
   
    #k_indices which close to the medoid
    sort = np.argsort(med_nottract_dm,axis = 1)[0]
    #print sort[:k+1]
    while (k>0 and med_nottract_dm[0][sort[k]]>=len_dis):
        k = k - 1
        
    
    #print k
    #close_indices = np.argsort(cst_dm,axis = 1)[:,0:k][0]
    close_indices = sort[0:k]
    
    #for idx in close_indices:
    #    tract_ext.append(not_tract_fil[idx])          
    #print 'close indices', len(close_indices)
    tract_ext_id = []
    for i in tracks_id:
         tract_ext_id.append(i)
    
    #print 'Before', len(tract_ext_id)
    
    for idx in close_indices:
        tract_ext_id.append(id_not_tract_fil[idx]) 
    #    print idx, id_not_tract_fil[idx]
      
    #print 'After', len(tract_ext_id)
    #tract_ext_id = [i for i in tracks_id]
    #tract_ext_id.append(id_not_tract_fil[i] for i in close_indices)
    
    save_pickle(out_fname, tract_ext_id)
    return tract_ext_id
    
def save_id_tract_ext_plus_sff(tracks_filename, id_file, num_proto, distance, out_fname_ext_sff, out_fname_ext = 'temp'): 
    tract_ext_id = save_id_tract_ext(tracks_filename,id_file, distance, out_fname_ext)
    return save_id_tract_plus_sff(tracks_filename, out_fname_ext, num_proto,distance, out_fname_ext_sff)
    
    
def visualize_tract(ren, tract,color=fvtk.red):    
    for i in np.arange(len(tract)):
        fvtk.add(ren, fvtk.line(tract[i], color, opacity=1.0))        
    return ren

def visualize_mapped(ren, tract2, mapping, color = fvtk.blue):
    for i in np.arange(len(mapping)):        
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], color, opacity=1.0))     
    return ren