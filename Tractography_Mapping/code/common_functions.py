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
            
def load_tract(tracks_filename, id_file):
    
    from dipy.io.dpy import Dpy
    from dipy.io.pickles import load_pickle
    dpr_tracks = Dpy(tracks_filename, 'r')
    all_tracks=dpr_tracks.read_tracks()
    dpr_tracks.close()
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]
    
    return tract
    
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