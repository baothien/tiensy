# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:43:34 2012

@author: bao
"""

import numpy as np
import nibabel as nib
import os.path as op
import pyglet
import os

#dipy modules

from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
from dipy.tracking.metrics import length
from dipy.tracking.vox2track import track_counts
from dipy.tracking.utils import density_map

#from dipy.segment.quickbundles import QuickBundles
#from dipy.viz.colormap import orient2rgb
#from dipy.reconst.dti import Tensor
#from dipy.tracking.propagation import EuDX
#import copy

#patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
#controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
#controls = [2,4,6,8,10,12,14,18,20,22,24]#12 controls16 miss R CST
#mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]    
def length_min(tracks):
    i_min = 0        
    for k in range(len(tracks)):
        if len(tracks[k])<len(tracks[i_min]):
            i_min = k
        
    return len(tracks[i_min])

def length_max(tracks):
    i_max = 0        
    for k in range(len(tracks)):
        if len(tracks[k])>len(tracks[i_max]):
            i_max = k
        
    return len(tracks[i_max])    

def length_avg(tracks):
    s = 0            
    for k in range(len(tracks)):
        s = s + len(tracks[k])
    return s/len(tracks)          
    
def truth_length_min(tracks):
    lmin = length(tracks[0], False)        
    for k in range(len(tracks)):
        if lmin>length(tracks[k]):
            lmin = length(tracks[k])        
    return lmin

def truth_length_max(tracks):
    lmax = length(tracks[0], False)        
    for k in range(len(tracks)):
        if lmax<length(tracks[k],False):
            lmax = length(tracks[k])       
    return lmax

def truth_length_avg(tracks):
    s = 0            
    for k in range(len(tracks)):
        s = s + length(tracks[k])
    return s/len(tracks)    
    
def volumns(vol):
    #input: a volume
    #count the number of voxel in volume which has the value > 0
    #return: N: number of voxel in volume which has the value > 0
    #        an array shape (N,3) of the intergers, where the (x,y,z) is the cordinate of voxel having value > 0
    s = vol.shape
    cordinate = []
    count = 0
    for x in np.arange(s[0]):
        for y in np.arange(s[1]):
            for z in np.arange(s[2]):
                if vol[x,y,z]>0:
                    cordinate.append((x,y,z))        
                    count = count + 1#vol[x,y,z]
    return count,cordinate
                

# for new segmentation 1210
#patients = [9,11] #patient_left only - there is no patient right in new segmentation 
#controls = [10,24]#controls_left 
#controls = [12,16,18]#controls_right  

patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
             

mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]     


if __name__ == '__main__':
    
    
   
    dir_name = 'data'
    num_seeds = 3
    features = []
    
    #print "\t Num fibers \t Size \t Length Min \t Length Max \t Length Average "         
    for a in ['L','R']:
    #for a in ['L']:#,'R']:        
        features =[]
        
        #features_name = dir_name + '/segmentation/patient_fiber_number_corticospinal_' + a + '_3M_temp.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_number_corticospinal_' + a + '_3M.txt'   
        
        
        #features_name = dir_name + '/segmentation/patient_fiber_len_min_corticospinal_' + a + '_3M.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_len_min_corticospinal_' + a + '_3M.txt'          
        
        #features_name = dir_name + '/segmentation/patient_fiber_len_max_corticospinal_' + a + '_3M.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_len_max_corticospinal_' + a + '_3M.txt'          
        
        #features_name = dir_name + '/segmentation/patient_fiber_len_avg_corticospinal_' + a + '_3M.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_len_avg_corticospinal_' + a + '_3M.txt'          
        
        #features_name = dir_name + '/segmentation/patient_fiber_truth_len_min_corticospinal_' + a + '_3M.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_truth_len_min_corticospinal_' + a + '_3M.txt'          
        
        #features_name = dir_name + '/segmentation/patient_fiber_truth_len_max_corticospinal_' + a + '_3M.txt'          
        #features_name = dir_name + '/segmentation/control_fiber_truth_len_max_corticospinal_' + a + '_3M.txt'          
        
        #features_name = dir_name + '/segmentation/patient_fiber_truth_len_avg_corticospinal_' + a + '_3M.txt'          
        features_name = dir_name + '/segmentation/control_fiber_truth_len_avg_corticospinal_' + a + '_3M.txt'          

        
        print features_name          
        #for p in patients:      
        for p in controls:        
            cst_file_name =  dir_name + '/segmentation/' + str(p) + '_corticospinal_' + a + '_' + str(num_seeds) + 'M.pkl'           
            #load id of tracks in CST left        
            tracks_id=load_pickle(cst_file_name)

        
#------------------------- number of fibers -----------------------------
#            features.append(len(tracks_id))                 
#------------------------- number of fibers -----------------------------
            

#----------------------------length ----------------------                
            subj = int(mapping[p])
            
            #load the tracks         
            tracks_filename = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'
            dpr_tracks = Dpy(tracks_filename, 'r')
            tensor_all_tracks=dpr_tracks.read_tracks();
            dpr_tracks.close()
            cst_tracks=[tensor_all_tracks[i] for i in tracks_id]
#    

            
            #len_min = length_min(cst_tracks)
            #features.append(len_min)                 
            
#            len_max = length_max(cst_tracks)
#            features.append(len_max) 
                
            #len_avg = length_avg(cst_tracks)
            #features.append(len_avg) 
            #len_min_truth = truth_length_min(cst_tracks)
            #features.append(len_min_truth) 
            #len_max_truth = truth_length_max(cst_tracks)
            #features.append(len_max_truth) 
            len_avg_truth = truth_length_avg(cst_tracks)
            features.append(len_avg_truth) 
#                       
#        print cst_file_name,"\t", cst_tracks.__len__(), "\t",cst_tracks.__sizeof__(), "\t" ,len_min,"\t", len_max,"\t", len_avg          
            
            #features.append(len_min_truth)                 
#            features.append(len_max_truth)  
            #features.append(len_avg_truth)  
#-------------------------------------------------------------------
            
#saving features        
        save_pickle(features_name,features)
                
        #for test and load later        
        features=load_pickle(features_name)
        print features