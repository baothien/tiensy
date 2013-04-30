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
#from dipy.tracking.utils import density_map

#from dipy.segment.quickbundles import QuickBundles
#from dipy.viz.colormap import orient2rgb
#from dipy.reconst.dti import Tensor
#from dipy.tracking.propagation import EuDX
#import copy

#patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
#controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
#patients = [2]
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
 
patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
#patients = [2]
#mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]    

# for new segmentation 1210
#patients = [9,11] #patient_left only - there is no patient right in new segmentation 
#controls = [10,24]#controls_left 
#controls = [12,16,18]#controls_right   
            

mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]     
if __name__ == '__main__':
    
    
   
    dir_name = 'data'
    num_seeds = 3  


#first compute the voxel cordinate of each fiber in CST
#it only needs to run one time at the first because all data are saved in file
#   
    #for p in np.arange(24)+1:
    for p in patients:#controls:
        for a in ['L']:#,'L']:
            print p               
            cst_file_name =  dir_name + '/segmentation/' + str(p) + '_corticospinal_' + a + '_3M.pkl'           
            print cst_file_name
            #load id of tracks in CST left        
            tracks_id=load_pickle(cst_file_name)
            
            subj = int(mapping[p])
            
            #load the tracks         
            tracks_filename = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'
            dpr_tracks = Dpy(tracks_filename, 'r')
            tensor_all_tracks=dpr_tracks.read_tracks();
            dpr_tracks.close()
            cst_tracks=[tensor_all_tracks[i] for i in tracks_id]


            #load the volume
            img = nib.load('data/'+str(subj)+'/MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz') 
            data = img.get_data()        
            vol_dims = data.shape
            #compute the number of fiber crossing every voxel
            tcs=track_counts(cst_tracks,vol_dims,return_elements=False)
            #print tcs
            #compute the volumn and the list of voxel that fiber crossing
            vol, list_index = volumns(tcs)
            #print vol
            #print list_index               
            cordinate_file_name =  dir_name + '/segmentation/cordinate_' + str(p) + '_corticospinal_' + a + '_3M.pkl'   
            print cordinate_file_name 
            save_pickle(cordinate_file_name,list_index)        
            print 'saving finish'


    #       for loading later
    #        cordinate_file_name =  dir_name + '/segmentation/cordinate_' + str(p) + '_corticospinal_R_3M.pkl'   
    #        cordinate_id=load_pickle(cordinate_file_name)
    #        print cordinate_id
    #         print len(cordinate_id)
    #        stop


##second compute the volumn of CST
#    print 'Name \t volumn'    
#    for a in ['L']:#,'R']:
#        for p in [17]:#patients:      
#        #for p in controls:        
#        #for p in np.arange(24)+1:
#            cordinate_file_name =  dir_name + '/segmentation/cordinate_' + str(p) + '_corticospinal_' + a + '_3M.pkl'           
#            cordinate_id=load_pickle(cordinate_file_name)
#            print cordinate_file_name, '\t', len(cordinate_id)            
#            
#            
#        # only for visulization detail
#           # print cordinate_id
#           # print len(cordinate_id)
#            
#           # for i in np.arange(len(cordinate_id)):
#           #     print cordinate_id[i]
#
#    
#
