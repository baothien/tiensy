# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:40:01 2012

@author: bao
"""
import os
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
    
visualize = False

tracks_filename=['tracks_dti_10K_linear.dpy','tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy']
#,
#                 'tracks_gqi_10K_linear.dpy','tracks_gqi_1M_linear.dpy','tracks_gqi_3M_linear.dpy']

qb_filename_15=['qb_dti_10K_linear_15.pkl','qb_dti_1M_linear_15.pkl','qb_dti_3M_linear_15.pkl']
#,
#                 'qb_gqi_10K_linear_15.pkl','qb_gqi_1M_linear_15.pkl','qb_gqi_3M_linear_15.pkl']

qb_filename_20=['qb_dti_10K_linear_20.pkl','qb_dti_1M_linear_20.pkl','qb_dti_3M_linear_20.pkl']
#,
#                 'qb_gqi_10K_linear_20.pkl','qb_gqi_1M_linear_20.pkl','qb_gqi_3M_linear_20.pkl']

qb_filename_30=['qb_dti_10K_linear_30.pkl','qb_dti_1M_linear_30.pkl','qb_dti_3M_linear_30.pkl']
#,
 #                'qb_gqi_10K_linear_30.pkl','qb_gqi_1M_linear_30.pkl','qb_gqi_3M_linear_30.pkl']

#dirname = "data_temp"
dirname = "data_qb"
def load_tracks(filename):
    dpr = Dpy(filename, 'r')
    tracks=dpr.read_tracks()
    dpr.close()
    return tracks
    
    
name1 = 0
name2 = 0
for root, dirs, files in os.walk(dirname):
#    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'):
#        base_dir = root+'/' 
#        filename = 'raw'
#        base_dir2 = base_dir+ 'DTI/'  
#        name1 = 1
#    if root.endswith('MP_Rage_1x1x1_ND_3'):
#        print root+'/T1_flirt_out.nii.gz'           
#        img = nib.load(root +'/T1_flirt_out.nii.gz')
#        data = img.get_data()
#        name2 = 1
#    if (name1==1 and name2==1):  
#     
    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'): 
        base_dir = root+'/' 
        filename = 'raw'
        base_dir2 = base_dir+ 'DTI/'  
        print base_dir+'../MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz'           
        img = nib.load(base_dir+'../MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz')
        data = img.get_data()
       
        for i in range(len(tracks_filename)):
            print '>>>>'            
            print base_dir2+tracks_filename[i]
            tracks=load_tracks(base_dir2+tracks_filename[i])            
            #shift in the center of the volume            
            tracks=[t-np.array(data.shape)/2. for t in tracks]

            print base_dir2+qb_filename_15[i]            
            qb=QuickBundles(tracks,15.,12)
            save_pickle(base_dir2+qb_filename_15[i],qb)
            
            print base_dir2+qb_filename_20[i]
            qb=QuickBundles(tracks,20.,12)
            save_pickle(base_dir2+qb_filename_20[i],qb)
            
            print base_dir2+qb_filename_30[i]
            qb=QuickBundles(tracks,30.,12)            
            save_pickle(base_dir2+qb_filename_30[i],qb)            
        name1=0
        name2=0
        
        #break
            

            

        
        


