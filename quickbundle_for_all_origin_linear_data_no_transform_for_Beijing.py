# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:34:20 2012

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:58:05 2012

@author: bao
"""

import os
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from dipy.external.fsl import flirt2aff,create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
from dipy.tracking.metrics import downsample

#==============quick bundles ===================================================

#import os
#import numpy as np
#import nibabel as nib
#from dipy.io.dpy import Dpy
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_sphere
    

tracks_filename_arr=['tracks_dti_10K_linear.dpy','tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy']
#,
#                 'tracks_gqi_10K_linear.dpy','tracks_gqi_1M_linear.dpy','tracks_gqi_3M_linear.dpy']

qb_filename_15=['qb_dti_10K_linear_15_no_transform.pkl','qb_dti_1M_linear_15_no_transform.pkl','qb_dti_3M_linear_15_no_transform.pkl']
#,
#                 'qb_gqi_10K_linear_15_no_transform.pkl','qb_gqi_1M_linear_15_no_transform.pkl','qb_gqi_3M_linear_15_no_transform.pkl']

qb_filename_20=['qb_dti_10K_linear_20_no_transform.pkl','qb_dti_1M_linear_20_no_transform.pkl','qb_dti_3M_linear_20_no_transform.pkl']
#,
#                 'qb_gqi_10K_linear_20_no_transform.pkl','qb_gqi_1M_linear_20_no_transform.pkl','qb_gqi_3M_linear_20_no_transform.pkl']

qb_filename_30=['qb_dti_10K_linear_30_no_transform.pkl','qb_dti_1M_linear_30_no_transform.pkl','qb_dti_3M_linear_30_no_transform.pkl']
#,
 #                'qb_gqi_10K_linear_30_no_transform.pkl','qb_gqi_1M_linear_30_no_transform.pkl','qb_gqi_3M_linear_30_no_transform.pkl']

def load_tracks(filename):
    dpr = Dpy(filename, 'r')
    tracks=dpr.read_tracks()
    dpr.close()
    return tracks
#=====================quick bundles ============================================

def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
        
def create_save_tracks(anisotropy,indices, seeds, low_thresh,filename):
    
    euler = EuDX(anisotropy, 
                        ind=indices, 
                        #odf_vertices=get_sphere('symmetric362'),
                        seeds=seeds, a_low=low_thresh)
                        #odf_vertices=get_sphere('symmetric362').vertices, 
    tracks = [track for track in euler]     
    dpw = Dpy(filename, 'w')
    dpw.write_tracks(tracks)
    dpw.close()
    
def warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename):
    fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    
    img_fa = nib.load(fa_filename)            

    flirt_affine= np.loadtxt(flirt_filename)    
        
    img_ref =nib.load(fsl_ref)
    
    #create affine matrix from flirt     
    mat=flirt2aff(flirt_affine,img_fa,img_ref)        

     #read tracks    
    dpr = Dpy(tracks_filename, 'r')
    tensor_tracks = dpr.read_tracks()
    dpr.close()
        
    #linear tranform for tractography
    tracks_warped_linear = transform_tracks(tensor_tracks,mat)        

    #save tracks_warped_linear    
    dpr_linear = Dpy(linear_filename, 'w')
    dpr_linear.write_tracks(tracks_warped_linear)
    dpr_linear.close()
     
    
#==============quick bundles ==================================
#dirname = "ALS/ALS_temp"
#dirname = "ADHD_data/2213364"
dirname = "ADHD_data/1087458"
for root, dirs, files in os.walk(dirname):
    if root.endswith('DTI64_1'): 
        base_dir = root+'/' 
        filename = 'raw'
        base_dir2 = base_dir+ 'DTI/'  
       
        for i in range(len(tracks_filename_arr)):
            print '>>>>'            
            print base_dir2+tracks_filename_arr[i]
            tracks=load_tracks(base_dir2+tracks_filename_arr[i]) 
            print len(tracks)           
            #tracks = [downsample(t, 12) - np.array(data.shape)/2. for t in tracks]
            tracks = [downsample(t, 12) for t in tracks]
           
            print base_dir2+qb_filename_15[i]            
            qb=QuickBundles(tracks,15.,12)
            save_pickle(base_dir2+qb_filename_15[i],qb)
            
            print base_dir2+qb_filename_20[i]
            qb=QuickBundles(tracks,20.,12)
            save_pickle(base_dir2+qb_filename_20[i],qb)
            
            print base_dir2+qb_filename_30[i]
            qb=QuickBundles(tracks,30.,12)            
            save_pickle(base_dir2+qb_filename_30[i],qb)            

        print('Done quickbundle for : ')
        print(base_dir2)
        #stop
