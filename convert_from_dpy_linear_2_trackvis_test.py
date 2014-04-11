# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:24:10 2013

@author: bao
"""
'''
Important Note:
   if the tract is original (not linear), then when converting to trackvis, the FA (original) should be used
   if the linear_tract is after warping (also the CST result - in MNI space), then when converting to trackvis, the FA_warped have to be used.
   Can not convert linear_tract (also CST result in MNI space) with the original FA --> wrong


'''


import os
import sys
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
#from dipy.io.pickles import load_pickle,save_pickle
from dipy.tracking.metrics import downsample
tracks_filename_arr=['tracks_dti_10K_linear.dpy']#,'tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy',
                     #'tracks_dti_10K.dpy','tracks_dti_1M.dpy','tracks_dti_3M.dpy']

trackvis_filename_arr=['tracks_dti_10K_linear_test_coregistration_130605_convert_fawarped+.trk']#,'tracks_dti_1M_linear.trk','tracks_dti_3M_linear.trk',
                       #'tracks_dti_10K.trk','tracks_dti_1M.trk','tracks_dti_3M.trk']
      
#dirname = "ALS/ALS_temp"
dirname = "ALS/ALS_Data/101"
for root, dirs, files in os.walk(dirname):
  if root.endswith('DTI'):
    #if root.endswith('101_32'):
    base_dir = root+'/'       
    basedir_recon = os.getcwd() + '/' + base_dir + '/'               
        
        #loading fa_image
    fa_file = basedir_recon + 'fa_warped.nii.gz'            
    fa_img = nib.load(fa_file)
    fa = fa_img.get_data()
    fa[np.isnan(fa)] = 0
    for i in range(len(tracks_filename_arr)):
        tracks_filename = basedir_recon + tracks_filename_arr[i]
        dpr_tracks = Dpy(tracks_filename, 'r')
        tensor_tracks_1=dpr_tracks.read_tracks();
        dpr_tracks.close()
        
        tensor_tracks = T
                
        #end of moi them vao 20130604
        #----------------------------------------------------------------------------        
    
        
        

        """
        We can now save the results in the disk. For this purpose we can use the
        TrackVis format (*.trk). First, we need to create a header.
        """

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = fa_img.get_header().get_zooms()[:3]
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = fa.shape

        """
        Then we need to input the streamlines in the way that Trackvis format expects them.
        """

        tensor_streamlines_trk = ((sl, None, None) for sl in tensor_tracks)

        ten_sl_fname = basedir_recon + trackvis_filename_arr[i]

        """
        Save the streamlines.
        """

        nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')
        
        print '        Create', ten_sl_fname
    print '>>>>>>>>>> Done ', base_dir 
    