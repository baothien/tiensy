# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:46:32 2014

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:09:38 2013

@author: bao
"""

import os
import sys
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
#from dipy.io.pickles import load_pickle,save_pickle
#tracks_filename_arr=['tracks_dti_10K_linear.dpy','tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy'] #use fa_warped
tracks_filename_arr=['tracks_dti_10K.dpy','tracks_dti_1M.dpy','tracks_dti_3M.dpy'] # use original fa

#trackvis_filename_arr=['tracks_dti_10K_linear.trk','tracks_dti_1M_linear.trk','tracks_dti_3M_linear.trk']
trackvis_filename_arr=['tracks_dti_10K.trk','tracks_dti_1M.trk','tracks_dti_3M.trk']
      

dirname = "Cambridge/subj_05/101_32"
for root, dirs, files in os.walk(dirname):
  if root.endswith('DTI'):
    base_dir = root#+'/'       
    basedir_recon = os.getcwd() + '/' + base_dir + '/'               
        
    #loading fa_image
    #with tracks_linear
    #fa_file = basedir_recon + 'fa_warped.nii.gz'            
     
    #with track original 
    fa_file = basedir_recon + 'fa.nii.gz'
    fa_img = nib.load(fa_file)
    fa = fa_img.get_data()
    fa[np.isnan(fa)] = 0
    for i in range(len(tracks_filename_arr)):
        tracks_filename = basedir_recon + tracks_filename_arr[i]
        dpr_tracks = Dpy(tracks_filename, 'r')
        tensor_tracks=dpr_tracks.read_tracks();
        dpr_tracks.close()

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
    
        



