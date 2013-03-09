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

      
dirname = "data_temp"
for root, dirs, files in os.walk(dirname):
    if root.endswith('DTI'):
    #if root.endswith('101_32'):
        base_dir = root+'/'       
        basedir_recon = os.getcwd() + '/' + base_dir #+ 'DTI/'               
        
        #loading fa_image
        fa_file = base_dir + 'fa.nii.gz'            
        fa_img = nib.load(fa_file)
        fa = fa_img.get_data()
        fa[np.isnan(fa)] = 0
                         
        tracks_filename = basedir_recon + 'tracks_dti_10K.dpy'
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

        ten_sl_fname = basedir_recon + 'tracks_dti_10K.trk'

        """
        Save the streamlines.
        """

        nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')
        
        print 'Done'



