# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:07:37 2012

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:06:42 2011

@author: bao
"""
import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.io.dpy import Dpy

#dirname = "data/111019"
#dirname = "data"
dirname= "data/1891215"
for root, dirs, files in os.walk(dirname):
    if root.endswith('DTI64_1'):
        base_dir = root+'/'

       
        basedir_recon = os.getcwd() + '/' + base_dir + 'DTI/'       
        #fdpyw = basedir_recon + 'tracks_dti_10K_warped.dpy'
        #fdpyw = basedir_recon + 'tracks_warped.dpy'
                
        #dpr_tracks_warped = Dpy(fdpyw, 'r')
        #tensor_tracks_warped = dpr_tracks_warped.read_tracks()
        #dpr_tracks_warped.close()

        data_filename = base_dir + 'DTI/data_resample.nii.gz'        
        print("Loading nifti data: %s" % data_filename)
        img = nib.load(data_filename)

        data = img.get_data()
        affine = img.get_affine()
        print affine
        #1/0



        #tracks_filename = basedir_recon + 'tracks_linear.dpy'
        tracks_filename = basedir_recon + 'tracks_dti_10K_linear.dpy'
        dpr_tracks = Dpy(tracks_filename, 'r')
        tensor_tracks=dpr_tracks.read_tracks();
        dpr_tracks.close()

        visualize = True
        if visualize:
            from dipy.viz import fvtk
            renderer = fvtk.ren()
            fvtk.add(renderer, fvtk.line(tensor_tracks[0:500], fvtk.red, opacity=1.0))
         #   fvtk.add(renderer, fvtk.line(tensor_tracks_warped[0:500], fvtk.green, opacity=1.0))
            fvtk.show(renderer)