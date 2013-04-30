# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:31:00 2013

@author: bao
"""

import os
import sys
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from nibabel import trackvis
#from dipy.io.pickles import load_pickle,save_pickle

      
dirname = "Segmentation/ROI"
for root, dirs, files in os.walk(dirname):
  for files_i in files:
    if files_i.endswith('.trk'):
    #if root.endswith('101_32'):
        
        base_dir = root+'/'       
        basedir_recon = os.getcwd() + '/' + base_dir #+ 'DTI/'
        
        
        #read from trackvis file
        streams, hdr = trackvis.read(basedir_recon + files_i)
        streamlines = [s[0] for s in streams]    
        print files_i
        new_files_i = files_i.replace('.trk','.dpy')
        #stop
        #write in .dpy file
        dpw = Dpy(basedir_recon +  new_files_i, 'w')
        dpw.write_tracks(streamlines)
        dpw.close()
        
        #read from .dpy file and visualize to test        
        tracks_filename = basedir_recon + new_files_i
        dpr_tracks = Dpy(tracks_filename, 'r')
        tensor_tracks=dpr_tracks.read_tracks();
        dpr_tracks.close()
        
        try:
            from dipy.viz import fvtk
        except ImportError:
            raise ImportError('Python vtk module is not installed')
            sys.exit()
        
        r=fvtk.ren()
        from dipy.viz.colormap import line_colors
        
        fvtk.add(r, fvtk.line(tensor_tracks, line_colors(tensor_tracks)))
        fvtk.show(r)        
        #print('Saving illustration as tensor_tracks.png')
        #fvtk.record(r, n_frames=1, out_path='tensor_tracking.png', size=(600, 600))
        
'''  
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
'''