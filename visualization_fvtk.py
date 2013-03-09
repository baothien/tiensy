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
#dirname= "data/1891215"
#dirname = "ADHD_data/1087458"
dirname = "ALS/ALS_temp"
for root, dirs, files in os.walk(dirname):
    #if root.endswith('DTI64_1'):
        #base_dir = root+'/'       
        #basedir_recon = os.getcwd() + '/' + base_dir + 'DTI/'       
        
        #fdpyw = basedir_recon + 'tracks_dti_10K_warped.dpy'
        #fdpyw = basedir_recon + 'tracks_warped.dpy'                
        #dpr_tracks_warped = Dpy(fdpyw, 'r')
        #tensor_tracks_warped = dpr_tracks_warped.read_tracks()
        #dpr_tracks_warped.close()

        #load tracks in .dpy format
#        tracks_filename = basedir_recon + 'tracks_dti_10K_linear.dpy'
#        dpr_tracks = Dpy(tracks_filename, 'r')
#        tensor_tracks=dpr_tracks.read_tracks();
#        dpr_tracks.close()
        
#-------------------------------------------------------------------------
#load tracks in .trk format     
     if root.endswith('trackvis_format'):        
        base_dir = root+'/'
        basedir_recon = os.getcwd() + '/' + base_dir + '/'       
        tracks_filename = basedir_recon + 'tracks_dti_10K.trk'
        dpr_tracks, hdr = nib.trackvis.read(tracks_filename)
        tensor_tracks =(sl[0] for sl in dpr_tracks) 
        #for i in [0,len(dpr_tracks)]:
        #    tensor_tracks[i] = dpr_tracks.__iter__(),dpr_tracks.next() 
        #tensor_tracks=dpr_tracks[0:]#.throw#.read_tracks();
        
        #end of load tracks in .trk format
#------------------------------------------------------------------------
        
        visualize = True
        if visualize:
            from dipy.viz import fvtk
            renderer = fvtk.ren()
            #fvtk.add(renderer, fvtk.line(tensor_tracks[0:500], fvtk.red, opacity=1.0))
            #   fvtk.add(renderer, fvtk.line(tensor_tracks_warped[0:500], fvtk.green, opacity=1.0))
            #for trackvis formart            
            fvtk.add(renderer, fvtk.line(tensor_tracks[0:500], fvtk.red, opacity=1.0))
         
            fvtk.show(renderer)