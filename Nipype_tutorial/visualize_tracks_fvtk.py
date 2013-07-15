# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:01:00 2013

@author: bao
"""


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


dirname = "data/dmri/temp6/"
#for visualization of dpy convert from trackvis format in DTI_Cattaneo folder        
        
tracks_filename = dirname + 'dti_tracks.dpy'
dpr_tracks = Dpy(tracks_filename, 'r')
tensor_tracks=dpr_tracks.read_tracks();
dpr_tracks.close()
print(len(tensor_tracks))      

visualize = True
if visualize:
    from dipy.viz import fvtk
    renderer = fvtk.ren()
    #fvtk.add(renderer, fvtk.line(tensor_tracks[0:500], fvtk.red, opacity=1.0))
    #   fvtk.add(renderer, fvtk.line(tensor_tracks_warped[0:500], fvtk.green, opacity=1.0))
    #for trackvis formart            
    fvtk.add(renderer, fvtk.line(tensor_tracks[0:500], fvtk.red, opacity=1.0))
 
    fvtk.show(renderer)