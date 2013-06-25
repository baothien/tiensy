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
from nibabel import trackvis
   
dirname = "PBC2009"
for root, dirs, files in os.walk(dirname):
 for fl in files:
  if fl.endswith('.trk'):
    #if root.endswith('101_32'):
    base_dir = root+'/'       
    basedir_recon = os.getcwd() + '/' + base_dir + '/'               

    fname = basedir_recon + fl
    print '     ',fname
    
    #read the trackvis format file
    streams, hdr = trackvis.read(fname)
    streamlines = [s[0] for s in streams]
    print (len(streamlines))  
    #print streamlines[0]
    
    #store streamlines into the dipy format .dpy
    dpy_fname = basedir_recon + 'tracks_dti.dpy'
    dpw = Dpy(dpy_fname, 'w')
    dpw.write_tracks(streamlines)    
    dpw.close()
    print '>>>> Done ', dpy_fname
    #stop
    
        
    
        



