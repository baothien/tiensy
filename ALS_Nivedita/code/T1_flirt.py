# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:00:49 2012

@author: bao
"""

from dipy.external.fsl import bet, pipe
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import warp_displacements, warp_displacements_tracks
import os

def create_displacements_T1(in_nii,affine_mat,out_nii,nonlin_nii,invw_nii,disp_nii,dispa_nii):
    commands=[]    
    commands.append('flirt -in '+in_nii+' -ref '+'/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -out '+ out_nii+' -omat ' + affine_mat+' -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear') 
    for c in commands:
        print(c)
        pipe(c)
 



visualize = False
dirname = "data/"
#dirname = "data_temp/"
for root, dirs, files in os.walk(dirname):
    if root.endswith('MP_Rage_1x1x1_ND_3'):
        base_dir = root+'/'
        filename = 'anatomy.nii.gz'
        base_filename = base_dir + filename        

        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g 0')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .4 -g -0.05 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g -0.02 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g -0.02 -c 151 177 90')         
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .15 -g 0.0 -c 151 177 90')                
        
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                        
        bet(base_filename, root+'/anatomybet.nii.gz',options=' -R -f .2 -g 0.0' )         
        T1_filename = base_dir + '/anatomybet.nii.gz'   
        basedir_recon = os.getcwd() + '/' + base_dir  
        
        print T1_filename
                   
        flirt_affine = basedir_recon + 'flirt_T1.mat'
        flirt_displacements = basedir_recon + 'displacements_T1.nii.gz'
        fsl_ref = '/usr/share/fsl/data/standard/MNI152lin_T1_1mm_brain.nii.gz'
        flirt_out = basedir_recon + 'T1_flirt_out.nii.gz'
        flirt_warp = basedir_recon + 'T1_warped.nii.gz'
        nonlin_nii = basedir_recon + 'nonlinear.nii.gz'
        invw_nii = basedir_recon + 'invw.nii.gz'
        disp_nii = basedir_recon + 'disp.nii.gz'
        dispa_nii = basedir_recon + 'dispa.nii.gz'
        
        
        #create_displacements_T1(T1_filename, flirt_affine, nonlin_nii, invw_nii, disp_nii, dispa_nii)
        create_displacements_T1(T1_filename, flirt_affine,flirt_out, nonlin_nii,invw_nii, disp_nii, dispa_nii)

        print "Finish creating displacements"