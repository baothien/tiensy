# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:00:49 2012

@author: bao
"""

from dipy.external.fsl import bet,pipe
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from dipy.external.fsl import warp_displacements, warp_displacements_tracks
import os

def create_displacements_T1(in_nii,affine_mat,out_nii,nonlin_nii,invw_nii,disp_nii,dispa_nii):
    commands=[]    
    commands.append('flirt -in '+in_nii+' -ref '+'/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -out '+ out_nii+' -omat ' + affine_mat+' -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear') 
    for c in commands:
        print(c)
        pipe(c)
 



dirname = "PBC2009"

for root, dirs, files in os.walk(dirname):
    if root.endswith('brain1_fr_raw'):
        base_dir = root+'/MPRAGE_1_from_structual_T1Space/'        
        T1_filename = base_dir + 'fbrain1_mprage1_bet.nii.gz'   
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

        print "Finish creating displacements"# -*- coding: utf-8 -*-

