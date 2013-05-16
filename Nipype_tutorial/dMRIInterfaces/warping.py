# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:57:15 2013

@author: bao
warping from native space into MNI-native space
"""

import os
import numpy as np
import nibabel as nib
from dipy.external.fsl import flirt2aff,create_displacements, warp_displacements
from dipy.io.dpy import Dpy


'''
warping the anatomy
'''
def create_displacements_T1(in_nii,ref, affine_mat,out_nii,nonlin_nii,invw_nii,disp_nii,dispa_nii):
    commands=[]    
    #commands.append('flirt -in '+in_nii+' -ref '+'/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -out '+ out_nii+' -omat ' + affine_mat+' -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear') 
    commands.append('flirt -in '+in_nii+' -ref '+ ref +' -out '+ out_nii+' -omat ' + affine_mat+' -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear')     
    for c in commands:
        print(c)
        pipe(c)
        
def warp_anatomy(input_filename, output_filename=None, output_fmatrix=None, input_ref='/usr/share/fsl/data/standard/MNI152lin_T1_1mm_brain.nii.gz'):     
    
    print 'Warping anatomy of ', input_filename, '...'    
          
    nonlin_nii = 'nonlinear.nii.gz'
    invw_nii = 'invw.nii.gz'
    disp_nii = 'disp.nii.gz'
    dispa_nii = 'dispa.nii.gz'
    
    if output_filename == None:
        filename_save = input_filename.split('.')[0]+'_flirt_out.nii.gz'
    else:
        filename_save = os.path.abspath(output_filename)
        
    if output_fmatrix == None:
        filename_save_matrix = input_filename.split('.')[0]+'_flirt.mat'
    else:
        filename_save_matrix = os.path.abspath(output_fmatrix)
        
    create_displacements_T1(input_filename, input_ref, filename_save_matrix, filename_save, nonlin_nii,invw_nii, disp_nii, dispa_nii)
    
    print 'Saving to ', filename_save, ' and ', filename_save_matrix 
    
    return filename_save, filename_save_matrix

    
def warp_fa(input_filename, output_filename=None, output_fmatrix=None, input_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'):
   
    print "Doing nonlinear (flirt/fnirt) registration of FA volume..."
    
    if output_filename == None:
        filename_save = input_filename.split('.')[0]+'_flirt_out.nii.gz'
    else:
        filename_save = os.path.abspath(output_filename)

    if output_fmatrix == None:
        filename_save_matrix = input_filename.split('.')[0]+'_flirt.mat'
    else:
        filename_save_matrix = os.path.abspath(output_fmatrix)    
    
    nonlin_nii = 'nonlinear.nii.gz'
    invw_nii = 'invw.nii.gz'
    disp_nii = 'disp.nii.gz'
    dispa_nii = 'dispa.nii.gz'

    print "Creating displacements...",
    create_displacements(input_filename, filename_save_matrix, nonlin_nii, invw_nii, disp_nii, dispa_nii,input_ref)
    print "Done."

    print "Warping FA...",
    warp_displacements(input_filename,
                       flaff = filename_save_matrix,
                       fdis = disp_nii,
                       fref = input_ref,
                       ffaw = filename_save)
                       
    print 'Saving to ', filename_save, ' and ', filename_save_matrix 
    
    return filename_save, filename_save_matrix
    
def load_tracks(input_filename):
    dpr = Dpy(input_filename, 'r')
    tracks=dpr.read_tracks()
    dpr.close()
    return tracks

def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
    
def warp_tracks(input_tracks_filename, input_flirt_fmatrix, input_fa_filename,  output_filename = None, input_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'):

    print 'Loading fa, flirt matrix ...'    
    img_fa = nib.load(input_fa_filename)            

    flirt_affine= np.loadtxt(input_flirt_fmatrix)    
        
    img_ref =nib.load(input_ref)
    
    #create affine matrix from flirt     
    mat=flirt2aff(flirt_affine,img_fa,img_ref)        

    #read tracks    
    print 'Loading tracks ...'
    tensor_tracks = load_tracks(input_tracks_filename)
        
    #linear tranform for tractography
    tracks_warped_linear = transform_tracks(tensor_tracks,mat)        

    if output_filename == None:
        filename_save = input_tracks_filename.split('.')[0]+'_linear.dpy'
    else:
        filename_save = os.path.abspath(output_filename)
        
    #save tracks_warped_linear    
    print 'Saving warped tracks into :', filename_save
    dpr_linear = Dpy(filename_save, 'w')
    dpr_linear.write_tracks(tracks_warped_linear)
    dpr_linear.close()
    
    return filename_save