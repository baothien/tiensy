# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:15:37 2013

@author: bao
"""
from preprocessing import *
from tracking import *
from warping import *
import nibabel as nib

data = '/home/bao/tiensy/Nipype_tutorial/data/dmri/raw_less.nii.gz'
path = '/home/bao/tiensy/Nipype_tutorial/data/dmri/temp4/'

#brain_extraction(data,path+'raw_bet.nii.gz')

#eddy_correction(path+'raw_bet.nii.gz')

#resample_voxel_size(path+'raw_bet_ecc.nii.gz')

#tensor_model(path+'raw_bet_ecc_iso.nii.gz',path+'raw.bvec',path+'raw.bval',path+'tensor_fa.nii.gz',path+'tensor_evecs.nii.gz')

#tracking(path+'tensor_fa.nii.gz',path+'tensor_evecs.nii.gz',output_filename = path+'tracks_dti.dpy')

#warp_fa(path+'tensor_fa.nii.gz', path+'tensor_fa_warped.nii.gz',path+'flirt.mat')

warp_tracks(path+'tracks_dti.dpy',path+'flirt.mat',path+'tensor_fa.nii.gz',path+'tracks_dti_warped.dpy')

'''
all function are tested successfully

'''