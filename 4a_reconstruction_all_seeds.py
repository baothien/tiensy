# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:56:54 2012

@author: bao
"""


import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
#from dipy.reconst.gqi import GeneralizedQSampling
from dipy.io.dpy import Dpy
from dipy.external.fsl import flirt2aff,create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
#this is new in dipy 0.6.0.dev
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX


def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
        
def create_save_tracks(anisotropy,indices, seeds, low_thresh,filename):
    #this is new features in new dipy -current 121011 0.6.0.dev
    #print "Computing EuDX reconstruction."
    euler = EuDX(anisotropy, 
                        ind=indices, 
                        odf_vertices=get_sphere('symmetric362').vertices, 
                        seeds=seeds, a_low=low_thresh)
  
    
    euler = EuDX(anisotropy, ind=indices, seeds=seeds, a_low=low_thresh)
    tracks = [track for track in euler]     
    dpw = Dpy(filename, 'w')
    dpw.write_tracks(tracks)
    dpw.close()
    
def warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename):
    fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    
    img_fa = nib.load(fa_filename)            

    flirt_affine= np.loadtxt(flirt_filename)    
        
    img_ref =nib.load(fsl_ref)
    
    #create affine matrix from flirt     
    mat=flirt2aff(flirt_affine,img_fa,img_ref)        

     #read tracks    
    dpr = Dpy(tracks_filename, 'r')
    tensor_tracks = dpr.read_tracks()
    dpr.close()
        
    #linear tranform for tractography
    tracks_warped_linear = transform_tracks(tensor_tracks,mat)        

    #save tracks_warped_linear    
    dpr_linear = Dpy(linear_filename, 'w')
    dpr_linear.write_tracks(tracks_warped_linear)
    dpr_linear.close()         
    
visualize = False
#dirname = "data_temp"
dirname = "data"
for root, dirs, files in os.walk(dirname):
    if root.endswith('DTI64_1'):
        base_dir = root+'/' 
        filename = 'DTI64'
        base_filename = base_dir + filename        

        nii_filename = base_filename + '_bet.nii.gz'
        bvec_filename = base_filename + '.bvec'
        bval_filename = base_filename + '.bval'        

        img = nib.load(nii_filename)
        data = img.get_data()
        affine = img.get_affine()

        bvals = np.loadtxt(bval_filename)
        gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient

        base_dir2 = base_dir+ 'DTI/'   
        
        tensors = Tensor(data, bvals, gradients, thresh=50)
        create_save_tracks(tensors.fa(), tensors.ind(), 10**4, .2, base_dir2+'tracks_dti_10K_new.dpy')
      #  create_save_tracks(tensors.fa(), tensors.ind(), 10**6, .2, base_dir2+'tracks_dti_1M.dpy')
      #  create_save_tracks(tensors.fa(), tensors.ind(), 3*10**6, .2, base_dir2+'tracks_dti_3M.dpy')
#        
#        gqs=GeneralizedQSampling(data,bvals,gradients)
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**4, .0239, base_dir2+'tracks_gqi_10K.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**6, .0239, base_dir2+'tracks_gqi_1M.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 3*10**6, .0239, base_dir2+'tracks_gqi_3M.dpy')
#        
             
        flirt_filename =base_dir2+'flirt.mat'
        fa_filename= base_dir2 + 'fa.nii.gz' 
        
        tracks_filename =  base_dir2+'tracks_dti_10K_new.dpy'
        linear_filename =  base_dir2+'tracks_dti_10K_linear_new.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

#        tracks_filename =  base_dir2+'tracks_dti_1M.dpy'
#        linear_filename =  base_dir2+'tracks_dti_1M_linear.dpy'        
#        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)
#
#        tracks_filename =  base_dir2+'tracks_dti_3M.dpy'
#        linear_filename =  base_dir2+'tracks_dti_3M_linear.dpy'        
#        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

#        
#        tracks_filename =  base_dir2+'tracks_gqi_10K.dpy'
#        linear_filename =  base_dir2+'tracks_gqi_10K_linear.dpy'        
#        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)
#        
#        tracks_filename =  base_dir2+'tracks_gqi_1M.dpy'
#        linear_filename =  base_dir2+'tracks_gqi_1M_linear.dpy'        
#        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)
#        
#        tracks_filename =  base_dir2+'tracks_gqi_3M.dpy'
#        linear_filename =  base_dir2+'tracks_gqi_3M_linear.dpy'        
#        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        print('Done')
        print(linear_filename)
        #break
