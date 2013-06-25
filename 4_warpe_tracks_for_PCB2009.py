# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:34:20 2012

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:58:05 2012

@author: bao
"""

import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
#from dipy.reconst.gqi import GeneralizedQSampling
from dipy.tracking.eudx import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import flirt2aff,create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
from dipy.tracking.metrics import downsample
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
#==============quick bundles ===================================================

from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_sphere
    

tracks_filename_arr=['tracks_dti_linear.dpy']

qb_filename_15=['qb_dti_linear_15.pkl']

qb_filename_20=['qb_dti_linear_20.pkl']

qb_filename_30=['qb_dti_linear_30.pkl']

def load_tracks(filename):
    dpr = Dpy(filename, 'r')
    tracks=dpr.read_tracks()
    dpr.close()
    return tracks
#=====================quick bundles ============================================

def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
        
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
     
    
dirname = "PBC2009/brain0_origintrack_newfa"

for root, dirs, files in os.walk(dirname):
    if root.endswith('DTI'):
        base_dir = root+'/' 
           
        #fa_filename = base_dir + 'dsi_fa.nii' 
        #fa_filename = base_dir + 'fbrain_FA.nii.gz' 
        fa_filename = base_dir + 'fa_resample.nii.gz' 
                     
        '''
        print "Doing nonlinear (flirt/fnirt) registration of FA volume."
        flirt_affine = base_dir + 'flirt.mat'
        flirt_displacements = base_dir + 'displacements.nii.gz'
        fsl_ref = '/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz'
        print "Reference:", fsl_ref
        flirt_warp = base_dir + 'fa_warped.nii.gz'
        nonlin_nii = base_dir + 'nonlinear.nii.gz'
        invw_nii = base_dir + 'invw.nii.gz'
        disp_nii = base_dir + 'disp.nii.gz'
        dispa_nii = base_dir + 'dispa.nii.gz'

        
        
        print "Creating displacements...",
        create_displacements(fa_filename, flirt_affine, nonlin_nii, invw_nii, disp_nii, dispa_nii,fsl_ref)
        print "Done."

        print "Warping FA...",
        warp_displacements(fa_filename,
                           flaff = flirt_affine,
                           fdis = disp_nii,
                           fref = fsl_ref,
                           ffaw = flirt_warp)
        print "Done warping fa."
        '''
        print "Doing warping tracks"
#-------------------------------------------------------------------------------        
        flirt_filename =base_dir+'flirt.mat'
        fa_filename= base_dir + 'fa_resample.nii.gz' 
        
        tracks_filename =  base_dir+'tracks_dti.dpy'
        linear_filename =  base_dir+'tracks_dti_linear.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)
        
        print 'Done warped tracks'
#-------------------------------------------------------------------------------
        

'''
#==============quick bundles ==================================       
        for i in range(len(tracks_filename_arr)):
            print '>>>>'            
            print base_dir+tracks_filename_arr[i]
            tracks=load_tracks(base_dir+tracks_filename_arr[i]) 
            print len(tracks)    

            tracks = [downsample(t, 12) for t in tracks]
            print base_dir+qb_filename_15[i]            
            qb=QuickBundles(tracks,15.,12)
            save_pickle(base_dir+qb_filename_15[i],qb)
            
            print base_dir+qb_filename_20[i]
            qb=QuickBundles(tracks,20.,12)
            save_pickle(base_dir+qb_filename_20[i],qb)
            
            print base_dir+qb_filename_30[i]
            qb=QuickBundles(tracks,30.,12)            
            save_pickle(base_dir+qb_filename_30[i],qb)            
#================ quick bundles ==============================================

        print('Done')
        print(linear_filename)
        #stop
'''