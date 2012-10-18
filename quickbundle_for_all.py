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
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import flirt2aff,create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
from dipy.tracking.metrics import downsample

#==============quick bundles ===================================================

#import os
#import numpy as np
#import nibabel as nib
#from dipy.io.dpy import Dpy
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_sphere
    

tracks_filename_arr=['tracks_dti_10K_linear.dpy']#,'tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy']
#,
#                 'tracks_gqi_10K_linear.dpy','tracks_gqi_1M_linear.dpy','tracks_gqi_3M_linear.dpy']

qb_filename_15=['qb_dti_10K_linear_15.pkl']#,'qb_dti_1M_linear_15.pkl','qb_dti_3M_linear_15.pkl']
#,
#                 'qb_gqi_10K_linear_15.pkl','qb_gqi_1M_linear_15.pkl','qb_gqi_3M_linear_15.pkl']

qb_filename_20=['qb_dti_10K_linear_20.pkl']#,'qb_dti_1M_linear_20.pkl','qb_dti_3M_linear_20.pkl']
#,
#                 'qb_gqi_10K_linear_20.pkl','qb_gqi_1M_linear_20.pkl','qb_gqi_3M_linear_20.pkl']

qb_filename_30=['qb_dti_10K_linear_30.pkl']#,'qb_dti_1M_linear_30.pkl','qb_dti_3M_linear_30.pkl']
#,
 #                'qb_gqi_10K_linear_30.pkl','qb_gqi_1M_linear_30.pkl','qb_gqi_3M_linear_30.pkl']

def load_tracks(filename):
    dpr = Dpy(filename, 'r')
    tracks=dpr.read_tracks()
    dpr.close()
    return tracks
#=====================quick bundles ============================================

def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
        
def create_save_tracks(anisotropy,indices, seeds, low_thresh,filename):
    
    euler = EuDX(anisotropy, 
                        ind=indices, 
                        #odf_vertices=get_sphere('symmetric362'),
                        seeds=seeds, a_low=low_thresh)
                        #odf_vertices=get_sphere('symmetric362').vertices, 
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
        create_save_tracks(tensors.fa(), tensors.ind(), 10**4, .2, base_dir2+'tracks_dti_10K.dpy')
        #create_save_tracks(tensors.fa(), tensors.ind(), 10**6, .2, base_dir2+'tracks_dti_1M.dpy')
        #create_save_tracks(tensors.fa(), tensors.ind(), 3*10**6, .2, base_dir2+'tracks_dti_3M.dpy')

#=====================  gqi ========================================        
#        gqs=GeneralizedQSampling(data,bvals,gradients)
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**4, .0239, base_dir2+'tracks_gqi_10K.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**6, .0239, base_dir2+'tracks_gqi_1M.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 3*10**6, .0239, base_dir2+'tracks_gqi_3M.dpy')
#=====================  gqi ========================================                
             
        flirt_filename =base_dir2+'flirt.mat'
        fa_filename= base_dir2 + 'fa.nii.gz' 
        
        tracks_filename =  base_dir2+'tracks_dti_10K.dpy'
        linear_filename =  base_dir2+'tracks_dti_10K_linear.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        #tracks_filename =  base_dir2+'tracks_dti_1M.dpy'
        #linear_filename =  base_dir2+'tracks_dti_1M_linear.dpy'        
        #warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        #tracks_filename =  base_dir2+'tracks_dti_3M.dpy'
        #linear_filename =  base_dir2+'tracks_dti_3M_linear.dpy'        
        #warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)


##=====================  gqi ========================================                
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
##=====================  gqi ========================================        



#==============quick bundles ==================================
#dirname = "data_temp"
#dirname = "data_qb"
#for root, dirs, files in os.walk(dirname):
#    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'): 
#        base_dir = root+'/' 
#        filename = 'raw'
#        base_dir2 = base_dir+ 'DTI/'  

        print base_dir+'../MPRAGE_1/T1_flirt_out.nii.gz'           
        img = nib.load(base_dir+'../MPRAGE_1/T1_flirt_out.nii.gz')
        data = img.get_data()
       
        for i in range(len(tracks_filename_arr)):
            print '>>>>'            
            print base_dir2+tracks_filename_arr[i]
            tracks=load_tracks(base_dir2+tracks_filename_arr[i]) 
            print len(tracks)
            #tracks = [downsample(t, 12) - np.array(data.shape[:3])/2. for t in tracks]
            tracks = [downsample(t, 12) - np.array(data.shape)/2. for t in tracks]
            #shift in the center of the volume            
            #tracks=[t-np.array(data.shape)/2. for t in tracks]
            #1/0
            print base_dir2+qb_filename_15[i]            
            qb=QuickBundles(tracks,15.,12)
            save_pickle(base_dir2+qb_filename_15[i],qb)
            
            print base_dir2+qb_filename_20[i]
            qb=QuickBundles(tracks,20.,12)
            save_pickle(base_dir2+qb_filename_20[i],qb)
            
            #print base_dir2+qb_filename_30[i]
            #qb=QuickBundles(tracks,30.,12)            
            #save_pickle(base_dir2+qb_filename_30[i],qb)            
#================ quick bundles ==============================================

        print('Done')
        print(linear_filename)
        #stop
