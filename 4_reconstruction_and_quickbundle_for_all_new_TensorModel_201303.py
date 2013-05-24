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
    

tracks_filename_arr=['tracks_dti_10K_linear.dpy','tracks_dti_1M_linear.dpy','tracks_dti_3M_linear.dpy']
#,
#                 'tracks_gqi_10K_linear.dpy','tracks_gqi_1M_linear.dpy','tracks_gqi_3M_linear.dpy']

qb_filename_15=['qb_dti_10K_linear_15.pkl','qb_dti_1M_linear_15.pkl','qb_dti_3M_linear_15.pkl']
#,
#                 'qb_gqi_10K_linear_15.pkl','qb_gqi_1M_linear_15.pkl','qb_gqi_3M_linear_15.pkl']

qb_filename_20=['qb_dti_10K_linear_20.pkl','qb_dti_1M_linear_20.pkl','qb_dti_3M_linear_20.pkl']
#,
#                 'qb_gqi_10K_linear_20.pkl','qb_gqi_1M_linear_20.pkl','qb_gqi_3M_linear_20.pkl']

qb_filename_30=['qb_dti_10K_linear_30.pkl','qb_dti_1M_linear_30.pkl','qb_dti_3M_linear_30.pkl']
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
        
def create_save_tracks(anisotropy,indices, vertices,seeds, low_thresh,filename):
    
    from dipy.tracking.eudx import EuDX

    eu = EuDX(FA, peak_indices, odf_vertices = vertices,seeds = seeds, a_low=low_thresh)
    tensor_tracks_old = [streamline for streamline in eu]
        
    #euler = EuDX(anisotropy, 
    #                    ind=indices, 
    #                    odf_vertices=get_sphere('symmetric362').vertices, 
    #                    seeds=seeds, a_low=low_thresh)                        
    #tensor_tracks_old = [track for track in euler]  
    
    #print len(tensor_tracks_old)
    tracks = [track for track in tensor_tracks_old if track.shape[0]>1]
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
     
    
#dirname = "Data/Nifti/Sub1"
dirname = "Data/Nifti"
for root, dirs, files in os.walk(dirname):
    #if root.endswith('DIFF'):
   # print root
    if root.endswith('DIFF'):
        base_dir = root+'/' 
        filename = 'raw'
        base_filename = base_dir + filename        

        nii_filename = base_dir + 'DTI/data_resample.nii.gz'
        #nii_filename = base_dir + 'raw.nii.gz'# raw_bet_ecc.nii.gz'
        bvec_filename = base_filename + '.bvec'
        bval_filename = base_filename + '.bval'        

        img = nib.load(nii_filename)
        data = img.get_data()
        affine = img.get_affine()
        print affine
        print data.shape

        #bvals = np.loadtxt(bval_filename)
        #gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
        #print bvals
        #print gradients        

#-----------------------------------------------------------
#   new dipy 20130319 -- change 

        bvals, bvecs = read_bvals_bvecs(bval_filename, bvec_filename)
        gtab = gradient_table(bvals, bvecs)

        zooms = img.get_header().get_zooms()[:3]
        new_zooms = (2., 2., 2.)        
        from dipy.align.aniso2iso import resample
        data2, affine2 = resample(data, affine, zooms, new_zooms)
        
        mask = data2[..., 0] > 50
        from dipy.reconst.dti import TensorModel
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(data2, mask)
        from dipy.reconst.dti import fractional_anisotropy
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        fa_img = nib.Nifti1Image(FA, img.get_affine())
        nib.save(fa_img, base_dir+'DTI/tensor_fa.nii.gz')
        evecs_img = nib.Nifti1Image(tenfit.evecs, img.get_affine())
        nib.save(evecs_img, base_dir+'DTI/tensor_evecs.nii.gz')
       
        fa_img = nib.load(base_dir+'DTI/tensor_fa.nii.gz')
        FA = fa_img.get_data()
        evecs_img = nib.load(base_dir+'DTI/tensor_evecs.nii.gz')
        evecs = evecs_img.get_data()
        FA[np.isnan(FA)] = 0
        
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')

        from dipy.reconst.dti import quantize_evecs
        peak_indices = quantize_evecs(evecs, sphere.vertices)
        #from dipy.tracking.eudx import EuDX
        #eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2)
        #tensor_streamlines = [streamline for streamline in eu]
        
#---------------------------------------------------------    
        base_dir2 = base_dir+ 'DTI/'           
        create_save_tracks(FA, peak_indices,sphere.vertices, 10**4, .2, base_dir2+'tracks_dti_10K.dpy')        
        create_save_tracks(FA, peak_indices,sphere.vertices, 10**6, .2, base_dir2+'tracks_dti_1M.dpy')        
        create_save_tracks(FA, peak_indices,sphere.vertices, 3*10**6, .2, base_dir2+'tracks_dti_3M.dpy')        
           
        
        
#---------------------------------------------------------------------------------- 
#  old version of Tensor       
#        tensors = Tensor(data, bvals, gradients, thresh=50)        
#        create_save_tracks(tensors.fa(), tensors.ind(), 10**4, .2, base_dir2+'tracks_dti_10K.dpy')        
#        create_save_tracks(tensors.fa(), tensors.ind(), 10**6, .2, base_dir2+'tracks_dti_1M.dpy')
#        create_save_tracks(tensors.fa(), tensors.ind(), 3*10**6, .2, base_dir2+'tracks_dti_3M.dpy')
#-----------------------------------------------------------------------------------       
    #=====================  gqi ========================================        
#        gqs=GeneralizedQSampling(data,bvals,gradients)
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**4, .0239, base_dir2+'tracks_gqi_10K.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 10**6, .0239, base_dir2+'tracks_gqi_1M.dpy')
#        create_save_tracks(gqs.qa(), gqs.ind(), 3*10**6, .0239, base_dir2+'tracks_gqi_3M.dpy')
    #=====================  gqi ========================================                
            
        print 'Done creating tractography'        
        
#-------------------------------------------------------------------------------        
        flirt_filename =base_dir2+'flirt.mat'
        fa_filename= base_dir2 + 'fa_resample.nii.gz' 
        
        tracks_filename =  base_dir2+'tracks_dti_10K.dpy'
        linear_filename =  base_dir2+'tracks_dti_10K_linear.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        tracks_filename =  base_dir2+'tracks_dti_1M.dpy'
        linear_filename =  base_dir2+'tracks_dti_1M_linear.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        tracks_filename =  base_dir2+'tracks_dti_3M.dpy'
        linear_filename =  base_dir2+'tracks_dti_3M_linear.dpy'        
        warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)

        
        #=====================  gqi ========================================                
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
        #=====================  gqi ========================================      
        
        print 'Done warped'
#-------------------------------------------------------------------------------



#==============quick bundles ==================================
#dirname = "data_temp"
#dirname = "data_qb"
#for root, dirs, files in os.walk(dirname):
#    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'): 
#        base_dir = root+'/' 
#        filename = 'raw'
#        base_dir2 = base_dir+ 'DTI/'  

        #print base_dir+'../ANATOMY/T1_flirt_out.nii.gz'           
        #img = nib.load(base_dir+'../ANATOMY/T1_flirt_out.nii.gz')
        #data = img.get_data()
       
        for i in range(len(tracks_filename_arr)):
            print '>>>>'            
            print base_dir2+tracks_filename_arr[i]
            tracks=load_tracks(base_dir2+tracks_filename_arr[i]) 
            print len(tracks)
            
            #shift in the center of the volume            
            #tracks=[t-np.array(data.shape)/2. for t in tracks]

            tracks = [downsample(t, 12) for t in tracks]
            print base_dir2+qb_filename_15[i]            
            qb=QuickBundles(tracks,15.,12)
            save_pickle(base_dir2+qb_filename_15[i],qb)
            
            print base_dir2+qb_filename_20[i]
            qb=QuickBundles(tracks,20.,12)
            save_pickle(base_dir2+qb_filename_20[i],qb)
            
            print base_dir2+qb_filename_30[i]
            qb=QuickBundles(tracks,30.,12)            
            save_pickle(base_dir2+qb_filename_30[i],qb)            
#================ quick bundles ==============================================

        print('Done')
        print(linear_filename)
        #stop
