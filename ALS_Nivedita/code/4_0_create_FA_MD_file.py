# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:06:57 2012

@author: bao
"""

import os
import numpy as np
import nibabel as nib
from dipy.external.fsl import flirt2aff, warp_displacements
from dipy.reconst.dti import Tensor
from dipy.io.dpy import Dpy

patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
#patients = [2]
mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]     

if __name__ == '__main__':
    
    
   
    dir_name = 'data'
    #dir_name = "data_temp"
    num_seeds = 3  

#creating FA and MD
#actually FA and FA_warped is already created from the step of tractography
#in this step, we only create MD
 
    for p in np.arange(24)+1:        
            print p         
            subj = int(mapping[p])            
            base_dir = dir_name +'/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/'
                        
#            filename = 'raw'
#            base_filename = base_dir + filename        
#            nii_filename = base_filename + 'bet.nii.gz'
#            bvec_filename = base_filename + '.bvec'
#            bval_filename = base_filename + '.bval'
#            
#            #print nii_filename
#            
#            img = nib.load(nii_filename)
#            data = img.get_data()
#            affine = img.get_affine()
#    
#            bvals = np.loadtxt(bval_filename)
#            gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
#    
#            tensors = Tensor(data, bvals, gradients, thresh=50)
#            #FA = tensors.fa()
#            #print(FA.shape)
##------------------------ 
#            MD=tensors.md() 
#            #print(MD.shape)   
##      
##------------------------
#        #fa_filename = base_dir + 'DTI/fa.nii.gz'
#        #print "Saving FA:", fa_filename
#        #fa_img = nib.Nifti1Image(data=FA, affine=affine)
#        #nib.save(fa_img, fa_filename)
#        
##------------------------
#            md_filename = base_dir + 'DTI/md.nii.gz'
#            print 'Saving MD: \t', md_filename, '\t' , MD.shape
#            md_img = nib.Nifti1Image(data=MD, affine=affine)
#            nib.save(md_img, md_filename)
        
        
 #---------------------------------------------------------------------------
#warping FA and MD       
#        print "Doing nonlinear (flirt/fnirt) registration of FA volume."
#        basedir_recon = os.path.join(os.getcwd(), base_dir) + 'DTI/'        
#        fa_filename = basedir_recon + 'fa.nii.gz'
#        flirt_affine = basedir_recon + 'flirt.mat'
#        flirt_displacements = basedir_recon + 'displacements.nii.gz'
#        fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
#        #print "Reference:", fsl_ref
#        #stop
#        flirt_warp = basedir_recon + 'fa_warped.nii.gz'
#        nonlin_nii = basedir_recon + 'nonlinear.nii.gz'
#        invw_nii = basedir_recon + 'invw.nii.gz'
#        disp_nii = basedir_recon + 'disp.nii.gz'
#        dispa_nii = basedir_recon + 'dispa.nii.gz'
#        
#        print "Creating displacements...",
#        create_displacements(fa_filename, flirt_affine, nonlin_nii, invw_nii, disp_nii, dispa_nii)
#        print "Done creating displacements."
#
#        print "Warping FA...",
#        warp_displacements(fa_filename,
#                           flaff = flirt_affine,
#                           fdis = disp_nii,
#                           fref = fsl_ref,
#                           ffaw = flirt_warp)
#        print "Done warping FA."


# filetemp = base_dir + 'DTI/fa_warped.nii.gz'

#not need to create displacements, because it has already been created when we warpe FA            
            md_filename = base_dir + 'DTI/md.nii.gz'
            fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'           
                       

            flirt_affine = base_dir + 'DTI/flirt.mat'
            disp_nii = base_dir + 'DTI/disp.nii.gz'
            flirt_warp = base_dir + 'DTI/md_warped.nii.gz'         
        

            print "Warping MD...",
            warp_displacements(md_filename,
                               flaff = flirt_affine,
                               fdis = disp_nii,
                               fref = fsl_ref,
                               ffaw = flirt_warp)
            print "Done warping MD."

    
            
            filetemp = base_dir + 'DTI/md_warped.nii.gz'
            img = nib.load(filetemp)
            md = img.get_data()
            print filetemp, '\t',np.shape(md)
           