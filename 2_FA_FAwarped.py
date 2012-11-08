# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:10:29 2012

@author: bao
"""
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.tracking.eudx import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import create_displacements, warp_displacements
from dipy.data import get_sphere
import os



#dirname = "data/1891215"
dirname = "ADHD"
b0_threshold = 50
#eudx_seeds = 1000
#eudx_fa_stop = 0.2
fsl_ref = '/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz'

for root, dirs, files in os.walk(dirname):    
    if root.endswith('DTI64_1'):
        
        base_dir = root+'/'            
        
        nii_filename = base_dir + 'DTI64.nii.gz'
        bval_filename = base_dir + 'DTI64.bval'
        bvec_filename = base_dir + 'DTI64.bvec'
        
        nii_bet_ecc_filename = base_dir + 'raw_bet_ecc.nii.gz'
        
        basedir_reconstruction = os.path.join(root + '/DTI/')
        print basedir_reconstruction
        # Silently build output directory if not present:
        try:
            os.makedirs(basedir_reconstruction)
        except OSError:
            pass


        bvals = np.loadtxt(bval_filename)
        gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
        assert(bvals.size==gradients.shape[0])
        print("Loading bvals and bvec: %s" % bvals.size)        
        
        print("Loading nifti data: %s" % nii_bet_ecc_filename)
        img = nib.load(nii_bet_ecc_filename)

        old_data = img.get_data()
        old_affine = img.get_affine()
                
        
        from dipy.align.aniso2iso import resample
        zooms=img.get_header().get_zooms()[:3]
        print 'old zooms:', zooms
        new_zooms=(2.,2.,2.)
        print 'new zoom', new_zooms        
        data,affine=resample(old_data,old_affine,zooms,new_zooms)
        
        
        print("Computing FA...")
        tensors = Tensor(data, bvals, gradients, thresh=b0_threshold)
        FA = tensors.fa()
        print("FA:", FA.shape)

#        print("Computing EuDX reconstruction.")
#        euler = EuDX(a=FA, ind=tensors.ind(), 
#                     odf_vertices=get_sphere('symmetric362').vertices,                      
#                     seeds=eudx_seeds, a_low=eudx_fa_stop)
#        tensor_tracks_old = [track for track in euler]   
#        tensor_tracks = [track for track in tensor_tracks_old if track.shape[0]>1]

        print("Saving results.")

        fa_filename = basedir_reconstruction + 'fa_resample.nii.gz'
        print "Saving FA:", fa_filename
        fa_img = nib.Nifti1Image(data=FA, affine=affine)
        nib.save(fa_img, fa_filename)
        
        data_filename = basedir_reconstruction + 'data_resample.nii.gz'
        print "Saving Data after resapling:", data_filename
        data_img = nib.Nifti1Image(data=data, affine=affine)
        nib.save(data_img, data_filename)
        
#        dpy_filename = basedir_reconstruction + 'tracks.dpy'
#        print "Saving tractography:", dpy_filename
#        dpw = Dpy(dpy_filename, 'w')
#        dpw.write_tracks(tensor_tracks)
#        dpw.close()

        print "Doing nonlinear (flirt/fnirt) registration of FA volume."
        flirt_affine = basedir_reconstruction + 'flirt.mat'
        flirt_displacements = basedir_reconstruction + 'displacements.nii.gz'
        print "Reference:", fsl_ref
        flirt_warp = basedir_reconstruction + 'fa_resample_warped.nii.gz'
        nonlin_nii = basedir_reconstruction + 'nonlinear.nii.gz'
        invw_nii = basedir_reconstruction + 'invw.nii.gz'
        disp_nii = basedir_reconstruction + 'disp.nii.gz'
        dispa_nii = basedir_reconstruction + 'dispa.nii.gz'

        print "Creating displacements...",
        create_displacements(fa_filename, flirt_affine, nonlin_nii, invw_nii, disp_nii, dispa_nii)
        print "Done."

        print "Warping FA...",
        warp_displacements(fa_filename,
                           flaff = flirt_affine,
                           fdis = disp_nii,
                           fref = fsl_ref,
                           ffaw = flirt_warp)
        print "Done."
