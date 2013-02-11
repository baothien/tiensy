# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:41:48 2012

@author: bao
"""

import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk

    
visualize = False
#dirname = "120523"
dirname = "data"
for root, dirs, files in os.walk(dirname):
   if root.endswith('DIFF2DEPI_EKJ_64dirs_14'):
   # if files.endswith('rawbet.nii.gz'):
        base_dir = root+'/'
        filename = 'raw'
        base_filename = base_dir + filename        
        nii_filename = base_filename + 'bet.nii.gz'
        bvec_filename = base_filename + '.bvec'
        bval_filename = base_filename + '.bval'
        
        img = nib.load(nii_filename)
        data = img.get_data()
        affine = img.get_affine()

        bvals = np.loadtxt(bval_filename)
        gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient

        tensors = Tensor(data, bvals, gradients, thresh=50)
        FA = tensors.fa()
        print(FA.shape)
 
        print "Computing EuDX reconstruction."
        euler = EuDX(a=FA, ind=tensors.ind(), seeds=100000, a_low=.2)

        tensor_tracks = [track for track in euler]

        tt = np.array(tensor_tracks, dtype=np.object)

#        if visualize:
#            from dipy.viz import fvtk
#            renderer = fvtk.ren()
#            fvtk.add(renderer, fvtk.line(tensor_tracks, fvtk.red, opacity=1.0))
#            fvtk.show(renderer)
#
        fa_filename = base_dir + 'DTI/fa.nii.gz'

        print "Saving FA:", fa_filename
        fa_img = nib.Nifti1Image(data=FA, affine=affine)
        nib.save(fa_img, fa_filename)

        dpy_filename = base_dir + 'DTI/tracks_dti_10K.dpy'
        print "Saving tractography:", dpy_filename
        dpw = Dpy(dpy_filename, 'w')
        dpw.write_tracks(tensor_tracks)
        dpw.close()    
        
        print "Doing nonlinear (flirt/fnirt) registration of FA volume."
        basedir_recon = os.path.join(os.getcwd(), base_dir) + 'DTI/'        
        fa_filename = basedir_recon + 'fa.nii.gz'
        flirt_affine = basedir_recon + 'flirt.mat'
        flirt_displacements = basedir_recon + 'displacements.nii.gz'
        fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
        #print "Reference:", fsl_ref
        #stop
        flirt_warp = basedir_recon + 'fa_warped.nii.gz'
        nonlin_nii = basedir_recon + 'nonlinear.nii.gz'
        invw_nii = basedir_recon + 'invw.nii.gz'
        disp_nii = basedir_recon + 'disp.nii.gz'
        dispa_nii = basedir_recon + 'dispa.nii.gz'
        
        print "Creating displacements...",
        create_displacements(fa_filename, flirt_affine, nonlin_nii, invw_nii, disp_nii, dispa_nii)
        print "Done creating displacements."

        print "Warping FA...",
        warp_displacements(fa_filename,
                           flaff = flirt_affine,
                           fdis = disp_nii,
                           fref = fsl_ref,
                           ffaw = flirt_warp)
        print "Done warping FA."

        print "Warping tractography with FA's displacements..."
        fdpyw = basedir_recon + 'tracks_dti_10K_warped.dpy'

        warp_displacements_tracks(fdpy=dpy_filename, ffa=fa_filename ,fmat=flirt_affine ,finv=invw_nii, fdis=disp_nii,fdisa=dispa_nii, fref=fsl_ref, fdpyw=fdpyw)
        print "Done."

        dpr = Dpy(fdpyw, 'r')
        tensor_tracks_warped = dpr.read_tracks()
        dpr.close()
    
#        visualize = False
        if visualize:
            from dipy.viz import fvtk
            renderer = fvtk.ren()
            fvtk.add(renderer, fvtk.line(tensor_tracks, fvtk.red, opacity=1.0))
            fvtk.add(renderer, fvtk.line(tensor_tracks_warped, fvtk.green, opacity=1.0))
            fvtk.show(renderer)
