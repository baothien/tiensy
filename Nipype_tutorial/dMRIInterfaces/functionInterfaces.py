# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:29:16 2013

@author: bao
"""

import nipype.interfaces.utility as util

from preprocessing import brain_extraction, eddy_correction, resample_voxel_size
from tracking import tensor_model, tracking
#from warping import warp_anatomy, warp_fa, warp_tracks

###### INTERFACE DEFINITION #######
# preprocessing
BrainExtraction = util.Function(input_names=['input_filename', 'output_filename'], 
                                             output_names=['bet_file'],
                                             function=brain_extraction, 
                                             imports=['from dipy.external.fsl import bet',                                                    
                                                      'import os'])
                                                      
EddyCorrection = util.Function(input_names=['input_filename', 'output_filename'], 
                                             output_names=['eddy_current_correction_file'],
                                             function=eddy_correction, 
                                             imports=['from dipy.external.fsl import eddy_correct',                                                    
                                                      'import os'])

ResampleVoxelSize = util.Function(input_names=['input_filename', 'output_filename'], 
                                             output_names=['resample_file'],
                                             function=resample_voxel_size, 
                                             imports=['from dipy.align.aniso2iso import resample',
                                                      'import nibabel as nib',
                                                      'import os'])
     
#tracking
TensorModel = util.Function(input_names=['input_filename_data', 'input_filename_bvecs', 'input_filename_bvals', 'output_filename_fa', 'output_filename_evecs'], 
                                             output_names=['tensor_fa_file','tensor_evecs_file'],
                                             function=tensor_model, 
                                             imports=['import nibabel as nib',
                                                      'import numpy as np',
                                                      'from dipy.io.gradients import read_bvals_bvecs',
                                                      'from dipy.core.gradients import gradient_table',
                                                      'from dipy.reconst.dti import TensorModel',
                                                      'from dipy.reconst.dti import fractional_anisotropy',
                                                      'import os'])                                                
                                                      
Tracking = util.Function(input_names=['input_filename_fa', 'input_filename_evecs', 'num_seeds', 'low_thresh', 'output_filename'], 
                                             output_names=['tracks_file'],
                                             function=tracking, 
                                             imports=['import nibabel as nib',
                                                      'import numpy as np',
                                                      'from dipy.io.gradients import read_bvals_bvecs',
                                                      'from dipy.core.gradients import gradient_table',
                                                      'from dipy.reconst.dti import TensorModel',
                                                      'from dipy.reconst.dti import fractional_anisotropy',
                                                      'from dipy.data import get_sphere',
                                                      'from dipy.reconst.dti import quantize_evecs',
                                                      'from dipy.tracking.eudx import EuDX', 
                                                      'from dipy.io.dpy import Dpy',
                                                      'import os'])                                                
 