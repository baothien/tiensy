# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:39:00 2013

@author: bao
"""

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from workflows.dmri.fsl.dti import create_eddy_correct_pipeline

from classInterfaces import BrainExtraction, EddyCorrection, ResampleVoxelSize, TensorModel, Tracking
 
path = '/home/bao/tiensy/Nipype_tutorial/data/dmri/temp/'
data = path+ 'raw.nii.gz'
                                        
###### WORKFLOW DEFINITION #######
wf=pe.Workflow(name="reconstructing_tractography")
wf.base_dir= path + 'results'


###### NODE DEFINITION #######
brain_extraction_node = pe.Node(BrainExtraction(), name="brain_extraction_node")
#eddy_current_correction_node = pe.Node(EddyCorrection(), name="eddy_current_correction_node")
eddy_current_correction_node = create_eddy_correct_pipeline("nipype_eddycorrect_wkf")
resample_voxel_size_node = pe.Node(ResampleVoxelSize(), name='resample_voxel_size_node')
tensor_model_node = pe.Node(TensorModel(), name='tensor_model_node')
tracking_node = pe.Node(Tracking(), name='tracking_node')


###### INPUT NODE DEFINITION #######
#inputs: brain_extraction_node
brain_extraction_node.inputs.input_filename=data
brain_extraction_node.inputs.output_filename = path + 'data_bet.nii.gz'

#inputs: eddy_current_correction_node
eddy_current_correction_node.inputs.output_filename = path + 'data_bet_ecc.nii.gz'

#inputs: resample_voxel_size_node
resample_voxel_size_node.inputs.output_filename = path + 'data_bet_ecc_iso.nii.gz'


#inputs: tensor_model_node
tensor_model_node.inputs.input_filename_bvecs = path + 'raw.bvec'
tensor_model_node.inputs.input_filename_bvals = path + 'raw.bval'

tensor_model_node.inputs.output_filename_fa = path + 'tensor_fa.nii.gz'
tensor_model_node.inputs.output_filename_evecs = path + 'tensor_evecs.nii.gz'

#inputs: tracking_node
tracking_node.inputs.num_seeds = 1000
tracking_node.inputs.low_thresh = 0.2
tracking_node.inputs.output_filename = path + 'dti_tracks.dpy'

###### NODE CONNECTIONS #######
wf.connect(brain_extraction_node,'bet_file', eddy_current_correction_node, 'input_filename')
wf.connect(eddy_current_correction_node,'eddy_current_correction_file', resample_voxel_size_node ,'input_filename')
wf.connect(resample_voxel_size_node,'resample_file',tensor_model_node ,'input_filename_data')
wf.connect(tensor_model_node, 'tensor_fa_file', tracking_node,'input_filename_fa')
wf.connect(tensor_model_node, 'tensor_evecs_file', tracking_node , 'input_filename_evecs')


###### GRAPH and RUN #######
wf.write_graph()
wf.run()
