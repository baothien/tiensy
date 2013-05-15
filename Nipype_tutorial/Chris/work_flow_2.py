'''
1. extract one volume from diffusion timeseries  ExtractROI
2. skullstrip it to create a mask BET
3. feed the mask and the diffusion timeseries (together with bval, bvec) into DTIFit
'''

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

wf = pe.Workflow(name = 'workflow_dti')
wf.base_dir = 'trento_temp'


#ExtractROI node
roi = pe.Node(interface = fsl.ExtractROI(), name = 'roi')
roi.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz'
roi.inputs.t_min=0 #default for the 
roi.inputs.t_size=1
roi.inputs.terminal_output = 'file'

#BET node
bet = pe.Node(interface = fsl.BET(), name = 'bet')
roi.inputs.terminal_output = 'file'
bet.inputs.mask = True#for saving the mask_file to use in the DTIFit model, in default the mask is not saved
bet.iterables = ('frac',[0.9,0.7,0.5])

wf.connect(roi,'roi_file', bet, 'in_file')

#DTIFit node
dti = pe.Node(interface = fsl.DTIFit(), name = 'dti_fit')
dti.inputs.bvals = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/bvals'
dti.inputs.bvecs = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/bvecs'
dti.inputs.dwi = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz' #not roi_file of roi node, because roi_file is only the region in timeseries not all volume
dti.inputs.terminal_output = 'file'
dti.inputs.save_tensor = True #for saving the tensor in 4D dimension

wf.connect(bet,'mask_file', dti, 'mask')

#wf.run(plugin='Linear') #running in linear
wf.run(plugin='MultiProc') #running in multiple processing

