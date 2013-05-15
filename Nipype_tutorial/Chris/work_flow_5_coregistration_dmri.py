'''
Eddy Current Correction

We want to correct for eddy currents in out dwi sequence. For that, we need to coregister all our volumes to a reference (one of the volumes, artibray)

'''

'''
1. extract one volume from diffusion timeseries  ExtractROI
2. skullstrip it to create a mask BET
3. feed the mask and the diffusion timeseries (together with bval, bvec) into DTIFit
'''

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

def create_eddy_current_correction_workflow():

	#!fslroi /home/bao/tiensy/Nipype_tutorial/data/dmri/raw.nii.gz /home/bao/tiensy/Nipype_tutorial/data/dmri/raw_less.nii.gz 0 10

	wf = pe.Workflow(name = 'workflow_dti_eddy')
	wf.base_dir = 'trento_example5'


	#ExtractROI node
	roi = pe.Node(interface = fsl.ExtractROI(), name = 'roi')
	#roi.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz'
	roi.inputs.in_file ='/home/bao/tiensy/Nipype_tutorial/data/dmri/raw_less.nii.gz'
		
	roi.inputs.t_min=0 #default for the 
	roi.inputs.t_size=1
	roi.inputs.terminal_output = 'file'

	#BET node
	bet = pe.Node(interface = fsl.BET(), name = 'bet')
	roi.inputs.terminal_output = 'file'
	bet.inputs.mask = True#for saving the mask_file to use in the DTIFit model, in default the mask is not 	saved
	#bet.iterables = ('frac',[0.9,0.7,0.5])
	bet.inputs.frac = 0.9	

	wf.connect(roi,'roi_file', bet, 'in_file')
	
	'''
	eddy current corection
	'''
	#split the 4D dimension into a list of 3D dimension
	split = pe.Node(interface = fsl.Split(), name = 'split')
	split.inputs.dimension = 't'
	split.inputs.in_file = '/home/bao/tiensy/Nipype_tutorial/data/dmri/raw_less.nii.gz'


	#coregistration -
	flirt = pe.MapNode(interface = fsl.FLIRT(), 
        	           name = 'flirt',
        	           iterfield = ['in_file'])
	flirt.inputs.no_search = True
	flirt.inputs.padding_size = 1
	flirt.inputs.dof = 12
	flirt.inputs.interp = 'spline'

	wf.connect(roi,'roi_file',flirt,'reference')
	'''
	or can use the nipype.utility.Select() to select one volume for reference

	import nipype.utility as util
	pick_ref = pe.Node(interface=util.Select(),name='pick_ref')
	pick_ref.inputs.index = 0

	wf.connect(split,'out_files',pick_ref,'inlist')
	wf.connect(pick_ref,'out_file',flirt,'reference')
	'''

	wf. connect(split,'out_files',flirt,'in_file')

	#merge all 3D volumes into 4D image
	merge = pe.Node(interface = fsl.Merge(), name = 'merge')
	merge.inputs.dimension = 't'

	wf.connect(flirt,'out_file',merge,'in_files')

	'''
	#DTIFit node
	dti = pe.Node(interface = fsl.DTIFit(), name = 'dti_fit')
	dti.inputs.bvals = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/bvals'
	dti.inputs.bvecs = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/bvecs'

	#not roi_file of roi node, because roi_file is only the region in timeseries not all volume
	dti.inputs.dwi = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz' 

	# or the dwi is the brain after coregistration  
	# wf.connect(merge,'merged_file',dti,'dwi')

	dti.inputs.terminal_output = 'file'
	dti.inputs.save_tensor = True #for saving the tensor in 4D dimension

	wf.connect(bet,'mask_file', dti, 'mask')
	'''

	#wf.run(plugin='MuliProc') #running in linear
	return wf
