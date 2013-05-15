'''
1. extract one volume from diffusion timeseries  ExtractROI
2. skullstrip it to create a mask BET
3. feed the mask and the diffusion timeseries (together with bval, bvec) into DTIFit
'''
#from work_flow_6_eddy_current_correction import create_eddy_current_correction_workflow,temp
from work_flow_6_eddy_current_correction import temp as tmp
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

def X(z):
    return z + 2

wf = pe.Workflow(name = 'workflow_dti')
wf.base_dir = 'trento_temp'
a = tmp()
#----------------------------------------------------------------------------
#inputs:  <ten_workflow>.inputs.<ten_node>.<input cua node>
#ouputs: <ten_node>.<output cua node>
#----------------------------------------------------------------------------

a.inputs.T_node.x = 10
a.inputs.T_node.y = 20
b =  pe.Node(name="X",interface=Function(input_names=['z'],
                                output_names = ['final'],                          
                         function=X)) 

#wf.connect(a,'a.outputs.S_node.sumS',b,'z')
wf.connect(a,'S_node.sumS',b,'z')
wf.write_graph('temp_graph')
wf.run(plugin='Linear')

'''

#using workflow as a node
eddy = pe.Node(interface = create_eddy_current_correction_workflow(), name = 'eddy')
eddy.inputs.inputspec.dwi = '/home/bao/tiensy/Nipype_tutorial/data/dmri/raw_less.nii.gz'
eddy.inputs.inputspec.index_ref = 0

#ExtractROI node
roi = pe.Node(interface = fsl.ExtractROI(), name = 'roi')
#roi.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz'
roi.inputs.t_min=0 #default for the 
roi.inputs.t_size=1
roi.inputs.terminal_output = 'file'

#wf.connect(eddy,'merge.merged_file',roi,'in_file')
wf.connect(eddy,'eddy.outputs.eddy_correction_file',roi,'in_file')

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
#dti.inputs.dwi = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/diff/data.nii.gz' #not roi_file of roi node, because roi_file is only the region in timeseries not all volume
dti.inputs.terminal_output = 'file'
dti.inputs.save_tensor = True #for saving the tensor in 4D dimension

wf.connect(bet,'mask_file', dti, 'mask')
wf.connect(eddy,'eddy_current_correction',dti,'dwi')

wf.run(plugin='MultiProc') #running in multiple processing

'''