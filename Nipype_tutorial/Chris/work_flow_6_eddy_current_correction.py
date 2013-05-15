'''
Eddy Current Correction

We want to correct for eddy currents in out dwi sequence. 
For that, we need to coregister all our volumes to a reference (one of the volumes, artibray)

'''

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from nipype. interfaces.utility import IdentityInterface	
from nipype.interfaces.utility import Function

def Txy(x,y):
    return x + y
    
def Sxy(x):
    return 2*x
    

def temp():
    w = pe.Workflow(name = 'wf_temp')
    T_node = pe.Node(name="T_node", 
                        interface=Function(input_names=['x','y'],
                        output_names=['sumT'],                          
                        function=Txy))    
    S_node = pe.Node(name="S_node", 
                        interface=Function(input_names=['x'],
                        output_names=['sumS'],                          
                        function=Sxy))
    w.connect(T_node,'sumT',S_node,'x')
    return w
    
def create_eddy_current_correction_workflow(name='eddy_current_correction'):

	wf = pe.Workflow(name = name)
	
	#for input
	inputspec = pe.Node(interface=IdentityInterface(fields=['dwi','ref_index']), name = 'inputspec')
	
	'''
	eddy current corection
	'''
	#split the 4D dimension into a list of 3D dimension
	split = pe.Node(interface = fsl.Split(), name = 'split')
	split.inputs.dimension = 't'
	
	pick_ref = pe.Node(interface=util.Select(),name='pick_ref')
	
	#coregistration -
	flirt = pe.MapNode(interface = fsl.FLIRT(), 
        	           name = 'flirt',
        	           iterfield = ['in_file'])
	flirt.inputs.no_search = True
	flirt.inputs.padding_size = 1
	flirt.inputs.dof = 12
	flirt.inputs.interp = 'spline'

	#merge all 3D volumes into 4D image
	merge = pe.Node(interface = fsl.Merge(), name = 'merge')
	merge.inputs.dimension = 't'
	
	wf.connect(inputspec,'dwi',split,'in_file')
	wf.connect(inputspec,'ref_index',pick_ref,'index')
	wf.connect(split,'out_files',pick_ref,'inlist')
	wf. connect(split,'out_files',flirt,'in_file')	

	wf.connect(pick_ref,'out',flirt,'reference')
	wf.connect(flirt,'out_file',merge,'in_files')

	#for output
	outputspec = pe.Node(interface=IdentityInterface(fields=['eddy_correct_file']), name = 'outputspec')
	wf.connect(merge,'merged_file',outputspec,'eddy_correct_file')
	return wf
