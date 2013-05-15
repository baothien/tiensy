'''
The task is to smooth three outputs of TSNR node (mean, stddev, and tsnr)

- To do this, we have to use MapNode

- But MapNode always take lishs - therefore we have to merge the three outputs of the TSNR node into a list, using MErge interfaces from 'utility' package (from nipype.interfaces.utility import Merge)

'''
from nipype.algorithms.misc import TSNR
from nipype.interfaces.utility import Merge
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

def wf_smooth(): 
	wf = pe.Workflow(name = 'workflow_mapnode_2')
	
	#for input
	inputspec = pe.Node(interface=IdentityInterface(fields=['file_name']), name = 'inputspec')
		
	realign = pe.Node(interface = fsl.MCFLIRT(), name = 'realign')
	
	wf.connect(inputspec,'file_name',realign,'in_file')
	
	
	smooth = pe.MapNode(interface = fsl.maths.IsotropicSmooth(), 
		         name = 'smooth') # run on each file in the list
	smooth.inputs.fwhm = 4
	
	wf.connect(merge,'out',smooth,'in_file')

	
	return wf

