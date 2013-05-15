'''
The task is to smooth three outputs of TSNR node (mean, stddev, and tsnr)

- To do this, we have to use MapNode

- But MapNode always take lishs - therefore we have to merge the three outputs of the TSNR node into a list, using MErge interfaces from 'utility' package (from nipype.interfaces.utility import Merge)

'''
from nipype.algorithms.misc import TSNR
from nipype.interfaces.utility import Merge
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

wf = pe.Workflow(name = 'workflow_mapnode_2')
wf.base_dir = 'trento_example4'  # if the folder does not exist, then it will be created automatically


realign = pe.Node(interface = fsl.MCFLIRT(), name = 'realign')
realign.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/BOLD/task001_run001/bold.nii.gz'

tsnr = pe.Node(interface=TSNR(), name = 'tsnr')
tsnr.inputs.regress_poly = 1
#tsnr.inputs.in_file = '/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/BOLD/task001_run001/bold_flirt.nii.gz'

wf.connect(realign,'out_file', tsnr, 'in_file')

merge = pe.Node(interface = Merge(3), name = 'merge')
wf.connect(tsnr,'stddev_file', merge, 'in1')
wf.connect(tsnr,'mean_file', merge, 'in2')
wf.connect(tsnr,'tsnr_file', merge, 'in3')


smooth = pe.MapNode(interface = fsl.maths.IsotropicSmooth(), 
                 name = 'smooth',
		 iterfield = ['in_file']) # run on each file in the list
smooth.inputs.fwhm = 4
wf.connect(merge,'out',smooth,'in_file')

wf.run()

