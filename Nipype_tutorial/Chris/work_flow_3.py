#from nipype.algorithms.misc import TSNR
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
from glob import glob

wf = pe.Workflow(name = 'workflow_mapnode')
wf.base_dir = 'trento_example3'

smooth = pe.MapNode(interface = fsl.maths.IsotropicSmooth(), 
                 name = 'smooth',
		 iterfield = ['in_file'])

smooth.inputs.fwhm = 4
smooth.inputs.in_file = glob('/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/BOLD/task001_run00*/bold_flirt_tsnr_mean.nii.gz')
#glob : Return a list of paths matching a pathname pattern

wf.add_nodes([smooth])
wf.run()

