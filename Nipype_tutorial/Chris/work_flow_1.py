
from nipype.algorithms.misc import TSNR
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

wf = pe.Workflow(name = 'my_workflow')
wf.base_dir = 'trento_temp'


realign = pe.Node(interface = fsl.MCFLIRT(), name = 'realign')
realign.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/BOLD/task001_run001/bold.nii.gz'

tsnr = pe.Node(interface=TSNR(), name = 'tsnr')
tsnr.inputs.regress_poly = 1

smooth = pe.Node(interface = fsl.maths.IsotropicSmooth(), name = 'smooth')
smooth.inputs.terminal_output = 'stream'
#which value should be used????
#smooth.inputs.fwhm = 0.5 # run only one value of smooth
smooth.iterables = ('fwhm',[4,6,8])  # run with many values of smooth

wf.connect(realign,'out_file', tsnr, 'in_file')

#smooth the detrended output of the tsnr node
wf.connect(tsnr,'detrended_file', smooth, 'in_file')

#wf.run(plugin='Linear')# run with only one value of smooth
wf.run(plugin='MultiProc') # run with many values of smooth
#wf.run(updatehash=True)
