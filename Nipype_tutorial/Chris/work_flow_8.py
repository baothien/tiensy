
#realign.inputs.in_file ='/home/bao/PhD_Trento/Nipype_tutorial/trento/trento/BOLD/task001_run001/bold.nii.gz'

	tsnr = pe.Node(interface=TSNR(), name = 'tsnr')
	tsnr.inputs.regress_poly = 1

wf.connect(realign,'out_file', tsnr, 'in_file')

	merge = pe.Node(interface = Merge(3), name = 'merge')
	wf.connect(tsnr,'stddev_file', merge, 'in1')
	wf.connect(tsnr,'mean_file', merge, 'in2')
	wf.connect(tsnr,'tsnr_file', merge, 'in3')


