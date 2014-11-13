# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:15:04 2014

@author: bao
"""

"""
Trackconverter at https://github.com/MarcCote/tractconverter

"""
import numpy as np
import tractconverter
sub = [202,204,205,206,209,212]

'''
input_path =  '/home/bao/tiensy/Lauren_registration/data_compare_mapping/tractography/'
output_path = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/tractography/'

in_format = str(".trk")
out_format = ".vtk"

for id_obj in np.arange(len(sub)):
	input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + str(sub[id_obj])+ "_data_brain.nii.gz"
	input_file = input_path   + str(sub[id_obj]) + "_tracks_dti_tvis" + in_format
	output_file = output_path + str(sub[id_obj]) + "_tracks_dti_tvis" + out_format
	
	input_format = tractconverter.detect_format(input_file)
	in_put = input_format(input_file, input_anatomy_ref)
	out_put = tractconverter.FORMATS['vtk'].create(output_file, in_put.hdr, input_anatomy_ref)
	tractconverter.convert(in_put, out_put)
	print "Done", output_file
'''
 
#convert vtk to trackvis
sub = [202,204,205,206,209,212]
input_path =  '/home/bao/tiensy/Lauren_registration/data_compare_mapping/out_registered_f750_l60/iteration_4/'
output_path = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/out_registered_f750_l60/iteration_4/'

#input_path =  '/home/bao/tiensy/Lauren_registration/data_compare_mapping/in_register/'
#output_path = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/in_register/'

in_format = ".vtk"
out_format = str(".trk")


for id_obj in np.arange(len(sub)):
	input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + str(sub[id_obj])+ "_data_brain.nii.gz"
 	input_file = input_path   + str(sub[id_obj]) + "_tracks_dti_tvis_reg" + in_format
  	output_file = output_path + str(sub[id_obj]) + "_tracks_dti_tvis_reg" + out_format
	#input_file = input_path   + str(sub[id_obj]) + "_tracks_dti_tvis" + in_format
	#output_file = output_path + str(sub[id_obj]) + "_tracks_dti_tvis" + out_format
 
	input_format = tractconverter.detect_format(input_file)
	in_put = input_format(input_file, input_anatomy_ref)
	out_put = tractconverter.FORMATS['trk'].create(output_file, in_put.hdr, input_anatomy_ref)
	tractconverter.convert(in_put, out_put)
	print "Done", output_file


