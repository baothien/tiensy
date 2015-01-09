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
 
"""
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

"""


#for CST_ROI_L
source_ids = [212, 202, 204, 209]
target_ids = [212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''

in_format = ".vtk"
out_format = str(".trk")

for s_id in np.arange(len(source_ids)):
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #=================================================================================================
            #Native space
            #=================================================================================================
            
            #native_ROI_Left
            
            #input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg/iteration_4/' + target + '_cst_L_tvis_ext_reg'+ in_format
            #output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg/iteration_4/' + target + '_cst_L_tvis_ext_reg'+ out_format
            input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + target + '_cst_L_tvis_ext_reg'+ in_format
            output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + target + '_cst_L_tvis_ext_reg'+ out_format
            
            input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + target + "_data_brain.nii.gz"
                       
            input_format = tractconverter.detect_format(input_file)
            in_put = input_format(input_file, input_anatomy_ref)
            out_put = tractconverter.FORMATS['trk'].create(output_file, in_put.hdr, input_anatomy_ref)
            tractconverter.convert(in_put, out_put)
            print "Done", output_file
            
            #input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg/iteration_4/' + source + '_corticospinal_L_tvis_reg'+ in_format
            #output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg/iteration_4/' + source + '_corticospinal_L_tvis_reg'+ out_format
            input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + source + '_corticospinal_L_tvis_reg'+ in_format
            output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + source + '_corticospinal_L_tvis_reg'+ out_format
            
            input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + source + "_data_brain.nii.gz"
            
            input_format = tractconverter.detect_format(input_file)
            in_put = input_format(input_file, input_anatomy_ref)
            out_put = tractconverter.FORMATS['trk'].create(output_file, in_put.hdr, input_anatomy_ref)
            tractconverter.convert(in_put, out_put)
            print "Done", output_file
            '''
            
            #native_ROI_Right
            
            #input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg/iteration_4/' + target + '_cst_R_tvis_ext_reg'+ in_format
            #output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg/iteration_4/' + target + '_cst_R_tvis_ext_reg'+ out_format
            input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + target + '_cst_R_tvis_ext_reg'+ in_format
            output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + target + '_cst_R_tvis_ext_reg'+ out_format
            input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + target + "_data_brain.nii.gz"
                       
            input_format = tractconverter.detect_format(input_file)
            in_put = input_format(input_file, input_anatomy_ref)
            out_put = tractconverter.FORMATS['trk'].create(output_file, in_put.hdr, input_anatomy_ref)
            tractconverter.convert(in_put, out_put)
            print "Done", output_file
            
            #input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg/iteration_4/' + source + '_corticospinal_R_tvis_reg'+ in_format
            #output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg/iteration_4/' + source + '_corticospinal_R_tvis_reg'+ out_format
            input_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + source + '_corticospinal_R_tvis_reg'+ in_format
            output_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg_f100_l25/iteration_4/' + source + '_corticospinal_R_tvis_reg'+ out_format
            input_anatomy_ref = "/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/" + source + "_data_brain.nii.gz"
            input_format = tractconverter.detect_format(input_file)
            in_put = input_format(input_file, input_anatomy_ref)
            out_put = tractconverter.FORMATS['trk'].create(output_file, in_put.hdr, input_anatomy_ref)
            tractconverter.convert(in_put, out_put)
            print "Done", output_file
            '''