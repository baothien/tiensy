# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:46:48 2015

@author: bao
This is to convert the CST segmented using ROI on trackvis tractography 
from trk format to vtk format
"""

"""
Trackconverter at https://github.com/MarcCote/tractconverter

"""
import numpy as np
import tractconverter
sub = [202,204,205,206,209,212]


#load pickle and save CST as trk file
from common_functions import load_tract, save_tracks_dpy, save_tract_trk

'''
#for CST_ROI_L
source_ids = [212, 202, 204, 209]

'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]



for s_id in np.arange(len(source_ids)):
    subj = str(source_ids[s_id])    
            
    #=================================================================================================
    #Native space
    #=================================================================================================
    '''
    #CST_Left
    #saving CST as trk file
    s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + subj + '_tracks_dti_tvis.trk'
    #cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_L_tvis.pkl'            
    #cst_tract = load_tract(s_file,cst_ind)    
    cst_ext_ind  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + subj + '_cst_L_tvis_ext.pkl'    
    cst_tract_ext = load_tract(s_file,cst_ext_ind)
    
    s_fa = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + subj + '/DTI/dti_fa_brain.nii.gz'
    #cst_trk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_L_tvis.trk'  
    #save_tract_trk(cst_tract, s_fa, cst_trk_fname)
    cst_ext_trk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_cst_L_tvis_ext.trk'            
    save_tract_trk(cst_tract_ext, s_fa, cst_ext_trk_fname)    

    #converting trk file to vtk file
    input_anatomy_ref = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/' + subj + '_data_brain.nii.gz'  
    #it is a link to '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + subj + '/DTI/data_brain.nii.gz'
    #cst_vtk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_L_tvis.vtk'
    cst_ext_vtk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_cst_L_tvis_ext.vtk'
    '''
    
    #CST_Right
    #saving CST as trk file
    s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + subj + '_tracks_dti_tvis.trk'
    #cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_R_tvis.pkl'            
    #cst_tract = load_tract(s_file,cst_ind)
    cst_ext_ind  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + subj + '_cst_R_tvis_ext.pkl'    
    cst_tract_ext = load_tract(s_file,cst_ext_ind)    
    
    s_fa = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + subj + '/DTI/dti_fa_brain.nii.gz'
    #cst_trk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_R_tvis.trk'            
    #save_tract_trk(cst_tract, s_fa, cst_trk_fname)
    cst_ext_trk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_cst_R_tvis_ext.trk'            
    save_tract_trk(cst_tract_ext, s_fa, cst_ext_trk_fname) 
    
    #converting trk file to vtk file
    input_anatomy_ref = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/anatomy/' + subj + '_data_brain.nii.gz'  
    #it is a link to '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + subj + '/DTI/data_brain.nii.gz'
    #cst_vtk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_corticospinal_R_tvis.vtk'
    cst_ext_vtk_fname = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + subj + '_cst_R_tvis_ext.vtk'    
   
        
    in_format = str(".trk")
    out_format = ".vtk"
    #input_file = cst_trk_fname
    #output_file = cst_vtk_fname
    
    input_file = cst_ext_trk_fname
    output_file = cst_ext_vtk_fname
    
    input_format = tractconverter.detect_format(input_file)
    in_put = input_format(input_file, input_anatomy_ref)
    out_put = tractconverter.FORMATS['vtk'].create(output_file, in_put.hdr, input_anatomy_ref)
    tractconverter.convert(in_put, out_put)
    print "Done", output_file
    
    
    
'''

input_path =  '/home/bao/tiensy/Lauren_registration/data_compare_mapping/tractography/'
output_path = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/tractography/'



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
'''

