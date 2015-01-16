# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:23:40 2015

@author: bao
"""

from common_functions import load_tract, Shannon_entropy, visualize_tract, length_min, length_max, length_avg, truth_length_min, truth_length_max, truth_length_avg
from dipy.viz import fvtk
import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and  var!="source_ids" and var!="s_id" and var!="np"]
    for var in all:
        del globals()[var]


        
    
#for CST_ROI_L
sub_ids = [212, 202, 204, 209]


'''
#for CST_ROI_R
sub_ids = [206, 204, 212, 205]
'''

visualize = False
print '---------------------------------------------------------------------------'
print 'Subject \t size CST  \t CST_Len_min  \t CST_Len_max \t CST_Entropy \t size CST_ext \t CST_ext_Len_min  \t CST_ext_Len_max  \t CST_ext_Len_avg \t CST_SFF_entropy \t CST_ext_Entropy'
for s_id in np.arange(len(sub_ids)):
    
    sub = str(sub_ids[s_id])
    
    """
    #native space
    
    tractography_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + sub + '_tracks_dti_tvis.trk'
       
    #CST Left
    cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + sub + '_corticospinal_L_tvis.pkl'            
    cst_sff_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_L_tvis_sff_in_ext.pkl'
    cst_ext_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_L_tvis_ext.pkl'
    '''
    
    #CST Right
    cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + sub + '_corticospinal_R_tvis.pkl'    
    cst_sff_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_R_tvis_sff_in_ext.pkl'        
    cst_ext_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_R_tvis_ext.pkl'
    '''
    """
    
    
    #MNI space
    
    tractography_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + sub + '_tracks_dti_tvis_linear.trk'
    
    
    #CST Left
    cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + sub + '_corticospinal_L_tvis.pkl'            
    cst_sff_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_L_tvis_sff_in_ext.pkl'
    cst_ext_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_L_tvis_ext.pkl'
    '''
    
    #CST Right
    cst_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + sub + '_corticospinal_R_tvis.pkl'    
    cst_sff_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_R_tvis_sff_in_ext.pkl'        
    cst_ext_ind = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + sub + '_cst_R_tvis_ext.pkl'
    '''
        
       
    cst = load_tract(tractography_file, cst_ind)
    cst_sff = load_tract(tractography_file, cst_sff_ind)
    cst_ext = load_tract(tractography_file, cst_ext_ind)
    
    #cst_len_min = length_min(cst)
    #cst_len_max = length_max(cst)
    #cst_len_avg = length_avg(cst)
    #cst_ext_len_min = length_min(cst_ext)
    #cst_ext_len_max = length_max(cst_ext)
    #cst_ext_len_avg = length_avg(cst_ext)
    
    cst_len_min = truth_length_min(cst)
    cst_len_max = truth_length_max(cst)
    cst_len_avg = truth_length_avg(cst)
    cst_ext_len_min = truth_length_min(cst_ext)
    cst_ext_len_max = truth_length_max(cst_ext)
    cst_ext_len_avg = truth_length_avg(cst_ext)
    
    
    cst_entropy = Shannon_entropy(cst)    
    cst_sff_entropy = Shannon_entropy(cst_sff)
    cst_ext_entropy = Shannon_entropy(cst_ext)    
    
    
               
    print sub,"\t", len(cst), "\t", cst_len_min,"\t", cst_len_max,"\t", cst_len_avg , "\t" , cst_entropy ,"\t", len(cst_ext), "\t", cst_ext_len_min,"\t", cst_ext_len_max,"\t", cst_ext_len_avg , "\t" , cst_sff_entropy, '\t', cst_ext_entropy         
                
    if (visualize==True):
        ren = fvtk.ren()
        ren = visualize_tract(ren, cst, fvtk.yellow)
        fvtk.show(ren)