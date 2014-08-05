# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:09:00 2014

@author: bao
"""
import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]

#for CST_ROI_L
source_ids = [212]#, 202, 204, 209]
target_ids = [212, 202]#, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''


for s_id in np.arange(len(source_ids)):
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
           #=================================================================================================
            #MNI space
            #=================================================================================================
                                 
            #MNI_ROI_left            
            pro = 40
            nn = 10
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'probability_mapping_sparse_density.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            arg6 = '-pr=' + str(pro) 
            arg7 = '-nn=' + str(nn) 
            arg8 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/prob_map_prob_map_' + source + '_' + target + '_cst_L_MNI_' + str(pro) + '_' + str(pro*2)+ '_sparse_density_' + str(nn) + '_neighbors.txt'            
            arg9 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/objective_function_' + source + '_' + target + '_cst_L_MNI_'+ str(pro) + '_' + str(pro*2)+ '_sparse_density_' + str(nn) + '_neighbors.pdf'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9]
            execfile(fname)
            clearall()
            
            
            '''
            #MNI_ROI_right                         
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'probability_mapping.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            arg6 = '-pr=50'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/probability_mapping/prob_map_' + source + '_' + target + '_cst_R_MNI.txt'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
            execfile(fname)
            clearall()
            '''