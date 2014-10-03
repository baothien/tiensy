# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:35:34 2014

@author: bao
"""


import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
'''
#for CST_ROI_L
source_ids = [212, 202, 204, 209]
target_ids = [212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206]#, 204, 212, 205]
target_ids = [204]#[206, 204, 212, 205]



for s_id in np.arange(len(source_ids)):
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #=================================================================================================
            #Native space
            #=================================================================================================
            
            '''
            #native_ROI_Left
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'tractography_mapping_cst_sff_in_ext_2_cst_ext.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            arg6 = '-pr=50'            
            arg7 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_native/map_best_' + source + '_' + target + '_cst_L_ann_200.txt'
            #arg8 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_native/map_1nn_' + source + '_' + target + '_cst_L_ann_200.txt'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
            execfile(fname)
            clearall()            
            '''
            
            '''
            #native_ROI_Right
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'tractography_mapping_cst_sff_in_ext_2_cst_ext.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            arg6 = '-pr=50'            
            arg7 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_native/map_best_' + source + '_' + target + '_cst_R_ann_400.txt'
            #arg8 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_native/map_1nn_' + source + '_' + target + '_cst_R_ann_400.txt'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
            execfile(fname)
            clearall() 
            '''
            
            
            #=================================================================================================
            #MNI space
            #=================================================================================================
            '''                       
            #MNI_ROI_left            
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'tractography_mapping_cst_sff_in_ext_2_cst_ext.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            arg6 = '-pr=50'            
            arg7 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_600_MNI.txt'
            #arg8 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_600_MNI.txt'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
            execfile(fname)
            clearall()
            '''
            
            
            #MNI_ROI_right                         
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'probability_fmin_new_implementation_mapping_cst_sff_in_ext_2_cst_ext.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractograph/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            arg6 = '-pr=50'            
            arg7 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_MNI_new_implementation.txt'
            #arg8 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_R_ann_100_MNI.txt'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
            execfile(fname)
            clearall()
            