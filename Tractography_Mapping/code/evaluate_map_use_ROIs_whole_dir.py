# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:34:44 2014

@author: bao
Evaluate the map using ROIs on whole directory
"""

import numpy as np


def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
'''
#for Left CST_ROI - native
source_ids =[ 202, 204, 206]#03]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 213]
target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210,213]
'''
'''
#for Right CST_ROI - native
source_ids =[203, 207, 210]#[201, 202, 203, 204, 205, 207,208, 210, 212, 213]
target_ids = [201, 202, 203, 204, 205, 207,208, 210, 212, 213]
'''

'''
#for Left CST_BOI - native
source_ids =[201, 202, 203]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 212]
target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212]
'''

#for Left CST_BOI - MNI -1000 # 100
source_ids =[203, 212]#[201, 203, 210, 212]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 212]
target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212]

for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    print 'Source: ', source_ids[s_id]
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            '''          
            # Native space for CST_BOI 
            #------------------------------------------------------------------------------------------
            #Left
            print 
            print 'Target: ', target
            
            fname = 'evaluate_map_use_ROIs.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_L_3M.pkl'
            arg2 = target            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + target + '_corticospinal_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_BOI/50_SFF_native/map_best_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_BOI/50_SFF_native/map_1nn_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg8 = '-vi=0'            #'-vi=1'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8]
            execfile(fname)
            clearall()  
            
            # end of native space for CST_BOI
            #------------------------------------------------------------------------------------------
            '''
            
            # MNI space for CST_BOI 
            #------------------------------------------------------------------------------------------
            #Left
            print 
            print 'Target: ', target
            
            fname = 'evaluate_map_use_ROIs.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_L_3M.pkl'
            arg2 = target            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M_linear.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + target + '_corticospinal_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_BOI/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_1000.txt'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_BOI/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_1000.txt'
            arg8 = '-vi=0'            #'-vi=1'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8]
            execfile(fname)
            clearall()  
            
            # end of native space for CST_BOI
            #------------------------------------------------------------------------------------------
            
            
            #------------------------------------------------------------------------------------------
            # Native space for CST_ROI
            '''
            #Left
            print 
            print 'Target: ', target
            
            fname = 'evaluate_map_use_ROIs.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + source + '_CST_ROI_L_3M.pkl'
            arg2 = target            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + target + '_CST_ROI_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_best_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_1nn_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg8 = '-vi=0'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8]
            execfile(fname)
            clearall() 
            '''
            '''
            #Right
            print 
            print 'Target: ', target
            
            fname = 'evaluate_map_use_ROIs.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + source + '_CST_ROI_R_3M.pkl'
            arg2 = target            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + target + '_CST_ROI_R_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + target + '_cst_R_3M_ext_plus_sff.pkl'
            arg6 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_best_' + source + '_' + target + '_cst_R_ann_100.txt'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_1nn_' + source + '_' + target + '_cst_R_ann_100.txt'
            arg8 = '-vi=0'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8]
            execfile(fname)
            clearall() 
            
            # end of native space for CST_ROI
            #------------------------------------------------------------------------------------------
            '''