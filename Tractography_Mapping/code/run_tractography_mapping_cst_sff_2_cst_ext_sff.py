# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:33:29 2014

@author: bao
"""
import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]


#source_ids = [201,202]#, 203, 204] #[ 208, 209, 210, 212] #[202, 203, 204, 205, 206, 207, 208, 209, 210, 212]
#target_ids = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212]

source_ids = [212]#4,205] #[ 208, 209, 210, 212] #[202, 203, 204, 205, 206, 207, 208, 209, 210, 212]
#target_ids = [212, 210, 209, 208, 207, 206, 205]#, 204]# 209, 210, 212]
target_ids = [201, 202, 203, 204,205,206,  207, 208, 209, 210, 212]


#source_ids = [204] #[ 208, 209, 210, 212] #[202, 203, 204, 205, 206, 207, 208, 209, 210, 212]
#target_ids = [205, 206, 207, 208, 209, 210, 212]

#source_ids = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212]
#target_ids = [201, 202, 203, 204, 205, 206]
#target_ids = [207, 208, 209, 210, 212]

for s_id in np.arange(len(source_ids)):
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #Native space
            #string = 'run tractography_mapping_cst_sff_2_cst_ext_sff.py /home/bao/tiensy/Tractography_Mapping/code/data/' + source + '_tracks_dti_3M.dpy /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + source + '_cst_L_3M_plus_sff.pkl /home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_tracks_dti_3M.dpy /home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_corticospinal_L_3M.pkl /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + target + '_cst_L_3M_ext_plus_sff.pkl -pr=50 -an=100 /home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_best_' + source + '_' + target + '_cst_L_ann_100.txt /home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_1nn_' + source + '_' + target + '_cst_L_ann_100.txt'
            #print string
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'tractography_mapping_cst_sff_2_cst_ext_sff.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + source + '_tracks_dti_3M.dpy'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + source + '_cst_L_3M_plus_sff.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_tracks_dti_3M.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_corticospinal_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '-pr=50'
            arg7 = '-an=100'
            arg8 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_best_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg9 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_1nn_' + source + '_' + target + '_cst_L_ann_100.txt'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9]
            execfile(fname)
            clearall()            
            
            '''
            #MNI space
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'tractography_mapping_cst_sff_2_cst_ext_sff.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + source + '_tracks_dti_3M_linear.dpy'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + source + '_cst_L_3M_plus_sff.pkl'
            arg3 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_tracks_dti_3M_linear.dpy'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_corticospinal_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '-pr=50'
            arg7 = '-an=1000'
            arg8 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_1000.txt'
            arg9 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_1000.txt'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9]
            execfile(fname)
            clearall()
            '''