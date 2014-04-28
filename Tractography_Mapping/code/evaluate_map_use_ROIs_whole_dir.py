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

source_ids =[202,203]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 212]

target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212]
            
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    print 'Source: ', source_ids[s_id]
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #Native space            
            print 
            print 'Target: ', target
            
            fname = 'evaluate_map_use_ROIs.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + source + '_corticospinal_L_3M.pkl'
            arg2 = target            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_tracks_dti_3M.trk'
            arg4 = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_corticospinal_L_3M.pkl'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/' + target + '_cst_L_3M_ext_plus_sff.pkl'
            arg6 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_best_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg7 = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_1nn_' + source + '_' + target + '_cst_L_ann_100.txt'
            arg8 = '-vi=1'            
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8]
            execfile(fname)
            clearall()     