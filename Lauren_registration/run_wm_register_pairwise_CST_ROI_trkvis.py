# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 17:30:25 2015

@author: bao
"""

import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]


#for CST_ROI_L
source_ids = [212, 202, 204, 209]
target_ids = [212, 202, 204, 209]


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
            #Native space
            #=================================================================================================
                        
            #native_ROI_Left
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'wm_register_multisubject.py'
            arg1 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/in_reg'
            #arg2 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg'
            arg2 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source + '_' + target + '/out_reg_f100_l25'
            arg3 = '-j 4'            
            arg4 = '-f 100'            
            arg5 = '-l 25'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5]
            execfile(fname)
            clearall()            
            '''
            
            #native_ROI_Right
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'wm_register_multisubject.py'
            arg1 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/in_reg'
            #arg2 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg'
            arg2 = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source + '_' + target + '/out_reg_f100_l25'
            arg3 = '-j 4'            
            arg4 = '-f 100'            
            arg5 = '-l 25'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5]
            execfile(fname)
            clearall()            
            '''