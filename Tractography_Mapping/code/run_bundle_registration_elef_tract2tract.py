# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 20:32:42 2014

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
source_ids = [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]



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
            fname = 'bundle_registration_elef_tract2tract.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis.trk'            
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            #arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_L_' + source + '_aligned_to_CST_L_ext_' + target + '_elef.dpy'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_L_' + source + '_aligned_to_CST_L_ext_' + target + '_elef_rand_200.dpy'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5]
            execfile(fname)
            clearall()            
            
            
            '''
            #native_ROI_Right
            print 
            print '-----------------------------------------------------------------------------------------'
            fname = 'bundle_registration_elef_tract2tract.py'
            arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
            arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'            
            arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis.trk'            
            arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            #arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_R_' + source + '_aligned_to_CST_R_ext_' + target + '_elef.dpy'
            arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_R_' + source + '_aligned_to_CST_R_ext_' + target + '_elef_rand_200.dpy'
            import sys
            sys.argv = [fname, arg1, arg2, arg3, arg4, arg5]            
            execfile(fname)
            clearall() 
           