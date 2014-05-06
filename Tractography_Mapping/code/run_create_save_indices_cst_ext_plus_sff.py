# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:16:23 2014

@author: bao
"""

import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]


#source_ids = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212]
source_ids = [202]#[201, 202, 203, 204, 205, 206, 207, 209, 210, 212,213]

for s_id in np.arange(len(source_ids)):   
    source = str(source_ids[s_id])
   
    
    #BOI segmentation
    #string = 'run create_save_indices_cst_ext_plus_sff.py /home/bao/tiensy/Tractography_Mapping/code/data/212_tracks_dti_3M.dpy /home/bao/tiensy/Tractography_Mapping/code/data/212_corticospinal_L_3M.pkl -pr=50 /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_plus_sff.pkl /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_ext.pkl /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_ext_plus_sff.pkl '
    #print string
    '''
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_plus_sff.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_L_3M.pkl'
    arg3 = '-pr=50'
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_L_3M_plus_sff.pkl'    
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_L_3M_ext.pkl'
    arg6 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_L_3M_ext_plus_sff.pkl'
    
    import sys
    sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6]
    execfile(fname)
    clearall()
    '''
    '''
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_plus_sff.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_R_3M.pkl'
    arg3 = '-pr=50'
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_R_3M_plus_sff.pkl'    
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_R_3M_ext.pkl'
    arg6 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_R_3M_ext_plus_sff.pkl'
    
    import sys
    sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6]
    execfile(fname)
    clearall()
    '''
    
    
    #ROI segmentation
    #string = 'run create_save_indices_cst_ext_plus_sff.py /home/bao/tiensy/Tractography_Mapping/code/data/212_tracks_dti_3M.dpy /home/bao/tiensy/Tractography_Mapping/code/data/212_corticospinal_L_3M.pkl -pr=50 /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_plus_sff.pkl /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_ext.pkl /home/bao/tiensy/Tractography_Mapping/code/data/50_SFF/212_cst_L_3M_ext_plus_sff.pkl '
    #print string
    '''
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_plus_sff.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + source + '_CST_ROI_L_3M.pkl'
    arg3 = '-pr=50'
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_L_3M_plus_sff.pkl'    
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_L_3M_ext.pkl'
    arg6 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_L_3M_ext_plus_sff.pkl'
    
    import sys
    sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6]
    execfile(fname)
    clearall()             
    '''
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_plus_sff.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + source + '_CST_ROI_R_3M.pkl'
    arg3 = '-pr=50'
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_R_3M_plus_sff.pkl'    
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_R_3M_ext.pkl'
    arg6 = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_R_3M_ext_plus_sff.pkl'
    
    import sys
    sys.argv = [fname, arg1, arg2, arg3, arg4, arg5, arg6]
    execfile(fname)
    clearall() 
    