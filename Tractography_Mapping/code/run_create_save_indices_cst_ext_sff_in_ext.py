# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:18:44 2014

@author: bao
"""

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
#Left
#source_ids =[212, 202, 204, 209]
#Right
source_ids =[205, 212, 204, 206]
for s_id in np.arange(len(source_ids)):   
    source = str(source_ids[s_id])    
    
    #ROI segmentation    
    '''    
    #native
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_sff_in_ext.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
    arg3 = '-pr=50'    
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_ext.pkl'
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'
    '''
    
    #MNI
    print 
    print '-----------------------------------------------------------------------------------------'
    fname = 'create_save_indices_cst_ext_sff_in_ext.py'
    arg1 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
    arg2 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis_MNI/' + source + '_corticospinal_R_tvis_MNI.pkl'
    arg3 = '-pr=50'    
    arg4 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_MNI/' + source + '_cst_R_tvis_ext_MNI.pkl'
    arg5 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_MNI/' + source + '_cst_R_tvis_sff_in_ext_MNI.pkl'
        
    
    import sys
    sys.argv = [fname, arg1, arg2, arg3, arg4, arg5]
    execfile(fname)
    clearall() 