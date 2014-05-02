# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:33:28 2014

@author: bao
"""
from dipy.tracking.distances import bundles_distances_mam
from dipy.segment.quickbundles import QuickBundles
from common_functions import load_tract
from dipy.viz import fvtk
from common_functions import visualize_tract

import numpy as np

#source_ids =[201, 202 ,203, 204, 205, 206, 207, 208, 209, 210, 212, 213]

#target_ids = [201, 202, 203, 204, 205, 206, 207,208, 209,210, 212, 213]

source_ids =[209]#, 202 ,203, 204, 205, 206, 207, 208, 209, 210, 212, 213]

target_ids = [212]#, 207]#, 204, 205, 206, 207,208, 209,210, 212, 213]
            
vis = True#False#True
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    source = str(source_ids[s_id]) 
    print 'Source: ', source
            
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
           
            target = str(target_ids[t_id])                     
            print 'Target: ', target
            
            s_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M_linear.dpy'
            #s_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
            s_idx = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_corticospinal_L_3M.pkl'    
            #s_cst = load_tract_trk(s_tracts,s_idx)      
            s_cst = load_tract(s_tracts,s_idx)  
            
            t_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M_linear.dpy'
            #t_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
            t_idx = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_corticospinal_L_3M.pkl'            
            #t_cst = load_tract_trk(t_tracts,t_idx)      
            t_cst = load_tract(t_tracts,t_idx) 
            
            '''
            dm = bundles_distances_mam(s_cst, t_cst)
            dm = np.array(dm, dtype=float)
            avg_dis = np.sum(dm) / (len(s_cst)*len(t_cst))
            min_dis = dm.min()
            max_dis = dm.max()           
                
            s_qb = QuickBundles(s_cst,200,18)
    
            s_medoid = s_qb.centroids[0]
            
            t_qb = QuickBundles(t_cst,200,18)
    
            t_medoid = t_qb.centroids[0]
            
            #print len(s_cst), '\t', len(t_cst), '\t', min_dis, '\t', max_dis, '\t', avg_dis, '\t', bundles_distances_mam([s_medoid],[t_medoid])[0][0]
            print min_dis, '\t', max_dis, '\t', avg_dis, '\t', bundles_distances_mam([s_medoid],[t_medoid])[0][0]
            '''
            
            if (vis==True):
                ren = fvtk.ren()       
                ren = visualize_tract(ren,s_cst,fvtk.red)                
                ren = visualize_tract(ren, t_cst, fvtk.blue)
                fvtk.show(ren)
        else:
            print '  '