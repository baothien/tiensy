# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:18:02 2014

@author: bao

Evaluate the mapping using two measurements: Jaccard and BFN indices
cst_sff_in_ext to cst_ext
all are in MNI space, 
segmentation: ROIs
tractography: reconstructed from trackvis software not dipy
result of the mapping as in 
     - tractography_mapping_cst_sff_in_ext_2_cst_ext.py
     - run_tractography_mapping_cst_sff_in_ext_2_cst_ext.py
note that the mapp saves the index of fiber in the source tract
with each fiber index i in source, mapp[i] is the index of fiber in target
       
"""
from common_functions import load_tract, Jac_BFN, visualize_tract
from dipy.io.pickles import load_pickle
from dipy.viz import fvtk
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


anneal = [100, 200, 400, 600, 800, 1000]
vol_dims = [182,218,182]
vis = False
for a_id in np.arange(len(anneal)):
    print "==================================================================="
    print "Anneal : ", anneal[a_id]
    print "\t\t Target \t not_map_Jac \t not_map_BFN \t map_Jac \t map_BFN" 
    for s_id in np.arange(len(source_ids)):
        #print "------------------------------------------"
        print source_ids[s_id]    
        for t_id in np.arange(len(target_ids)):        
            if (target_ids[t_id] != source_ids[s_id]):                
                source = str(source_ids[s_id])
                target = str(target_ids[t_id])
                '''
                #Left
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                '''
                #Right
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                
                s_cst = load_tract(s_file, s_cst_idx)
                t_cst = load_tract(t_file, t_cst_idx)
                t_cst_ext = load_tract(t_file, t_cst_ext_idx)
                map_all = load_pickle(map_file)
                
                cst_len = len(s_cst)
                mapped = map_all[:cst_len]
                
                mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
                
                jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
                jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
                
                print "\t\t", target_ids[t_id], "\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
                #print "Before mapping: ", jac0, bfn0
                #print "After mapping: ", jac1, bfn1
               
                if vis:
                   #visualize target cst and mapped source cst - yellow and blue
                    ren = fvtk.ren()                
                    ren = visualize_tract(ren, s_cst, fvtk.yellow)
                    ren = visualize_tract(ren, t_cst, fvtk.blue)
                    ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                    fvtk.show(ren)