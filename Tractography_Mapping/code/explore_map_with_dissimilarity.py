# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:19:54 2014

@author: bao

Explore the mapping of prototypes by creating the common dissimilarity space
after aligning the prototypes and the mapped prototypes:

Note:   cst_sff_in_ext of source to cst_ext of target (50 prototypes selected by SFF)
        all are in MNI space, 
        segmentation: ROIs
        tractography: reconstructed from trackvis software not dipy
        result of the mapping as in 
            - tractography_mapping_cst_sff_in_ext_2_cst_ext.py
            - run_tractography_mapping_cst_sff_in_ext_2_cst_ext.py
            note that the mapp saves the index of fiber in the source tract
            with each fiber index i in source, mapp[i] is the index of fiber in target

        cst_s and cst_t is the source and target of cst
        
Step 0: Source prototypes (pr_s) and mapped prototypes (in target tracts_extension) (call mapped_pr_s_in_t)
        are aligned together
        
Step 1: Compute the dissimilarity of source tract based on pr_s - called dis_s
        Compute the dissimilarity of target tract extension based on mapped_pr_s_in_t (mapped of pr_s in target extension)
        - called dis_t_ext

Step 3: Compute the kd-tree of target tract extension based on dis_t_ext - call kdt_t_ext

Step 4: Segment the cst_s in the target using nearest neighbor of each fiber_source 
        in the space of the kdt_t_ext - result is nn_cst_s_in_t
        
Step 5: Compute JAC and BFN between cst_t and nn_cst_s_in_t

"""

from common_functions import load_tract, Jac_BFN, vol_corr_notcorr, visualize_tract
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.io.pickles import load_pickle
from dipy.viz import fvtk
import numpy as np


from sklearn.neighbors import KDTree, BallTree

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
source_ids = [206, 204, 212, 205]#[204, 212, 205]#
target_ids = [206, 204, 212, 205]#[206, 204, 212, 205]


vol_dims = [182,218,182]
vis = False#True#False


#-------------------------------------------------------------------
#            Annealing
#-------------------------------------------------------------------
anneal = [100, 200, 400, 600, 800, 1000]#[100]#
num_pro = 50
for a_id in np.arange(len(anneal)):
    print "==================================================================="
    print "Anneal : ", anneal[a_id]
    #print "\t\t Target \t not_map_Jac \t not_map_BFN \t map_Jac \t map_BFN" 
    print "\t\t Target \t vol correct(not map) \t vol not correct (not map) \t vol correct(mapping+dis) \t vol not correct (mapping+dis)" 
    for s_id in np.arange(len(source_ids)):
        #print "------------------------------------------"
        print source_ids[s_id]    
        for t_id in np.arange(len(target_ids)):        
            if (target_ids[t_id] != source_ids[s_id]):                
                source = str(source_ids[s_id])
                target = str(target_ids[t_id])
                
                
                #-------------------------------------------------------------------------------------------------------------------------------------------
                #Mapping method
                '''
                #Left
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'

                s_cst_sff_in_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'                                
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
                
                #annealing
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                """
                ##probability
                ##nn = 10
                ##map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/prob_map_prob_map_' + source + '_' + target + '_cst_L_MNI_full_full' + '_sparse_density_' + str(nn) + '_neighbors.txt' 
                """
                
                '''
                #Right
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'

                s_cst_sff_in_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'                
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                   
                #annealing
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                """
                ##probability
                ##nn = 10
                ##map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/prob_map_prob_map_' + source + '_' + target + '_cst_R_MNI_full_full' + '_sparse_density_' + str(nn) + '_neighbors.txt' 
                """
                
                #-------end of mapping method-----------------------------------------
                #-------------------------------------------------------------------------------------------------------------------------------------------
                
                #==============================================================================================================================================================                
                """
                #-------------------------------------------------------------------------------------------------------------------------------------------
                #Lauren_method of group registration
                indir = 'out_registered_f750_l60'            
                #indir = 'out_registered_f300_l75'            
                s_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + source + '_tracks_dti_tvis_reg.trk'
                t_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + target + '_tracks_dti_tvis_reg.trk'
            
                
                #Left
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'

                s_cst_sff_in_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'                
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f300_l75/map_1nn_'+ source + '_' + target + '_cst_sff_in_ext_L_Lauren.txt'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f750_l60/map_1nn_'+ source + '_' + target + '_cst_sff_in_ext_L_Lauren.txt'
                
                '''                
                #Right
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'

                s_cst_sff_in_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'                
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f300_l75/map_1nn_'+ source + '_' + target + '_cst_sff_in_ext_R_Lauren.txt'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f750_l60/map_1nn_'+ source + '_' + target + '_cst_sff_in_ext_R_Lauren.txt'
                
                '''
                #-------end of Lauren group registration method-----------------------------------------
                
                """
                
                #==============================================================================================================================================================                
                
                
                s_cst = load_tract(s_file, s_cst_idx)
                t_cst = load_tract(t_file, t_cst_idx)
                
                s_cst_sff_in_ext = load_tract(s_file, s_cst_sff_in_ext_idx)
                t_cst_ext = load_tract(t_file, t_cst_ext_idx)
                
                #---------------------------------------------------
                #for normal mapping
                map_all = load_pickle(map_file)
                
                #-----------------------------------------------                
                
                
                cst_len = len(s_cst)
                
                pr_s = s_cst_sff_in_ext[-num_pro:]
                
                #Step 0: Source prototypes (pr_s) and mapped prototypes (in target tracts_extension) 
                #        (call mapped_pr_s_in_t) are aligned together        
                mapped_pr = map_all[-num_pro:]
                
                mapped_pr_s_in_t = [t_cst_ext[idx] for idx in mapped_pr]
                
                #Step 1: Compute the dissimilarity of source tract based on pr_s - called dis_s
                #       Compute the dissimilarity of target tract extension based on mapped_pr_s_in_t (mapped of pr_s in target extension)- called dis_t_ext
                
                dis_s = bundles_distances_mam(s_cst, pr_s)
                dis_t_ext = bundles_distances_mam(t_cst_ext, mapped_pr_s_in_t)

                #Step 3: Compute the kd-tree of target tract extension based on dis_t_ext - call kdt_t_ext
                kdt_t_ext = BallTree(dis_t_ext,leaf_size=30) # KDTree(dis_t_ext) 
                                             
            
                #Step 4: Segment the cst_s in the target using nearest neighbor of each fiber_source 
                #        in the space of the kdt_t_ext - result is nn_cst_s_in_t
        
                k = 1         
                dst_nn, idx_nn = kdt_t_ext.query(dis_s, k)

                #print 'Distance'
                #print dst_nn
                #print 'Index '
                #print idx_nn                
                
                mapped_s_cst = [t_cst_ext[idx_nn[i][0]] for i in np.arange(cst_len)]
                
                #Step 5: Compute JAC and BFN between cst_t and nn_cst_s_in_t
                
                #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
                #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
                #print "\t\t", target_ids[t_id], "\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
                #print "Before mapping: ", jac0, bfn0
                #print "After mapping: ", jac1, bfn1
                
                cor0, ncor0 = vol_corr_notcorr(s_cst, t_cst, vol_dims, disp=False)
                cor1, ncor1 = vol_corr_notcorr(mapped_s_cst, t_cst, vol_dims, disp=False)
                print "\t\t", target_ids[t_id], "\t", cor0,"\t",  ncor0, "\t", cor1,"\t",  ncor1
                
                
               
                if vis:
                   #visualize target cst and mapped source cst - yellow and blue
                    ren = fvtk.ren()                
                    ren = visualize_tract(ren, s_cst, fvtk.yellow)
                    ren = visualize_tract(ren, t_cst, fvtk.blue)
                    ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                    fvtk.show(ren)
                    