# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:24:45 2014

@author: bao
"""

from common_functions import load_tract, load_whole_tract, Jac_BFN, visualize_tract, visualize_tract_transparence
from dipy.io.pickles import load_pickle
from dipy.viz import fvtk
import numpy as np


def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
        

#for CST_ROI_L
source_ids = [204]#[212, 202, 204, 209]
target_ids = [202]#, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [205]# [206, 204, 212, 205]
target_ids = [206]#[206, 204, 212, 205]
'''

vol_dims = [182,218,182]
vis = True#False


#-------------------------------------------------------------------
#            Annealing
#-------------------------------------------------------------------
anneal = [1000]#, 200, 400, 600, 800, 1000]

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
                
                #Left
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_100_MNI.txt'
                
                '''
                #Right
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                '''                
                s_cst = load_tract(s_file, s_cst_idx)
                #tracto = load_whole_tract(s_file)
                t_cst = load_tract(t_file, t_cst_idx)
                t_cst_ext = load_tract(t_file, t_cst_ext_idx)
                map_all = load_pickle(map_file)
                
                cst_len = len(s_cst)
                mapped = map_all[:cst_len]
                
                mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
                
#                s1_idx = load_pickle(s_cst_idx)                
#                s_idx = [ i for i in s1_idx] 
#                print s1_idx
#                t1_idx = load_pickle(t_cst_idx)      
#                t_idx = [i for i in t1_idx]
#                print t_idx
#                s_TP_idx = []
#                s_FN_idx = []
#                for k in np.arange(cst_len):
#                    j = mapped[k]
#                    if j in t1_idx:
#                        s_TP_idx.append(s_idx[k])
#                    else:
#                        s_FN_idx.append(s_idx[k])
#                print len(s_idx), len(s_TP_idx), len(s_FN_idx)
#                
#                stop
                
                
                #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=True, save=False)
                print 'mapped'
                jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=True, save=False)
                
                print "\t\t", target_ids[t_id], "\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
                
                #print "Before mapping: ", jac0, bfn0
                #print "After mapping: ", jac1, bfn1
               
                if vis:
                   #visualize target cst and mapped source cst - yellow and blue
                    ren = fvtk.ren() 
                    ren.SetBackground(255,255,255)
                    #ren = visualize_tract(ren, s_cst, fvtk.blue)
                    ren = visualize_tract_transparence(ren, t_cst_ext, fvtk.red, 0.1)                    
                    #ren = visualize_tract_transparence(ren, t_cst, fvtk.green, 0.8, 1)
                    ren = visualize_tract_transparence(ren, mapped_s_cst, fvtk.blue, 100.,1.5)
                    fvtk.show(ren)
                    

'''
#------------------------------------------------------------
#          1-NN 
#------------------------------------------------------------                 
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])

            
                        
            #Left
            s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_100_MNI.txt'
            
            
            #Right
            s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_R_ann_100_MNI.txt'
            

                
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
'''
'''
#------------------------------------------------------------
#          probability mapping
#------------------------------------------------------------                 
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #Left
            
            
            
            #Right
            nn = 10#15#10            
            
            s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
            
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'            
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
            
            
            map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/prob_map_prob_map_' + source + '_' + target + '_cst_R_MNI_full_full' + '_sparse_density_' + str(nn) + '_neighbors.txt'                            
            
            
            s_cst = load_tract(s_file, s_cst_idx)
            t_cst = load_tract(t_file, t_cst_idx)
            s_cst_sff = load_tract(s_file, s_cst_sff_idx)
            t_cst_ext = load_tract(t_file, t_cst_ext_idx)            
            
            
            map_all = load_pickle(map_file)
            
            from common_functions import init_prb_state_sparse
            
            prb_map12_init, cs_idxs = init_prb_state_sparse(s_cst_sff,t_cst_ext,nn) 
            
            #aang lam do den fan nay
            #load cai map va chon cai co prob cao nhat
            
            cst_len = len(s_cst)
            map_tmp = map_all[:cst_len]
            
            map_idxs = [map_tmp[i].argsort()[-1] for i in np.arange(cst_len)]
            
            mapped = [cs_idxs[i][map_idxs[i]] for i in np.arange(cst_len)]
            
            #print map_tmp[:10], map_idxs[:10], cs_idxs[:10], mapped[:10]            
            
            #stop
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
            jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=True)#True)#False)
            
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
'''