# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:07:51 2014

@author: bao
"""
from common_functions import load_tract, Jac_BFN, visualize_tract
from dipy.io.pickles import load_pickle, save_pickle
from dipy.viz import fvtk
import numpy as np
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.tracking.metrics import length

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
        

#for CST_ROI_L
source_ids = [212, 202, 204, 209]#[212, 202, 204, 209]
target_ids = [212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]# [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''

vol_dims = [128,128,80]
vis = False#True
save = False#True


def mapping_nn(tractography1, tractography2):
    #print 'Compute the 1-nn from source to target'
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)    
    return mapping12_coregistration_1nn


#------------------------------------------------------------
#          1-NN 
#------------------------------------------------------------                 
print "The coregistration+1NN gives a mapping12 with the following measurement:"
print "\t\t Target \t not_map_Jac \t not_map_BFN \t map_Jac \t map_BFN" 
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source_sub = str(source_ids[s_id])
            target_sub = str(target_ids[t_id])
       

            #indir = 'out_registered_defaultpara'            
            indir = 'out_registered_f750_l60'            
            s_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + source_sub + '_tracks_dti_tvis_reg.trk'
            t_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + target_sub + '_tracks_dti_tvis_reg.trk'
            
            
            #Left            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_L_tvis.pkl'
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source_sub + '_cst_L_tvis_sff_in_ext.pkl'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_L_tvis_ext.pkl'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_L_Lauren.txt'
            '''
            
            #Right
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_R_tvis.pkl'
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source_sub + '_cst_R_tvis_sff_in_ext.pkl'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_R_tvis_ext.pkl'
            
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_R_Lauren.txt'
            '''
            
            source = load_tract(s_file,s_cst_sff_idx)
            target = load_tract(t_file,t_cst_ext_idx)
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                print 'Saving 1-NN tract based: ', out_file
                save_pickle(out_file, map_all)
            
            s_cst = load_tract(s_file, s_cst_idx)
            t_cst = load_tract(t_file, t_cst_idx)
            t_cst_ext = load_tract(t_file, t_cst_ext_idx)
            
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
            jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            print "\t\t", target_ids[t_id], "\t", len(source), "\t", len(target),"\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
            #print "Before mapping: ", jac0, bfn0
            #print "After mapping: ", jac1, bfn1
           
            if vis:
               #visualize target cst and mapped source cst - yellow and blue
                ren = fvtk.ren()                
                ren = visualize_tract(ren, s_cst, fvtk.yellow)
                ren = visualize_tract(ren, t_cst, fvtk.blue)
                ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                fvtk.show(ren)
 

