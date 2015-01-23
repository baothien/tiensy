# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:07:51 2014

@author: bao
"""
from common_functions import load_tract, load_whole_tract, Jac_BFN,Jac_BFN2, vol_corr_notcorr, visualize_tract
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
source_ids = [202]#[212, 202, 204, 209]#[212, 202, 204, 209]
target_ids = [204]#[212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]# [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''

vol_dims = [128,128,80]
vis = True#False#True
save = False#True#False#True


def mapping_nn(tractography1, tractography2):
    #print 'Compute the 1-nn from source to target'
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)    
    return mapping12_coregistration_1nn


#------------------------------------------------------------
#          1-NN 
#------------------------------------------------------------                 
print "The coregistration+1NN gives a mapping12 with the following measurement:"
#print "\t\t Target \t Lauren_Jac \t Lauren_BFN \t 1NN_Lauren_Jac \t 1_NN_Lauren_BFN" 
print "\t\t Target \t vol correct(Lauren_pairwise) \t vol not correct (Lauren_pairwise) \t vol correct(Lauren_pairwise_1NN) \t vol not correct (Lauren_pairwise_1NN)" 
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source_sub = str(source_ids[s_id])
            target_sub = str(target_ids[t_id])
       
            #---------------------------------------------------------------------------------------
            #This is for computing cor and not-cor number of voxel with pairwise registration using Lauren method
            # used for registering cst_ext to cst_ext
            
            
           
            #Left            
            
            #s_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + source_sub + '_cst_L_tvis_ext_reg.trk'
            #t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + target_sub + '_cst_L_tvis_ext_reg.trk'
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CSText2CSText/Lauren_pair_CSText2CSText_f300_l75_1NN/map_1nn_pairwise_reg_CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_Lauren_f300_l75.txt'
            
            s_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + source_sub + '_cst_L_tvis_ext_reg.trk'
            t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + target_sub + '_cst_L_tvis_ext_reg.trk'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CSText2CSText/Lauren_pair_CSText2CSText_f100_l25_1NN/map_1nn_pairwise_reg_CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_Lauren_f100_l25.txt'
            
            s_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_L_tvis.pkl'            
            t_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            '''
            
            
            #Right                           
            #s_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + source_sub + '_cst_R_tvis_ext_reg.trk'
            #t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + target_sub + '_cst_R_tvis_ext_reg.trk'
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CSText2CSText/Lauren_pair_CSText2CSText_f300_l75_1NN/map_1nn_pairwise_reg_CST_R_ext_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_Lauren_f300_l75.txt'
            
            s_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + source_sub + '_cst_R_tvis_ext_reg.trk'
            t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg_cstext2cstext/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + target_sub + '_cst_R_tvis_ext_reg.trk'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CSText2CSText/Lauren_pair_CSText2CSText_f100_l25_1NN/map_1nn_pairwise_reg_CST_R_ext_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_Lauren_f100_l25.txt'
            
            s_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_R_tvis.pkl'            
            t_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            '''
            
            
            source_ext = load_whole_tract(s_cst_ext_file)
            target_ext = load_whole_tract(t_cst_ext_file)
            
            s_cst_idx = load_pickle(s_cst_idx_file) 
            source = source_ext[:len(s_cst_idx)]            
            
            t_cst_idx = load_pickle(t_cst_idx_file)            
            target = target_ext[:len(t_cst_idx)]   
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target_ext
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                #print 'Saving 1-NN tract based: ', out_file
                save_pickle(out_file, map_all)
            
            s_cst = source
            t_cst = target
            t_cst_ext = target_ext
            
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #jac0, bfn0 = Jac_BFN2(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #print "\t\t", target_ids[t_id], "\t", len(source), "\t", len(target),"\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
            #print "Before mapping: ", jac0, bfn0
            #print "After mapping: ", jac1, bfn1
            
            
            cor0, ncor0 = vol_corr_notcorr(s_cst, t_cst, vol_dims, disp=False)
            cor1, ncor1 = vol_corr_notcorr(mapped_s_cst, t_cst, vol_dims, disp=False)                
                
            print "\t\t", target_ids[t_id], "\t", cor0,"\t",  ncor0, "\t", cor1,"\t",  ncor1
               
            
           
            if vis:
                from common_functions import show_both_bundles
                show_both_bundles([s_cst, t_cst],
                      #colors=[fvtk.colors.orange, fvtk.colors.red],
                      colors=[fvtk.colors.green, fvtk.colors.blue],
                      show=True,
                      fname='Lauren_reg_only_202_204_L.png')
                
                show_both_bundles([t_cst, mapped_s_cst],
                      colors=[fvtk.colors.blue, fvtk.colors.red],
                      show=True,
                      fname='Lauren_reg_1NN_202_204_L.png')
                """
                #visualize target cst and mapped source cst - yellow and blue
                ren = fvtk.ren() 
                ren.SetBackground(255,255,255)
                #ren = visualize_tract(ren, s_cst, fvtk.yellow)
                #ren = visualize_tract(ren, t_cst, fvtk.blue)
                ren = visualize_tract(ren, s_cst, fvtk.colors.yellow)
                ren = visualize_tract(ren, t_cst, fvtk.colors.blue)
                fvtk.show(ren)
                #ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                fvtk.clear(ren)
                ren = visualize_tract(ren, t_cst, fvtk.colors.blue)
                ren = visualize_tract(ren, mapped_s_cst, fvtk.colors.red)
                fvtk.show(ren)
                """
            
            
            #----------------------------------------------------------------------------------------            
            
            """
            #---------------------------------------------------------------------------------------
            #This is for computing cor and not-cor number of voxel with pairwise registration using Lauren method
            # used for registering cst only to cst_ext
           
            
            '''
            #Left            
            s_cst_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + source_sub + '_corticospinal_L_tvis_reg.trk'
            t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + target_sub + '_cst_L_tvis_ext_reg.trk'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CST2CSText/Lauren_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_Lauren.txt'
            
            #s_cst_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + source_sub + '_corticospinal_L_tvis_reg.trk'
            #t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Left/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + target_sub + '_cst_L_tvis_ext_reg.trk'
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CST2CSText/Lauren_pair_CST2CSText_f100_l25_1NN/map_1nn_pairwise_reg_CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_Lauren_f100_l25.txt'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            
            '''
            
            
            #Right            
            #s_cst_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + source_sub + '_corticospinal_R_tvis_reg.trk'
            #t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg/iteration_4/' + target_sub + '_cst_R_tvis_ext_reg.trk'
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CST2CSText/Lauren_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_Lauren.txt'
            
            s_cst_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + source_sub + '_corticospinal_R_tvis_reg.trk'
            t_cst_ext_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/pairwise_reg/CST_ROI_trkvis_Right/' + source_sub + '_' + target_sub + '/out_reg_f100_l25/iteration_4/' + target_sub + '_cst_R_tvis_ext_reg.trk'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Lauren_pair_CST2CSText/Lauren_pair_CST2CSText_f100_l25_1NN/map_1nn_pairwise_reg_CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_Lauren_f100_l25.txt'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            
            
            
            source = load_whole_tract(s_cst_file)
            target_ext = load_whole_tract(t_cst_ext_file)
            
            t_cst_idx = load_pickle(t_cst_idx)            
            target = target_ext[:len(t_cst_idx)]   
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target_ext
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                #print 'Saving 1-NN tract based: ', out_file
                save_pickle(out_file, map_all)
            
            s_cst = source
            t_cst = target
            t_cst_ext = target_ext
            
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #jac0, bfn0 = Jac_BFN2(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #print "\t\t", target_ids[t_id], "\t", len(source), "\t", len(target),"\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
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
            
            
            #----------------------------------------------------------------------------------------            
            """
           
            """
            #---------------------------------------------------------------------------------------
            #This is for computing JAC and BFN of when group registration using Lauren method
            
            
            #indir = 'out_registered_f300_l75'            
            indir = 'out_registered_f750_l60'            
            s_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + source_sub + '_tracks_dti_tvis_reg.trk'
            t_file = '/home/bao/tiensy/Lauren_registration/data_compare_mapping/' + indir + '/iteration_4/' + target_sub + '_tracks_dti_tvis_reg.trk'
            
            
            #Left            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_L_tvis.pkl'
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source_sub + '_cst_L_tvis_sff_in_ext.pkl'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_L_tvis_ext.pkl'
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f300_l75/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_L_Lauren.txt'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f750_l60/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_L_Lauren.txt'
            
            '''            
            #Right
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_R_tvis.pkl'
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source_sub + '_cst_R_tvis_sff_in_ext.pkl'
            
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_R_tvis_ext.pkl'
            
            #out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f300_l75/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_R_Lauren.txt'
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_Lauren_1NN/Lauren_group_f750_l60/map_1nn_' + source_sub + '_' + target_sub + '_cst_sff_in_ext_R_Lauren.txt'
            '''
            
            
            source = load_tract(s_file,s_cst_sff_idx)
            target = load_tract(t_file,t_cst_ext_idx)
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                #print 'Saving 1-NN tract based: ', out_file
                save_pickle(out_file, map_all)
            
            s_cst = source[:-50]#remove 50 SFF prototypes
            t_cst_ext = target
            t_cst = load_tract(t_file,t_cst_idx)
            
            
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #jac0, bfn0 = Jac_BFN2(s_cst, t_cst, vol_dims, disp=False)            
            #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)
                                  
            #print "\t\t", target_ids[t_id], "\t", len(source), "\t", len(target),"\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
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
            


            #---------------------------------------------------------------------------------------
            
            """  
            
           
            
            
