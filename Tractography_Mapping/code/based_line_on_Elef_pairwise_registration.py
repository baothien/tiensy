# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:10:35 2015

@author: bao
"""
from common_functions import load_tract, load_whole_tract,vol_corr_notcorr, TP_FP_TP_FN, Jac_BFN, Jac_BFN_1, Jac_BFN2, visualize_tract
from dipy.io.pickles import load_pickle, save_pickle
from dipy.viz import fvtk
import numpy as np
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.tracking.metrics import length

def mapping_nn(tractography1, tractography2):
    #print 'Compute the 1-nn from source to target'
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)    
    return mapping12_coregistration_1nn

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
        

#for CST_ROI_L
source_ids = [209]#[212, 202, 204, 209]#[212, 202, 204, 209]
target_ids = [212]#[212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]# [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''

#vol_dims = [182,218,182]#MNI, voxel = [1,1,1]
vol_dims =[128,128,80]# [128,128,70] #native, voxel = [2,2,2]
vis = True#False#True
save = False#True

#------------------------------------------------------------
#          1-NN 
#------------------------------------------------------------  
#print 'Left 50 random'               
print "The coregistration+1NN gives a mapping12 with the following measurement:"
print "\t\t Target \t Elef_cor \t Elef_not_cor \t Elef_1NN_cor \t Elef_1NN_not_cor" 
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source_sub = str(source_ids[s_id])
            target_sub = str(target_ids[t_id])
            
            #---------------------------------------------------------------------------------------
            #This is for computing cor and not-cor number of voxel with pairwise registration using Eleftherios method
            # used for registering cst_ext to cst_ext
            
            
            #Left            
            s_cst_ext_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.dpy'#100.dpy 200.dpy
            out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/Elef_pair_CSText2CSText_1NN/map_1nn_pairwise_reg_CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.txt'

            s_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_L_tvis.pkl'
            t_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'                        
            t_cst_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            t_cst_ext_idx_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_L_tvis_ext.pkl'
            
            #s_cst_ext_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.dpy'#100.dpy 200.dpy
            #out_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/Elef_pair_CSText2CSText_1NN/map_1nn_pairwise_reg_CST_L_ext_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.txt'

            #s_cst_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_L_tvis.pkl'
            #t_file  = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'                        
            #t_cst_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            #t_cst_ext_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_L_tvis_ext.pkl'

            
            '''
            #Right            
            s_cst_ext_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/CST_R_ext_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.dpy'
            out_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CSText2CSText/Elef_pair_CSText2CSText_1NN/map_1nn_pairwise_reg_CST_R_ext_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_50_fiberpoint_40.txt'

            s_cst_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source_sub + '_corticospinal_R_tvis.pkl'
            t_file  = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'                        
            t_cst_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            t_cst_ext_idx_file = '/home/nilab/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_R_tvis_ext.pkl'
            '''
            
            
            s_cst_idx = load_pickle(s_cst_idx_file)  
            
            source_ext = load_whole_tract(s_cst_ext_file)
            source = source_ext[:len(s_cst_idx)]
            
            target = load_tract(t_file, t_cst_ext_idx_file)
            
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                #print 'Saving 1-NN Eleftherios tract based method: ', out_file
                save_pickle(out_file, map_all)
                
            s_cst = source
            t_cst = load_tract(t_file, t_cst_idx_file)           
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [target[idx] for idx in mapped]
            
            new_s_cst = []
            for k in np.arange(cst_len):
                new_s_cst.append(np.array(s_cst[k], dtype = np.float32))
                
           
            #jac0, bfn0 = Jac_BFN(new_s_cst, t_cst, vol_dims, disp=False)
            #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #jac0, bfn0 = Jac_BFN2(new_s_cst, t_cst, vol_dims, disp=False)
            #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)

            

            #cor0, ncor0 = vol_corr_notcorr(new_s_cst, t_cst, vol_dims, disp=False)
            #cor1, ncor1 = vol_corr_notcorr(mapped_s_cst, t_cst, vol_dims, disp=False)                
                
            #print "\t\t", target_ids[t_id], "\t", cor0,"\t",  ncor0, "\t", cor1,"\t",  ncor1
               
            
           
            if vis:
                s_file_native = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source_sub + '_tracks_dti_tvis.trk'
                t_file_native = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'
                s_cst_native = load_tract(s_file_native, s_cst_idx_file)
                t_cst_native = load_tract(t_file_native, t_cst_idx_file)
                t_cst_ext_native = load_tract(t_file_native, t_cst_ext_idx_file)
                TP_mapped,FP_mapped,TP_source, FN_source = TP_FP_TP_FN(s_cst_native, t_cst_native, t_cst_ext_native, mapped)
                
                #TP_mapped,FP_mapped,TP_source, FN_source = TP_FP_TP_FN(s_cst, t_cst, target, mapped)
                from common_functions import show_both_bundles
                       
                show_both_bundles([TP_mapped, FP_mapped],
                      #colors=[fvtk.colors.orange, fvtk.colors.red],
                      colors=[fvtk.colors.red, fvtk.colors.green],
                      show=True,
                      fname='Elef_reg_1NN_' + source_sub + '_'+ target_sub + '_L_TP_red_FP_green_in_mapped_show_in_native.png')
                
                
                show_both_bundles([TP_source, FN_source],
                      colors=[fvtk.colors.blue, fvtk.colors.yellow],
                      show=True,
                      fname='Elef_reg_1NN_' + source_sub + '_'+ target_sub + '_L_TP_blue_FN_yellow_in_source_show_in_native.png')

                import matplotlib.pyplot as plt
                plt.xlabel('streamline id')
                plt.ylabel('Frequency')
                plt.title('Eleftherios streamline based registration method')
                plt.axis([0,len(target)/3,0,50])
                n, bins, patches = plt.hist(mapped, bins = len(target), range=[0,len(target)])
                plt.show()
                #plt.savefig('Histogram_Elef_reg_1NN_' + source_sub + '_'+ target_sub + '_L.png')   
                
                '''      
                from common_functions import show_both_bundles
                show_both_bundles([s_cst, t_cst],
                      #colors=[fvtk.colors.orange, fvtk.colors.red],
                      colors=[fvtk.colors.green, fvtk.colors.blue],
                      show=True,
                      fname='Elef_reg_only_204_202_L.png')
                
                show_both_bundles([t_cst, mapped_s_cst],
                      colors=[fvtk.colors.blue, fvtk.colors.red],
                      show=True,
                      fname='Elef_reg_1NN_204_202_L.png')
                '''
                """
                #visualize target cst and mapped source cst - yellow and blue
                ren = fvtk.ren()                
                ren = visualize_tract(ren, s_cst, fvtk.yellow)
                ren = visualize_tract(ren, t_cst, fvtk.blue)
                ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                fvtk.show(ren)
                """
                
                                 
                
                
            """            
            '''
            #Left            
            s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef.dpy'
            out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef.txt'
            
            #s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_100.dpy'
            #out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_100.txt'
                        
            #s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_200.dpy'
            #out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_L_' + source_sub + '_aligned_to_CST_L_ext_' + target_sub + '_elef_rand_200.txt'

            t_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'                        
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_L_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_L_tvis_ext.pkl'
            

            
            '''
            #Right            
            #s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef.dpy'
            #out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef.txt'

            #s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_100.dpy'
            #out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_100.txt'
            
            s_cst_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_200.dpy'
            out_file = out_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/Elef_pair_CST2CSText/Elef_pair_CST2CSText_1NN/map_1nn_pairwise_reg_CST_R_' + source_sub + '_aligned_to_CST_R_ext_' + target_sub + '_elef_rand_200.txt'

            t_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target_sub + '_tracks_dti_tvis.trk'                        
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target_sub + '_corticospinal_R_tvis.pkl'
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target_sub + '_cst_R_tvis_ext.pkl'
            
            
            
              
            source = load_whole_tract(s_cst_file)
            target = load_tract(t_file, t_cst_ext_idx)
            
            
            #print len(source), len(target)
            
            tractography1 = source
            tractography2 = target
            
                       
            map_all = mapping_nn(tractography1, tractography2)
            
            if save:            
                #print 'Saving 1-NN Eleftherios tract based method: ', out_file
                save_pickle(out_file, map_all)
                
            s_cst = source
            t_cst = load_tract(t_file, t_cst_idx)           
            
            cst_len = len(s_cst)
            mapped = map_all[:cst_len]
            
            mapped_s_cst = [target[idx] for idx in mapped]
            
            new_s_cst = []
            for k in np.arange(cst_len):
                new_s_cst.append(np.array(s_cst[k], dtype = np.float32))
                
           
            #jac0, bfn0 = Jac_BFN(new_s_cst, t_cst, vol_dims, disp=False)
            #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
            
            #jac0, bfn0 = Jac_BFN2(new_s_cst, t_cst, vol_dims, disp=False)
            #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)

            

            cor0, ncor0 = vol_corr_notcorr(new_s_cst, t_cst, vol_dims, disp=False)
            cor1, ncor1 = vol_corr_notcorr(mapped_s_cst, t_cst, vol_dims, disp=False)                
                
            print "\t\t", target_ids[t_id], "\t", cor0,"\t",  ncor0, "\t", cor1,"\t",  ncor1
               
            
           
            if vis:
               #visualize target cst and mapped source cst - yellow and blue
                ren = fvtk.ren()                
                ren = visualize_tract(ren, s_cst, fvtk.yellow)
                ren = visualize_tract(ren, t_cst, fvtk.blue)
                ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                fvtk.show(ren)
                
            """