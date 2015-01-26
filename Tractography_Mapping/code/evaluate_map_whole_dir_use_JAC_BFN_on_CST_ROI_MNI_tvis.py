# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:18:02 2014

@author: bao

Evaluate the mapping using two measurements: Jaccard and BFN indices
cst_sff_in_ext to cst_ext (50 prototypes selected by SFF)
all are in MNI space, 
segmentation: ROIs
tractography: reconstructed from trackvis software not dipy
result of the mapping as in 
     - tractography_mapping_cst_sff_in_ext_2_cst_ext.py
     - run_tractography_mapping_cst_sff_in_ext_2_cst_ext.py
note that the mapp saves the index of fiber in the source tract
with each fiber index i in source, mapp[i] is the index of fiber in target
      """
from common_functions import load_tract, Jac_BFN, Jac_BFN2, vol_corr_notcorr, TP_FP_TP_FN, vol_tracts, Jac_BFN_1, visualize_tract, Shannon_entropy
from dipy.io.pickles import load_pickle, save_pickle
from dipy.viz import fvtk
import numpy as np


def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]
        

#for CST_ROI_L
source_ids =[204]# [212, 202, 204, 209]#[202]#
target_ids = [202]#[212, 202, 204, 209]#[204]#


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]# [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''


vol_dims = [182,218,182]
vis = True#False#True#False#True#False

"""
#-------------------------------------------------------------------
#            Check the result of annealing is the same as 1NN or not
#-------------------------------------------------------------------
source_ids = [202]#[212, 202, 204, 209]
target_ids = [212]#[212, 202, 204, 209]


'''
#for CST_ROI_R
source_ids = [206, 204, 212, 205]# [206, 204, 212, 205]
target_ids = [206, 204, 212, 205]
'''

anneal = [1000]#[100, 200, 400, 600, 800, 1000]

for a_id in np.arange(len(anneal)):
    print "==================================================================="
    print "Anneal : ", anneal[a_id]
    #print "\t\t Target \t not_map_Jac \t not_map_BFN \t map_Jac \t map_BFN \t 1NN_diff_ANN \t 1NN_same_ANN \t 1NN_intersec_target \t ANN_intersec_target" 
    #print "\t\t Target \t len s_cst \t 1NN_diff_ANN \t 1NN_same_ANN \t 1NN_intersec_target \t ANN_intersec_target" 
    print "\t\t Target \t len s_cst \t 1NN_entropy \t ANN_entropy"
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
                map_file_ann = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                map_file_1nn = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                '''
                
                #Right
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                
                map_file_ann = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                map_file_1nn = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_R_ann_100_MNI.txt'
                '''
                
                 
                s_cst = load_tract(s_file, s_cst_idx)
                t_cst = load_tract(t_file, t_cst_idx)
                t_cst_ext = load_tract(t_file, t_cst_ext_idx)
                map_all_ann = load_pickle(map_file_ann)
                map_all_1nn = load_pickle(map_file_1nn)
                
                cst_len = len(s_cst)
                mapped_ann = map_all_ann[:cst_len]
                mapped_1nn = map_all_1nn[:cst_len]
                
                
                # this is for printing the number of mapping is the same as 1-NN/ target or not
                same = 0
                diff = 0
                for k in np.arange(cst_len):
                    if mapped_ann[k]!=mapped_1nn[k]:
                        diff = diff + 1
                    else:
                        same = same + 1
                
                mapped_ann = np.array(mapped_ann, dtype = int)                                
                set_mapped_ann = set(mapped_ann)
                
                mapped_1nn = np.array(mapped_1nn, dtype = int)                                
                set_mapped_1nn = set(mapped_1nn)

                t_idx = np.arange(len(t_cst))
                set_t_cst_idx = set(t_idx)
                
                #print set_mapped_ann
                #print set_mapped_1nn
                #print set_t_cst_idx
                
                
                #ren = fvtk.ren()                                
                #ren = visualize_tract(ren, t_cst, fvtk.colors.blue)                
                #ren = visualize_tract(ren, t_cst_ext[:len(t_cst)], fvtk.colors.yellow)
                #fvtk.show(ren)
                    
                t_1nn = set_mapped_1nn & set_t_cst_idx
                t_ann = set_mapped_ann & set_t_cst_idx
                                
                mapped_s_cst = [t_cst_ext[idx] for idx in mapped_ann]
                
                #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
                #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
                
                #print "\t\t", target_ids[t_id],"\t",  cst_len, "\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1, "\t",  same, "\t", diff
                print "\t\t", target_ids[t_id],"\t",  cst_len, "\t",  diff, "\t", same,  "\t", len(t_1nn), "\t", len(t_ann)
                
                if vis:
                   
                    mapped_t_1nn_tract = [t_cst_ext[l] for l in t_1nn]
                    mapped_t_ann_tract = [t_cst_ext[j] for j in t_ann]
                    
                    mapped_1nn_tract = [t_cst_ext[l] for l in mapped_1nn]
                    mapped_ann_tract = [t_cst_ext[j] for j in mapped_ann]
                    jac0, bfn0 = Jac_BFN(mapped_1nn_tract, t_cst, vol_dims, disp=True)
                    jac1, bfn1 = Jac_BFN(mapped_ann_tract, t_cst, vol_dims, disp=True)
                    print jac0, bfn0
                    print jac1, bfn1
                    
                    ren = fvtk.ren()                
                    
                    ren = visualize_tract(ren, mapped_t_ann_tract, fvtk.colors.blue)
                    ren = visualize_tract(ren, mapped_t_1nn_tract, fvtk.colors.yellow)
                    #ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                    
                    #ren = visualize_tract(ren, s_cst, fvtk.yellow)
                    #ren = visualize_tract(ren, t_cst, fvtk.blue)
                    #ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                    fvtk.show(ren)
                '''
                                
                # this is for printing the entropy of mapping
                mapped_s_cst_1nn = [t_cst_ext[idx] for idx in mapped_1nn]
                mapped_s_cst_ann = [t_cst_ext[idx] for idx in mapped_ann]
                entropy_1nn = Shannon_entropy(mapped_s_cst_1nn)
                entropy_ann = Shannon_entropy(mapped_s_cst_ann)
                entropy_t = Shannon_entropy(t_cst)
                print "\t", target_ids[t_id], "\t",  cst_len,"\t", entropy_1nn, "\t", entropy_ann, "\t", entropy_t
                '''
                
                
"""
#-------------------------------------------------------------------
#            Annealing
#-------------------------------------------------------------------
anneal = [800]#[100, 200, 400, 600, 800, 1000]
print 'mapping'
for a_id in np.arange(len(anneal)):
    print "==================================================================="
    print "Anneal : ", anneal[a_id]
    #print "\t\t Target \t not_map_Jac \t not_map_BFN \t map_Jac \t map_BFN" 
    print "\t\t Target \t vol correct(flirt) \t vol not correct (flirt) \t vol correct(mapping) \t vol not correct (mapping)" 
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
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_L_ann_' + str(anneal[a_id]) + '_MNI.txt'
                '''
                
                #Right
                s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
                t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
                s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
                t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'
                t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_R_tvis_ext.pkl'
                #map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_best_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                map_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/results/result_cst_sff_in_ext_2_cst_ext/50_SFF_MNI/map_1nn_' + source + '_' + target + '_cst_R_ann_' + str(anneal[a_id]) + '_MNI.txt'
                '''               
                
                s_cst = load_tract(s_file, s_cst_idx)
                t_cst = load_tract(t_file, t_cst_idx)
                t_cst_ext = load_tract(t_file, t_cst_ext_idx)
                map_all = load_pickle(map_file)
                
                #this is only for mapping - to conver voxel size from 1,1,1 to 2,2,2
                s_cst = .5 * s_cst
                t_cst = .5 * t_cst                
                t_cst_ext = .5 * t_cst_ext
                #vol_dims = [182*2,218*2,182*2]
                
                # end of this is only for mapping - to conver voxel size from 1,1,1 to 2,2,2
                                
                
                
                cst_len = len(s_cst)
                mapped = map_all[:cst_len]
                
                mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
                
                #jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
                #jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)
                #jac0, bfn0 = Jac_BFN2(s_cst, t_cst, vol_dims, disp=False)
                #jac1, bfn1 = Jac_BFN2(mapped_s_cst, t_cst, vol_dims, disp=False)
                #print "\t\t", target_ids[t_id], "\t", jac0,"\t",  bfn0, "\t", jac1,"\t",  bfn1
                
                                
                cor0, ncor0 = vol_corr_notcorr(s_cst, t_cst, vol_dims, disp=False)
                cor1, ncor1 = vol_corr_notcorr(mapped_s_cst, t_cst, vol_dims, disp=False)                

                vl_s, vl_t = vol_tracts(s_cst, t_cst, vol_dims, disp=False)
                
                print "\t\t", target_ids[t_id], "\t", cor0,"\t",  ncor0, "\t", cor1,"\t",  ncor1, "\t", vl_s,"\t",  vl_t
                
                
                if vis:
                    s_file_native = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'
                    t_file_native = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis.trk'
                    s_cst_native = load_tract(s_file_native, s_cst_idx)
                    t_cst_native = load_tract(t_file_native, t_cst_idx)
                    t_cst_ext_native = load_tract(t_file_native, t_cst_ext_idx)
                    TP_mapped,FP_mapped,TP_source, FN_source = TP_FP_TP_FN(s_cst_native, t_cst_native, t_cst_ext_native, mapped)
                    from common_functions import show_both_bundles
                    
                    show_both_bundles([TP_mapped, FP_mapped],
                          #colors=[fvtk.colors.orange, fvtk.colors.red],
                          colors=[fvtk.colors.red, fvtk.colors.green],
                          show=True,
                          fname='Flirt_reg_1NN_' + source + '_'+ target + '_L_TP_red_FP_green_in_mapped_show_in_native.png')
                    
                    
                    show_both_bundles([TP_source, FN_source],
                          colors=[fvtk.colors.blue, fvtk.colors.yellow],
                          show=True,
                          fname='Flirt_reg_1NN_' + source + '_'+ target + '_L_TP_blue_FN_yellow_in_source_show_in_native.png')
    
                    import matplotlib.pyplot as plt
                    plt.xlabel('streamline id')
                    plt.ylabel('frequency')
                    plt.title('Flirt, voxel based registration method')
                    plt.axis([0,len(t_cst_ext)/3,0,50])
                    n, bins, patches = plt.hist(mapped, bins = len(t_cst_ext), range=[0,len(t_cst_ext)])
                    plt.show()
                    plt.savefig('Histogram_Flirt_reg_1NN_' + source + '_'+ target + '_L.png')   
                    '''
                    from common_functions import show_both_bundles
                    show_both_bundles([mapped_s_cst, t_cst],                      
                      colors=[fvtk.colors.green, fvtk.colors.blue],
                      show=True,
                      fname='Flirt_reg_only_204_202_L.png')
                    '''
                    '''
                    show_both_bundles([s_cst, t_cst],
                      #colors=[fvtk.colors.orange, fvtk.colors.red],
                      colors=[fvtk.colors.green, fvtk.colors.blue],
                      show=True,
                      fname='Flirt_reg_only_204_202_L.png')
                
                    show_both_bundles([t_cst, mapped_s_cst],
                      colors=[fvtk.colors.blue, fvtk.colors.red],
                      show=True,
                      fname='Flirt_reg_1NN_204_202_L.png')
                     '''
                    """  
                    #visualize target cst and mapped source cst - yellow and blue
                    ren = fvtk.ren()                
                    ren = visualize_tract(ren, s_cst, fvtk.yellow)
                    ren = visualize_tract(ren, t_cst, fvtk.blue)
                    ren = visualize_tract(ren, mapped_s_cst, fvtk.red)
                    fvtk.show(ren)
                    """
                
                #import matplotlib.pyplot as plt
                #n, bins, patches = plt.hist(mapped, bins = len(t_cst_ext), range=[0,len(t_cst_ext)])
                #l = plt.plot(bins, y, 'r--', linewidth=1)
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
            nn = 5#10
            
            s_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.trk'
            t_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + target + '_tracks_dti_tvis_linear.trk'            
            
            s_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_L_tvis.pkl'
            t_cst_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_L_tvis.pkl'
            
            s_cst_sff_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_L_tvis_sff_in_ext.pkl'            
            t_cst_ext_idx = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/50_SFF_in_ext/ROI_seg_native/' + target + '_cst_L_tvis_ext.pkl'
            
            
            map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_prob_map/prob_map_prob_map_' + source + '_' + target + '_cst_L_MNI_full_full' + '_sparse_density_' + str(nn) + '_neighbors.txt'                            
            
            
            
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
            
            
            #load cai map va chon cai co prob cao nhat
            
            #-------------------------------------------------------
            #only for saving the higest prob map            
            #cst_sff_len = len(s_cst_sff)
            
            #map_idxs_all = [map_all[i].argsort()[-1] for i in np.arange(cst_sff_len)]
            
            #mapped_all = [cs_idxs[i][map_idxs_all[i]] for i in np.arange(cst_sff_len)]
            
            #save_pickle(map_file+'choose_highest.txt',mapped_all)
            
            #stop

            # end only for saving the higest prob map            
            #-------------------------------------------------------
            
            
            cst_len = len(s_cst)
            map_tmp = map_all[:cst_len]
            
            map_idxs = [map_tmp[i].argsort()[-1] for i in np.arange(cst_len)]
            
            mapped = [cs_idxs[i][map_idxs[i]] for i in np.arange(cst_len)]
            
            #print map_tmp[:10], map_idxs[:10], cs_idxs[:10], mapped[:10]            
            
            #stop
            
            mapped_s_cst = [t_cst_ext[idx] for idx in mapped]
            
            jac0, bfn0 = Jac_BFN(s_cst, t_cst, vol_dims, disp=False)
            jac1, bfn1 = Jac_BFN(mapped_s_cst, t_cst, vol_dims, disp=False)#True)#True)#False)
            
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