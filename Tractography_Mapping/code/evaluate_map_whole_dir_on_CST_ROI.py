# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:40:29 2014

@author: bao


Evaluate the mapping
Local: cst to cst
Global:- cst to cst_ext
       - cst_sff to cst_ext_sff
"""
from common_functions import minus, overlap
from dipy.io.pickles import load_pickle
import numpy as np
import argparse

def quantity_in(tract1, tract2):
    '''
    return the quantity of the fiber appear in both tracts
    '''
    return len(overlap(tract1,tract2))



import numpy as np

#for Left CST_ROI
source_ids =[202, 204, 206]#03]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 213]
target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210,213]

#for Right CST_ROI
#source_ids =[203, 207, 210]#, 213]#[201, 202, 203, 204, 205, 207,208, 210, 212, 213]
#target_ids = [201, 202, 203, 204, 205, 207,208, 210, 212, 213]

for s_id in np.arange(len(source_ids)):
    print source_ids[s_id]
    print 'Target id \t Len source CST \t Len target CST \t number of fiber inside target \t number of fiber outside target'
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            
            
            '''
            native space CST_SFF_2_CST_EXT_SFF
            '''
                       
            #Left
            s_ind    = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + source + '_CST_ROI_L_3M.pkl'
            t_ind    = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + target + '_CST_ROI_L_3M.pkl'
            
            #map_file_nn = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_1nn_' + source + '_'+ target + '_cst_L_ann_100.txt'           
            #map_all = load_pickle(map_file_nn)            
            map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_best_' + source + '_'+ target + '_cst_L_ann_100.txt'
            map_all = load_pickle(map_file) 
            '''
            #Right
            
            s_ind    = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + source + '_CST_ROI_R_3M.pkl'
            t_ind    = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + target + '_CST_ROI_R_3M.pkl'
            
            #map_file_nn = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_1nn_' + source + '_'+ target + '_cst_R_ann_100.txt'           
            #map_all = load_pickle(map_file_nn)            
            map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff_ROI/50_SFF_native/map_best_' + source + '_'+ target + '_cst_R_ann_100.txt'
            map_all = load_pickle(map_file)  
            '''
            '''
            MNI space with 100 annealing iterations
            '''         
            
            
            '''
            MNI space with 1000 annealing iterations
            ''' 
           
           
           
            
            s_cst = load_pickle(s_ind)
            t_cst = load_pickle(t_ind)
            
            
            cst_len = len(s_cst)
            mapp = map_all[:cst_len]
            
            t_id_ar =  np.arange(len(t_cst))
            
            
            t_cst_minus_map = minus(t_id_ar, mapp)
            num_in = len(t_cst) - len(t_cst_minus_map)
            
            num_in_1 = quantity_in(t_id_ar, mapp)#, t_id_ar)
            
            map_minus_t_cst = minus(mapp,t_id_ar)
            num_out = len(map_minus_t_cst)
            num_replicate = cst_len - num_out
            
            print '\t',target, '\t', cst_len, '\t', len(t_cst), '\t', num_replicate, '\t', num_out #,'\t', num_in_1
            #print 'Len of cst - source and target, and mapping', cst_len, len(t_cst), len(map_all), len(mapp)
            #print 'Number of fiber inside target', num_in
            #print 'Number of fiber replicating outside target', num_out
            #print 'Number of fiber replicating in the target', num_replicate