# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:58:36 2014

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:22:21 2014

@author: bao
Evaluate the mapping
Local: cst to cst
Global:- cst to cst_ext
       - cst_sff to cst_ext_sff
"""
from common_functions import minus
from dipy.io.pickles import load_pickle
import numpy as np
import argparse

def quantity_in(tract1, tract2):
    '''
    return the quantity of the fiber appear in both tracts
    '''
    return len(overlap(tract1,tract2))


'''
#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Evaluation the tractography mapping for the whole directory",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.1')
                                 
parser.add_argument(
                    'inputSourceCSTIndex',
                    help='The file name of source CST index')                    
parser.add_argument(
                    'inputTargetCSTIndex',
                    help='The file name of target CST index')
parser.add_argument(
                    'inputMapping',
                    help='The file name of the mapping')
                    

args = parser.parse_args()

print "=========================="
print "Source CST index:       ", args.inputSourceCSTIndex
print "Target CST index:       ", args.inputTargetCSTIndex
print "Mapping:    ", args.inputMapping
print "=========================="
'''

import numpy as np

source_ids = [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 212]

target_ids = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212]

for s_id in np.arange(len(source_ids)):
    print source_ids[s_id]
    print 'Target id \t Len source CST \t Len target CST \t number of fiber inside target \t number of fiber outside target'
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            s_ind    = '/home/bao/tiensy/Tractography_Mapping/code/data/' + source + '_corticospinal_L_3M.pkl'
            t_ind    = '/home/bao/tiensy/Tractography_Mapping/code/data/' + target + '_corticospinal_L_3M.pkl'
            map_file = '/home/bao/tiensy/Tractography_Mapping/code/results/result_cst_sff_cst_ext_sff/50_SFF/map_best_' + source + '_'+ target + '_cst_L_ann_100.txt'
            
            s_cst = load_pickle(s_ind)
            t_cst = load_pickle(t_ind)
            map_all = load_pickle(map_file)
            
            cst_len = len(s_cst)
            mapp = map_all[:cst_len]
            
            t_id_ar =  np.arange(len(t_cst))
            
            
            t_cst_minus_map = minus(t_id_ar, mapp)
            num_in = len(t_cst) - len(t_cst_minus_map)
            
            map_minus_t_cst = minus(mapp,t_id_ar)
            num_out = len(map_minus_t_cst)
            num_replicate = cst_len - num_out
            
            print '\t',target, '\t', cst_len, '\t', len(t_cst), '\t', num_out , '\t', num_replicate
            #print 'Len of cst - source and target, and mapping', cst_len, len(t_cst), len(map_all), len(mapp)
            #print 'Number of fiber inside target', num_in
            #print 'Number of fiber replicating outside target', num_out
            #print 'Number of fiber replicating in the target', num_replicate