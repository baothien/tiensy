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



#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Evaluation the tractography mapping",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.0')
                                 
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


s_ind = args.inputSourceCSTIndex
t_ind = args.inputTargetCSTIndex
map_file = args.inputMapping


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

print 'Len of cst - source and target, and mapping', cst_len, len(t_cst), len(map_all), len(mapp)
print 'Number of fiber inside target', num_in
print 'Number of fiber replicating outside target', num_out
print 'Number of fiber replicating in the target', num_replicate

