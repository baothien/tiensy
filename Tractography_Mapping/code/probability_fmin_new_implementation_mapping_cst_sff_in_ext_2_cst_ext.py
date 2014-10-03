# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:57:49 2014

@author: bao
"""
import time
import numpy as np
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.io.pickles import save_pickle
from common_functions import cpu_time, load_tract, visualize_tract, plot_smooth
import matplotlib.pyplot as plt
import os
import argparse
from probability_fmin_new_implementation import probability_map_new_ipl

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Tractography mapping with probability - new implementation based on graph matching - Loss = A- P.B.P^T",
                                 epilog="Written by Bao Thien Nguyen, tbnguyen@fbk.eu.",
                                 version='1.0')

parser.add_argument(
                    'inputSourceTractography',
                    help='The file name of source whole-brain tractography as .dpy/.trk format.')
parser.add_argument(
                    'inputSourceCSTSFFIndex',
                    help='The file name of source CST plus SFF index')

parser.add_argument(
                    'inputTargetTractography',
                    help='The file name of target whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputTargetCSTIndex',
                    help='The file name of target CST index')
parser.add_argument(
                    'inputTargetCSTExtIndex',
                    help='The file name of target CST extension index')

parser.add_argument(
                    '-pr', action='store', dest='inputNumPrototypes', type=int,
                    help='The number of prototypes') 
                    
parser.add_argument(
                    'outputMap_prob',
                    help='The output file name of the probability mapping')                    

args = parser.parse_args()

print "=========================="
print "Source tractography:       ", args.inputSourceTractography
#print "Source CST plus SFF index:       ", args.inputSourceCSTSFFIndex
print "Target tractography:       ", args.inputTargetTractography
#print "Target CST index:       ", args.inputTargetCSTIndex
#print "Target CST extension index:       ", args.inputTargetCSTExtIndex
#print "Number of prototypes:      ", args.inputNumPrototypes 
print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTSFFIndex

t_file = args.inputTargetTractography
t_ind = args.inputTargetCSTExtIndex
t_cst = args.inputTargetCSTIndex

num_pro = args.inputNumPrototypes 

map_prob = args.outputMap_prob


vis = False

source_cst = load_tract(s_file,s_ind)

target_cst_ext = load_tract(t_file,t_ind)

print len(source_cst), len(target_cst_ext)

tractography1 = source_cst[:100]#[-num_pro:]
tractography2 = target_cst_ext[:200]#[:num_pro]
#tractography2 = target_cst_ext[:num_pro*2]

print "Source", len(tractography1)
print "Target", len(tractography2)


size1 = len(tractography1)
size2 = len(tractography2)

if vis:
    ren = fvtk.ren() 
    ren = visualize_tract(ren, tractography1, fvtk.yellow)
    ren = visualize_tract(ren, tractography2, fvtk.blue)
    fvtk.show(ren)
  
print 'Optimizing ...........................'    
mapp_best = probability_map_new_ipl(tractography1, tractography2, size1, size2)

