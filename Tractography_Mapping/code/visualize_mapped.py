# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:07:32 2014

@author: bao
visualize the mapping and the source for evaluation
"""

from common_functions import *
import argparse

def visualize_source_mappedsource(ren, tract1, tract2, mapping, color1=fvtk.red, color2=fvtk.blue): 
    for i in np.arange(len(tract1)):
        fvtk.add(ren, fvtk.line(tract1[i], color1, opacity=1.0))
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], color2, opacity=1.0))     
    return ren
      
def visualize_diff_color(ren, tract1, tract2, mapping):
    
    #c = fvtk.line(lines, fvtk.green)    
    #fvtk.add(r,c)
    colors = [fvtk.red,fvtk.green, fvtk.blue, fvtk.white,fvtk.yellow, fvtk.gray,fvtk.hot_pink]#fvtk.cyan,fvtk.dark_blue,fvtk.dark_green,fvtk.dark_red,fvtk.golden,
    for i in np.arange(len(tract1)):
        fvtk.add(ren, fvtk.line(tract1[i], colors[i % len(colors)], opacity=1.0))
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], colors[i % len(colors)], opacity=1.0))
     
    return ren

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Evaluation the tractography mapping using local assessment (mapping CST to CST)",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.0')

parser.add_argument(
                    'inputSourceTractography',
                    help='The file name of source whole-brain tractography as .dpy format.')
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
                    'inputTargetCSTExtSFFIndex',
                    help='The file name of target CST extension plus SFF index')
parser.add_argument(
                    'inputMap_best',
                    help='The output file name of the best mapping')                    
parser.add_argument(
                    'inputMap_1nn',
                    help='The output file name of the 1-NN mapping')
                    
parser.add_argument(
                    '-pr', action='store', dest='inputNumPrototypes', type=int,
                    help='The number of prototypes') 
                                        
args = parser.parse_args()

print "=========================="
print "Source tractography:       ", args.inputSourceTractography
print "Source CST plus SFF index:       ", args.inputSourceCSTSFFIndex
print "Target tractography:       ", args.inputTargetTractography
print "Target CST index:       ", args.inputTargetCSTIndex
print "Target CST extension plus SFF index:       ", args.inputTargetCSTExtSFFIndex
print "Number of prototypes:      ", args.inputNumPrototypes 
print "Map best file:      ", args.inputMap_best
print "Map 1_NN file:      ", args.inputMap_1nn

print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTSFFIndex

t_file = args.inputTargetTractography
t_ind = args.inputTargetCSTExtSFFIndex
t_cst = args.inputTargetCSTIndex

num_pro = args.inputNumPrototypes 

map_best_fn = args.inputMap_best
map_1nn_fn = args.inputMap_1nn

source_cst = load_tract(s_file,s_ind)

target_cst = load_tract(t_file,t_ind)

print len(source_cst), len(target_cst)

tractography1 = source_cst
tractography2 = target_cst

print "Source", len(tractography1)
print "Target", len(tractography2)

from dipy.io.pickles import load_pickle
mapping12_best = load_pickle(map_best_fn)
mapping12_coregistration_1nn = load_pickle(map_1nn_fn)

#visualize source and mapped source - red and blue
ren = fvtk.ren()
ren = visualize_source_mappedsource(ren, tractography1[:- num_pro], tractography2, mapping12_best[:-num_pro])
fvtk.show(ren)

#visualize target cst and mapped source cst - yellow and blue
ren1 = fvtk.ren()
target_cst_only = load_tract(t_file,t_cst)
ren1 = visualize_tract(ren1, target_cst_only, fvtk.yellow)
ren1 = visualize_mapped(ren1, tractography2, mapping12_best[:- num_pro], fvtk.blue)
fvtk.show(ren1)
