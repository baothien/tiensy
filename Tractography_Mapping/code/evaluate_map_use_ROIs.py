# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:34:43 2014

@author: bao

Evaluate the mapping
Local: cst to cst
Global:- cst to cst_ext
       - cst_sff to cst_ext_sff
Using the ROIs defined by Nivedita of CST

"""
from common_functions import *
from intersect_roi import *
from dipy.viz import fvtk
import argparse

def extract_mapped_tract(s_idx, tract2, mapping): 
    mapped_tract = []
    for i in np.arange(len(s_idx)):        
        mapped_tract.append(tract2[mapping[i]])     
    
    mapped_tract = np.array(mapped_tract,dtype=np.object)
    
    return mapped_tract
    
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
                                 description="Evaluation the tractography mapping using two ROIs defined by expert",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.0')


parser.add_argument(
                    'inputSourceCSTIndex',
                    help='The file name of source CST index')
parser.add_argument(
                    'inputTargetSubject',
                    help='The target subject')
parser.add_argument(
                    'inputTargetTractography',
                    help='The file name of target whole-brain tractography as .trk format.')
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
                    
#parser.add_argument(
#                    '-pr', action='store', dest='inputNumPrototypes', type=int,
#                    help='The number of prototypes') 
parser.add_argument(
                    '-vi', action='store', dest='inputVisualize', type=int,
                    help='Visulize or not (0/1)') 
                                        
args = parser.parse_args()

if (args.inputVisualize==1):
    print "=========================="
    print "Source CST plus SFF index:       ", args.inputSourceCSTIndex
    print "Target subject:       ", args.inputTargetSubject
    print "Target tractography:       ", args.inputTargetTractography
    print "Target CST index:       ", args.inputTargetCSTIndex
    print "Target CST extension plus SFF index:       ", args.inputTargetCSTExtSFFIndex
    #print "Number of prototypes:      ", args.inputNumPrototypes 
    print "Map best file:      ", args.inputMap_best
    print "Map 1_NN file:      ", args.inputMap_1nn
    
    print "=========================="

ROIs_subject_L = {'101': (np.array([67.5,67.5,19.5], dtype=np.float32), np.array([75.5,67.5,39.5], dtype=np.float32)),
                  '102': (np.array([68.5,60.5,17.5], dtype=np.float32), np.array([75.5,58.5,39.5], dtype=np.float32)),
	            '103': (np.array([68.5,62.5,19.5], dtype=np.float32), np.array([75.5,61.5,37.5], dtype=np.float32)),                  
                  '104': (np.array([69.5,63.5,19.5], dtype=np.float32), np.array([74.5,61.5,36.5], dtype=np.float32)),
                  '105': (np.array([67.5,65.5,13.5], dtype=np.float32), np.array([73.5,64.5,36.5], dtype=np.float32)), 
                  '106': (np.array([70.5,63.5,19.5], dtype=np.float32), np.array([76.5,62.5,38.5], dtype=np.float32)), 
                  '107': (np.array([67.5,69.5,15.5], dtype=np.float32), np.array([76.5,69.5,35.5], dtype=np.float32)), 
                  '109': (np.array([67.5,64.5,15.5], dtype=np.float32), np.array([75.5,62.5,37.5], dtype=np.float32)),  
                  '111': (np.array([67.5,65.5,16.5], dtype=np.float32), np.array([76.5,70.5,40.5], dtype=np.float32)),   
                  '112': (np.array([69.5,65.5,20.5], dtype=np.float32), np.array([74.5,65.5,35.5], dtype=np.float32)), 
                  '113': (np.array([69.5,57.5,23.5], dtype=np.float32), np.array([79.5,60.5,40.5], dtype=np.float32)),
                  '201': (np.array([69.5,66.5,20.5], dtype=np.float32), np.array([74.5,66.5,38.5], dtype=np.float32)), 
                  '202': (np.array([69.5,64.5,22.5], dtype=np.float32), np.array([74.5,65.5,38.5], dtype=np.float32)), 
                  '203': (np.array([68.5,65.5,19.5], dtype=np.float32), np.array([75.5,65.5,38.5], dtype=np.float32)), 
                  '204': (np.array([67.5,68.5,13.5], dtype=np.float32), np.array([75.5,66.5,35.5], dtype=np.float32)), 
                  '205': (np.array([68.5,61.5,16.5], dtype=np.float32), np.array([75.5,62.5,38.5], dtype=np.float32)), 
                  '206': (np.array([68.5,67.5,20.5], dtype=np.float32), np.array([75.5,69.5,39.5], dtype=np.float32)), 
                  '207': (np.array([67.5,68.5,15.5], dtype=np.float32), np.array([75.5,64.5,36.5], dtype=np.float32)), 
                  '208': (np.array([69.5,66.5,21.5], dtype=np.float32), np.array([75.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([69.5,68.5,18.5], dtype=np.float32), np.array([75.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([69.5,64.5,20.5], dtype=np.float32), np.array([76.5,65.5,39.5], dtype=np.float32)), 
                  '212': (np.array([67.5,65.5,16.5], dtype=np.float32), np.array([77.5,66.5,38.5], dtype=np.float32)), 
                  '213': (np.array([66.5,69.5,18.5], dtype=np.float32), np.array([73.5,70.5,38.5], dtype=np.float32)),   
		 }
   
ROIs_subject_R = {'101': (np.array([61.5,66.5,20.5], dtype=np.float32), np.array([55.5,66.5,40.5], dtype=np.float32)),
                  '102': (np.array([61.5,59.5,17.5], dtype=np.float32), np.array([53.5,59.5,40.5], dtype=np.float32)),
	            '103': (np.array([62.5,62.5,20.5], dtype=np.float32), np.array([54.5,61.5,39.5], dtype=np.float32)),                  
                  '104': (np.array([59.5,64.5,19.5], dtype=np.float32), np.array([54.5,63.5,36.5], dtype=np.float32)),
                  '105': (np.array([62.5,66.5,12.5], dtype=np.float32), np.array([55.5,66.5,35.5], dtype=np.float32)), 
                  '106': (np.array([63.5,62.5,19.5], dtype=np.float32), np.array([53.5,63.5,40.5], dtype=np.float32)), 
                  '107': (np.array([61.5,67.5,15.5], dtype=np.float32), np.array([54.5,69.5,35.5], dtype=np.float32)), 
                  '109': (np.array([61.5,64.5,12.5], dtype=np.float32), np.array([53.5,62.5,37.5], dtype=np.float32)),  
                  '111': (np.array([61.5,65.5,15.5], dtype=np.float32), np.array([54.5,70.5,38.5], dtype=np.float32)),   
                  '112': (np.array([59.5,65.5,22.5], dtype=np.float32), np.array([54.5,64.5,34.5], dtype=np.float32)), 
                  '113': (np.array([61.5,56.5,23.5], dtype=np.float32), np.array([53.5,58.5,41.5], dtype=np.float32)),
                  '201': (np.array([59.5,66.5,20.5], dtype=np.float32), np.array([53.5,66.5,38.5], dtype=np.float32)), 
                  '202': (np.array([61.5,64.5,22.5], dtype=np.float32), np.array([53.5,65.5,38.5], dtype=np.float32)), 
                  '203': (np.array([59.5,65.5,19.5], dtype=np.float32), np.array([53.5,64.5,37.5], dtype=np.float32)), 
                  '204': (np.array([62.5,68.5,13.5], dtype=np.float32), np.array([54.5,68.5,35.5], dtype=np.float32)), 
                  '205': (np.array([63.5,61.5,16.5], dtype=np.float32), np.array([54.5,60.5,38.5], dtype=np.float32)), 
                  '206': (np.array([60.5,67.5,20.5], dtype=np.float32), np.array([54.5,68.5,39.5], dtype=np.float32)), 
                  '207': (np.array([61.5,68.5,15.5], dtype=np.float32), np.array([53.5,67.5,34.5], dtype=np.float32)), 
                  '208': (np.array([60.5,65.5,21.5], dtype=np.float32), np.array([53.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([61.5,67.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([63.5,64.5,20.5], dtype=np.float32), np.array([54.5,67.5,39.5], dtype=np.float32)), 
                  '212': (np.array([61.5,65.5,16.5], dtype=np.float32), np.array([54.5,65.5,38.5], dtype=np.float32)), 
                  '213': (np.array([59.5,70.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)),   
		 }

ROIs_subject = ROIs_subject_R
Rs = [5.,5.]#np.array([2.,2.],dtype=np.float32)
   
s_ind = args.inputSourceCSTIndex
t_sub = args.inputTargetSubject
t_file = args.inputTargetTractography
t_ind = args.inputTargetCSTExtSFFIndex
t_cst = args.inputTargetCSTIndex

#num_pro = args.inputNumPrototypes 

map_best_fn = args.inputMap_best
map_1nn_fn = args.inputMap_1nn

target_cst = load_tract_trk(t_file,t_ind)

from dipy.io.pickles import load_pickle
s_idx = load_pickle(s_ind)

tractography2 = target_cst

print "\t Len source", len(s_idx)
print "\t Len target", len(tractography2)


mapping12_best = load_pickle(map_best_fn)
mapping12_coregistration_1nn = load_pickle(map_1nn_fn)


mapped_tract = extract_mapped_tract(s_idx, tractography2, mapping12_best)#[:-num_pro])



ROIs = ROIs_subject[t_sub]
ROIs = [[r[0],128.-r[1],r[2]] for r in ROIs]

common = intersec_ROIs(mapped_tract, ROIs, Rs, vis=True)

print "\t The number of fibers through the ROIs ", len(common)
#print "Done evaluate using ROIs"

#visualize target cst and mapped source cst - yellow and blue
if (args.inputVisualize==1):
    ren1 = fvtk.ren()
    target_cst_only = load_tract_trk(t_file,t_cst)
    ren1 = visualize_tract(ren1, target_cst_only, fvtk.yellow)
    #ren1 = visualize_mapped(ren1, tractography2, mapping12_best[:- num_pro], fvtk.blue)
    ren1 = visualize_mapped(ren1, tractography2, mapping12_best[:-50], fvtk.blue)
    fvtk.add(ren1, fvtk.sphere(ROIs[0],Rs[0],color = fvtk.red, opacity=1.0)) 
    fvtk.add(ren1, fvtk.sphere(ROIs[1],Rs[1],color = fvtk.blue, opacity=1.0)) 
    fvtk.show(ren1)
    
