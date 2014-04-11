# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:58:31 2014

@author: bao

create and save the extension of cst and the extension of cst plus sff prototypes
"""

from common_functions import *
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
import argparse

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
                    'inputSourceCSTIndex',
                    help='The file name of source CST index')
                    
parser.add_argument(
                    '-pr', action='store', dest='inputNumPrototypes', type=int,
                    help='The number of prototypes') 
                                  
parser.add_argument(
                    'outputCSTSFFIndex',
                    help='The output file name of CST plus SFF indices ')                    
parser.add_argument(
                    'outputCSTExtIndex',
                    help='The output file name of CST extension indices ') 
parser.add_argument(
                    'outputCSTExtSFFIndex',
                    help='The output file name of CST extension plus SFF indices ') 
                    
                    
args = parser.parse_args()

print "=========================="
print "Source tractography:       ", args.inputSourceTractography
print "Source CST index:       ", args.inputSourceCSTIndex
print "Number of SFF prototypes:    ", args.inputNumPrototypes
print "Out CST plus SFF index file:       ", args.outputCSTSFFIndex
print "Out CST extension index file:      ", args.outputCSTExtIndex
print "Out CST extension plus SFF index file:   ", args.outputCSTExtSFFIndex

print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTIndex

n_pro = args.inputNumPrototypes
cst_sff_file =  args.outputCSTSFFIndex
cst_ext_file =  args.outputCSTExtIndex
cst_ext_sff_file = args.outputCSTExtSFFIndex


cst_sff_id = save_id_tract_plus_sff(s_file,s_ind, n_pro, bundles_distances_mam, cst_sff_file)
cst_sff = load_tract(s_file,cst_sff_file)
ren = fvtk.ren()
ren = visualize_tract(ren,cst_sff)
#fvtk.show(ren)

cst_ext_sff_id = save_id_tract_ext_plus_sff(s_file, s_ind, n_pro, bundles_distances_mam, cst_ext_sff_file, cst_ext_file)
cst_ext_sff = load_tract(s_file, cst_ext_sff_file )
ren = visualize_tract(ren,cst_ext_sff,fvtk.yellow)
fvtk.show(ren)
