# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:23:19 2014

@author: bao
"""

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
                    'outputCSTExtIndex',
                    help='The output file name of CST extension indices ') 
parser.add_argument(
                    'outputCSTExtSFFInExtIndex',
                    help='The output file name of CST extension plus SFF indices ') 
                    
                    
args = parser.parse_args()

print "=========================="
print "Source tractography:       ", args.inputSourceTractography
print "Source CST index:       ", args.inputSourceCSTIndex
print "Number of SFF prototypes:    ", args.inputNumPrototypes
print "Out CST extension index file:      ", args.outputCSTExtIndex
print "Out CST plus SFF in extension index file:   ", args.outputCSTExtSFFInExtIndex

print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTIndex

n_pro = args.inputNumPrototypes
cst_ext_file =  args.outputCSTExtIndex
cst_ext_sff_in_ext_file = args.outputCSTExtSFFInExtIndex


save_id_tract_plus_sff_in_ext(s_file, s_ind, n_pro, bundles_distances_mam, cst_ext_file, cst_ext_sff_in_ext_file,  1.5/3.,  4.5 , 6.2/2.)
#save_id_tract_plus_sff_in_ext(tracks_filename, id_file, num_proto, distance,  out_fname_ext , out_fname_sff_in_ext, thres_len= 2.2/3., thres_vol = 1.4 , thres_dis = 3./2.):

'''
ren = fvtk.ren()
cst_ext_sff_in_ext = load_tract(s_file, cst_ext_sff_in_ext_file )
ren = visualize_tract(ren,cst_ext_sff_in_ext,fvtk.yellow)

#cst_ext = load_tract(s_file,cst_ext_file)
#ren = visualize_tract(ren,cst_ext, fvtk.red)

cst = load_tract(s_file,s_ind)
ren = visualize_tract(ren,cst, fvtk.blue)


fvtk.show(ren)
'''