import argparse
import os
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
from dipy.viz import fvtk

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Saving the index CST to the real CST",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.0')

parser.add_argument(
                    'inputDirectory',
                    help='A directory of whole-brain tractography as .dpy format.')
parser.add_argument(
                    'outputDirectory',
                    help='The output directory will be created if it does not exist.')
parser.add_argument(
                    'cstidxDirectory',
                    help='The directory of the referened index of Corticalspinal Tracts')
parser.add_argument(
                    '-verbose', dest="flag_verbose",
                    help='Verbose. (currently not used)')


args = parser.parse_args()

if not os.path.isdir(args.inputDirectory):
    print "Error: Input directory", args.inputDirectory, "does not exist."
    exit()

if not os.path.isdir(args.cstidxDirectory):
    print "Error: Anatomy directory", args.inputDirectory, "does not exist."
    exit()

outdir = args.outputDirectory
if not os.path.exists(outdir):
    print "Output directory", outdir, "does not exist, creating it."
    os.makedirs(outdir)

if args.flag_verbose:
    print "Verbose display message is ON."
else:
    print "Verbose display message is OFF."
verbose = args.flag_verbose


print "Starting saving cortinalspinal tract..."
print ""
print "=====input directory ======\n", args.inputDirectory
print "=====output directory =====\n", args.outputDirectory
print "=====CST index directory =====\n", args.cstidxDirectory
print "=========================="

input_path = args.inputDirectory + '/'
output_path = args.outputDirectory + '/'
cstidx_path = args.cstidxDirectory + '/'

sub = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]  

#input_path =  '/Users/bao/Desktop/ALS_VTK_format_reg_results_400sample_120length/iteration_4/'
#output_path = '/Users/bao/Desktop/ALS_VTK_format_reg_results_400sample_120length/iteration_4/'
#cstidx_path = '/Users/bao/Desktop/ALS_data/Segmentation/BOI/index_pkl/'

for id_file in np.arange(len(sub)): 
	tracks_filename = input_path + str(mapping[sub[id_file]])+'_tracts_3M_registered.dpy'
	dpr_tracks = Dpy(tracks_filename, 'r')
	tensor_all_tracks=dpr_tracks.read_tracks()
	dpr_tracks.close()
	tracks_id_left = load_pickle( cstidx_path + str(sub[id_file])+'_corticospinal_L_3M.pkl')
	tracks_id_right = load_pickle(cstidx_path + str(sub[id_file])+'_corticospinal_R_3M.pkl')
	
	cst_left = [tensor_all_tracks[i] for i  in tracks_id_left]
	cst_right = [tensor_all_tracks[i] for i  in tracks_id_right]

	cst_left_fname = output_path + str(mapping[sub[id_file]])+'_corticospinal_L_3M_registered.dpy'

	cst_right_fname = output_path + str(mapping[sub[id_file]])+'_corticospinal_R_3M_registered.dpy'

	"""
	Save the streamlines.
	"""

	dpw = Dpy(cst_left_fname, 'w')
	dpw.write_tracks(cst_left)    
	dpw.close()
      print len(cst_left)
	print '>>>> Done ', cst_left_fname

	dpw = Dpy(cst_right_fname, 'w')
	dpw.write_tracks(cst_right)    
	dpw.close()
      print len(sct_right)
	print '>>>> Done ', cst_right_fname
	 
        



