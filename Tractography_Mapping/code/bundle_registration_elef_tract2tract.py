# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 19:58:16 2014

@author: bao
"""

"""
==========================
Direct Bundle Registration
==========================

This example explains how you can register two bundles from two different
subjects directly in native space [Garyfallidis14]_.

To show the concept we will use two pre-saved cingulum bundles.

This is run under  https://github.com/Garyfallidis/dipy.git branche bundle_registration
(url: https://github.com/Garyfallidis/dipy/tree/bundle_registration)
"""

from dipy.viz import fvtk
from time import sleep
from dipy.io.pickles import load_pickle
#from dipy.data import two_cingulum_bundles
import numpy as np
import argparse
from common_functions import load_tract, save_tracks_dpy


#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Registration the moving tract to the static tract with Eleftherios method",
                                 epilog="Written by Bao Thien Nguyen, bao@fbk.eu",
                                 version='1.0')

parser.add_argument(
                    'inputSourceTractography',
                    help='The file name of source whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputSourceTractIndex',
                    help='The file name of source tract index')
parser.add_argument(
                    'inputTargetTractography',
                    help='The file name of target whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputTargetTractIndex',
                    help='The file name of target tract index')

parser.add_argument(
                    'outputSourceTractAligned',
                    help='The file name of source tract after aligning')

args = parser.parse_args()

print "=========================="
#print "Source tractography:       ", args.inputSourceTractography
print "Source tract index:       ", args.inputSourceTractIndex
#print "Target tractography:       ", args.inputTargetTractography
print "Target tract index:       ", args.inputTargetTractIndex
print "Out put source tract aligned:       ", args.outputSourceTractAligned
#print "=========================="

s_file = args.inputSourceTractography
s_ind = args.inputSourceTractIndex

t_file = args.inputTargetTractography
t_ind = args.inputTargetTractIndex

out_file = args.outputSourceTractAligned

s_tract = load_tract(s_file,s_ind)
t_tract = load_tract(t_file,t_ind)

          

from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     vectorize_streamlines,
                                     BundleMinDistance)

"""
An important step before running the registration is to resample the streamlines
so that they both have the same number of points per streamline. Here we will
use 10 points.
"""
points = 20#10#20
s_tract = vectorize_streamlines(s_tract, points)
t_tract = vectorize_streamlines(t_tract, points)

"""
Let's say now that we want to move the ``s_tract`` (moving) so that it can be
aligned with ``t_tract`` (static). Here is how this is done.
"""
num_fiber = 300#50#100,200
s_rand_idx = np.random.randint(low = 0, high = len(s_tract), size = min(num_fiber,len(s_tract)))
s_tract_tmp = [s_tract[i] for i in s_rand_idx]

t_rand_idx = np.random.randint(low = 0, high = len(t_tract), size = min(num_fiber,len(t_tract)))
t_tract_tmp = [t_tract[i] for i in t_rand_idx]

#s_tract_tmp = s_tract[:50]
#t_tract_tmp = t_tract[:50]

#x0 = np.array([0, 0, 0, 0, 0, 0.])

affine = True

if affine:
    
    x0 = np.array([0, 0, 0, 0, 0, 0., 1, 1, 1, 0, 0, 0])
    
    metric = BundleMinDistance()
    method = 'L-BFGS-B'
    bounds = [(-20, 20), (-20, 20), (-20, 20),
              (-30, 30), (-30, 30), (-30, 30),
              (0.5, 1.5), (0.5, 1.5), (0.5, 1.5),
              (-1, 1), (-1, 1), (-1, 1)]

if not affine:
    
    x0 = np.array([0, 0, 0, 0, 0, 0.])
    #default is BundleMinDistanceFast, rigid and L-BFGS-B
    metric = BundleMinDistance()
    method = 'Powell'#L-BFGS-B'
    bounds = None
    #bounds = [(-20, 20), (-20, 20), (-20, 20),
    #          (-30, 30), (-30, 30), (-30, 30)]

srr = StreamlineLinearRegistration(metric=metric, x0=x0, bounds=bounds)

srm = srr.optimize(static=t_tract_tmp, moving=s_tract_tmp)

"""
After the optimization is finished we can apply the learned transformation to
``s_tract``.
"""

s_tract_aligned = srm.transform(s_tract)

save_tracks_dpy(s_tract_aligned, out_file)
print 'Saved:  ' , out_file

                
def show_both_bundles(bundles, colors=None, show=False, fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = fvtk.streamtube(bundle, color, linewidth=0.3)
        lines.RotateX(-90)
        lines.RotateZ(90)
        fvtk.add(ren, lines)
    if show:
        fvtk.show(ren)
    if fname is not None:
        sleep(1)
        fvtk.record(ren, n_frames=1, out_path=fname, size=(900, 900))

vis = False#True#False
cal = False
if vis:
    show_both_bundles([s_tract, t_tract],
                      colors=[fvtk.colors.orange, fvtk.colors.red],
                      show=True,
                      fname='before_registration.png')


    """
    .. figure:: before_registration.png
       :align: center
    
       **Before bundle registration**.
    """

    show_both_bundles([s_tract_aligned, t_tract],
                      colors=[fvtk.colors.orange, fvtk.colors.red],
                      show=True,
                      fname='after_registration.png')

    """
    .. figure:: after_registration.png
       :align: center
    
       **After bundle registration**.
    
    .. [Garyfallidis14] Garyfallidis et. al, "Direct native-space fiber bundle
                        alignment for group comparisons", ISMRM, 2014.
    
    """
if cal:
    from common_functions import streamlines_to_vol
    vol1, vol2, intersec = streamlines_to_vol(s_tract_aligned, t_tract, [128,128,70], disp=False)
    from common_functions import volumn_intersec
    cou1, cou2, inter_cou = volumn_intersec(s_tract_aligned, t_tract, [128,128,70], [1,1,1], disp=True)

