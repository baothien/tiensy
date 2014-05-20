"""
==========================
Direct Bundle Registration
==========================

This example explains how you can register two bundles from two different
subjects directly in native space [Garyfallidis14]_.

To show the concept we will use two pre-saved cingulum bundles.
"""

from dipy.viz import fvtk
from time import sleep
from dipy.io.pickles import load_pickle
from dipy.data import two_cingulum_bundles
import numpy as np

#cb_subj1, cb_subj2 = two_cingulum_bundles()
'''
source = '205'
target = '209'
s_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
s_idx = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_L_3M.pkl'
#s_idx = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_R_3M_plus_sff.pkl'

t_file = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
t_idx = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + target + '_corticospinal_L_3M.pkl'
#s_idx = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/BOI_seg/' + source + '_cst_R_3M_plus_sff.pkl'
'''

source = '213'
target = '202'
s_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
s_idx = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + source + '_CST_ROI_R_3M.pkl'
#s_idx = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_R_3M_plus_sff.pkl'

t_file = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
t_idx = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + target + '_CST_ROI_R_3M.pkl'
#s_idx = '/home/bao/tiensy/Tractography_Mapping/data/50_SFF_plus_ext/ROI_seg/' + source + '_cst_R_3M_plus_sff.pkl'


from common_functions import load_tract

cb_subj1 = load_tract(s_file,s_idx)
cb_subj2 = load_tract(t_file,t_idx)

          

from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     vectorize_streamlines,
                                     BundleMinDistance)

"""
An important step before running the registration is to resample the streamlines
so that they both have the same number of points per streamline. Here we will
use 20 points.
"""

cb_subj1 = vectorize_streamlines(cb_subj1, 20)
cb_subj2 = vectorize_streamlines(cb_subj2, 20)

"""
Let's say now that we want to move the ``cb_subj2`` (moving) so that it can be
aligned with ``cb_subj1`` (static). Here is how this is done.
"""
cb_subj1 = cb_subj1[:50]
cb_subj2 = cb_subj2[:50]

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

srm = srr.optimize(static=cb_subj1, moving=cb_subj2)

"""
After the optimization is finished we can apply the learned transformation to
``cb_subj2``.
"""

cb_subj2_aligned = srm.transform(cb_subj2)


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


show_both_bundles([cb_subj1, cb_subj2],
                  colors=[fvtk.colors.orange, fvtk.colors.red],
                  show=True,
                  fname='before_registration.png')


"""
.. figure:: before_registration.png
   :align: center

   **Before bundle registration**.
"""

show_both_bundles([cb_subj1, cb_subj2_aligned],
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
from common_functions import streamlines_to_vol
vol1, vol2, intersec = streamlines_to_vol(cb_subj1, cb_subj2_aligned, [128,128,70], disp=False)