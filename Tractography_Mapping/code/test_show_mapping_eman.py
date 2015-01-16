# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:34:28 2015

@author: bao
"""

import numpy as np
from nibabel import trackvis
from dipy.tracking.distances import bundles_distances_mam
from show_mapping import visualize


def transform(s):
    s_new = s + (s**0.7) * 5.0
    # s_new = s + np.log(s) * 5.0
    # s_new = s + np.sin(s / 50.0) * 10.0
    return s_new


if __name__ == '__main__':

    distortion = False

    length_min = 20
    filename = 'data/HCP_subject124422_100Kseeds/tracks_dti_100K.trk'
    print("Loading %s" % filename)
    streamlines, header = trackvis.read(open(filename))
    print("Removing streamlines shorter than %s points and transforming to array." % length_min)
    streamlines = np.array(filter(lambda x: len(x) >= length_min, [s[0] for s in streamlines]), dtype=np.object)
    
    size_A = 100
    size_B = 20
    print("Size A: %s" % size_A)
    print("Size B: %s" % size_B)
    idx_A = np.random.permutation(len(streamlines))[:size_A]
    streamlines_A = streamlines[idx_A]
    idx_B = idx_A[:size_B] # np.random.permutation(len(streamlines))[:size_B]
    streamlines_B = streamlines[idx_B]

    if distortion:
        streamlines_A = np.array([transform(s) for s in streamlines_A], dtype=np.object)

    print("Computing 1NN mapping.")
    dm_AB = bundles_distances_mam(streamlines_A, streamlines_B)
    mappingAB_1nn = dm_AB.argmin(1)

    print("Visualizing streamlines and mapping.")
    visualize(streamlines_A, streamlines_B, mappingAB_1nn, line='tube', color_A='auto', color_B='auto')