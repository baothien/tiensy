# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:34:02 2015

@author: bao
"""

import numpy as np
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors


def visualize(streamlines_A, streamlines_B, mappingAB, line='line', shift=np.array([0.0, 0.0, 200.0]), color_A=fvtk.colors.white, color_B=fvtk.colors.green, color_line=fvtk.colors.yellow):
    assert(len(mappingAB) == len(streamlines_A))
    assert(mappingAB.max() <= len(streamlines_B))
    if line == 'line':
        line = fvtk.line
        linewidth = 2.0
        linekwargs = {'linewidth':linewidth}
    elif line == 'tube':
        line = fvtk.streamtube
        linewidth = 1.0
        linekwargs = {'linewidth':linewidth, 'lod':False}
    else:
        raise Exception

    if color_A == 'auto':
        color_A = line_colors(streamlines_A)

    if color_B == 'auto':
        color_B = line_colors(streamlines_B)
            
    streamlines_B_shifted = np.array([s + shift for s in streamlines_B])
    midpointA = [s[int(s.shape[0] / 2)] for s in streamlines_A]
    midpointB = [s[int(s.shape[0] / 2)] for s in streamlines_B_shifted]

    ren = fvtk.ren()
    fvtk.add(ren, line(streamlines_A.tolist(), colors=color_A, **linekwargs))
    fvtk.add(ren, line(streamlines_B_shifted.tolist(), colors=color_B, **linekwargs))
    fvtk.add(ren, fvtk.line(zip(midpointA, midpointB), colors=color_line, opacity=0.5, linewidth=0.5))
    fvtk.show(ren)