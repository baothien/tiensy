# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:38:57 2013

@author: bao
"""
import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import numpy as np
#import pylab as pl
#import mpl_toolkits.mplot3d.axes3d as p3
#from sklearn.datasets.samples_generator import make_swiss_roll

#from sklearn.cluster import Ward
from hierarchical import Ward

import matplotlib.pyplot as plt
from dipy.io.pickles import save_pickle,load_pickle
import time as time

#tree_name = 'Results/101/101_15000_tracks_third_trial.tree'
#tree_name = 'Results/101/101_full_tracks_fifth_trial.tree'

#tree_name = 'Results/201/201_15000_tracks_first_trial.tree'
#tree_name = 'Results/201/201_full_tracks_first_trial.tree'

#tree_name = 'Results/109/109_15000_tracks_first_trial.tree'
#tree_name = 'Results/109/109_full_tracks_first_trial.tree'

#tree_name = 'Results/205/205_15000_tracks_first_trial.tree'
#tree_name = 'Results/205/205_full_tracks_first_trial.tree'

#tree_name = 'Results/210/210_15000_tracks_first_trial.tree'
#tree_name = 'Results/210/210_full_tracks_first_trial.tree'

#tree_name = 'Results/210/210_15000_tracks_50_neighbors_modified_ward_full_tree_first_trial.tree'
tree_name = 'Results/101/101_full_tracks_50_neighbors_modified_ward_full_tree_fifth_trial.tree'

'''
compute the histogram of each node: how many node split to a given child nodes

'''


tree = load_pickle(tree_name)
cut = tree.best_cut()
cut.insert(len(cut),[tree.height_[len(tree.height_)-1],0.])
for i in np.arange(len(cut)):
    height = cut[i][0]
    if i==0:
        h = 0.
    else:
        h = cut[i-1][0]
    t0 = time.time()
    guillotines = tree.cut(height)
    t_cut = time.time() - t0
    print 'Time for cutting at height ',height, ' :  ',t_cut
    count = [len(tree.children_list(node, h)) for node in guillotines]
    #print count
    plt.figure()
    #plt.hist(count,len(np.unique(count)),range=(0.5,max(count)+0.5))
    plt.hist(count,max(count),range=(0.5,max(count)+0.5))       
    title = '15000 tracks: cut from height =  ' + str(height) + '  to  ' + str(h)
    #title = 'Full tracks: cut from height =  ' + str(height) + '  to  ' + str(h)
    text = '\nCutting time at height ' + str(height) + ' : ' + str(t_cut)+'s'
    plt.title(title+text)
    plt.xlabel('Num of children')
    plt.ylabel('Histogram')
    
    #plt.figtext(text)
    plt.show()
    
  
    
    
    
    