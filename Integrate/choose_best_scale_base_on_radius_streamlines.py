# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:58:15 2013

@author: bao
"""

import numpy as np
import nibabel as nib

import time as time
import matplotlib.pyplot as plt
from sklearn import metrics
#from sklearn.cluster import Ward
from hierarchical import Ward, _hc_cut

from dipy.io.pickles import save_pickle,load_pickle

#for computing the CPU time, not elapsed time
import resource
def cpu_time():
    	return resource.getrusage(resource.RUSAGE_SELF)[0]

def compute_silhouette_score(X, tree, metric_measure):
	'''
	n : sample sizes |X|
	num of clusters, k = [1..n]
	for each value of k
	      P_k: partition of X having k cluster (based on the maximum distance (or the radius) of a cluster) 
	      compute silhouette score for P_k 

	input:  X : data 
		tree: ward tree
		matric_measure ('euclidean', ...)
	output: float array 1D size n
		value of silhouette score of partion P_k 
	'''
    	n = len(X)
    	score = np.zeros(n-1)
	print 'Length : ', n

    	for i in range(n-1):
       	#canot calculate the silhouette score for only one cluster
           #should start from 2 clusters 
           k = i + 2
		print '\n Cutting at k = ', k
        	label = _hc_cut(k,tree.children_, tree.n_leaves_)
		print '\n Compute score ...'
        	s = metrics.silhouette_score(X, label, metric = metric_measure)
		#s = silhouette_score_block(X, label, metric= metric_measure , sample_size=None	)	
		score[k-2] = s
    	return score
	

def plot_result(x, y,  maker, label , x_label, y_label, title):
    	plt.figure()            
   	plt.plot(x, y, maker, label = label, markersize = 8.0)
    	plt.legend(loc='lower right')
    	plt.xlabel(x_label)
    	plt.ylabel(y_label)
    	plt.title(title)
    	plt.show()

#--------------------------------------------------------
if __name__ == '__main__':
	label = 'silhouette score'
	maker = 'ko--'
 
     sub_id = 101
     
     #load data
     print 'Loading data and tree ...'
     #data = load_pickle('data_1K.txt')
     #tree = load_pickle("test_1K.tre")
     tree_name = 'Results/' + str(sub_id) + '/' + str(sub_id) +'_full_tracks_50_neighbors_modified_ward_full_tree.tree'
     tree = load_pickle(tree_name)	    
	
     
     #print 'Computing score for each partition ...'    
     sil_scores = compute_silhouette_score(data, tree, 'euclidean')
     #sil_scores = load_pickle('silhouette_score_data_1K.txt')
     plot_result(range(len(data))[1:], sil_scores, maker,label, "num of clusters","silhouette score","Silhouette score for all partition of simulated data")

