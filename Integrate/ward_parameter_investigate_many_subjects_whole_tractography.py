# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:58:15 2013

@author: bao
"""

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
#from sklearn.cluster import Ward
from hierarchical import Ward

#for computing the CPU time, not elapsed time
import resource
def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]
 
'''
figure: type of data
        'small_dataset'; 'median_dataset'; 'big_dataset';
data_id: 101, ..., 109, 201, ..., 213 (detail in ALS dataset)
return: tractography 
'''
def load_data(figure, data_id):   
    
    if figure=='small_dataset':
        filename = 'ALS_Data/'+ str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'        
    elif figure=='median_dataset':
        filename = 'ALS_Data/' + str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_1M.dpy'    
    elif figure=='big_dataset':
        filename = 'ALS_Data/' + str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    return tracks
    
'''
This function compute the silhouette score for the data with no limitation of the data size
This overcomes the limitation of memory issue of the current version of Silhouette score in scikit-learn
First, the whole dataset will be clustered
Second, the silhouette score is only calculated on a small set of the dataset using sample_size parameter of Silhouette Score function
Iteratively running this and then take the mean score for the whole dataset
'''
#def compute_silhouette(data, distance,prototype_policies, num_prototypes, iterations, verbose=False, size_limit=1000):
def compute_silhouette_sample_size(data_ids, prototype_policie, num_prototype, num_cluster, metric_measure, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil = np.zeros((len(data_ids), len(num_neighbors),iterations))
    time = np.zeros((len(data_ids), len(num_neighbors),iterations))

    for m, data_id in enumerate(data_ids):
        if verbose: print 'Subject/Control: ', data_id
        data = load_data('big_dataset',data_id)   

        X = compute_disimilarity(data, bundles_distances_mam, prototype_policie, num_prototype,len(data))
        
        for j, num_neigh in enumerate(num_neighbors):
            print "number of neighbors:", num_neigh, " - ", 
            
            #print '\n',k,
            stdout.flush()
            if verbose: print("Generating at %s neighbor" % num_neigh),                
            
            connectivity = kneighbors_graph(X, n_neighbors=num_neigh)                          
            
            st = cpu_time()#time.clock()
            ward = Ward(n_clusters=num_cluster, compute_full_tree=False, connectivity=connectivity).fit(X)
            #t = time.clock() - st
            t = cpu_time() - st
            
            label = ward.labels_
            score = 0.
            for k in range(iterations):                    
                score_temp = metrics.silhouette_score(X, label, metric = metric_measure,sample_size=20000)
                score = score + score_temp
            score = score/iterations
            sil[m,j,k] = score
            time[m,j,k] = t
                #print score, '\t', time
            print
    return sil, time



def plot_results(rho , num_neighbors, label_policies, color_policies, y_label):
    plt.figure()
    
    for m, prototype_policy in enumerate(label_policies):
        mean = rho[m,:,:].mean(1)
        std = rho[m,:,:].std(1)
        errorbar = std # 3.0 * std / np.sqrt(rho.shape[2])
        
        plt.plot(num_neighbors, mean, color_policies[m], label = label_policies[m], markersize = 8.0)
            
        plt.fill(np.concatenate([num_neighbors, num_neighbors[::-1]]),
                 np.concatenate([mean - errorbar, (mean + errorbar)[::-1]]),
                 alpha=.25,fc='black',  ec='None')
    plt.legend(loc='lower right')
    plt.xlabel("number of neighbors $(k)$")
    plt.ylabel(y_label)
    plt.show()


###############################################################################
if __name__ == '__main__':

    np.random.seed(0)
   
      
    
    data_ids = [101]#,102,201,202]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [25]# ,50, 75, 100]#[10,20,30,40,50,60,70,80,90,100]
    iterations = 5
    prototype_policies = ['random', 'fft', 'sff']
    #fit with num_tracks    
    label_policies =['50k']# ['500','1K','5K','10K']#,'15K','All']
    color_policies = ['ko--']#, 'kx:', 'k^-','k*-' ]
    verbose = True    
       
    sil, time = compute_silhouette_sample_size(data_ids, prototype_policies[0], num_prototype, 
                                   num_cluster, 'euclidean', 
                                   num_neighbors, iterations, verbose)
    plot_results(sil, num_neighbors,label_policies,color_policies,"Silhouette score")
    plot_results(time, num_neighbors,label_policies,color_policies,"Time for hierarchical tree")
    
    
    
    
    
    
    
        