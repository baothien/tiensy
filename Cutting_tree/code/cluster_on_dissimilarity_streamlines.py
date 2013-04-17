import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances

import time

import numpy as np
import pylab as pl


if __name__ == '__main__':

    np.random.seed(0)

    figure = 'big_dataset' # 'small_dataset' #     
    if figure=='small_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'
        prototype_policies = ['random', 'fft', 'sff']
        color_policies = ['ko--', 'kx:', 'k^-']
    elif figure=='big_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
        prototype_policies = ['random', 'sff']
        color_policies = ['ko--', 'k^-']    
    num_trks = [0,15000,10000,5000,1000,500]
    num_clusters = [150,50,50,50,50,50]
    num_prototypes = 40
    
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    t_batch=0
    t_batch_random=0
    for i in range(len(num_trks)):   
      if (i==1):
        ##############################################################################
        # define some parameters
        num_cluster = num_clusters[i] 
        print 'number of cluster: ', num_cluster
        num_trk = num_trks[i]        
        if num_trk!=0:
            tracks = tracks[:num_trk]
        print "tracks:", tracks.size
    
        t0 = time.time()
        data_disimilarity = compute_disimilarity(tracks, bundles_distances_mam, prototype_policies[0], num_prototypes,tracks.size)
        t_disi = time.time() - t0
        print 'Dissimilarity: ', t_disi        
                
        ##############################################################################
        # Compute clustering with Means        
        if (i!=0):        
            X = data_disimilarity
            k_means = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
            t0 = time.time()
            k_means.fit(X)
            t_batch = time.time() - t0
            k_means_labels = k_means.labels_
            k_means_cluster_centers = k_means.cluster_centers_
            k_means_labels_unique = np.unique(k_means_labels)
        
        ##############################################################################
        # Compute clustering with MiniBatchKMeans
        X = data_disimilarity    
        batch_size = 100
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_cluster, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(X)
        t_mini_batch = time.time() - t0
        mbk_means_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(mbk_means_labels)       
        
        ##############################################################################
        # Compute clustering with Means
        if (i!=0):        
            X = data_disimilarity
            k_means = KMeans(init='random', n_clusters=num_cluster, n_init=10)
            t0 = time.time()
            k_means.fit(X)
            t_batch_random = time.time() - t0     
        
        ##############################################################################
        # Compute clustering with MiniBatchKMeans
        X = data_disimilarity    
        batch_size = 100
        mbk = MiniBatchKMeans(init='random', n_clusters=num_cluster, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(X)
        t_mini_batch_random = time.time() - t0            
        
        print "dissimilarity \t kmean (kmeans++) \t minibatch (kmeans++) \t kmean (random) \t minibatch (random)"
        print t_disi,'\t', t_batch, '\t', t_mini_batch, '\t', t_batch_random, '\t', t_mini_batch_random
    
    
        
