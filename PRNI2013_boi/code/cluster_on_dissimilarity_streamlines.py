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

    figure = 'big_dataset' # 'small_dataset' # #
    #print os.pardir
    if figure=='small_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'
        prototype_policies = ['random', 'fft', 'sff']
        color_policies = ['ko--', 'kx:', 'k^-']
    elif figure=='big_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
        prototype_policies = ['random', 'sff']
        color_policies = ['ko--', 'k^-']
    #num_prototypes =  [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #iterations = 50
    #num_trks = [1000,5000,10000,15000,20000,25000,0]
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
        num_cluster = num_clusters[i] #150 # 300 #tracks.size / 50
        print 'number of cluster: ', num_cluster
        num_trk = num_trks[i]        
        if num_trk!=0:
            tracks = tracks[:num_trk]# 15000] #1000 5000 10000 15000            
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
            #k_means_labels = k_means.labels_
            #k_means_cluster_centers = k_means.cluster_centers_
            #k_means_labels_unique = np.unique(k_means_labels)
        
        
        ##############################################################################
        # Compute clustering with MiniBatchKMeans
        X = data_disimilarity    
        batch_size = 100
        mbk = MiniBatchKMeans(init='random', n_clusters=num_cluster, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(X)
        t_mini_batch_random = time.time() - t0
        #mbk_means_labels = mbk.labels_
        #mbk_means_cluster_centers = mbk.cluster_centers_
        #mbk_means_labels_unique = np.unique(mbk_means_labels)        
        
        print "dissimilarity \t kmean (kmeans++) \t minibatch (kmeans++) \t kmean (random) \t minibatch (random)"
        print t_disi,'\t', t_batch, '\t', t_mini_batch, '\t', t_batch_random, '\t', t_mini_batch_random
    
        '''  
        ##############################################################################
        # Plot result
      
        fig = pl.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        #colors = ['#4EACC5', '#FF9C34', '#4E9A06','#4FACC5', '#FFFC34', '#7E9A06',
        #          '#BEA05F', '#1F9F3F', '#419AF6','#33ACC5', '#F29C34', '#8E9A06']
        
        # We want to have the same colors for the same cluster from the
        # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
        # closest one.
        
        distance = euclidean_distances(k_means_cluster_centers,
                                       mbk_means_cluster_centers,
                                       squared=True)
        order = distance.argmin(axis=1)
        
        # KMeans
        #make color map for every clusters
        colors = np.ones((num_cluster,3))
        for k in range(num_cluster):
            colors[k] = np.random.rand(3)
        
        ax = fig.add_subplot(1, 3, 1)
        #for k, col in zip(range(num_cluster), colors):
        for k in range(num_cluster):
            col = colors[k]        
            #all of members in the cluster k
            #my_members is an array of bool, only member in cluster k has the value True, otherwhile is False
            my_members = k_means_labels == k
            
            cluster_center = k_means_cluster_centers[k]
            ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            #ax.plot(tracks[my_members][:0][0,0], tracks[my_members][:0][0,1], 'w',
            #        markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)        
        ax.set_title('KMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        pl.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
            t_batch, k_means.inertia_))
        
        # MiniBatchKMeans
        ax = fig.add_subplot(1, 3, 2)
        for k, col in zip(range(num_cluster), colors):
            my_members = mbk_means_labels == order[k]
            cluster_center = mbk_means_cluster_centers[order[k]]
            ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        ax.set_title('MiniBatchKMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        pl.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
                (t_mini_batch, mbk.inertia_))
        
        # Initialise the different array to all False
        different = (mbk_means_labels == 4)
        ax = fig.add_subplot(1, 3, 3)
        
        for l in range(num_cluster):
            different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
        
        identic = np.logical_not(different)
        ax.plot(X[identic, 0], X[identic, 1], 'w',
                markerfacecolor='#bbbbbb', marker='.')
        ax.plot(X[different, 0], X[different, 1], 'w',
                markerfacecolor='m', marker='.')
        ax.set_title('Difference')
        ax.set_xticks(())
        ax.set_yticks(())
        pl.text(-3.5, 1.8, 'disimilarity time: %.2fs' % 
        (t_disi))
        
        pl.show()
        
        '''
        