import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import BallTree

import time

import numpy as np
import pylab as pl


if __name__ == '__main__':

    np.random.seed(0)
    figure = 'big_dataset'    
    if figure=='small_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'
        prototype_policies = ['random', 'fft', 'sff']
        color_policies = ['ko--', 'kx:', 'k^-']
    elif figure=='big_dataset':
        filename = 'ALS_Data/101/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
        prototype_policies = ['random', 'sff']
        color_policies = ['ko--', 'k^-']
    
    num_trks = [500,1000,5000,10000,15000,0]
    num_clusters = [50,150,250]
    num_prototypes = 40
    
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks_all = dpr.read_tracks()
    dpr.close()
    tracks_all = np.array(tracks_all, dtype=np.object)
    t_batch=0
    t_batch_random=0
    print "Num tracks \t num clusters \t btree \t dissimilarity \t minibatch (random) \t medoids \t exhaustive search \t smart exhaustive search"
    for i in range(len(num_trks)):   
      
        ##############################################################################
        # define some parameters
        num_trk = num_trks[i]        
        if num_trk!=0:
            tracks = tracks_all[:num_trk]
        else:
            tracks = tracks_all
        print "tracks:", tracks.size
    
        t0 = time.time()
        data_disimilarity = compute_disimilarity(tracks, bundles_distances_mam, prototype_policies[0], num_prototypes,tracks.size)
        t_disi = time.time() - t0
        
        #for convert from centroids to medoids        
        t0 = time.time()
        bt = BallTree(data_disimilarity)
        t_btree = time.time() - t0
        
        for j in range(len(num_clusters)):
            num_cluster = num_clusters[j]        
            ##############################################################################
            # Compute clustering with MiniBatchKMeans
            X = data_disimilarity    
            batch_size = 1000
            mbk = MiniBatchKMeans(init='random', n_clusters=num_cluster, batch_size=batch_size,
                                  n_init=10, max_no_improvement=10, verbose=0)
            t0 = time.time()
            mbk.fit(X)
            t_mini_batch_random = time.time() - t0
            
            mbk_means_labels = mbk.labels_
            mbk_means_cluster_centers = mbk.cluster_centers_
            mbk_means_labels_unique = np.unique(mbk_means_labels)
 
            #  ------------------------------------------        
            #  compute medoids from centroids of MBKM
            #  ------------------------------------------        
            t0 = time.time()
            for i in range(len(mbk_means_cluster_centers)):
                medoid = bt.query(mbk_means_cluster_centers[i], k=1, return_distance=False)
            t_medoids_mini_batch_random = time.time() - t0              
           
            #print "exhaustive search of the medoids:",
            medoids_exh = np.zeros(len(mbk_means_cluster_centers), dtype=np.int)
            t0 = time.time()
            for i, centroid in enumerate(mbk.cluster_centers_):
                tmp = X - centroid
                medoids_exh[i] = (tmp * tmp).sum(1).argmin()
            t_exh_query = time.time() - t0
            #print t_exh_query, "sec"
        
        
            #print "exhaustive smarter search of the medoids:",
            medoids_exhs = np.zeros(len(mbk_means_cluster_centers), dtype=np.int)
            t0 = time.time()
            for i, centroid in enumerate(mbk.cluster_centers_):
                idx_i = np.where(mbk.labels_==i)[0]
                if idx_i.size == 0: idx_i = [0]
                tmp = X[idx_i] - centroid
                medoids_exhs[i] = idx_i[(tmp * tmp).sum(1).argmin()]
            t_smart_exh_query = time.time() - t0
            #print t_exhs_query, "sec"           
           
           
            print len(tracks),'\t',len(mbk_means_cluster_centers),'\t',t_btree,'\t', t_disi,'\t',  t_mini_batch_random, '\t', t_medoids_mini_batch_random, '\t', t_exh_query,'\t',t_smart_exh_query
            