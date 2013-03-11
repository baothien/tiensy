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

    figure = 'small_dataset' # #'big_dataset' # 
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
    num_prototypes = 40
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    tracks = tracks[:500]
    print "tracks:", tracks.size

    #rho = compute_correlation(tracks, bundles_distances_mam, prototype_policies, num_prototypes, iterations)
    #plot_results(rho, num_prototypes, prototype_policies, color_policies)
    data_disimilarity = compute_disimilarity(tracks, bundles_distances_mam, prototype_policies[0], num_prototypes)
    
    ##############################################################################
    # define some parameters
    num_clusters = tracks.size / 50
    print num_clusters
    X = data_disimilarity
    ##############################################################################
    # Compute clustering with Means
    
    k_means = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    
    ##############################################################################
    # Compute clustering with MiniBatchKMeans
    batch_size = 40
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    mbk_means_labels_unique = np.unique(mbk_means_labels)
    
    ##############################################################################
    # Plot result
    
    fig = pl.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06','#4FACC5', '#FFFC34', '#7E9A06',
              '#BEA05F', '#1F9F3F', '#419AF6','#33ACC5', '#F29C34', '#8E9A06']
    
    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    
    distance = euclidean_distances(k_means_cluster_centers,
                                   mbk_means_cluster_centers,
                                   squared=True)
    order = distance.argmin(axis=1)
    
    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(num_clusters), colors):
    #for k in range(num_clusters):
    #    col = col + 10
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    pl.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
        t_batch, k_means.inertia_))
    
    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(num_clusters), colors):
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
    
    for l in range(num_clusters):
        different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
    
    identic = np.logical_not(different)
    ax.plot(X[identic, 0], X[identic, 1], 'w',
            markerfacecolor='#bbbbbb', marker='.')
    ax.plot(X[different, 0], X[different, 1], 'w',
            markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())
    
    pl.show()