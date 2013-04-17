import time
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import BallTree

if __name__ == '__main__':

    np.random.seed(0)

    n_samples = 250000
    centers = 50
    n_features = 40
    n_clusters = 150

    print "Generate sample data."
    X, labels_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features)
    print "X.shape:", X.shape

    # print "Compute clustering with Means:"

    # k_means = KMeans(init='k-means++', n_clusters=300, n_init=10)
    # t0 = time.time()
    # k_means.fit(X)
    # t_batch = time.time() - t0
    # k_means_labels = k_means.labels_
    
    
    print "MBKM clustering time:",
    init = 'k-means++'
    init = 'random'
    mbk = MiniBatchKMeans(init=init, n_clusters=n_clusters, batch_size=1000,
                          n_init=3, max_no_improvement=5, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    print t_mini_batch

    print "BallTree construction",
    t0 = time.time()
    bt = BallTree(X)
    t_balltree = time.time() - t0
    print t_balltree, "sec"

    print n_clusters, "neareast neighbor queries (medoids):",
    medoids_mbk = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    for i, centroid in enumerate(mbk.cluster_centers_):
        medoids_mbk[i] = bt.query(centroid, k=1, return_distance=False)
    t_bt_query = time.time() - t0
    print t_bt_query, "sec"


    print "exhaustive search of the medoids:",
    medoids_exh = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    for i, centroid in enumerate(mbk.cluster_centers_):
        tmp = X - centroid
        medoids_exh[i] = (tmp * tmp).sum(1).argmin()
    t_exh_query = time.time() - t0
    print t_exh_query, "sec"


    print "exhaustive smarter search of the medoids:",
    medoids_exhs = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    for i, centroid in enumerate(mbk.cluster_centers_):
        idx_i = np.where(mbk.labels_==i)[0]
        if idx_i.size == 0: idx_i = [0]
        tmp = X[idx_i] - centroid
        medoids_exhs[i] = idx_i[(tmp * tmp).sum(1).argmin()]
    t_exhs_query = time.time() - t0
    print t_exhs_query, "sec"
