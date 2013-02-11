from dipy.io.dpy import Dpy
import dissimilarity_common
from dipy.tracking.distances import bundles_distances_mam
import time
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree

if __name__ == '__main__':

    np.random.seed(0)

    k = 30
    
    filename = 'data/subj_05/101_32/DTI/tracks_gqi_1M_linear.dpy'
    print "Loading data:", filename
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)

    print "Computing %s prototypes" % k
    t = time.time()
    idx = dissimilarity_common.subset_furthest_first(S=tracks, k=k, distance=bundles_distances_mam, c=3.0)
    print time.time() - t, "sec."

    print "Dissimilarity projection"
    t = time.time()
    dataset = bundles_distances_mam(tracks, tracks[idx])
    print time.time() - t, "sec."
    
    print "Building KDTree"
    kdtree = cKDTree(dataset)

    print "Building BallTree"
    balltree = BallTree(dataset)
    
