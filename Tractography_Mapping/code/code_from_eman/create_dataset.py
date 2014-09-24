import numpy as np
from dipy.io.dpy import Dpy
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from prototypes import subset_furthest_first as sff
from prototypes import furthest_first_traversal as fft
from scipy.spatial import distance_matrix

def create_dataset_artificial(size1, size2, same=True):
    print("Dateset creation.")
    if same:
        X = np.random.rand(max([size1, size2]), 2)
        X1 = X[:size1]
        X2 = X[:size2]
        dm = distance_matrix(X, X)
        dm1 = dm[:size1, :size1]
        dm2 = dm[:size2, :size2]
        sigma1 = sigma2 = np.median(dm)
    else:
        X1 = np.random.rand(size1, 2)
        X2 = np.random.rand(size2, 2)
        dm1 = distance_matrix(X1, X1)
        dm2 = distance_matrix(X2, X2)
        sigma1 = np.median(dm1)
        sigma2 = np.median(dm2)

    A = np.exp(- dm1 * dm1 / (sigma1 ** 2))
    B = np.exp(- dm2 * dm2 / (sigma2 ** 2))
    return A, B


def create_dataset_from_tractography(size1, size2, same=True):
    if same: assert(size2 >= size1)
    filename = 'data/tracks_dti_10K_linear.dpy'
    print "Loading", filename
    dpr = Dpy(filename, 'r')
    tractography = dpr.read_tracks()
    dpr.close()
    print len(tractography), "streamlines"
    print "Removing streamlines that are too short"
    tractography = filter(lambda x: len(x) > 20, tractography) # remove too short streamlines
    print len(tractography), "streamlines"    
    tractography = np.array(tractography, dtype=np.object)

    print "Creating two simulated tractographies of sizes", size1, "and", size2
    if same:
        ids = fft(tractography, k=max([size1, size2]), distance=bundles_distances_mam)
        tractography1 = tractography[ids[:size1]]
    else:
        # ids1 = np.random.permutation(len(tractography))[:size1]
        # ids1 = sff(tractography, k=size1, distance=bundles_distances_mam)
        ids1 = fft(tractography, k=size1, distance=bundles_distances_mam)
        tractography1 = tractography[ids1[:size1]]

    if same:
        tractography2 = tractography[ids[:size2]]
    else:
        # ids2 = np.random.permutation(len(tractography))[:size2]
        # ids2 = sff(tractography, k=size2, distance=bundles_distances_mam)
        ids2 = fft(tractography, k=size2, distance=bundles_distances_mam)
        tractography2 = tractography[ids2]
        
    print "Done."

    print "Computing the distance matrices for each tractography."
    dm1 = bundles_distances_mam(tractography1, tractography1)
    dm2 = bundles_distances_mam(tractography2, tractography2)

    print("Computing similarity matrices.")
    sigma2 = np.mean([np.median(dm1), np.median(dm2)]) ** 2.0
    print("sigma2 = %f" % sigma2)
    A = np.exp(-dm1 * dm1 / sigma2)
    B = np.exp(-dm2 * dm2 / sigma2)

    # Note: the optimization works even using distance instead of similarity:
    # A = dm1
    # B = dm2

    return A, B



def create_sparse_dataset_from_tractography(size1, size2, same=True):
    pass
