from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr as correlation
from sys import stdout
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy


def furthest_first_traversal(S, k, distance, permutation=True):
    """This is the farthest first (ff) traversal algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See: http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    if permutation:
        idx = np.random.permutation(S.shape[0])
        S = S[idx]       
    else:
        idx = np.arange(S.shape[0], dtype=np.int)
    T = [0]
    while len(T) < k:
        z = distance(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]


def subset_furthest_first(S, k, distance, permutation=True, c=2.0):
    """Stochastic scalable version of ff based in a random subset of a
    specific size.

    See: D. Turnbull and C. Elkan, Fast Recognition of Musical Genres
    Using RBF Networks, IEEE Trans Knowl Data Eng, vol. 2005, no. 4,
    pp. 580-584, 17.
    
    http://cseweb.ucsd.edu/users/elkan/250Bfall2006/oct18.html
    Lemma:  Given k equal-size sets and any constant c >1, with high
    probability  ck log k  random points intersect each set.
    REFERENCE??
    """
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        idx = np.random.permutation(S.shape[0])[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, distance, permutation=False)]


if __name__ == '__main__':

    np.random.seed(0)

    filename = 'data/subj_05/101_32/DTI/tracks_dti_10K.dpy'
    k = 10
    num_prototypes = [5,10,20,30,50]
    prototype_policies = ['random', 'fft', 'sff']
    color_policies = ['r', 'b', 'g']
    iterations = 10
    verbose = True
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    # tracks = tracks[:]
    print "tracks:", tracks.size

    print "Computing distance matrix and similarity matrix (original space):",
    od = bundles_distances_mam(tracks,tracks)     
    print od.shape
    original_distances = squareform(od)

    rho = np.zeros((len(prototype_policies), len(num_prototypes),iterations))

    for m, prototype_policy in enumerate(prototype_policies):
        for j, num_proto in enumerate(num_prototypes):
            for k in range(iterations):
                print k                                
                if verbose: print("Generating %s prototypes as" % num_proto),
                if prototype_policy=='random':
                    if verbose: print("random subset of the initial data.")
                    prototype_idx = np.random.permutation(tracks.size)[:num_proto]
                    prototype = [tracks[i] for i in prototype_idx]
                elif prototype_policy=='fft':
                    if verbose: print("using the furthest first traversal aglorithm.")
                    prototype_idx = furthest_first_traversal(tracks, num_proto, bundles_distances_mam)
                    prototype = [tracks[i] for i in prototype_idx]
                elif prototype_policy=='sff':
                    if verbose: print("using the subset furthest first aglorithm.")
                    prototype_idx = subset_furthest_first(tracks, num_proto, bundles_distances_mam)
                    prototype = [tracks[i] for i in prototype_idx]                
                else:
                    raise Exception                

                if verbose: print("Computing dissimilarity matrix.")
                data_dissimilarity = bundles_distances_mam(tracks, prototype)

                if verbose: print("Computing distance matrix (dissimilarity space).")
                dissimilarity_distances = pdist(data_dissimilarity, metric='euclidean')

                rho[m,j,k] = correlation(original_distances, dissimilarity_distances)[0]

    for m, prototype_policy in enumerate(prototype_policies):
        mean = rho[m,:,:].mean(1)
        std = rho[m,:,:].std(1)
        color = color_policies[m]
        plt.plot(num_prototypes, mean, color, label=prototype_policy)
        plt.plot(num_prototypes, mean + std, color)
        plt.plot(num_prototypes, mean - std, color)

    plt.legend(loc='lower right')
