import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr as correlation
from sys import stdout
import matplotlib.pyplot as plt
import pickle

import nibabel as nib
from fos.data import get_track_filename
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from scipy.stats import spearmanr
from dipy.io.dpy import Dpy

def furthest_first_traversal_tracks(S, k, rho=bundles_distances_mdf, permutation=True):
#def furthest_first_traversal_tracks(S, k, rho=bundles_distances_mam, permutation=True):
    """This is the farthest first (ff) traversal algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See: http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    
    len_S = len(S)    
    if permutation:
        idx = np.random.permutation(len_S)                
        S = [S[i] for i in idx]          
    else:
        idx = np.arange(len_S, dtype=np.int)
        
    T = [0]
    while len(T) < k:
        z = rho(S, [S[i] for i in T]).min(1).argmax()
        #z = rho(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]

def subset_furthest_first_tracks(S, k, rho=bundles_distances_mdf, permutation=True, c=10.0):
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
    len_S = len(S)
        
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        #idx = np.random.permutation(S.shape[0])[:size]       
        idx = np.random.permutation(len_S)[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    #return furthest_first_traversal_tracks(S[idx], k, rho, permutation=False)
    S_temp=[S[i] for i in idx]    
    return idx[furthest_first_traversal_tracks(S_temp, k, rho, permutation=False)]

def my_squareform (arr, n):
    
    b = []        
    for i in np.arange(n-1):
        for j in np.arange(i+1,n):            
            b.append(arr[i,j])            
    return b

def compute_correlation_tracks (tracks, prototype_policy, nums_prototypes, iterations):
    
    dist_thr = 10.0
    pts = 12
    epsilon =30000
    
    np.random.seed(1)
  
    print('Computing QuickBundles:'),
    qb = QuickBundles(tracks, dist_thr=dist_thr, pts=pts)    
    tracks_downsampled = qb.downsampled_tracks()
    tracks_qb = qb.virtuals()
    tracks_qbe, tracks_qbei = qb.exemplars()
    print(len(tracks_qb))

    tracks = tracks_downsampled 
           
    num_points = len(tracks)
    score_sample = np.zeros((len(nums_prototypes), iterations))
    print "Num prototypes: "        
    for j, num_prototypes in enumerate(nums_prototypes):
        print num_prototypes,
        stdout.flush()      
        for k in range(iterations):                              
            if prototype_policy=='subset':                
                prototype_idx = np.random.permutation(num_points)[:num_prototypes]
                prototype = [tracks[i] for i in prototype_idx]
            elif prototype_policy=='kcenter':
                prototype_idx = furthest_first_traversal_tracks(tracks, num_prototypes)
                prototype = [tracks[i] for i in prototype_idx]
            elif prototype_policy=='sff':
                prototype_idx = subset_furthest_first_tracks(tracks, num_prototypes)
                prototype = [tracks[i] for i in prototype_idx]                
            else:
                raise Exception                
            
            #choosing random m tracks from all tracks for estimating correlation 
            #to reduce time and memory           
            
            num_sub_tracks=1000           
            sub_tracks_idx = np.random.permutation(num_points)[:num_sub_tracks]
            sub_tracks = [tracks[i] for i in sub_tracks_idx] 

            #print("Computing distance matrix and similarity matrix (original space):"),
            od = bundles_distances_mdf(sub_tracks,sub_tracks)                
            n = len(sub_tracks)
            original_distances = my_squareform(od,n)
            
            #print("Computing dissimilarity matrix.")                
            data_dissimilarity = bundles_distances_mdf(sub_tracks, prototype)
                       
            #print("Computing distance matrix (dissimilarity space).")
            dissimilarity_distances = pdist(data_dissimilarity, metric='euclidean')
            dd = squareform(dissimilarity_distances)

            #print("Compute correlation between distances before and after projection.")            
            correlation_pval_od_dd = correlation(original_distances, dissimilarity_distances) 
            #print("correlation_od_dd = %s" % correlation_pval_od_dd[0])

            idx_epsilon = original_distances<epsilon            
            correlation_pval_epsilon = correlation(od[idx_epsilon], dd[idx_epsilon])                 
            #print("correlation_od_dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon[0]))
            
            original_distances_unbiased = od.diagonal(1)[::2]
            dissimilarity_distances_unbiased = dd.diagonal(1)[::2]
            correlation_pval_unbiased = correlation(original_distances_unbiased, dissimilarity_distances_unbiased)
             #print("Unbiased correlation od dd = %s" % correlation_pval_unbiased[0])
       
            idx_epsilon_unbiased = original_distances_unbiased<epsilon
            correlation_pval_epsilon_unbiased = correlation(original_distances_unbiased[idx_epsilon_unbiased], dissimilarity_distances_unbiased[idx_epsilon_unbiased])
           
            print("Unbiased correlation od dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon_unbiased[0]))                               
            
            score_sample[j, k] = correlation_pval_epsilon_unbiased[0]

    return score_sample

