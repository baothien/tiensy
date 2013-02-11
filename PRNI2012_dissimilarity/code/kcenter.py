"""The k-center problem: fix any metric space (X, \rho). The k-center
problem asks: given a set S and an integer k, what is the smallest
\rho for which you can find an \epsilon-cover of S of size k?

See:
http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf

def furthest_first_traversal(S, k, rho=cdist, permutation=True):
    """This is the farthest first (ff) traversal algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See: http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    #20120224
    len_S = len(S)
    #20120224
    if permutation:
        #idx = np.random.permutation(S.shape[0])
        idx = np.random.permutation(lens_S)
        S = S[idx]       
    else:
        #idx = np.arange(S.shape[0], dtype=np.int)
        idx = np.arange(len_S, dtype=np.int)
    T = [0]
    while len(T) < k:
        z = rho(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]
    
def furthest_first_traversal_fast(S, k, rho=cdist, permutation=True):
    """This is the farthest first (ff) traversal algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See: http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf

    THIS IMPLEMENTATION IS MUCH FASTER WHEN k IS BIG.
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
    D = np.zeros((k, S.shape[0]))
    for i in range(k):
        D[i] = rho(S, S[[T[i]]]).squeeze()
        z = D[:i+1].min(0).argmax()
        T.append(z)
    return idx[T]


def subset_furthest_first(S, k, rho=cdist, permutation=True, c=2.0):
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
    #20120224
    len_S = len(S)
    #20120224
    
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        #idx = np.random.permutation(S.shape[0])[:size]       
        idx = np.random.permutation(len_S)[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, rho, permutation=False)]
    
def subset_furthest_first_fast(S, k, rho=cdist, permutation=True, c=2.0):
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
    return idx[furthest_first_traversal_fast(S[idx], k, rho, permutation=False)]
    

#===============================================================================
#                          for tractography - 20120225
#===============================================================================
def furthest_first_traversal_tracks(S, k, rho=bundles_distances_mdf, permutation=True):
#def furthest_first_traversal_tracks(S, k, rho=bundles_distances_mam, permutation=True):
    """This is the farthest first (ff) traversal algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See: http://cseweb.ucsd.edu/~dasgupta/291/lec1.pdf
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    
    #20120224
    len_S = len(S)
    
    #20120224
    if permutation:
        idx = np.random.permutation(len_S)        
        #S = np.random.permutation(len_S)                 
        S = [S[i] for i in idx]  
        #S = S[idx]
    else:
        #idx = np.arange(S.shape[0], dtype=np.int)
        idx = np.arange(len_S, dtype=np.int)
        
    T = [0]
    while len(T) < k:
        z = rho(S, [S[i] for i in T]).min(1).argmax()
        #z = rho(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]

#    T = [S[0]]    
#    print "T = ", T        
#    while len(T) < k:
#        z = rho(S, T).min(1).argmax()
#        T.append(S[z])
#    return T

def subset_furthest_first_tracks(S, k, rho=bundles_distances_mdf, permutation=True, c=10.0):
#def subset_furthest_first_tracks(S, k, rho=bundles_distances_mam, permutation=True, c=2.0):
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
    #20120224
    len_S = len(S)
    #20120224
    
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        #idx = np.random.permutation(S.shape[0])[:size]       
        idx = np.random.permutation(len_S)[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    #return furthest_first_traversal_tracks(S[idx], k, rho, permutation=False)
    S_temp=[S[i] for i in idx]    
    return furthest_first_traversal_tracks(S_temp, k, rho, permutation=False)
    
#===============================================================================
#                       end of for tractography - 20120225
#===============================================================================

if __name__ == '__main__':

    from dissimilarity_2d import generate_data, generate_data2

    np.random.seed(1)

    num_points = 100
    num_prototypes = 10
    distance = 'euclidean' 

    generator = generate_data

    print("Generate a 2D dataset."),
    data = generator(num_points)
    print(data.shape)

    # prototype_idx = farthest_first_traversal(data, k=num_prototypes, rho=cdist)

    prototype_idx = subset_furthest_first_fast(data, k=num_prototypes, rho=cdist)
    prototype = data[prototype_idx]

    visualize = True
    if visualize:
        plt.figure()
        plt.plot(data[:,0], data[:,1], 'ko')
        plt.plot(prototype[:,0], prototype[:,1], 'rx', markersize=16, markeredgewidth=4)
        for i in range(num_prototypes):
            plt.text(prototype[i,0], prototype[i,1], str(i))
