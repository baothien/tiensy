""" Code to map CST.

From Bao:
'CST_segmentation by experts which is far away in MNI space:
(205,210) (201,205) (210,212) (209,212)  
ROIs that are far away in MNI space:  (205,209); (205,213)'
"""

import numpy as np
from nibabel.trackvis import read
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from prototypes import subset_furthest_first
# from hashlib import md5
from sklearn.neighbors import KDTree

# This is the dictionary from subject ID to segmentation ID
subject_segmentation = {201:  2,
                        205: 10,
                        209: 18,
                        210: 20,
                        212: 22
    }

def load_or_create(subject, side, len_threshold=20, k=100, outdir='data_als/cache/', seed=0):
    filename = 'data_als/%d/tracks_dti_3M_linear.trk' % subject

    print "Loading", filename
    streamlines, header = read(filename)
    streamlines = np.array(streamlines, dtype=np.object)[:,0]

    # hd = md5(streamlines).hexdigest()
    # print "hexdigest:", hd

    filename_cst = 'data_als/%d/%d_corticospinal_%s_3M.pkl'
    filename_cst = filename_cst % (subject, subject_segmentation[subject], side)
    print "Loading CST", filename_cst
    cst_ids = np.load(filename_cst)
    # cst_streamlines = streamlines[cst_ids]

    print "Building the dissimilarity representation."
    try:
        filename_prototypes = outdir+'Pi_ids_%d_%s.npy' % (subject, side)
        print "Trying to load", filename_prototypes
        Pi_ids = np.load(filename_prototypes)
        print "Done."
    except IOError:
        print "Not found."
        print "Creating prototypes."
        lenghts = np.array([len(s) for s in streamlines])
        streamlines_long_ids = np.where(lenghts > len_threshold)[0] # using long streamlines heuristic
        distance = bundles_distances_mam
        np.random.seed(seed)
        Pi_ids = streamlines_long_ids[subset_furthest_first(streamlines[streamlines_long_ids], k=k, distance=distance)] # using long streamlines heuristic
        print "Saving", filename_prototypes
        np.save(filename_prototypes, Pi_ids)
        Pi = streamlines[Pi_ids]
        
    try:
        filename_dr = outdir+'dr_%d_%s.npy' % (subject, side)
        print "Trying to load", filename_dr
        dr = np.load(filename_dr)
        print "Done."
    except IOError:
        print "Not found."
        print "Computing the dissimilarity matrix."
        dr = bundles_distances_mam(streamlines, Pi).astype(np.float32)
        print "Saving", filename_dr
        np.save(filename_dr, dr.astype(np.float32))

    return streamlines, cst_ids, Pi_ids, dr


def loss(dm_A, dm_mapped_B):
    """Loss function for mapping.
    """
    tmp = (dm_mapped_B - dm_A)[np.triu_indices(dm_A.shape[0], k=1)]
    return (tmp * tmp).sum() # sqrt if you want

def show_Pi_mapping(streamlines_A, streamlines_B, Pi_ids_A, mapping_Pi):
        r = fvtk.ren()
        Pi_viz_A = fvtk.streamtube(streamlines_A[Pi_ids_A], fvtk.colors.red)
        fvtk.add(r, Pi_viz_A)
        # Pi_viz_B = fvtk.streamtube(streamlines_B[Pi_ids_B], fvtk.colors.blue)
        # fvtk.add(r, Pi_viz_B)
        Pi_viz_A_1nn_B = fvtk.streamtube(streamlines_B[mapping_Pi], fvtk.colors.white)
        fvtk.add(r, Pi_viz_A_1nn_B)
        fvtk.show(r)
    
def greedy_optimization(dm_sA, dm_sA_mapping_B, dr_B, mapping_A_in_B, kdt, idxs_to_optimize, neighborhood_size=100, optimization_steps=1000, seed=0):
    np.random.seed(seed)
    print
    print "Loss:", loss(dm_sA, dm_sA_mapping_B)
    print "Greedy optimization."
    for n in range(optimization_steps):
        flag = False
        # print n
        # pick one streamline at random from sA:
        i = idxs_to_optimize[np.random.randint(len(idxs_to_optimize))]
        # retrieve d(s_i^A, s_j^A) for each j!=i :
        d_i_A = dm_sA[i,:]
        # retrieve d(s_{\phi(i)}^B, s__{\phi(j)}^B) for each j!=i :
        d_phii_B = dm_sA_mapping_B[i,:]
        # compute loss(i):
        tmp = d_i_A - d_phii_B
        partial_loss_i = (tmp * tmp).sum()
        # print "l(i):", partial_loss_i
        
        # retrieve a neighborhood of phi(i):   (phi(i) excluded)
        neighbors = kdt.query(dr_B[mapping_A_in_B[i]], k=neighborhood_size, return_distance=False).squeeze()[1:]
        # compute the change in loss when switching from phi(i) to each of the neighbors, and greedily keep just the improvements:
        best_partial_loss = partial_loss_i
        best_candidate = mapping_A_in_B[i]
        d_best_candidate_B = d_phii_B
        for candidate in neighbors:
            # computing new distances:
            d_candidate_B = bundles_distances_mam([streamlines_B[candidate]], streamlines_B[mapping_A_in_B]).squeeze()
            d_candidate_B[i] = 0.0 # fixing distance i with proper value
            # computing new partial loss:
            tmp = d_i_A - d_candidate_B
            l_candidate = (tmp * tmp).sum()
            # updating the best_candidate:
            if l_candidate < best_partial_loss:
                print "Improvement:", best_partial_loss - l_candidate
                best_partial_loss = l_candidate
                best_candidate = candidate
                d_best_candidate_B = d_candidate_B
                flag = True

        # If optimization happened, update mapping and dm_sA_mapping_B and compute new loss:
        if flag:
            mapping_A_in_B[i] = best_candidate
            dm_sA_mapping_B[i, :] = d_best_candidate_B
            dm_sA_mapping_B[:, i] = d_best_candidate_B

            print "Loss:", loss(dm_sA, dm_sA_mapping_B)

    return mapping_A_in_B, dm_sA_mapping_B


if __name__ == '__main__':

    subject_A = 210
    subject_B = 205
    side = 'L' # or 'R'
    show = False
    k = 100
    
    filename = 'data_als/%d/tracks_dti_3M_linear.trk'
    filename_A = filename % subject_A
    filename_B = filename % subject_B

    streamlines_A, cst_ids_A, Pi_ids_A, dr_A = load_or_create(subject_A, side, k=k)
    streamlines_B, cst_ids_B, Pi_ids_B, dr_B = load_or_create(subject_B, side, k=k)

    if show:
        r = fvtk.ren()
        cst_viz_A = fvtk.streamtube(cst_streamlines_A, fvtk.colors.red)
        cst_viz_B = fvtk.streamtube(cst_streamlines_B, fvtk.colors.blue)
        fvtk.add(r, cst_viz_A)
        fvtk.add(r, cst_viz_B)
        fvtk.show(r)
        
    if show:
        r = fvtk.ren()
        Pi_viz_A = fvtk.streamtube(streamlines_A[Pi_ids_A], fvtk.colors.red)
        fvtk.add(r, Pi_viz_A)
        Pi_viz_B = fvtk.streamtube(streamlines_B[Pi_ids_B], fvtk.colors.blue)
        fvtk.add(r, Pi_viz_B)
        fvtk.show(r)

    print "Computing the distance matrix between Pi_A streamlines."
    dm_Pi_A = bundles_distances_mam(streamlines_A[Pi_ids_A], streamlines_A[Pi_ids_A])
    print "Computing the distance matrix between Pi_B streamlines."
    dm_Pi_B = bundles_distances_mam(streamlines_B[Pi_ids_B], streamlines_B[Pi_ids_B])
    print "Loss:", loss(dm_Pi_A, dm_Pi_B)

    print "Computing KDTree on B with the dissimilarity representation."
    dr_dim = 100
    print "Using the first", dr_dim, "prototypes."
    kdt = KDTree(dr_B[:,:dr_dim])
    print "Computing the dissimilarity representation of prototypes A in B."
    dr_Pi_A_in_B = bundles_distances_mam(streamlines_A[Pi_ids_A], streamlines_B[Pi_ids_B])
    print "Computing the initial mapping."
    print "Retrieving the nearest-neighbors of prototypes A in B."
    Pi_ids_A_1nn_B = kdt.query(dr_Pi_A_in_B[:,:dr_dim], k=1, return_distance=False).squeeze()
    mapping_Pi = Pi_ids_A_1nn_B.copy()
    mapping_Pi_initial = mapping_Pi.copy()

    print "Computing the distance matrix between mapping_Pi streamlines."
    dm_Pi_A_1nn_B = bundles_distances_mam(streamlines_B[mapping_Pi], streamlines_B[mapping_Pi])
    dm_Pi_mapping_B = dm_Pi_A_1nn_B.copy()

    print "Loss:", loss(dm_Pi_A, dm_Pi_mapping_B)

    if show:
        show_Pi_mapping(streamlines_A, streamlines_B, Pi_ids_A, mapping_Pi)

    np.random.seed(0)
    print
    print "Loss:", loss(dm_Pi_A, dm_Pi_mapping_B)
    print "Greedy optimization."
    neighborhood_size = 100
    optimization_steps = 1000
    for n in range(optimization_steps):
        flag = False
        # print n
        # pick one streamline at random from Pi_A:
        i = np.random.randint(k)
        # retrieve d(s_i^A, s_j^A) for each j!=i :
        d_i_A = dm_Pi_A[i,:]
        # retrieve d(s_{\phi(i)}^B, s__{\phi(j)}^B) for each j!=i :
        d_phii_B = dm_Pi_mapping_B[i,:]
        # compute loss(i):
        tmp = d_i_A - d_phii_B
        partial_loss_i = (tmp * tmp).sum()
        # print "l(i):", partial_loss_i
        
        # retrieve a neighborhood of phi(i):   (phi(i) excluded)
        neighbors = kdt.query(dr_B[mapping_Pi[i]], k=neighborhood_size, return_distance=False).squeeze()[1:]
        # compute the change in loss when switching from phi(i) to each of the neighbors, and greedily keep just the improvements:
        best_partial_loss = partial_loss_i
        best_candidate = mapping_Pi[i]
        d_best_candidate_B = d_phii_B
        for candidate in neighbors:
            # computing new distances:
            d_candidate_B = bundles_distances_mam([streamlines_B[candidate]], streamlines_B[mapping_Pi]).squeeze()
            d_candidate_B[i] = 0.0 # fixing distance i with proper value
            # computing new partial loss:
            tmp = d_i_A - d_candidate_B
            l_candidate = (tmp * tmp).sum()
            # updating the best_candidate:
            if l_candidate < best_partial_loss:
                print "Improvement:", best_partial_loss - l_candidate
                best_partial_loss = l_candidate
                best_candidate = candidate
                d_best_candidate_B = d_candidate_B
                flag = True

        # If optimization happened, update mapping and dm_Pi_mapping_B and compute new loss:
        if flag:
            mapping_Pi[i] = best_candidate
            dm_Pi_mapping_B[i, :] = d_best_candidate_B
            dm_Pi_mapping_B[:, i] = d_best_candidate_B

            print "Loss:", loss(dm_Pi_A, dm_Pi_mapping_B)

    if show:
        show_Pi_mapping(streamlines_A, streamlines_B, Pi_ids_A, mapping_Pi)


    print
    print "Optimizing mapping of random streamlines."
    np.random.seed(0)
    M = 100 # number of the random streamlines to map
    # generating random ids of streamlines from A
    s_ids_A = np.random.permutation(len(streamlines_A))[:M]
    # computing the dissimilarity representation of random streamlines in B
    dr_sA_in_B = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_B[Pi_ids_B])
    # getting the closest (1nn) streamlines in B to the random ones:
    s_ids_A_mapping_B = kdt.query(dr_sA_in_B, k=1, return_distance=False).squeeze()
    # 1nn is the initial mapping of the random streamlines:
    mapping_A_in_B = s_ids_A_mapping_B.copy()
    # computing the dissimlarity matrix of the random streamlines in A
    dm_sA = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_A[s_ids_A])
    # computing the initial dissimlarity matrix of the mapped random streamlines in B
    dm_sA_mapping_B = bundles_distances_mam(streamlines_B[mapping_A_in_B], streamlines_B[mapping_A_in_B])

    # Fixing seed for stochastic greedy optimization:
    np.random.seed(0)
    # setting which ids (wrt the distance matrix) to optimize (all of them)
    idxs_to_optimize = np.arange(dm_sA.shape[0])
    # actual optimization:
    mapping_A_in_B, dm_sA_mapping_B = greedy_optimization(dm_sA, dm_sA_mapping_B, dr_B, mapping_A_in_B, kdt, idxs_to_optimize, neighborhood_size=100, optimization_steps=1000)
    

    print
    print "Optimizing mapping of CST but considering (and NOT optimizing) prototypes too."
    s_ids_A = cst_ids_A
    # computing the dissimilarity representation of the CST in B
    dr_sA_in_B = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_B[Pi_ids_B])
    # getting the closest (1nn) streamlines in B to the CST:
    s_ids_A_mapping_B = kdt.query(dr_sA_in_B, k=1, return_distance=False).squeeze()
    # 1nn is the initial mapping of the CST and we add the prototypes with the best mapping computed before:
    mapping_A_in_B = np.concatenate([s_ids_A_mapping_B.copy(), mapping_Pi.copy()])
    # adding prototypes of A to the set of ids considered here:
    s_ids_A = np.concatenate([s_ids_A, Pi_ids_A])
    
    # computing the dissimlarity matrix of the random streamlines in A
    dm_sA = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_A[s_ids_A])
    # computing the initial dissimlarity matrix of the mapped random streamlines in B
    dm_sA_mapping_B = bundles_distances_mam(streamlines_B[mapping_A_in_B], streamlines_B[mapping_A_in_B])

    # Fixing seed for stochastic greedy optimization:
    np.random.seed(0)
    # setting which ids to optimize: just CST!
    idxs_to_optimize = np.arange(len(cst_ids_A))
    # actual optimization:
    mapping_A_in_B, dm_sA_mapping_B = greedy_optimization(dm_sA, dm_sA_mapping_B, dr_B, mapping_A_in_B, kdt, idxs_to_optimize, neighborhood_size=100, optimization_steps=100, seed=0)

    # extracting the optimized mapping of CST
    cst_ids_A_mapping_B = mapping_A_in_B[:len(cst_ids_A)]
    # extracting initial 1nn mapping of CST:
    cst_ids_A_mapping_B_1nn = s_ids_A_mapping_B

    if show:
        # show 1nn mapped CST A
        # show optimized mapped CST A
        r = fvtk.ren()
        # show CST A
        # Pi_viz_A = fvtk.streamtube(streamlines_A[cst_ids_A], fvtk.colors.red)
        # fvtk.add(r, Pi_viz_A)
        Pi_viz_A_1nn = fvtk.streamtube(streamlines_B[cst_ids_A_mapping_B_1nn], fvtk.colors.blue)
        fvtk.add(r, Pi_viz_A_1nn)
        Pi_viz_A_optimized = fvtk.streamtube(streamlines_B[cst_ids_A_mapping_B], fvtk.colors.green)
        fvtk.add(r, Pi_viz_A_optimized)
        fvtk.show(r)
        

    print
    print "A new idea: mapping the translated CST."
    s_ids_A = cst_ids_A
    # computing the dissimilarity representation of the CST in B
    dr_sA_in_B = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_B[Pi_ids_B])
    # getting the closest (1nn) streamlines in B to the CST:
    s_ids_A_mapping_B_1nn = kdt.query(dr_sA_in_B, k=1, return_distance=False).squeeze()
    # 1nn is the initial mapping of the CST and we add the prototypes with the best mapping computed before:
    mapping_A_in_B = s_ids_A_mapping_B_1nn.copy()

    # computing the dissimlarity matrix of the random streamlines in A
    dm_sA = bundles_distances_mam(streamlines_A[s_ids_A], streamlines_A[s_ids_A])
    # computing the initial dissimlarity matrix of the mapped random streamlines in B
    dm_sA_mapping_B = bundles_distances_mam(streamlines_B[mapping_A_in_B], streamlines_B[mapping_A_in_B])

    l = loss(dm_sA, dm_sA_mapping_B)
    print "Initial Loss:", l

    # computing the centroid of CST in the dissimilarity space:
    centroid_dr = dr_sA_in_B.mean(0)
    # getting the closest (1nn) streamlines in B to the centroid:
    # retrieve a neighborhood of phi(centroid):   (phi(centroid) excluded)
    neighbors = kdt.query(centroid_dr, k=neighborhood_size, return_distance=False).squeeze()[1:]

    best_mapping = mapping_A_in_B
    best_loss = l
    for i, candidate in enumerate(neighbors):
        candidate_dr = dr_B[candidate]
        delta = candidate_dr - centroid_dr
        dr_sA_in_B_new = dr_sA_in_B + delta
        s_ids_A_mapping_B_new = kdt.query(dr_sA_in_B_new, k=1, return_distance=False).squeeze()
        mapping_A_in_B_new = s_ids_A_mapping_B_new.copy()
        dm_sA_mapping_B_new = bundles_distances_mam(streamlines_B[mapping_A_in_B_new], streamlines_B[mapping_A_in_B_new])
        l = loss(dm_sA, dm_sA_mapping_B_new)
        print i, ") Loss:", l
        if l < best_loss:
            best_loss = l
            best_mapping = mapping_A_in_B_new
            print "Best Loss:", best_loss

    # extracting the optimized mapping of CST
    cst_ids_A_mapping_B = best_mapping

    if show:
        # show 1nn mapped CST A
        # show optimized mapped CST A
        r = fvtk.ren()
        # show CST A
        # Pi_viz_A = fvtk.streamtube(streamlines_A[cst_ids_A], fvtk.colors.red)
        # fvtk.add(r, Pi_viz_A)
        Pi_viz_A_1nn = fvtk.streamtube(streamlines_B[cst_ids_A_mapping_B_1nn], fvtk.colors.blue)
        fvtk.add(r, Pi_viz_A_1nn)
        Pi_viz_A_optimized = fvtk.streamtube(streamlines_B[cst_ids_A_mapping_B], fvtk.colors.green)
        fvtk.add(r, Pi_viz_A_optimized)
        fvtk.show(r)
