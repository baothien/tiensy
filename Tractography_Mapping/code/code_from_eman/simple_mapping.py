from dipy.io.dpy import Dpy
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.viz import fvtk

if __name__ == '__main__':

    np.random.seed(0)

    filename = 'data/tracks_dti_10K_linear.dpy'
        
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)

    size1 = 100
    size2 = 100
    tracks1 = tracks[np.random.permutation(len(tracks))[:size1]]
    tracks2 = tracks[np.random.permutation(len(tracks))[:size2]]
    # solution = np.random.permutation(len(tracks1))
    # tracks2 = tracks1[solution]
    # inverse_solution = np.argsort(solution)

    dm1 = bundles_distances_mam(tracks1, tracks1)
    dm2 = bundles_distances_mam(tracks2, tracks2)

    print "For each s1 and for each s2 we reorder their distances and create a mapping from the first ordered ids to the second ordered ids"
    print "Then we compute the loss of this mapping"
    print "This approach is suboptimal but should provide a very good guess."

    idx2_best = None
    loss_best = 100000000
    mapping12_best = None
    for i1 in range(size1):
        idx1 = np.argsort(dm1[i1])

        for i2 in range(size2):
            idx2 = np.argsort(dm2[i2])
            mapping12 = np.argsort(idx2[np.argsort(idx1)]) # this line is tricky and create the mapping as desiderd. It works correctly because if tracks2 is just a reshuffling of tracks1 then it leads to loss=0, as expected.
            loss = np.linalg.norm(dm2 - dm1[mapping12][:,mapping12])
            # print i2, loss
            if loss < loss_best:
                idx2_best = idx2
                loss_best = loss
                mapping12_best = mapping12
                actual_distance = np.sum([mam_distances(tracks1[mapping12_best][i], tracks2[i]) for i in range(size1)])
                print i1, i2, loss, actual_distance
        

    tracks3 = tracks1[mapping12_best].copy()
    for i in range(size1):
        tracks3[i] = tracks3[i] - tracks3[i].mean(0)
        tracks3[i] = tracks3[i] + tracks2[i].mean(0)
    
    r = fvtk.ren()
    # s = fvtk.streamtube(tracks2, fvtk.colors.carrot)
    # fvtk.add(r, s)
    # s = fvtk.streamtube(tracks3, fvtk.colors.blue)
    # fvtk.add(r, s)
    # s = fvtk.streamtube(tracks, fvtk.colors.red)
    # fvtk.add(r, s)
    fvtk.show(r)
