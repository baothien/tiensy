import numpy as np
from dipy.io.dpy import Dpy
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from sklearn.metrics import pairwise_distances
from dipy.viz import fvtk
import networkx as nx
from sklearn.neighbors import KDTree, BallTree

def avg_mam_distance_numpy(s1, s2):
    """Just NumPy with very compact broadcasting
    """
    dm = s1[:, None, :] - s2[None, :, :]
    dm = (dm * dm).sum(2)
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())

if __name__ == '__main__':
    
    filename = '/home/ele/datasets/Garyfallidis_2011/PROC_MR10032_CLEAN/subj_01/101_32/DTI/tracks_dti_10K_linear.dpy'
        
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object) # [:100]

    dm = bundles_distances_mam(tracks, tracks)

    # Problem:
    # Given a streamline s1 find the most similar streamline s2!=s1
    # where similarity is in term of similarity of the neighbouroods
    # (in terms of distances) between s1 and s2.
    # Idea for a solution:
    # 0) sort distances of tracks from s0 = d0
    # 1) for each s1 in the neighbourood of s0:
    # 1.1) sort distances of tracks from s1 = d1
    # 2) for each s2 in tracks:
    # 2.1) sort distances of tracks from s2 = d2
    # 2.2) tot = the Euclidean distance between d2 and d0
    # 2.3) for each s3 in the neighbourhood of s2, (closests first):
    # 2.3.1) sort distances of tracks from s3 = d3
    # 2.4.1) compute minimum the Euclidean distance between d3 and all d1s
    # 2.4.2) sum this minimum distance to tot
    # tot is the final distance between s0 and s2
    # the s2 with minimum distance is the desired streamline

    np.random.seed(0)
    prototypes_id = np.random.permutation(dm.shape[0])[:200]
    dp = dm[:,prototypes_id] # dissimilarity projection
    
    kdt = BallTree(dp) # KDTree(dp)
    
    radius = 100
    k = 10

    sid = 9
    
    idx1 = kdt.query_radius(dp[sid], radius)[0]
    # idx1 = kdt.query(dp[sid], k)[1][0]
    dm_small1 = dm[idx1][:,idx1]
    e1 = dm_small1[np.triu_indices(dm_small1.shape[0],1)]

    spgk = np.zeros(dm.shape[0])
    for i in range(dm.shape[0]):
        idx2 = kdt.query_radius(dp[i], radius)[0]
        # idx2 = kdt.query(dp[i], k)[1][0]
        dm_small2 = dm[idx2][:,idx2]
        e2 = dm_small2[np.triu_indices(dm_small2.shape[0],1)]

        spgk[i] = np.multiply.outer(np.exp(-e1), np.exp(-e2)).sum()
        print i, spgk[i]

    
    r = fvtk.ren()
    # lines = tracks
    # c = fvtk.streamtube(lines, fvtk.colors.red)
    # fvtk.add(r,c)
    # lines = tracks[np.argsort(spgk)[-30:]]
    # c = fvtk.streamtube(lines, fvtk.colors.red)
    # fvtk.add(r,c)
    
    lines = tracks[idx1]
    c = fvtk.streamtube(lines, fvtk.colors.red)
    fvtk.add(r,c)
    lines = [tracks[sid]]
    c = fvtk.streamtube(lines, fvtk.colors.cyan)
    fvtk.add(r,c)
    # lines = tracks[idx1]
    # c = fvtk.streamtube(lines, fvtk.colors.cyan)
    # fvtk.add(r,c)
    # lines = tracks[sid]
    # c = fvtk.streamtube(lines, fvtk.colors.cyan)
    # fvtk.add(r,c)

    best = np.argsort(spgk)[-1] # np.argmax(spgk)
    lines = tracks[kdt.query_radius(dp[best], radius)[0]]
    c = fvtk.streamtube(lines, fvtk.colors.carrot)
    fvtk.add(r,c)
    lines = [tracks[best]]
    c = fvtk.streamtube(lines, fvtk.colors.green)
    fvtk.add(r,c)

    fvtk.show(r)
     
