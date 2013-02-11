from dipy.io.dpy import Dpy
import numpy as np
import matplotlib.pyplot as plt


def load_tracks(filename):
    
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    tracks = tracks[:100]
    print "tracks:", tracks.size
    
    return tracks

def visualize(score_mean,score_std, nums_prototypes,iterations):    

    plt.plot(nums_prototypes, score_mean, 'k')
    std_mean = score_std / np.sqrt(iterations)
    plt.xlabel('Number of prototypes ($p$)')
    plt.ylabel('Correlation')                
    plt.plot(nums_prototypes, score_mean + 2 * std_mean, 'r')
    plt.plot(nums_prototypes, score_mean - 2 * std_mean, 'r')
    plt.show()

   
     