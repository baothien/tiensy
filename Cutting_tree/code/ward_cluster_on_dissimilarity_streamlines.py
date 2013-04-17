#print __doc__

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
#from sklearn.cluster import Ward
from hierarchical import Ward
from sklearn.datasets.samples_generator import make_swiss_roll

def R_function(parameters, x_range, x):
    i = 0
    while (i < x_range and x>=1.*i/x_range):        
            i = i + 1
    i = i -1
    #print 'x = ', x, 'i = ',i
    return parameters[i][0]* x * x + parameters[i][1]*x + parameters[i][2]
    


###############################################################################
figure = 'big_dataset' # 'small_dataset' #     
if figure=='small_dataset':
    filename = 'ALS_Data/210/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'
    prototype_policies = ['random', 'fft', 'sff']
    color_policies = ['ko--', 'kx:', 'k^-']
elif figure=='big_dataset':
    filename = 'ALS_Data/210/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
    prototype_policies = ['random', 'sff']
    color_policies = ['ko--', 'k^-']    
num_trks = [0]#15000]#[0,15000,10000,5000,1000,500]
num_clusters = [50]#[150,50,50,50,50,50]
num_prototypes = 40


print "Loading tracks."
dpr = Dpy(filename, 'r')
tracks = dpr.read_tracks()
dpr.close()
tracks = np.array(tracks, dtype=np.object)
t_batch=0
t_batch_random=0
for i in range(len(num_trks)):     
    ##############################################################################
    # define some parameters
    num_cluster = num_clusters[i] 
    print 'number of cluster: ', num_cluster
    num_trk = num_trks[i]        
    if num_trk!=0:
        tracks = tracks[:num_trk]
    print "tracks:", tracks.size

    t0 = time.time()
    data_disimilarity = compute_disimilarity(tracks, bundles_distances_mam, prototype_policies[0], num_prototypes,tracks.size)
    t_disi = time.time() - t0
    print 'Time for dissimilarity: ', t_disi
    X = data_disimilarity


'''
    # Generate data (swiss roll dataset)
n_samples = 8
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise)
    # Make it thinner
X[:, 1] *= .5
'''

'''
###############################################################################
# Compute clustering
print "Compute unstructured hierarchical clustering..."
st = time.time()
ward = Ward(compute_full_tree=True).fit(X)
label = ward.labels_
print "Elapsed time: ", time.time() - st
print "Number of points: ", label.size


###############################################################################
# Plot result
fig = pl.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
              'o', color=pl.cm.jet(np.float(l) / np.max(label + 1)))
pl.title('Without connectivity constraints')
#stop
'''
###############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors

from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=50)

###############################################################################
# Compute clustering
print "Compute structured hierarchical clustering..."
st = time.time()
#ward = Ward(n_clusters=num_cluster, connectivity=connectivity).fit(X)
ward = Ward(compute_full_tree=True, connectivity=connectivity).fit(X)
label = ward.labels_
print "Elapsed time: ", time.time() - st
print "Number of points: ", label.size
#stop
#-----------------------------------------------------------------------------

# saving the result
from dipy.io.pickles import save_pickle,load_pickle
#save_pickle('210_15000_tracks_first_trial.tree',ward)
#save_pickle('210_full_tracks_first_trial.tree',ward)
#print 'Saving tree: done'
#ward = load_pickle('210_full_tracks_first_trial.tree')
#-----------------------------------------------------------------------------

###############################################################################
# Plot result
fig = pl.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
              'o', color=pl.cm.jet(float(l) / np.max(label + 1)))
pl.title('With connectivity constraints')

###############################################################################
# plot the R_alpha function
fig = plt.figure()
x = np.linspace(0, 1, 2000)
y = [R_function(ward.R_alpha_,ward.height_[len(ward.height_)-1],x_i) for x_i in x]
ax = fig.add_subplot(111)
ax.plot(x, y, 'o')
ax.set_xlabel('scale: ' + str(1./ward.height_[len(ward.height_)-1]))
ax.set_ylabel('R_alpha')
plt.title('R_alpha fucntion')
plt.show()


    
        
    

