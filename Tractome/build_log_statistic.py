# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:35:51 2014

@author: bao
"""

import argparse
import numpy as np
import pickle

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# these functions are copied from the common_functions.py
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Inertia(tract):
    
    from dissimilarity_common import compute_dissimilarity
    from dipy.tracking.distances import bundles_distances_mam
    from sklearn import metrics
    from sklearn.cluster import MiniBatchKMeans#, KMeans
    #from sklearn.metrics.pairwise import euclidean_distances
    diss = compute_dissimilarity(tract, distance=bundles_distances_mam, prototype_policy='sff', num_prototypes=20)
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=1, batch_size=100,
                              n_init=10, max_no_improvement=10, verbose=0)        
    mbk.fit(diss)
        
    labels = mbk.labels_        
    print labels    
    #labels = np.zeros(len(tract))
    #sil = metrics.silhouette_score(diss, labels, metric='euclidean')
    return mbk.inertia_#, sil 
    
def Shannon_entropy(tract):
    '''
    compute the Shannon Entropy of a set of tracks as defined by Lauren
    H(A) = (-1/|A|) * sum{log(1/|A|)* sum[p(f_i|f_j)]} 
    where p(f_i|f_j) = exp(-d(f_i,f_j)*d(f_i,f_j))
    '''
    from dipy.tracking.distances import bundles_distances_mam
    import numpy as np
    #if number of fiber is too large, just sample
    if len(tract) > 3500:
        #from dissimilarity_common import subset_furthest_first
        #prototype_idx = subset_furthest_first(tract, 500, bundles_distances_mam)
        #prototype = [tract[i] for i in prototype_idx]
        
        prototype_idx = np.random.permutation(tract.shape[0])[:3500]
        prototype = [tract[i] for i in prototype_idx]
        
        dm = bundles_distances_mam(prototype, prototype)
    else:
        dm = bundles_distances_mam(tract, tract)
    
    dm = np.array(dm, dtype =float)
    
    dm2 = dm**2
    
    A = len(dm)
    theta = 10.
    
    pr_all = np.exp((-dm**2)/theta)
    
    pr_i = (1./A) * np.array([sum(pr_all[i]) for i in np.arange(A)])
        
    entropy = (-1./A) * sum([np.log(pr_i[i]) for i in np.arange(A)])
    print entropy
    return entropy
    
def load_whole_tract(tracks_filename):
    
    from dipy.io.pickles import load_pickle
    if (tracks_filename[-3:]=='dpy'):
        from dipy.io.dpy import Dpy
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        import nibabel as nib
        streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
        all_tracks = np.array([s[0] for s in streams], dtype=np.object)    

    all_tracks = np.array(all_tracks,dtype=np.object)
    return all_tracks

def load_tract(tracks_filename, id_file):
    

    from dipy.io.pickles import load_pickle
    if (tracks_filename[-3:]=='dpy'):
        from dipy.io.dpy import Dpy
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        import nibabel as nib
        streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
        all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]
    
    tract = np.array(tract,dtype=np.object)
    return tract
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Creating the log of segmentation from the seglog file",
                                 epilog="Written by Bao Thien Nguyen, bao@fbk.eu",
                                 version='1.0')

parser.add_argument(
                    'inputTractography',
                    help='The file name of whole-brain tractography in .dpy/.trk format')
parser.add_argument(
                    'inputSegmentation',
                    help='The file name of the segmentation in .dpy/.trk format')
parser.add_argument(
                    'inputLog',
                    help='The file name of the log segmentation.')
                    
#parser.add_argument(
#                    'inputTargetCSTIndex',
#                    help='The file name of target CST index')

args = parser.parse_args()

print "=========================="
print "Tractography: ", args.inputTractography
print "Segmentation: ", args.inputSegmentation
print "Log (history) file: ", args.inputLog
print "=========================="

tracto_file = args.inputTractography
seg_file = args.inputSegmentation
log_file = args.inputLog


tractography = load_whole_tract(tracto_file)
segmentation = load_whole_tract(seg_file)

print 'Number of fiber in segmentation', len(segmentation)

print "Loading history session file"
log_info = pickle.load(open(log_file)) 
state = log_info['segmsession']  

str_file = log_info['structfilename']
tract_file = log_info['tractfilename']

print str_file, '\t', tract_file

print 'Len - Cluster ', len(state['clusters'])
#print 'Selected', state['selected']

#print 'History', state['simple_history']
#print 'History pointer', state['simple_history_pointer']
history = state['simple_history']
pointer = len(history)#state['simple_history_pointer']
print 'Len of history', len(history)
#print 'History', history[pointer]

pre_total_fiber = len(tractography)
cur_pointer = 0

log_seg = []

while cur_pointer < pointer:
    cur_state = history[cur_pointer]
    total_fiber_state = 0
    tract_ids_state = []
    for medoid, ids in cur_state.iteritems():
        total_fiber_state = total_fiber_state + len (ids)
        for j in ids:
            tract_ids_state.append(j)
    tract = [tractography[i] for i  in tract_ids_state]
    tract = np.array(tract,dtype=np.object)
    print len(tract)
    entr = Shannon_entropy(tract)
    iner = Inertia(tract)/len(tract)    
    
    
    if total_fiber_state != pre_total_fiber:
        #action_type, number of clusters selected, number of fiber remain
        log_seg.append(['Remove',len(cur_state),total_fiber_state, entr, iner])
    else:
        log_seg.append(['Recluster',len(cur_state),total_fiber_state, entr, iner])
        
    cur_pointer = cur_pointer + 1
    pre_total_fiber = total_fiber_state

       
        
print
print 'Log of action: action_type, number of clusters selected, number of fiber remain\n',
for i in np.arange(pointer):
    
    print 'Action ', i,': ', log_seg[i][0], log_seg[i][1], log_seg[i][2], log_seg[i][3], log_seg[i][4]
    #print 'Action ', i,': ', log_seg[i]

print 'Number of fiber in segmentation', len(segmentation)

'''
seg_info={'structfilename':self.structpath, 'tractfilename':self.tracpath, 'segmsession':state}



       
def load_segmentation(self, segpath=None):
        """
        Loading file containing a previous segmentation
        """
        print "Loading saved session file"
        segm_info = pickle.load(open(segpath)) 
        state = segm_info['segmsession']  
            
        self.structpath=segm_info['structfilename']
        self.tracpath=segm_info['tractfilename']   

        # load T1 volume registered in MNI space
        print "Loading structural information file"

        self.loading_structural(self.structpath)

        # load tractography
        self.loading_full_tractograpy(self.tracpath)
        
        #self.streamlab.set_state(state)        
        # the history and other things are saved in state
        self.clusters_reset(state['clusters'])
        self.selected = state['selected']
        self.expand = state['expand']
        self.simple_history = state['simple_history']
        self.simple_history_pointer = state['simple_history_pointer']
        
            
        self.scene.update()

def get_state(self):
        """Create a dictionary from which it is possible to
        reconstruct the current Manipulator.
        """
        state = {}
        state['clusters'] = copy.deepcopy(self.clusters)
        state['selected'] = copy.deepcopy(self.selected)
        state['expand'] = self.expand
        state['simple_history'] = copy.deepcopy(self.simple_history)
        state['simple_history_pointer'] = copy.deepcopy(self.simple_history_pointer)
        return state
def set_state(self, state):
        """Set the current object with a given state. Useful to
        serialize the Manipulator to file etc.
        """
        self.clusters_reset(state['clusters'])
        self.selected = state['selected']
        self.expand = state['expand']
        self.simple_history = state['simple_history']
        self.simple_history_pointer = state['simple_history_pointer']
        self.recluster_action()
'''