from __future__ import division
import numpy as np


from data_processing import compute_correlation_tracks
from interface import load_tracks, visualize

def init():
    filename_tracks = '/data/subj_01/101_32/DTI/tracks_dti_3M.dpy'
    k = 10
    num_prototypes = [5,10,20,30,50]
    prototype_policies =  ['sff', 'kcenter','subset']    
    iterations = 50
    
    distance = 'euclidean'
    metric_original = 'euclidean'
    metric_dissimilarity = 'euclidean'    
    

if __name__ == '__main__':

#    init()
    filename_tracks = 'data/subj_01/101_32/DTI/tracks_dti_10K.dpy'    
    num_prototypes = [5,10,20,30,50]
    prototype_policies =  ['sff', 'kcenter','subset']    
    iterations = 50
    
    distance = 'euclidean'
    metric_original = 'euclidean'
    metric_dissimilarity = 'euclidean'    
           
    for prototype_policy in prototype_policies:
        print("Prototype policy: %s" % prototype_policy)    
        score_sample_list = [] 
     
        print('Loading data.')
        print filename_tracks       
        tracks = load_tracks(filename_tracks)
        
        score_sample = compute_correlation_tracks (tracks, prototype_policy,num_prototypes,iterations)
        score_sample_list.append(score_sample)            
       
        score_sample = np.hstack(score_sample_list)
    
        score_mean = score_sample.mean(1)
        score_std = score_sample.std(1)
        print("Average Score corr: %s" % score_mean)
        print("Std Score corr: %s" % score_std)
        
        visualize(score_mean, score_std, num_prototypes,iterations)
        
        