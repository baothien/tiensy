# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:51:23 2012

@author: bao
"""

from __future__ import division
import os
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

from kcenter import subset_furthest_first_tracks, furthest_first_traversal_tracks

def my_squareform (arr, n):
    
    #b = [a[i,j] for i in np.arange(n) and j in np.arange(i+1,n) ]
    b = []        
    for i in np.arange(n-1):
        for j in np.arange(i+1,n):            
            b.append(arr[i,j])            
    return b

def compute_correlation_tracks (tracks_file_name, prototye_policy,nums_prototypes, iterations):
    dist_thr = 10.0
    pts = 12
    epsilon = 15#1.0
    
    np.random.seed(0)
    
    print "Loading tracks."
    dpr = Dpy(tracks_file_name, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    print "tracks:", tracks.size        
   
   
    print('Computing QuickBundles:'),
    qb = QuickBundles(tracks, dist_thr=dist_thr, pts=pts)    
    tracks_downsampled = qb.downsampled_tracks()
    tracks_qb = qb.virtuals()
    tracks_qbe, tracks_qbei = qb.exemplars()
    print(len(tracks_qb))

    tracks = tracks_downsampled[:100]       
    #tracks = [tracks_downsampled[i] for i in [1,2,3,4,5,10]]       
    
    verbose = False
    if verbose: print("Computing distance matrix and similarity matrix (original space):"),
#        original_distances = pdist(data, metric='euclidean')
#        od = squareform(original_distances)        
    od = bundles_distances_mdf(tracks,tracks)     
    #print od      
    n = len(tracks)
    original_distances = my_squareform(od,n)
    #print original_distances
    #stop
    
    num_points = len(tracks)
    score_sample = np.zeros((len(nums_prototypes), iterations))
    print "num prototypes: "        
    for j, num_prototypes in enumerate(nums_prototypes):
        print num_prototypes,
        stdout.flush()
        print 'k = '
        for k in range(iterations):
            print k                                
            if verbose: print("Generating %s prototypes as" % num_prototypes),
            if prototype_policy=='subset':
                if verbose: print("random subset of the initial data.")
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
            
            if verbose: print("Computing dissimilarity matrix.")                
            data_dissimilarity = bundles_distances_mdf(tracks, prototype)
            #prototype_dissimilarity = cdist(prototype, prototype, metric=distance)
           
            if verbose: print("Computing distance matrix (dissimilarity space).")
            dissimilarity_distances = pdist(data_dissimilarity, metric='euclidean')
            dd = squareform(dissimilarity_distances)

            if verbose: print("Compute correlation between distances before and after projection.")
            #correlation_pval_od_dd = correlation(original_distances, dissimilarity_distances)
            correlation_pval_od_dd = correlation(original_distances, dissimilarity_distances) 
            print correlation_pval_od_dd                             
            if verbose: print("correlation_od_dd = %s" % correlation_pval_od_dd[0])

            idx_epsilon = original_distances<epsilon
            #correlation_pval_epsilon = correlation(original_distances[idx_epsilon], dissimilarity_distances[idx_epsilon])
            correlation_pval_epsilon = correlation(od[idx_epsilon], dd[idx_epsilon])                 
            if verbose: print("correlation_od_dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon[0]))
            
            original_distances_unbiased = od.diagonal(1)[::2]
            dissimilarity_distances_unbiased = dd.diagonal(1)[::2]
            correlation_pval_unbiased = correlation(original_distances_unbiased, dissimilarity_distances_unbiased)
 
            if verbose: print("Unbiased correlation od dd = %s" % correlation_pval_unbiased[0])
       
            idx_epsilon_unbiased = original_distances_unbiased<epsilon
            correlation_pval_epsilon_unbiased = correlation(original_distances_unbiased[idx_epsilon_unbiased], dissimilarity_distances_unbiased[idx_epsilon_unbiased])
           
            #if verbose:
            print("Unbiased correlation od dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon_unbiased[0]))                               
            
            score_sample[j, k] = correlation_pval_epsilon_unbiased[0]

    return score_sample


if __name__ == '__main__':

#    a = np.arange(16) 
#    b = a.reshape(4,4)
#    print b
#    c = my_squareform(b,4)
#    print c
#    stop    
      
    prototype_policies = ['sff', 'kcenter','subset']#['subset', 'kcenter', 'sff']
    #prototype_policy =  'kcenter' #'subset' # 'kcenter' # 'sff'   # 'draw' #'cover'  #  'kmeans' # 'kmeans++' # 
    iterations = 50#80#50 # number of iterations when drawing prototypes at random
    nums_prototypes = range(1,400,10)#(1,20)#(1,40)#(1,20)#20 # num_points    

    distance = 'euclidean' # 'canberra' # 'mahalanobis' # 'hamming' # 'chebyshev' # 'correlation' # 'cosine' # 
    metric_original = 'euclidean'
    metric_dissimilarity = 'euclidean'    
       
    for prototype_policy in prototype_policies:
        print("Distance function used to create the dissimilarity representation: %s" % distance)
        print("Domain metric of the problem at hand (used to compute dm_original): %s" % metric_original)
        print("Metric between vectors of the dissimilarity space: %s" % metric_dissimilarity)
        print("Prototype policy: %s" % prototype_policy)
    
        score_sample_list = [] 
     
        dirname = "data/"
        for root, dirs, files in os.walk(dirname):
            if root.endswith('101_32'):
                filename = root+'/DTI/tracks_dti_10K.dpy'              
                print('Loading data.')
                print filename
        
        #   add in 120217
        #filename = 'data/subj_05/101_32/DTI/tracks_dti_10K.dpy'
     
#----------------------------------------------------------------------------------     
#        delta = 10.0 # in mm.
#        min_length = 5.0 # in mm
#
#        print "Loading tracks."
#        dpr = Dpy(filename, 'r')
#        tracks = dpr.read_tracks()
#        dpr.close()
#        tracks = np.array(tracks, dtype=np.object)
#        print "tracks:", tracks.size        
#       
#       
#        print('Computing QuickBundles:'),
#        qb = QuickBundles(tracks, dist_thr=dist_thr, pts=pts)    
#        tracks_downsampled = qb.downsampled_tracks()
#        tracks_qb = qb.virtuals()
#        tracks_qbe, tracks_qbei = qb.exemplars()
#        print(len(tracks_qb))
#
#        tracks = tracks_downsampled       
#        #tracks = [tracks_downsampled[i] for i in [1,2,3,4,5,10]]       
#        
#        if verbose: print("Computing distance matrix and similarity matrix (original space):"),
##        original_distances = pdist(data, metric='euclidean')
##        od = squareform(original_distances)        
#        od = bundles_distances_mdf(tracks,tracks)     
#        #print od      
#        n = len(tracks)
#        original_distances = my_squareform(od,n)
#        #print original_distances
#        #stop
#        
#        num_points = len(tracks)
#        score_sample = np.zeros((len(nums_prototypes), iterations))
#        print "num prototypes: "        
#        for j, num_prototypes in enumerate(nums_prototypes):
#            print num_prototypes,
#            stdout.flush()
#            print 'k = '
#            for k in range(iterations):
#                print k                                
#                if verbose: print("Generating %s prototypes as" % num_prototypes),
#                if prototype_policy=='subset':
#                    if verbose: print("random subset of the initial data.")
#                    prototype_idx = np.random.permutation(num_points)[:num_prototypes]
#                    prototype = [tracks[i] for i in prototype_idx]
#                elif prototype_policy=='kcenter':
#                    prototype_idx = furthest_first_traversal_tracks(tracks, num_prototypes)
#                    prototype = [tracks[i] for i in prototype_idx]
#                elif prototype_policy=='sff':
#                    prototype_idx = subset_furthest_first_tracks(data, num_prototypes)
#                    prototype = [tracks[i] for i in prototype_idx]                
#                else:
#                    raise Exception                
#                
#                if verbose: print("Computing dissimilarity matrix.")                
#                data_dissimilarity = bundles_distances_mdf(tracks, prototype)
#                #prototype_dissimilarity = cdist(prototype, prototype, metric=distance)
#               
#                if verbose: print("Computing distance matrix (dissimilarity space).")
#                dissimilarity_distances = pdist(data_dissimilarity, metric='euclidean')
#                dd = squareform(dissimilarity_distances)
#
#                if verbose: print("Compute correlation between distances before and after projection.")                
#                correlation_pval_od_dd = correlation(original_distances, dissimilarity_distances) 
#                print correlation_pval_od_dd                             
#                if verbose: print("correlation_od_dd = %s" % correlation_pval_od_dd[0])
#
#                idx_epsilon = original_distances<epsilon
#                #correlation_pval_epsilon = correlation(original_distances[idx_epsilon], dissimilarity_distances[idx_epsilon])
#                correlation_pval_epsilon = correlation(od[idx_epsilon], dd[idx_epsilon])                 
#                if verbose: print("correlation_od_dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon[0]))
#                
#                original_distances_unbiased = od.diagonal(1)[::2]
#                dissimilarity_distances_unbiased = dd.diagonal(1)[::2]
#                correlation_pval_unbiased = correlation(original_distances_unbiased, dissimilarity_distances_unbiased)
# 
#                if verbose: print("Unbiased correlation od dd = %s" % correlation_pval_unbiased[0])
#           
#                idx_epsilon_unbiased = original_distances_unbiased<epsilon
#                correlation_pval_epsilon_unbiased = correlation(original_distances_unbiased[idx_epsilon_unbiased], dissimilarity_distances_unbiased[idx_epsilon_unbiased])
#               
#                if verbose: print("Unbiased correlation od dd (d<=%s) = %s" % (epsilon, correlation_pval_epsilon_unbiased[0]))                               
#
#                score_sample[j, k] = correlation_pval_epsilon_unbiased[0]
#--------------------------------------------------------------------------------------                
        
                score_sample = compute_correlation_tracks (filename, prototype_policy,nums_prototypes,iterations)
                score_sample_list.append(score_sample)
            
        #stop                
        
        score_sample = np.hstack(score_sample_list)
    
        score_mean = score_sample.mean(1)
        score_std = score_sample.std(1)
        print("Average Score corr: %s" % score_mean)
        print("Std Score corr: %s" % score_std)
        
        visualize = False#True #False
        if visualize:
            plt.plot(nums_prototypes, score_mean, 'k')
            std_mean = score_std / np.sqrt(iterations)
            plt.xlabel('number of prototypes ($p$)')
            plt.ylabel('correlation')
            title =   '120301_20h_400_10pro_c10_tracks_score_gen_'+prototype_policy
            plt.title(title)
            plt.plot(nums_prototypes, score_mean + 2 * std_mean, 'r')
            plt.plot(nums_prototypes, score_mean - 2 * std_mean, 'r')
            plt.show()
    
        filename = '120301_20h_400_10pro_c10_tracks_score_gen_'+prototype_policy+'.npy'
        print "Saving results in", filename
        np.save(filename, score_sample)
        filename_pickle = '120301_20h_400_10pro_c10_tracks_score_gen_'+prototype_policy+'.pickle'
        pickle.dump(score_sample_list, open(filename_pickle, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        
