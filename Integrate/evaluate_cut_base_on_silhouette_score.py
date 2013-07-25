# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:43:44 2013

@author: bao
"""
import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
#from sklearn.cluster import Ward
#from sklearn.cluster.hierarchical import _hc_cut
from hierarchical import Ward, _hc_cut
from dipy.io.pickles import save_pickle,load_pickle

#from silhouette_score_modified import *
from silhouette_score_modified_parallel import *
from evaluate_cut_base_on_split_factor import remove_valley, heuristic_modified_cuts

#for computing the CPU time, not elapsed time
import resource
def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]
 
'''
Load the dissimilarity approximation of the real tractography
data_id: 101, ..., 109, 201, ..., 213 (detail in ALS dataset)
return: disimilarity of tractography with 40 prototyes 
'''
def load_data_dis(data_id,num_prototype=40):       
    
    filename = 'Results/' + str(data_id) + '/' + str(data_id)+'_data_disimilarity_full_tracks_' + str(num_prototype) + '_prototyes_random_130516.dis'       
    print "Loading tracks dissimilarity"
    dis = load_pickle(filename)
    return dis
    
'''
Load the dissimilarity approximation of the real tractography
data_id: 101, ..., 109, 201, ..., 213 (detail in ALS dataset)
return: disimilarity of tractography with 40 prototyes 
'''
def load_ward_tree(data_id,num_neigh=50):       
    
    #file_name = 'Results/' + str(data_id) + '/' + str(data_id)+'_full_tracks_' + str(num_neigh) + '_neighbors_original_ward_stop_50_clusters.tree'            
    file_name = 'Results/' + str(data_id) + '/' + str(data_id)+'_full_tracks_' + str(num_neigh) + '_neighbors_modified_ward_full_tree_130516.tree'            
    print "Loading ward tree "
    ward = load_pickle(file_name)
    return ward
    
'''
This function compute the silhouette score for the data with no limitation of the data size
This overcomes the limitation of memory issue of the current version of Silhouette score in scikit-learn
First, the whole dataset will be clustered
Second, the silhouette score is only calculated on a small set of the dataset using sample_size parameter of Silhouette Score function
Iteratively running this and then take the mean score for the whole dataset
'''
#def compute_silhouette(data, distance,prototype_policies, num_prototypes, iterations, verbose=False, size_limit=1000):
def compute_silhouette_sample_size(data_ids, prototype_policie, num_prototype, num_cluster, metric_measure, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil = np.zeros((len(data_ids), len(num_neighbors),iterations))
    time = np.zeros((len(data_ids), len(num_neighbors),iterations))

    for m, data_id in enumerate(data_ids):
        if verbose: print 'Subject/Control: ', data_id
        X = load_data_dis(data_id,num_prototype)          
        
        for j, num_neigh in enumerate(num_neighbors):  
            
            print "\tGenerating at ", num_neigh, " neighbors"                     

            #if ward is stop at 50 clusters                        
            ward = load_ward_tree(data_id,num_neigh)              
            label = ward.labels_            
            
            #if ward is full tree
            #ward = load_ward_tree(data_id,num_neigh)              
            
            #label = _hc_cut(num_cluster, ward.children_,ward.n_leaves_)            
            
            print"\t Computing silhouette score"
            for k in range(iterations): 
                st = cpu_time()                   
                #score = metrics.silhouette_score(X, label, metric = metric_measure,sample_size=20000)
                score = silhouette_score_block(X, label, metric = metric_measure,sample_size=50000,n_jobs=2)
                t = cpu_time()-st                
                sil[m,j,k] = score
                time[m,j,k] = t
                
            print '\t \t silhouette score', sil
            print '\t \t Time for silhouette score', time
    return sil, time


'''
This function compute the silhouette score for the data with no limitation of the data size
This overcomes the limitation of memory issue of the current version of Silhouette score in scikit-learn
First, the whole dataset will be clustered
Second, the silhouette score is calculated based on the modified fuction implemented by Alexandre https://gist.github.com/AlexandreAbraham/5544803
in the file silhouette_score_modified
Compute the silhoette score for the whole tree H of dataset X
S = [s_min, ..., s_max]
E(H) = 1/|S| * sum ( E(cut(s_i)) ) with all s_i in S
E(cut(s_i)) is the silhouette score of X with labels in cut(s_i)
H is a modified ward object
'''
def compute_silhouette_block_tree(X, ward, metric_measure, verbose=False):
    #extract all cut of H
    num_cuts = ward.height_[len(ward.children_) + ward.n_leaves_-1]
    if verbose==True:    
        print "Computing silhouette score :"
        print "Height : ",num_cuts
    sil = np.zeros(num_cuts)
    for height_cut in np.arange(num_cuts):
        if verbose: print ".....scales: ", height_cut
        num_clusters = len(ward.cut(height_cut))
        label = _hc_cut(num_clusters, ward.children_,ward.n_leaves_)
        iterations = 4
        temp = np.zeros(interations)
        for t in range(iterations):                                 
                score = silhouette_score_block(X, label, metric = metric_measure,sample_size=50000,n_jobs=2)                
                temp[t] = score                                        
        sil[height_cut] = temp[:].mean(0)
        if verbose: print ".....score: ", score
    mean = sil[:].mean(0)
    std = sil[:].std(0)    
    return mean, std


def compute_silhouette_block_scales(X, ward, scales, metric_measure, verbose=False):   
    
    if verbose==True:    
        print "Computing silhouette score :"
        print "Scales : ", scales
    sil = np.zeros(len(scales))
    for i in np.arange(len(scales)):
        height_cut = scales[i]
        if verbose: print ".....scales: ", height_cut
        num_clusters = len(ward.cut(height_cut))
        label = _hc_cut(num_clusters, ward.children_,ward.n_leaves_)
        iterations = 1
        temp = np.zeros(iterations)
        for t in range(iterations):  
                st = cpu_time()                                  
                score = silhouette_score_block(X, label, metric = metric_measure,sample_size=5000,n_jobs=2)                
                t = cpu_time()-st  
                if verbose: print "......time", t
                temp[t] = score                                        
        sil[i] = temp[:].mean(0)             
        if verbose: print ".....score: ", score
    mean = sil[:].mean(0)
    std = sil[:].std(0)    
    return mean, std

'''
This function compute the silhouette score for the data with no limitation of the data size
This overcomes the limitation of memory issue of the current version of Silhouette score in scikit-learn
First, the whole dataset will be clustered
Second, the silhouette score is calculated based on the modified fuction implemented by Alexandre https://gist.github.com/AlexandreAbraham/5544803
in the file silhouette_score_modified
'''
def compute_silhouette_block_whole_tree(data_ids, prototype_policie, num_prototype, metric_measure, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil_mean = np.zeros((len(data_ids), len(num_neighbors),iterations))
    sil_std = np.zeros((len(data_ids), len(num_neighbors),iterations))

    for m, data_id in enumerate(data_ids):
        if verbose: print 'Subject/Control: ', data_id
        X = load_data_dis(data_id,num_prototype)          
        
        for j, num_neigh in enumerate(num_neighbors):            
            stdout.flush()
            if verbose: print "\tGenerating at ", num_neigh, " neighbors"                     
                        
            ward = load_ward_tree(data_id,num_neigh)                        
            
            if verbose: print"\t Computing silhouette score"
            for k in range(iterations):                                    
                score_mean, score_std  = compute_silhouette_block_tree(X, ward, metric_measure, verbose)                
                sil_mean[m,j,k] = score_mean
                sil_std[m,j,k] = score_std
            if verbose: print sil_mean, '\t', sil_std
    return sil_mean, sil_std

def compute_silhouette_block_best_cuts(data_ids, prototype_policie, num_prototype, metric_measure, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil_mean = np.zeros((len(data_ids), len(num_neighbors),iterations))
    sil_std = np.zeros((len(data_ids), len(num_neighbors),iterations))

    for m, data_id in enumerate(data_ids):
        if verbose: print 'Subject/Control: ', data_id
        X = load_data_dis(data_id,num_prototype)          
        
        for j, num_neigh in enumerate(num_neighbors):            
            stdout.flush()
            if verbose: print "\tGenerating at ", num_neigh, " neighbors"                     
                        
            ward = load_ward_tree(data_id,num_neigh)
            
            #extract the best cuts
            cut = ward.best_cut()
            print 'origin cut', cut
            remove_valley(cut)
            print 'after remove valley', cut          
                   
            cut_scales_ori = [s[0] for s in cut] 
            temp_scales = heuristic_modified_cuts(cut_scales_ori[:4],3)
            temp_scales_1 = heuristic_modified_cuts(cut_scales_ori[4:],4,temp_scales[len(temp_scales)-1])
            cut_scales = np.concatenate((temp_scales,temp_scales_1))
            if verbose: print cut_scales_ori
            if verbose: print cut_scales
            
            
            if verbose: print"\t Computing silhouette score at best scales"
            for k in range(iterations):                                    
                score_mean, score_std  = compute_silhouette_block_scales(X, ward,cut_scales, metric_measure, verbose)                
                sil_mean[m,j,k] = score_mean
                sil_std[m,j,k] = score_std
            if verbose: print sil_mean, '\t', sil_std
    return sil_mean, sil_std


###############################################################################
if __name__ == '__main__':

    np.random.seed(0)      
    
    data_ids = [101]#,201,109,210]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [50]#[25 ,50, 75, 100]
    iterations = 1
    prototype_policies = ['random', 'fft', 'sff']   
   
    verbose = True    
       
    #if verbose: print "Compute silhouette score for whole tree "
    #sil_mean, sil_std = compute_silhouette_block_whole_tree(data_ids, prototype_policies[0], 
    #                                                      num_prototype, 'euclidean', 
    #                                                      num_neighbors, iterations, verbose)
    
    if verbose: print "Compute silhouette score for best cuts"
    sil_mean_best_cuts, sil_std_best_cuts = compute_silhouette_block_best_cuts(data_ids, prototype_policies[0], 
                                                          num_prototype, 'euclidean', 
                                                          num_neighbors, iterations, verbose)
    
    
    '''    
    filename = 'Results/all_subjects_silhouette_time_sub_sample_50K_iterations_8.txt.txt'       
    time =load_pickle(filename)
    filename = 'Results/all_subjects_silhouette_score_sub_sample_50K_iterations_8.txt.txt'       
    sil =load_pickle(filename)
    '''

    #save_pickle('all_subjects_silhouette_score_sub_sample_50K_iterations_8_140513.txt',sil)
    #save_pickle('all_subjects_silhouette_time_sub_sample_50K_iterations_8_140513.txt',time)
        
   