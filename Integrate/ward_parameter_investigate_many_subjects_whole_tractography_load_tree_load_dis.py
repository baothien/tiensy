# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:56:23 2013

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
from sklearn.cluster import Ward
from sklearn.cluster.hierarchical import _hc_cut
#from hierarchical import Ward
from dipy.io.pickles import save_pickle,load_pickle

#from silhouette_score_modified import *
from silhouette_score_modified_parallel import *

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
    
    filename = 'Results/' + str(data_id) + '/' + str(data_id)+'_data_disimilarity_full_tracks_' + str(num_prototype) + '_prototyes_random.dis'       
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
    file_name = 'Results/' + str(data_id) + '/' + str(data_id)+'_full_tracks_' + str(num_neigh) + '_neighbors_original_ward.tree'            
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
'''
#def compute_silhouette(data, distance,prototype_policies, num_prototypes, iterations, verbose=False, size_limit=1000):
def compute_silhouette_block(data_ids, prototype_policie, num_prototype, num_cluster, metric_measure, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil = np.zeros((len(data_ids), len(num_neighbors),iterations))
    time = np.zeros((len(data_ids), len(num_neighbors),iterations))

    for m, data_id in enumerate(data_ids):
        if verbose: print 'Subject/Control: ', data_id
        X = load_data_dis(data_id,num_prototype)          
        
        for j, num_neigh in enumerate(num_neighbors):            
            stdout.flush()
            print "\tGenerating at ", num_neigh, " neighbors"                     
                        
            ward = load_ward_tree(data_id,num_neigh)            
            label = ward.labels_
            
            print"\t Computing silhouette score"
            for k in range(iterations):                    
                st = cpu_time()
                score = silhouette_score_block(X, label, metric = metric_measure,n_jobs=2)
                t = cpu_time() - st
                sil[m,j,k] = score
                time[m,j,k] = t
            print sil, '\t', time            
    return sil, time



def plot_results(rho , num_neighbors, label_policies, color_policies, y_label):
    plt.figure()
    
    for m, prototype_policy in enumerate(label_policies):
        mean = rho[m,:,:].mean(1)
        std = rho[m,:,:].std(1)
        errorbar = std # 3.0 * std / np.sqrt(rho.shape[2])
        
        plt.plot(num_neighbors, mean, color_policies[m], label = label_policies[m], markersize = 8.0)
            
        plt.fill(np.concatenate([num_neighbors, num_neighbors[::-1]]),
                 np.concatenate([mean - errorbar, (mean + errorbar)[::-1]]),
                 alpha=.25,fc='black',  ec='None')
    plt.legend(loc='lower right')
    plt.xlabel("number of neighbors $(k)$")
    plt.ylabel(y_label)
    plt.show()


###############################################################################
if __name__ == '__main__':

    np.random.seed(0)      
    
    data_ids = [101,201,109,210]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [25 ,50, 75, 100]#[10,20,30,40,50,60,70,80,90,100]
    iterations = 1
    prototype_policies = ['random', 'fft', 'sff']
    #fit with num_tracks    
    label_policies =['101', '201','109','210']#,'10K']#,'15K','All']
    color_policies = ['ko--', 'kx:', 'k^-','k*-' ]
    verbose = True    
       
    
    sil, time = compute_silhouette_sample_size(data_ids, prototype_policies[0], num_prototype, 
                                   num_cluster, 'euclidean', 
                                   num_neighbors, iterations, verbose)
    '''                               
    sil, time = compute_silhouette_block(data_ids, prototype_policies[0], num_prototype, 
                                   num_cluster, 'euclidean', 
                                   num_neighbors, iterations, verbose)
    '''
    
    '''    
    filename = 'Results/all_subjects_silhouette_time_sub_sample_50K_iterations_8.txt.txt'       
    time =load_pickle(filename)
    filename = 'Results/all_subjects_silhouette_score_sub_sample_50K_iterations_8.txt.txt'       
    sil =load_pickle(filename)
    '''

    #save_pickle('all_subjects_silhouette_score_sub_sample_50K_iterations_8_140513.txt',sil)
    #save_pickle('all_subjects_silhouette_time_sub_sample_50K_iterations_8_140513.txt',time)
        
    plot_results(sil, num_neighbors,label_policies,color_policies,"Silhouette score for whole tractography")
    plot_results(time, num_neighbors,label_policies,color_policies,"Time for calculating silhouette score with sub_sample 50K")
    