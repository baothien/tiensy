# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:58:15 2013

@author: bao
"""

import numpy as np
import nibabel as nib

import time as time
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.cluster import Ward
#from hierarchical import Ward

#for computing the CPU time, not elapsed time
import resource
def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]
    
def compute_silhouette(data, prototype_policie, num_prototype, num_cluster, metric_measure, sub_data_sizes, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil = np.zeros((len(sub_data_sizes), len(num_neighbors),iterations))
    time = np.zeros((len(sub_data_sizes), len(num_neighbors),iterations))

    for m, sub_data_size in enumerate(sub_data_sizes):
        if verbose: print 'Sub data size: ', sub_data_size
        if sub_data_size!=0:
            sub_data = data[:sub_data_size]
        else:
            sub_data = data      

        X = compute_disimilarity(sub_data, bundles_distances_mam, prototype_policie, num_prototype,len(sub_data))
        
        for j, num_neigh in enumerate(num_neighbors):
            print "number of neighbors:", num_neigh, " - ", 
            for k in range(iterations):
                print '\n',k,
                #stdout.flush()
                if verbose: print("Generating at %s neighbor" % num_neigh),                
                
                connectivity = kneighbors_graph(X, n_neighbors=num_neigh)                          
                
                st = cpu_time()#time.clock()
                ward = Ward(n_clusters=num_cluster, connectivity=connectivity).fit(X)#compute_full_tree=False, 
                #t = time.clock() - st
                t = cpu_time() - st
                
                label = ward.labels_
                score = metrics.silhouette_score(X, label, metric = metric_measure)
            
                sil[m,j,k] = score
                time[m,j,k] = t
                #print score, '\t', time
            print
    return sil, time

def compute_silhouette_simulated_data(num_cluster, metric_measure, sub_data_sizes, num_neighbors, iterations, verbose=False):
    print "Computing silhouette score :"
    sil = np.zeros((len(sub_data_sizes), len(num_neighbors),iterations))
    time = np.zeros((len(sub_data_sizes), len(num_neighbors),iterations))

    for m, sub_data_size in enumerate(sub_data_sizes):
        if verbose: print 'Sub data size: ', sub_data_size
	###############################################################################
	# Generate data (swiss roll dataset)
    	noise = 0.05
    	X, _ = make_swiss_roll(sub_data_size, noise)
    	# Make it thinner
    	X[:, 1] *= .5        
      
        for j, num_neigh in enumerate(num_neighbors):
            print "number of neighbors:", num_neigh, " - ", 
            for k in range(iterations):
                print '\n',k,
                #stdout.flush()
                if verbose: print("Generating at %s neighbor" % num_neigh),                
                
                connectivity = kneighbors_graph(X, n_neighbors=num_neigh)                          
                
                st = cpu_time()#time.clock()
                ward = Ward(n_clusters=num_cluster, connectivity=connectivity).fit(X)#, compute_full_tree=False).fit(X)
                #t = time.clock() - st
                t = cpu_time() - st
                
                label = ward.labels_
                score = metrics.silhouette_score(X, label, metric = metric_measure)
            
                sil[m,j,k] = score
                time[m,j,k] = t
                #print score, '\t', time
            print
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

    num_points = [500, 1000, 5000]#,10000]#,15000]#,50000]#[50,100,150,0]##,15000]#[0,15000,10000,5000,1000,500]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [25 ,50, 75, 100]#[10,20,30,40,50,60,70,80,90,100]
    iterations = 1
    prototype_policies = ['random', 'fft', 'sff']
    #fit with num_tracks    
    label_policies = ['500','1K','5K']#,'10K']#,'15K','All']
    color_policies = ['ko--', 'kx:', 'k^-']#,'k*-' ]
    verbose = True
    
    sil, time = compute_silhouette_simulated_data(num_cluster, 'euclidean', 
                                   num_points, num_neighbors, iterations, verbose)
    plot_results(sil, num_neighbors,label_policies,color_policies,"Silhouette score")
    plot_results(time, num_neighbors,label_policies,color_policies,"Time for hierarchical tree")
    
    
    
    
    
    
    
        
