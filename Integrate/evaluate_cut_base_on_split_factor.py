# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:17:48 2013

@author: bao
"""

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import numpy as np

#from sklearn.cluster import Ward
from hierarchical import Ward

import matplotlib.pyplot as plt
from dipy.io.pickles import save_pickle,load_pickle
import time as time

#tree_name = 'Results/101/101_15000_tracks_third_trial.tree'
#tree_name = 'Results/101/101_full_tracks_fifth_trial.tree'

#tree_name = 'Results/201/201_15000_tracks_first_trial.tree'
#tree_name = 'Results/201/201_full_tracks_first_trial.tree'

#tree_name = 'Results/109/109_15000_tracks_first_trial.tree'
#tree_name = 'Results/109/109_full_tracks_first_trial.tree'

#tree_name = 'Results/205/205_15000_tracks_first_trial.tree'
#tree_name = 'Results/205/205_full_tracks_first_trial.tree'

#tree_name = 'Results/210/210_15000_tracks_50_neighbors_modified_ward_full_tree_first_trial.tree'
#tree_name = 'Results/210/210_full_tracks_50_neighbors_modified_ward_full_tree_first_trial.tree'

#tree_name = 'Results/101/101_full_tracks_50_neighbors_modified_ward_full_tree_fifth_trial.tree'
#tree_name = 'test.tree'
def random_elements(list_element, k):
    '''
    draw a uniformly random k elements from the list_element
    input: 
           list_element: a list of elements
           k: number of elements need to choose
    output:
           list of k elements in list_element
           if k>len(list_element) then whole list_element is returned
    '''
    n = len(list_element)
    if k >= n:
        return list_element
    result =[]
    idx = np.random.permutation(n)[:k]
    result = [list_element[i] for i in idx]
    return result

def split_factor_node(tree, node_idx, h):
    '''
    calculate the split factor of a node to the heigh h
    (number of children nodes of node_idx, which childern_nodes have the height is heigh h)
    input: 
        tree: a ward_modified tree
        node_idx: index of a node
        h: the heigh to calculate the split of the node_idx
    output:
        int (long) the split factor of node_idx to the heigh h
    '''
    # from the height tree.height_[node_idx] the node would be split, ortherwhile it still one node
    if tree.height_[node_idx]<h:
        return 1
    children = tree.children_list(node_idx,h)
    result = len(children)
    return result
    #return len(tree.children_list(node_idx, height))
    
def split_factor_setnodes(tree, node_idxes, h):
    '''
    calculate the split factor of a list nodes to the heigh h
    sum of all split_factor of each node in list nodes
    input: 
        tree: a ward_modified tree
        node_idxes: array of node index
        heigh: the heigh to calculate the split of the node_idxes
    output:
        int (long) the split factor of array node_idxes to the heigh h
    '''
    s = 0
    for node in node_idxes:
        s1 = split_factor_node(tree, node, h)
        s = s + s1
    return s

def plot_result(split, num_cuts, color_policy, label, title=None):
    plt.figure()    
    
    mean = split[:,:].mean(1)
    std = split[:,:].std(1)
    errorbar = std #3.0 * std / np.sqrt(split.shape[1]) #std
        
    plt.plot(num_cuts, mean, color_policy, label=label, markersize=8.0)
    plt.fill(np.concatenate([num_cuts, num_cuts[::-1]]),
                 np.concatenate([mean - errorbar, (mean + errorbar)[::-1]]),
                 alpha=.25,fc='black',  ec='None')
    plt.legend(loc='lower right')
    plt.xlabel("cut scales ")
    plt.ylabel("split factor ")
    plt.title = title
    plt.show()
    
def plot_results(plt, split, num_cuts, color_policy, label):
     
    mean = split[:,:].mean(1)
    std = split[:,:].std(1)
    errorbar = std #3.0 * std / np.sqrt(split.shape[1]) #std
        
    plt.plot(num_cuts, mean, color_policy, label=label, markersize=8.0)
    plt.fill(np.concatenate([num_cuts, num_cuts[::-1]]),
                 np.concatenate([mean - errorbar, (mean + errorbar)[::-1]]),
                 alpha=.25,fc='black',  ec='None')

def remove_valley(cuts):
    '''
    if s(i-1) + 1 = s(i) and s(i)+1 = s(i+1) : cut at i is a valley between two peaks
       and (R_(i-1)>R (i) && R(i)<R(i+1))
       then remove i
    '''
    n = len(cuts)
    i = 1
    while (i<n-1):
        if ((cuts[i-1][0] + 1 == cuts[i][0]) and (cuts[i][0] +1 == cuts[i+1][0])):
            if (((cuts[i-1][1]>cuts[i][1]) and (cuts[i+1][1]>cuts[i][1])) 
            or ((cuts[i-1][1]<cuts[i][1]) and (cuts[i+1][1]<cuts[i][1]))):
                cuts.remove(cuts[i])
                n = n - 1
                i = i - 1
        i = i + 1
    return cuts        

def heuristic_modified_cuts(scales, dis_thres,star_scale=0):
    '''
    modify the cutting scales with heuristic rule:
        the distance between scale s(i) and s(i+1) should not be larger than the distance 
        if (s(i+1) - s(i) > distance_threshold):
            k = (s(i+1) - s(i))/distance_threshold
            insert k cut scale between s(i) and s(i+1)
    input: scales: integer 1D array of cutting scales
           dis_thres: the maximum distance between two cutting scale
           start_scale: where the started scale
    output: integer 1D array of modified scales
    '''
    results = [star_scale] # the first cut is defined as the star cut scale
    res_i = 0   #position of the processed scale in results
    scl_i = 0   #position of the processing scale in scales
    np.round(5/2)
    while (scl_i<len(scales)):
        d = scales[scl_i] - results[res_i]
        if d > dis_thres:
            #k = d // dis_thres  - 1 
            k = np.round(d / dis_thres)  - 1 
            if ((k+1)*dis_thres) < d:
                k = k + 1
            #l = d / (k+1) #the real distance between 2 adding scales
            #print d, ' ', k 
            
            for j in np.arange(k):                
                results.append(np.round(results[res_i-j] + d*(j+1)/(k+1)))
                res_i = res_i + 1
        results.append(scales[scl_i])
        res_i = res_i + 1
        scl_i = scl_i + 1
    #remove the firt cut sclae added before
    results.remove(star_scale)
    return results     

def run_one_subject(subject_id):
    #tree_name = 'Results/' + str(subject_id) + '/' + str(subject_id) +'_full_tracks_50_neighbors_modified_ward_full_tree.tree'
    tree_name = 'Results/' + str(subject_id) + '/' + str(subject_id) +'_full_tracks_50_neighbors_modified_ward_full_tree_130516.tree'
    #'Result/210/210_full_tracks_50_neighbors_modified_ward_full_tree.tree' 
    tree = load_pickle(tree_name)
    k1 = 15    
    iterations = 50
    cut = tree.best_cut()    
    #print 'origin cut', cut
    
    remove_valley(cut)
    #print 'after remove valley', cut
    
    cut_scales_ori = [s[0] for s in cut] 
    temp_scales = heuristic_modified_cuts(cut_scales_ori[:4],3)
    temp_scales_1 = heuristic_modified_cuts(cut_scales_ori[4:],4,temp_scales[len(temp_scales)-1])
    cut_scales = np.concatenate((temp_scales,temp_scales_1))
    
    split = np.zeros((len(cut_scales),iterations)) 
    
    for j in np.arange(len(cut_scales)):
        
        #run from the top cut to the bottom cut    
        i = len(cut_scales) - j - 1
    
        height = cut_scales[i]   
        t0 = time.time()
        guillotines = tree.cut(height)
        t_cut = time.time() - t0
        print 'Time for cutting at height ',height, ' :  ',t_cut
        
        #the heigh of the next cut
        if i==0:
            h = 0.
        else:
            h = cut_scales[i-1]
        print 'Compute split factor of ', height,' to  scale', h
        for k in np.arange(iterations):            
            random_nodes = random_elements(guillotines, k1)
            print '\t  trial ', k, ':   ', random_nodes
            split[i,k] = split_factor_setnodes(tree,random_nodes,h)
            
    title = '\nSplit factor for cutting time of ' + tree_name 
    plot_result(split, cut_scales, '-kx', '210', title)    
    
def run_on_multi_subjects():
    k1 = 15    
    iterations = 25    
    plt.figure()   
    
    subjects =['101','109','201','205','210']
    colors = ['ko--', 'kx:', 'k^-','k*-','v-.' ]
    
    for m, sub_id in enumerate(subjects):
        
        #tree_name = 'Results/' + str(sub_id) + '/' + str(sub_id) +'_full_tracks_50_neighbors_modified_ward_full_tree.tree'
        tree_name = 'Results/' + str(sub_id) + '/' + str(sub_id) +'_full_tracks_50_neighbors_modified_ward_full_tree_130516.tree'
        tree = load_pickle(tree_name)
        
        cut = tree.best_cut()
        print 'origin cut', cut
        remove_valley(cut)
        print 'after remove valley', cut
        
        cut_scales_ori = [s[0] for s in cut] 
        temp_scales = heuristic_modified_cuts(cut_scales_ori[:4],3)
        temp_scales_1 = heuristic_modified_cuts(cut_scales_ori[4:],4,temp_scales[len(temp_scales)-1])
        cut_scales = np.concatenate((temp_scales,temp_scales_1))
        #print cut_scales_ori
        #print cut_scales
        
        #cut_scales = cut_scales_ori
        #cut_scales = heuristic_modified_cuts(cut_scales,4)
        
        split = np.zeros((len(cut_scales),iterations)) 
        
        for j in np.arange(len(cut_scales)):            
            #run from the top cut to the bottom cut    
            i = len(cut_scales) - j - 1
        
            height = cut_scales[i]               
            guillotines = tree.cut(height)            
                     
            #the heigh of the next cut
            if i==0:
                h = 0.
            else:
                h = cut_scales[i-1]
            
            for k in np.arange(iterations):                                
                random_nodes = random_elements(guillotines, k1)                                
                split[i,k] = split_factor_setnodes(tree,random_nodes,h)       
                
        plot_results(plt, split, cut_scales, colors[m], sub_id) 
        print sub_id, ' : ', cut_scales
    
    plt.legend(loc='upper right')
    plt.xlabel("cut scales ")
    plt.ylabel("split factor ")
    plt.title = '\n Evaluating the cut based on split factor'
    plt.show()
    
run_on_multi_subjects()
#run_one_subject()
#s = [5,7,10,16]
#print 's = ', s
#t = heuristic_modified_cuts(s,3)
#print 't = ', t