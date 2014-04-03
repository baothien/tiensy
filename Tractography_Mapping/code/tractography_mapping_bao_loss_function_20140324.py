# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:06:24 2014

@author: bao
"""

"""
 mapping prototype of tract1 to find the coressponding in tract2
 using the new pyramid matching kernel
 Grauman & Darrel, IJML 2007 - The pyramid match kernel: efficient learning with sets of features
"""

import numpy as np
from dipy.io.dpy import Dpy
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from simulated_annealing import anneal, transition_probability, temperature_boltzmann, temperature_cauchy
from dissimilarity_common_20130925 import *
from dipy.tracking.metrics import length
from sklearn.neighbors import KDTree, BallTree
    
def Pyramid_KN(set1, set2, weight=1., num_bins = 128):
    """Compute the similarity of two set
       similarity = weight * si_patt + (1-weight)* si_scale
    """
    loss_scale = 0.0# * np.max(set1)/np.max(set2)
    
    #normalize two set to 128    
    #max_dis = num_bins    
    #nl_set1 = np.array(set1*max_dis/np.max(set1),dtype=float)
    #nl_set2 = np.array(set2*max_dis/np.max(set2),dtype=float)
    #his_set1 = np.histogram(nl_set1,num_bins,(0,max_dis))[0]
    #his_set2 = np.histogram(nl_set2,num_bins,(0,max_dis))[0]
    
    nl_set1 = set1
    nl_set2 = set2
    
    scale_max = max(np.max(set1),np.max(set2))
    #print set1
    #print set2
        
    bins_temp = num_bins
    si_patt = 0.
    pre_sum_min = 0.    
    while (bins_temp>0):
        his_set1 = np.histogram(nl_set1,bins_temp,(0,scale_max))[0]
        his_set2 = np.histogram(nl_set2,bins_temp,(0,scale_max))[0]    
        sum_min = np.minimum(his_set1,his_set2).sum()
        diff = sum_min - pre_sum_min
        pre_sum_min = sum_min
        si_patt = si_patt + diff * 2**(-1.*num_bins/bins_temp)
        bins_temp = bins_temp/2
        #print 'sum_min: ', sum_min, ' diff: ', diff, '  si_patt', si_patt
        
        
    
    #print set1
    #print set2
    #print nl_set1
    #print nl_set2
    #print his_set1
    #print his_set2    
    loss_patt = min(len(set1),len(set2))/2. - si_patt
    loss = weight * loss_patt + (1.0 - weight)* loss_scale

    return loss
        
  
def random_mapping(size1, size2):
    """Generate a random mapping from from 1 to 2.
    """
    return np.random.randint(0, size1, size=size2)


def permutation_mapping(size1, size2):
    """Generate a random mapping from from 1 to 2 using permutations.
    
    This function handles all cases: size1 < size2, size1=size2, size1>size2.

    The mapping has repetitions (the minimum necessary) only if size2 < size1.
    """
    return np.random.permutation(np.repeat(np.arange(size2),np.ceil(size1/float(size2))))[:size1]


def random_search(loss_function, random_state_function, iterations=10000):
    """Generate multiple random mappings and return the best one.
    """
    print "Computing an initial state by random sampling."
    loss_best = np.finfo(np.double).max
    mapping12_best = None
    print "  Step) \t Best loss"
    for i in range(iterations):
        mapping12 = random_state_function(size1, size2)
        loss = loss_function(mapping12)
        if loss < loss_best:
            loss_best = loss
            mapping12_best = mapping12
            print "%6d) \t %s" % (i, loss_best)

    return mapping12_best, loss_best


def random_choice(values, size=1):
    """pick one value at random according to probabilities
    proportional to values.

    size: currently unsupported.
    """
    values_normalized = values / values.sum()
    values_normalized_with0 = np.zeros(len(values) + 1)
    values_normalized_with0[1:] = values_normalized
    choice = np.where(values_normalized_with0.cumsum() > np.random.rand())[0][0] - 1
    return choice

def visualize(ren, tract1, tract2, mapping):
    
    #c = fvtk.line(lines, fvtk.green)    
    #fvtk.add(r,c)
    colors = [fvtk.red,fvtk.green, fvtk.blue, fvtk.white,fvtk.yellow, fvtk.gray,fvtk.hot_pink]#fvtk.cyan,fvtk.dark_blue,fvtk.dark_green,fvtk.dark_red,fvtk.golden,
    for i in np.arange(len(tract1)):
        fvtk.add(ren, fvtk.line(tract1[i], colors[i % len(colors)], opacity=1.0))
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], colors[i % len(colors)], opacity=1.0))
     
    return ren

if __name__ == '__main__':

    do_random_search = False#True#False#True
    do_simulated_annealing = True

    iterations_random = 10000
    iterations_anneal = 1000

    two_tractography = True    
    
    if two_tractography == False:
        
        filename = 'data/tracks_dti_10K_linear.dpy'
        print "Loading", filename
        dpr = Dpy(filename, 'r')
        tractography = dpr.read_tracks()
        dpr.close()
        print len(tractography), "streamlines"
        print "Removing streamlines that are too short"
        tractography = filter(lambda x: len(x) > 40, tractography) # remove too short streamlines
        print len(tractography), "streamlines"    
        tractography = np.array(tractography, dtype=np.object)
    
        size1 = 100
        size2 = 100
        print "Creating two simulated tractographies of sizes", size1, "and", size2
        np.random.seed(1) # this is the random seed to create tractography1 and tractography2
        ids1 = np.random.permutation(len(tractography))[:size1]
        # ids1 = sff(tractography, k=size1, distance=bundles_distances_mam)
        # ids1 = fft(tractography, k=size1, distance=bundles_distances_mam)
        tractography1 = tractography[ids1]
        print "Done."
        
        ids2 = np.random.permutation(len(tractography1))
        tractography2 = tractography1[ids2]
        
        '''
        adding new streamlines to test mapping
        '''  
        new_tractography = []
        for i in np.arange(len(ids2)):
            new_tractography.append(tractography2[i])
            
        ids3 = np.random.permutation(len(tractography))[:2*size1]
        for i in np.arange(len(ids3)):
            if i not in ids2:
                new_tractography.append(tractography[i])
                
        tractography2 = np.array(new_tractography,dtype = np.object)
        
        size2 = len(tractography2)
        print 'New length ', size2
        
        '''
        ids2 = np.random.permutation(len(tractography))[:size2]
        # ids2 = sff(tractography, k=size2, distance=bundles_distances_mam)
        # ids2 = fft(tractography, k=size2, distance=bundles_distances_mam)
        tractography2 = tractography[ids2]
        # solution = permutation_mapping(size1, size2)
        # tractography2 = tractography1[solution]
        # inverse_solution = np.argsort(solution)
        '''
        print "Done."
        
        whole_tract1 = tractography
        whole_tract2 = tractography
    
    else:
    
        #begin of working with two tractography
        
        filename_1 = 'data/101_tracks_dti_10K_linear.dpy'    
        filename_2 = 'data/104_tracks_dti_10K_linear.dpy'    
        
        #prototype_policies = ['random', 'fft', 'sff']
        #num_prototypes = 200
        
        size1 = 150#100
        size2 = 400#100
    
        print "Loading tracks."
        dpr_1 = Dpy(filename_1, 'r')
        tracks_1_all = dpr_1.read_tracks()
        dpr_1.close()
        
        tracks_1 = filter(lambda x: len(x) > 20, tracks_1_all)
        '''
        tracks_1 = []
        for st in tracks_1_all:
            if (length(st)>25):
                tracks_1.append(st)
        '''
        tracks_1 = np.array(tracks_1, dtype=np.object)
        
        
        dpr_2 = Dpy(filename_2, 'r')
        tracks_2_all = dpr_2.read_tracks()
        dpr_2.close()

        tracks_2 = filter(lambda x: len(x) > 20, tracks_2_all)
        '''        
        tracks_2 = []
        for st in tracks_2_all:
            if (length(st)>25):
                tracks_2.append(st)
        '''
        tracks_2 = np.array(tracks_2, dtype=np.object) 
        
        
        print len(tracks_1), len(tracks_2)   
        
        "mapping from prototypes of one tractography to the whole second tractography"        
        
        print "Creating two simulated tractographies of sizes", size1, "and", size2
        np.random.seed(1) # this is the random seed to create tractography1 and tractography2
        ids1 = np.random.permutation(len(tracks_1))[:size1]
        # ids1 = sff(tractography, k=size1, distance=bundles_distances_mam)
        # ids1 = fft(tractography, k=size1, distance=bundles_distances_mam)
        tractography1 = tracks_1[ids1]        
        
        ids2 = np.random.permutation(len(tracks_2))[:size2]
        # ids1 = sff(tractography, k=size1, distance=bundles_distances_mam)
        # ids1 = fft(tractography, k=size1, distance=bundles_distances_mam)
        tractography2 = tracks_2[ids2]
        print "Done."
        
        whole_tract1 = tracks_1
        whole_tract2 = tracks_2
        #end of working with two tractography

    print "Computing the distance matrices for each tractography."
    dm1 = bundles_distances_mam(tractography1, tractography1)
    dm2 = bundles_distances_mam(tractography2, tractography2)   
    
    dm1_all = bundles_distances_mam(whole_tract1, whole_tract1)
    dm2_all = bundles_distances_mam(whole_tract2, whole_tract2)
    
    dm1_all_small = bundles_distances_mam(tractography1, whole_tract1)
    dm2_all_small = bundles_distances_mam(tractography2, whole_tract2)
    
    kdt1 = BallTree(dm1_all) # KDTree(dp)
    kdt2 = BallTree(dm2_all) # KDTree(dp)

    def loss_function_Eman(mapping12):
        """Computes the loss function of a given mapping.

        This is the 'energy_function' of simulated annealing.
        """
        global dm1, dm2
        loss = np.linalg.norm(dm1[np.triu_indices(size1)] - dm2[mapping12[:,None], mapping12][np.triu_indices(size1)])
        return loss

    def loss_function(mapping12):
        """Computes the loss function of a given mapping.

        Using the graph kernel of two sets of distance.
        """
        
        global tractography1, tractography2
        global dm1_all, dm1_all_small, dm2_all, dm2_all_small
        global kdt1, kdt2
        k = 10
        
        radius = 150  
        loss = 0.0
        for sid in np.arange(len(tractography1)):               
            #idx1 = kdt1.query_radius(dm1_all_small[sid], radius)[0]
            idx1 = kdt1.query(dm1_all_small[sid], k)[1][0]            
            dm_small1 = dm1_all[idx1][:,idx1]
            e1 = dm_small1[np.triu_indices(dm_small1.shape[0],1)]
                    
            #idx2 = kdt2.query_radius(dm2_all_small[mapping12[sid]], radius)[0]
            idx2 = kdt2.query(dm2_all_small[mapping12[sid]], k)[1][0]
            dm_small2 = dm2_all[idx2][:,idx2]
            e2 = dm_small2[np.triu_indices(dm_small2.shape[0],1)]
            
            #loss = loss + Graph_KN(e1, e2, weight=1., num_bins = 128)
            #similarity = similarity + Pyramid_KN(e1, e2, weight=1., num_bins = 128)
            loss = loss +  Pyramid_KN(e1, e2, weight=1., num_bins = 128)
            
        return loss

    def neighbour(mapping12):
        """Computes the next state given the current state by
        re-assigning the mapping of one streamline at random.
        """
        global size1, size2
        source = np.random.randint(0, size1)
        destination = np.random.randint(0, size2)
        mapping12[source] = destination
        return mapping12


    def neighbour2(mapping12):
        """Computes the next state given the current state by
        switching the mapping of two streamlines.
        """
        global size1, size2
        pair = np.random.randint(0, size1, size=2)
        mapping12[pair[0]], mapping12[pair[1]] = mapping12[pair[1]], mapping12[pair[0]]
        return mapping12


    def neighbour3(mapping12):
        """Computes the next state given the current state.
        """
        global size1, size2
        return informed_random_mapping(size1, size2)


    def neighbour4(mapping12):
        """Computes the next state given the current state.

        Change the mapping of one random streamline in a greedy way.
        """
        global size1, size2
        source = np.random.randint(0, size1)
        loss = np.zeros(size2)
        for i, destination in enumerate(range(size2)):
            mapping12[source] = destination
            loss[i] = loss_function(mapping12)

        mapping12[source] = np.argmin(loss) # greedy! (WORKS WELL!)
        # mapping12[source] = random_choice(loss.max() - loss + loss.min()) # stochastic greedy (WORKS BAD!)
        return mapping12
        

    def informed_random_mapping(size1, size2):
        """Choose one source streamline and one destination streamline
        and then map all other streamlines according to their relative
        distances to the source and destination streamlines.
        """
        global dm1, dm2
        mapping12 = -np.ones(size1, dtype=np.int)
        source = np.random.randint(0, size1)
        destination = np.random.randint(0, size2)
        source_sorted_distance_id = np.argsort(dm1[source])
        destination_sorted_distance_id = np.argsort(dm2[destination])
        # Compute a random subset of destination of size source:
        subset_destination = np.sort((np.random.rand(size1) * size2).astype(np.int)) # with repetitions
        # subset_destination = np.sort(permutation_mapping(size1, size2)) # without repetitions (or the minimum possible)
        destination_sorted_distance_id_subset = destination_sorted_distance_id[subset_destination]
        mapping12[source_sorted_distance_id] = destination_sorted_distance_id_subset
        return mapping12
        

    if do_random_search:
        print
        print "Random Search."
        np.random.seed(1) # this is the random seed of the optimization process
        mapping12_best_rs, energy_best_rs = random_search(loss_function=loss_function, random_state_function=informed_random_mapping, iterations=iterations_random)
        print "Best loss:", energy_best_rs

    print
    print "The best coregistration+1NN gives a mapping12 with the following loss:"
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)
    loss_coregistration_1nn = loss_function(mapping12_coregistration_1nn)
    print "loss =", loss_coregistration_1nn

    if do_simulated_annealing:
        print
        print "Simulated Annealing"
        np.random.seed(1) # this is the random seed of the optimization process
        # initial_state = random_mapping(size1, size2)
        #initial_state = mapping12_best_rs.copy()
        initial_state = mapping12_coregistration_1nn.copy()
        mapping12_best, energy_best = anneal(initial_state=initial_state, energy_function=loss_function, neighbour=neighbour4, transition_probability=transition_probability, temperature=temperature_boltzmann, max_steps=iterations_anneal, energy_max=0.0, T0=200.0, log_every=1000)    
   
    ren = fvtk.ren() 
    ren = visualize(ren,tractography1[:6],tractography2,mapping12_best)
    fvtk.show(ren)
    
    ren1 = fvtk.ren() 
    ren1 = visualize(ren1,tractography1[6:12],tractography2,mapping12_best[6:])
    fvtk.show(ren1)
    