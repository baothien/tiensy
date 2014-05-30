# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:33:27 2014

@author: bao

==========================================
Evaluation the mapping with local assessment

==========================================

Mapping CST-SFF-in-Extension to CST-Extension

"""
print(__doc__)


import numpy as np
from dipy.io.dpy import Dpy
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.tracking.metrics import length
from simulated_annealing import anneal, transition_probability, temperature_boltzmann, temperature_cauchy
#from prototypes import furthest_first_traversal as fft
from common_functions import *
import argparse
  
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
   
def visualize_source_mappedsource(ren, tract1, tract2, mapping, color1=fvtk.red, color2=fvtk.blue): 
    for i in np.arange(len(tract1)):
        fvtk.add(ren, fvtk.line(tract1[i], color1, opacity=1.0))
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], color2, opacity=1.0))     
    return ren
      
def visualize_diff_color(ren, tract1, tract2, mapping):
    
    #c = fvtk.line(lines, fvtk.green)    
    #fvtk.add(r,c)
    colors = [fvtk.red,fvtk.green, fvtk.blue, fvtk.white,fvtk.yellow, fvtk.gray,fvtk.hot_pink]#fvtk.cyan,fvtk.dark_blue,fvtk.dark_green,fvtk.dark_red,fvtk.golden,
    for i in np.arange(len(tract1)):
        fvtk.add(ren, fvtk.line(tract1[i], colors[i % len(colors)], opacity=1.0))
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], colors[i % len(colors)], opacity=1.0))
     
    return ren
 
   
def informed_random_mapping(size1, size2, dm1, dm2):
    """Choose one source streamline and one destination streamline
    and then map all other streamlines according to their relative
    distances to the source and destination streamlines.
    """
    #global dm1, dm2
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
    
def tracts_mapping(tractography1, tractography2, loss_function, neighbour, iterations_anneal):
        
    print
    print "The best coregistration+1NN gives a mapping12 with the following loss:"
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)
    loss_coregistration_1nn = loss_function(mapping12_coregistration_1nn)
    print "loss =", loss_coregistration_1nn

    #iterations_anneal = 100
    print "Simulated Annealing"
    np.random.seed(1) 
    initial_state =  mapping12_coregistration_1nn.copy()
    mapping12_best, energy_best = anneal(initial_state=initial_state, energy_function=loss_function, neighbour=neighbour, transition_probability=transition_probability, temperature=temperature_boltzmann, max_steps=iterations_anneal, energy_max=0.0, T0=200.0, log_every=1000)

    return mapping12_coregistration_1nn, loss_coregistration_1nn, mapping12_best, energy_best


def tracts_mapping1(tractography1, tractography2, loss_function, neighbour, iterations_anneal_now,pre_map_file):

    ann = [100, 200, 400, 600, 800, 1000]        
    iterations_anneal_pre = 0
    
    if iterations_anneal_now<=100:
        dm12 = bundles_distances_mam(tractography1, tractography2)
        mapping12_coregistration_1nn = np.argmin(dm12, axis=1)
    else:
        k = (iterations_anneal_now/200) - 1
        iterations_anneal_pre = ann[k]
        from dipy.io.pickles import load_pickle
        mapping12_coregistration_1nn = load_pickle(pre_map_file)
        
    iterations_anneal = iterations_anneal_now - iterations_anneal_pre
    
    print "Iteration: ", iterations_anneal_now, iterations_anneal_pre, iterations_anneal
    print "The previous coregistration gives a mapping12 with the following loss:"        
    loss_coregistration_1nn = loss_function(mapping12_coregistration_1nn)
    print "loss =", loss_coregistration_1nn

    #iterations_anneal = 100
    print "Simulated Annealing"
    np.random.seed(1) 
    initial_state =  mapping12_coregistration_1nn.copy()
    mapping12_best, energy_best = anneal(initial_state=initial_state, energy_function=loss_function, neighbour=neighbour, transition_probability=transition_probability, temperature=temperature_boltzmann, max_steps=iterations_anneal, energy_max=0.0, T0=200.0, log_every=1000)

    return mapping12_coregistration_1nn, loss_coregistration_1nn, mapping12_best, energy_best


#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Evaluation the tractography mapping using local assessment (mapping CST to CST)",
                                 epilog="Written by Bao Thien Nguyen, bao@bwh.harvard.edu.",
                                 version='1.0')

parser.add_argument(
                    'inputSourceTractography',
                    help='The file name of source whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputSourceCSTSFFIndex',
                    help='The file name of source CST plus SFF index')

parser.add_argument(
                    'inputTargetTractography',
                    help='The file name of target whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputTargetCSTIndex',
                    help='The file name of target CST index')
parser.add_argument(
                    'inputTargetCSTExtIndex',
                    help='The file name of target CST extension index')

parser.add_argument(
                    '-pr', action='store', dest='inputNumPrototypes', type=int,
                    help='The number of prototypes') 
                    
parser.add_argument(
                    '-an', action='store', dest='inputAnneal_iterations', type=int,
                    help='The number of interation for annealing')  
                                  
parser.add_argument(
                    'outputMap_best',
                    help='The output file name of the best mapping')                    
parser.add_argument(
                    'outputMap_1nn',
                    help='The output file name of the 1-NN mapping')
'''                    
parser.add_argument(
                    'inputMap_pre',
                    help='The input file name of the previous mapping')                    
'''
args = parser.parse_args()

print "=========================="
#print "Source tractography:       ", args.inputSourceTractography
print "Source CST plus SFF index:       ", args.inputSourceCSTSFFIndex
#print "Target tractography:       ", args.inputTargetTractography
#print "Target CST index:       ", args.inputTargetCSTIndex
print "Target CST extension index:       ", args.inputTargetCSTExtIndex
#print "Number of prototypes:      ", args.inputNumPrototypes 
#print "Anneal iterations:      ", args.inputAnneal_iterations 
#print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTSFFIndex

t_file = args.inputTargetTractography
t_ind = args.inputTargetCSTExtIndex
t_cst = args.inputTargetCSTIndex

num_pro = args.inputNumPrototypes 

map_best_fn = args.outputMap_best
map_1nn_fn = args.outputMap_1nn
#pre_map_fn = args.inputMap_pre

iterations_anneal = args.inputAnneal_iterations 

source_cst = load_tract(s_file,s_ind)

#target_cst, target_cst_ext, medoid_target_cst = load_cst(t_file,t_ind,True)
target_cst = load_tract(t_file,t_ind)

print len(source_cst), len(target_cst)

tractography1 = source_cst
tractography2 = target_cst

#print "Source", len(tractography1)
#print "Target", len(tractography2)


#print "Computing the distance matrices for each tractography."
dm1 = bundles_distances_mam(tractography1, tractography1)
dm2 = bundles_distances_mam(tractography2, tractography2)

size1 = len(tractography1)
size2 = len(tractography2)
#iterations_anneal = 1000

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
    
def loss_function(mapping12):
    """Computes the loss function of a given mapping.

    This is the 'energy_function' of simulated annealing.
    """
    global dm1, dm2
    loss = np.linalg.norm(dm1[np.triu_indices(size1)] - dm2[mapping12[:,None], mapping12][np.triu_indices(size1)])
    return loss
    
#mapping12_coregistration_1nn, loss_coregistration_1nn, mapping12_best, energy_best = tracts_mapping1(tractography1, tractography2, loss_function,neighbour4, iterations_anneal, pre_map_fn)
mapping12_coregistration_1nn, loss_coregistration_1nn, mapping12_best, energy_best = tracts_mapping(tractography1, tractography2, loss_function,neighbour4, iterations_anneal)
print "Best enegery of annealing: ", energy_best

from dipy.io.pickles import save_pickle
save_pickle(map_best_fn,mapping12_best)
save_pickle(map_1nn_fn,mapping12_coregistration_1nn)
print 'Saved ', map_best_fn
print 'Saved ', map_1nn_fn
'''
#visualize source and mapped source - red and blue
ren3 = fvtk.ren()
ren3 = visualize_source_mappedsource(ren3, tractography1[:- num_pro], tractography2, mapping12_best[:-num_pro])
fvtk.show(ren3)

#visualize target cst and mapped source cst - yellow and blue
ren4 = fvtk.ren()
target_cst_only = load_tract(t_file,t_cst)
ren4 = visualize_tract(ren4, target_cst_only, fvtk.yellow)
ren4 = visualize_mapped(ren4, tractography2, mapping12_best[:- num_pro], fvtk.blue)
fvtk.show(ren4)
'''





