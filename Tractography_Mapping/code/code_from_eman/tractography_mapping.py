import numpy as np
from dipy.io.dpy import Dpy
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from simulated_annealing import anneal, transition_probability, temperature_boltzmann, temperature_cauchy
from prototypes import subset_furthest_first as sff
from prototypes import furthest_first_traversal as fft

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


if __name__ == '__main__':

    do_random_search = True
    do_simulated_annealing = True
    do_simulated_annealing_from_1nn = False
    show = False

    iterations_random = 10000
    iterations_anneal = 1000

    filename = 'data/tracks_dti_10K_linear.dpy'
    print "Loading", filename
    dpr = Dpy(filename, 'r')
    tractography = dpr.read_tracks()
    dpr.close()
    print len(tractography), "streamlines"
    print "Removing streamlines that are too short"
    tractography = filter(lambda x: len(x) > 20, tractography) # remove too short streamlines
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
    ids2 = np.random.permutation(len(tractography))[:size2]
    # ids2 = sff(tractography, k=size2, distance=bundles_distances_mam)
    # ids2 = fft(tractography, k=size2, distance=bundles_distances_mam)
    tractography2 = tractography[ids2]
    # solution = permutation_mapping(size1, size2)
    # tractography2 = tractography1[solution]
    # inverse_solution = np.argsort(solution)
    print "Done."

    print "Computing the distance matrices for each tractography."
    dm1 = bundles_distances_mam(tractography1, tractography1)
    dm2 = bundles_distances_mam(tractography2, tractography2)


    def loss_function(mapping12):
        """Computes the loss function of a given mapping.

        This is the 'energy_function' of simulated annealing.
        """
        global dm1, dm2
        loss = np.linalg.norm(dm1[np.triu_indices(size1)] - dm2[mapping12[:,None], mapping12][np.triu_indices(size1)])
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


    if do_simulated_annealing:
        print
        print "Simulated Annealing"
        np.random.seed(1) # this is the random seed of the optimization process
        # initial_state = random_mapping(size1, size2)
        initial_state = mapping12_best_rs.copy()
        mapping12_best, energy_best = anneal(initial_state=initial_state, energy_function=loss_function, neighbour=neighbour4, transition_probability=transition_probability, temperature=temperature_boltzmann, max_steps=iterations_anneal, energy_max=0.0, T0=200.0, log_every=1000)

    print
    print "The best coregistration+1NN gives a mapping12 with the following loss:"
    dm12 = bundles_distances_mam(tractography1, tractography2)
    mapping12_coregistration_1nn = np.argmin(dm12, axis=1)
    loss_coregistration_1nn = loss_function(mapping12_coregistration_1nn)
    print "loss =", loss_coregistration_1nn

    if do_simulated_annealing_from_1nn:
        print
        print "Simulated Annealing"
        np.random.seed(1) # this is the random seed of the optimization process
        initial_state = mapping12_coregistration_1nn
        mapping12_best_from_1nn, energy_best_from_1nn = anneal(initial_state=initial_state, energy_function=loss_function, neighbour=neighbour4, transition_probability=transition_probability, temperature=temperature_boltzmann, max_steps=iterations_anneal, energy_max=0.0, T0=200.0, log_every=1000)

    if show:
        from dipy.viz import fvtk
        sid = 22
        r = fvtk.ren()
        # fvtk.add(r, fvtk.streamtube(tractography1, fvtk.colors.orange))
        # fvtk.add(r, fvtk.streamtube(tractography2, fvtk.colors.blue))
        fvtk.add(r, fvtk.streamtube([tractography1[sid]], fvtk.colors.red))
        fvtk.add(r, fvtk.streamtube([tractography2[mapping12_best[sid]]], fvtk.colors.cyan))
        fvtk.add(r, fvtk.streamtube([tractography2[mapping12_coregistration_1nn[sid]]], fvtk.colors.green))
        fvtk.show(r)

