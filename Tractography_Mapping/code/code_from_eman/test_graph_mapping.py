import numpy as np
from dipy.io.dpy import Dpy
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from graph_mapping import *
# from scipy.sparse import lil_matrix
from tractography_mapping import permutation_mapping
from prototypes import subset_furthest_first as sff
from prototypes import furthest_first_traversal as fft

if __name__ == '__main__':

    np.random.seed(0)

    size1 = 3
    size2 = 5

    # Some checks to avoid memory issues with 4D probabilistic mapping:
    assert(size1 <= 100)
    assert(size2 <= 100)

    # Some checks to avoid unnecessary too slow computations:
    slow = False
    fast = True
    if size1 <= 50 and size2 <= 50:
        slow = True
        
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

    print "Creating two simulated tractographies of sizes", size1, "and", size2
    # np.random.seed(0) # this is the random seed to create tractography1 and tractography2
    ids1 = np.random.permutation(len(tractography))[:size1]
    # ids1 = sff(tractography, k=size1, distance=bundles_distances_mam)
    ids1 = fft(tractography, k=size1, distance=bundles_distances_mam)
    ids = fft(tractography, k=max([size1, size2]), distance=bundles_distances_mam)
    tractography1 = tractography[ids[:size1]]
    print "Done."
    # ids2 = np.random.permutation(len(tractography))[:size2]
    # ids2 = sff(tractography, k=size2, distance=bundles_distances_mam)
    # ids2 = fft(tractography, k=size2, distance=bundles_distances_mam)
    # tractography2 = tractography[ids2]
    # solution = permutation_mapping(size1, size2)
    # tractography2 = tractography1[solution]
    # inverse_solution = np.argsort(solution)
    tractography2 = tractography[ids[:size2]]
    print "Done."

    print "Computing the distance matrices for each tractography."
    dm1 = bundles_distances_mam(tractography1, tractography1)
    dm2 = bundles_distances_mam(tractography2, tractography2)

    print("Computing similarity matrices.")
    sigma2 = np.mean([np.median(dm1), np.median(dm2)]) ** 2.0
    print("sigma2 = %f" % sigma2)
    sm1 = np.exp(-dm1 * dm1 / sigma2)
    sm2 = np.exp(-dm2 * dm2 / sigma2)

    print("")
    print("Generating random (deterministic) mapping.")
    mapping12 = np.random.randint(size2, size=size1)
    if slow: print("Loss (slow): %s" % loss(sm1, sm2, mapping12))
    if fast: print("Loss (fast): %s" % loss_fast(sm1, sm2, mapping12))
    assert(loss(sm1, sm2, mapping12) == loss_fast(sm1, sm2, mapping12))

    print("")
    print("Generating the equivalent probabilistic mapping.")
    probabilistic_mapping12 = np.zeros((size1, size2), dtype=np.int)
    probabilistic_mapping12[np.arange(size1, dtype=np.int), mapping12] = 1
    if slow: print("Probabilistic Loss (slow): %s" % probabilistic_loss(sm1, sm2, probabilistic_mapping12, verbose=False))
    if fast: print("Probabilistic Loss (fast): %s" % probabilistic_loss_fast(sm1, sm2, probabilistic_mapping12))

    print("")
    print("Generating a random probabilistic mapping.")
    probabilistic_mapping12 = np.random.rand(size1, size2)
    # probabilistic_mapping12 = np.ones((size1, size2)) # equal probability
    probabilistic_mapping12 = probabilistic_mapping12 / probabilistic_mapping12.sum(1)[:,None]
    # probabilistic_mapping12 = probabilistic_mapping12 / probabilistic_mapping12.sum()
    if slow: print("Probabilistic Loss (slow): %s" % probabilistic_loss(sm1, sm2, probabilistic_mapping12, verbose=False))
    if fast: print("Probabilistic Loss (fast): %s" % probabilistic_loss_fast(sm1, sm2, probabilistic_mapping12))

    # print("")
    # print("Generating a sort of equivalent 4D probabilistic mapping (p(i,j,k,l)=p(i,k)p(j,l))")
    # probabilistic_mapping = np.multiply.outer(probabilistic_mapping12, probabilistic_mapping12).transpose(0,2,1,3)
    # probabilistic_mapping /= probabilistic_mapping.sum()
    # if slow: print("Probabilistic Loss 4D (slow): %s" % probabilistic_loss4D(sm1, sm2, probabilistic_mapping, verbose=False))
    # if fast: print("Probabilistic Loss 4D (fast): %s" % probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping))

    print("")
    print("Generating at random a new 4D probabilistic mapping")
    # probabilistic_mapping = np.random.rand(size1, size1, size2, size2)
    probabilistic_mapping = np.ones((size1, size1, size2, size2)) # equal probability mapping
    for i in range(size1):
        probabilistic_mapping[i, i, :, :] = np.eye(size2)

    probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]
    if slow: print("Probabilistic Loss 4D (slow): %s" % probabilistic_loss4D(sm1, sm2, probabilistic_mapping, verbose=False))
    if fast: print("Probabilistic Loss 4D (fast): %s" % probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping))

    print("")
    print("Gradient Descent.")
    alpha = 0.002
    L = np.zeros(20)
    for i in range(len(L)):
        print(i)
        probabilistic_mapping = gradient_descent(probabilistic_mapping, sm1, sm2, alpha)
        # if slow: print("Probabilistic Loss 4D (slow): %s" % probabilistic_loss4D(sm1, sm2, probabilistic_mapping, verbose=False))
        L[i] = probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping)
        if fast: print("Probabilistic Loss 4D (fast): %s" % L[i])


    print("")
    if (size1 == size2) and (sm1 == sm2).all(): # In case the two tractography are the same one, i.e. loss=0.0:

        print("Testing the ideal solution to the 4D mapping")
        probabilistic_mapping_ideal = np.zeros((size1, size1, size2, size2))
        for i in range(size1):
            for j in range(size2):
                probabilistic_mapping_ideal[i, j, i, j] = 1.0

        if slow: print("Probabilistic Loss 4D (slow): %s" % probabilistic_loss4D(sm1, sm2, probabilistic_mapping_ideal, verbose=False))
        if fast: print("Probabilistic Loss 4D (fast): %s" % probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping_ideal))
        assert(probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping_ideal) == 0.0)
        print("Testing whether the gradient of the ideal solution is zero everywhere.")
        for i in range(size1):
            for j in range(size1):
                for k in range(size2):
                    for l in range(size2):
                        assert(gradient_probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping_ideal, i, j, k, l) == 0.0)

        assert((gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping_ideal) == 0.0).all())
        print("OK.")

        print("")
        print("Gradient Descent from a close point to the ideal solution.")
        alpha = 0.002
        probabilistic_mapping = np.random.rand(size1, size1, size2, size2) * 0.01 + probabilistic_mapping_ideal
        for i in range(size1):
            probabilistic_mapping[i, i, :, :] = 0.0

        probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]
        L = np.zeros(20)
        for i in range(len(L)):
            print(i)
            probabilistic_mapping = gradient_descent(probabilistic_mapping, sm1, sm2, alpha)
            L[i] = probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping)
            if fast: print("Probabilistic Loss 4D (fast): %s" % L[i])


    print("")
    print("Testing the vectorized (fast) computation of the gradient of the 4D loss against the non-vectorized (slow) one.")
    gradient = np.zeros((size1, size1, size2, size2))
    for i in range(size1):
        for j in range(size1):
            for k in range(size2):
                for l in range(size2):
                    gradient[i, j, k, l] = gradient_probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping, i, j, k, l)

    gradient_fast = gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping)

    # Minimal differences may occur because of numerical issues.
    np.testing.assert_almost_equal(gradient, gradient_fast, decimal=12)
    print("OK.")


    print("")
    print("Fast (vectorized) Gradient Descent.")
    alpha = 0.003
    L = np.zeros(500)
    # probabilistic_mapping = np.random.rand(size1, size1, size2, size2) * 0.01 + probabilistic_mapping_ideal
    # for i in range(size1):
    #     probabilistic_mapping[i, i, :, :] = 0.0

    # probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]    
    probabilistic_mapping = np.ones((size1, size1, size2, size2)) # equal probability mapping
    for i in range(size1):
        probabilistic_mapping[i, i, :, :] = np.eye(size2)

    probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]
    # probabilistic_mapping = np.ones((size1, size1, size2, size2)) # equal probability mapping
    # for i in range(size1):
    #     probabilistic_mapping[i, i, :, :] = np.eye(size2)

    # probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]
    for i in range(len(L)):
        print(i)
        probabilistic_mapping = gradient_descent_fast(probabilistic_mapping, sm1, sm2, alpha)
        # if slow: print("Probabilistic Loss 4D (slow): %s" % probabilistic_loss4D(sm1, sm2, probabilistic_mapping, verbose=False))
        L[i] = probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping)
        if fast: print("Probabilistic Loss 4D (fast): %s" % L[i])


    print("")
    print("Optimization using scipy's fmin_*.")
    from scipy.optimize import fmin_slsqp, fmin, fmin_cg, fmin_l_bfgs_b

    def f(x):
        probabilistic_mapping = x.reshape(size1, size1, size2, size2)
        # for i in range(size1):
        #     probabilistic_mapping[i, i, :, :] = np.eye(size2)

        probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]    
        return probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping)

    def grad_f(x):
        probabilistic_mapping = x.reshape(size1, size1, size2, size2)
        # for i in range(size1):
        #     probabilistic_mapping[i, i, :, :] = np.eye(size2)

        probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]    
        return gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping).flatten()

    # def f_eqcons(x):
    #     probabilistic_mapping = x.reshape(size1, size1, size2, size2)
    #     return probabilistic_mapping.sum(3).sum(2) - 1.0

    x0 = probabilistic_mapping.flatten()
    # x = fmin_slsqp(func=f, x0=x0, eqcons=[], f_eqcons=None, ieqcons=[], f_ieqcons=None, bounds=[], fprime=grad_f, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=1.4901161193847656e-08)

    # x = fmin(f, x0, disp=True)

    probabilistic_mapping = np.ones((size1, size1, size2, size2)) # equal probability mapping
    for i in range(size1):
        probabilistic_mapping[i, i, :, :] = np.eye(size2)

    probabilistic_mapping /= probabilistic_mapping.sum(3).sum(2)[:,:,None,None]
    x0 = probabilistic_mapping.flatten()
    # x = fmin_cg(f, x0, grad_f)
    x, loss_best, d = fmin_l_bfgs_b(f, x0, grad_f)
    # x = fmin_slsqp(func=f, x0=x0, fprime=grad_f)

    probabilistic_mapping = x.reshape(size1, size1, size2, size2)


    print("")
    print("Gradient Descent for the sqaured graph mapping loss.")
    alpha = 0.001
    L = np.zeros(200)
    P = np.ones((size1, size2))
    P /= P.sum(1)[:,None]
    for i in range(len(L)):
        print(i)
        P = gradient_descent_loss_graph_mapping(P, sm1, sm2, alpha)
        L[i] = loss_graph_mapping_squared(sm1, sm2, P)
        if fast: print("Normalized Squared Probabilistic Loss 2D (fast): %s" % L[i])


    print("")
    print("Gradient Descent for the sqaured normalized graph mapping loss.")
    alpha = 0.00000005
    L = np.zeros(100)
    P = np.random.rand(size1, size2)
    for i in range(len(L)):
        print(i)
        P = gradient_descent_loss_graph_mapping_normalized_squared(P, sm1, sm2, alpha)
        L[i] = loss_graph_mapping_normalized_squared(sm1, sm2, P)
        if fast: print("Normalized Squared Probabilistic Loss 2D (fast): %s" % L[i])


    from scipy.optimize import fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b, fmin_tnc, fmin_cobyla, fmin_slsqp
    print("")
    loss_best = 1e6
    def f(x):
        global loss_best
        P = x.reshape(size1, size2)
        loss = loss_graph_mapping_normalized_squared(sm1, sm2, P)
        if loss < loss_best:
            loss_best = loss
            print loss_best
            
        return loss

    def grad_f(x):
        P = x.reshape(size1, size2)
        gradient = gradient_loss_graph_mapping_normalized_squared_slow(sm1, sm2, P)
        return gradient.flatten()

    # x0 = np.random.rand(size1 * size2)
    # x0 = np.ones(size1 * size2)
    x0 = (np.eye(size1, size2) + np.random.rand(size1, size2) * 0.01).flatten()
    # xopt, fopt, iter, funcalls, warnflag = fmin(f, x0)
    xopt = fmin_powell(f, x0, disp=True, full_output=False, maxiter=10)
    # xopt = fmin_cg(f, x0, disp=True, full_output=False)
    # xopt = fmin_bfgs(f, x0, disp=True, full_output=False)
    # xopt = fmin_ncg(f, x0, fprime=grad_f, disp=True, full_output=False)
    # xopt, fopt, d = fmin_l_bfgs_b(f, x0, fprime=grad_f, disp=True)
    # xopt = fmin_tnc(f, x0, fprime=grad_f, disp=True)
    # xopt = fmin_cobyla(f, x0, [], disp=True)
    # xopt = fmin_slsqp(func=f, x0=x0, fprime=grad_f)

    
    P = xopt.reshape(size1, size2)
    P = np.abs(P / P.sum(1)[:,None])
