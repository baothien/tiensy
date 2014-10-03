import numpy as np
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from create_dataset import create_dataset_from_tractography, create_dataset_artificial
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import fmin_powell
from time import time


counter = 0
loss_best = 1e10
#P_best = None


def init_prb_state(size1, size2):
    '''
    normal Gausan distribution
    '''
    prb = np.zeros((size1, size2)) 
    prb[:,:] = 1./size2
    return np.array(prb,dtype='float')
    
    
def init_prb_state_1(tract1, tract2):
    '''
    distribution based on the convert of distance
    '''   
    
    dm12 = bundles_distances_mam(tract1, tract2)
        
    from common_functions import normalize_sum_row_1
    prb = normalize_sum_row_1(dm12)
    
    return np.array(prb,dtype='float')  

def init_prb_state_sparse(tract1, tract2, nearest = 10):
    '''
    distribution based on the convert of distance
    '''   
    
    dm12 = bundles_distances_mam(tract1, tract2)
    
    print dm12
    
    cs_idxs = [dm12[i].argsort()[:nearest] for i in np.arange(len(tract1))] #chosen indices
    ncs_idxs = [dm12[i].argsort()[nearest:] for i in np.arange(len(tract1))] #not chosen indices
    
    size1 = len(tract1)
    
    for i in np.arange(size1):
        dm12[i][ncs_idxs[i]] = 0      
    
    '''
    test sparse optimzation
    '''
    
    print dm12
    
    
    from common_functions import normalize_sum_row_1
    prb = normalize_sum_row_1(dm12)
    
    print prb
    
    return np.array(prb,dtype='float'), cs_idxs
    
def init_prb_state_2(size1, size2):
    '''
    random distribution
    '''   
    prb_1 = np.random.rand(size1, size2) 
    
    from common_functions import normalize_sum_row_1
    prb = normalize_sum_row_1(prb_1)
    
    return np.array(prb,dtype='float') 
    

def loss(A, B, P):
    """Loss function of probabilistic tractography mapping.
    """
    return norm(A - dot(P, dot(B, P.T)))


def loss_faster(A, B, P):
    """A slightly faster implementation of loss().
    """
    tmp = A - dot(P, dot(B, P.T))
    return (tmp * tmp).sum()


def loss_normalized(A, B, P):
    Q = np.abs(P) / np.abs(P).sum(1)[:,None]
    return loss_faster(A, B, Q)


def f(x, size1, size2, A, B):
    """Wrapper of the loss function.
    """
    global counter, loss_best, P_best
    P = x.reshape(size1, size2)
    L = loss_normalized(A, B, P)
    if (counter % 1000) == 0: print("%d: %e" % (counter, L))
    counter += 1
    if L < loss_best:
        loss_best = L
        P_best = P
        
    return L


def probability_map_new_ipl(tractography1, tractography2, size1, size2):
    
    print("Defining the initial flat probabilistic mapping, to be optimized.")
    P = np.ones((size1, size2))
    P = P / P.sum(1)[:,None]
    x0 = P.flatten()
    
    print "Computing the distance matrices for each tractography."
    dm1 = bundles_distances_mam(tractography1, tractography1)
    dm2 = bundles_distances_mam(tractography2, tractography2)

    print("Computing similarity matrices.")
    sigma2 = np.mean([np.median(dm1), np.median(dm2)]) ** 2.0
    print("sigma2 = %f" % sigma2)
    A = np.exp(-dm1 * dm1 / sigma2)
    B = np.exp(-dm2 * dm2 / sigma2)

    # Note: the optimization works even using distance instead of similarity:
    # A = dm1
    # B = dm2
    
    print("")
    print("Optimization...")
    print("iteration : Loss")
    t0 = time()
    xopt = fmin_powell(f, x0, args=(size1, size2, A, B), disp=True, full_output=False, maxiter=4, ftol=1.0e-4)
    print("Optimization done in %f secs." % (time() - t0))

    print("")
    # Popt = xopt.reshape(size1, size2)
    # mapping_opt = Popt.argmax(1)
    P_best = xopt.reshape(size1, size2)    
    mapping_best = P_best.argmax(1)
    print("Best Loss = %s" % loss_best)
    print("Best mapping = %s" % mapping_best)  
    
    return mapping_best
    
    
'''    
if __name__ == '__main__':

    np.random.seed(3)

    size1 = 50
    size2 = 60

    same = True # whether the first tractography is a subset of the second one

    A, B = create_dataset_from_tractography(size1, size2, same)
    # A, B = create_dataset_artificial(size1, size2, same)
    
    mapping_best = probability_map(A,B, size1, size2)
#    
#    print("Defining the initial flat probabilistic mapping, to be optimized.")
#    P = np.ones((size1, size2))
#    P = P / P.sum(1)[:,None]
#    x0 = P.flatten()
#
#    print("")
#    print("Optimization...")
#    print("iteration : Loss")
#    t0 = time()
#    xopt = fmin_powell(f, x0, args=(size1, size2, A, B), disp=True, full_output=False, maxiter=4, ftol=1.0e-4)
#    print("Optimization done in %f secs." % (time() - t0))
#
#    print("")
#    # Popt = xopt.reshape(size1, size2)
#    # mapping_opt = Popt.argmax(1)
#    mapping_best = P_best.argmax(1)
#    print("Best Loss = %s" % loss_best)
#    print("Best mapping = %s" % mapping_best)
#    
    
    if same:
        print("accuracy: %f" % (mapping_best == np.arange(size1)).mean())
'''