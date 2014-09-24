import numpy as np
from create_dataset import create_dataset_from_tractography, create_dataset_artificial
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import fmin_powell
from time import time


counter = 0
loss_best = 1e10
P_best = None


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


if __name__ == '__main__':

    np.random.seed(3)

    size1 = 50
    size2 = 60

    same = True # whether the first tractography is a subset of the second one

    A, B = create_dataset_from_tractography(size1, size2, same)
    # A, B = create_dataset_artificial(size1, size2, same)

    print("Defining the initial flat probabilistic mapping, to be optimized.")
    P = np.ones((size1, size2))
    P = P / P.sum(1)[:,None]
    x0 = P.flatten()

    print("")
    print("Optimization...")
    print("iteration : Loss")
    t0 = time()
    xopt = fmin_powell(f, x0, args=(size1, size2, A, B), disp=True, full_output=False, maxiter=4, ftol=1.0e-4)
    print("Optimization done in %f secs." % (time() - t0))

    print("")
    # Popt = xopt.reshape(size1, size2)
    # mapping_opt = Popt.argmax(1)
    mapping_best = P_best.argmax(1)
    print("Best Loss = %s" % loss_best)
    print("Best mapping = %s" % mapping_best)

    if same:
        print("accuracy: %f" % (mapping_best == np.arange(size1)).mean())
