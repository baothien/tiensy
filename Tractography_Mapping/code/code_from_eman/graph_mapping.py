import numpy as np


def loss(sm1, sm2, mapping12):
    """Basic definition of the deterministic loss function. Just for
    testing purpose.
    """
    L = 0.0
    for i, mi in enumerate(mapping12):
        for j, mj in enumerate(mapping12):
            if j == i:
                continue
            
            L += (sm1[i,j] - sm2[mi,mj]) ** 2.0

    return np.sqrt(L)


def loss_fast(sm1, sm2, mapping12):
    """Fast implementation of the deterministic loss function, using
    conventions from graph matching.
    """
    return np.linalg.norm(sm1 - sm2[mapping12[:,None], mapping12], ord='fro')


def probabilistic_loss(sm1, sm2, probabilistic_mapping12, verbose=False):
    """Basic definition of the probabilistic loss function. Just for
    testing purpose.
    """
    L = 0.0
    for i in range(sm1.shape[0]):
        # if verbose: print("i %d" % i)
        for j in range(sm1.shape[0]):
            if j == i:
                continue

            tot = 0.0
            for k in range(sm2.shape[0]):
                for l in range(sm2.shape[0]):
                    tot += probabilistic_mapping12[i, k] * probabilistic_mapping12[j, l] * sm2[k, l]

            # if verbose: print("(%d,%d) : %f" % (i, j, tot))
            L += (sm1[i, j] - tot) ** 2.0

    return np.sqrt(L)
    

def probabilistic_loss_fast(sm1, sm2, probabilistic_mapping12):
    """Fast implementation of the probabilistic loss.
    """
    tmp = np.dot(probabilistic_mapping12, np.dot(sm2, probabilistic_mapping12.T))
    # This is necessary because we want to skip diagonal elements both
    # in tmp and sm1 (and they are 1.0 in sm1):
    np.fill_diagonal(tmp, 1.0)
    return np.linalg.norm(sm1 - tmp, ord='fro')


def probabilistic_loss4D(sm1, sm2, probabilistic_mapping, verbose=False):
    """Basic definition of the probabilistic loss function with 4D
    probabilistic_mapping. Just for testing purpose.
    """
    L = 0.0
    for i in range(sm1.shape[0]):
        # if verbose: print("i %d" % i)
        for j in range(sm1.shape[0]):
            # if j == i:
            #     continue

            tot = 0.0
            for k in range(sm2.shape[0]):
                for l in range(sm2.shape[0]):
                    tot += probabilistic_mapping[i, j, k, l] * sm2[k, l]

            # if verbose: print("(%d,%d) : %f" % (i, j, tot))
            L += (sm1[i, j] - tot) ** 2.0

    return np.sqrt(L)


def probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping):
    """Fast implementation of the probabilistic loss with 4D
    probabilistic_mapping.
    """
    tmp = (probabilistic_mapping * sm2).sum(3).sum(2)
    # This is necessary because we want to skip diagonal elements both
    # in tmp and sm1:
    # np.fill_diagonal(tmp, 1.0)
    return np.linalg.norm(sm1 - tmp, ord='fro')


# def gradient_probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping, i, j, k, l):
#     """Partly vectorized computation of the gradient of the 4D loss.
#     """
#     if i!=j:
#         tmp = (probabilistic_mapping[i, j, :, :] * sm2).sum()
#         return -2.0 * (sm1[i, j] - tmp) * sm2[k, l]
#     else:
#         return 0.0


# def gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping):
#     """Fully vectorized computation of the gradient of the 4D loss.
#     """
#     tmp = (probabilistic_mapping * sm2).sum(3).sum(2)
#     gradient = -2.0 * np.multiply.outer((sm1 - tmp), sm2)
#     for i in range(sm1.shape[0]):
#         gradient[i, i, :, :] = 0.0
#     return gradient


def gradient_probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping, i, j, k, l):
    """Partly vectorized computation of the gradient of the 4D loss.
    """
    tmp = (probabilistic_mapping[i, j, :, :] * sm2).sum()
    return -2.0 * (sm1[i, j] - tmp) * sm2[k, l]


def gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping):
    """Fully vectorized computation of the gradient of the 4D loss.
    """
    tmp = (probabilistic_mapping * sm2).sum(3).sum(2)
    gradient = -2.0 * np.multiply.outer((sm1 - tmp), sm2)
    return gradient


def gradient_descent(probabilistic_mapping, sm1, sm2, alpha):
    """Simple, non-vectorized, gradient descent algorithm to minimize
    the 4D probabilistic loss.
    """
    probabilistic_mapping_new = np.empty(probabilistic_mapping.shape)
    for i in range(sm1.shape[0]):
        for j in range(sm1.shape[0]):
            for k in range(sm2.shape[0]):
                for l in range(sm2.shape[0]):
                    if i == j and j!=l:
                        probabilistic_mapping_new[i, j, k, l] = 0.0
                        continue

                    probabilistic_mapping_new[i, j, k, l] = probabilistic_mapping[i, j, k, l] - alpha * gradient_probabilistic_loss4D_fast(sm1, sm2, probabilistic_mapping, i, j, k, l)

    # Normalizing coefficients so that it is a valid probabilistic_mapping:
    probabilistic_mapping_new = np.abs(probabilistic_mapping_new)
    tmp = probabilistic_mapping_new.sum(3).sum(2)
    probabilistic_mapping_new = np.nan_to_num(probabilistic_mapping_new / tmp[:,:,None,None])
    return probabilistic_mapping_new


def gradient_descent_fast(probabilistic_mapping, sm1, sm2, alpha):
    """Vectorized gradient descent algorithm to minimize the 4D
    probabilistic loss.
    """
    gradient = gradient_probabilistic_loss4D_fast_fast(sm1, sm2, probabilistic_mapping)
    # for i in range(sm1.shape[0]):
    #     for k in range(sm2.shape[0]):
    #         for l in range(sm2.shape[0]):
    #             if k != l:
    #                 gradient[i, i, k, l] = 0.0

    probabilistic_mapping_new = probabilistic_mapping - alpha * gradient

    # Normalizing coefficients so that it is a valid probabilistic_mapping:
    probabilistic_mapping_new = np.abs(probabilistic_mapping_new)
    tmp = probabilistic_mapping_new.sum(3).sum(2)
    probabilistic_mapping_new = np.nan_to_num(probabilistic_mapping_new / tmp[:,:,None,None])
    return probabilistic_mapping_new

# GRAPH MAPPING SECTION

def loss_graph_mapping_squared_slow(sm1, sm2, P):
    """Graph mapping squared loss. (Slow) reference implementation.
    """
    loss = 0.0
    for i in range(sm1.shape[0]):
        for j in range(sm1.shape[0]):
            tmp = sm1[i, j]
            for k in range(sm2.shape[0]):
                for l in range(sm2.shape[0]):
                    tmp -= P[i, k] * P[j, l] * sm2[k, l]

            loss += tmp * tmp

    return loss


def loss_graph_mapping_slow(sm1, sm2, P):
    """Graph mapping loss. (Slow) reference implementation.
    """
    return np.sqrt(loss_graph_mapping_squared_slow(sm1, sm2, P))
    

def loss_graph_mapping_squared(sm1, sm2, P):
    """Squared graph mapping loss.
    """
    tmp = sm1 - np.dot(P, np.dot(sm2, P.T))
    return (tmp * tmp).sum()


def loss_graph_mapping(sm1, sm2, P):
    """Graph mapping loss.
    """
    # return np.sqrt(loss_graph_mapping_squared(sm1, sm2, P))
    # This returns a slightly different number in the last decimals:
    return np.linalg.norm(sm1 - np.dot(P, np.dot(sm2, P.T)), ord='fro')


def gradient_loss_graph_mapping(sm1, sm2, P):
    loss = loss_graph_mapping(sm1, sm2, P)
    return -2.0 * loss * (np.dot(sm2, P.T).T + np.dot(P, sm2))


def gradient_descent_loss_graph_mapping(P, sm1, sm2, alpha):
    """Vectorized gradient descent algorithm to minimize the graph
    mapping 2D probabilistic squared loss.
    """
    gradient = gradient_loss_graph_mapping(sm1, sm2, P)
    P_new = P - alpha * gradient

    # Normalizing coefficients so that it is a valid P:
    P_new = np.abs(P_new)
    P_new = np.nan_to_num(P_new / P_new.sum(1)[:,None])
    return P_new


def loss_graph_mapping_normalized_squared_slow(sm1, sm2, P):
    """Graph mapping squared loss. (Slow) reference implementation.
    """
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    return loss_graph_mapping_squared_slow(sm1, sm2, Q)


def loss_graph_mapping_normalized_slow(sm1, sm2, P):
    """Graph mapping loss. (Slow) reference implementation.
    """
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    return loss_graph_mapping_slow(sm1, sm2, Q)
    

def loss_graph_mapping_normalized(sm1, sm2, P):
    """Graph mapping loss. Vectorized.
    """
    # P = np.abs(P)
    # return np.linalg.norm(sm1 - np.dot(P, np.dot(sm2, P.T)) / P.sum(1), ord='fro')
    Q = np.abs(P / P.sum(1)[:,None])
    return loss_graph_mapping(sm1, sm2, Q)


def loss_graph_mapping_normalized_squared(sm1, sm2, P):
    """Squared graph mapping loss.
    """
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    # tmp = sm1 - np.dot(Q, np.dot(sm2, Q.T))
    # return (tmp * tmp).sum()
    return loss_graph_mapping_squared(sm1, sm2, Q)


def gradient_loss_graph_mapping_normalized_squared(sm1, sm2, P):
    """THIS IS MOST PROBABLY INCORRECT!
    """
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    gradient = -2.0 * np.dot((sm1 - np.dot(Q, np.dot(sm2, Q.T))), (np.dot(sm2, Q.T).T + np.dot(Q, sm2))) # gradient of the normalized loss
    gradient_normalization = np.dot(np.sign(P).T, ((P.sum(1)[:,None] - P) / np.dot(P.sum(1), P.sum(1)[:,None])))
    # gradient *= np.sign(P) * (P.sum(1)[:,None] - P) / (P.sum(1) * P.sum(1))[:,None] # gradient of the normalization/constraints
    return np.dot(gradient, gradient_normalization)


def gradient_loss_graph_mapping_normalized_squared_slow(sm1, sm2, P):
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    gradient = np.zeros((sm1.shape[0], sm2.shape[0]))
    gradient_Q = np.zeros((sm1.shape[0], sm2.shape[0]))
    for n in range(sm1.shape[0]):
        for m in range(sm2.shape[0]):
            gradient1 = 0.0
            gradient2 = 0.0
            for j in range(sm1.shape[0]):
                tmp0 = sm1[n, j]
                for l in range(sm2.shape[0]):
                    for k in range(sm2.shape[0]):
                        tmp0 -= Q[n, k] * sm2[k, l] * Q[j, l]

                tmp1 = 0.0
                for l in range(sm2.shape[0]):
                    tmp1 += sm2[m, l] * Q[j, l]

                gradient1 += tmp0 * tmp1

            for i in range(sm1.shape[0]):
                tmp2 = sm1[i, n]
                for l in range(sm2.shape[0]):
                    for k in range(sm2.shape[0]):
                        tmp2 -= Q[i, k] * sm2[k, l] * Q[n, l]

                tmp3 = 0.0
                for k in range(sm2.shape[0]):
                    tmp3 += Q[i, k] * sm2[k, m]
        
                gradient2 += tmp2 * tmp3

            gradient[n, m] = -2.0 * (gradient1 + gradient2)

            gradient_Q[n, m] = np.sign(Q[n, m]) * (P[n, :].sum() - P[n, m])
            gradient[n, m] = gradient[n, m] * gradient_Q[n, m]

    return gradient


def gradient_loss_graph_mapping_normalized_squared_slow2(sm1, sm2, P):
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    gradient = np.zeros((sm1.shape[0], sm2.shape[0]))
    gradient_Q = np.zeros((sm1.shape[0], sm2.shape[0]))
    for n in range(sm1.shape[0]):
        for m in range(sm2.shape[0]):
            gradient1 = 0.0
            for j in range(sm1.shape[0]):
                tmp0 = sm1[n, j]
                for k in range(sm2.shape[0]):
                    for l in range(sm2.shape[0]):
                        tmp0 -= Q[n, k] * Q[j, l] * sm2[k, l]

                tmp1 = 0.0
                for l in range(sm2.shape[0]):
                    tmp1 += Q[j, l] * sm2[m, l]

                gradient1 += tmp0 * tmp1

            gradient2 = 0.0            
            for i in range(sm1.shape[0]):
                tmp2 = sm1[i, n]
                for k in range(sm2.shape[0]):
                    for l in range(sm2.shape[0]):
                        tmp2 -= Q[i, k] * Q[n, l] * sm2[k, l]

                tmp3 = 0.0
                for k in range(sm2.shape[0]):
                    tmp3 += Q[i, k] * sm2[k, m]

                gradient2 += tmp2 * tmp3
                
            gradient[n, m] = -2.0 * (gradient1 + gradient2)

            tmp = P[n, :].sum()
            gradient_Q[n, m] = np.sign(Q[n, m]) * (tmp - P[n, m]) / (tmp * tmp)
            gradient[n, m] = gradient[n, m] * gradient_Q[n, m]
            
    return gradient


def gradient_loss_graph_mapping_normalized_squared_slow2_stable(sm1, sm2, P):
    Q = np.abs(P / P.sum(1)[:,None]) # normalization
    gradient = np.zeros((sm1.shape[0], sm2.shape[0]))
    gradient_Q = np.zeros((sm1.shape[0], sm2.shape[0]))
    for n in range(sm1.shape[0]):
        for m in range(sm2.shape[0]):
            gradient1 = 0.0
            gradient11 = np.empty(sm1.shape[0])
            for j in range(sm1.shape[0]):
                tmp0 = sm1[n, j]
                tmp00 = np.empty((sm2.shape[0], sm2.shape[0]))
                for k in range(sm2.shape[0]):
                    for l in range(sm2.shape[0]):
                        tmp00[k, l] = Q[n, k] * Q[j, l] * sm2[k, l]
                        
                tmp0 -= np.sort(tmp00.flatten()).sum() # stable sum
                tmp1 = 0.0
                tmp11 = np.empty(sm2.shape[0])
                for l in range(sm2.shape[0]):
                    tmp11[l] = Q[j, l] * sm2[m, l]

                tmp1 += np.sort(tmp11.flatten()).sum() # stable sum
                gradient11[j] = tmp0 * tmp1

            gradient1 += np.sort(gradient11.flatten()).sum()

            gradient2 = 0.0
            gradient22 = np.empty(sm1.shape[0])
            for i in range(sm1.shape[0]):
                tmp2 = sm1[i, n]
                tmp22 = np.empty((sm2.shape[0], sm2.shape[0]))
                for k in range(sm2.shape[0]):
                    for l in range(sm2.shape[0]):
                        tmp22[k, l] = Q[i, k] * Q[n, l] * sm2[k, l]

                tmp2 -= np.sort(tmp22.flatten()).sum()
                tmp3 = 0.0
                tmp33 = np.empty(sm2.shape[0])
                for k in range(sm2.shape[0]):
                    tmp33[k] = Q[i, k] * sm2[k, m]

                tmp3 += np.sort(tmp33.flatten()).sum()
                gradient22[i] = tmp2 * tmp3

            gradient2 += np.sort(gradient22.flatten()).sum()
                
            gradient[n, m] = -2.0 * (gradient1 + gradient2)

            tmp = P[n, :].sum()
            gradient_Q[n, m] = np.sign(Q[n, m]) * (tmp - P[n, m]) / (tmp * tmp)
            gradient[n, m] = gradient[n, m] * gradient_Q[n, m]
            
    return gradient


# THIS DOES NOT WORK: logsubexp requires loga > logb, which is not always true in this case
# from inference_with_classifiers.logvar import *
# def gradient_loss_graph_mapping_normalized_squared_slow2_logspace(sm1, sm2, P):
#     Q = np.abs(P / P.sum(1)[:,None]) # normalization
#     gradient = np.zeros((sm1.shape[0], sm2.shape[0]))
#     gradient_Q = np.zeros((sm1.shape[0], sm2.shape[0]))
#     for n in range(sm1.shape[0]):
#         for m in range(sm2.shape[0]):
#             gradient1 = np.log(0.0)
#             for j in range(sm1.shape[0]):
#                 tmp0 = np.log(sm1[n, j])
#                 for k in range(sm2.shape[0]):
#                     for l in range(sm2.shape[0]):
#                         tmp0 = logsubexp(tmp0, np.log(Q[n, k]) + np.log(Q[j, l]) + np.log(sm2[k, l]))

#                 tmp1 = np.log(0.0)
#                 for l in range(sm2.shape[0]):
#                     tmp1 = logaddexp(tmp1, np.log(Q[j, l]) + np.log(sm2[m, l]))

#                 gradient1 = logaddexp(gradient1, tmp0 + tmp1)

#             gradient2 = np.log(0.0)
#             for i in range(sm1.shape[0]):
#                 tmp2 = np.log(sm1[i, n])
#                 for k in range(sm2.shape[0]):
#                     for l in range(sm2.shape[0]):
#                         tmp2 = logsubexp(tmp2, np.log(Q[i, k]) + np.log(Q[n, l]) + np.log(sm2[k, l]))

#                 tmp3 = np.log(0.0)
#                 for k in range(sm2.shape[0]):
#                     tmp3 = logaddexp(tmp3, np.log(Q[i, k]) * np.log(sm2[k, m]))

#                 gradient2 = logaddexp(gradient2, tmp2 + tmp3)
                
#             gradient[n, m] = logaddexp(gradient1, gradient2)

#             tmp = P[n, :].sum()
#             gradient_Q[n, m] = np.sign(Q[n, m]) * (tmp - P[n, m]) / (tmp * tmp)
            
#     return -2.0 * np.exp(gradient) * gradient_Q

            
def gradient_descent_loss_graph_mapping_normalized_squared(P, sm1, sm2, alpha):
    """Vectorized gradient descent algorithm to minimize the graph
    mapping 2D probabilistic squared loss.
    """
    gradient = gradient_loss_graph_mapping_normalized_squared_slow2(sm1, sm2, P)
    P_new = P - alpha * gradient
    return P_new


