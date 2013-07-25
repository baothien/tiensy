# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:40:15 2013

@author: bao
"""

"""Dominant set clustering: iteratively find the dominant set and then
remove it from the dataset.
"""
import numpy as np
from dominant_set import dominant_set


def ds_clustering(clusters,support_vectors, f_values, new_element):
    '''
    clustering the new element 
    Efficient Out-of-Sample extension of Dominant set clusters
    Massimiliano et. al., NIPS 2004
    for all h in S: if sum(a(h,i)*x(h)  > f(x)  then i is assigned to S)      
    '''
    if clusters ==None or support_vectors==None or new_element == None:
        return None
    sum_axs = []
    for i in np.arange(len(clusters)):
        S = clusters[i]
        S_old = S.copy()
        x = support_vectors[i]
        
        #print 'len S ', len(S), 'len x', len(x)
        
        from sklearn.metrics import euclidean_distances , pairwise_distances
        #euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
        new_arr = [new_element]
        dis = pairwise_distances(new_arr,S, metric='sqeuclidean')
        sigma2 = np.median(dis)    
        a_hj = np.exp(-dis / sigma2)      

        #print dis, a_hj
        sum_ax = 0.        
        for h in np.arange(len(S_old)):            
            sum_ax = sum_ax + a_hj[0][h]*x[h]
        #print 'i =',i,' sum_ax', sum_ax, 'f_values ', f_values[i]   
        sum_axs.append(sum_ax)
        
    #print np.argmax(sum_axs), '  ', np.max(sum_axs)   
    if np.max(sum_axs) >= 0.5*f_values[np.argmax(sum_axs)]:
            return np.argmax(sum_axs)
    return None
            

if __name__ == '__main__':

    from sklearn.metrics import pairwise_distances
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    

    np.random.seed(1)

    n = 10000
    d = 2    

    X, y = make_blobs(n, d, centers=3)

    D = pairwise_distances(X, metric='sqeuclidean')

    sigma2 = np.median(D)
    
    A = np.exp(-D / sigma2)    

    if d==2:
        plt.figure()
        for yi in np.unique(y):
            plt.plot(X[y==yi,0], X[y==yi,1], 'o')

        plt.title('Dataset')

    clusters = []
    support_vectors = []
    f_values = []
    num_clusters = 3
    colors = ['ro','go','bo']
    colors_out = ['rx','gx','bx']
    colors_out_new = ['r^','g^','b^']
    i = 0
    while num_clusters > i and A.size > 10:
        x = dominant_set(A, epsilon=2e-4)
        cutoff = 0.05*np.median(x[x>0])
        #cutoff = np.max(x) * 0.0005

#        plt.figure()
#        plt.plot(X[x<=cutoff,0], X[x<=cutoff,1], 'bo')
#        plt.plot(X[x>cutoff,0], X[x>cutoff,1], 'ro')
#        plt.title("Dominant set")

        clusters.append(X[x>cutoff,:])
        support_vectors.append(x[x>cutoff])
        f_values.append(np.dot(x,A.dot(x)))     
       
        
        # remove the dominant set
        idx = x <= cutoff
        A = A[idx, :][:, idx]
        X = X[idx, :]
        
        i = i + 1
                   
    #plot clusters
    plt.figure()
    for i in np.arange(num_clusters):
        c = clusters[i]
        x = []   
        y = []       
        for j in np.arange(len(c)):                       
            x.append(c[j][0])
            y.append(c[j][1])
        plt.plot(x,y, colors[i])

    #cluster outliers
    if len(X) >0:
        for i in np.arange(len(X)):
            print X[i]
            label = ds_clustering(clusters,support_vectors,f_values, X[i])
            if label ==None:                
                plt.plot(X[i][0],X[i][1], 'ko')
            else:
                plt.plot(X[i][0],X[i][1], colors_out[label])
                
    #cluster new elements
    n = 1000
    d = 2    
    X_new, y_new = make_blobs(n, d, centers=3)
    for i in np.arange(len(X_new)):
        label = ds_clustering(clusters,support_vectors,f_values, X_new[i])
        if label ==None:                
            plt.plot(X_new[i][0],X_new[i][1], 'k^')
        else:
            plt.plot(X_new[i][0],X_new[i][1], colors_out_new[label])

    plt.show()
        
        
