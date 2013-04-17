"""Hierarchical Agglomerative Clustering

These routines perform some hierachical agglomerative clustering of some
input data. Currently, only Ward's algorithm is implemented.

Authors : Vincent Michel, Bertrand Thirion, Alexandre Gramfort,
          Gael Varoquaux
License: BSD 3 clause
"""
from heapq import heapify, heappop, heappush, heappushpop
import warnings

import numpy as np
from scipy import sparse
from scipy.cluster import hierarchy

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._csgraph import cs_graph_components
from sklearn.externals.joblib import Memory
from sklearn.metrics import euclidean_distances
from sklearn.utils import array2d

from sklearn.cluster import _hierarchical
from sklearn.cluster._feature_agglomeration import AgglomerationTransform

'''
return the value of function R_alpha
R_alpha_function = A*x^2 + B*x + C
where A, B, C = A[i], B[i], C[i]  if (x>=1.*i/x_range and x<1.*(i+1)/x_range)
'''
def R_alpha_function(parameters, x_range, x):
    i = 0
    while (i < x_range and x>=1.*i/x_range):        
            i = i + 1
    i = i -1
    #print 'x = ', x, 'i = ',i
    return parameters[i][0]* x * x + parameters[i][1]*x + parameters[i][2]

###############################################################################
# Ward's algorithm

def ward_tree(X, connectivity=None, n_components=None, copy=True,
              n_clusters=None):
    """Ward clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account a some topological
    structure between samples.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered

    connectivity : sparse matrix.
        connectivity matrix. Defines for each sample the neigbhoring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.

    copy : bool (optional)
        Make a copy of connectivity or work inplace. If connectivity
        is not of LIL type there will be a copy in any case.

    n_clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    Returns
    -------
    children : 2D array, shape (n_nodes, 2)
        list of the children of each nodes.
        Leaves of the tree have empty list of children.

    n_components : sparse matrix.
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree

    parents : 1D array, shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        if n_clusters is not None:
            warnings.warn('Early stopping is implemented only for '
                          'structured Ward clustering (i.e. with '
                          'explicit connectivity.', stacklevel=2)
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.int)
        return children_, 1, n_samples, None

    # Compute the number of nodes
    if n_components is None:
        n_components, labels = cs_graph_components(connectivity)

    # Convert connectivity matrix to LIL with a copy if needed
    if sparse.isspmatrix_lil(connectivity) and copy:
        connectivity = connectivity.copy()
    elif not sparse.isspmatrix(connectivity):
        connectivity = sparse.lil_matrix(connectivity)
    else:
        connectivity = connectivity.tolil()

    if n_components > 1:
        warnings.warn("the number of connected components of the "
                      "connectivity matrix is %d > 1. Completing it to avoid "
                      "stopping the tree early." % n_components)
        connectivity = _fix_connectivity(X, connectivity, n_components, labels)
        n_components = 1

    if n_clusters is None:
        n_nodes = 2 * n_samples - n_components
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters

    if (connectivity.shape[0] != n_samples
            or connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s '
                         'when X is %s' % (connectivity.shape, X.shape))

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.int)
    coord_col = np.array(coord_col, dtype=np.int)

    # build moments as a list
    moments_1 = np.zeros(n_nodes)
    moments_1[:n_samples] = 1
    moments_2 = np.zeros((n_nodes, n_features))
    moments_2[:n_samples] = X
    inertia = np.empty(len(coord_row), dtype=np.float)
    _hierarchical.compute_ward_dist(moments_1, moments_2, coord_row, coord_col,
                                    inertia)
    inertia = zip(inertia, coord_row, coord_col)
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.int)
    heights = np.zeros(n_nodes)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []

    not_visited = np.empty(n_nodes, dtype=np.int8)

    # recursive merge loop
    for k in xrange(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j], heights[k] = k, k, inert
        children.append([i, j])
        used_node[i] = used_node[j] = False

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[l].append(k) for l in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.int)
        coord_row = np.empty_like(coord_col)
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float)

        _hierarchical.compute_ward_dist(moments_1, moments_2,
                                        coord_row, coord_col, ini)
        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx]))
            for idx in xrange(n_additions)]

    # Separate leaves in children (empty lists up to now)    
    n_leaves = n_samples
    children = np.array(children)  # return numpy array for efficient caching
    parent = np.array(parent)
    #print 'parent = ', parent
#--------moi them vao 04-04-2013 -----------------------    
    #heights = np.array(heights)  # return numpy array for efficient caching
    

    return children, n_components, n_leaves, parent


###############################################################################
# For non fully-connected graphs

def _fix_connectivity(X, connectivity, n_components, labels):
    """
    Warning: modifies connectivity in place
    """
    for i in range(n_components):
        idx_i = np.where(labels == i)[0]
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.where(labels == j)[0]
            Xj = X[idx_j]
            D = euclidean_distances(Xi, Xj)
            ii, jj = np.where(D == np.min(D))
            ii = ii[0]
            jj = jj[0]
            connectivity[idx_i[ii], idx_j[jj]] = True
            connectivity[idx_j[jj], idx_i[ii]] = True
    return connectivity

###############################################################################
# Functions for cutting  hierarchical clustering tree


def _hc_cut(n_clusters, children, n_leaves):
    """Function cutting the ward tree for a given number of clusters.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.

    children : list of pairs. Length of n_nodes
        List of the children of each nodes.
        Leaves have empty list of children and are not stored.

    n_leaves : int
        Number of leaves of the tree.

    Returns
    -------
    labels : array [n_samples]
        cluster labels for each point

    """
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree
    nodes = [-(max(children[-1]) + 1)]
    for i in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.int)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label


###############################################################################
# Class for Ward hierarchical clustering

class Ward(BaseEstimator, ClusterMixin):
    """Ward hierarchical clustering: constructs a tree and cuts it.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to find.

    connectivity : sparse matrix.
        Connectivity matrix. Defines for each sample the neigbhoring
        samples following a given structure of the data.
        Default is None, i.e, the hiearchical clustering algorithm is
        unstructured.

    memory : Instance of joblib.Memory or string
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    copy : bool
        Copy the connectivity matrix or work inplace.

    n_components : int (optional)
        The number of connected components in the graph defined by the \
        connectivity matrix. If not set, it is estimated.

    compute_full_tree: bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of cluster and using caching, it may
        be advantageous to compute the full tree.


    Attributes
    ----------
    `children_` : array-like, shape = [n_nodes, 2]
        List of the children of each nodes.  Leaves of the tree do not appear.

    `labels_` : array [n_samples]
        cluster labels for each point

    `n_leaves_` : int
        Number of leaves in the hiearchical tree.

    `n_components_` : sparse matrix.
        The estimated number of connected components in the graph.

    """

    def __init__(self, n_clusters=2, memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, copy=True, n_components=None,
                 compute_full_tree='auto'):
        self.n_clusters = n_clusters
        self.memory = memory
        self.copy = copy
        self.n_components = n_components
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.R_alpha_ = None

    def fit(self, X):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.

        Returns
        -------
        self
        """
        memory = self.memory
        X = array2d(X)
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory, verbose=0)

        if not self.connectivity is None:
            if not sparse.issparse(self.connectivity):
                raise TypeError("`connectivity` should be a sparse matrix or "
                                "None, got: %r" % type(self.connectivity))

            if (self.connectivity.shape[0] != X.shape[0] or
                    self.connectivity.shape[1] != X.shape[0]):
                raise ValueError("`connectivity` does not have shape "
                                 "(n_samples, n_samples)")

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            # Early stopping is likely to give a speed up only for
            # a large number of clusters. The actual threshold
            # implemented here is heuristic
            compute_full_tree = self.n_clusters > max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        self.children_, self.n_components_, self.n_leaves_, self.parent_ = \
            memory.cache(ward_tree)(X, self.connectivity,
                                    n_components=self.n_components,
                                    copy=self.copy, n_clusters=n_clusters)
        # Cut the tree 
        parents=self.parent_
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters, self.children_,
                                   self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reasign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        #---------------------------------------------------------------------
        # moi them vao 04-04-2013 de chon nhat cut cho cay            
        # each node we need height of that node, and alpha_min, alpha_max
        # compute height of each node, only in the case of compute full tree
        '''
        #New Attributes
        ----------
        `height_` : int, array [n_nodes+n_leaves]
            List of the hieght of each node of the tree (included leaves of the tree, which have height value 0)
    
        `alpha_` : int, array-like, shape = [n_nodes+n_leaves,2] 
            List of alpha_min, alpha_max of each nodes ((included leaves of the tree, which have alpha_min value 0)
            It is not true alpha, just the height of that node --> to get the true alpha: height/heigh_max
        
        `R_alpha_` : float, array-like,  shape = [height_max,3], corresponding to A, B and C; where height_max = height_[n_nodes+n_leaves - 1]
            List of A, B, C of the R_alpha function
            R_alpha(C) = 1/2 * (alpha_max(C)-alpha_min(C)) + 2*(alpha_max(C) - alpha)(alpha-alpha_min(C))/(alpha_max(C)-alpha_min(C))
            R(alpha) = 1/n * sum(|C|R_alpha(C)) for all C in P_alpha 
            ref: Pascal Pon 2011, Post-processing hierarchial community structures: Quality improvements and multi-scale view
         
        '''
        n_nodes = self.n_leaves_ + len(self.children_)               
        self.height_=np.zeros(n_nodes,dtype=np.int)         
        self.alpha_ = []
        if (self.compute_full_tree == True):                   
            for k in range(len(self.children_)):   
                 i = k + self.n_leaves_
                 if (self.height_[i]== 0):
                     self.height_[i]=self.height(i)            
            for i in range(n_nodes):   
              alpha_min = self.height_[i]
              alpha_max = self.height_[self.parent_[i]]
              self.alpha_.append([alpha_min,alpha_max])  
              
            self.compute_R_alpha()
            self.best_cut_=[]
        return self
    
    '''
       calculate the height of a given node 
       input: node: the index of a node, start from 0
              the maximum value of node is n_leaves + len(self.children)
    '''
    def height(self, node):    
        if node<self.n_leaves_:
             return 0
        left = self.children_[node-self.n_leaves_][0]
        right = self.children_[node-self.n_leaves_][1]
        if (left >= self.n_leaves_ and self.height_[left] ==0):
             self.height_[left] = self.height(left)
        if (right >= self.n_leaves_ and self.height_[right] ==0):
             self.height_[right] = self.height(right)
        return 1 + max(self.height_[left],self.height_[right])
         
    '''
       cut the hieararchical tree at the height_cut
       input: 
              height_cut: the height that to be cut the tree, start from 0
                          the maximum value of height is self.height_[n_leaves + len(self.children)-1]
       output:
              the list of the node
    '''
    def cut(self, height_cut):    
        if  (height_cut < 0 or height_cut > self.height_[self.n_leaves_ + len(self.children_)-1]):
            return None 
        if height_cut < 1:
            return np.arange(self.n_leaves_)
        if height_cut ==self.height_[self.n_leaves_ + len(self.children_)-1]:
            return [self.n_leaves_ + len(self.children_)-1]        
        node_list=[]
        for i in np.arange(self.n_leaves_+ len(self.children_)):
            [h_min, h_max] = self.alpha_[i]
            if (height_cut >= h_min and height_cut < h_max):
                node_list.append(i)    
        return node_list
            
    '''
        list all of elements belonging to a given node
        input:  
              node: the index of a node, start from 0
              the maximum value of node is n_leaves + len(self.children) 
        output: 
               the list of the index of elements (leaf nodes) belonging to that node        
    '''
    def node_list(self,node):
        if (node<0 or node > (self.n_leaves_ + len(self.children_))):
            return None
        if node <self.n_leaves_:
            return [node]
        if (node == self.n_leaves_ + len(self.children_) - 1):
            return np.arange(self.n_leaves_)
        [left,right]  = self.children_[node-self.n_leaves_]
        result_list = self.node_list(left)
        list_right = self.node_list(right)
        for a in list_right:
            result_list.append(a)
        return result_list  
        
    '''
        list all of nodes at the height h belonging to a given node
        input:  
              node: the index of a node, start from 0
              the maximum value of node is n_leaves + len(self.children) 
              height h: the height of the children level
        output: 
               the list of the index of the children at level height  h belonging to that node        
    '''
    def children_list(self,node,height):
        if (node<0 or node > (self.n_leaves_ + len(self.children_))):
            return None
        if node <self.n_leaves_ and height>=0 and height<self.height_[node]:
            return [node]            
        max_height = self.height_[self.n_leaves_ + len(self.children_)-1]
        if (height <0 or height > max_height):
            return None
        if self.height_[node]==height:
            return [node]
        result = []   
        for c in self.children_[node - self.n_leaves_]:
            [al_min,al_max] = self.alpha_[c]
            if (height<al_min):                            
                for x in self.children_list(c,height):
                    result.append(x) 
            if (al_min <= height and height < al_max):
                result.append(c)
            #if (height>=al_max): do nothing, not add c into the result
        return result            
        
    '''
        compute the A_alpha, B_alpha, and C_alpha 
        input:  
               index of node calculated
        output: 
                A_alpha, B_alpha, and C_alpha         
        R_alpha(C) = 1/2 * (alpha_max(C)-alpha_min(C)) + 2*(alpha_max(C) - alpha)(alpha-alpha_min(C))/(alpha_max(C)-alpha_min(C))
        R(alpha) = 1/n * sum(|C|R_alpha(C)) for all C in P_alpha 
        ref: Pascal Pon 2011, Post-processing hierarchial community structures: Quality improvements and multi-scale view
        Note: don't compute for the root, because at root (index == self.n_leaves_ + len(self.children_)-1 ), alpha_min and alpha_max are equal 
    '''
    def compute_ABC(self,node):
        if node <0 or node >= (self.n_leaves_ + len(self.children_)-1):
            return None
        [al_min, al_max] = self.alpha_[node]
        h = self.height_[self.n_leaves_ + len(self.children_)-1]         
        alpha_min = 1.* al_min / h
        alpha_max = 1.* al_max / h
        card = len(self.node_list(node))
        #print 'h = ', h, '\t alpha_min = ', alpha_min, '\t alpha_max = ',alpha_max, '\t card = ',card
        A_alpha = -2. * card / (alpha_max - alpha_min)
        B_alpha = 2.* card * (alpha_max + alpha_min) / (alpha_max - alpha_min)
        C_alpha = 0.5 * card * (alpha_max*alpha_max + alpha_min*alpha_min - 6 * alpha_min*alpha_max) / (alpha_max - alpha_min)
        
        return A_alpha, B_alpha, C_alpha
         
    '''
        compute the best scale for visulization of the tree
        input:  
               none
        output: 
               the list of the scale        
        R_alpha(C) = 1/2 * (alpha_max(C)-alpha_min(C)) + 2*(alpha_max(C) - alpha)(alpha-alpha_min(C))/(alpha_max(C)-alpha_min(C))
        R(alpha) = 1/n * sum(|C|R_alpha(C)) for all C in P_alpha 
        ref: Pascal Pon 2011, Post-processing hierarchial community structures: Quality improvements and multi-scale view
    '''
    def best_scale(self, thres_1=0., thres_2=1.):        
#        n = self.n_leaves_
#        result = []
#        h = self.height_[self.n_leaves_ + len(self.children_)-1]
#        for k in range(h):            
#            A = B = C = 0.
#            nodes = self.cut(k)
#            for node in nodes:
#                A_alpha, B_alpha, C_alpha = self.compute_ABC(node)
#                A = A + A_alpha
#                B = B + B_alpha
#                C = C + C_alpha
#            
#            A = 1.* A / n
#            B = 1.* B / n
#            C = 1.* C / n
        result = []        
        h = self.height_[self.n_leaves_ + len(self.children_)-1]
        for k in range(h):            
            [A,B,C] = self.R_alpha_[k]
#            delta = B*B - 4.*A*C            
#            if ( delta>= 0):
#                al_1 = (-B + np.sqrt(delta)) / (2. * A)
#                al_2 = (-B - np.sqrt(delta)) / (2. * A)
#                if ((al_1 >= ((thres_1+1.*k) /h) and (al_1< ((thres_2 + 1.*k) / h))) or ((al_2 >= (thres_1+1.*k)/h) and (al_2< (thres_2+ 1.*k) / h))):
#                    result.append(k)
#            print 'delta = ', delta
#            if delta>=0: print al_1, al_2             
            if (A<>0):
                al_local_max = -1.* B / (2. * A)                
                if (al_local_max >= ((thres_1+1.*k) /h) and (al_local_max< ((thres_2 + 1.*k) / h))):
                    r_alpha = A * al_local_max * al_local_max + B * al_local_max + C
                    result.append([k,r_alpha])
            #print 'alpha_locacal_max = ', al_local_max
        #print 'height: ', h    
        return result
        
    '''
        compute the R(alpha) function of the tree
        input:  
               none
        output: 
               the list of the parameters (A,B,C) for each scale        
        R_alpha(C) = 1/2 * (alpha_max(C)-alpha_min(C)) + 2*(alpha_max(C) - alpha)(alpha-alpha_min(C))/(alpha_max(C)-alpha_min(C))
        R(alpha) = 1/n * sum(|C|R_alpha(C)) for all C in P_alpha 
        ref: Pascal Pon 2011, Post-processing hierarchial community structures: Quality improvements and multi-scale view
    '''
    def compute_R_alpha(self):
        if (self.R_alpha_<>None):
            return self.R_alpha_
        n = self.n_leaves_
        result = []
        h = self.height_[self.n_leaves_ + len(self.children_)-1]
        for k in range(h):            
            A = B = C = 0.
            nodes = self.cut(k)
            for node in nodes:
                A_alpha, B_alpha, C_alpha = self.compute_ABC(node)
                A = A + A_alpha
                B = B + B_alpha
                C = C + C_alpha
            
            A = 1.* A / n
            B = 1.* B / n
            C = 1.* C / n
            result.append([A,B,C])        
        self.R_alpha_ = result
        return result
    '''
        estimate the best cut of the tree
        input:  
               none
        output: 
               the list of the best cuts (each cuts corresponding to a value of height)              
       
    '''
    def best_cut(self,step=1):
        best_scale = self.best_scale()
        result = [] 
        h = len(best_scale)
        k = step
        while k < h-step:                                    
            accept = True
            #check for the local maximum
            for j in range(step):
                i = j+1
                if ((best_scale[k][1]<best_scale[k-i][1]) or (best_scale[k][1]<best_scale[k+i][1])):
                    accept = False
            if accept == True:
                result.append(best_scale[k])
            accept = True
            #check for the local minimum
            for j in range(step):
                i = j + 1
                if ((best_scale[k][1]>best_scale[k-i][1]) or (best_scale[k][1]>best_scale[k+i][1])):
                    accept = False
            if accept == True:
                result.append(best_scale[k])
            k = k + 1       
            
        self.best_cut_ = result
        return result    
###############################################################################
# Ward-based feature agglomeration

class WardAgglomeration(AgglomerationTransform, Ward):
    """Feature agglomeration based on Ward hierarchical clustering

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters.

    connectivity : sparse matrix
        connectivity matrix. Defines for each feature the neigbhoring
        features following a given structure of the data.
        Default is None, i.e, the hiearchical agglomeration algorithm is
        unstructured.

    memory : Instance of joblib.Memory or string
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    copy : bool
        Copy the connectivity matrix or work inplace.

    n_components : int (optional)
        The number of connected components in the graph defined by the
        connectivity matrix. If not set, it is estimated.

    compute_full_tree: bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of cluster and using caching, it may
        be advantageous to compute the full tree.


    Attributes
    ----------
    `children_` : array-like, shape = [n_nodes, 2]
        List of the children of each nodes.
        Leaves of the tree do not appear.

    `labels_` : array [n_samples]
        cluster labels for each point

    `n_leaves_` : int
        Number of leaves in the hiearchical tree.

    """

    def fit(self, X, y=None, **params):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The data

        Returns
        -------
        self
        """
        return Ward.fit(self, X.T, **params)
