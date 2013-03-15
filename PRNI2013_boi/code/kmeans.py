def kmeans(tracks, num_clusters=10):
    r""" Efficient tractography clustering

    Every track can needs to have the same number of points.
    Use dipy.tracking.metrics.downsample to restrict the number of points

    Parameters
    -----------
    tracks : sequence
        of tracks as arrays, shape (N,3) .. (N,3) where N=points
    d_thr : float, average euclidean distance threshold

    Returns
    --------
    C : dict

    Examples
    ----------
    >>> from dipy.tracking.distances import local_skeleton_clustering
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),
            np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
            np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
            np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
            np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
            np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
            np.array([[0,0,0],[0,1,0],[0,2,0]])]
    >>> C=local_skeleton_clustering(tracks,d_thr=0.5,3)

    Notes
    ------
    The distance calculated between two tracks::

        t_1       t_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ the euclidean distance between t_1[0] and
    t_2[0], $b$ between t_1[1] and t_2[1] and $c$ between t_1[2] and t_2[2].
    Also the same with t2 flipped (so t_1[0] compared to t_2[2] etc).

    Visualization
    --------------

    It is possible to visualize the clustering C from the example
    above using the fvtk module::

        from dipy.viz import fvtk
        r=fvtk.ren()
        for c in C:
            color=np.random.rand(3)
            for i in C[c]['indices']:
                fvtk.add(r,fvtk.line(tracks[i],color))
        fvtk.show(r)

    See also
    ---------
    dipy.tracking.metrics.downsample
    """
    cdef :
        cnp.ndarray[cnp.float32_t, ndim=2] track
        LSC_Cluster *cluster
        long lent = 0,lenC = 0, dim = 0, points=0
        long i=0, j=0, c=0, i_k=0, rows=0 ,cit=0
        float *ptr, *hid, *alld
        float d[2],m_d,cd_thr        
        long *flip
        
    points=len(tracks[0])
    dim = points*3
    rows = points
    cd_thr = d_thr
    
    #Allocate and copy memory for first cluster
    cluster=<LSC_Cluster *>realloc(NULL,sizeof(LSC_Cluster))
    cluster[0].indices=<long *>realloc(NULL,sizeof(long))
    cluster[0].hidden=<float *>realloc(NULL,dim*sizeof(float))    
    cluster[0].indices[0]=0
    track=np.ascontiguousarray(tracks[0],dtype=f32_dt)
    ptr=<float *>track.data    
    for i from 0<=i<dim:        
        cluster[0].hidden[i]=ptr[i]    
    cluster[0].N=1
    
    #holds number of clusters
    lenC = 1

    #store memmory for the hid variable
    hid=<float *>realloc(NULL,dim*sizeof(float))

    #Work with the rest of the tracks        
    lent=len(tracks)    
    for it in range(1,lent):        
        track=np.ascontiguousarray(tracks[it],dtype=f32_dt)
        ptr=<float *>track.data
        cit=it
        
        with nogil:
            
            alld=<float *>calloc(lenC,sizeof(float))
            flip=<long *>calloc(lenC,sizeof(long))
            for k from 0<=k<lenC:
                for i from 0<=i<dim:
                    hid[i]=cluster[k].hidden[i]/<float>cluster[k].N                
                
                #track_direct_flip_3dist(&ptr[0],&ptr[3],&ptr[6],&hid[0],&hid[3],&hid[6],d)
                #track_direct_flip_3dist(ptr,ptr+3,ptr+6,hid,hid+3,hid+6,<float *>d)                
                track_direct_flip_dist(ptr, hid,rows,<float *>d)
                
                if d[1]<d[0]:
                    d[0]=d[1]
                    flip[k]=1
                alld[k]=d[0]
            
            m_d = biggest_float
            #find minimum distance and index    
            for k from 0<=k<lenC:
                if alld[k] < m_d:
                    m_d=alld[k]
                    i_k=k
            
            if m_d < cd_thr:                
                if flip[i_k]==1:#correct if flipping is needed
                    for i from 0<=i<rows:
                        for j from 0<=j<3:                                                       
                            cluster[i_k].hidden[i*3+j]+=ptr[(rows-1-i)*3+j]
                else:
                     for i from 0<=i<rows:
                        for j from 0<=j<3:                                                       
                            cluster[i_k].hidden[i*3+j]+=ptr[i*3+j]
                cluster[i_k].N+=1
                cluster[i_k].indices=<long *>realloc(cluster[i_k].indices,cluster[i_k].N*sizeof(long))
                cluster[i_k].indices[cluster[i_k].N-1]=cit
                
            else:#New cluster added
                lenC+=1
                cluster=<LSC_Cluster *>realloc(cluster,lenC*sizeof(LSC_Cluster))
                cluster[lenC-1].indices=<long *>realloc(NULL,sizeof(long))
                cluster[lenC-1].hidden=<float *>realloc(NULL,dim*sizeof(float))    
                cluster[lenC-1].indices[0]=cit
                for i from 0<=i<dim:        
                    cluster[lenC-1].hidden[i]=ptr[i]    
                cluster[lenC-1].N=1
                            
            free(alld)
            free(flip)    
    
            
    #Copy results to a dictionary
    
    C={}
    for k in range(lenC):
        
        C[k]={}
        C[k]['hidden']=np.zeros(points*3,dtype=np.float32)
        
        for j in range(points*3):
            C[k]['hidden'][j]=cluster[k].hidden[j]            
        C[k]['hidden'].shape=(points,3)
            
        C[k]['N']=cluster[k].N
        C[k]['indices']=np.zeros(cluster[k].N,dtype=np.int64)
        
        for i in range(cluster[k].N):            
            C[k]['indices'][i]=cluster[k].indices[i]            
        
        C[k]['indices']=list(C[k]['indices'])
    
    #Free memory
    with nogil:   
        
        for k from 0<=k<lenC:
            free(cluster[k].indices)
            free(cluster[k].hidden)
        free(cluster)
    
    return C

