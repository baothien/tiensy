#print __doc__

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
#from sklearn.cluster import Ward
from hierarchical import Ward
from sklearn.neighbors import kneighbors_graph
from dipy.io.pickles import save_pickle,load_pickle 

#for computing the CPU time, not elapsed time
import resource
def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]
    

def R_function(parameters, x_range, x):
    i = 0
    while (i < x_range and x>=1.*i/x_range):        
            i = i + 1
    i = i -1
    #print 'x = ', x, 'i = ',i
    return parameters[i][0]* x * x + parameters[i][1]*x + parameters[i][2]
   
def load_data(figure, data_id):   
    
    if figure=='small_dataset':
        filename = 'ALS_Data/'+ str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_10K.dpy'        
    elif figure=='median_dataset':
        filename = 'ALS_Data/' + str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_1M.dpy'    
    elif figure=='big_dataset':
        filename = 'ALS_Data/' + str(data_id) + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy'
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    return tracks

###############################################################################
###############################################################################
if __name__ == '__main__':

    np.random.seed(0)     
    
    data_ids = [210]#,201]#,202]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [50]#100]#[25 ,50]#, 75, 100]#[10,20,30,40,50,60,70,80,90,100]      

    for data_id in data_ids:
        print 'Subject/Control: ', data_id
        
        #data = load_data('big_dataset',data_id)   

        #X = compute_disimilarity(data, bundles_distances_mam, 'random', num_prototype,len(data))        
        
        #file_name_dis = 'Results/'+str(data_id)+'/'+str(data_id)+'_data_disimilarity_full_tracks_' + str(num_prototype) + '_prototyes_random_modified_ward_full_tree_130524.dis'            
        file_name_dis = 'Results/'+str(data_id)+'/'+str(data_id)+'_data_disimilarity_full_tracks_40_prototyes_random_130516.dis'        
        #save_pickle(file_name_dis,X)        
        #print 'Saving data_disimilarity: ',file_name_dis,' - done'
        X = load_pickle(file_name_dis)

        for num_neigh in num_neighbors:                    
            print "\tGenerating at ", num_neigh, " neighbor"                
            
            file_name = str(data_id)+'_full_tracks_' + str(num_neigh) + '_neighbors_modified_ward_full_tree_130516_new.tree'            
            connectivity = kneighbors_graph(X, n_neighbors=num_neigh)                          
            
            st = cpu_time()#time.clock()
            ward = Ward(n_clusters=num_cluster, compute_full_tree=True, connectivity=connectivity).fit(X)
            #t = time.clock() - st
            t = cpu_time() - st
            
            #-----------------------------------------------------------------------------
            # saving the result                       
            save_pickle(file_name,ward)
            print '\tSaving tree: ',file_name,' - done'
            
            