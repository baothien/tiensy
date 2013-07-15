#print __doc__

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

import time as time
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#from sklearn.cluster import Ward
from hierarchical import Ward
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
     
if __name__ == '__main__':

    np.random.seed(0)     
    
    data_ids =[109]#205]#205]#,205]# [101, 109,201,205]#[101,109, 201, 205, 210]#,201]#,202]
    num_cluster = 50
    num_prototype = 40
    num_neighbors = [50] 
    marker = ['-']#[':', '--', '-','-.']
    
    plt.figure()  

    for mk, data_id in enumerate(data_ids):
        print 'Subject/Control: ', data_id                       
        #file_name_dis = 'Results/'+str(data_id)+'/'+str(data_id)+'_data_disimilarity_full_tracks_' + str(num_prototype) + '_prototyes_random_modified_ward_full_tree_130524.dis'            
        file_name_dis = 'Results/'+str(data_id)+'/'+str(data_id)+'_data_disimilarity_full_tracks_40_prototyes_random_130516.dis'        
        #save_pickle(file_name_dis,X)        
        #print 'Saving data_disimilarity: ',file_name_dis,' - done'
        X = load_pickle(file_name_dis)

        for num_neigh in num_neighbors:                    
            print "\tGenerating at ", num_neigh, " neighbor"                
            #101_full_tracks_50_neighbors_modified_ward_full_tree_130516.tree
            file_name = 'Results/'+str(data_id)+'/'+ str(data_id)+'_full_tracks_' + str(num_neigh) + '_neighbors_modified_ward_full_tree_130516.tree'                        
            ward = load_pickle(file_name)                   
            
            ###############################################################################
            # plot the R_alpha function
            #fig = plt.figure()
            x = np.linspace(0, 1, 6*ward.height_[len(ward.height_)-1])
            y = [R_function(ward.R_alpha_,ward.height_[len(ward.height_)-1],x_i) for x_i in x]
                            
            #plt.plot(x, y, 'g-', label = '201', markersize = 1.2)            
            #plt.plot(x, y,linestyle='-','rD', label = str(data_id), markersize = 1.8)            
            plt.plot(x, y, marker[mk],color ='black', label = str(data_id), markersize = 8)            
    #       plt.xlabel('scale: ' + str(1./ward.height_[len(ward.height_)-1]))
            #markers_on = np.linspace(0, 1, ward.height_[len(ward.height_)-1])+0.5/ ward.height_[len(ward.height_)-1]
            #y_markers_on = [R_function(ward.R_alpha_,ward.height_[len(ward.height_)-1],x_i+0.5/ ward.height_[len(ward.height_)-1]) for x_i in markers_on]
            #plt.plot(markers_on, y_markers_on, 'rD')
    plt.legend(loc='lower right')    
    plt.xlabel('scale')
    plt.ylabel('goodness')
    #plt.title('Choosing scale based on goodness of a cut')            
    plt.show()

    
        
    

