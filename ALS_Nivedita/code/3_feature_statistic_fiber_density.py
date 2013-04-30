# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:30:34 2012

@author: bao
"""

import numpy as np
import nibabel as nib

#dipy modules
#from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
#from dipy.tracking.metrics import length




patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]    

if __name__ == '__main__':
    
    dir_name = 'data'
    num_seeds = 3  
    
    print 'Name \t number of fibers \t volumn \t fiber density'    
    for a in ['L','R']:
        #for p in patients:      
        for p in controls:        
            #compute number of fibers
            cst_file_name =  dir_name + '/segmentation/' + str(p) + '_corticospinal_' + a + '_' + str(num_seeds) + 'M.pkl'           
            #load id of tracks in CST left        
            tracks_id=load_pickle(cst_file_name)
            number_fibers = len(tracks_id)
                        
            #compute the volumn of CST
            cordinate_file_name =  dir_name + '/segmentation/cordinate_' + str(p) + '_corticospinal_' + a + '_' + str(num_seeds) + 'M.pkl'           
            cordinate_id=load_pickle(cordinate_file_name)
            volumn =  len(cordinate_id)           
            
            #compute fiber density = number_fibers/volumn
            fiber_density =  number_fibers*1./volumn           
            print cordinate_file_name, '\t', number_fibers, '\t', volumn, '\t', fiber_density