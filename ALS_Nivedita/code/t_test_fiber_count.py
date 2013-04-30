import numpy as np
import scipy.stats as ss
from dipy.io.pickles import load_pickle,save_pickle

if __name__ == '__main__':

    #np.random.seed(1) #to ensure that the result is reproducity - have the same value for any run time

#    n_P = 14
#    n_C = 14
#    vol_mean_P = 200
#    vol_sigma_P = 50
#    vol_mean_C = 300
#    vol_sigma_C = 50

#    n_P = 13
#    n_C = 14
#    vol_mean_P = 200
#    vol_sigma_P = 30
#    vol_mean_C = 300
#    vol_sigma_C = 50
#    n = n_P + n_C
#
#    vol_P = np.random.normal(loc=vol_mean_P, scale=vol_sigma_P, size=n_P)
#    vol_C = np.random.normal(loc=vol_mean_C, scale=vol_sigma_C, size=n_C)
    
    for a in ['L','R']:#,'R']:   
        #patient_name = 'data/segmentation/patient_fiber_number_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_number_corticospinal_' + a + '_3M.txt'   
        
        #patient_name = 'data/segmentation/patient_fiber_len_min_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_len_min_corticospinal_' + a + '_3M.txt'   
        
        #patient_name = 'data/segmentation/patient_fiber_len_max_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_len_max_corticospinal_' + a + '_3M.txt'   
        
        #patient_name = 'data/segmentation/patient_fiber_len_avg_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_len_avg_corticospinal_' + a + '_3M.txt'   
        
        #patient_name = 'data/segmentation/patient_fiber_truth_len_min_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_truth_len_min_corticospinal_' + a + '_3M.txt'   
        
        #patient_name = 'data/segmentation/patient_fiber_truth_len_max_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_truth_len_max_corticospinal_' + a + '_3M.txt'   

        #patient_name = 'data/segmentation/patient_fiber_truth_len_avg_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_truth_len_avg_corticospinal_' + a + '_3M.txt'   

#        patient_name = 'data/segmentation/patient_fiber_volumn_corticospinal_' + a + '_3M.txt'          
#        control_name = 'data/segmentation/control_fiber_volumn_corticospinal_' + a + '_3M.txt'   

        #patient_name = 'data/segmentation/patient_fiber_density_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_density_corticospinal_' + a + '_3M.txt'   
 
        #patient_name = 'data/segmentation/patient_fiber_fa_corticospinal_' + a + '_3M.txt'          
        #control_name = 'data/segmentation/control_fiber_fa_corticospinal_' + a + '_3M.txt'   
 
        patient_name = 'data/segmentation/patient_fiber_md_corticospinal_' + a + '_3M.txt'          
        control_name = 'data/segmentation/control_fiber_md_corticospinal_' + a + '_3M.txt'   
 
        features_P=load_pickle(patient_name)
        features_C=load_pickle(control_name)
        print patient_name, features_P
        print control_name, features_C                 
       
        t, p_value = ss.ttest_ind(features_P, features_C)
        print a        
        print 't = ',t
        print 'p = ',  p_value
