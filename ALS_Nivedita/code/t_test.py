import numpy as np
import scipy.stats as ss

if __name__ == '__main__':

    np.random.seed(1) #to ensure that the result is reproducity - have the same value for any run time

#    n_P = 14
#    n_C = 14
#    vol_mean_P = 200
#    vol_sigma_P = 50
#    vol_mean_C = 300
#    vol_sigma_C = 50

    n_P = 13
    n_C = 14
    vol_mean_P = 200
    vol_sigma_P = 30
    vol_mean_C = 300
    vol_sigma_C = 50
    n = n_P + n_C

    vol_P = np.random.normal(loc=vol_mean_P, scale=vol_sigma_P, size=n_P)
    vol_C = np.random.normal(loc=vol_mean_C, scale=vol_sigma_C, size=n_C)
        
    t, p_value = ss.ttest_ind(vol_P, vol_C)
    print t
    print p_value
