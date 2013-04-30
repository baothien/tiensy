# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:14:55 2012

@author: bao
"""
import os
import numpy as np
import nibabel as nib
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle

patients = [1,3,5,7,9,11,13,15,17,19,21,23] #12 patients
controls = [2,4,6,8,10,12,14,16,18,20,22,24]#12 controls16 miss R CST
#patients = [2]
mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]     

if __name__ == '__main__':    
    dir_name = 'data'
    #dir_name = "data_temp"
    num_seeds = 3  

    
    print 'Name \t volumn \t mean FA \t mean MD'    
    for a in ['L','R']:#,'R']:
        for p in patients:      
        #for p in controls:              
            subj = int(mapping[p])            
            #load the cordinates of each point in fibers of CST
            cordinate_file_name =  'data' + '/segmentation/cordinate_' + str(p) + '_corticospinal_' + a + '_' + str(num_seeds) + 'M.pkl'           
            cordinate_id=load_pickle(cordinate_file_name)                      
            
            base_dir = dir_name +'/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/'
            #loading fa_warped
            fa_warp = base_dir + 'fa_warped.nii.gz'            
            img = nib.load(fa_warp)
            fa = img.get_data()
            #print fa_warp, '\t',np.shape(fa)
   
            #loading md_wapred
            md_warp = base_dir + 'md_warped.nii.gz'
            img = nib.load(md_warp)
            md = img.get_data()
            #print md_warp, '\t',np.shape(fa)
            MD = FA = 0.
            volumn = len(cordinate_id)
            for x,y,z in cordinate_id:
                MD = MD + md[x,y,z]
                FA = FA + fa[x,y,z]
            print cordinate_file_name, '\t',volumn,'\t',FA/volumn,'\t', MD/volumn
                