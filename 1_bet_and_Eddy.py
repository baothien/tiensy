# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:17:53 2012

@author: bao
"""
from dipy.external.fsl import bet, eddy_correct
import os


visualize = False
#dirname = "data/1891215"
#dirname = "ADHD"
#dirname = "Sample/2020162"
dirname = "PBC2009/brain1_fr_raw"
for root, dirs, files in os.walk(dirname):    
    
    if root.endswith('MPRAGE_1_from_structual_T1Space'):
        #   anatomy bet    
        base_dir = root+'/'
        #filename = 'T1_anatomy.nii.gz'
        filename = 'fbrain1_mprage1.nii.gz'        
        base_filename = base_dir + filename        

        #bet(base_filename, root+'/fbrain0_mprage1_bet.nii.gz',options=' -R -f .2 -g 0')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .4 -g -0.05 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g -0.02 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g -0.02 -c 151 177 90')         
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .15 -g 0.0 -c 151 177 90')                        
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                
        
        bet(base_filename, root+'/bet_new.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                
    '''    
    if root.endswith('DTI'):
        
        #difusion tensor image - brain extraction 
        
        base_dir = root+'/'
        filename = 'raw.nii.gz'        
        base_filename = base_dir + filename        

        bet(base_filename, root+'/raw_bet.nii.gz',options=' -R -F -f .2 -g 0')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .4 -g -0.05 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g -0.02 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g -0.02 -c 151 177 90')         
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .15 -g 0.0 -c 151 177 90')                        
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                
        
        #bet(base_filename, root+'/raw_bet.nii.gz',options=' -R -f .2 -g 0.0' )      #-c 151 177 90')                  
        
        #eddy correction 
        
        eddy_correct(root+'/raw_bet.nii.gz',root+'/raw_bet_ecc.nii.gz')#,ref=0)
     '''
