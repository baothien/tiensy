# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:17:53 2012

@author: bao
"""
from dipy.external.fsl import bet, eddy_correct
import os


visualize = False
dirname = "data"
for root, dirs, files in os.walk(dirname):    
    if root.endswith('MPRAGE_1'):
        '''   anatomy bet    '''
        base_dir = root+'/'
        filename = 'defaced_MPRAGE.nii.gz'
        base_filename = base_dir + filename        

        bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g 0')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .4 -g -0.05 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g -0.02 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g -0.02 -c 151 177 90')         
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .15 -g 0.0 -c 151 177 90')                        
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                
        
        #bet(base_filename, root+'/raw_bet.nii.gz',options=' -R -f .2 -g 0.0' )      #-c 151 177 90')                
        
    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'):
        '''
        difusion tensor image - brain extraction 
        '''
        #base_dir = root+'/'
        #filename = 'DTI64.nii.gz'
        #base_filename = base_dir + filename        

        #bet(base_filename, root+'/raw_bet.nii.gz',options=' -F -f .2 -g 0')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .4 -g -0.05 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .2 -g -0.02 -c 151 177 90')
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .1 -g -0.02 -c 151 177 90')         
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -F -f .15 -g 0.0 -c 151 177 90')                        
        #bet(base_filename, root+'/rawbet.nii.gz',options=' -R -f .2 -g 0.0 -c 151 177 90')                
        
        #bet(base_filename, root+'/raw_bet.nii.gz',options=' -R -f .2 -g 0.0' )      #-c 151 177 90')                  
        '''
        eddy correction 
        '''
        eddy_correct(root+'/rawbet.nii.gz',root+'/rawbet_ecc.nii.gz')#,ref=0)
     
