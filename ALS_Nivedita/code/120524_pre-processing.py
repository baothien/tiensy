# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:11:29 2012

@author: bao
"""

import os
from dipy.external.fsl import bet

#------------------------------------------------------------------------
#rename all the data
#dirname = "data"#/120523"
##dirname = "data_temp"
#for root, dirs, files in os.walk(dirname):
#   if root.endswith('MP_Rage_1x1x1_ND_3'):
#     for filename in files:        
#        if filename.endswith('.nii.gz'):
#           os.rename(root+'/'+filename,root+'/anatomy.nii.gz')
#        #if filename.endswith('.bal'):
        #   os.rename(root+'/'+filename,root+'/raw.bval')
        #if filename.endswith('.bvec'):
        #   os.rename(root+'/'+filename,root+'/raw.bvec')
#------------------------------------------------------------------------

#------------------------------------------------------------------------
#  brain extraction           
           
#from dipy.external.fsl import bet
#dirname = "data"
dirname = "112_bvecbval"#"120523"
for root, dirs, files in os.walk(dirname):  
  #if root.endswith('MP_Rage_1x1x1_ND_3'):
    for filename in files:
        if filename.endswith('raw.nii.gz'):
             bet(root+'/'+filename, root+'/rawbet.nii.gz')
#        if filename.endswith('anatomy.nii.gz'):
#             bet(root+'/'+filename, root+'/anatomybet.nii.gz')#            
#-------------------------------------------------------------------------            
            
            
            #if filename.endswith('.bval'):
             #   os.rename(root+'/'+filename,root+'/raw.bval')
              #  print(root+filename)
            #if filename.endswith('.bvec'):
             #   os.rename(root+'/'+filename,root+'/raw.bvec')
              #  print(root+filename)
            #if filename.endswith('raw.'):
             #   os.rename(root+'/'+filename,root+'/raw.nii')
              #  print(root+filename)
        #for file in files:
         #   if file.endswith('.dpy'):
          #      print(file)
   


#for root, dirs, files in os.walk(dirname):
#    if root.endswith('101_32'):
#        base_dir = root+'/' 
#        filename = 'raw'
#        base_filename = base_dir + filename        
#
#        nii_filename = base_filename + 'bet.nii.gz'
#        bvec_filename = base_filename + '.bvec'
#        bval_filename = base_filename + '.bval'        
#
#        img = nib.load(nii_filename)
#        data = img.get_data()
#        affine = img.get_affine()
#
#        bvals = np.loadtxt(bval_filename)
#        gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
#
#        base_dir2 = base_dir+ 'DTI/'   
    

