# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:21:33 2012

@author: bao
"""
import os
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy

dirname = "data/"
for root, dirs, files in os.walk(dirname):
    if root.endswith('DIFF2DEPI_EKJ_64dirs_14'):
        base_dir = root+'/'       
        basedir_recon = os.getcwd() + '/' + base_dir + 'DTI/'               
        fdpyw = basedir_recon + 'tracks_dti_10K_warped.dpy'                
        dpr_tracks_warped = Dpy(fdpyw, 'r')
        tensor_tracks_warped = dpr_tracks_warped.read_tracks()
        dpr_tracks_warped.close()
                
        s = 0        
        for k in range(len(tensor_tracks_warped)):
            s = s + len(tensor_tracks_warped[k])
        i_min = 0        
        i_max = 0
        for k in range(len(tensor_tracks_warped)):
            if len(tensor_tracks_warped[k])<len(tensor_tracks_warped[i_min]):
                i_min = k
            if len(tensor_tracks_warped[k])>len(tensor_tracks_warped[i_max]):
                i_max = k
        
#        print base_dir
#        print "Number of tracts ",tensor_tracks_warped.__len__()
#        print "Size", tensor_tracks_warped.__sizeof__()        
#        print "Min of length tract", len(tensor_tracks_warped[i_min])
#        print "Max of length tract", len(tensor_tracks_warped[i_max])   
#        #print tensor_tracks_warped[len(tensor_tracks_warped)-1]        
#        print 'Average tract length', s/len(tensor_tracks_warped)

#        print "Num tracts   Size     Min     Max    Average "        
        print base_dir,"\t", tensor_tracks_warped.__len__(), "\t",tensor_tracks_warped.__sizeof__(), "\t" ,len(tensor_tracks_warped[i_min]),"\t", len(tensor_tracks_warped[i_max]), "\t", s/len(tensor_tracks_warped)          
        
