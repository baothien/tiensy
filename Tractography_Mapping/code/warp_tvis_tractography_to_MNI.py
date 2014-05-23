# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:48:31 2014

@author: bao
"""
import numpy as np
from common_functions import warp_tracks_linearly, load_whole_tract
def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="warp_tracks_linearly" and var!="load_whole_tract" and var!="save" and var!="vis" and var!="np"]
    for var in all:
        del globals()[var]
        

source_ids =[201 , 202, 203, 204,  205, 206, 207, 208, 209, 210, 212,213]  
vis = False
save = True
        
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    print 'Source: ', source_ids[s_id]
    source = str(source_ids[s_id])

    flirt_filename = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + source+'/DIFF2DEPI_EKJ_64dirs_14/DTI/flirt.mat'
    fa_filename= '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + source + '/DTI/dti_fa_brain.nii.gz' 
        
    tracks_filename = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + source + '/DTI/dti.trk'
    linear_filename =  '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + source + '/DTI/dti_linear.dpy'        
    warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename)
    print 'Saved ', linear_filename
    tract_linear = load_whole_tract(linear_filename)
    if save:
        from common_functions import save_tract_trk
        fa_warped = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + source+'/DIFF2DEPI_EKJ_64dirs_14/DTI/fa_warped.nii.gz'     
        fname_out = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + source + '/DTI/dti_linear.trk'        
                
        save_tract_trk(tract_linear, fa_warped, fname_out )
        print 'Saved ', fname_out
    if vis:
        tract = load_whole_tract(tracks_filename)        
        from dipy.viz import fvtk
        from common_functions import visualize_tract
        ren = fvtk.ren()
        visualize_tract(ren, tract,color=fvtk.red)
        visualize_tract(ren, tract_linear,color=fvtk.blue)
        fvtk.show(ren)
    clearall()
        