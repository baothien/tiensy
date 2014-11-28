# -*- coding: utf-8 -*-
"""
Created on Mon May 12 18:48:10 2014

@author: bao
compute Shanon Entropy for ranking
"""

from common_functions import load_tract,load_tract_trk, load_whole_tract_trk, Shannon_entropy, visualize_tract
from dipy.viz import fvtk
import numpy as np

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and  var!="source_ids" and var!="s_id" and var!="np"]
    for var in all:
        del globals()[var]


        
    
source_ids =[201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212,213]

visualize = False#True#True#False#True# False
print '---------------------------------------------------------------------------'
print 'Source \t Len \t Entropy '
for s_id in np.arange(len(source_ids)):
    
    source = str(source_ids[s_id])
    
    #tract_file = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/Segmentation/ROI/' + source + '/cst_right.trk'        
    #tract = load_whole_tract_trk(tract_file) 

    
    #tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_R_control/' + source + '_CST_ROI_R_3M.pkl'
    #tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/ROI_seg/CST_ROI_L_control/' + source + '_CST_ROI_L_3M.pkl'
    #tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/BOI_seg/' + source + '_corticospinal_R_3M.pkl'
    #tract_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'        
    #tract = load_tract(tract_file,tracks_ind_file)     
    
    tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'
    tract_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis.trk'        
        
    tract = load_tract_trk(tract_file,tracks_ind_file)     
     
   
    entropy = Shannon_entropy(tract)    
    
    print source, '\t', len(tract), '\t ', entropy 
    
    if (visualize==True):
        ren = fvtk.ren()
        ren = visualize_tract(ren, tract, fvtk.yellow)
        fvtk.show(ren)