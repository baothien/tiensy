# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 18:48:49 2014

@author: bao
"""

import numpy as np
from dipy.viz import fvtk
from common_functions import load_whole_tract, load_whole_tract_trk, load_tract, load_tract_trk, visualize_tract
from dipy.io.pickles import load_pickle, save_pickle

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="s_id" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]

ROIs_native_voxel_L = {'101': (np.array([67.5,67.5,19.5], dtype=np.float32), np.array([75.5,67.5,39.5], dtype=np.float32)),
                  '102': (np.array([68.5,60.5,17.5], dtype=np.float32), np.array([75.5,58.5,39.5], dtype=np.float32)),
	            '103': (np.array([68.5,62.5,19.5], dtype=np.float32), np.array([75.5,61.5,37.5], dtype=np.float32)),                  
                  '104': (np.array([69.5,63.5,19.5], dtype=np.float32), np.array([74.5,61.5,36.5], dtype=np.float32)),
                  '105': (np.array([67.5,65.5,13.5], dtype=np.float32), np.array([73.5,64.5,36.5], dtype=np.float32)), 
                  '106': (np.array([70.5,63.5,19.5], dtype=np.float32), np.array([76.5,62.5,38.5], dtype=np.float32)), 
                  '107': (np.array([67.5,69.5,15.5], dtype=np.float32), np.array([76.5,69.5,35.5], dtype=np.float32)), 
                  '109': (np.array([67.5,64.5,15.5], dtype=np.float32), np.array([75.5,62.5,37.5], dtype=np.float32)),  
                  '111': (np.array([67.5,65.5,16.5], dtype=np.float32), np.array([76.5,70.5,40.5], dtype=np.float32)),   
                  '112': (np.array([69.5,65.5,20.5], dtype=np.float32), np.array([74.5,65.5,35.5], dtype=np.float32)), 
                  '113': (np.array([69.5,57.5,23.5], dtype=np.float32), np.array([79.5,60.5,40.5], dtype=np.float32)),
                  '201': (np.array([69.5,66.5,20.5], dtype=np.float32), np.array([74.5,66.5,38.5], dtype=np.float32)), 
                  '202': (np.array([69.5,64.5,22.5], dtype=np.float32), np.array([74.5,65.5,38.5], dtype=np.float32)), 
                  '203': (np.array([68.5,65.5,19.5], dtype=np.float32), np.array([75.5,65.5,38.5], dtype=np.float32)), 
                  '204': (np.array([67.5,68.5,13.5], dtype=np.float32), np.array([75.5,66.5,35.5], dtype=np.float32)), 
                  '205': (np.array([68.5,61.5,16.5], dtype=np.float32), np.array([75.5,62.5,38.5], dtype=np.float32)), 
                  '206': (np.array([68.5,67.5,20.5], dtype=np.float32), np.array([75.5,69.5,39.5], dtype=np.float32)), 
                  '207': (np.array([67.5,68.5,15.5], dtype=np.float32), np.array([75.5,64.5,36.5], dtype=np.float32)), 
                  '208': (np.array([69.5,66.5,21.5], dtype=np.float32), np.array([75.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([69.5,68.5,18.5], dtype=np.float32), np.array([75.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([69.5,64.5,20.5], dtype=np.float32), np.array([76.5,65.5,39.5], dtype=np.float32)), 
                  '212': (np.array([67.5,65.5,16.5], dtype=np.float32), np.array([77.5,66.5,38.5], dtype=np.float32)), 
                  '213': (np.array([66.5,69.5,18.5], dtype=np.float32), np.array([73.5,70.5,38.5], dtype=np.float32)),   
		 }
   
ROIs_native_voxel_R = {'101': (np.array([61.5,66.5,20.5], dtype=np.float32), np.array([55.5,66.5,40.5], dtype=np.float32)),
                  '102': (np.array([61.5,59.5,17.5], dtype=np.float32), np.array([53.5,59.5,40.5], dtype=np.float32)),
	            '103': (np.array([62.5,62.5,20.5], dtype=np.float32), np.array([54.5,61.5,39.5], dtype=np.float32)),                  
                  '104': (np.array([59.5,64.5,19.5], dtype=np.float32), np.array([54.5,63.5,36.5], dtype=np.float32)),
                  '105': (np.array([62.5,66.5,12.5], dtype=np.float32), np.array([55.5,66.5,35.5], dtype=np.float32)), 
                  '106': (np.array([63.5,62.5,19.5], dtype=np.float32), np.array([53.5,63.5,40.5], dtype=np.float32)), 
                  '107': (np.array([61.5,67.5,15.5], dtype=np.float32), np.array([54.5,69.5,35.5], dtype=np.float32)), 
                  '109': (np.array([61.5,64.5,12.5], dtype=np.float32), np.array([53.5,62.5,37.5], dtype=np.float32)),  
                  '111': (np.array([61.5,65.5,15.5], dtype=np.float32), np.array([54.5,70.5,38.5], dtype=np.float32)),   
                  '112': (np.array([59.5,65.5,22.5], dtype=np.float32), np.array([54.5,64.5,34.5], dtype=np.float32)), 
                  '113': (np.array([61.5,56.5,23.5], dtype=np.float32), np.array([53.5,58.5,41.5], dtype=np.float32)),
                  '201': (np.array([59.5,66.5,20.5], dtype=np.float32), np.array([53.5,66.5,38.5], dtype=np.float32)), 
                  '202': (np.array([61.5,64.5,22.5], dtype=np.float32), np.array([53.5,65.5,38.5], dtype=np.float32)), 
                  '203': (np.array([59.5,65.5,19.5], dtype=np.float32), np.array([53.5,64.5,37.5], dtype=np.float32)), 
                  '204': (np.array([62.5,68.5,13.5], dtype=np.float32), np.array([54.5,68.5,35.5], dtype=np.float32)), 
                  '205': (np.array([63.5,61.5,16.5], dtype=np.float32), np.array([54.5,60.5,38.5], dtype=np.float32)), 
                  '206': (np.array([60.5,67.5,20.5], dtype=np.float32), np.array([54.5,68.5,39.5], dtype=np.float32)), 
                  '207': (np.array([61.5,68.5,15.5], dtype=np.float32), np.array([53.5,67.5,34.5], dtype=np.float32)), 
                  '208': (np.array([60.5,65.5,21.5], dtype=np.float32), np.array([53.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([61.5,67.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([63.5,64.5,20.5], dtype=np.float32), np.array([54.5,67.5,39.5], dtype=np.float32)), 
                  '212': (np.array([61.5,65.5,16.5], dtype=np.float32), np.array([54.5,65.5,38.5], dtype=np.float32)), 
                  '213': (np.array([59.5,70.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)),   
		 }


ROIs_MNI_mm_L = { }
		 

ROIs_MNI_mm_R = { '201': (np.array([8.62874 , -9.41393 , -24.2606], dtype=np.float32), np.array([19.0414,  -3.57136  ,14.6015], dtype=np.float32)), 
                  '202': (np.array([6.1493  ,-14.9657  ,-17.0077], dtype=np.float32), np.array([21.1829 , -5.38302  ,15.2539], dtype=np.float32)), 
                  '203': (np.array([9.66368 , -13.2517 , -21.209], dtype=np.float32), np.array([21.6034 , -9.8891  ,14.3478], dtype=np.float32)), 
                  '204': (np.array([4.77806 , -4.03347 , -31.1325], dtype=np.float32), np.array([19.6632,  3.09118,  13.7962], dtype=np.float32)), 
                  '205': (np.array([4.75283,	-28.9714	,-26.1153], dtype=np.float32), np.array([20.9713,  -21.4114,  17.7547], dtype=np.float32)), 
                  '206': (np.array([5.04659,	-6.50777	,-25.4048], dtype=np.float32), np.array([17.7666,  4.79244 , 11.532], dtype=np.float32)),
                  '207': (np.array([2.468,	-5.73293	,-26.9222], dtype=np.float32), np.array([19.8247,  0.969612,  9.93099], dtype=np.float32)), 
                  '208': (np.array([8.09244,	-12.48	,-19.6686], dtype=np.float32), np.array([19.4842,  -2.69769,  15.5083], dtype=np.float32)),
                  '209': (np.array([5.84651,	-4.36627	,-25.1219], dtype=np.float32), np.array([21.6349,  9.98443 , 14.5165], dtype=np.float32)), 
                  '210': (np.array([2.83449,	-18.3056	,-22.5702], dtype=np.float32), np.array([20.0729,  -2.4762 , 13.8269], dtype=np.float32)),
                  '212': (np.array([4.69006,	-14.1204	,-30.7804], dtype=np.float32), np.array([20.7944,  -4.30026,  12.3812], dtype=np.float32)), 
                  '213': (np.array([7.01227,	2.70327	,-27.7935], dtype=np.float32), np.array([19.0292,  10.7559 , 12.6319], dtype=np.float32))                  
		 }
   
ROIs_MNI_voxel_R = { '201': (np.array([81.3713,  116.586  ,47.7394], dtype=np.float32), np.array([70.9586,  122.429  ,86.6015], dtype=np.float32)), 
                    '202': (np.array([83.8507 , 111.034  ,54.9923], dtype=np.float32), np.array([68.8171 , 120.617  ,87.2539], dtype=np.float32)), 
                    '203': (np.array([80.3363 , 112.748 , 50.791], dtype=np.float32), np.array([68.3966  ,116.111  ,86.3478], dtype=np.float32)), 
                    '204': (np.array([85.2219 , 121.967,  40.8675], dtype=np.float32), np.array([70.3368 , 129.091,  85.7962], dtype=np.float32)), 
                    '205': (np.array([85.2472 , 97.0286 , 45.8847], dtype=np.float32), np.array([69.0287,  104.589 , 89.7547], dtype=np.float32)), 
                    '206': (np.array([84.9534 , 119.492 , 46.5952], dtype=np.float32), np.array([72.2334,  130.792 , 83.532], dtype=np.float32)),
                    '207': (np.array([87.532  ,120.267  ,45.0778], dtype=np.float32), np.array([70.1753,  126.97 , 81.931], dtype=np.float32)), 
                    '208': (np.array([81.9076 , 113.52  ,52.3314], dtype=np.float32), np.array([70.5158 , 123.302  ,87.5083], dtype=np.float32)),
                    '209': (np.array([84.1535 , 121.634 , 46.8781], dtype=np.float32), np.array([68.3651,  135.984 , 86.5165], dtype=np.float32)), 
                    '210': (np.array([87.1655 , 107.694 , 49.4298], dtype=np.float32), np.array([69.9271,  123.524 , 85.8269], dtype=np.float32)),
                    '212': (np.array([85.3099 , 111.88  ,41.2196], dtype=np.float32), np.array([69.2056 , 121.7  ,84.3812], dtype=np.float32)), 
                    '213': (np.array([82.9877 , 128.703 , 44.2065], dtype=np.float32), np.array([70.9708,  136.756,  84.6319], dtype=np.float32))      
                    }

ROIs_subject = ROIs_native_voxel_R
Rs = [2.,2.]#np.array([2.,2.],dtype=np.float32)
  
source_ids =[202]#,202]#,203, 204, 205, 206, 207, 208, 209, 210, 212,213]
visualize = True# False
        
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    print 'Source: ', source_ids[s_id]
    source = str(source_ids[s_id])
    
    #Native space                    
    tract_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'        
    #tract_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M_saving_from_trackvis.trk'        
    tracks_ind_file = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/Segmentation/ROI/index/'+ source + '_NILAB_bao_cst_right.pkl'
    tracks = load_tract(tract_file,tracks_ind_file)
    
    '''
    tract_file1 = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M_saving_from_trackvis.trk'            
    tracks1 = load_tract_trk(tract_file1,tracks_ind_file)
    ren = fvtk.ren()
    ren = visualize_tract(ren, tracks, fvtk.yellow)
    ren = visualize_tract(ren, tracks1, fvtk.blue)
    '''
    #tract_file = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/Segmentation/ROI/'+ source + '/NILAB_bao_cst_left.trk'
    #tracks = load_whole_tract(tract_file)
    #print len(tracks)

    #tract_file = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.trk'        
    #tracks = load_whole_tract_trk(tract_file)
    
    
    from intersect_roi import *

    ROIs = ROIs_subject[source]
    ROIs = [[r[0],128.-r[1],r[2]] for r in ROIs]
    print ROIs

    common = intersec_ROIs_dpy(tracks, ROIs, Rs, vis = True)
    #common = intersec_ROIs(tracks, ROIs, Rs, vis = True)

    print "\t Total ", len(tracks), " and  the number of fibers cross the ROIs ", len(common)
    
    #save_pickle(tracks_ind_file, common)
    #print "Saved file", tracks_ind_file
    
    segment = [tracks[i] for i  in common]
    
    if (visualize==True):
        ren = fvtk.ren()
        ren = visualize_tract(ren, segment, fvtk.yellow)
        fvtk.add(ren, fvtk.sphere(ROIs[0],Rs[0],color = fvtk.red, opacity=1.0)) 
        fvtk.add(ren, fvtk.sphere(ROIs[1],Rs[1],color = fvtk.blue, opacity=1.0)) 
        fvtk.show(ren)
    #clearall() 
    
    
          