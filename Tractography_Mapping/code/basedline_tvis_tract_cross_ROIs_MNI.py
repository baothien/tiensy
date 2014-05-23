# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:30:13 2014

@author: bao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:34:06 2014

@author: bao
"""

import numpy as np
from intersect_roi import *
from common_functions import *

def clearall():
    all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall" and var!="i" and var!="source_ids" and var!="t_id" and var!="target_ids"  and var!="np"]
    for var in all:
        del globals()[var]

ROIs_native_voxel_L_LPS = {'101': (np.array([67.5,67.5,19.5], dtype=np.float32), np.array([75.5,67.5,39.5], dtype=np.float32)),
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
   
ROIs_native_voxel_R_LPS = {'101': (np.array([61.5,66.5,20.5], dtype=np.float32), np.array([55.5,66.5,40.5], dtype=np.float32)),
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
   
ROIs_native_voxel_L_LAS = {
                  '201': (np.array([69.5,61.5,20.5], dtype=np.float32), np.array([74.5,61.5,38.5], dtype=np.float32)), 
                  '202': (np.array([69.5,63.5,22.5], dtype=np.float32), np.array([74.5,62.5,38.5], dtype=np.float32)), 
                  '203': (np.array([68.5,62.5,19.5], dtype=np.float32), np.array([75.5,62.5,38.5], dtype=np.float32)), 
                  '204': (np.array([67.5,59.5,13.5], dtype=np.float32), np.array([75.5,61.5,35.5], dtype=np.float32)), 
                  '205': (np.array([68.5,66.5,16.5], dtype=np.float32), np.array([75.5,65.5,38.5], dtype=np.float32)), 
                  '206': (np.array([68.5,60.5,20.5], dtype=np.float32), np.array([75.5,58.5,39.5], dtype=np.float32)), 
                  '207': (np.array([67.5,59.5,15.5], dtype=np.float32), np.array([75.5,63.5,36.5], dtype=np.float32)), 
                  '208': (np.array([69.5,61.5,21.5], dtype=np.float32), np.array([75.5,61.5,39.5], dtype=np.float32)), 
                  '209': (np.array([69.5,59.5,18.5], dtype=np.float32), np.array([75.5,57.5,38.5], dtype=np.float32)), 
                  '210': (np.array([69.5,63.5,20.5], dtype=np.float32), np.array([76.5,62.5,39.5], dtype=np.float32)), 
                  '212': (np.array([67.5,62.5,16.5], dtype=np.float32), np.array([77.5,61.5,38.5], dtype=np.float32)), 
                  '213': (np.array([66.5,58.5,18.5], dtype=np.float32), np.array([73.5,57.5,38.5], dtype=np.float32)),   
		 }
 
ROIs_native_voxel_R_LAS = {  
                  '201': (np.array([59.5,61.5,20.5], dtype=np.float32), np.array([53.5,61.5,38.5], dtype=np.float32)), 
                  '202': (np.array([61.5,63.5,22.5], dtype=np.float32), np.array([53.5,62.5,38.5], dtype=np.float32)), 
                  '203': (np.array([59.5,62.5,19.5], dtype=np.float32), np.array([53.5,63.5,37.5], dtype=np.float32)), 
                  '204': (np.array([62.5,59.5,13.5], dtype=np.float32), np.array([54.5,59.5,35.5], dtype=np.float32)), 
                  '205': (np.array([63.5,66.5,16.5], dtype=np.float32), np.array([54.5,67.5,38.5], dtype=np.float32)), 
                  '206': (np.array([60.5,60.5,20.5], dtype=np.float32), np.array([54.5,59.5,39.5], dtype=np.float32)), 
                  '207': (np.array([61.5,59.5,15.5], dtype=np.float32), np.array([53.5,60.5,34.5], dtype=np.float32)), 
                  '208': (np.array([60.5,62.5,21.5], dtype=np.float32), np.array([53.5,61.5,39.5], dtype=np.float32)), 
                  '209': (np.array([61.5,60.5,18.5], dtype=np.float32), np.array([53.5,57.5,38.5], dtype=np.float32)), 
                  '210': (np.array([63.5,63.5,20.5], dtype=np.float32), np.array([54.5,60.5,39.5], dtype=np.float32)), 
                  '212': (np.array([61.5,62.5,16.5], dtype=np.float32), np.array([54.5,62.5,38.5], dtype=np.float32)), 
                  '213': (np.array([59.5,57.5,18.5], dtype=np.float32), np.array([53.5,57.5,38.5], dtype=np.float32)),   
		 }
   
ROIs_MNI_mm_R_LPS = { '201': (np.array([8.62874 , -9.41393 , -24.2606], dtype=np.float32), np.array([19.0414,  -3.57136  ,14.6015], dtype=np.float32)), 
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



   
ROIs_MNI_voxel_R_LPS = { '201': (np.array([81.3713,  116.586  ,47.7394], dtype=np.float32), np.array([70.9586,  122.429  ,86.6015], dtype=np.float32)), 
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

ROIs_MNI_mm_R_LAS = { '201': (np.array([9.5151,  -21.6881,  -23.5397], dtype=np.float32), np.array([19.9277,  -15.8455,  15.3224], dtype=np.float32)), 
                  '202': (np.array([7.09124,  -18.0495,  -17.5348], dtype=np.float32), np.array([22.0211,  -12.2146 , 15.3124], dtype=np.float32)), 
                  '203': (np.array([10.6311,  -19.9636,  -21.5577], dtype=np.float32), np.array([22.5732,  -12.8569 , 13.6323], dtype=np.float32)), 
                  '204': (np.array([5.46255,  -22.4484,  -29.327], dtype=np.float32), np.array([20.3477 , -15.3237  ,15.6017], dtype=np.float32)), 
                  '205': (np.array([6.41053,  -21.1963,  -28.1187], dtype=np.float32), np.array([22.8845,  -10.051  ,15.2646], dtype=np.float32)), 
                  '206': (np.array([7.06925,  -20.502 , -23.4381], dtype=np.float32), np.array([20.0706 , -12.8911  ,14.3022], dtype=np.float32)), 
                  '207': (np.array([4.80061,  -23.1131,  -24.661], dtype=np.float32), np.array([21.8621 , -12.7848  ,11.4995], dtype=np.float32)), 
                  '208': (np.array([8.95311,  -19.1617,  -19.6871], dtype=np.float32), np.array([20.2606,  -13.0605 , 16.0102], dtype=np.float32)), 
                  '209': (np.array([6.69344,  -18.9436,  -23.4522], dtype=np.float32), np.array([22.3157,  -16.0562 ,18.32], dtype=np.float32)), 
                  '210': (np.array([3.79915,  -21.5296,  -22.955], dtype=np.float32), np.array([20.8427 , -17.4947  ,15.7456], dtype=np.float32)), 
                  '212': (np.array([5.53725,  -21.2825,  -30.5078], dtype=np.float32), np.array([21.6416,  -11.4623 , 12.6537], dtype=np.float32)), 
                  '213': (np.array([8.0348 , -23.115 , -24.9937], dtype=np.float32), np.array([20.0517,  -15.0624  ,15.4317], dtype=np.float32))
                  
		 }
   
ROIs_MNI_voxel_R_LAS = { '201': (np.array([80.4849 , 104.312,  48.4603], dtype=np.float32), np.array([70.0723,	110.154,	87.3224], dtype=np.float32)), 
                  '202': (np.array([82.9088,  107.951,  54.4652], dtype=np.float32), np.array([67.9789,  113.785 , 87.3124], dtype=np.float32)), 
                  '203': (np.array([79.3689,  106.036,  50.4423], dtype=np.float32), np.array([67.4268,  113.143 , 85.6323], dtype=np.float32)), 
                  '204': (np.array([84.5375,  103.552,  42.673], dtype=np.float32), np.array([69.6523 , 110.676  ,87.6017], dtype=np.float32)), 
                  '205': (np.array([83.5895,  104.804,  43.8813], dtype=np.float32), np.array([67.1155,  115.949 , 87.2646], dtype=np.float32)), 
                  '206': (np.array([82.9308,  105.498,  48.5619], dtype=np.float32), np.array([69.9294,  113.109 , 86.3022], dtype=np.float32)), 
                  '207': (np.array([85.1994,  102.887,  47.339], dtype=np.float32), np.array([68.1379 , 113.215  ,83.4995], dtype=np.float32)), 
                  '208': (np.array([81.0469,  106.838,  52.3129], dtype=np.float32), np.array([69.7394,  112.939 , 88.0102], dtype=np.float32)), 
                  '209': (np.array([83.3066,  107.056,  48.5478], dtype=np.float32), np.array([67.6843,  109.944 , 90.32], dtype=np.float32)), 
                  '210': (np.array([86.2009,  104.47 , 49.045], dtype=np.float32), np.array([69.1573  ,108.505  ,87.7456], dtype=np.float32)), 
                  '212': (np.array([84.4627,  104.718 , 41.4922], dtype=np.float32), np.array([68.3584 , 114.538,  84.6537], dtype=np.float32)), 
                  '213': (np.array([81.9652,  102.885,  47.0063], dtype=np.float32), np.array([69.9483 , 110.938,  87.4317], dtype=np.float32))
                  
		 }
   
ROIs_MNI_mm_L_LAS = { '201': (np.array([-10.3412 , -21.6062,  -24.2413], dtype=np.float32), np.array([-21.7706,  -15.6736 , 13.849], dtype=np.float32)), 
                  '202': (np.array([-8.78028,  -17.7085,  -18.0784], dtype=np.float32), np.array([-19.6417,  -11.3197,  13.8856], dtype=np.float32)), 
                  '203': (np.array([-6.96279,  -19.7711,  -21.4505], dtype=np.float32), np.array([-20.4235,  -13.9604,  16.047], dtype=np.float32)), 
                  '204': (np.array([-4.75505,  -22.2761,  -29.5815], dtype=np.float32), np.array([-22.4882,  -10.7623,  13.9448], dtype=np.float32)), 
                  '205': (np.array([-3.91027,  -20.6348,  -28.7087], dtype=np.float32), np.array([-20.7184,  -11.2782,  13.2737], dtype=np.float32)), 
                  '206': (np.array([-8.81087,  -21.5911,  -22.662], dtype=np.float32), np.array([-21.474,  -17.5948  ,16.7414], dtype=np.float32)), 
                  '207': (np.array([-6.88173,  -23.6724,  -24.0839], dtype=np.float32), np.array([-21.244,  -8.57914 , 16.5004], dtype=np.float32)), 
                  '208': (np.array([-8.23852,  -20.7518,  -20.1414], dtype=np.float32), np.array([-21.6605,  -12.4482,  14.2637], dtype=np.float32)), 
                  '209': (np.array([-9.87361,  -20.4807,  -23.3255], dtype=np.float32), np.array([-23.1675,  -15.0294,  17.6905], dtype=np.float32)), 
                  '210': (np.array([-8.12573,  -21.4152,  -23.1788], dtype=np.float32), np.array([-22.8169,  -13.1437,  14.1574], dtype=np.float32)), 
                  '212': (np.array([-6.37431,  -20.9995,  -29.9575], dtype=np.float32), np.array([-24.0466,  -12.3513,  15.1388], dtype=np.float32)), 
                  '213': (np.array([-5.9726,  -21.2182 , -25.2633], dtype=np.float32), np.array([-19.9647,  -15.0644 , 15.4683], dtype=np.float32))
                  
		 }
   
ROIs_MNI_voxel_L_LAS = { '201': (np.array([100.341 , 104.394 , 47.7587], dtype=np.float32), np.array([111.771,  110.326 , 85.849], dtype=np.float32)), 
                  '202': (np.array([98.7803,  108.291,  53.9216], dtype=np.float32), np.array([109.642,  114.68 , 85.8856], dtype=np.float32)), 
                  '203': (np.array([96.9628,  106.229,  50.5495], dtype=np.float32), np.array([110.424,  112.04 , 88.047], dtype=np.float32)), 
                  '204': (np.array([94.755 , 103.724 , 42.4185], dtype=np.float32), np.array([112.488 , 115.238 , 85.9448], dtype=np.float32)), 
                  '205': (np.array([93.9103,  105.365,  43.2913], dtype=np.float32), np.array([110.718,  114.722,  85.2737], dtype=np.float32)), 
                  '206': (np.array([98.8109,  104.409,  49.338], dtype=np.float32), np.array([111.474 , 108.405 , 88.7414], dtype=np.float32)), 
                  '207': (np.array([96.8817,  102.328,  47.9161], dtype=np.float32), np.array([111.244 , 117.421,  88.5004], dtype=np.float32)), 
                  '208': (np.array([98.2385,  105.248,  51.8586], dtype=np.float32), np.array([111.66  ,113.552 , 86.2637], dtype=np.float32)), 
                  '209': (np.array([99.8736,  105.519,  48.6745], dtype=np.float32), np.array([113.168 , 110.971,  89.6905], dtype=np.float32)), 
                  '210': (np.array([98.1257,  104.585,  48.8212], dtype=np.float32), np.array([112.817 , 112.856,  86.1574], dtype=np.float32)), 
                  '212': (np.array([96.3743,  105.001, 42.0425], dtype=np.float32), np.array([114.047  ,113.649 , 87.1388], dtype=np.float32)), 
                  '213': (np.array([95.9726,  104.782 ,46.7367], dtype=np.float32), np.array([109.965  ,110.936 , 87.4683], dtype=np.float32))
                  
		 }


ROIs_subject = ROIs_MNI_voxel_R_LAS
Rs = [4.,4.]#np.array([2.,2.],dtype=np.float32)
 
#Left 
#source_ids = [212, 202, 204, 209]
#target_ids = [212, 202, 204, 209]

#Right
source_ids = [205, 212, 204, 206]
target_ids = [205, 212, 204, 206]
 
visualize = True#False#True# False
   
print "Check the tractography of source crossing the ROIs of target"
print " when both tractography and ROIs are in MNI space"     
for i in np.arange(len(source_ids)):
    print '----------------------------------------------------------------------------------'
    print 'Source: ', source_ids[i]
    source = str(source_ids[i])    
    for j in np.arange(len(target_ids)):
        if target_ids[j]!=source_ids[i]:
            
            target = str(target_ids[j]) 
            print '\t\t target', target
            #MNI space trackvis tractography
            #tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/ROI_seg_tvis/' + source + '_corticospinal_L_tvis.pkl'
            #tract_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.dpy'        
            #tracks = load_tract(tract_file,tracks_ind_file)
            
            tract_file = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + source + '_tracks_dti_tvis_linear.dpy'        
            tracks = load_whole_tract(tract_file)
            
            
            from intersect_roi import *
            ROIs = ROIs_subject[target]
             
            common = intersec_ROIs(tracks, ROIs, Rs, vis = True)
            
            print "\t Total ", len(tracks), " and  the number of fibers cross the ROIs ", len(common)
                
            tracks_ind_file  = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/baseline_anatomy_MNI/' + source + '_tractography_cross_'+ target+'_ROIs_R_tvis_MNI.pkl'
            save_pickle(tracks_ind_file, common)
            print "Saved file", tracks_ind_file
            
            #print "Done evaluate using ROIs"
            tracks_cross = [tracks[k] for k in common]
            if (visualize==True):
                ren = fvtk.ren()
                #ren = visualize_tract(ren, tracks, fvtk.yellow)
                ren = visualize_tract(ren, tracks_cross, fvtk.yellow)
                fvtk.add(ren, fvtk.sphere(ROIs[0],Rs[0],color = fvtk.red, opacity=1.0)) 
                fvtk.add(ren, fvtk.sphere(ROIs[1],Rs[1],color = fvtk.blue, opacity=1.0)) 
                fvtk.show(ren)
            #clearall()  