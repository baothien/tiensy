# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:33:57 2014

@author: bao

Check the overlap between ROIs defined by Nivedita of CST

"""
from common_functions import *
from intersect_roi import *
from dipy.viz import fvtk
import argparse


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
                  '207': (np.array([61.5,68.5,15.5], dtype=np.float32), np.array([53.5,67.5,35.5], dtype=np.float32)), 
                  '208': (np.array([60.5,65.5,21.5], dtype=np.float32), np.array([53.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([61.5,67.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([63.5,64.5,20.5], dtype=np.float32), np.array([54.5,67.5,39.5], dtype=np.float32)), 
                  '212': (np.array([61.5,65.5,16.5], dtype=np.float32), np.array([54.5,65.5,38.5], dtype=np.float32)), 
                  '213': (np.array([59.5,70.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)),   
		 }


ROIs_MNI_mm_L = { }
		 

ROIs_MNI_mm_R = { '201': (np.array([68.9658,  -107.168,  -88.9914], dtype=np.float32), np.array([75.5253,  -103.849,  -67.7217], dtype=np.float32)), 
                  '202': (np.array([70.7157,  -107.595,  -79.3391], dtype=np.float32), np.array([78.9463,  -102.219,  -61.8877], dtype=np.float32)), 
                  '203': (np.array([72.7948,  -96.4262,  -78.0331], dtype=np.float32), np.array([79.6056,  -94.7181,  -57.8735], dtype=np.float32)), 
                  '204': (np.array([74.4013,  -94.7595,  -86.0524], dtype=np.float32), np.array([82.4992,  -89.5193,  -62.8367], dtype=np.float32)), 
                  '205': (np.array([64.6355,  -112.075,  -83.7875], dtype=np.float32), np.array([73.3966,  -108.585,  -59.082], dtype=np.float32)), 
                  '206': (np.array([70.7387,  -91.3969,  -94.1828], dtype=np.float32), np.array([77.9126,  -86.0224,  -72.7127], dtype=np.float32)),
                  '207': (np.array([63.5117,  -90.849 , -84.7916], dtype=np.float32), np.array([73.0966 , -88.8838 , -63.5763], dtype=np.float32)), 
                  '208': (np.array([67.584 , -108.624 , -76.0948], dtype=np.float32), np.array([74.4819 , -102.788 , -57.5295], dtype=np.float32)), 
                  '209': (np.array([75.7967,  -94.8282,  -93.0827], dtype=np.float32), np.array([83.7935,  -88.5361,  -70.935], dtype=np.float32)), 
                  '210': (np.array([71.5205,  -112.892,  -86.6334], dtype=np.float32), np.array([80.3851,  -105.708,  -65.9964], dtype=np.float32)), 
                  '212': (np.array([61.3057,  -122.257,  -70.2849], dtype=np.float32), np.array([70.2135,  -114.291,  -47.4686], dtype=np.float32)),
                  '213': (np.array([71.3302,  -99.5765,  -89.3854], dtype=np.float32), np.array([78.9528,  -94.3791,  -66.637], dtype=np.float32))                                    
		 }
   
ROIs_MNI_voxel_R = { '201': (np.array([21.0342,  18.832 , -16.9914], dtype=np.float32), np.array([14.4747,  22.151 , 4.2783], dtype=np.float32)), 
                    '202': (np.array([19.2843 , 18.405  ,-7.3391], dtype=np.float32), np.array([11.0537  ,23.781  ,10.1123], dtype=np.float32)), 
                    '203': (np.array([17.2052 , 29.5738 , -6.0331], dtype=np.float32), np.array([10.3944  ,31.2819,  14.1265], dtype=np.float32)), 
                    '204': (np.array([15.5987 , 31.2405 , -14.0524], dtype=np.float32), np.array([7.5008  ,36.4807,  9.1633], dtype=np.float32)), 
                    '205': (np.array([25.3645 , 13.925  ,-11.7875], dtype=np.float32), np.array([16.6034  ,17.415 , 12.918], dtype=np.float32)), 
                    '206': (np.array([19.2613 , 34.6031 , -22.1828], dtype=np.float32), np.array([12.0874 , 39.9776,  -0.7127], dtype=np.float32)), 
                    '207': (np.array([26.4883 , 35.151  ,-12.7916], dtype=np.float32), np.array([16.9034  ,37.1162 , 8.4237], dtype=np.float32)), 
                    '208': (np.array([22.416  ,17.376  ,-4.0948], dtype=np.float32), np.array([15.5181  ,23.212  ,14.4705], dtype=np.float32)), 
                    '209': (np.array([14.2033 ,31.1718 , -21.0827], dtype=np.float32), np.array([6.2065 , 37.4639,  1.065], dtype=np.float32)), 
                    '210': (np.array([18.4795 ,13.108  ,-14.6334], dtype=np.float32), np.array([9.6149  ,20.292 , 6.0036], dtype=np.float32)), 
                    '212': (np.array([28.6943 , 3.743  ,1.7151], dtype=np.float32), np.array([19.7865  ,11.709 , 24.5314], dtype=np.float32)), 
                    '213': (np.array([18.6698 , 26.4235 , -17.3854], dtype=np.float32), np.array([11.0472,  31.6209,  5.363], dtype=np.float32)), 
                    }

def nativevoxel2MNImm_all(ROIs_sub):
    
    ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212, 213]           
    
    for i in np.arange(len(ids)):
        sub = str(ids[i])
        anatomy = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/MP_Rage_1x1x1_ND_3/anatomy.nii.gz'
        flirt_T1_mat = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/MP_Rage_1x1x1_ND_3/flirt_T1.mat'
        ROIs = ROIs_sub[sub]        
        #echo "156 111 145" | img2stdcoord -img /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz -std $FSLDIR/data/standard/MNI152_T1_1mm -xfm /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox
        native2MNI_mm(ROIs[0]+0.5, anatomy, flirt_T1_mat)
        native2MNI_mm(ROIs[1]+0.5, anatomy, flirt_T1_mat)        
        

def MNImm2MNIvoxel_all(ROIs_sub):
    
    ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212, 213]            
    
    for i in np.arange(len(ids)):
        sub = str(ids[i])        
        ROIs = ROIs_sub[sub]   
        #print ROIs
        MNImm2MNIvoxel(ROIs[0])
        MNImm2MNIvoxel(ROIs[1])
           


#nativevoxel2MNImm_all(ROIs_native_voxel_R)
MNImm2MNIvoxel_all(ROIs_MNI_mm_R)

stop

ROIs_subject = ROIs_subject_R
Rs = [5.,5.]#np.array([2.,2.],dtype=np.float32)

source_ids =[202,203]# [201]#, 202, 203]#, 204, 205, 206, 207, 208, 209, 210, 212]

target_ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212]
            
vis = True
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    print 'Source: ', source_ids[s_id]
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #Native space            
            print 
            print 'Target: ', target            

           
               
            if (vis==True):
                ren1 = fvtk.ren()       
                fvtk.add(ren1, fvtk.sphere(ROIs[0],Rs[0],color = fvtk.red, opacity=1.0)) 
                fvtk.add(ren1, fvtk.sphere(ROIs[1],Rs[1],color = fvtk.blue, opacity=1.0)) 
                fvtk.show(ren1)
    
