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


def nativevoxel2MNImm_all(ROIs_sub):
    
    ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212, 213]           
    
    for i in np.arange(len(ids)):
        sub = str(ids[i])        
        
        fa = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/fa.nii.gz'
        flirt_mat = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/flirt.mat'
        ROIs = ROIs_sub[sub]        
        #echo "156 111 145" | img2stdcoord -img /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz -std $FSLDIR/data/standard/MNI152_T1_1mm -xfm /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox
        #nativevoxel2MNImm(ROIs[0]+0.5, fa, flirt_mat)
        nativevoxel2MNImm(ROIs[1]+0.5, fa, flirt_mat)        
        

def MNImm2MNIvoxel_all(ROIs_sub):
    
    ids = [201, 202, 203, 204,205, 206, 207,208, 209,210, 212, 213]            
    
    for i in np.arange(len(ids)):
        sub = str(ids[i])        
        ROIs = ROIs_sub[sub]   
        #print ROIs
        #MNImm2MNIvoxel(ROIs[0])
        MNImm2MNIvoxel(ROIs[1])
           


#nativevoxel2MNImm_all(ROIs_native_voxel_R)
MNImm2MNIvoxel_all(ROIs_MNI_mm_R)
stop
#mklink()

#ROIs_subject = ROIs_MNI_voxel_R
ROIs_subject = ROIs_native_voxel_R
Rs = np.array([5.,5.],dtype=np.float32)

source_ids =[201, 202 ,203, 204, 205, 206, 207, 208, 209, 210, 212, 213]

target_ids = [201, 202, 203, 204, 205, 206, 207,208, 209,210, 212, 213]
            
vis = False#True
for s_id in np.arange(len(source_ids)):
    print '-----------------------------------------------------------------------------------------'
    source = str(source_ids[s_id])
    s_ROIs = ROIs_subject[source] 
    
    print 'Source: ', source
        
    
    intersec_1 = []
    intersec_2 = []
    dis_1 = []
    dis_2 = []
    for t_id in np.arange(len(target_ids)):
        if target_ids[t_id] != source_ids[s_id]:
           
            target = str(target_ids[t_id])                     
            t_ROIs = ROIs_subject[target]       
            
            #print s_ROIs, t_ROIs
            #d1, sphere_inter_1 = spheres_intersection(s_ROIs[0], Rs[0], t_ROIs[0], Rs[0])
            #d2, sphere_inter_2 = spheres_intersection(s_ROIs[1], Rs[1], t_ROIs[1], Rs[1])
            
            d1, sphere_inter_1 = spheres_intersection(s_ROIs[0]+0.5, Rs[0], t_ROIs[0]+0.5, Rs[0])
            d2, sphere_inter_2 = spheres_intersection(s_ROIs[1]+0.5, Rs[1], t_ROIs[1]+0.5, Rs[1])
            #intersec_1.append(sphere_inter_1)
            #intersec_2.append(sphere_inter_2)
            #dis_1.append(d1)
            #dis_2.append(d2)
            #print d1, sphere_inter_1
            print d2, sphere_inter_2
            
            if (vis==True):
                print 'Target: ', target
                
                #s_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M_linear.dpy'
                s_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_tracks_dti_3M.dpy'
                s_idx = '/home/bao/tiensy/Tractography_Mapping/data/' + source + '_corticospinal_L_3M.pkl'    
                #s_cst = load_tract_trk(s_tracts,s_idx)      
                s_cst = load_tract(s_tracts,s_idx)  
                
                #t_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M_linear.dpy'
                t_tracts = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_tracks_dti_3M.dpy'
                t_idx = '/home/bao/tiensy/Tractography_Mapping/data/' + target + '_corticospinal_L_3M.pkl'            
                #t_cst = load_tract_trk(t_tracts,t_idx)      
                t_cst = load_tract(t_tracts,t_idx) 
                print d1, sphere_inter_1, d2, sphere_inter_2
                ren = fvtk.ren()       
                ren = visualize_tract(ren,s_cst,fvtk.red)
                fvtk.add(ren, fvtk.sphere(s_ROIs[0],Rs[0],color = fvtk.yellow, opacity=1.0))
                fvtk.add(ren, fvtk.sphere(s_ROIs[1],Rs[1],color = fvtk.white, opacity=1.0))
                ren = visualize_tract(ren, t_cst, fvtk.blue)
                fvtk.add(ren, fvtk.sphere(t_ROIs[0],Rs[0],color = fvtk.blue, opacity=1.0)) 
                fvtk.add(ren, fvtk.sphere(t_ROIs[1],Rs[1],color = fvtk.blue, opacity=1.0)) 
                fvtk.show(ren)
        else:
            print '  '
    #print np.array(dis_1,dtype=np.float)
    #print np.array(intersec_1,dtype=np.float)
    #print np.array(dis_2,dtype=np.float)
    #print np.array(intersec_2,dtype=np.float)
    #print dis_1
    #print intersec_1
    #print dis_2
    #print intersec_2
