# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:37:20 2014

@author: bao
Some common functions for working with tractography
"""
import numpy as np
from dipy.io.dpy import Dpy
from dipy.viz import fvtk
from dipy.tracking.metrics import length
from dipy.io.pickles import load_pickle, save_pickle
from dissimilarity_common_20130925 import subset_furthest_first as sff
from dipy.tracking.metrics import downsample
import resource
   
def mklink():
    import os
    ids = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213]
    for i in np.arange(len(ids)):
        sub = str(ids[i])
        arg1 = 'ln -s '
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M_linear.dpy '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M_linear.dpy'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M_linear.trk '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M_linear.trk'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.dpy '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M.dpy'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/' + sub + '/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.trk '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/' + sub + '_tracks_dti_3M.trk'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + sub + '/DTI/dti.trk '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + sub + '_tracks_dti_tvis.trk'
        
        #arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + sub + '/DTI/dti_linear.dpy '
        #arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + sub + '_tracks_dti_tvis_linear.dpy' 

        arg2 = '/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Segmentation_CST_francesca/' + sub + '/DTI/dti_linear.trk '
        arg3 = '/home/bao/tiensy/Tractography_Mapping/data/trackvis_tractography/tvis_tractography/' + sub + '_tracks_dti_tvis_linear.trk' 
        
        full_cmd = arg1 + arg2 + arg3        
        os.system(full_cmd)
 
def dipy_version():
    import dipy
    dipy_ver = dipy.__version__
    from distutils.version import StrictVersion

    dipy_ver = StrictVersion(dipy_ver.split('.dev')[0])
    return dipy_ver
 
def cpu_time():
    	return resource.getrusage(resource.RUSAGE_SELF)[0]
  

def center_streamlines(streamlines):
    """ Move streamlines to the origin

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of 2D ndarrays of shape[-1]==3
    inv_shift : ndarray
        Translation in x,y,z to go back in the initial position

    """
    import numpy as np
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center

def vectorize_streamlines(streamlines, no_pts):
    """ Resample all streamlines to the same number of points
    """
    return [downsample(s, no_pts) for s in streamlines]
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for quantizing the    overlap between two tracts
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''
def concat(tract1, tract2):
    
    res = []
    for i in np.arange(len(tract1)):
        res.append(tract1[i])
    
    for j in np.arange(len(tract2)):
        res.append(tract2[j])
    
    return res
    
def overlap(A1, A2):
    '''
    Return the overlap array between two arrays: A1 and A2
    Note that if in A1, one element appears many times, then these appearance are different
    So, the return is an array not a set.
    '''
    overlap = []
    for i in np.arange(len(A1)):
        if (A1[i] in A2):
            overlap.append(A1[i])
    return overlap
    
def minus(A1, A2):
    '''
    Return the A1 - A2
    Note that if in A1, one element appears many times, then these appearance are different
    So, the return is an array not a set.
    '''
    minus = []
    for i in np.arange(len(A1)):
        if (A1[i] not in A2):
            minus.append(A1[i])
    return minus
 
def spheres_intersection(point1, radius1, point2, radius2):
    '''
    calculate the volume of two spheres' intersection
    http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    '''
    #distance between two center points
    
    import math 
    d = math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]) + (point1[2]-point2[2])*(point1[2]-point2[2]))
    
    if (d>=(radius1 + radius2)):
        return d, 0.

    volume = math.pi * ((radius1 + radius2 - d)*(radius1 + radius2 - d)) * (d*d + 2*d*radius1 - 3*(radius1*radius1) + 2*d*radius2 +6*radius1*radius2 - 3*(radius2*radius2)) / (12*d)
    
    return d, volume
    
def volumn_intersec(tract1, tract2, vol_dims, voxel_size, disp=False):
    from dipy.tracking.vox2track import track_counts
    
    #compute the number of fiber crossing every voxel
    tcv1 = track_counts(tract1,vol_dims, voxel_size, return_elements=False)
    tcv2 = track_counts(tract2,vol_dims, voxel_size, return_elements=False)

    count = 0    
    count1 = 0
    count2 = 0
    for x in np.arange(vol_dims[0]):
        for y in np.arange(vol_dims[1]):
            for z in np.arange(vol_dims[2]):
                if tcv1[x,y,z]>0:
                    count1 = count1 + 1
                    #count1 = count1 + tcv1[x,y,z]
                    
                if tcv2[x,y,z]>0:
                    count2 = count2 + 1
                    #count2 = count2 + tcv2[x,y,z]
                
                if tcv1[x,y,z]>0 and tcv2[x,y,z]>0:
                    count = count + 1
                    #count = count +  min(tcv1[x,y,z],tcv2[x,y,z])
                    
    if disp:
        viz_vol1(tcv1,fvtk.colors.red)
        viz_vol1(tcv2,fvtk.colors.blue)
    return count1, count2, count
       
def streamlines_to_vol(static1, moving1, vol_dims,
                       disp=False, save=False):

    #static_centered, static_shift = center_streamlines(static)
    #moving_centered, moving_shift = center_streamlines(moving)
    
    #spts = np.concatenate(static_centered, axis=0)
    #spts = np.round(spts).astype(np.int)

    #mpts = np.concatenate(moving_centered, axis=0)
    #mpts = np.round(mpts).astype(np.int)
    
    static = vectorize_streamlines(static1, 20)
    moving = vectorize_streamlines(moving1, 20)
    
    spts = np.concatenate(static, axis=0)
    spts = np.round(spts).astype(np.int)

    mpts = np.concatenate(moving, axis=0)
    mpts = np.round(mpts).astype(np.int)
    
    vol = np.zeros(vol_dims)
    for index in spts:
        i, j, k = index
        vol[i, j, k] = 1

    vol1 = np.zeros(vol_dims)
    for index in mpts:
        i, j, k = index
        vol1[i, j, k] = 1
          
    intersec =np.zeros(vol_dims)
    for index in spts:
        i, j, k = index
        if (vol1[i, j, k] == 1):
            intersec[i,j,k] = 1
    if save:
        import nibabel as nib
        nib.save(nib.Nifti1Image(vol.astype('uint16'), np.eye(4)), 'vol.nii.gz')
        nib.save(nib.Nifti1Image(vol1.astype('uint16'), np.eye(4)), 'vol1.nii.gz')
        nib.save(nib.Nifti1Image(intersec.astype('uint16'), np.eye(4)), 'intersec.nii.gz')
    
    
    '''
    if disp:
        viz_vol(vol)
        viz_vol(vol1)        
        viz_vol(intersec)
    '''
    if disp:
        dipy_ver = dipy_version()
        #print dipy_ver        
        from distutils.version import StrictVersion
        minimize_version = StrictVersion('0.7') 
        
        if dipy_ver > minimize_version:
            viz_vol1(vol,fvtk.colors.blue)
            viz_vol1(vol1,fvtk.colors.green)                
            viz_vol1(intersec,fvtk.colors.red)
        else:
            viz_vol1(vol,fvtk.blue)
            viz_vol1(vol1,fvtk.green)                
            viz_vol1(intersec,fvtk.red)
         
    return vol, vol1, intersec
    
    
def Jaccard_vol(volA, volB, intersecAB):
    '''
    calculate the Jaccard index of two volume A, B
    volA, volB is the volume of two set A, B (float value)
    intersecAB is the volume of the intersection between set A and set B
    J(A,B) = intersecAB/(min{volA, volB})    
    '''
    jac = (1.*intersecAB)/(1.*min(volA, volB))
    return jac
    
def BFN_vol(volA, volB, intersecAB):
    '''
    calculate the BFN (Balance False Negative) index of two volume A, B
    volA, volB is the volume of two set A, B (float value)
    intersecAB is the volume of the intersection between set A and set B (float value)
    BFN(A,B) = min{volA - intersecAB, volB - intersecAB}/volB    
    '''
    BFN = (1.*min(volA - intersecAB, volB - intersecAB))/(1.*volB)
    return BFN
 
def real_volumn(vol, vol_dims):
    '''
    calculate the number of voxel that has the value 1 in the vol
    in vol, the value of a voxel is 1 if there is any fiber going through that voxel
    otherwhile the value of that voxel is set to 0
    '''
    [x,y,z] = vol_dims
    count = 0
    for i in np.arange(x):
        for j in np.arange(y):
            for k in np.arange(z):
                if vol[i,j,k]==1:
                    count = count + 1
    
    return count
    
def Jac_BFN(tract1, tract2, vol_dims, disp=False, save=False):
    '''
    calculate the Jaccard and BFN indices of two given tract
    vol_size is the volume dimension of two tracts (two tracts have the same dimension of brain)
    disp: visualize or not
    save: save the volumn of each tract and the volumn of the interesection
    
    '''
    vol1, vol2, intersec = streamlines_to_vol(tract1, tract2, vol_dims, disp, save)
    
    real_vol1 = real_volumn(vol1, vol_dims)
    real_vol2 = real_volumn(vol2, vol_dims)
    real_intersec = real_volumn(intersec, vol_dims)
    
    jac = Jaccard_vol(real_vol1, real_vol2, real_intersec)
    bfn = BFN_vol(real_vol1, real_vol2, real_intersec)
    
    return jac, bfn       
    
def Jaccard_index(set_A, set_B):
    '''
    calculate the Jaccard index of two set A, B
    A,B should be the numppy array
    J(A,B) = card(A & B)/(min{card(A), card(B)})
    '''
    set_A = set(set_A)
    set_B = set(set_B)
    lA = len(set_A)
    lB = len(set_B)
    common =set_A.intersection(set_B)
    lC = len(common)
    #print common
    #print set(set_A)
    #print set(set_B)
    
    return (1.*lC)/(1.*min(lA, lB))

def BFN_index(set_A, set_B, map_A2B):
    '''
    calculate the BNF index of two set A, B based on the mapping from A to B  (balance false negative)
    A, B, and mapp_A2B should be the numppy array
    map_A2B[i] is element in B that A[i] is mapped (not the index)
    BFN(A,B|map_A2B) = min{card(un_used),card(missing)}/card(B)
    un_used = {A[i_a] in A | map_A2B(i_a)] not in B, for all i_a in index set of A }
    missing = {b in B | b not in mapp_A2B}
    
    '''
    un_used = []
    for i_a in np.arange(len(set_A)):
        if map_A2B[i_a] not in set_B:
            un_used.append(set_A[i_a])
            
    missing = []
    for b in set_B:
        if b not in map_A2B:
            missing.append(b)
            
    l_u = len(un_used)
    l_m = len(missing)
    print un_used, l_u
    print missing, l_m
    
    return (1.*min(l_u,l_m))/len(set_B)
    
def BFN_index_idxmap(set_A, set_B, map_A2B):
    '''
    calculate the BNF index of two set A, B based on the mapping from A to B  (balance false negative)
    A, B, and mapp_A2B should be the numppy array
    map_A2B[i] is the index of element in B that A[i] is mapped (not the element of B, just index of that element)
    BFN(A,B|map_A2B) = min{card(un_used),card(missing)}/card(B)
    un_used = {A[i_a] in A | B[map_A2B(i)] not in B, for all i_a in index set of A }
    missing = {B[i_b] in B | i_b not in mapp_A2B, for all i_b in index set of B}
    
    '''
    un_used = []
    for i_a in np.arange(len(set_A)):
        if map_A2B[i_a] not in np.arange(len(set_B)):
            un_used.append(set_A[i_a])
            
    missing = []
    for i_b in np.arange(len(set_B)):
        if i_b not in map_A2B:
            missing.append(set_B[i_b])
            
    l_u = len(un_used)
    l_m = len(missing)
    print un_used, l_u
    print missing, l_m
    
    return (1.*min(l_u,l_m))/len(set_B)
    
    
    
    
    
    

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for converting between native and MNI space
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''

def nativevoxel2MNImm(point, anatomy, flirt_mat ):
    import os
    cmd = 'echo "' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '" | img2stdcoord '
    arg1 = '-img ' + anatomy + ' '#/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz '
    #arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm -xfm '
    arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -xfm '
    arg3 = flirt_mat + ' -vox'#'/home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox'
    full_cmd = cmd + arg1 + arg2 + arg3    
    #os.system('echo "156 111 145" | img2stdcoord -img /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/anatomy.nii.gz -std $FSLDIR/data/standard/MNI152_T1_1mm -xfm /home/bao/Personal/PhD_at_Trento/Research/ALS_Nivedita_Bao/Code/data/109/MP_Rage_1x1x1_ND_3/flirt_T1.mat -vox')
    #print full_cmd    
    #print point    
    os.system(full_cmd)

def MNImm2MNIvoxel(point):
    import os
    #echo "-36.18 -21.52 41.28" | std2imgcoord -img $FSLDIR/data/standard/MNI152_T1_1mm -std $FSLDIR/data/standard/MNI152_T1_1mm -vox -
    cmd = 'echo "' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '" | std2imgcoord '
    #arg1 = '-img $FSLDIR/data/standard/MNI152_T1_1mm '
    #arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm -vox -'  
    arg1 = '-img $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz '
    arg2 = '-std $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -vox -'  
    
    full_cmd = cmd + arg1 + arg2    
    #print point    
    os.system(full_cmd)    


            
            
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for load and save tracts both in .dpy and .trk format
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''
def load_whole_tract(tracks_filename):
    
    from dipy.io.pickles import load_pickle
    if (tracks_filename[-3:]=='dpy'):
        from dipy.io.dpy import Dpy
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        import nibabel as nib
        streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
        all_tracks = np.array([s[0] for s in streams], dtype=np.object)    

    all_tracks = np.array(all_tracks,dtype=np.object)
    return all_tracks
    
def load_tract(tracks_filename, id_file):
    

    from dipy.io.pickles import load_pickle
    if (tracks_filename[-3:]=='dpy'):
        from dipy.io.dpy import Dpy
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        import nibabel as nib
        streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
        all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]
    
    tract = np.array(tract,dtype=np.object)
    return tract
   
def load_whole_tract_trk(tracks_filename):
    '''
    load tract from trackvis format
    '''
    import nibabel as nib
    streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
    all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    return all_tracks
    
def load_tract_trk(tracks_filename, id_file):
    '''
    load tract from trackvis format
    '''
    import nibabel as nib
    streams,hdr=nib.trackvis.read(tracks_filename,points_space='voxel')
    all_tracks = np.array([s[0] for s in streams], dtype=np.object)
    
    from dipy.io.pickles import load_pickle
    tracks_id = load_pickle(id_file)
    tract = [all_tracks[i] for i  in tracks_id]
    
    tract = np.array(tract,dtype=np.object)  
    
    return tract
    
def save_tract_trk(tract, fa_file, fname_out ):
    import nibabel as nib
    fa_img = nib.load(fa_file)
    fa = fa_img.get_data()
    fa[np.isnan(fa)] = 0
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = fa_img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = fa.shape

    """
    Then we need to input the streamlines in the way that Trackvis format expects them.
    """

    tract_trk = ((sl, None, None) for sl in tract)
    
    """
    Save the streamlines.
    """

    nib.trackvis.write(fname_out, tract_trk, hdr, points_space='voxel')

 
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for create and save fiber id of a tract (extension, prototype, ...)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''   
def save_id_tract_plus_sff(tracks_filename, id_file, num_proto, distance, out_fname):
   
    if (tracks_filename[-3:]=='dpy'):
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        all_tracks = load_whole_tract_trk(tracks_filename)
    
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]    
    
    not_tract_fil = []
    id_not_tract_fil = []
    min_len = min(len(i) for i in tract)
    #print 'min_len of cst', min_len
    min_len = min_len*2.2/3#2./3.2# - 20
    for i in np.arange(len(all_tracks)):
        if (i not in tracks_id) and (length(all_tracks[i]) > min_len):
            not_tract_fil.append(all_tracks[i])
            id_not_tract_fil.append(i)
    
    not_tract_fil = np.array(not_tract_fil,dtype=np.object)        
    sff_pro_id = sff(not_tract_fil, num_proto, distance)        
    
    tract_sff_id = []
    for i in tracks_id:
        tract_sff_id.append(i)
        
    for idx in sff_pro_id:        
        tract_sff_id.append(id_not_tract_fil[idx])
        
    #tract_sff_id.append(id_not_tract_fil[i] for i in sff_pro_id)
    print len(tract), len(tract_sff_id)
    save_pickle(out_fname, tract_sff_id)
    return tract_sff_id
   
def save_id_tract_ext(tracks_filename, id_file,  distance, out_fname):
    
    
    if (tracks_filename[-3:]=='dpy'):
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        all_tracks = load_whole_tract_trk(tracks_filename)    
    
    
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]    
    
    not_tract_fil = []
    id_not_tract_fil = []
    min_len = min(len(i) for i in tract)
    #print 'min_len of cst', min_len
    min_len = min_len*2.2/3#2./3.2# - 20
    #min_len = min_len*2./3.2#2./3.2# - 20
    for i in np.arange(len(all_tracks)):
        if (i not in tracks_id) and (length(all_tracks[i]) > min_len):
            not_tract_fil.append(all_tracks[i])
            id_not_tract_fil.append(i)
       
    #k = np.round(len(tract)*2.)#1.2)       
    k = np.round(len(tract)*1.2)       
            
    from dipy.segment.quickbundles import QuickBundles
    
    qb = QuickBundles(tract,200,18)
    
    medoid_tract = qb.centroids[0]
    
    med_nottract_dm =  distance([medoid_tract], not_tract_fil)
    med_tract_dm =  distance([medoid_tract], tract)
    
    tract_rad = med_tract_dm[0][np.argmax(med_tract_dm[0])]
    len_dis = tract_rad * 2.8/2.
   
    #k_indices which close to the medoid
    sort = np.argsort(med_nottract_dm,axis = 1)[0]
    #print sort[:k+1]
    while (k>0 and med_nottract_dm[0][sort[k]]>=len_dis):
        k = k - 1
        
    
    #print k
    #close_indices = np.argsort(cst_dm,axis = 1)[:,0:k][0]
    close_indices = sort[0:k]
    
    #for idx in close_indices:
    #    tract_ext.append(not_tract_fil[idx])          
    #print 'close indices', len(close_indices)
    tract_ext_id = []
    for i in tracks_id:
         tract_ext_id.append(i)
    
    #print 'Before', len(tract_ext_id)
    
    for idx in close_indices:
        tract_ext_id.append(id_not_tract_fil[idx]) 
    #    print idx, id_not_tract_fil[idx]
      
    #print 'After', len(tract_ext_id)
    #tract_ext_id = [i for i in tracks_id]
    #tract_ext_id.append(id_not_tract_fil[i] for i in close_indices)
    
    save_pickle(out_fname, tract_ext_id)
    return tract_ext_id

def save_id_tract_ext1(tracks_filename, id_file,  distance, out_fname, thres_len= 2.2/3., thres_vol = 1.2 , thres_dis = 2.8/2.):
    
    print thres_len, thres_vol, thres_dis
    if (tracks_filename[-3:]=='dpy'):
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        all_tracks = load_whole_tract_trk(tracks_filename)    
    
    
    tracks_id = load_pickle(id_file)
    	
    tract = [all_tracks[i] for i  in tracks_id]    
    
    not_tract_fil = []
    id_not_tract_fil = []
    min_len = min(len(i) for i in tract)
    #print 'min_len of cst', min_len
    min_len = min_len*thres_len
    
    for i in np.arange(len(all_tracks)):
        if (i not in tracks_id) and (length(all_tracks[i]) > min_len):
            not_tract_fil.append(all_tracks[i])
            id_not_tract_fil.append(i)
       
    k = np.round(len(tract) * thres_vol  )     
            
    from dipy.segment.quickbundles import QuickBundles
    
    qb = QuickBundles(tract,200,18)
    
    medoid_tract = qb.centroids[0]
    
    med_nottract_dm =  distance([medoid_tract], not_tract_fil)
    med_tract_dm =  distance([medoid_tract], tract)
    
    tract_rad = med_tract_dm[0][np.argmax(med_tract_dm[0])]
    len_dis = tract_rad * thres_dis# 2.8/2.
   
    #k_indices which close to the medoid
    sort = np.argsort(med_nottract_dm,axis = 1)[0]
    #print sort[:k+1]
    while (k>0 and med_nottract_dm[0][sort[k]]>=len_dis):
        k = k - 1
        
    
    #print k
    close_indices = sort[0:k]
    
    #for idx in close_indices:
    #    tract_ext.append(not_tract_fil[idx])          
    #print 'close indices', len(close_indices)
    tract_ext_id = []
    for i in tracks_id:
         tract_ext_id.append(i)
    
    #print 'Before', len(tract_ext_id)
    
    for idx in close_indices:
        tract_ext_id.append(id_not_tract_fil[idx]) 
    #    print idx, id_not_tract_fil[idx]
      
    #print 'After', len(tract_ext_id)
    #tract_ext_id = [i for i in tracks_id]
    #tract_ext_id.append(id_not_tract_fil[i] for i in close_indices)
    
    save_pickle(out_fname, tract_ext_id)
    return tract_ext_id
  
def save_id_tract_ext_plus_sff(tracks_filename, id_file, num_proto, distance, out_fname_ext_sff, out_fname_ext = 'temp'): 
    tract_ext_id = save_id_tract_ext(tracks_filename,id_file, distance, out_fname_ext)
    return save_id_tract_plus_sff(tracks_filename, out_fname_ext, num_proto,distance, out_fname_ext_sff)


#def save_id_tract_plus_sff_in_ext(tracks_filename, id_file, num_proto, distance,  out_fname_ext , out_fname_sff_in_ext, thres_len= 2.2/3., thres_vol = 1.4 , thres_dis = 3./2.):
def save_id_tract_plus_sff_in_ext(tracks_filename, id_file, num_proto, distance,  out_fname_ext , out_fname_sff_in_ext, thres_len= 2.2/3., thres_vol = 1.4 , thres_dis = 3./2.):
    
    
    tract_ext_id = save_id_tract_ext1(tracks_filename,id_file, distance, out_fname_ext, thres_len, thres_vol , thres_dis)
    
    if (tracks_filename[-3:]=='dpy'):
        dpr_tracks = Dpy(tracks_filename, 'r')
        all_tracks=dpr_tracks.read_tracks()
        dpr_tracks.close()
    else:
        all_tracks = load_whole_tract_trk(tracks_filename)
    
    tracks_id = load_pickle(id_file)
    	
    ext_not_tract_id = []
    ext_not_tract = []
    for idx in tract_ext_id:
        if idx not in tracks_id:
            ext_not_tract.append(all_tracks[idx])
            ext_not_tract_id.append(idx)
        
          
    ext_not_tract = np.array(ext_not_tract,dtype=np.object)        
    sff_pro_id = sff(ext_not_tract, num_proto, distance)        
    
    tract_sff_in_ext_id = []
    for i in tracks_id:
        tract_sff_in_ext_id.append(i)
        
    for k in sff_pro_id:        
        tract_sff_in_ext_id.append(ext_not_tract_id[k])
        
    #tract_sff_id.append(id_not_tract_fil[i] for i in sff_pro_id)
    print len(tracks_id), len(tract_sff_in_ext_id), len(tract_ext_id)
    save_pickle( out_fname_sff_in_ext, tract_sff_in_ext_id)
    return tract_sff_in_ext_id
 
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for warping tractography from native to MNI space
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''   
def transform_tracks(tracks,affine):        
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]
        
def warp_tracks_linearly(flirt_filename,fa_filename, tracks_filename,linear_filename):
    import nibabel as nib
    from dipy.external.fsl import flirt2aff
    
    fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    
    img_fa = nib.load(fa_filename)            

    flirt_affine= np.loadtxt(flirt_filename)    
        
    img_ref =nib.load(fsl_ref)
    
    #create affine matrix from flirt     
    mat=flirt2aff(flirt_affine,img_fa,img_ref)        

    #read tracks    
    tensor_tracks = load_whole_tract(tracks_filename)    
        
    #linear tranform for tractography
    tracks_warped_linear = transform_tracks(tensor_tracks,mat)        

    #save tracks_warped_linear    
    dpr_linear = Dpy(linear_filename, 'w')
    dpr_linear.write_tracks(tracks_warped_linear)
    dpr_linear.close()

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for quantizing the tracts (such as entropy, length, FA ...)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''
def Shannon_entropy(tract):
    '''
    compute the Shannon Entropy of a set of tracks as defined by Lauren
    H(A) = (-1/|A|) * sum{log(1/|A|)* sum[p(f_i|f_j)]} 
    where p(f_i|f_j) = exp(-d(f_i,f_j)*d(f_i,f_j))
    '''
    from dipy.tracking.distances import bundles_distances_mam
    import numpy as np
    dm = bundles_distances_mam(tract, tract)
    
    dm = np.array(dm, dtype =float)
    
    dm2 = dm**2
    
    A = len(tract)
    theta = 10.
    
    pr_all = np.exp((-dm**2)/theta)
    
    pr_i = (1./A) * np.array([sum(pr_all[i]) for i in np.arange(A)])
    
    #sum_all = np.sum(pr_i)
    #print pr_i
    #print sum_all
    #pr_i = (1./sum_all) * pr_i
    #print pr_i
    
    #entropy = (-1./A) * sum([pr_i[i]*np.log(pr_i[i]) for i in np.arange(A)])
    entropy = (-1./A) * sum([np.log(pr_i[i]) for i in np.arange(A)])
    #entropy = (-1.) * sum([pr_i[i]*(np.log2(pr_i[i])) for i in np.arange(A)])
    
    #print dm2
    #print pr_all
    #print pr_i
    
    return entropy
    
def Silhouette_Inertia(tract):
    
    from dissimilarity_common import compute_dissimilarity
    from dipy.tracking.distances import bundles_distances_mam
    from sklearn import metrics
    from sklearn.cluster import MiniBatchKMeans#, KMeans
    #from sklearn.metrics.pairwise import euclidean_distances
    diss = compute_dissimilarity(tract, distance=bundles_distances_mam, prototype_policy='sff', num_prototypes=20)
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=1, batch_size=100,
                              n_init=10, max_no_improvement=10, verbose=0)        
    mbk.fit(diss)
        
    labels = mbk.labels_        
    print labels    
    #labels = np.ones(len(tract))
    sil = metrics.silhouette_score(diss, labels, metric='euclidean')
    return sil, mbk.inertia_

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for visualization: a tract, a volumn, ...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''

#visualize a volumn without color
def viz_vol(vol):
   
    
    ren = fvtk.ren()
    vl = 100*vol
    v = fvtk.volume(vl)
    fvtk.add(ren, v)    
    fvtk.show(ren)
    
#visualize a volumn with a specific color   
def viz_vol1(vol,color):
       
    ren = fvtk.ren()
    ren.SetBackground(255,255,255)
    d1, d2, d3 = vol.shape
    
    point = []
    for i in np.arange(d1):
        for j in np.arange(d2):
            for k in np.arange(d3):    
                if (vol[i, j, k] == 1):
                    point.append([i,j,k])
    pts = fvtk.point(point,color, point_radius=0.85)#0.85)    
    fvtk.add(ren, pts)
    fvtk.show(ren)  


def visualize_tract(ren, tract, color=None):  
    if color == None:
        dipy_ver = dipy_version()
        #print dipy_ver        
        from distutils.version import StrictVersion
        minimize_version = StrictVersion('0.7') 
        
        if dipy_ver > minimize_version:
            color = fvtk.colors.red       
        else:
            color = fvtk.red            

    for i in np.arange(len(tract)):
        fvtk.add(ren, fvtk.line(tract[i], color, opacity=1.0))        
    return ren

def visualize_tract_transparence(ren, tract, color=None, tran = 1.0, lwidth=1.0):  
    if color == None:
        dipy_ver = dipy_version()
        #print dipy_ver        
        from distutils.version import StrictVersion
        minimize_version = StrictVersion('0.7') 
        
        if dipy_ver > minimize_version:
            color = fvtk.colors.red       
        else:
            color = fvtk.red            

    for i in np.arange(len(tract)):
        fvtk.add(ren, fvtk.line(tract[i], color, opacity=tran, linewidth=lwidth))        
    return ren

def visualize_mapped(ren, tract2, mapping, color=None):
    for i in np.arange(len(mapping)):        
        fvtk.add(ren, fvtk.line(tract2[mapping[i]], color, opacity=1.0))     
    return ren
   

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for plotting data
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''   
def plot_smooth(plt, x, y, ori = False):
    
    n = len(x)
    xi = np.linspace(x.min(),x.max(),100*n)
    
    
    from scipy.interpolate import spline        
    yi = spline(x, y, xi) 
    
    if ori:    
        plt.plot(x,y,'o',xi, yi)
    else:
        plt.plot(xi, yi) 
    
        
    '''
    #don't use it 
    from scipy.interpolate import interp1d
    yi = interp1d(x, y, kind='cubic')

    if ori:    
        plt.plot(x,y,'o',xi, yi(xi))
    else:
        plt.plot(xi, yi(xi)) 
    '''
    
    #plt.plot(x,y,'o',xi, yi(xi),'--')
    #plt.scatter(x,y,'o',xi, ynew(xi),'--')
  
def plot_smooth_label(plt, x, y, marker, label, ori = False):
    
    n = len(x)
    xi = np.linspace(x.min(),x.max(),100*n)
    
    
    from scipy.interpolate import spline        
    yi = spline(x, y, xi) 
    
    if ori:    
        plt.plot(x,y,'o', color = 'black')
        
    plt.plot(xi, yi, linestyle = marker, color = 'black', label = label, linewidth=1.85)        
    
        

def smooth(x,beta):
    import numpy
    """ kaiser window smoothing """
    window_len=11
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = numpy.kaiser(window_len,beta)
    y = numpy.convolve(w/w.sum(),s,mode='valid')
    return y[5:len(y)-5] 
     
#Let's test it on a random sequence:   
def test_smooth():
    import numpy
    import pylab
    beta = [2,4,16,32]

    pylab.figure()
    # random data generation
    y = numpy.random.random(100)*100 
    for i in range(100):
        y[i]=y[i]+i**((150-i)/80.0) # modifies the trend
    
    # smoothing the data
    pylab.figure(1)
    pylab.plot(y,'-k',label="original signal",alpha=.3)
    for b in beta:
        yy = smooth(y,b) 
        pylab.plot(yy,label="filtered (beta = "+str(b)+")")
    pylab.legend()
    pylab.show()  
    
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for operating on matrix
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''

def normalize_sum_row_1(mat):
    #normalize a matrix so that: sum(row_i) = 1 for all row_i
    r,c = np.shape(mat)
    sr = [np.sum(np.abs(mat[i, :])) for i in np.arange(r)]
    temp_mat = np.array([np.abs(mat[i,:])*1./sr[i] for i in np.arange(r)],dtype=float)
    
    return temp_mat
    
 
'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Some functions for probability mapping
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
'''
 
def init_prb_state_sparse(tract1, tract2, nearest = 10):
    '''
    distribution based on the convert of distance
    '''   
    from dipy.tracking.distances import bundles_distances_mam
    
    dm12 = bundles_distances_mam(tract1, tract2)
    
    #print dm12
    
    cs_idxs = [dm12[i].argsort()[:nearest] for i in np.arange(len(tract1))] #chosen indices
    ncs_idxs = [dm12[i].argsort()[nearest:] for i in np.arange(len(tract1))] #not chosen indices

    size1 = len(tract1)
    
    for i in np.arange(size1):
        cs_idxs[i].sort()
        ncs_idxs[i].sort()
        dm12[i][ncs_idxs[i]] = 0      
    
    '''
    test sparse optimzation
    '''
    #print cs_idxs
    #print dm12
    
    prb = np.zeros((size1,nearest))
 
    for i in np.arange(size1):
        prb[i] = dm12[i][cs_idxs[i]]
       
    from common_functions import normalize_sum_row_1
    prb = normalize_sum_row_1(prb)
    
    #print prb
    #stop
    return np.array(prb,dtype='float'),np.array(cs_idxs, dtype = 'float')   