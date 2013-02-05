import os.path as op
import pyglet
from dipy.io.pickles import load_pickle,save_pickle
import Tkinter, tkFileDialog

import numpy as np
import nibabel as nib
from dipy.segment.quickbundles import QuickBundles
from dipy.viz.fos.streamshow import StreamlineLabeler
#from dipy.viz.fos.streamwindow import Window
#from dipy.viz.fos.guillotine import Guillotine
from streamwindow import Window  # for removing the Right Panel
from guillotine import Guillotine # for binding slice moving key
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import downsample
from fos import Scene, Init, Run
from dipy.tracking.metrics import length

class SubjectSelector(object):
    def __init__(self, parent, default_value):
        self.parent = parent
        self.s = Tkinter.Scale(self.parent, from_=1, to=24, width=25, length=300, orient=Tkinter.HORIZONTAL)
        self.s.set(default_value)
        self.s.pack()
        self.b = Tkinter.Button(self.parent, text='OK', command=self.ok)
        self.b.pack(side=Tkinter.BOTTOM)
    def ok(self):
        self.value = self.s.get()
        self.parent.destroy()


def rotation_matrix(axis, theta_degree):
    theta = 1. * theta_degree * np.pi / 180.
    axis = 1. * axis / np.sqrt(np.dot(axis,axis))
    a = np.cos(theta / 2)
    b, c, d = - axis * np.sin(theta / 2)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


if __name__ == '__main__':
    
    #load T1 volume registered in MNI space 
#    dname='/home/bao/tiensy/ADHD/2213364/'
#    img = nib.load(dname+'session_1/MPRAGE_1/T1_flirt_out.nii.gz')
#    #dname='/home/eg309/Devel/fos_legacy/applications/'
#    #img = nib.load(dname+'data/subj_05/MPRAGE_32/T1_flirt_out.nii.gz')
#    data = img.get_data()
#    affine = img.get_affine()
#
#    #load the tracks registered in MNI space 
#    #fdpyw = dname+'data/subj_05/101_32/DTI/tracks_gqi_1M_linear.dpy'    
#    #dpr = Dpy(fdpyw, 'r')
#    fdpyw = dname+'session_1/DTI64_1/DTI/tracks_dti_1M_linear.dpy'    
#    dpr = Dpy(fdpyw, 'r')
#    T = dpr.read_tracks()
#    dpr.close() 
    
    num_seeds = 3    
    
    root = Tkinter.Tk()
    root.wm_title('Subject Selection')
    subsel = SubjectSelector(root, default_value=1)
    root.wait_window()
    mapping  = [0,101,201,102,202,103,203,104,204,105,205,106,206,107,207,109,208,110,209,111,210,112,212,113,213]    
    subj = mapping[subsel.value]
     
     
    #load the volume
    img = nib.load('ALS/ALS_Data/'+str(subj)+'/MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz') 
    data = img.get_data()
    affine = img.get_affine()
    print len(data)
    
    #load the tracks
    tracks_filename = 'ALS/ALS_Data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'    
    dpr_tracks = Dpy(tracks_filename, 'r')
    T = dpr_tracks.read_tracks();
    dpr_tracks.close()    
        
    #print len(T)
    print len(T)   
   
    T = T[:200]
        
    T = [t for t in T if length(t)>= 15]

#    T = [downsample(t, 18) - np.array(data.shape[:3]) / 2. for t in T]
#    axis = np.array([1, 0, 0])
#    theta = - 90. 
#    T = np.dot(T,rotation_matrix(axis, theta))
#    axis = np.array([0, 1, 0])
#    theta = 180. 
#    T = np.dot(T, rotation_matrix(axis, theta))
#    
    #load initial QuickBundles with threshold 30mm
    #fpkl = dname+'data/subj_05/101_32/DTI/qb_gqi_1M_linear_30.pkl'
    #qb=QuickBundles(T, 10., 18)
    #save_pickle(fpkl,qb)
    #qb=load_pickle(fpkl)

    qb=QuickBundles(T, 20., 12)
    #save_pickle(fpkl,qb)
    #qb=load_pickle(fpkl)
    Init()

    title = '[F]oS Streamline Interaction and Segmentation'
    w = Window(caption = title, 
                width = 1200, 
                height = 800, 
                bgcolor = (.5, .5, 0.9), right_panel=True)

    scene = Scene(scenename = 'Main Scene', activate_aabb = False)
    
    #create the interaction system for tracks 
    tl = StreamlineLabeler('Bundle Picker', 
                        qb,qb.downsampled_tracks(), 
                        vol_shape=data.shape[:3], 
                        tracks_alpha=1,
                        affine=affine)

    guil = Guillotine('Volume Slicer', data, affine)

    scene.add_actor(guil)
    scene.add_actor(tl)

    w.add_scene(scene)
    w.refocus_camera()

    Run()
