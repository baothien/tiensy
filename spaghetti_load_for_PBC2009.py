import os.path as op
import pyglet
from dipy.io.pickles import load_pickle,save_pickle
import Tkinter, tkFileDialog

import numpy as np
import nibabel as nib
from dipy.segment.quickbundles import QuickBundles
from dipy.viz.fos.streamshow import StreamlineLabeler
from dipy.viz.fos.streamwindow import Window
from dipy.viz.fos.guillotine import Guillotine
#from streamwindow import Window  # for removing the Right Panel
#from guillotine import Guillotine # for binding slice moving key
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import downsample
from fos import Scene, Init, Run
from dipy.tracking.metrics import length

class SubjectSelector(object):
    def __init__(self, parent, default_value):
        self.parent = parent
        self.s = Tkinter.Scale(self.parent, from_=1, to=3, width=25, length=100, orient=Tkinter.HORIZONTAL)
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
    #mapping  = ['0','brain0_origintrack_newfa','brain1','brain2']
    mapping  = ['0','brain0_fr_raw','brain1_fr_raw','brain2']
    subj = mapping[subsel.value]
     
     
    #load the volume
    dir_path = 'PBC2009/' + subj    
    #img = nib.load(dir_path+'/MPRAGE_1_from_structual_T1Space/T1_flirt_out.nii.gz')
    #print dir_path+'/MPRAGE_1_from_structual_T1Space/T1_flirt_out.nii.gz'
    
    #img = nib.load(dir_path+'/MPRAGE_1_from_correct/fbrain0_T1on_dsi_B0Anz_bet.nii.gz')
    #print dir_path+'/MPRAGE_1_from_correct/fbrain0_T1on_dsi_B0Anz_bet.nii.gz'
    
    #img = nib.load(dir_path+'/MPRAGE_1/anatomy_bet.nii.gz')


    #file_name = dir_path+'/MPRAGE_1_from_correct/T1_flirt_out.nii.gz' #bi cat mat mot khuc
    ##file_name = dir_path+'/MPRAGE_1_from_correct/fbrain0_T1on_dsi_B0Anz_bet.nii.gz'
    #file_name = dir_path+'/MPRAGE_1_from_structuals/T1_flirt_out.nii.gz' #not good
    ###file_name = dir_path+'/MPRAGE_1_from_structuals/T1_anatomy_bet.nii.gz' 
    file_name = dir_path+'/MPRAGE_1_from_structual_T1Space/T1_flirt_out.nii.gz' #good
    ##file_name = dir_path+'/MPRAGE_1_from_structual_T1Space/fbrain0_mprage1_bet.nii.gz'
    
    #file_name = dir_path+'/Structuals_DSISpace/T1_flirt_out.nii.gz' #hoi toi
    img = nib.load(file_name)
    print file_name
    
    
    data = img.get_data()
    data = np.interp(data, [data.min(), data.max()], [0, 255])
    affine = img.get_affine()
    print len(data)
    
    #load the tracks
    #tracks_filename = dir_path + '/DTI/tracks_dti_linear.dpy'    
    #tracks_filename = dir_path + '/DTI/tracks_dti.dpy'    
    #dpr_tracks = Dpy(tracks_filename, 'r')
    #T = dpr_tracks.read_tracks();
    #dpr_tracks.close()    
        
    #print len(T)   
   
    #T = T[:10000]
    #T = [t for t in T if length(t)>= 1]     
    
    #qb=QuickBundles(T, 20., 12)
    
    fpkl = dir_path + '/DTI/qb_dti_10K_linear_15.pkl'    
    qb=load_pickle(fpkl)

    Init()

    title = '[F]oS Streamline Interaction and Segmentation'
    w = Window(caption = title, 
                width = 1200, 
                height = 800, 
                bgcolor = (.5, .5, 0.9), right_panel=False)

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
