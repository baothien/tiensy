import os.path as op
import pyglet
from dipy.io.pickles import load_pickle,save_pickle
import Tkinter, tkFileDialog
from PySide import QtCore


import numpy as np
import nibabel as nib
from dipy.segment.quickbundles import QuickBundles
#from dipy.viz.fos.streamshow import StreamlineLabeler
#from dipy.viz.fos.streamwindow import Window
from streamshow_loadtrack_color import StreamlineLabeler
from streamwindow_rendering import Window
from dipy.viz.fos.guillotine import Guillotine
#from guillotine_for_fading import Guillotine
#from streamwindow import Window  # for remove Right Panel
#from guillotine import Guillotine # for slice moving
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import downsample
from fos import Scene, Init, Run
#from dipy.tracking.metrics import length


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
    
    #root = Tkinter.Tk()
    #root.withdraw()
    #dir_name = 'ALS/ALS_Segmentation'
    #tracks_chosen_filename = tkFileDialog.askopenfilename(parent=root,initialdir=dir_name)          
    #num_seeds = 3 #1M  3M     
    #tracks_id=load_pickle(tracks_chosen_filename)     
      
    
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
    data = np.interp(data, [data.min(), data.max()], [0, 255])
    affine = img.get_affine()
    print len(data)       
    
    #id-tracks filenames
    id_tracts_filenames = ['ALS/ALS_Segmentation/BOI/index_pkl/'+str(subsel.value)+'_corticospinal_L_3M.pkl',
                           'ALS/ALS_Segmentation/BOI/index_pkl/'+str(subsel.value)+'_corticospinal_R_3M.pkl',
                           'ALS/ALS_Segmentation/BOI/index_pkl/'+str(subsel.value)+'_test_colors_1.pkl']     
     
    tracts_colors = [np.array([255.0, 0.0, 0.0], dtype='f4'),
                     np.array([0.0, 0.0, 255.0], dtype='f4'),
                     np.array([0.0, 255.0, 0.0], dtype='f4')]    
 
   #load the tracks
    tracks_filename = 'ALS/ALS_Data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'    
    dpr_tracks = Dpy(tracks_filename, 'r')
    tensor_all_tracks=dpr_tracks.read_tracks();
    dpr_tracks.close()
    
    #load  the first track_ids
    tracks_id=load_pickle(id_tracts_filenames[0])                 
    T = [tensor_all_tracks[i] for i in tracks_id]
    
    tracks_colors = []
    for i in np.arange(len(T)):
        tracks_colors.append(tracts_colors[0])
    #print len(tracks_id)    
    
    #load all the rest track_ids
    for j in np.arange(len(id_tracts_filenames)-1):
        tracks_id_other=load_pickle(id_tracts_filenames[j+1]) 
        for i in tracks_id_other:                        
            T.append(tensor_all_tracks[i] )
        
        #print len(tracks_id_other)

        for k in np.arange(len(tracks_id_other)):
            tracks_colors.append(tracts_colors[j+1])
        
        
   # print 'tracks: ', len(T), '  colors:  ', len(tracks_colors)
    
    
    qb=QuickBundles(T, 0., 18)
    #save_pickle(fpkl,qb)
    #qb=load_pickle(fpkl)   
    
   
   
    
    Init()

    title = '[F]oS Streamline Interaction and Segmentation'
    w = Window(caption = title, 
                width = 1200, 
                height = 800, 
                bgcolor = (.5, .5, 0.9), right_panel=False)

    scene = Scene(scenename = 'Main Scene', activate_aabb = False)
    
    #color_selected = np.array([255.0, 0.0, 0.0], dtype='f4')
    #create the interaction system for tracks 
    tl = StreamlineLabeler('Bundle Picker', 
                        qb,qb.downsampled_tracks(), 
                        colors = tracks_colors,#color_selected,
                        vol_shape=data.shape[:3], 
                        tracks_alpha=0.51,
                        affine=affine)    
    
    guil = Guillotine('Volume Slicer', data, affine)

    scene.add_actor(guil)
    scene.add_actor(tl)    

    w.add_scene(scene)
    w.refocus_camera()

    Run()
    w.glWidget.world.camera.rotate_around_focal(90.* np.pi / 180.,"right")
    w.glWidget.world.camera.rotate_around_focal(90.* np.pi / 180.,"yup")
    
    #---------------------------------------------
    # rendering and save the file
    #---------------------------------------------
    
    # F1: fullscreen
    # F2: next frame
    # F3: previous frame
    # F4: start rotating
    # F5: save screen
    #F6: render multiple frame and save
    # F12: reset camera
    # Esc: close window
    
    
    
    #import time
    
    #time.sleep(5)                        
    
    #first rotation
    
#    w.recording(n_frames=50, rotation = True)                           
#      
#    #second hide virtuals
#    tl.hide_virtuals = True
#    w.recording(n_frames=10, rotation = False)                         
#    w.recording(n_frames=40, rotation = True)                         
#    
#    
#    #show virtuals - hide anatomy
#    tl.hide_virtuals = False
#    guil.show_i = False
#    w.recording(n_frames=7, rotation = False)                         
#    w.recording(n_frames=20, rotation = True)                         
#    guil.show_j = False
#    w.recording(n_frames=5, rotation = False)                         
#    w.recording(n_frames=20, rotation = True)                         
#    guil.show_k = False     
#    w.recording(n_frames=5, rotation = False)                         
#    w.recording(n_frames=20, rotation = True)                         
#    
#    guil.show_i = True
#    guil.show_j = True
#    guil.show_k = True  
#    
#    w.recording(n_frames=40, rotation = True)                         
    
    
    fr1 = 80
    fr2 = 80
    fr3 = 160
    w.recording(n_frames=fr1, start_frame = 0, rotation = True, save=True, out_path='/home/bao/tiensy/temp/')
    
    #second hide/show tracts
    tl.hide_virtuals = True #not self.hide_virtuals True
    w.recording(n_frames=fr2/2, start_frame = fr1, rotation = False, save=True, out_path='/home/bao/tiensy/temp/')
    tl.hide_virtuals = False
    w.recording(n_frames=fr2/2, start_frame = fr1+fr2/2, rotation = True, save=True, out_path='/home/bao/tiensy/temp/')

    #third hide/show anatomy
    guil.show_i = False
    w.recording(n_frames=fr3/4, start_frame = fr1+fr2, rotation = False, save=True, out_path='/home/bao/tiensy/temp/')
    guil.show_j = False
    w.recording(n_frames=fr3/4, start_frame = fr1+fr2+fr3/4, rotation = False, save=True, out_path='/home/bao/tiensy/temp/')
    guil.show_k = False
    w.recording(n_frames=fr3/4, start_frame = fr1+fr2+fr3*2/4, rotation = True, save=True, out_path='/home/bao/tiensy/temp/')
    guil.show_i = True
    guil.show_j = True
    guil.show_k = True    
    w.recording(n_frames=fr3/4, start_frame = fr1+fr2+fr3*3/4, rotation = True, save=True, out_path='/home/bao/tiensy/temp/')
    
        
