# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:18:26 2012

@author: bao
"""

import numpy as np
import nibabel as nib
import os.path as op
import pyglet

#fos modules
from fos.actor.axes import Axes
from fos import World, Window, WindowManager
from labeler import TrackLabeler
from fos.actor.slicer import Slicer
#from slicer import Slicer

#dipy modules
from dipy.segment.quickbundles import QuickBundles
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
from dipy.viz.colormap import orient2rgb
import copy
import Tkinter, tkFileDialog


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

if __name__ == '__main__':

#   choose the file to visualie
    root = Tkinter.Tk()
    root.wm_title('Select the segmentation result')
    root.withdraw()
    dir_name = 'Tracts'
    tracks_chosen_filename = tkFileDialog.askopenfilename(parent=root,initialdir=dir_name)          
#    tracks_chosen_filename = 'Segmentation/s201_corticospinal_left_3M_Nivedita.pkl'
    tracks_id=load_pickle(tracks_chosen_filename)
         
    subj = 201
    num_seeds = 3 #1M  3M 

    root = Tkinter.Tk()
    root.wm_title('Subject Selection')
    subsel = SubjectSelector(root, default_value=1)
    root.wait_window()
    subj = subsel.value
  
   
    #load the tracks
    tracks_filename = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'
    dpr_tracks = Dpy(tracks_filename, 'r')
    tensor_all_tracks=dpr_tracks.read_tracks();
    dpr_tracks.close()
    tensor_tracks=[tensor_all_tracks[i] for i in tracks_id]
        
    print len(tensor_tracks)
    print len(tracks_id)
    
    #load the volume
    img = nib.load('data/'+str(subj)+'/MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz') 
    data = img.get_data()
    affine = img.get_affine()
    print len(data)
  
    tensor_tracks=[t-np.array(data.shape)/2. for t in tensor_tracks]
    
    qb=QuickBundles(tensor_tracks,1.,12)
    #visualisation part        
    tl = TrackLabeler(qb,qb.downsampled_tracks(),vol_shape=data.shape,tracks_alpha=1,virtuals_line_width=5.0)
    sl = Slicer(affine,data)#,alpha=255)    
    #one way connection
    tl.slicer=sl
    #OpenGL coordinate system axes    
    ax = Axes(100)
    #add the actors to the world    
    w=World()
    w.add(tl)
    w.add(sl)
    w.add(ax)
    #create a window
    wi = Window(caption="Visualization the segmentation",\
                bgcolor=(0.3,0.3,0.6,1),width=1366,height=730)
    #attach the world to the window
    wi.attach(w)
    #create a manager which can handle multiple windows
    wm = WindowManager()
    wm.add(wi)    
    wm.run()
    print('Everything is running ;-)')
    
