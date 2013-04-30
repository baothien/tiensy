# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:46:57 2012

@author: bao
"""

import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
#from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
from dipy.viz import fvtk

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

    root = Tkinter.Tk()
    root.withdraw()
    #dir_name = 'data/201/DIFF2DEPI_EKJ_64dirs_14/DTI'
    dir_name = 'New_Seg_1210'
    tracks_chosen_filename = tkFileDialog.askopenfilename(parent=root,initialdir=dir_name)          
    #tracks_chosen_filename = './s201_corticospinal_left_3M_Nivedita.pkl'
    tracks_id=load_pickle(tracks_chosen_filename)
    
    subj = 213
    num_seeds = 3 #1M  3M
        
    #root = Tkinter.Tk()
    #root.wm_title('Subject Selection')
    #subsel = SubjectSelector(root, default_value=1)
    #root.wait_window()
    #subj = subsel.value
    
    tracks_filename = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'
    #tracks_filename = dirname + '/tracks_dti_3M_linear.dpy'
    dpr_tracks = Dpy(tracks_filename, 'r')
    tensor_all_tracks=dpr_tracks.read_tracks();
    dpr_tracks.close()
    tensor_tracks=[tensor_all_tracks[i] for i in tracks_id]
        
    print len(tensor_tracks)
    print len(tracks_id)
    visualize = True
    if visualize:
        
        renderer = fvtk.ren()
        fvtk.add(renderer, fvtk.line(tensor_tracks[:], fvtk.red, opacity=1.0))  
        fvtk.show(renderer)
