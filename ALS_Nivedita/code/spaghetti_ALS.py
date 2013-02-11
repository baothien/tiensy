import numpy as np
import nibabel as nib
import os.path as op

import pyglet
#pyglet.options['debug_gl'] = True
#pyglet.options['debug_x11'] = True
#pyglet.options['debug_gl_trace'] = True
#pyglet.options['debug_texture'] = True

#fos modules
from fos.actor.axes import Axes
from fos import World, Window, WindowManager
from labeler_ALS import TrackLabeler
from fos.actor.slicer import Slicer
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

    
    subj = 1
    num_seeds = 3 #1M  3M
    qb_dist = 30 #20 15
    
    root = Tkinter.Tk()
    root.wm_title('Subject Selection')
    subsel = SubjectSelector(root, default_value=1)
    root.wait_window()
    subj = subsel.value

    #load the volume
    #img = nib.load('/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz')
    img = nib.load('data/'+str(subj)+'/MP_Rage_1x1x1_ND_3/T1_flirt_out.nii.gz')
    #img = nib.load('/home/eg309/Desktop/out.nii.gz')
    data = img.get_data()
    affine = img.get_affine()
    
    #load the tracks
    fdpyw = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_'+str(num_seeds)+'M_linear.dpy'
    
    dpr = Dpy(fdpyw, 'r')
    T = dpr.read_tracks()
    dpr.close()
    
    #T=T[:50000]
    #T=[t-np.array(data.shape)/2. for t in T]
    
    fpkl = 'data/'+str(subj)+'/DIFF2DEPI_EKJ_64dirs_14/DTI/qb_dti_'+str(num_seeds)+'M_linear_'+str(qb_dist)+'.pkl'
    #qb=QuickBundles(T,10.,12)

    qb=load_pickle(fpkl)
    #visualisation part        
    tl = TrackLabeler(qb,qb.downsampled_tracks(),vol_shape=data.shape,tracks_alpha=1,virtuals_line_width=5.0)
    sl = Slicer(affine,data) # ,alpha=255)    
    #one way connection
    tl.slicer=sl
    #OpenGL coordinate system axes    
    ax = Axes(100)
    x,y,z=data.shape
    #add the actors to the world    
    w=World()
    w.add(tl)
    w.add(sl)
    w.add(ax)
    #create a window
    wi = Window(caption="Interactive Spaghetti using Diffusion Imaging in Python (dipy.org) and Free On Shades (fos.me)",\
                bgcolor=(0.3,0.3,0.6,1),width=1366,height=730)
    #attach the world to the window
    wi.attach(w)
    #create a manager which can handle multiple windows
    wm = WindowManager()
    wm.add(wi)    
    wm.run()
    print('Everything is running ;-)')
    
