import numpy as np
from fos import Window, Scene
#from fos.actor.slicer import Slicer
from slicer_for_fading import Slicer
from pyglet.gl import *
from PySide.QtCore import Qt


class Guillotine(Slicer):
    """ Head slicer actor

    Notes
    ------
    Coordinate Systems
    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    http://www.slicer.org/slicerWiki/index.php/Coordinate_systems
    http://eeg.sourceforge.net/mri_orientation_notes.html

    """
    step = 1
    def draw(self):
        
        #i slice
        if self.show_i: 
            glPushMatrix()
            glRotatef(90, 0., 1., 0)
            glRotatef(90, 0., 0., 1.)
            self.tex.update_quad(self.texcoords_i, self.vertcoords_i)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()
        
        #j slice
        if self.show_j:
            glPushMatrix()
            glRotatef(180, 0., 1., 0) # added for fsl convention
            glRotatef(90, 0., 0., 1.)
            self.tex.update_quad(self.texcoords_j, self.vertcoords_j)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()

        #k slice
        if self.show_k:
            glPushMatrix()
            glRotatef(90, 1., 0, 0.)
            glRotatef(90, 0., 0., 1)
            glRotatef(180, 1., 0., 0.) # added for fsl
            self.tex.update_quad(self.texcoords_k, self.vertcoords_k)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()

    def right2left(self, step):
        if self.i + step < self.I:
            self.slice_i(self.i + step)
        else:
            self.slice_i(self.I - 1)

    def left2right(self, step):
        if self.i - step >= 0:
            self.slice_i(self.i - step)
        else:
            self.slice_i(0)

    def inferior2superior(self, step):
        if self.k + step < self.K:
            self.slice_k(self.k + step)
        else:
            self.slice_k(self.K - 1)

    def superior2inferior(self, step):
        if self.k - step >= 0:
            self.slice_k(self.k - step)
        else:
            self.slice_k(0)

    def anterior2posterior(self, step):
        if self.j + step < self.J:
            self.slice_j(self.j + step)
        else:
            self.slice_j(self.J - 1)

    def posterior2anterior(self, step):
        if self.j - step >= 0:
            self.slice_j(self.j - step)
        else:
            self.slice_j(0)

    def reset_slices(self):
        self.slice_i(self.I / 2)
        self.slice_j(self.J / 2)
        self.slice_k(self.K / 2)

    def slices_ijk(self, i, j, k):
        self.slice_i(i)
        self.slice_j(j)
        self.slice_k(k)
        
    def show_coronal(self, bool=True):
        self.show_k = bool

    def show_axial(self, bool=True):
        self.show_i = bool

    def show_saggital(self, bool=True):
        self.show_j = bool

    def show_all(self, bool=True):
        self.show_i = bool
        self.show_j = bool
        self.show_k = bool

    def process_messages(self,messages):
        msg=messages['key_pressed']
        #print 'Processing messages in actor', self.name, 
#        #' key_press message ', msg
        if msg!=None:
            self.process_keys(msg,None)
#        msg=messages['mouse_position']            
#        #print 'Processing messages in actor', self.name, 
#        #' mouse_pos message ', msg
#        if msg!=None:
#            self.process_mouse_position(*msg)
        pass
            
    def process_keys(self,symbol,modifiers):
        
        #NEEDS to change
        #if modifiers & Qt.Key_Shift:            
        #    print 'Shift'
            #print("Increase step.")
        #    self.step=5             
        if symbol == Qt.Key_Up:
            print 'Up'
            self.inferior2superior(self.step)
            self.step=1 
        if symbol == Qt.Key_Down:
            print 'Down'            
            self.superior2inferior(self.step)                     
            self.step=1
        if symbol == Qt.Key_Left:
            print 'Left'
            self.left2right(self.step)                        
            self.step=1         
        if symbol == Qt.Key_Right:
            print 'Right'
            self.right2left(self.step)
            self.step=1            
        if symbol == Qt.Key_PageUp:
            print 'PgUp'
            self.posterior2anterior(self.step)
            self.step=1
        if symbol == Qt.Key_PageDown:
            print 'PgDown'
            self.anterior2posterior(self.step)
            self.step=1
        #HIDE SLICES
        if symbol == Qt.Key_0:
            print('0 - Show/Hide all')
            if self.show_i==True | self.show_j==True | self.show_k==True :
                self.show_i = False
                self.show_j = False
                self.show_k = False
            else:                
                self.show_i = True
                self.show_j = True
                self.show_k = True
        if symbol == Qt.Key_1:
            print('1')
            self.show_axial(not self.show_i)
        if symbol == Qt.Key_2:
            print('2')
            self.show_saggital(not self.show_j)
        if symbol == Qt.Key_3:
            print('3')
            self.show_coronal(not self.show_k)
        if symbol == Qt.Key_Home:
            print('Home')
            self.reset_slices()            
#        if symbol == Qt.Key_End:            
#            print('End')
#            if self.alpha>0:
#                self.alpha-=1
#                self.sli=self.update_slice(0,self.vxi)
#                self.slj=self.update_slice(1,self.vxj)
#                self.slk=self.update_slice(2,self.vxk)
    

if __name__ == '__main__':

    import nibabel as nib    
    
    dname = '/usr/share/fsl/data/standard/'
    fname = dname + 'FMRIB58_FA_1mm.nii.gz'

    fname = '/home/eg309/Data/trento_processed/subj_01/MPRAGE_32/rawbet.nii.gz'
    img=nib.load(fname)
    data = img.get_data()
    data = np.interp(data, [data.min(), data.max()], [0, 255])

    from fos import Init, Run

    Init()

    window = Window(caption="[F]OS", bgcolor=(0.4, 0.4, 0.9))
    scene = Scene(activate_aabb=False)
    guil = Guillotine('VolumeSlicer', data)
    scene.add_actor(guil)
    window.add_scene(scene)
    window.refocus_camera()

    #window = Window()
    window.show()
    #print get_ipython()

    Run()
