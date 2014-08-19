# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:54:37 2014

@author: bao


 mapping prototype of tract1 to find the coressponding in tract2
 using the new loss function based on probability
 
 result is the probability of each mapp 
 
 input: tract1(size n), and tract2(size m)
 output: the probability mapping matrix (n,m)
         where pr(i,j) is the probability of streamline i in tract1 mapped to streamline j in tract2
         for each row, sum(pr(i,j)) (with all j) should be equal to 1
"""
import time
import numpy as np
from dipy.viz import fvtk
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from dipy.io.pickles import save_pickle
from common_functions import cpu_time, load_tract, visualize_tract, plot_smooth
import matplotlib.pyplot as plt
import os
import argparse
        
def init_prb_state(size1, size2):
    '''
    normal Gausan distribution
    '''
    prb = np.zeros((size1, size2)) 
    prb[:,:] = 1./size2
    return np.array(prb,dtype='float')
    
    
def init_prb_state_1(tract1, tract2):
    '''
    distribution based on the convert of distance
    '''   
    
    dm12 = bundles_distances_mam(tract1, tract2)
        
    from common_functions import normalize_sum_row_1
    prb = normalize_sum_row_1(dm12)
    
    return np.array(prb,dtype='float')  

def init_prb_state_sparse(tract1, tract2, nearest = 10):
    '''
    distribution based on the convert of distance
    '''   
    
    dm12 = bundles_distances_mam(tract1, tract2)
    
    print dm12
    
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
    
   
def create_bounds_sparse(size1, size2):
    ''' test sparse optimization'''
    bnds = []             
    for i in np.arange(size1):        
        for j in np.arange(size2):                        
                bnds.append((0.00000,1))            
            
    #print bnds

    bnds = tuple(bnds)    
    return bnds


def create_constrains(size):
    cons = []
    for i in np.arange(size):
        cons.append({'type': 'eq', 'fun': constr_sum_1, 'args' : (size , i)})
    cons = tuple(cons)
    return cons
    
def create_constrains_ineq(size):
    cons = []
    for i in np.arange(size):
        cons.append({'type': 'ineq', 'fun': constr_sum_1, 'args' : (size , i)})
        cons.append({'type': 'ineq', 'fun': constr_1_sum, 'args' : (size , i)})
    cons = tuple(cons)
    return cons
    
#------------------------------------------------------------------------------  
#loss function for optimize by SLSQP with bounds and constrains
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#f = sum((y - f(p,x))^2)
 
#def constr_sum_1(p, y_dm1, x_dm2, k):
def constr_sum_1(p, size1, k):    
    p_2D = np.reshape(p,(size1,-1))
    sp = np.sum(p_2D[k,:])
       
    return 1 - sp

def constr_1_sum(p, size1, k):    
    p_2D = np.reshape(p,(size1,-1))
    sp = np.sum(p_2D[k,:])
       
    return sp - 1


'''
sparse optimization
'''
import numba as nb
#from numba import autojit, jit

#@nb.jit('float[:,:](int8, int8, float[:,:],float[:,:], float[:,:], float[:,:])')
#@nb.autojit
def avg_dis_sparse(size1, size2, dm1, dm2, p, cs_idxs):
    #ad = np.zeros((size1, size1))      
    tmp = 0.
    prb_map12 = np.reshape(p,(size1, -1))
    for i in np.arange(size1):
        for j in np.arange(size1):
            if j!=i:
                t = 0.
                for k in np.arange(size2):
                    for l in np.arange(size2):                
                     #   ad[i,j] = ad[i,j] + np.sum(prb_map12[i,k]*prb_map12[j,l]*dm2[cs_idxs[i,k],cs_idxs[j,l]])
                       t = t + np.sum(prb_map12[i,k]*prb_map12[j,l]*dm2[cs_idxs[i,k],cs_idxs[j,l]])
                tmp = tmp + (dm1[i,j] - t)*(dm1[i,j] - t)
    return tmp# ad

#fast_avg = jit(float(int, int, float[:,:], float[:,:], float[:], int[:,:]))(avg_dis_sparse)
fast_avg = nb.jit(avg_dis_sparse)
#@jit    
def f_1D_slsqp_sparse(p, y_dm1, x_dm2, cs_idxs):    
    """
    Computes the loss function of a given probability mapping.
    dm1, and dm2 are the distance matrices of tract1 and tract2 repestively        
    """
    size1 = np.shape(y_dm1)[0]
    size2 = np.shape(cs_idxs)[1]
    '''
    ad = avg_dis_sparse(size1, size2, x_dm2, p, cs_idxs)    
    c = (y_dm1 - ad).flatten()
    loss = np.sum(c*c)
        
    return loss
    '''

    return avg_dis_sparse(size1, size2, y_dm1, x_dm2, p, cs_idxs)
    #return fast_avg(size1, size2, y_dm1, x_dm2, p, cs_idxs)

#@jit    
def inter_loss_sparse(p):
    global L, y_dm1, x_dm2, cs_idxs
    l = f_1D_slsqp_sparse(p, y_dm1, x_dm2, cs_idxs)
    L.append(l)
    
    return l
    
#inter_loss_sparse = nb.jit(inter_loss_sparse_1)

#@jit
#functions for calculating the gradients
def A_2_sparse(j, n, m, p_tmp, x_dm2_tmp, size2, cs_idxs): 
    #p_tmp = np.reshape(p,(-1, size2))  
    #x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    A1 = 0.
    A2 = 0.
    for k in np.arange(size2):
        for l in np.arange(size2):
            #nk = cs_idxs[n][k] 
            #jl = cs_idxs[j][l]
            A1 = A1 + p_tmp[n,k] * p_tmp[j,l] * x_dm2_tmp[cs_idxs[n,k],cs_idxs[j,l]]

    for l in np.arange(size2):    
        A2 = A2 + p_tmp[j,l] * x_dm2_tmp[cs_idxs[n,m],cs_idxs[j,l]]

    #A2 = np.sum(p_tmp[j,:] * x_dm2_tmp[m,:])
        
    return A1, A2
    
#A_2_sparse = nb.jit(A_2_sparse_1)

#@jit    
def B_2_sparse(i, n, m, p_tmp, x_dm2_tmp, size2, cs_idxs):  
    #p_tmp = np.reshape(p,(-1, size2))  
    #x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    B1 = 0.
    B2 = 0.
    for k in np.arange(size2):
        for l in np.arange(size2):
            B1 = B1 + p_tmp[i,k] * p_tmp[n,l] * x_dm2_tmp[cs_idxs[i,k],cs_idxs[n,l]]
        
        B2 = B2 + p_tmp[i,k] * x_dm2_tmp[cs_idxs[i,k],cs_idxs[n,m]]
    return B1, B2
    
#B_2_sparse = nb.jit(B_2_sparse_1)
#@jit    
def gradient_f_1D_slsqp_sparse(p, y_dm1, x_dm2, cs_idxs):
    #size1 number of fibers in source tract
    #size2 number of fibers in target tract (nearest neighbors)
    size1 = np.shape(y_dm1)[0]
    size2 = np.shape(cs_idxs)[1]
    
    p_tmp = np.reshape(p,(-1, size2))  
    
    jac = np.empty([size1,size2])
    
    for n in np.arange(size1):
        for m in np.arange(size2):
            
            grd = 0.            
            sum1 = 0.            
            for j in np.arange(size1):
                if j!=n:
                    A1, A2 = A_2_sparse(j, n, m, p_tmp, x_dm2, size2, cs_idxs)                
                    sum1 = sum1 + ((y_dm1[n,j] - A1) * A2) 
                    
            sum2 = 0.            
            for i in np.arange(size1):       
                if i!=n:
                    B1, B2 = B_2_sparse(i, n, m, p_tmp, x_dm2, size2, cs_idxs)                
                    sum2 = sum2 + ((y_dm1[i,n] - B1) * B2) 
                    
            grd = - 2.*(sum1 + sum2)
            
            jac[n,m] = grd
            
    return jac.flatten() 

#gradient_f_1D_slsqp_sparse = nb.jit(gradient_f_1D_slsqp_sparse_1)
    
'''
end of sparse optimization
'''
 
def scipy_slsqp_sparse(prb_map12_init, y_dm1, x_dm2, cs_idxs, max_nfe=50000, vis=False):
    
    
    '''
    #test the gradient
    from scipy.optimize import check_grad
    err = check_grad(f_1D_slsqp_sparse, gradient_f_1D_slsqp_sparse, prb_map12_init.flatten(), y_dm1, x_dm2, cs_idxs) #correct
    

    print 'Check grad: ', err
    stop
    '''
    
    
    from scipy.optimize import minimize, fmin_slsqp
    size1 = np.shape(y_dm1)[0]
    size2 = np.shape(cs_idxs)[1]
    
    bnds = create_bounds_sparse(size1,size2)
    cons = create_constrains(size1)
    #stop
    
    t0 = cpu_time()  
    
    
    #bounds and constrains
    #if vis:
    #    print 'Optimizing based on slsqp with bounds and constrains'
    #res = minimize(f_1D_slsqp, prb_map12_init.flatten(),args=(y_dm1, x_dm2), method='SLSQP', bounds=bnds,
    #            constraints = cons, options = ({'maxiter' : max_nfe, 'disp': True}))
    if vis:
        print 'Optimizing based on slsqp with bounds and constrains. Also gradient is provided'
        
    
    res = minimize(f_1D_slsqp_sparse, prb_map12_init.flatten(), args=(y_dm1, x_dm2, cs_idxs), 
                   method='SLSQP',
                   jac = gradient_f_1D_slsqp_sparse,
                   bounds=bnds, 
                   constraints = cons, 
                   callback = inter_loss_sparse,
                   options = ({'maxiter' : max_nfe, 'disp': True}))                   
    
    t_opt = cpu_time() - t0
    
    plsq = res.x
    
    prb_map12 = np.reshape(plsq,(size1,-1))    
   
    
    #print 'Number of iteration', res.nit
    #print 'Final value of object func:', res.fun    
    #print 'Exit mode', res.status, res.message
    
    
    if vis:        
        #print plsq        
        #print 'Probability mapping: ',  prb_map12  
        begin_err = f_1D_slsqp_sparse(prb_map12_init.flatten(), y_dm1, x_dm2, cs_idxs)
        print 'Loss function before optimizing: ', begin_err
        final_err = f_1D_slsqp_sparse(plsq, y_dm1, x_dm2, cs_idxs)
        print 'Loss function after optimizing: ', final_err 
        print 'Optimizing cpu time: ', t_opt    
       
        
    return t_opt, prb_map12

 

#end of SLSQP with bounds and constains
#------------------------------------------------------------------------------




#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
                                 description="Tractography mapping with probability",
                                 epilog="Written by Bao Thien Nguyen, tbnguyen@fbk.eu.",
                                 version='1.0')

parser.add_argument(
                    'inputSourceTractography',
                    help='The file name of source whole-brain tractography as .dpy/.trk format.')
parser.add_argument(
                    'inputSourceCSTSFFIndex',
                    help='The file name of source CST plus SFF index')

parser.add_argument(
                    'inputTargetTractography',
                    help='The file name of target whole-brain tractography as .dpy format.')
parser.add_argument(
                    'inputTargetCSTIndex',
                    help='The file name of target CST index')
parser.add_argument(
                    'inputTargetCSTExtIndex',
                    help='The file name of target CST extension index')

parser.add_argument(
                    '-pr', action='store', dest='inputNumPrototypes', type=int,
                    help='The number of prototypes') 

parser.add_argument(
                    '-nn', action='store', dest='inputNumNeighbors', type=int,
                    help='The number of neighbors') 
                    
parser.add_argument(
                    'outputMapProbFile',
                    help='The output file name of the probability mapping')                    

parser.add_argument(
                    'outputObjFuncFile',
                    help='The output file name of the objective function eveluation (pdf or png)')   

args = parser.parse_args()

print "=========================="
print "Source tractography:       ", args.inputSourceTractography
#print "Source CST plus SFF index:       ", args.inputSourceCSTSFFIndex
print "Target tractography:       ", args.inputTargetTractography
#print "Target CST index:       ", args.inputTargetCSTIndex
#print "Target CST extension index:       ", args.inputTargetCSTExtIndex
#print "Number of prototypes:      ", args.inputNumPrototypes 
print "=========================="

#if not os.path.isdir(args.inputDirectory):
#    print "Error: Input directory", args.inputDirectory, "does not exist."
#    exit()


s_file = args.inputSourceTractography
s_ind = args.inputSourceCSTSFFIndex

t_file = args.inputTargetTractography
t_ind = args.inputTargetCSTExtIndex
t_cst = args.inputTargetCSTIndex

num_pro = args.inputNumPrototypes 
nearest = args.inputNumNeighbors

map_prob = args.outputMapProbFile

obj_func_file = args.outputObjFuncFile
#obj_func_file = os.path.join(os.path.curdir, 'objective_function_'+ str(num_pro) + '_' + str(num_pro) + '_sparse_density_' + str(nearest) + '_neighbors.pdf')
vis = False
save = True#False

source_cst = load_tract(s_file,s_ind)

target_cst_ext = load_tract(t_file,t_ind)

print len(source_cst), len(target_cst_ext)

tractography1 = source_cst[-num_pro:]
tractography2 = target_cst_ext[:num_pro]
#tractography2 = target_cst_ext[:num_pro*2]

print "Source", len(tractography1)
print "Target", len(tractography2)
print "Neighbors", nearest


#print "Computing the distance matrices for each tractography."
dm1 = bundles_distances_mam(tractography1, tractography1)
dm2 = bundles_distances_mam(tractography2, tractography2)

size1 = len(tractography1)
size2 = len(tractography2)

if vis:
    ren = fvtk.ren() 
    ren = visualize_tract(ren, tractography1, fvtk.yellow)
    ren = visualize_tract(ren, tractography2, fvtk.blue)
    fvtk.show(ren)
  

y_dm1 = dm1
x_dm2 = dm2
L = []
print 'Optimizing ...........................'    

max_nfe = 50000

#optimizing with SLSPQ  ---------------------
#-----------with bounds and (constrains OR NO constrains)
for t in np.arange(1):
    L = []
    print '-------------------- Iteration = ', t      
      
    #sparse optimization    
    prb_map12_init, cs_idxs = init_prb_state_sparse(tractography1,tractography2,nearest) 
    
    t_opt, prb_map12 = scipy_slsqp_sparse(prb_map12_init.flatten(), y_dm1, x_dm2, cs_idxs, max_nfe,vis = True)
    
    #print 'Time :', t_opt
    

    #check the result 
    print 'Map : ', prb_map12
    for i in np.arange(size1):
        print 'sum row ', i , np.sum(prb_map12[i,:])

    #compare to mapping results
    if t==0:
        pre_map12 = np.copy(prb_map12_init)            
    norm = np.linalg.norm(prb_map12 - pre_map12)
    print "Norm of results - with previous: ", norm
    pre_map12 = np.copy(prb_map12)
    
         
    #print L      
    plot_smooth(plt, np.arange(len(L)), L, False)       
    
    if save:
        save_pickle(map_prob, prb_map12)
    
plt.title('Loss function ')  
plt.xlabel('Gradient evaluations')  
if save:
    #plt.savefig(os.path.join(os.path.curdir, 'objective_function_'+ str(num_pro) + '_' + str(num_pro) + '_sparse_density_' + str(nearest) + '_neighbors.pdf'))
    plt.savefig(obj_func_file)
plt.show()

#end of optimize with slsqp





   



 




       