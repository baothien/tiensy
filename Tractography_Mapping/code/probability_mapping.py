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
from common_functions import cpu_time, load_tract, visualize_tract
import argparse

def avg_dis(size1, size2, dm2, prb_map12):
    ad = np.zeros((size1, size1))  
    for i in np.arange(size1):
        for j in np.arange(size1):
            for l in np.arange(size2):
                ad[i,j] = ad[i,j] + np.sum(prb_map12[i,l]*prb_map12[j,:]*dm2[l,:])
    return ad
#def loss_function(size1, size2, dm1, dm2, prb_map12):
def loss_function(dm1, dm2, prb_map12):
    """
    Computes the loss function of a given probability mapping.
    dm1, and dm2 are the distance matrices of tract1 and tract2 repestively        
    """
    size1 = len(dm1)
    size2 = len(dm2)      
    ad = avg_dis(size1, size2, dm2, prb_map12)
    loss = np.linalg.norm(dm1 - ad)
    return loss
  
def avg_dis_1D(size1, size2, dm2, prb_map12):
    ad = np.zeros((size1*size1))  
    for i in np.arange(size1):
        for j in np.arange(size1):
            for l in np.arange(size2):
                for k in np.arange(size2):
                    ad[i*size1 +j] = ad[i*size1+j] + prb_map12[i*size2+l]*prb_map12[j*size2+k]*dm2[l*size2+k]
    return ad  
#def loss_function(size1, size2, dm1, dm2, prb_map12):
def loss_function_1D(dm1, dm2, prb_map12):
    """
    Computes the loss function of a given probability mapping.
    dm1, and dm2 are the distance matrices of tract1 and tract2 repestively        
    """
    size1 = np.sqrt(len(dm1))
    size2 = np.sqrt(len(dm2))
    
    ad = avg_dis_1D(size1, size2, dm2, prb_map12)
    loss = np.linalg.norm(dm1 - ad)
    return loss
        
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
    
    
#------------------------------------------------------------------------------       
#function for optimize by leastsq
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#error = y - f(p,x)
def residuals(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    #print 'len p',  len(p)
    #print size1, size2
    #print 'probability ', p
    #print 'dm1 ', y_dm1
    #print 'dm2 ', x_dm2
    ad = avg_dis_1D(size1, size2, x_dm2, p) 
    err = y_dm1-ad
    return err

def scipy_leastsq(prb_map12_init, y_dm1, x_dm2, vis=False):
    from scipy.optimize import leastsq, fmin_slsqp, fmin_bfgs, fmin_powell
    
    t0 = cpu_time()
    plsq = leastsq(residuals, prb_map12_init, args=(y_dm1, x_dm2))
    #plsq = fmin_bfgs(residuals, prb_map12_init, args=(y_dm1, x_dm2))
    #prb_map12 = plsq[0]
    
    t_opt = cpu_time() - t0
    mapp = np.reshape(prb_map12,(size1,-1))
    if vis:
        print 'Probability mapping: ', prb_map12
        print 'Map after reshape :', mapp
        final_loss = loss_function_1D(y_dm1, x_dm2, prb_map12)
        print 'Loss function after optimizing: ', final_loss        
        print 'Optimizing cpu time: ', t_opt    
    return t_opt, mapp

#end of leastsq
#------------------------------------------------------------------------------


def create_bounds(size):
    bnds = []
    for i in np.arange(size):        
        bnds.append((0.00001,1))
    bnds = tuple(bnds)
    return bnds
    
def create_constrains(size):
    cons = []
    for i in np.arange(size):
        cons.append({'type': 'eq', 'fun': constr_sum_1, 'args' : (size , i)})
    cons = tuple(cons)
    return cons
    

#------------------------------------------------------------------------------  
#loss function for optimize by SLSQP with bounds and constrains
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#f = sum((y - f(p,x))^2)
def f_1D_slsqp(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    #print 'len p',  len(p)
    #print size1, size2
    #print 'probability ', p
    #print 'dm1 ', y_dm1
    #print 'dm2 ', x_dm2
    ad = avg_dis_1D(size1, size2, x_dm2, p) 
    f = np.linalg.norm(y_dm1-ad)
    return f

#functions for calculating the gradients
def A(j, n, m, p, x_dm2, size2):    
    A1 = 0.
    A2 = 0.
    for l in np.arange(size2):
        for k in np.arange(size2):
            A1 = A1 + p[n*size2 + k] * p[j*size2 + l] * x_dm2[k*size2 + l]
        A2 = A2 + p[j*size2 + l] * x_dm2[m*size2 + l]
    return A1, A2
    
def B(i, n, m, p, x_dm2, size2):    
    B1 = 0.
    B2 = 0.
    for k in np.arange(size2):
        for l in np.arange(size2):
            B1 = B1 + p[i*size2 + k] * p[n*size2 + l] * x_dm2[k*size2 + l]
        B2 = B2 + p[i*size2 + k] * x_dm2[k*size2 + m]
    return B1, B2
    
#functions for calculating the gradients
def A_2(j, n, m, p_tmp, x_dm2_tmp, size2): 
    #p_tmp = np.reshape(p,(-1, size2))  
    #x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    A1 = 0.
    A2 = 0.
    for l in np.arange(size2):
        for k in np.arange(size2):
            A1 = A1 + p_tmp[n,k] * p_tmp[j,l] * x_dm2_tmp[k,l]
        
        A2 = A2 + p_tmp[j,l] * x_dm2_tmp[m,l]
    return A1, A2
    
def B_2(i, n, m, p_tmp, x_dm2_tmp, size2):  
    #p_tmp = np.reshape(p,(-1, size2))  
    #x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    B1 = 0.
    B2 = 0.
    for k in np.arange(size2):
        for l in np.arange(size2):
            B1 = B1 + p_tmp[i,k] * p_tmp[n,l] * x_dm2_tmp[k,l]
        
        B2 = B2 + p_tmp[i,k] * x_dm2_tmp[k,m]
    return B1, B2
 
#functions for calculating the gradients
def A_3(j, m, p, x_dm2, size2):    
    A = 0.    
    for l in np.arange(size2):        
        A = A + p[j,l] * x_dm2[m,l]        
    return A
    
def B_3(i, m, p, x_dm2, size2):    
    B = 0.    
    for k in np.arange(size2):        
        B = B + p[i, k] * x_dm2[k, m]
    return B
 
def gradient_f_1D_slsqp_3(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    p_tmp = np.reshape(p,(-1, size2))  
    y_dm1_tmp = np.reshape(y_dm1,(-1, size1))  
    x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    jac = np.empty([size1,size2])
    
    for n in np.arange(size1):
        for m in np.arange(size2):
            
            sum1 = 0.
            sum2 = 0.
            for j in np.arange(size1):
                #if j!=n:
                A = A_3(j, m, p_tmp, x_dm2_tmp, size2)                
                sum1 = sum1 + (A*A - 2.*y_dm1_tmp[n,j] * A) 
                    
                #if j!=n:
                B = B_3(j, m, p_tmp, x_dm2_tmp, size2)                
                sum2 = sum2 + (B*B - 2.*y_dm1_tmp[j,n] * B) 
            grd = - 2 * p_tmp[n,m] * (sum1 + sum2)
            
            jac[n,m] = grd
            
    return jac.flatten()   
  
#Gradient of loss function for optimize by SLSQP with bounds and constrains
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#f = sum((y - f(p,x))^2)
#return the vector that the number of element is equal to the size of parameter needed to optimized p
def gradient_f_1D_slsqp(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    p_tmp = np.reshape(p,(-1, size2))  
    y_dm1_tmp = np.reshape(y_dm1,(-1, size1))  
    x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    jac = np.empty([size1,size2])
    
    for n in np.arange(size1):
        for m in np.arange(size2):
            
            sum1 = 0.
            sum2 = 0.
            for j in np.arange(size1):
                if j!=n:
                    A1, A2 = A_2(j, n, m, p_tmp, x_dm2_tmp, size2)                
                    sum1 = sum1 + ((y_dm1_tmp[n,j] - A1) * A2) 
                    
                if j!=n:
                    B1, B2 = B_2(j, n, m, p_tmp, x_dm2_tmp, size2)                
                    sum2 = sum2 + ((y_dm1_tmp[j,n] - B1) * B2) 
            grd = - 2.*(sum1 + sum2)
            
            jac[n,m] = grd
            
    return jac.flatten()   

#function for optimize by SLSQP with bounds and no constrains
def f_1D_slsqp_no_constrains(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    prob_temp = np.reshape(p,(size1,-1))    
    from common_functions import normalize_sum_row_1
    p_temp = normalize_sum_row_1(prob_temp)    
    ad = avg_dis_1D(size1, size2, x_dm2, p_temp.flatten())  
    f = np.linalg.norm(y_dm1-ad)
    return f
    
#def constr_sum_1(p, y_dm1, x_dm2, k):
def constr_sum_1(p, size1, k):    
    p_2D = np.reshape(p,(size1,-1))
    sp = np.sum(p_2D[k,:])
       
    return 1 - sp
   
def scipy_slsqp(prb_map12_init, y_dm1, x_dm2, max_nfe=50000, vis=False):
    from scipy.optimize import minimize
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    bnds = create_bounds(size1*size2)
    cons = create_constrains(size1)

    t0 = cpu_time()  
    
    '''
    #bounds and constrains
    #if vis:
    #    print 'Optimizing based on slsqp with bounds and constrains'
    #res = minimize(f_1D_slsqp, prb_map12_init.flatten(),args=(y_dm1, x_dm2), method='SLSQP', bounds=bnds,
    #            constraints = cons, options = ({'maxiter' : max_nfe, 'disp': True}))
    if vis:
        print 'Optimizing based on slsqp with bounds and constrains. Also gradient is provided'
    res = minimize(f_1D_slsqp, prb_map12_init.flatten(), args=(y_dm1, x_dm2), 
                   method='SLSQP', 
                   jac = gradient_f_1D_slsqp,
                   bounds=bnds, constraints = cons, 
                   options = ({'maxiter' : max_nfe, 'disp': True}))
    '''

    #test the gradient
    from scipy.optimize import check_grad
    err = check_grad(f_1D_slsqp, gradient_f_1D_slsqp_3, prb_map12_init.flatten(), y_dm1, x_dm2)

    #err = check_grad(f_1D_slsqp_no_constrains, gradient_f_1D_l_bfgs_b, prb_map12_init.flatten(), y_dm1, x_dm2)


    print 'Check grad: ', err
    
    stop

    
    #bounds and no constrains
    if vis:
        print 'Optimizing based on slsqp with bounds and NO constrains'
    res = minimize(f_1D_slsqp_no_constrains, prb_map12_init.flatten(),args=(y_dm1, x_dm2), 
                   method='SLSQP', 
                   jac = gradient_f_1D_l_bfgs_b, #gradient_f_1D_slsqp,
                   bounds=bnds, 
                   options = ({'maxiter' : max_nfe, 'disp': True}))
    

            
    t_opt = cpu_time() - t0
    
    plsq = res.x
    
    prb_map12 = np.reshape(plsq,(size1,-1))
    
    if vis:        
        print plsq        
        print 'Probability mapping: ',  prb_map12  
        begin_err = f_1D_slsqp_no_constrains(prb_map12_init.flatten(), y_dm1, x_dm2)
        print 'Loss function before optimizing: ', begin_err
        final_err = f_1D_slsqp_no_constrains(plsq, y_dm1, x_dm2)
        print 'Loss function after optimizing: ', final_err 
        print 'Optimizing cpu time: ', t_opt    
    
    return t_opt, prb_map12
#end of SLSQP with bounds and constains
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------  
#function for optimize by L-BFGS-B with/without bounds ONLY, NO constains
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#f = sum((y - f(p,x))^2)
def f_1D_l_bfgs_b(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    #should nomalize the probability to satisfy the constrains of sum(row_i) = 1 for all i
    
    prob_temp = np.reshape(p,(size1,-1))    
    from common_functions import normalize_sum_row_1
    p_temp = normalize_sum_row_1(prob_temp)    
    ad = avg_dis_1D(size1, size2, x_dm2, p_temp.flatten()) 
    f = np.linalg.norm(y_dm1-ad)
    return f

#Gradient of loss function for optimize by L-BFGS-B with no bounds and no constrains
#p: parameter to optimize - prb_map12
#y: source    (dm1)   
#x: target    (dm2)
#f = sum((y - f(p,x))^2)
#return the vector that the number of element is equal to the size of parameter needed to optimized p
def gradient_f_1D_l_bfgs_b(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    p_temp = np.reshape(p,(-1, size2))
    
    from common_functions import normalize_sum_row_1
    p_tmp = normalize_sum_row_1(p_temp) 
    
    y_dm1_tmp = np.reshape(y_dm1,(-1, size1))  
    x_dm2_tmp = np.reshape(x_dm2,(-1, size2))  
    
    jac = np.empty([size1,size2])
    
    for n in np.arange(size1):
        for m in np.arange(size2):
            
            sum1 = 0.
            sum2 = 0.
            for j in np.arange(size1):
                A1, A2 = A_2(j, n, m, p_tmp, x_dm2_tmp, size2)                
                sum1 = sum1 + ((y_dm1_tmp[n,j] - A1) * A2) 
                
                B1, B2 = B_2(j, n, m, p_tmp, x_dm2_tmp, size2)                
                sum2 = sum2 + ((y_dm1_tmp[j,n] - B1) * B2) 
            grd = - 2.*(sum1 + sum2)
            
            jac[n,m] = grd
            
    return jac.flatten()   

   
def scipy_l_bfgs_b(prb_map12_init, y_dm1, x_dm2, max_nfe=50000, vis=False):
    
    from scipy.optimize import minimize
    size1 = np.sqrt(len(y_dm1)) #number of fibers in source tract
    size2 = np.sqrt(len(x_dm2)) #number of fibers in target tract
    
    #bnds = create_bounds(size1*size2)
    
    t0 = cpu_time()  

    '''    
    #with bounds only, and no constrains
    #if vis:
    #    print 'Optimizing based on l_bfgs_b with bounds only, NO constrains'
    #res = minimize(f_1D_l_bfgs_b, prb_map12_init.flatten(),args=(y_dm1, x_dm2), 
    #                method='L-BFGS-B', bounds=bnds, 
    #               options = ({'maxiter' : max_nfe, 'disp': True}))   
    '''
    
    '''
    #with no bounds no constrains   
    if vis:
        print 'Optimizing based on l_bfgs_b with NO bounds and NO constrains'
    res = minimize(f_1D_l_bfgs_b, prb_map12_init.flatten(),args=(y_dm1, x_dm2),
                   method='L-BFGS-B', 
                   options = ({'maxiter' : max_nfe, 'disp': True}))
    '''

    #with no bounds no constrains, with gradient provided use CG
    if vis:
        print 'Optimizing based on CG with NO bounds, NO constrains, and Gradient provided'
    res = minimize(f_1D_l_bfgs_b, prb_map12_init.flatten(),args=(y_dm1, x_dm2), 
                   method='CG',#Newton-CG', 
                   jac = gradient_f_1D_l_bfgs_b, 
                   options = ({'maxiter' : max_nfe, 'disp': True}))
                   
    t_opt = cpu_time() - t0
    
    plsq = res.x
    
    prb_map12 = np.reshape(plsq,(size1,-1))
    
    if vis:
                
        print plsq        
        print 'Probability mapping: ',  prb_map12 
        begin_err = f_1D_l_bfgs_b(prb_map12_init.flatten(), y_dm1, x_dm2)
        print 'Loss function before optimizing: ', begin_err   
        final_err = f_1D_l_bfgs_b(plsq, y_dm1, x_dm2)
        print 'Loss function after optimizing: ', final_err 
        print 'Optimizing cpu time: ', t_opt    
    
    return t_opt, prb_map12
#end of L-BFGS-B with bounds ONLY, NO constains
#------------------------------------------------------------------------------


#these functions just for test    
def constr1(p, y_dm1, x_dm2):
    size1 = np.sqrt(len(y_dm1))
    p_2D = np.reshape(p,(size1,-1))
    sp = [np.sum(p_2D[i,:]) for i in np.arange(len(p_2D))]
    sp = np.array(sp,dtype = float)
    temp = 1
    #print 'sum row', sp
    for i in np.arange(len(p_2D)):
        if sp[i]>1:
            temp = 1 - sp[i]
    
    return temp
    
def constr2(p, y_dm1, x_dm2):
    temp = 1
    for i in np.arange(len(p)):
        if p[i]<0:
            temp = p[i]
    return temp
    

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
                    'outputMap_prob',
                    help='The output file name of the probability mapping')                    

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

map_prob = args.outputMap_prob


vis = False

source_cst = load_tract(s_file,s_ind)

target_cst_ext = load_tract(t_file,t_ind)

print len(source_cst), len(target_cst_ext)

tractography1 = source_cst[-num_pro:]
tractography2 = target_cst_ext[:num_pro]
#tractography2 = target_cst_ext[:num_pro*2]

print "Source", len(tractography1)
print "Target", len(tractography2)


#print "Computing the distance matrices for each tractography."
dm1 = bundles_distances_mam(tractography1, tractography1)
dm2 = bundles_distances_mam(tractography2, tractography2)

size1 = len(tractography1)
size2 = len(tractography2)


#prb_map12_init = init_prb_state(size1, size2)   #normal distribution
prb_map12_init = init_prb_state_1(tractography1,tractography2) #distribution based on distance

print prb_map12_init
init_loss = loss_function(dm1, dm2, prb_map12_init)

print 'init loss', init_loss


if vis:
    ren = fvtk.ren() 
    ren = visualize_tract(ren, tractography1, fvtk.yellow)
    ren = visualize_tract(ren, tractography2, fvtk.blue)
    fvtk.show(ren)
    

y_dm1 = dm1.flatten()
x_dm2 = dm2.flatten()

init_loss = loss_function(dm1, dm2, prb_map12_init)
print 'Loss function before optimizing: ', init_loss

print 'Optimizing ...........................'    

max_nfe = 50000
#from scipy.optimize import leastsq, fmin_slsqp,fmin_bfgs, fmin_powell, fmin_cobyla

#optimizing with SLSPQ  ---------------------
#-----------with bounds and constrains
t_opt, prb_map12 = scipy_slsqp(prb_map12_init.flatten(), y_dm1, x_dm2, max_nfe,vis = True)




'''
#-----------with bounds and no constrains
t_opt, prb_map12 = scipy_slsqp(prb_map12_init.flatten(), y_dm1, x_dm2, max_nfe,vis = True)
from common_functions import normalize_sum_row_1
prb_map12 =normalize_sum_row_1(prb_map12) 
'''

'''
#optimizing with L_BFGS_B---------------------
t_opt, prb_map12 = scipy_l_bfgs_b(prb_map12_init.flatten(), y_dm1, x_dm2, max_nfe,vis = True)
from common_functions import normalize_sum_row_1
prb_map12 =normalize_sum_row_1(prb_map12) 
'''



print 'Time :', t_opt
print 'Map : ', prb_map12




#test the result 
for i in np.arange(size1):
    print 'sum row ', i , np.sum(prb_map12[i,:])
    
final_loss = loss_function(dm1, dm2, prb_map12)
print 'Loss function after optimizing: ', final_loss



'''
#leastsq
plsq = leastsq(residuals, prb_map12_init, args=(y_dm1, x_dm2))
#plsq = fmin_bfgs(residuals, prb_map12_init, args=(y_dm1, x_dm2))
#prb_map12 = plsq[0]
print prb_map12
final_loss = loss_function_1D(y_dm1, x_dm2, prb_map12)
print 'final loss', final_loss

mapp = np.reshape(prb_map12,(size1,-1))
print mapp
'''

'''
#cobyla
t0 = time.time()
plsq = fmin_cobyla(f_1D, prb_map12_init.flatten(), cons=[constr_temp1], args=(y_dm1, x_dm2),  rhobeg=0.20, rhoend=0.001, maxfun = max_nfe)
t_opt = time.time() - t0
print 'Optimizing time', t_opt
#cons=[constr1, constr2, constr3]
print plsq[0], plsq[15]
'''


'''
#test optimize least square
x = np.arange(0,6e-2,6e-2/30)
A,k,theta = 10, 1.0/3e-2, np.pi/6
y_true = A*np.sin(2*np.pi*k*x+theta)
y_meas = y_true + 2*np.random.randn(len(x))

def residuals(p, y, x):
     A,k,theta = p
     err = y-A*np.sin(2*np.pi*k*x+theta)
     return err

def peval(x, p):
     return p[0]*np.sin(2*np.pi*p[1]*x+p[2])

p0 = [8, 1/2.3e-2, np.pi/3]
print np.array(p0)
from scipy.optimize import leastsq
plsq = leastsq(residuals, p0, args=(y_meas, x))
print plsq[0]

print np.array([A, k, theta])

import matplotlib.pyplot as plt
plt.plot(x,peval(x,plsq[0]),x,y_meas,'o',x,y_true)
plt.title('Least-squares fit to noisy data')
plt.legend(['Fit', 'Noisy', 'True'])
plt.show()
'''        