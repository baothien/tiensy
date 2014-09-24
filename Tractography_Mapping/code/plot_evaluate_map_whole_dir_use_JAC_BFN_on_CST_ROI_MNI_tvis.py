# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:18:56 2014

@author: bao
"""


import matplotlib.pyplot as plt
import numpy as np

y_loss_left = { '212_202': np.array([152.90,	   120.10,	104.35,	81.10,	77.46	,	75.73,	74.64], dtype=np.float32),
                '212_204': np.array([160.34,	   129.47,	108.00,	82.57	,	75.38	,	74.33	,	73.99], dtype=np.float32),
                '212_209': np.array([175.90,    133.16,	118.11,	90.97	,	86.12	,	84.29	,	83.91], dtype=np.float32),
                '202_212': np.array([448.64,    354.61,	318.01,	276.97,	244.21,	223.16,	217.38], dtype=np.float32),
                '202_204': np.array([245.38,    199.57,	170.84,	138.06,	118.11,	114.15,	112.83], dtype=np.float32),
                '202_209': np.array([362.08,    312.11,	277.50,	239.51,	216.29,	202.46,	198.25], dtype=np.float32),
                '204_212': np.array([556.25,    474.89,	433.69,	396.14,	377.71,	342.32,	334.68], dtype=np.float32),
                '204_202': np.array([364.27,    296.30,	262.80,	229.61,	208.43,	192.63,	185.94], dtype=np.float32),
                '204_209': np.array([476.83,    425.78,	395.61,	359.16,	339.43,	320.88,	309.80], dtype=np.float32),
                '209_212': np.array([221.30,	   165.95,	143.65,	115.36,	111.93,	111.06,	110.62], dtype=np.float32),
                '209_202': np.array([175.38,	   140.86,	127.96,	94.53	,	93.16	,	93.09	,	92.91], dtype=np.float32),
                '209_204': np.array([207.64,	   156.68,	129.55,	99.05	,	94.77	,	93.48	,     92.76	], dtype=np.float32) 
         }
         
y_loss_right = { '206_204': np.array([227.35,	178.37,	147.06,	126.13,	121.22,	120.71,	120.54], dtype=np.float32),
                '206_212': np.array([327.18, 	254.52,	199.29,	157.90,	149.14,	148.52,	148.22], dtype=np.float32),
                '206_205': np.array([335.80, 	274.18,	236.02,	204.12,	196.22,	193.51,	192.62], dtype=np.float32),
                '204_206': np.array([224.26,    	189.19,	176.71,	145.91,	139.74,	135.42,	134.30], dtype=np.float32),
                '204_212': np.array([403.33,    	324.19,	280.84,	222.60,	198.74,	182.36,	176.02], dtype=np.float32),
                '204_205': np.array([397.33,    	304.96,	270.75,	227.41,	207.82,	201.84,	199.44], dtype=np.float32),
                '212_206': np.array([137.87,    	110.26,	95.72	,    78.33	,	76.42,	75.73	,	74.13], dtype=np.float32),
                '212_204': np.array([170.20,    	128.50,	111.23,	91.43	,	85.30,	85.20	,	84.41], dtype=np.float32),
                '212_205': np.array([241.24,    	178.86,	153.84,	128.08,	120.68,	117.42,	112.12], dtype=np.float32),
                '205_206': np.array([137.14,    	99.35,	85.66,	74.04,	68.85,	64.50,	62.84], dtype=np.float32),
                '205_204': np.array([128.32,    	100.07,	91.28,	67.92,	65.72,	64.77,	64.73], dtype=np.float32),
                '205_212': np.array([190.38,    	132.69,	109.82,	85.61,	74.59,	70.06,	66.95], dtype=np.float32)
         }

y_jac_left = { '212_202': np.array([0.07,	0.62,	0.63,	0.62,	0.59,	0.60,	0.59], dtype=np.float32),
                '212_204': np.array([0.12,	0.39,	0.46,	0.42,	0.46,	0.45,	0.42], dtype=np.float32),
                '212_209': np.array([0.09,	0.75,	0.76,	0.88,	0.84,	0.87,	0.85], dtype=np.float32),
                '202_212': np.array([0.07,	0.54,	0.56,	0.55,	0.54,	0.53,	0.53], dtype=np.float32),
                '202_204': np.array([0.11,	0.71,	0.70,	0.72,	0.76,	0.76,	0.76], dtype=np.float32),
                '202_209': np.array([0.02,	0.38,	0.39,	0.36,	0.34,	0.33,	0.28], dtype=np.float32),
                '204_212': np.array([0.12,	0.66,	0.62,	0.56,	0.52,	0.53,	0.52], dtype=np.float32),
                '204_202': np.array([0.11,	0.77,	0.78,	0.77,	0.81,	0.81,	0.82], dtype=np.float32),
                '204_209': np.array([0.07,	0.49,	0.49,	0.50,	0.49,	0.45,	0.42], dtype=np.float32),
                '209_212': np.array([0.09,	0.70,	0.67,	0.75,	0.67,	0.67,	0.66], dtype=np.float32),
                '209_202': np.array([0.02,	0.51,	0.57,	0.63,	0.63,	0.63,	0.63], dtype=np.float32),
                '209_204': np.array([0.07,	0.23,	0.22,	0.27,	0.26,	0.26,	0.30], dtype=np.float32)
         }
         
y_bfn_left = {  '212_202': np.array([0.58,	0.12,	0.13,	0.13,	0.15,	0.16,	0.16], dtype=np.float32),
                '212_204': np.array([0.59,	0.23, 0.21,	0.22,	0.18,	0.19,	0.19], dtype=np.float32),
                '212_209': np.array([0.84,	0.14, 0.14,	0.08,	0.10,	0.08,	0.09], dtype=np.float32),
                '202_212': np.array([0.93,	0.30,	0.29,	0.30,	0.32,	0.33,	0.33], dtype=np.float32),
                '202_204': np.array([0.89,	0.16,	0.19,	0.20,	0.16,	0.16,	0.16], dtype=np.float32),
                '202_209': np.array([0.98,	0.60,	0.61,	0.64,	0.66,	0.61,	0.68], dtype=np.float32),
                '204_212': np.array([0.88,	0.17,	0.20,	0.23,	0.21,	0.20,	0.19], dtype=np.float32),
                '204_202': np.array([0.83,	0.12,	0.14,	0.14,	0.11,	0.12,	0.11], dtype=np.float32), 
                '204_209': np.array([0.93,	0.27,	0.32,	0.32,	0.36,	0.42,	0.41], dtype=np.float32),
                '209_212': np.array([0.91,	0.17,	0.20,	0.13,	0.16,	0.15,	0.16], dtype=np.float32),
                '209_202': np.array([0.65,	0.15,	0.14,	0.11,	0.12,	0.12,	0.12], dtype=np.float32),
                '209_204': np.array([0.67,	0.25,	0.28,	0.27,	0.29,	0.28,	0.26], dtype=np.float32)
 
         }
         
y_jac_right = { '206_204': np.array([0.22,	0.55,	0.57,	0.54,	0.53,	0.49,	0.47], dtype=np.float32),
                '206_212': np.array([0.13,	0.69,	0.66,	0.67,	0.76,	0.76,	0.75], dtype=np.float32),
                '206_205': np.array([0.15,	0.62,	0.63,	0.52,	0.57,	0.65,	0.58], dtype=np.float32),
                '204_206': np.array([0.22,	0.69,	0.68,	0.66,	0.64,	0.67,	0.66], dtype=np.float32),
                '204_212': np.array([0.20,	0.69,	0.63,	0.57,	0.54,	0.49,	0.47], dtype=np.float32),
                '204_205': np.array([0.18,	0.47, 0.43,	0.40,	0.39,	0.37,	0.38], dtype=np.float32),
                '212_206': np.array([0.13,	0.58, 0.57,	0.57,	0.54,	0.54,	0.55], dtype=np.float32),
                '212_204': np.array([0.20,	0.55,	0.51,	0.51,	0.60,	0.61, 0.63], dtype=np.float32),
                '212_205': np.array([0.10,	0.50,	0.45,	0.44,	0.35,	0.27,	0.27], dtype=np.float32),
                '205_206': np.array([0.15,	0.81,	0.89,	0.88,	0.91,	0.85,	0.82], dtype=np.float32),
                '205_204': np.array([0.18,	0.52,	0.48,	0.51,	0.59,	0.60,	0.59], dtype=np.float32),
                '205_212': np.array([0.10,	0.77,	0.87,	0.93,	0.93,	0.90,	0.90], dtype=np.float32)
 
         }
         

y_bfn_right = { '206_204': np.array([0.65,	0.27,	0.27,	0.25,	0.25,	0.25,	0.24], dtype=np.float32),
                '206_212': np.array([0.87,	0.17,	0.22,	0.20,	0.15,	0.14,	0.14], dtype=np.float32),
                '206_205': np.array([0.85,	0.38,	0.37,	0.48,	0.40,	0.28,	0.34], dtype=np.float32),
                '204_206': np.array([0.78,	0.20,	0.21,	0.26,	0.26,	0.23,	0.24], dtype=np.float32),
                '204_212': np.array([0.80,	0.20,	0.25,	0.33,	0.33,	0.34,	0.33], dtype=np.float32),
                '204_205': np.array([0.82,	0.47,	0.55,	0.59,	0.61,	0.63,	0.61], dtype=np.float32),
                '212_206': np.array([0.77,	0.25,	0.28,	0.26,	0.29,	0.28,	0.30], dtype=np.float32),
                '212_204': np.array([0.58,	0.20,	0.22,	0.20,	0.16,	0.16,	0.16], dtype=np.float32),
                '212_205': np.array([0.90,	0.44,	0.54,	0.56,	0.65,	0.73,	0.73], dtype=np.float32),
                '205_206': np.array([0.50,	0.07,	0.03, 0.04,	0.03,	0.06,	0.07], dtype=np.float32),
                '205_204': np.array([0.40,	0.18, 0.19,	0.17,	0.13,	0.13,	0.13], dtype=np.float32),
                '205_212': np.array([0.60,	0.08,	0.05,	0.03,	0.03,	0.04,	0.04], dtype=np.float32)

         }
         



#for CST_ROI_L
source_ids = [209]#[212, 202, 204, 209]
target_ids = [202, 204, 212 ]#209, 212]#, 212]#, 204, 202]#, 202, 209]#, 212, 209]# [212, 202, 204, 209]

'''
#for CST_ROI_R
source_ids = [205]#, 204, 212, 205]
target_ids = [ 204, 206, 212]#[206, 204, 212, 205]
'''

marker = ['-','--','-.', ':']
x = [0, 100, 200, 400, 600, 800, 1000]
mk = 0

y_all = y_loss_left
#y_all = y_loss_right
#y1_all = y_jac_left
#y2_all = y_bfn_left
x = x[0:]
for s_id in np.arange(len(source_ids)):
    #print "------------------------------------------"
    print source_ids[s_id]    
    for t_id in np.arange(len(target_ids)):        
        if (target_ids[t_id] != source_ids[s_id]):                
            source = str(source_ids[s_id])
            target = str(target_ids[t_id])
            
            #y1 = y1_all[source+'_'+target][1:]
            #y2 = y2_all[source+'_'+target][1:]
            
            #print y1, y2
            mk = t_id
            #plt.plot(x, y1, marker[mk],color ='black', label = 'JAC_' + source+'_'+target, markersize = 8,linewidth=2.)            
            #plt.plot(x, y2, marker[mk],color ='black', label = 'BFN_' +  source+'_'+target, markersize = 8,linewidth=2.)            
            
            #plt.plot(y2, y1, marker[mk],color ='black', label = 'JAC_' + source+'_'+target, markersize = 8,linewidth=2.)            
            
            #110 is len of 205 CST_R_ext - for normalization
            #len_cst = 110
            #145 is len of 209 CST_L_ext - for normalization
            len_cst = 145
            y = 1./len_cst * y_all[source+'_'+target][0:] 
            from common_functions import plot_smooth_label
            
            plot_smooth_label(plt, np.array(x,int), y, marker[mk], source+'_'+target, True)#False) 
            #plt.plot(x, y, 'g-', color ='black', label = source+'_'+target, markersize = 8,linewidth=2.)            
            #plt.plot(x, y, 'g-', label = '201', markersize = 1.2)            
            #plt.plot(x, y,linestyle='-','rD', label = str(data_id), markersize = 1.8)            
            #plt.plot(x, y, marker[mk],color ='black', label = source+'_'+target, markersize = 8,linewidth=2.)            
            #plt.xlabel('scale: ' + str(1./ward.height_[len(ward.height_)-1]))
            #markers_on = np.linspace(0, 1, ward.height_[len(ward.height_)-1])+0.5/ ward.height_[len(ward.height_)-1]
            #y_markers_on = [R_function(ward.R_alpha_,ward.height_[len(ward.height_)-1],x_i+0.5/ ward.height_[len(ward.height_)-1]) for x_i in markers_on]
            #plt.plot(markers_on, y_markers_on, 'rD')
plt.legend(loc='up right')    
plt.xlabel('annealing number', fontsize=17)
plt.ylabel('normalized loss function value', fontsize=17)
#plt.title('Choosing scale based on goodness of a cut')            
plt.show()
plt.savefig('loss_anneal_' + str(source_ids[0]) + '_all_R')