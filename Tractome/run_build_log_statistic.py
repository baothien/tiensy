# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:44:37 2014

@author: bao
"""

in_dir = '/home/bao/tiensy/APSS_Sarubbo/'
sub = ['COAL_270845_300914', 'MALO_250448_151014']
sub_short = ['CA', 'MA']
tract = ['AF_left','AF_right','IFOF_left', 'SLF_Ind_Ant_left', 'SLF_Ind_Post_left', 'SLF_Ind_Ant_right', 'SLF_Ind_Post_right' ]
s_idx = 1 #1,5,6 
#s_idx = 0 #0,2,3,4 
t_idx = 6

print '-----------------------------------------------------------------------------------------'
fname = 'build_log_statistic.py'
tracto_fname = in_dir + sub[s_idx] + '/TRK/Tractography/'+ sub[s_idx] + '_3M_apss.trk'
seg_fname = in_dir + sub[s_idx] + '/TRK/Dissection/Tractome/'+ sub_short[s_idx] + '_' + tract[t_idx] + '_scene.trk.trk'
log_fname = in_dir + sub[s_idx] + '/TRK/Dissection/Tractome/'+ sub_short[s_idx] + '_' + tract[t_idx] + '_scene.seg.seg'

import sys
sys.argv = [fname, tracto_fname, seg_fname, log_fname]
execfile(fname)