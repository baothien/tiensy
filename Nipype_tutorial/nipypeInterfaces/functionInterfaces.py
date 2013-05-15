# -*- coding: utf-8 -*-

import nipype.interfaces.utility as util

from NILabMNELibrary import select_trial_label, create_3Dmatrix, encoding, classification

###### INTERFACE DEFINITION #######
SelectTrialLabel = util.Function(input_names=['event_id', 'trigger_channels', 'coi', 'input_filename', 'output_filename'], 
                                             output_names=['trial_segmentation_file'],
                                             function=select_trial_label, 
                                             imports=['import numpy as np',
                                                      'import mne',
                                                      'import pickle',
                                                      'import os'])


Create3DMatrix = util.Function(input_names=['epochs_dim', 'ch_type', 'input_filename', 'input_filename_trial','output_filename'],
                                          output_names=['output_3Dmatrix'],
                                          function=create_3Dmatrix, 
                                          imports=['import numpy as np',
                                                   'import mne',
                                                   'import pickle',
                                                   'import os'])
                                                   

Encoding = util.Function(input_names=['lb', 'ub', 'new_freq', 't_start', 't_stop', 'input_filename', 'output_filename'],
                                          output_names=['output_encoding'],
                                          function=encoding, 
                                          imports=['import numpy as np',
                                                   'import scipy.io as spio',
                                                   'from nitime.timeseries import TimeSeries',
                                                   'from nitime.analysis import FilterAnalyzer',
                                                   'from scipy.signal import resample',
                                                   'import pickle',
                                                   'import os'])
                                                                                                    
                                                   
Classification = util.Function(input_names=['cv','input_filename'],
                                         output_names=['accuracy','mean_ch_accuracy'],
                                         function=classification,
                                         imports=['import numpy as np',
                                                  'import pickle',
                                                  'from sklearn.linear_model import LogisticRegression',
                                                  'from sklearn.lda import LDA',
                                                  'from sklearn.svm import SVC',
                                                  'from sklearn import cross_validation',
                                                  'from sklearn.metrics import confusion_matrix'])