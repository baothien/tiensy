# -*- coding: utf-8 -*-
"""
workFlow

@author: nilab
"""
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from functionInterfaces import SelectTrialLabel, Create3DMatrix, Encoding, Classification

                                         
###### WORKFLOW DEFINITION #######
aWf=pe.Workflow(name="aWf")
aWf.base_dir="/home/nilab/nipype_tutorial/theFunctions/results"

###### NODE DEFINITION #######
select_trial_label_node = pe.Node(SelectTrialLabel, name="select_trial_label_node")
create_3Dmatrix_node = pe.Node(Create3DMatrix, name="create_3Dmatrix_node")
encoding_node = pe.Node(Encoding, name='encoding_node')
classification_node = pe.Node(Classification, name='classification_node')

###### INPUT NODE DEFINITION #######
#inputs: select_trial_label_node
select_trial_label_node.inputs.input_filename='/home/nilab/nipype_tutorial/pipelineMEG1/data/121120Adriano01.fif'
select_trial_label_node.inputs.event_id={1: ('left', 'body'),
                                         2: ('left', 'face', 'female'),
                                         3: ('left', 'face', 'male'),
                                         4: ('left', 'house'),
                                         5: ('right', 'body'),
                                         6: ('right', 'face', 'female'),
                                         7: ('right', 'face','male'),
                                         8: ('right', 'house'),
                                         9: ('left', 'body', 'flipped'),
                                         10: ('left', 'face', 'female', 'flipped'),
                                         11: ('left', 'face', 'male', 'flipped'),
                                         12: ('left', 'house', 'flipped'),
                                         13: ('right', 'body', 'flipped'),
                                         14: ('right', 'face', 'female', 'flipped'),
                                         15: ('right', 'face', 'male', 'flipped'),
                                         16: ('right', 'house', 'flipped'),
                                         17: ('fixation'),
                                         18: ('correct answer'),
                                         19: ('incorrect answer'),
                                         }
select_trial_label_node.inputs.trigger_channels=['STI001', 'STI002', 'STI003', 'STI004', 'STI005']
select_trial_label_node.inputs.coi =  [[1,2,3,4], [5,6,7,8]]
select_trial_label_node.inputs.output_filename = 'TrialTime'

#inputs: create_3Dmatrix_node
tmin=-0.250 
tmax=0.750 
create_3Dmatrix_node.inputs.epochs_dim = [tmin, tmax]
create_3Dmatrix_node.inputs.ch_type = {'meg': True,
        					     'eeg': False,
        					     'eog': False,
        					     'ecg': False,
        					     'emg': False,
        					     'stim': False,
        					     'exclude': [],
        					     'selection': [],
        					     }
create_3Dmatrix_node.inputs.input_filename = select_trial_label_node.inputs.input_filename
create_3Dmatrix_node.inputs.output_filename = '3dMatrix'

#inputs: encoding_node
encoding_node.inputs.lb = 1
encoding_node.inputs.ub = 100
encoding_node.inputs.new_freq = 200
encoding_node.inputs.t_start = 0.0
encoding_node.inputs.t_stop = 0.5
encoding_node.inputs.output_filename = 'Encoding'

#inputs: classification_node
classification_node.inputs.cv = 5

###### NODE CONNECTIONS #######
aWf.connect(select_trial_label_node,"trial_segmentation_file",create_3Dmatrix_node,"input_filename_trial")
aWf.connect(create_3Dmatrix_node, 'output_3Dmatrix', encoding_node, 'input_filename')
aWf.connect(encoding_node, 'output_encoding', classification_node, 'input_filename')

###### GRAPH and RUN #######
aWf.write_graph()
aWf.run()
