# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:25:47 2013

@author: nilab
"""

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

from NILabMNELibrary import select_trial_label, create_3Dmatrix, encoding, classification

from nipype import logging
iflogger = logging.getLogger('interface')

import nibabel as nb
import numpy as np
from sys import stdout
import os

class SelectTrialLabelInputSpec(BaseInterfaceInputSpec):
    event_id = traits.Dict(desc="id stimuli", mandatory=True)
    trigger_channels = traits.List(desc="Channels carrying information about stimuli presentation.",mandatory=True)
    coi = traits.Array(desc="Classes of stimuli to be classified",mandatory=True)
    input_filename = File(exists=True,desc="FIF file to be parsed",mandatory=True)
    output_filename = File(exists=False,desc="Output file name",mandatory=False)
  
class SelectTrialLabelOutputSpec(TraitedSpec):
    trial_segmentation_file=File(exists=True ,desc="Output file name")
    

class SelectTrialLabel(BaseInterface):
    input_spec = SelectTrialLabelInputSpec
    output_spec = SelectTrialLabelOutputSpec
    
    def _run_interface(self, runtime):
        self._out_file = select_trial_label(self.inputs.event_id,self.inputs.trigger_channels,self.inputs.coi,self.inputs.input_filename,self.inputs.output_filename)
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["trial_segmentation_file"] = os.path.abspath(self._out_file)
        return outputs

###

class Create3DMatrixInputSpec(BaseInterfaceInputSpec):
    epochs_dim=traits.Array(desc="Inital and final time of the epochs",mandatory=True)
    ch_type=traits.Dict(desc="Type of the channel. E.g.: MEG", mandatory=True)
    input_filename = File(exists=True,desc="FIF file to be parsed",mandatory=True)
    input_filename_trial = File(exists=True,desc="File containing Trials",mandatory=True)
    output_filename = File(exists=False,desc="Output file name",mandatory=False)
    
  
class Create3DMatrixOutputSpec(TraitedSpec):
    output_3Dmatrix=File(exists=True ,desc="Output file name")
    

class Create3DMatrix(BaseInterface):
    input_spec = Create3DMatrixInputSpec
    output_spec = Create3DMatrixOutputSpec
    
    def _run_interface(self, runtime):
        self._out_file = create_3Dmatrix(self.inputs.epochs_dim, self.inputs.ch_type, self.inputs.input_filename, self.inputs.input_filename_trial, self.inputs.output_filename)
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_3Dmatrix"] = os.path.abspath(self._out_file)
        return outputs

###

class EncodingInputSpec(BaseInterfaceInputSpec):

    input_filename = File(exists=True,desc="Pickle File to be encoded",mandatory=True)
    lb = traits.Float(desc="Lower bound of the bandwith for the bandpass filtering",mandatory=True)
    ub = traits.Float(desc="Upper bound of the bandwith for the bandpass filtering",mandatory=True)
    new_freq = traits.Float(desc="Sampling frequency for resampling",mandatory=True)
    t_start = traits.Float(desc="Starting time point",mandatory=True)
    t_stop = traits.Float(desc="Ending time point",mandatory=True)
    output_filename = File(exists=False,desc="Output file name",mandatory=False)
    
  
class EncodingOutputSpec(TraitedSpec):
    output_encoding=File(exists=True ,desc="Output file name")
    

class Encoding(BaseInterface):
    input_spec = EncodingInputSpec
    output_spec = EncodingOutputSpec
    
    def _run_interface(self, runtime):
        self._out_file = encoding(self.inputs.input_filename, self.inputs.lb, self.inputs.ub, self.inputs.new_freq, self.inputs.t_start, self.inputs.t_stop, self.inputs.output_filename)
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_encoding"] = os.path.abspath(self._out_file)
        return outputs


###

class ClassificationInputSpec(BaseInterfaceInputSpec):
    
    cv=traits.Int(desc="Number of folds for classification",mandatory=True)
    input_filename = File(exists=True,desc="Pickle File for the classification",mandatory=True)
  
class ClassificationOutputSpec(TraitedSpec):
     
    output_variables=traits.Array(exists=True ,desc="Output variables")
    

class Classification(BaseInterface):
    input_spec = ClassificationInputSpec
    output_spec = ClassificationOutputSpec
    
    def _run_interface(self, runtime):
        self._out_vars = classification(self.inputs.cv,self.inputs.input_filename)
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_variables"] = self._out_vars
        return outputs

