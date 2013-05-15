# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as spio
import mne
import pickle
import os

from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from scipy.signal import resample

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


def select_trial_label(event_id, trigger_channels, coi, input_filename, output_filename=None):    
    
    mne.set_log_level('WARNING')
    print "loading dataset " + input_filename
    raw = mne.fiff.Raw(input_filename)
    print
    print "Extract events from channels",
    print trigger_channels
    channels_num = len(trigger_channels)
    triggers = []
    for trigger_channel in trigger_channels:
        print trigger_channel,
        tmp = mne.find_events(raw, stim_channel=trigger_channel)[:,0] 
        triggers.append(tmp)

    trigger_times = np.unique(np.concatenate(triggers)) #concatenate trigger event and order them without repetitions
    threshold_faulty_hw = 5
    print "Fix triggers times occurring <"+str(threshold_faulty_hw)+" timesteps from the previous trigger (due to faulty hardware)"
    idx_triggers_to_fix = np.where(np.diff(trigger_times) < threshold_faulty_hw)[0]
    fix = dict(zip(trigger_times[idx_triggers_to_fix+1], trigger_times[idx_triggers_to_fix]))
    print "Actual triggers times to fix due to faulty HW:", fix
    
    trigger_times = np.concatenate([[trigger_times[0]], trigger_times[1:][np.diff(trigger_times) >= threshold_faulty_hw]])

    for ttf in fix.keys():
        for trigger in triggers:
            trigger[trigger==ttf] = fix[ttf] # this is the actual fix

    # some checks:
    tmp = np.unique(np.concatenate(triggers))
    assert((tmp == trigger_times).all())

    print "Binary to decimal conversion of triggers."
    trigger_decimal = np.zeros(trigger_times.size, dtype=np.int)
    for i, t in enumerate(trigger_times):
        for bit, trigger in enumerate(triggers):
            if (trigger == t).any():
                trigger_decimal[i] += 2**(channels_num - bit - 1) # right lsb
                
    
    trigger_decimal_unique = np.unique(trigger_decimal)
    summary = [(trigger_decimal==v).sum() for v in trigger_decimal_unique]
    print "Summary:"
    print trigger_decimal_unique
    print summary

    triggers_of_interest = np.concatenate(coi)
    print "triggers_of_interest:", triggers_of_interest
    print "corresponding to:"
    for toi in triggers_of_interest:
        print '\t', event_id[toi]

    triggers_mask = np.logical_or.reduce([trigger_decimal==v for v in triggers_of_interest])
    
    if output_filename == None:
        filename_save = input_filename.split('.')[0]+'_trial_time.pickle'
    else:
        filename_save = os.path.abspath(output_filename)
    
    print "Saving to:", filename_save
    pickle.dump({'trigger_times': trigger_times[triggers_mask],
                 'trigger_decimal': trigger_decimal[triggers_mask],
                 'coi': coi,
                 }, open(filename_save, 'w'),
                protocol = pickle.HIGHEST_PROTOCOL)
    
    return filename_save
       



def create_3Dmatrix(epochs_dim, ch_type, input_filename, input_filename_trial, output_filename=None):
    
    mne.set_log_level('WARNING')
    raw = mne.fiff.Raw(input_filename)

    datatrial = pickle.load(open(input_filename_trial))
    trigger_times = datatrial['trigger_times']
    trigger_decimal = datatrial['trigger_decimal']
    coi = datatrial['coi']
    
    print
    print "Get the indexes of just the MEG channels"
    picks = mne.fiff.pick_types(raw.info, meg=ch_type['meg'], eeg=ch_type['eeg'], stim=ch_type['stim'], exclude=ch_type['exclude']) #only meg channels
    
    events = np.vstack([trigger_times, np.zeros(len(trigger_times), dtype=np.int), trigger_decimal]).T

    print "Extracting Epochs for each condition for the contrast."
    baseline = (None, 0) # means from the first instant to t = 0
    reject = {}
    tmin = epochs_dim[0]
    tmax = epochs_dim[1]
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=baseline, preload=False, reject=reject)

    X = epochs.get_data()
    y = epochs.events[:,2]
    label = np.zeros(len(y))
    for i, yi in enumerate(y):
        if np.sum(yi == coi[0]):
            label[i] = 1
            
    if output_filename == None:
	filename_save = input_filename_trial.split('.')[0]+'_3D.pickle'
    else:
	filename_save = os.path.abspath(output_filename)

    print "Saving to:", filename_save
    pickle.dump({'X': X,
	         'y': label,
	         'tmin': tmin,
	         'sfreq': raw.info['sfreq']
	         }, open(filename_save, 'w'),
	        protocol = pickle.HIGHEST_PROTOCOL)
                
    return filename_save
    

"""
Preprocessing steps of the Body-Face-House dataset of Thomas Hartmann.
BandPass filtering 1 --> 100 Hz
Resampling to 200 Hz
Windowing 0 --> 500 ms
Grand z-scoring
"""

def encoding(input_filename, lb, ub, new_freq, t_start, t_stop, output_filename=None):    
    #load from pickle file

    data = pickle.load(open(input_filename))
    X = data['X']
    y = data['y']
    tmin = data['tmin']
    sfreq = data['sfreq']
    
    #session = np.concatenate([[i]*ss for i, ss in enumerate(session_size)])
    #XX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

#    print "Bandpass filtering [%s,%s]Hz" % (lb, ub)
#    XX_filtered = []
#    for i in range(X.shape[0]):
#        print i,
#        stdout.flush()
#        T = TimeSeries(data=X[i,:,:], sampling_rate=sfreq, t0=tmin, time_unit='s')
#        F = FilterAnalyzer(T, ub=ub, lb=lb)
#        XX_filtered.append(F.filtered_boxcar)
#
#    XX_filtered = np.array(XX_filtered).swapaxes(1,2)
#    print
        
    XX_filtered = X    
    to_frequency = new_freq # Hz
    if to_frequency != sfreq:
        print "Resampling to", to_frequency, "Hz."
        XX_filtered_resampled = resample(XX_filtered, num=to_frequency, axis = 2)  #ATT num=to_freq only because the signal is 
    else:
        XX_filtered_resampled = XX_filtered

    print "windowing."
    #t_start = 0.0
    #t_stop = 0.5
    idx_start = int((t_start-tmin) * to_frequency)
    idx_stop = int((t_stop-tmin) * to_frequency)
    XX_filt_res_wind = XX_filtered_resampled[:,:,idx_start:idx_stop]
    # XX_filt_res_wind = XX_filtered_resampled
    
    print "Grand z-scoring."
    XX_filt_res_wind_z = (XX_filt_res_wind - XX_filt_res_wind.mean()) / XX_filt_res_wind.std()
    
    if output_filename == None:
        filename_save = input_filename.split('.')[0]+'_encod.pickle'
    else:
        filename_save = os.path.abspath(output_filename)
    
    print "Saving to:", filename_save
    pickle.dump({'X': XX_filt_res_wind_z,
                 'y': y
                 }, open(filename_save, 'w'),
                protocol = pickle.HIGHEST_PROTOCOL)
                
    return filename_save
    
    """
Classification step of the Body-Face-House dataset of Thomas Hartmann.

"""
def decoding(X, y, cv, clf):
    """Plain classification based decoding.
    """
    skf = cross_validation.StratifiedKFold(y, n_folds=cv)
    cm = []
    for i, (train, test) in enumerate(skf):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        cm.append(confusion_matrix(y[test], y_pred))
        #print i, ')', cm[-1].diagonal().sum() / float(cm[-1].sum())
        #print cm[-1]
    
   
    #print "Average accuracy:",
    CM = np.sum(cm, 0)
    return CM.diagonal().sum() / float(CM.sum())
  
###########################################################

def classification(cv, input_filename): 
    from NILabMNELibrary import decoding
    #load from pickle file    
    
    print "Loading data."
    data = pickle.load(open(input_filename))
    X = data['X']
    y = data['y']
    
    clf = LogisticRegression(penalty='l2')
    # clf = SVC()    
    # clf = LDA()
    acc_Ch = np.zeros(X.shape[1]) 
    for i in np.arange(X.shape[1]):
        acc_Ch[i] = decoding(X[:,i,:], y, cv, clf)
    
#    accs = np.zeros(len(np.arange(1,20,2)))
#    for i, n_ch in enumerate(np.arange(1,20,2)):
#        ch_sel = np.random.rand(n_ch)
#        ch_sel = np.round(ch_sel*305).astype(int)
#        X_appo = X[:,ch_sel,:]
#        acc = decoding(X_appo.reshape(X_appo.shape[0], X_appo.shape[1]*X_appo.shape[2]), y, cv, clf)
#        accs[i] = np.mean(acc)
#    
    
    acc_tot = decoding(X.reshape(X.shape[0], X.shape[1] * X.shape[2]), y, cv, clf)
    acc_Ch_tot = np.mean(acc_Ch)
    
    print "Accuracy: " + str(acc_tot)
    print "Mean channel accuracy: " + str(acc_Ch_tot)
    
    return acc_tot, acc_Ch_tot
