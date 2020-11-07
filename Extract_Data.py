#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
import scipy.io
import gc as garbageCollector
import mne

def import_signals(file_name):
    return scipy.io.loadmat(file_name)['val']

def loadSignals(recordName, dataPath):
    signals = scipy.io.loadmat(dataPath + recordName + '_LFP.mat')
    garbageCollector.collect()
    return signals


def load_time(recordName, dataPath):
    print("Time path", dataPath+ recordName + '_TIME.mat')
    time = scipy.io.loadmat(dataPath + recordName + '_TIME.mat')['INT_TIME'][0]
    garbageCollector.collect()
    return time


def convert_signal_to_array(x, time, downstream=True):
    print(time)
    data_map = {}
    desired_keys = ["PrL_Cx", "Md_Thal", "IL_Cx", "BLA", "Acb_Sh", "Acb_Core", "mSNC", "mDHip", "lSNC", "lDHip", "L_VTA", "R_VTA"]
    for k in desired_keys:
        data_map[k] = []
    
    for key in x:
        if "_" in key and key[-1]!="_":
            newkey = key[:key.rindex("_")]
            if newkey in data_map:
                data_map[newkey].append(x[key])
    
    # add right and left VTA
    data_map["VTA"] = data_map["L_VTA"] + data_map["R_VTA"]
    del data_map["L_VTA"]
    del data_map["R_VTA"]
    print("VTA SHAPE", len(data_map["VTA"]))
    
    # average the lsits
    for key in data_map:
        print(np.array(data_map[key]).shape)
        data_map[key] = np.mean(np.array(data_map[key]), axis=0).ravel()

    # find starts and stops
    fs=1000
    print(time)
    starts_and_stops=np.r_[0:300*fs,time[0]*fs:(time[0]+time[1])*fs, time[2]*fs:(time[2]+time[3])*fs]

    # compile into a single array
    if downstream:
        ans= np.array([data_map[k][starts_and_stops] for k in data_map]) 
    else:
        ans= np.array([data_map[k][:] for k in data_map]) 
    print("The shape of the array is", ans.shape)
    garbageCollector.collect()
    return ans

def extractWholeRecord(recordName,
                       dataPath, downstream=True):

    signals = loadSignals(recordName, dataPath + os.sep + "LFP_Data" + os.sep)
    
    time = load_time(recordName, dataPath + os.sep + "INT_TIME" + os.sep)
    
    signals = convert_signal_to_array(signals, time, downstream=downstream)
    
    #4th order butterworth
    signals = mne.filter.filter_data(data=signals, sfreq=1000, l_freq=None, h_freq=55, method='fir', fir_window='hamming')    
    
    ## 1000 -> 250Hz downsample
    # removing the downsample
    signals = signals[:, 0::1]

    garbageCollector.collect()

    return np.transpose(signals)


def import_labels(recordName, dataPath):
    # imports all the sleep stages as numbers in in array. A negative 1 corresponds to an undefined label.
    time = load_time(recordName, dataPath + os.sep + "INT_TIME" + os.sep)
    print(time)
    #sampling rate
    fs=1000 #removing downsample
    labels=np.zeros(1200*fs)
    labels[time[1]*fs:(2*time[1])*fs] = 1
    labels[2*time[1]*fs:(2*time[1]+time[3])*fs] = 2
    garbageCollector.collect()
    return labels

if __name__=="__main__":
    root = os.path.join("Mouse_Training_Data", "")
    print(root)
    x = extractWholeRecord(recordName = "MouseCKA1_030515_HCOFTS", dataPath = root)
    print(len(x[0]))
    y = import_labels(recordName = "MouseCKA1_030515_HCOFTS", dataPath = root)
    print(len(y))
    
    print("num0", sum(y==0))
    print("num1", sum(y==1))
    print("num2", sum(y==2))

