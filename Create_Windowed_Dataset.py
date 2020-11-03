#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
from Extract_Data import loadSignals, extractWholeRecord, import_labels
from sklearn import preprocessing
from scipy import stats

def preprocess_file(recordName):
    root = os.path.join("Mouse_Training_Data", "")
    # returns all channels of the eeg, 30Hz hamming low pass filtered
    x = extractWholeRecord(recordName = recordName, dataPath = root)
    y = import_labels(recordName = recordName, dataPath = root)
    print(x.shape)
    print(y.shape)
    sampling_rate = 100 # in Hz
    window_size = 30 # in sec
    
    print("Before preprocessing shape is:", x.shape)
    total_windows = len(x)//(sampling_rate*window_size)
    print("Total windows:", total_windows) 
    xwindows=[]
    sleep_labels=[]

    for i in range(total_windows):
        xval = x[sampling_rate*window_size*i:sampling_rate*window_size*(i+1)]
        yval = y[sampling_rate*window_size*i:sampling_rate*window_size*(i+1)]

        
        mode, mode_count = stats.mode(yval)

        #normalized channel wize for zero mean and unit sd
        xval = preprocessing.scale(xval,axis=0)
        #xval = np.expand_dims(xval, axis=0)
        #print(xval.shape)
        #print(xwindows.shape)
        xwindows.append(xval)
        #mode is returned as a list so just get the first index
        sleep_labels.append(mode[0])
        
        
    x_rp = extractWholeRecord(recordName = recordName, dataPath = root, downstream=False)
    print("Before preprocessing shape is:", x_rp.shape)
    total_windows = len(x_rp)//(sampling_rate*window_size)
    print("Total windows:", total_windows) 
    xwindows_rp=[]
    start_times=[]
    for i in range(total_windows):
        xval = x_rp[sampling_rate*window_size*i:sampling_rate*window_size*(i+1)]
        xval = preprocessing.scale(xval,axis=0)
        xwindows_rp.append(xval)
        start_times.append(window_size*i)
    
    
    print("XWINDOWS IS SIZE", np.array(xwindows).shape)
    print("X_RP_WINDOWS IS SIZE", np.array(xwindows_rp).shape)
    
    root = os.path.join("Mouse_Training_Data", "Windowed_Data", recordName, "")
    os.makedirs(root, exist_ok=True)
    np.save(file=root + os.sep + recordName+"_Windowed_Preprocess", arr=np.array(xwindows))
    np.save(file=root + os.sep + recordName+"_Windowed_Label", arr=np.array(sleep_labels))
    np.save(file=root + os.sep + recordName+"_Windowed_StartTime", arr=np.array(start_times))
    np.save(file=root + os.sep + recordName+"_Windowed_Pretext_Preprocess", arr=np.array(xwindows_rp))


if __name__=="__main__":
    f=open(os.path.join("training_names.txt"),'r')
    lines = f.readlines()
    for line in lines:
        print(line.strip())
        preprocess_file(line.strip())
    f.close()
    