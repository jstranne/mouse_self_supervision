#!/usr/bin/env python3
"""
Jason Stranne
"""
import numpy as np
import os
import sys
import gc
from RP_Downstream_Trainer import DownstreamNet, Downstream_Dataset, print_class_counts, num_correct, reduce_dataset_size
from RP_Downstream_Trainer import smallest_class_len, restrict_training_size_per_class
from RP_train_all_at_once import train_end_to_end_RP_combined
sys.path.insert(0, '..')
from Stager_net_pratice import StagerNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from RP_data_loader import Custom_RP_Dataset
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset

def train_different_classes(RP_train_gen, RP_val_gen, train_set, test_set, epochs, sample_list):
    outList=[]
    balanced_acc_out=[]
    for i in sample_list:
        # print(i)
        acc, balanced_acc = train_end_to_end_RP_combined(RP_train_gen, RP_val_gen, train_set, test_set, i, epochs, 3)
        outList.append(acc)
        balanced_acc_out.append(balanced_acc)
    return outList, balanced_acc_out


root = os.path.join("Mouse_Training_Data", "Windowed_Data", "")

datasets_list=[]
print('Loading Data')
f=open(os.path.join("training_names.txt"),'r')
lines = f.readlines()
for line in lines:
    recordName=line.strip()
    print('Processing', recordName)
    data_file=root+recordName+os.sep+recordName
    datasets_list.append(Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=3, sfreq=1000))
f.close()


training_set = torch.utils.data.ConcatDataset(datasets_list)

data_len = len(training_set)
print("dataset len is", len(training_set))

train_len = int(data_len*0.8)
val_len = data_len - train_len

training_set, validation_set = torch.utils.data.random_split(training_set, [train_len, val_len])

print("one dataset is", len(datasets_list[0]))

params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 40
training_generator = torch.utils.data.DataLoader(training_set, **params)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

print("len of the dataloader is:",len(training_generator))

# cuda setup if allowed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0




# actual training

root = os.path.join("Mouse_Training_Data", "Windowed_Data", "")
datasets_list=[]
print('Loading Data')
f=open(os.path.join("training_names.txt"),'r')
lines = f.readlines()
x_vals = []
y_vals = []
groups = []
index = 0
for line in lines:
    recordName=line.strip()
    print('Processing', recordName)
    data_file=root+recordName+os.sep+recordName
    d = Downstream_Dataset(path=data_file)
    x_vals.append(d.data)
    y_vals.append(d.labels)
    groups.append(np.ones(len(d.labels))*index)
    index+=1
    
f.close()


x_vals = np.vstack(x_vals)
y_vals = np.concatenate(y_vals, axis=0)
groups = np.concatenate(groups, axis=0)
print(x_vals.shape)
print(y_vals.shape)
print(groups.shape)


logo = LeaveOneGroupOut()
logo.get_n_splits(x_vals, y_vals, groups)

result_dict = {}
# dtype=torch.int32
for train_index, test_index in logo.split(x_vals, y_vals, groups):
    group_num = groups[test_index][0]
    print("Leaving out mouse number:", group_num)
    training_set = TensorDataset(torch.tensor(x_vals[train_index], dtype=torch.float), torch.tensor(y_vals[train_index], dtype=torch.long))
    test_set = TensorDataset(torch.tensor(x_vals[test_index], dtype=torch.float), torch.tensor(y_vals[test_index], dtype=torch.long))
    print(len(training_set))
    smallest_class = smallest_class_len(training_set, 3)
    num_samples=[]
    temp=1
    while temp < smallest_class:
        num_samples.append(temp)
        temp*=10
    num_samples.append(None)
    # def train_different_classes(RP_train_gen, RP_val_gen, train_set, test_set, epochs, sample_list):

    accuracy, balanced_accuracy = train_different_classes(training_generator, validation_generator, training_set, test_set, 50, num_samples)
    for num_pos, acc in zip(num_samples, accuracy):
        if num_pos not in result_dict:
            result_dict[num_pos] = []
        result_dict[num_pos].append(acc)
    print(result_dict)
    
for k in result_dict:
    result_dict[k] = np.mean(result_dict[k])
print(result_dict)