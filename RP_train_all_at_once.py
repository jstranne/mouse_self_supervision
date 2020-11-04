#!/usr/bin/env python3
"""
Jason Stranne
"""
import numpy as np
import os
import sys
import gc
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from torchsummary import summary
from Stager_net_pratice import StagerNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from Custom_data_loader import Custom_RP_Dataset

class RelPosNet_Downstream(nn.Module):
    def __init__(self):
        super(RelPosNet_Downstream, self).__init__()
        # self.conv1 = nn.Conv2d(1, 2, (2,1), stride=(1,1))

        #we want 2 filters?
        self.stagenet=StagerNet()
        self.linear = nn.Linear(100,1)
        self.linear_downstream = nn.Linear(100,5)
        
    def forward(self, x1, x2, second_layer_inin):
        if x1==None and x2==None:
            second_layer_inin = self.stagenet(second_layer_inin)
            return x1, self.linear_downstream(second_layer_inin)
        x1 = self.stagenet(x1)
        x2 = self.stagenet(x2)
        x1 = self.linear(torch.abs(x1-x2))
        second_layer_inin = self.stagenet(second_layer_inin)
        return x1, self.linear_downstream(second_layer_inin)


class Downstream_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, path):
        datapath = path+"_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath = path+"_Windowed_SleepLabel.npy"
        self.labels = np.load(datapath)
        
        #need to removed the -1 labels (unknown)
        unknown=np.where(self.labels<0)
        self.labels=np.delete(self.labels,unknown)
        self.data=np.delete(self.data,unknown, axis=0)
        print("labels shape", self.labels.shape)
        print("data shape", self.data.shape)
        print("removed", len(unknown[0]), "unknown entries")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.from_numpy(self.data[index,:,:]).float()
        Y = torch.from_numpy(np.array(self.labels[index])).long()
        return X, Y

    
def print_class_counts(y_pred):
#         zero = (torch.argmax(y_pred, dim=1)==0).float().sum()
#         print("zero", zero)
#         one = (torch.argmax(y_pred, dim=1)==1).float().sum()
#         print("one", one)
#         two = (torch.argmax(y_pred, dim=1)==2).float().sum()
#         print("two", two)
#         three = (torch.argmax(y_pred, dim=1)==3).float().sum()
#         print("three", three)
#         four = (torch.argmax(y_pred, dim=1)==4).float().sum()
#         print("four", four)
    n1 = (torch.argmax(y_pred, dim=1)==-1).float().sum()
    if n1.item() > 0:
        print("There are", n1.item(), "Unknown labels that showed up")

        
def num_correct_RP(ypred, ytrue):
    return ((ypred* ytrue) > 0).float().sum().item()        

def num_correct_downstream(ypred, ytrue):
    #print(ypred)
    #print(torch.argmax(ypred, dim=1))
    #torch.argmax(a, dim=1)
    return (torch.argmax(ypred, dim=1)==ytrue).float().sum().item()

def reduce_dataset_size(dataset, num_of_each):
    print(dataset.shape)
    a = dataset[:, 0]
    print(a)


def smallest_class_len(training_set):
    indecies=[[],[],[],[],[]]

    for i in range(len(training_set)):
            # puts in in the list based on the label
            indecies[training_set[i][1].item()].append(i)

    smallest_class = len(indecies[0])
    for num in range(len(indecies)):
        smallest_class=min(smallest_class, len(indecies[num]))
    return smallest_class


def restrict_training_size_per_class(training_set, samples_per_class):
    indecies=[[],[],[],[],[]]

    for i in range(len(training_set)):
            # puts in in the list based on the label
            indecies[training_set[i][1].item()].append(i)

    smallest_class = 0
    for num in range(len(indecies)):
        smallest_class=min(smallest_class, len(indecies[num]))
        random.shuffle(indecies[num])

    for num in range(len(indecies)):
        smallest_class=min(smallest_class, len(indecies[num]))
        random.shuffle(indecies[num])

    for num in range(len(indecies)):
        indecies[num]=indecies[num][:samples_per_class]

    flat=itertools.chain.from_iterable(indecies)
    to_use = list(flat)
    random.shuffle(to_use)

    return torch.utils.data.Subset(training_set, to_use)



def train_end_to_end_RP_combined(RP_training_generator, training_set, validation_set, pos_labels_per_class, max_epochs, verbose=False):
#     root = os.path.join("..","training", "")

#     ## Setting up RP
#     datasets_list=[]
#     print('Loading Data')
#     f=open(os.path.join("..","training_names.txt"),'r')
#     lines = f.readlines()
#     for line in lines:
#         recordName=line.strip()
#         print('Processing', recordName)
#         data_file=root+recordName+os.sep+recordName
#         datasets_list.append(Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=30, sfreq=100))
#     f.close()

#     RP_training_set = torch.utils.data.ConcatDataset(datasets_list)

#     params = {'batch_size': 256,
#               'shuffle': True,
#               'num_workers': 6}
#     RP_training_generator = torch.utils.data.DataLoader(RP_training_set, **params)
    
    
    
#     ## Set up downstream dataloader
#     datasets_list=[]
#     print('Loading Data')
#     f=open(os.path.join("..","training_names.txt"),'r')
#     lines = f.readlines()
#     for line in lines:
#         recordName=line.strip()
#         print('Processing', recordName)
#         data_file=root+recordName+os.sep+recordName
#         datasets_list.append(Downstream_Dataset(path=data_file))
#         d = Downstream_Dataset(path=data_file)
#         print(d.labels.shape)
#     f.close()


#     dataset = torch.utils.data.ConcatDataset(datasets_list)
#     data_len = len(dataset)
#     print("dataset len is", len(dataset))

#     train_len = int(data_len*0.6)
#     val_len = data_len - train_len
     
#     training_set, validation_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    train_set_reduced = restrict_training_size_per_class(training_set, pos_labels_per_class)
    
    # we want to weight the downstream more when we have more data there, so weight with 10*log10(number of each example)+1
    
    loss_weighting = 1 + np.log10(len(train_set_reduced))*2
    
    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}
    training_generator_downstream = torch.utils.data.DataLoader(train_set_reduced, **params)
    validation_generator_downstream = torch.utils.data.DataLoader(validation_set, **params)
    
    
    print("downstream batches:", len(training_generator_downstream))
    print("pretext batches:", len(RP_training_generator))
    
    
    
    
    
    
    
    model = RelPosNet_Downstream().to(device)

    #defining training parameters
    print("Start Training Full")
    loss_fn_rp = torch.nn.SoftMarginLoss(reduction='sum')
    loss_fn_downstream = nn.CrossEntropyLoss()
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

    for epoch in range(max_epochs):
        running_loss = 0
        correct_pretext = 0
        total_pretext = 0
        correct_downstream = 0
        total_downstream = 0
        for i, vals in enumerate(zip(itertools.cycle(training_generator_downstream), RP_training_generator)):
            X1, X2, y = vals[1]
            x_down, y_down = vals[0]
            #print(X1.shape)
            #print(y.shape)
            # Transfer to GPU
            X1, X2, y, x_down, y_down = X1.to(device), X2.to(device), y.to(device), x_down.to(device), y_down.to(device)
            #print(X1.shape)
            y_pred_stager, y_pred_downstream = model(X1, X2, x_down)
            loss = loss_fn_rp(y_pred_stager, y) + loss_weighting*loss_fn_downstream(y_pred_downstream, y_down)
            #calculate accuracy
            correct_pretext += num_correct_RP(y_pred_stager,y)
            total_pretext += len(y)
            correct_downstream += num_correct_downstream(y_pred_downstream,y_down)
            total_downstream += len(y_down)

            #print("batch:", loss.item())

            #zero gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            running_loss+=loss.item()
        
        
        model.train=False
        val_correct=0
        val_total=0
        temp_val = None
        for x, y in validation_generator_downstream:
            x, y = x.to(device), y.to(device)
            temp_val, y_pred = model(None, None, x)
            val_correct += num_correct_downstream(y_pred,y)
            val_total += len(y)
        model.train=True
        
        
        if verbose:
            print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(RP_training_generator)))
            print('[Epoch %d] accuracy pretext: %.3f' %
                              (epoch + 1, correct_pretext/total_pretext))
            print('[Epoch %d] accuracy downstream: %.3f' %
                              (epoch + 1, correct_downstream/total_downstream))
            print('Validation accuracy downstream: %.3f' %
                                  (val_correct/val_total))
    return val_correct/val_total
        


if __name__=="__main__":
    root = os.path.join("..","training", "")

    ## Setting up RP
    datasets_list=[]
    print('Loading Data')
    f=open(os.path.join("..","training_names.txt"),'r')
    lines = f.readlines()
    for line in lines:
        recordName=line.strip()
        print('Processing', recordName)
        data_file=root+recordName+os.sep+recordName
        datasets_list.append(Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=30, sfreq=100))
    f.close()

    RP_training_set = torch.utils.data.ConcatDataset(datasets_list)

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}
    RP_training_generator = torch.utils.data.DataLoader(RP_training_set, **params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    
    
    
    ## Set up downstream dataloader
    datasets_list=[]
    print('Loading Data')
    f=open(os.path.join("..","training_names.txt"),'r')
    lines = f.readlines()
    for line in lines:
        recordName=line.strip()
        print('Processing', recordName)
        data_file=root+recordName+os.sep+recordName
        datasets_list.append(Downstream_Dataset(path=data_file))
        d = Downstream_Dataset(path=data_file)
        print(d.labels.shape)
    f.close()


    dataset = torch.utils.data.ConcatDataset(datasets_list)
    data_len = len(dataset)
    print("dataset len is", len(dataset))

    train_len = int(data_len*0.6)
    val_len = data_len - train_len
     
    training_set, validation_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    pos_labels_per_class = 1
    train_set_reduced = restrict_training_size_per_class(training_set, pos_labels_per_class)
    
    # we want to weight the downstream more when we have more data there, so weight with 10*log10(number of each example)+1
    
    loss_weighting = 1 + np.log10(len(train_set_reduced))*2
    
    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}
    training_generator_downstream = torch.utils.data.DataLoader(train_set_reduced, **params)
    validation_generator_downstream = torch.utils.data.DataLoader(validation_set, **params)
    
    
    print("downstream batches:", len(training_generator_downstream))
    print("pretext batches:", len(RP_training_generator))
    
    
    
    
    
    
    
    model = RelPosNet_Downstream().to(device)

    #defining training parameters
    print("Start Training Full")
    loss_fn_rp = torch.nn.SoftMarginLoss(reduction='sum')
    loss_fn_downstream = nn.CrossEntropyLoss()
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

    max_epochs=20
    for epoch in range(max_epochs):
        running_loss = 0
        correct_pretext = 0
        total_pretext = 0
        correct_downstream = 0
        total_downstream = 0
        for i, vals in enumerate(zip(itertools.cycle(training_generator_downstream), RP_training_generator)):
            X1, X2, y = vals[1]
            x_down, y_down = vals[0]
            #print(X1.shape)
            #print(y.shape)
            # Transfer to GPU
            X1, X2, y, x_down, y_down = X1.to(device), X2.to(device), y.to(device), x_down.to(device), y_down.to(device)
            #print(X1.shape)
            y_pred_stager, y_pred_downstream = model(X1, X2, x_down)
            loss = loss_fn_rp(y_pred_stager, y) + loss_weighting*loss_fn_downstream(y_pred_downstream, y_down)
            #calculate accuracy
            correct_pretext += num_correct_RP(y_pred_stager,y)
            total_pretext += len(y)
            correct_downstream += num_correct_downstream(y_pred_downstream,y_down)
            total_downstream += len(y_down)

            #print("batch:", loss.item())

            #zero gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            running_loss+=loss.item()
        
        
        model.train=False
        val_correct=0
        val_total=0
        temp_val = None
        for x, y in validation_generator_downstream:
            x, y = x.to(device), y.to(device)
            temp_val, y_pred = model(None, None, x)
            val_correct += num_correct_downstream(y_pred,y)
            val_total += len(y)
        model.train=True
        
        print('[Epoch %d] loss: %.3f' %
                          (epoch + 1, running_loss/len(RP_training_generator)))
        print('[Epoch %d] accuracy pretext: %.3f' %
                          (epoch + 1, correct_pretext/total_pretext))
        print('[Epoch %d] accuracy downstream: %.3f' %
                          (epoch + 1, correct_downstream/total_downstream))
        print('Validation accuracy downstream: %.3f' %
                              (val_correct/val_total))
    
    
    

# print(model.stagenet)
# stagenet_save_path = os.path.join("..", "models", "RP_stagernet.pth")
# torch.save(model.stagenet.state_dict(), stagenet_save_path)
    
    
    
    
    
    

    
    
    
        
#     train_end_to_end("RP_stagernet.pth", training_set, validation_set, 1000, 150)
    
    
    
    
    
# basic idea is dont freeze the weights
# pass data in, use to make a rel pos example, 
# then also use for downstream