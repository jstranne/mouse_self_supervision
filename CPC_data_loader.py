#!/usr/bin/env python3
"""
Jason Stranne
"""
import numpy as np
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import random
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Temporal_Shuffling import TemporalShufflingNet
from CPC_Network import CPC_Net
import torch
import torch.nn as nn
from math import floor
from tqdm import tqdm


class Custom_CPC_Dataset(torch.utils.data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, path, Nc, Np, Nb):
        # 'Initialization'
        datapath = path + "_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath = path + "_Windowed_StartTime.npy"
        self.start_times = np.load(datapath)
        print(self.data.shape)
        self.total_windows = len(self.data)
        self.total_points = floor(0.05*self.total_windows)

        #set Nc and Np
        self.Nc = Nc
        self.Np = Np
        self.Nb = Nb
        #get starting points
        self.Xc_starts = self.getXc_starts(self.total_points, self.Nc + self.Np)


    def __len__(self):
        'Denotes the total number of samples'

        return len(self.Xc_starts)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        # X1 = torch.from_numpy(self.data[self.pairs[index, 0], :, :]).float()
        # X2 = torch.from_numpy(self.data[self.pairs[index, 1], :, :]).float()
        # y = torch.from_numpy(np.array([self.labels[index]])).float()
        Xc = torch.from_numpy(self.data[self.Xc_starts[index]:self.Xc_starts[index]+self.Nc, :, :]).float()
        Xp = torch.from_numpy(self.data[self.Xc_starts[index] + self.Nc:self.Xc_starts[index] + self.Nc + self.Np, :, :]).float()
        Xb = []

        for i in range(self.Xc_starts[index]+self.Nc, self.Xc_starts[index] + self.Nc+self.Np):
            Xb.append(self.generate_negative_sample_list(self.Xc_starts[index]))

        Xb = torch.from_numpy(np.array(Xb)).float()
        return Xc, Xp, Xb


    def getXc_starts(self, total_points, buffer_needed):
        startList = []
        for i in range(total_points):
            startList.append(np.random.randint(low=0, high=self.total_windows-buffer_needed))
        return np.array(startList)


    def generate_negative_sample_list(self, xc_start):
        toReturn = []
        for i in range(self.Nb):
            toReturn.append(self.random_Nb_Sample(xc_start))
        return toReturn


    def random_Nb_Sample(self, xcStart):
        num = xcStart # will cause while loop to run
        count = 0
        while xcStart <= num <= (xcStart + self.Nc + self.Np):
            num = np.random.randint(low=0, high=self.total_windows)
            count += 1
            if count > 1000:
                raise Exception("impossible to find a valid Nb sample, need to debug")

        if count == 0:
            raise Exception("never found an Nb, need to debug")
        # cant be in the range (start+Nc+Np)
        return self.data[num, :, :]
          
def customLoss(input_data):
    #input data should be in the shape [batch, np, nb+1]
    # the first index of the nb+1 is the correct one
    lsoft = nn.LogSoftmax(dim=2)
    soft = lsoft(input_data)[:, :, 0]
    return -torch.sum(soft)
        
def num_correct(ypred):
    argmx = torch.argmax(ypred, dim=2)
    correct = (torch.sum(argmx==0)).float().sum().item()
    return correct, argmx.nelement()

    
if __name__=="__main__":

    root = os.path.join("Mouse_Training_Data", "Windowed_Data", "")

    datasets_list=[]
    print('Loading Data')
    f=open(os.path.join("training_names.txt"),'r')
    lines = f.readlines()
    NP_num = 16
    for line in lines:
        recordName=line.strip()
        print('Processing', recordName)
        data_file=root+recordName+os.sep+recordName
        datasets_list.append(Custom_CPC_Dataset(path=data_file, Nc=10, Np=NP_num, Nb=2))
    f.close()

    
    model_save_path = f"CPC_stagernet_{NP_num}.pth"
    

    training_set = torch.utils.data.ConcatDataset(datasets_list)
    
    data_len = len(training_set)
    print("dataset len is", len(training_set))

    train_len = int(data_len*0.8)
    val_len = data_len - train_len
    
    training_set, validation_set = torch.utils.data.random_split(training_set, [train_len, val_len])
    
    

    print("one dataset is", len(datasets_list[0]))

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 12}
    
    max_epochs = 100
    
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    print("len of the dataloader is:",len(training_generator))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = CPC_Net(Np=16, channels=11).to(device)

    print("Start Training")
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

    min_val_loss = float("inf")
    min_state = None
    patience = 0


    for epoch in range(max_epochs):
        model.train()
        running_loss=0
        correct=0
        total=0
        for Xc, Xp, Xb in tqdm(training_generator):
            gc.collect()
            #print(X1.shape)
            #print(y.shape)
            # Transfer to GPU
            Xc, Xp, Xb= Xc.to(device), Xp.to(device), Xb.to(device)
            #print(X1.shape)
            y_pred = model(Xc, Xp, Xb)
            loss = customLoss(y_pred)

            #calculate accuracy
            new_correct, new_total = num_correct(y_pred)
            correct += new_correct
            total += new_total

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
            
            
        with torch.no_grad():
            # model.train=False
            model.eval()
            val_correct=0
            val_total=0
            for Xc, Xp, Xb in validation_generator:
                Xc, Xp, Xb= Xc.to(device), Xp.to(device), Xb.to(device)
                #print(X1.shape)
                y_pred = model(Xc, Xp, Xb)
                new_correct, new_total = num_correct(y_pred)
                val_correct += new_correct
                val_total += new_total

            zero_one_val = 1-val_correct/val_total
            if zero_one_val < min_val_loss:
                patience = 0
                min_val_loss = zero_one_val
                saved_model = model.state_dict()
            else:
                patience += 1
                if patience >= 6:
                    print("EARLY STOPPING")
                    model.load_state_dict(saved_model)
                    stagenet_save_path = os.path.join("models", model_save_path)
                    torch.save(model.stagenet.state_dict(), stagenet_save_path)
                    sys.exit()
        # model.train=True
        model.train()
            
            
            
            
        
        print('[Epoch %d] loss: %.3f' %
                          (epoch + 1, running_loss/len(training_generator)))
        print('[Epoch %d] accuracy: %.3f' %
                          (epoch + 1, correct/total))
        print('[Epoch %d] val 0-1: %.5f' %
                          (epoch + 1, zero_one_val))




    print(model.stagenet)
    stagenet_save_path = os.path.join("models", model_save_path)
    torch.save(model.stagenet.state_dict(), stagenet_save_path)
