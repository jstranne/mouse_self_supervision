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
import torch
import time


class Custom_TS_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, path, total_points, tpos, tneg, windowSize, sfreq):
        #'Initialization'
        datapath=path+"_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath=path+"_Windowed_StartTime.npy"
        self.start_times = np.load(datapath)
        print(self.data.shape)
        self.total_windows = len(self.data)
        self.trios, self.labels = self.get_pairs_and_labels(size=total_points, tpos=tpos, tneg=tneg, windowSize=windowSize)
    def __len__(self):
        'Denotes the total number of samples'
        
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X1 = torch.from_numpy(self.data[self.trios[index,0],:,:]).float()
        X2 = torch.from_numpy(self.data[self.trios[index,1],:,:]).float()
        X3 = torch.from_numpy(self.data[self.trios[index,2],:,:]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()

        return X1, X2, X3, y
    
    def get_pairs_and_labels(self, size, tpos, tneg, windowSize):
        """
        Assigns all of the pairs of input and labels for the pretext task. Ones refer to 
        positive pretext tasks and -1 correcsponds to negative tasks.
        """
        
        pairs = []
        labels = []
        
        # order is t, t', t''
        
        for i in range(size):
            tempval=np.random.randint(low=0, high=self.total_windows)
            
            secondval = self.return_pos_index(index=tempval, tpos=tpos, windowSize=windowSize)
            
            if(np.abs(self.start_times[tempval]-self.start_times[secondval])>tpos): #checking if we got a bad label
                    print("skipping bad label")
                    continue
            
            if random.random() < 0.5: # randomly a positive or a negative example
                outval = 1
                # we need to check if its impossible to return a pos label
                if(np.abs(secondval-tempval)<=1):
                    print("skipping since it is impossible to return a positive label")
                    continue
                print("lowval", tempval)
                print("highval", secondval)
                unknown_val=np.random.randint(low=tempval+1, high=secondval)
                    
            else:
                outval = -1
                unknown_val = self.return_neg_index(tempval, tneg, windowSize)

                # Double check we didnt mess up. If this is printed while executing the script, something is wrong
                if(np.abs(self.start_times[tempval]-self.start_times[unknown_val])<tneg):
                    print("ERROR, messed up neg label")
                    continue
                
            
            pairs.append([tempval, unknown_val, secondval])
            labels.append(outval)
            pairs.append([secondval, unknown_val, tempval])
            labels.append(outval)
            
        
        pairs = np.array(pairs)
        labels=np.array(labels)
        print(labels.shape)
        return pairs, labels
    
    def return_pos_index(self, index, tpos, windowSize):
        # return a positive example
        maximum = min(len(self.data),index+(tpos//windowSize)+1) #since non inclusive

        return np.random.randint(index, maximum)
    
    def return_neg_index(self, index, tneg, windowSize):
        # return a negative example
        
        midlow=max(0,index-(tneg//windowSize))
        midhigh =  min(len(self.data)-1,index+(tneg//windowSize))

        assert (midlow>0 or midhigh<len(self.data))
        # check if it is even possible to return a negative index
        trial = np.random.randint(0, len(self.data))
        while(trial >= midlow and trial <= midhigh):
            # keep trying until we find one in the correct range
            trial = np.random.randint(0, len(self.data))
        return trial
           
        
def num_correct(ypred, ytrue):
    # count the number of times that we predcted the pretext task correctly (signs match means correct)
    return ((ypred* ytrue) > 0).float().sum().item()

if __name__=="__main__":

    root = os.path.join("Mouse_Training_Data", "Windowed_Data", "")

    datasets_list=[]
    print('Loading Data')
    f=open(os.path.join("training_names.txt"),'r')
    lines = f.readlines()
    tpos_val=30
    tneg_val=120
    for line in lines:
        recordName=line.strip()
        print('Processing', recordName)
        data_file=root+recordName+os.sep+recordName
        datasets_list.append(Custom_TS_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=3, sfreq=1000))
    f.close()
    
    model_save_path = f"TS_stagernet_{tpos_val}_{tneg_val}.pth"

    # recordName="tr03-0078"
    # data_file=root+recordName+os.sep+recordName
    # training_set=Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=30, sfreq=100)

    training_set = torch.utils.data.ConcatDataset(datasets_list)
    
    
    data_len = len(training_set)
    print("dataset len is", len(training_set))

    train_len = int(data_len*0.8)
    val_len = data_len - train_len
    
    training_set, validation_set = torch.utils.data.random_split(training_set, [train_len, val_len])
    
    

    print("one dataset is", len(datasets_list[0]))

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}
    max_epochs = 100
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    print("len of the dataloader is:",len(training_generator))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = TemporalShufflingNet(channels=11).to(device)

    #defining training parameters
    print("Start Training")
    
    min_val_loss = float("inf")
    min_state = None
    patience = 0
    
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

    # t_neg=0
    # t_pos=0
    
    
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        running_loss=0
        correct=0
        total=0
        for X1,X2,X3, y in training_generator:
            #print(X1.shape)
            #print(y.shape)
            # Transfer to GPU
            X1, X2, X3, y = X1.to(device), X2.to(device), X3.to(device), y.to(device)
            #print(X1.shape)
            y_pred = model(X1, X2, X3)
            loss = loss_fn(y_pred, y)

            #calculate accuracy
            correct += num_correct(y_pred,y)
            total += len(y)

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
            model.eval()
            val_correct=0
            val_total=0
            for X1,X2,X3, y in validation_generator:
                X1, X2, X3, y = X1.to(device), X2.to(device), X3.to(device), y.to(device)
                y_pred = model(X1, X2, X3)
                val_correct += num_correct(y_pred,y)
                val_total += len(y)

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
        model.train()
        
        print('{} seconds'.format(time.time() - t0))
        print('[Epoch %d] loss: %.3f' %
                          (epoch + 1, running_loss/len(training_generator)))
        print('[Epoch %d] accuracy: %.3f' %
                          (epoch + 1, correct/total))
        print('[Epoch %d] val 0-1: %.5f' %
                          (epoch + 1, zero_one_val))




    print(model.stagenet)
    stagenet_save_path = os.path.join("models", model_save_path)
    torch.save(model.stagenet.state_dict(), stagenet_save_path)
