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
from sklearn.metrics import balanced_accuracy_score


class DownstreamNet(nn.Module):
    def __init__(self, trained_stager, classes):
        super(DownstreamNet, self).__init__()
        self.stagenet=trained_stager
        self.linear = nn.Linear(100,3) # 5 labels
        
    def forward(self, x):
        x = self.stagenet(x)
        x = self.linear(x)
        return x

class Downstream_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, path):
        datapath = path+"_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath = path+"_Windowed_Label.npy"
        self.labels = np.load(datapath)
        
        #need to removed the -1 labels (unknown)
        unknown=np.where(self.labels<0)
        self.labels=np.delete(self.labels,unknown)
        self.data=np.delete(self.data,unknown, axis=0)
        # print("labels shape", self.labels.shape)
        # print(self.labels)
        # print("data shape", self.data.shape)
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

def num_correct(ypred, ytrue):
    #print(ypred)
    #print(torch.argmax(ypred, dim=1))
    #torch.argmax(a, dim=1)
    return (torch.argmax(ypred, dim=1)==ytrue).float().sum().item()

def reduce_dataset_size(dataset, num_of_each):
    print(dataset.shape)
    a = dataset[:, 0]
    print(a)


def smallest_class_len(training_set, k):
    indecies=[[] for x in range(k)]

    for i in range(len(training_set)):
            # puts in in the list based on the label
            # print(training_set[i][1].item())
            indecies[training_set[i][1].item()].append(i)

    smallest_class = len(indecies[0])
    for num in range(len(indecies)):
        # print(len(indecies[num]))
        smallest_class=min(smallest_class, len(indecies[num]))
    return smallest_class


def restrict_training_size_per_class(training_set, samples_per_class, k):
    indecies=[[] for x in range(k)]

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



def train_end_to_end(stagernet_path, train_set, test_set, pos_labels_per_class, max_epochs, classes, verbose=False):
    

    gc.collect()
    
    
    data_len = len(train_set)
    train_len = int(data_len*0.8)
    val_len = data_len - train_len
    train_set, val_set = torch.utils.data.random_split(train_set, [train_len, val_len])
    
    
    
    
    train_set_reduced = restrict_training_size_per_class(train_set, pos_labels_per_class, classes)

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}
    training_generator = torch.utils.data.DataLoader(train_set_reduced, **params)
    validation_generator = torch.utils.data.DataLoader(val_set, **params)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    print("len of the dataloader is:",len(training_generator))
    
    
    if stagernet_path == "full_supervision":
        trained_stage = StagerNet(11)
    else:    
        trained_stage = StagerNet(11)
        if stagernet_path:
            trained_stage.load_state_dict(torch.load("models"+os.sep+stagernet_path))

        for p in trained_stage.parameters():
            p.requires_grad = False
            #print(p)

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = DownstreamNet(trained_stage, classes).to(device)

    #defining training parameters
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)


    print("Start Training")
    min_val_loss = float("inf")
    min_state = None
    patience = 0

    for epoch in range(max_epochs):
        running_loss=0
        correct=0
        total=0
        for x, y in training_generator:
            #print(X1.shape)
            # Transfer to GPU
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            #print("batch:", loss.item())

            #accuracy
            correct += num_correct(y_pred,y)
            total += len(y)


            print_class_counts(y_pred)


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
        for x, y in validation_generator:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            val_correct += num_correct(y_pred,y)
            val_total += len(y)
            val_balanced_acc = balanced_accuracy_score(y.cpu(), torch.argmax(y_pred, dim=1).cpu().data.numpy())*len(y)
        model.train=True
        
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
                model.train=False
                test_correct=0
                test_total=0
                for x, y in test_generator:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    test_correct += num_correct(y_pred,y)
                    test_total += len(y)
                    test_balanced_acc = balanced_accuracy_score(y.cpu(), torch.argmax(y_pred, dim=1).cpu().data.numpy())*len(y)
                model.train=True
                return test_correct/test_total, test_balanced_acc/test_total
                


        # val_outputs = model()
        # print(epoch)
        if verbose:
            print('[Epoch %d] Training loss: %.3f' %
                              (epoch + 1, running_loss/len(training_generator)))
            print('Training accuracy: %.3f' %
                              (correct/total))

            print('Validation accuracy: %.3f' %
                              (val_correct/val_total))
    # return answer at the end
    model.train=False
    test_correct=0
    test_total=0
    for x, y in test_generator:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        test_correct += num_correct(y_pred,y)
        test_total += len(y)
        test_balanced_acc = balanced_accuracy_score(y.cpu(), torch.argmax(y_pred, dim=1).cpu().data.numpy())*len(y)
    model.train=True
    return test_correct/test_total, test_balanced_acc/test_total
    #return val_correct/val_total, val_balanced_acc/val_total



if __name__=="__main__":
    root = os.path.join("Mouse_Training_Data", "Windowed_Data", "")

    datasets_list=[]
    print('Loading Data')
    f=open(os.path.join("training_names.txt"),'r')
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
        
    train_end_to_end("RP_stagernet.pth", training_set, validation_set, 1000, 150)