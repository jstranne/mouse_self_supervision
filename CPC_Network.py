import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Stager_net_pratice import StagerNet
import numpy as np
import gc


class CPC_Net(nn.Module):
    def __init__(self, Np, channels):
        super(CPC_Net, self).__init__()
        self.stagenet = StagerNet(channels)
        h_dim=100
        ct_dim=100
        self.gru = nn.GRU(ct_dim, h_dim, 1, batch_first=True)
        self.BilinearList = nn.ModuleList()
        for i in range(Np):
            self.BilinearList.append(nn.Bilinear(in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False))
        
        
        self.sample_bilin = nn.Bilinear(in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False)

        self.logsoftmax = nn.LogSoftmax()


    def forward(self, Xc, Xp, Xb_array):
        
        gc.collect()
        
        #print(Xb_array.shape)
        #a = torch.empty([32, 16, 10, 100], dtype=Xp.dtype, device=Xp.device)
#         for batch in range(list(Xb_array.shape)[0]):
#             for i in range(list(Xb_array.shape)[1]):
#                 for j in range(list(Xb_array.shape)[2]):
#                     gc.collect()
#                     a[:, i, j, :] = self.stagenet(torch.squeeze(Xb_array[:, i, j, :, :]))
        
        Xb_array = [[self.stagenet(torch.squeeze(Xb_array[:, i, j, :, :])) for i in range(list(Xb_array.shape)[1])] for j in range(list(Xb_array.shape)[2])]         
        for i in range(len(Xb_array)):
            Xb_array[i] = torch.stack(Xb_array[i])
#         Xb_array = [torch.stack(Xb_array[i]) for i in range(len(Xb_array))]
        Xb_array = torch.stack(Xb_array)
        Xb_array = Xb_array.permute(2, 1, 0, 3) 
        
        
        Xc = [self.stagenet(torch.squeeze(Xc[:, x, :, :])) for x in range(list(Xc.shape)[1])]
        Xc = torch.stack(Xc)
        Xc = Xc.permute(1, 0, 2) 
        
        Xp_new = [self.stagenet(torch.squeeze(Xp[:, x, :, :])) for x in range(list(Xp.shape)[1])]
        Xp_new = torch.stack(Xp_new)
        Xp_new = Xp_new.permute(1, 0, 2) 
        
        
        output, hn = self.gru(Xc)
        hn=torch.squeeze(hn)
        Xp_new = Xp_new.unsqueeze(2)
        
        
        Xp_new = torch.cat((Xp_new, Xb_array), 2)
        
        
        output_cat = torch.empty([list(Xb_array.shape)[0], 16, 11], dtype=Xp_new.dtype, device=Xp_new.device)
        
        for batch in range(list(Xp_new.shape)[0]):
            for predicted in range(list(Xp_new.shape)[1]):
                #bilinear = self.BilinearList[predicted]
                for sample in range(list(Xp_new.shape)[2]):
                    
                    output_cat[batch, predicted, sample] = self.BilinearList[predicted](hn[batch, :], Xp_new[batch, predicted, sample, :])
        # print(output_cat.shape)
        
        return output_cat


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model_stager = StagerNet().to(device)
    model = TemporalShufflingNet().to(device)
    # print(model)


    x1 = torch.randn(2, 3000, 2)
    x2 = torch.randn(2, 3000, 2)
    x3 = torch.randn(2, 3000, 2)
    y = torch.randn(2, 1)
    
    x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
    


    print("Start Training")
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)
    for t in range(20):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x1, x2, x3)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()