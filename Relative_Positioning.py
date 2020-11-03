import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Stager_net_pratice import StagerNet

class RelPosNet(nn.Module):
    def __init__(self, channels):
        super(RelPosNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 2, (2,1), stride=(1,1))

        #we want 2 filters?
        self.stagenet=StagerNet(channels)
        self.linear = nn.Linear(100,1)
        
    def forward(self, x1, x2):
        x1 = self.stagenet(x1)
        x2 = self.stagenet(x2)
        #print('X1', x1.size())
        #print('X2', x2.size())
        
        #the torch.abs() is able to emulate the grp
        x1 = self.linear(torch.abs(x1-x2))
        return x1


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model_stager = StagerNet().to(device)
    model = RelPosNet().to(device)
    # print(model)


    x1 = torch.randn(2, 3000, 2)
    x2 = torch.randn(2, 3000, 2)
    y = torch.randn(2, 1)

    x1, x2, y = x1.to(device), x2.to(device), y.to(device)



    print("Start Training")
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)
    for t in range(20):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x1, x2)

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
        
        
      