import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# Create random Tensors to hold inputs and outputs
x = torch.randn(2, 3000)

class StagerNet(nn.Module):
    def __init__(self, channels):
        super(StagerNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 2, (2,1), stride=(1,1))

        #we want 2 filters?
        self.conv1 = nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.dense1 = nn.Linear(208*channels,100)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
    def forward(self, x):
        # Input should be C by T (2 by 3000)
        x = torch.unsqueeze(x, 1)
        # change into C by T by 1 extending
        # print("Initial Size", x.size())
        
        # convolve with C filters to 1 by T by C
        #linear activation ??
        x = self.conv1(x)
        # print("After spatial conv", x.size())
        
        #permute to C T I
        x = x.permute(0, 3, 2, 1)
        #print("after permutation", x.size())
        
        x = self.conv2(x)
        #print("After First Temporal Conv", x.size())
        
        x = F.relu(F.max_pool2d(x, (13,1)))
        # print("Relu and maxpool", x.size())
        
        x = self.batchnorm1(x)
        #print("Batchnorm1", x.size())
        
        x = self.conv3(x)
        # print("After Second Temporal Conv", x.size())
        
        x = F.relu(F.max_pool2d(x, (13, 1)))
        # print("Relu and maxpool", x.size())
        
        x = self.batchnorm2(x)
        #print("Batchnorm2", x.size())
        
        x = torch.flatten(x,1) #flatten all but batch
        #print("Flattened", x.size())
        
        x = F.dropout(x, p=0.5)
        #print("dropout", x.size())
        
        x = self.dense1(x)
        #print("Dense", x.size())
        return x



    # BatchNorm2d


    # 2D temporal convolution to get C by T by 8
    #Activation is relu, mode same

    # Maxpool 2D

    # 2D temporal convolution to get C by T by 8
    # Activation is relu, mode same

    # Maxpool 2D

    #Flatten

    # Dropout

    # dense so output is 5



    ##################################333333
    # torch.nn.Linear(D_in, H),
    # torch.nn.ReLU(),
    # torch.nn.Linear(H, D_out),

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = StagerNet(2).to(device)
    print(model)
    # summary(model, (1, 28, 28))


    #we really want 1 channel so we can get the right convolution
    x = torch.randn(2, 3000)
    print(x.size())
    summary(model, (3000, 2))
    print("test")

    # loss_fn = torch.nn.MSELoss(reduction='sum')
    #
    # # Use the optim package to define an Optimizer that will update the weights of
    # # the model for us. Here we will use Adam; the optim package contains many other
    # # optimization algorithms. The first argument to the Adam constructor tells the
    # # optimizer which Tensors it should update.
    # learning_rate = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # for t in range(500):
    #     # Forward pass: compute predicted y by passing x to the model.
    #     y_pred = model(x)
    #
    #     # Compute and print loss.
    #     loss = loss_fn(y_pred, y)
    #     if t % 100 == 99:
    #         print(t, loss.item())
    #
    #     # Before the backward pass, use the optimizer object to zero all of the
    #     # gradients for the variables it will update (which are the learnable
    #     # weights of the model). This is because by default, gradients are
    #     # accumulated in buffers( i.e, not overwritten) whenever .backward()
    #     # is called. Checkout docs of torch.autograd.backward for more details.
    #     optimizer.zero_grad()
    #
    #     # Backward pass: compute gradient of the loss with respect to model
    #     # parameters
    #     loss.backward()
    #
    #     # Calling the step function on an Optimizer makes an update to its
    #     # parameters
    #     optimizer.step()