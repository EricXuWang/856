#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class Logistic_Regression(nn.Module):
    def __init__(self):
        super(Logistic_Regression, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 3),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)

        return x

# Fully connected neural network with one hidden layer
class TwoNN(nn.Module):
    def __init__(self, input_size):
        super(TwoNN, self).__init__()
        self.activation = nn.ReLU(True)
        self.fc1 = nn.Linear(in_features=input_size, out_features=200, bias=True)
        self.fc2 = nn.Linear(in_features=200, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=10, bias=True)
    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        # we done't need to add softmax, because it has been integrated in the nn.CrossEntropyLoss()
        return x

class CNN(nn.Module):
    def __init__(self, Channels, num_fnn):
        super(CNN, self).__init__()
        self.activation = nn.ReLU(True)
        # hidden_channels = 32
        # num_hiddens = 512
        self.conv1 = nn.Conv2d(in_channels = Channels, out_channels = 32, kernel_size = (5, 5),padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5),padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2),padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2),padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features = num_fnn, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 10)

    def forward(self, x):
        # print(x.shape)
        x = self.activation(self.conv1(x))

        # print(x.shape)
        x = self.maxpool1(x)

        # print(x.shape)
        x = self.activation(self.conv2(x))

        # print(x.shape)
        x = self.maxpool2(x)

        # print(x.shape)
        x = self.flatten(x)

        # print(x.shape)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x






class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x








# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)

#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#         return x

    
# class TwoNN(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(TwoNN, self).__init__()
#         self.activation = nn.ReLU(True)
#         self.fc1 = nn.Linear(in_features=dim_in, out_features=200, bias=True)
#         self.fc2 = nn.Linear(in_features=200, out_features=200, bias=True)
#         self.fc3 = nn.Linear(in_features=200, out_features=dim_out, bias=True)
#
#     def forward(self, x):
# #         if x.ndim == 4:
# #             x = x.view(x.size(0), -1)
#         # The shape attribute for numpy arrays returns the dimensions of the array.
#         # If Arr has m rows and m columns, then Arr.shape is (m,n).
#         # So Arr.shape[0] is m and Arr.shape[1] is n. Also, Arr.shape[-1] is n, Arr.shape[-2] is m.
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         # print(x.shape)
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.fc3(x)
#         # do not have softmax, because nn.CrossEntropyLoss() has it
#         # print(x.shape)
#         return x
#
# model = TwoNN(1*28*28,200,10)
# model.parameters
# total = sum([param.nelement() for param in model.parameters()])
# total = 199.210
    

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         # nn.Conv2d(self, in_channels, out_channels, kernel_size)
#         # The number of input channels is args.num_channels
#         # The number of output channels is 10
#         # Kernel size = 5 means that the shape of kernel is 5*5 shape
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         # The number of input channels is 10
#         # The number of output channels is 20
#         # Kernel size = 5 means that the shape of kernel is 5*5 shape
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         # Dropout with probability p (default = 0.5)
#         self.conv2_drop = nn.Dropout2d()
#         # torch.nn.Linear(in_features, out_features, bias=True)
#         # The number of input features is 320
#         # The number of output features is 50
#         self.fc1 = nn.Linear(320, 50)
#         # The number of input features is 50
#         # The number of output features is args.num_classes
#         self.fc2 = nn.Linear(50, args.num_classes)

#     def forward(self, x):
#         # structure: conv1 - pooling - relu
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         # structure: conv2 - drop - pool - relu 
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         # flatten 
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         # full connected layer - relu
#         x = F.relu(self.fc1(x))
#         # F.dropout works with training=self.training
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x

    
#
# class CNN(nn.Module):
#     def __init__(self, Channels):
#         super(CNN, self).__init__()
#         self.activation = nn.ReLU(True)
#         hidden_channels = 32
#         num_hiddens = 512
#         self.conv1 = nn.Conv2d(in_channels=Channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
#
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
#         self.flatten = nn.Flatten()
#
#         self.fc1 = nn.Linear(in_features=(hidden_channels * 2)*7*7, out_features=num_hiddens, bias=True)
#         self.fc2 = nn.Linear(in_features=num_hiddens, out_features=10, bias=True)
#
#     def forward(self, x):
#         x = self.activation(self.conv1(x))
#         x = self.maxpool1(x)
#
#         x = self.activation(self.conv2(x))
#         x = self.maxpool2(x)
#         x = self.flatten(x)
#
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)
        # return x
    
    
# model = CNN(1*28*28,32,512,10)
# model.parameters    
# total = sum([param.nelement() for param in model.parameters()])
# total    
    
    
#
#
#
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         # The number of input channels is 3
#         # The number of output channels is 6
#         # The shape of kernels are 5*5
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)
