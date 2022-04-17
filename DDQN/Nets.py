#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


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
        self.conv1 = nn.Conv2d(in_channels = Channels, out_channels = 32, kernel_size = (5, 5),padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5),padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2),padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2),padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features = num_fnn, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 10)

    def forward(self, x):

        x = self.activation(self.conv1(x))

        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))

        x = self.maxpool2(x)

        x = self.flatten(x)

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
        self.fc1 = nn.Linear(16 * 5 * 5, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


