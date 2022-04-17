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
