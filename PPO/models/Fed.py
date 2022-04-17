#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    # The received models are dictonaries
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        
        # Calculate the sume of all models 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        
        # Calculate the averaged global model
        w_avg[k] = torch.div(w_avg[k], len(w))
        
        # Return the global model
    return w_avg
