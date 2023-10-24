# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 2023

@author: Wale
"""
import torch
import torch.nn as nn
import torch.nn.functional as tn_func
from torch.autograd import Variable

# Construct AI network
# Brain
class CNN(nn.Module):
    xl : torch.Tensor
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        num_neurons = self.count_neurons((1, 80, 80))
        output = 40
        self.full_connection1 = nn.Linear(in_features = num_neurons, out_features = output)
        self.full_connection2 = nn.Linear(in_features = output, out_features = number_actions)
        
    def count_neurons(self, image_dimensions):
        x = Variable(torch.rand(1, *image_dimensions))
        x = tn_func.relu(tn_func.max_pool2d(self.convolution1(x), 3, 2))
        x = tn_func.relu(tn_func.max_pool2d(self.convolution2(x), 3, 2))
        x = tn_func.relu(tn_func.max_pool2d(self.convolution2(x), 5, 2))
        return x.data.view(1, -1).size(1)
    
    def foward(self, x):
        x = tn_func.relu(tn_func.max_pool2d(self.convolution1(x), 3, 2))
        x = tn_func.relu(tn_func.max_pool2d(self.convolution2(x), 3, 2))
        x = tn_func.relu(tn_func.max_pool2d(self.convolution2(x), 5, 2))        
        x = x.view(x.size(0), -1)
        x = tn_func.relu(self.full_connection1(x))
        x = self.full_connection2(x)
        return x
