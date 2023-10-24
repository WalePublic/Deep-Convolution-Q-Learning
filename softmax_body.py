# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 2023

@author: Wale
"""
import torch.nn as nn
import torch.nn.functional as tn_func


# Body
class SoftmaxBody(nn.Module()):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    def forward(self, outputs):
        probabilities = tn_func.softmax(outputs * self.T)
        actions = probabilities.multinomial()
        return actions
