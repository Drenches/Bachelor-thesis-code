# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:36:00 2020

@author: dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pdb

# N_STATES = 3
# N_ACTIONS = 4
N_STATES = 10
ACTION = [100, 500, 1000, 2000, 4000, 5000]
N_ACTIONS = len(ACTION)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

net = Net()
print(net)
