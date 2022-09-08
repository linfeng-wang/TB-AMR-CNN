#%%
from pyexpat import model
import statistics
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
import torchvision.transforms as transforms
from itertools import chain
from sklearn import metrics as met
import pickle
import icecream as ic
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

def conv_block1(in_f, out_f, conv_dropout_rate):
    return nn.Sequential(
        nn.Dropout(p=conv_dropout_rate, inplace=False),
        nn.Conv1d(in_f, out_f, kernel_size = 3),
        nn.BatchNorm1d(3),
        nn.ReLU(),
        nn.MaxPool1d(3),  #check here

        )
    
def dense_block1(in_f, out_f, dense_dropout_rate, *args, **kwargs):
    return nn.Sequential(
        nn.Dropout(p=dense_dropout_rate, inplace=False),
        nn.Linear(in_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(3, stride = 1),
        nn.ReLU(),
    )

class raw_seq_model(nn.Module):
    def __init__(self, 
                 in_c, 
                 n_classes, 
                 num_filters = 64,
                 filter_length=25,
                 num_conv_layers=2,     
                 num_dense_layers=2,
                 conv_dropout_rate = 0.2,
                 dense_dropout_rate = 0.5,
                 bias = True, 
                 num_classes = num_classes,
                 return_logits = False
                 ):
        super(raw_seq_model, self).__init__() #why do i need to put model name again
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.return_logits = return_logits

        self.conv_layer1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=filter_length)
        self.batch_norm = nn.BatchNorm1d(num_filters)        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride = 1)
        self.conv_block1 = conv_block1(num_filters, num_filters, kernel_size=3, padding=1)
        self.dense_block1 = dense_block1(num_filters, 256, kernel_size=3, padding=1)

        self.linear_logit = nn.Linear(num_filters, 13) 
        self.linear_no_logit = nn.Linear(num_filters, 13) 
        self.predictions = nn.Sigmoid()
        self.global_maxpool = F.max_pool2d(x, kernel_size=x.size()[2:])


    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        for i in range(1, self.num_conv_layers + 1):
            x = self.conv_block1(x)
                    
        x = self.global_maxpool(x)

        for i in range(1, self.num_dense_layers + 1):
            x = self.dense_block1(x)

        if self.return_logits:
            prediction = self.linear_logit(x)
        
        else:
            prediction = self.linear_logit(x)
            prediction = self.predictions = nn.Sigmoid(prediction)
        return prediction
        


