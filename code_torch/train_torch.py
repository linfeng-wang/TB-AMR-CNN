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

import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

import util
import model_torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#%%
# load training data
seqs_df, res_all = util.load_data.get_main_dataset()
N_samples = seqs_df.shape[0]
DRUGS = util.DRUGS
assert set(DRUGS) == set(res_all.columns)
N_drugs = len(DRUGS)

#%%
# load the CRyPTIC samples as test data
seqs_cryptic, res_cryptic = util.load_data.get_cryptic_dataset()
# make sure the loci are in the same order as in the training data
seqs_cryptic = seqs_cryptic[seqs_df.columns]

#%%
seq = 'ATGCN'
#%%
def one_hot(seq):
    seq = list(seq)
    o_encoder = OneHotEncoder(sparse=False)
    o_encode = o_encoder.fit(np.array(seq).reshape(-1, 1))
    return o_encode

oh_encoder = one_hot(seq)
#%%
#%%

#%%
seq_array = []
for x in range(len(seqs_df)):
    seq = "".join(list(seqs_df.iloc[x,:]))
    seq = np.array(seq).reshape(-1, 1)
    print(seq)break
    # trans = oh_encoder.transform(seq)
    seq_array.append(seq)

#%%

# %%
res_all = res_all.fillna(0)
#%%
# %%
np.array(seq).reshape(1, -1)

#%%



#%%
x_tensor = torch.from_numpy(seqs_df).float()
y_tensor = torch.from_numpy(res_all).float()






class RawReadDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)
