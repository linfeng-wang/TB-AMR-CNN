#%%
from array import array
from cmath import nan
from pyexpat import model
import statistics
from tkinter.ttk import Separator
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
# import icecream as ic

import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from importlib import reload
import util
import model_torch_simple
from torchmetrics import Accuracy
from tqdm import tqdm
import argparse

#%%

# lr = 0.001
# dr = 0.2
#%%

parser = argparse.ArgumentParser(description='Ioniazid prediction model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-lr", "--learning_rate", help='Learning rate for the model(between 10e-6 and 1)',default=0.001)
parser.add_argument("-dr", "--dropout_rate", help='Dropout rate for hte model layers (between 0 and 1)',default=0.2)

args = parser.parse_args()

lr = float(args.learning_rate)
dr = float(args.dropout_rate)
#%%
model_torch_simple = reload(model_torch_simple)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# load training data
train_data = pd.read_csv('data/training_data.csv')
val_data = pd.read_csv('data/validation_data.csv')

seqs_df_agg = train_data["KatG"].tolist()
res_all_combined = train_data["ISONIAZID"].tolist()

seqs_cryptic_agg = val_data["KatG"].tolist()
res_cryptic_combined = val_data["ISONIAZID"].tolist()

class RawReadDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y    
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

dataset = RawReadDataset(seqs_df_agg, res_all_combined) # dataset = CustomDataset(x_tensor, y_tensor)

#%%
def masked_BCE_from_logits(y_true, y_pred_logits):
    """
    Computes the BCE loss from logits and tolerates NaNs in `y_true`.
    """
    loss = nn.BCELoss()
    accuracy = Accuracy().to(device)
  
    # print("non_nan_ids:",non_nan_ids)
    # print("y_true.size:",y_true.size())
    y_pred_logits = y_pred_logits.squeeze(dim = -1)
    y_pred_logits =  y_pred_logits.to(device)
    y_true = torch.Tensor(y_true).to(device)
    y_true = y_true.float()
    y_pred_logits = y_pred_logits.float()

    # print("y_pred_logits.size:",y_pred_logits.size())
    # print(y_pred_logits)
    # print(y_pred_logits_non_nan)
    loss_value = loss(y_pred_logits, y_true)
    y_true = y_true.int()
    acc = accuracy(y_pred_logits, y_true)

    return loss_value, acc

#%%
train_dataset, val_dataset = random_split(dataset, [int(len(seqs_df_agg)*0.8), len(seqs_df_agg)-int(len(seqs_df_agg)*0.8)])
train_loader = DataLoader(dataset=train_dataset, batch_size=128)
val_loader = DataLoader(dataset=val_dataset, batch_size=128)

def one_hot_torch(seq):
    oh = []
    for sample in seq:
        sample = torch.ByteTensor(list(bytes(sample, "utf-8")))
        acgt_bytes = torch.ByteTensor(list(bytes("ACGT", "utf-8")))
        arr = torch.zeros((len(sample), 4), dtype=torch.int8)
        arr[sample == acgt_bytes[0], 0] = 1
        arr[sample == acgt_bytes[1], 1] = 1
        arr[sample == acgt_bytes[2], 2] = 1
        arr[sample == acgt_bytes[3], 3] = 1
        oh.append(arr)
    return oh

def my_padding(seq_tuple):
    list_x_ = list(seq_tuple)
    max_len = len(max(list_x_, key=len))
    for i, x in enumerate(list_x_):
        list_x_[i] = x + "N"*(max_len-len(x))
    return list_x_


#%%
# Testing conditions
#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#reloading model and running
model_torch_simple = reload(model_torch_simple)

# hyper-parameters
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        x = x.float()
        yhat = model(x)
        loss, acc = loss_fn(y, yhat)
        loss = torch.Tensor(loss)
        loss.requires_grad_()
        loss.backward()
        optimizer.step()
        # lrs.append(optimizer.param_groups[0]["lr"])
        # scheduler.step(0) # 0 used for learning rate policy 'plateau'
        optimizer.zero_grad()
        return loss.item(), acc
        
    return train_step


#%%
# for x_batch, y_batch in train_loader:
#     x_batch = my_padding(x_batch)
#     x_batch = one_hot_torch(x_batch)
#     x_batch = torch.stack(x_batch, dim=1).to(device)
#     x_batch = x_batch.permute(1, 2, 0).to(device)
#     print(x_batch.size())
#     break
#%%

# from torchviz import make_dot
# x = torch.randn(2, 4, 56).to(device)
# m = model_torch_simple.raw_seq_model().to(device)
# y = m(x)
# make_dot(y, params=dict(list(m.named_parameters()))).render("cnn_torchviz", format="png")

#%%




model = model_torch_simple.raw_seq_model(dense_dropout_rate = dr).to(device) # model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_step = make_train_step(model, masked_BCE_from_logits, optimizer)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,min_lr=1e-6, verbose=True)
n_epochs = 20
training_losses = []
validation_losses = []
lrs = []
training_acc = []
val_acc = []


# print(model.state_dict())

# training
for epoch in tqdm(range(n_epochs)):
    batch_losses = []
    batch_acc_train = []
    for x_batch, y_batch in train_loader:
        x_batch = my_padding(x_batch)
        x_batch = one_hot_torch(x_batch)
        x_batch = torch.stack(x_batch, dim=1).to(device)
        x_batch = x_batch.permute(1, 2, 0).to(device)
        # print(x_batch.size())
        # print(y_batch.size())
        # print(y_batch)
        y_batch = torch.Tensor(y_batch).to(device)

        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss)
        batch_acc_train.append(acc.item())
        #break
    # print(batch_acc_train)
    # print(type(batch_acc_train))
    epoch_acc_training = np.mean(batch_acc_train)
    training_acc.append(epoch_acc_training)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        batch_acc_val = []
        for x_val, y_val in val_loader:
            x_val = my_padding(x_val)
            x_val = one_hot_torch(x_val)
            x_val = torch.stack(x_val, dim=1).to(device)
            x_val = x_val.float()
            x_val = x_val.permute(1, 2, 0).to(device)
            # y_val = torch.stack(y_val, dim=1).to(device)
            model.eval()
            yhat = model(x_val)
            val_loss, acc = masked_BCE_from_logits(y_val, yhat)#.item()
            val_losses.append(val_loss.item())
            batch_acc_val.append(acc.item())
            #break
        
        epoch_acc_val = np.mean(batch_acc_val)
        val_acc.append(epoch_acc_val)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)
    
    # scheduler.step(validation_loss)
    lrs.append(optimizer.param_groups[0]["lr"])
    # print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
    # print(f"[{epoch+1}] Training Accuracy: {epoch_acc_training:.3f}\t Validation Accuracy: {epoch_acc_val:.3f}")
    # print("="*20)
    #break
# print(model.state_dict())
# torch.save(model.state_dict(), '/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/pytorch_model-simple')

# model = model_torch_batch.raw_seq_model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

d = {'training_losses':training_losses,
     'validation_losses':validation_losses,
     'Training Accuracy': training_acc, 
     'Validation Accuracy': val_acc}
history = pd.DataFrame.from_dict(d)
pd.DataFrame(history).to_csv("/mnt/storageG1/lwang/TB-AMR-CNN/code_torch/training-history-simple.csv",index=False)

fig, ax = plt.subplots()
x = np.arange(1, n_epochs+1, 1)
ax.plot(x, history["training_losses"],label='Training')
ax.plot(x, history["validation_losses"],label='Validation')
ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Loss")
ax.set_xticks(np.arange(0, n_epochs+1, 10))
ax.set_title(f'Learning_rate:{lr}-Dropout_rate:{dr}')
# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig(f"IND-model_LR:{lr}-DR:{dr}-LOSS.png")

fig, ax = plt.subplots()
x = np.arange(1, n_epochs+1, 1)
ax.plot(x, history["Training Accuracy"],label='Training')
ax.plot(x, history["Validation Accuracy"],label='Validation')
ax.legend()
ax.set_xlabel("Number of Epoch")
ax.set_ylabel("Accuracy")
ax.set_xticks(np.arange(0, n_epochs+1, 10))
ax.set_title(f'Learning_rate:{lr}-Dropout_rate:{dr}')

# ax_2 = ax.twinx()
# ax_2.plot(history["lr"], "k--", lw=1)
# ax_2.set_yscale("log")
# ax.set_ylim(ax.get_ylim()[0], history["training_losses"][0])
ax.grid(axis="x")
fig.tight_layout()
fig.show()

fig.savefig(f"IND-model_LR:{lr}-DR:{dr}-ACC.png")

# %%
