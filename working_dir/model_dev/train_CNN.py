##############################################################################
# Template to train VAE models for PD2 data. To run, change the following:
#    * model_path
#    * epochs
#    * learning_rate = 1e-4
#    * z_hidden
#    * beta
#    * the partition being loaded from vae.io.load_partition
#
#
#
# It is currently set to save the model at every epoch. This can be changed by 
# setting ncheckpoints = a list of epochs to save at. Or set this equal to 1 to
# just save at the end of training.
#
# It is advised not to change, scale_factor, offset. Also, tracelength must be
# 1625 for the model to work with the current data. This will hopefully be made
# more flexable in future versions. 
#
# To run, from the current directory, >>> python train_model.py
#
##############################################################################






import torch
from torch.utils import data
import vae
from glob import glob
import numpy as np
import pandas as pd
import time
from vae import TRACE_PATH_DS, \
                META_PATH_DS, \
                LABEL_PATH_DS, \
                RQ_DF_PATH, \
                EVENT_FILE_MAP_PATH, \
                PARTITION_PATH, \
                PD2_LABEL_COLUMNS
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
import os


### load data partition
partition = vae.io.load_partition(PARTITION_PATH, 'good_triggers_cal')

### Define params
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 20, 
          'pin_memory' : True,
          'drop_last' : False}

# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/v6_8_flat_beta50/'
# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/v6_8_flat_beta100/'
# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/v6_8_flat_beta5/'
# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/v6_8_flat_beta1e-2/'
# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/bagged_data/6_dims_beta1/'
# model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/bagged_data/20_dims_beta1/'
model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/good_triggers_cal/15_dims_beta1/'

file_mapping = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v6/file_mapping.h5'
tracelength = 1625
epochs = 250

scale_factor = 2.5e-6
offset = 0.25
learning_rate = 1e-4
z_hidden = 15
# beta = 50
# beta = 100
beta = 1
saverate = 4




# make dirs for model to save
isdir = os.path.isdir(model_path)
if not isdir:
    os.makedirs(model_path)
    
# Save settings    
settings = {'scale_factor' : scale_factor, 'offset' : offset, 'learning_rate' : learning_rate,
           'z_hidden' : z_hidden, 'beta' : beta, 'tracelength' : tracelength}

with open(model_path+'settings.pkl', 'wb') as setfile:
    pkl.dump(settings, setfile)

saveinds = np.arange(1, epochs, saverate)


### Generators
training_set = vae.PD2dataset(partition['train'], 
                              labels=None,
                              map_path=file_mapping,
                              max_normed=False,
                              baseline_sub=True, 
                              offset=offset,
                              scaledata=scale_factor,
                              tracelength=tracelength)
train_loader = data.DataLoader(training_set, **params)

test_set = vae.PD2dataset(partition['validation'], 
                          labels=None, 
                          map_path=file_mapping,
                          max_normed=False, 
                          baseline_sub=True, 
                          offset=offset,
                          scaledata=scale_factor,
                          tracelength=tracelength)
test_loader = data.DataLoader(test_set, **params)


### Load Model
model = vae.VAE(z_hidden).to(device)

### Print model
summary(model.encoder, (1, 1624))
summary(model.decoder, (1, z_hidden))


### Train model
trainer = vae.Trainer(model=model, 
                      train_dl=train_loader,
                      test_dl=test_loader, 
                      optimizer_type='Adam', 
                      loss_func=vae.beta_mse_loss, 
                      nepochs=epochs,
                      lr=learning_rate,
                      loss_kwargs={'beta':beta},
                      savemodel=True,
                      savename=f'{model_path}model',
                      ncheckpoints='all')
trainer.train(verbose=True, calc_test_loss=True)



        
