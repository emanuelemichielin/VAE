##############################################################################
# Template to train VAE models for PD2 data. To run, change the following:
#    * model_path
#    * epochs
#    * learning_rate  
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
partition = vae.io.load_partition(PARTITION_PATH, 'good_triggers_no_low_tight')

### Define params
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 10, 
          'pin_memory' : True,
          'drop_last' : False}
#####
truncate = 812
######

# extra_model_params = {'kernels' : [3]*4,
#                       'strides' : [2]*4, 
#                       'final_pad' : 0,
#                       'final_kernel' : 10}
# extra_model_params = {'kernels' : [3]*4,
#                       'strides' : [1]*4, 
#                       'final_pad' : 1,
#                       'final_kernel' : 1}
# extra_model_params = {'kernels' : [9]*4,
#                       'strides' : [3]*4, 
#                       'final_pad' : 1,
#                       'final_kernel' : 10} 
# extra_model_params = {'kernels' : [19]*4,
#                       'strides' : [2]*4, 
#                       'final_pad' : 1,
#                       'final_kernel' : 12}

extra_model_params = {'tracelength' : truncate,
                      'kernels' : [8]*4,
                      'strides' : [2]*4, 
                      'final_pad' : 1,
                      'final_kernel' : 5}




#model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/good_triggers_cal/45_dims_beta_point5_lr1e-3_drop_dif_offset/'
model_path = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/Run44_short_wiener_trunc/good_triggers_no_low_tight/25_b_5_s2k8_nodrop_sf5_lr1e4/'


file_mapping = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_short_wiener/file_mapping.h5'
tracelength = 1625


epochs = 10000

#scale_factor = 4e-6
scale_factor = 5e-6
offset = 0.25
learning_rate = 1e-4
z_hidden = 25
beta = .5
dropout=False

verbose=True# if you want the model to print its progress
calc_test=True#if you want the testing loss calculated every epoch

# if you want to load a saved model to continue, change this to true and provide the path to the epoch to start from
load_saved=False
loadpath='/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/models/v8/good_triggers_cal/30_bpoint5_full_s2k3_nodrop/model_epoch0370.pt'


# make dirs for model to save
isdir = os.path.isdir(model_path)
if not isdir:
    os.makedirs(model_path)
    
# Save settings    
settings = {'scale_factor' : scale_factor, 'offset' : offset, 'learning_rate' : learning_rate,
           'z_hidden' : z_hidden, 'beta' : beta, 'tracelength' : tracelength, 'usedropout':dropout, 
           'model_params' : extra_model_params}

with open(model_path+'settings.pkl', 'wb') as setfile:
    pkl.dump(settings, setfile)




### Generators
training_set = vae.PD2dataset(partition['train'], 
                              labels=None,
                              map_path=file_mapping,
                              max_normed=False,
                              baseline_sub=True, 
                              offset=offset,
                              scaledata=scale_factor,
                              tracelength=tracelength,
                              truncate=truncate)
train_loader = data.DataLoader(training_set, **params)

test_set = vae.PD2dataset(partition['validation'], 
                          labels=None, 
                          map_path=file_mapping,
                          max_normed=False, 
                          baseline_sub=True, 
                          offset=offset,
                          scaledata=scale_factor,
                          tracelength=tracelength,
                          truncate=truncate)
test_loader = data.DataLoader(test_set, **params)



### Load Model
#model = vae.VAE(z_hidden, usedropout_encode=dropout).to(device)
#model = vae.VAE(z_hidden, usedropout_encode=dropout, kernels=[4]*4, strides=[2]*4, final_kernel=11, final_pad=0).to(device)
#model = vae.VAE(z_hidden, usedropout_encode=dropout, kernels=[8]*4, strides=[4]*4, final_kernel=5, final_pad=0).to(device)
#model = vae.VAE(z_hidden, strides=[4]*4, final_pad=0, final_kernel=5).to(device)
#model = vae.VAE(z_hidden, usedropout_encode=dropout, kernels=[3]*4, strides=[2]*4, final_pad=0, final_kernel=10).to(device)

model = vae.VAE(z_hidden, usedropout_encode=dropout, **extra_model_params).to(device)
### Print model
summary(model.encoder, (1, truncate))
summary(model.decoder, (1, z_hidden))



### Train model
if load_saved:
    print('====> loading saved model')
    trainer = vae.load_checkpoint(path=loadpath,
                                  z_dims=z_hidden,
                                  train_dl=train_loader,
                                  test_dl=test_loader, 
                                  optimizer_type='Adam', 
                                  loss_func=vae.beta_mse_loss, 
                                  nepochs=epochs,
                                  lr=learning_rate,
                                  loss_kwargs={'beta':beta},
                                  savemodel=True,
                                  savename=f'{model_path}model',
                                  ncheckpoints='all',
                                  extra_model_params=extra_model_params)
else:
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
trainer.train(verbose=verbose, calc_test_loss=calc_test)



        
