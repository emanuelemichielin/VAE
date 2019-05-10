# VAE

Collection of tools for testing the idea of using a Variational autoencoder for event reconstruction. The directory `working_dir` holds various working files and notebooks, and the directory `vae` holds the VAE module tools. 

The tools can be generalized to any single or multi-channel time series data, but most of the IO functionality of this module at the moment only applies to (private) data from a UCB Pyle group R&D detector called PD2.

It is highly recommended that this be run on a system with a GPU and multiple CPUs.

### installation

To install and use these tools, from the top level directory do the following in the command line:

```
>>> python setup.py clean
>>> python setup.py install --user
```

Among the standard packages included with Anaconda, the following dependencies are needed: `deepdish`, `QETpy`, and `RQpy`, `PyTorch`. Most of these can be installed via

```
>>> pip install -r requirements.txt
```
Except `RQpy` which is not yet avaliable on PyPi. To install `RQpy` clone the repository https://github.com/ucbpylegroup/RQpy and follow the install instructions in the README.md. It is also recommended that the most current development version of `QETpy` is used, which can be cloned and installed from https://github.com/ucbpylegroup/QETpy

Both `QETpy` and `RQpy` are only used for processing raw data. These will be made into optional dependancies later.

### Usage 

There are two useful files in `working_dir/model_dev/` that will demostrate the usage of the tools. 

`train_CNN.py` is a file used to train the VAE model
`visualize_trained_models.ipynb` is a Jupyter notebook used to load saved models and test performace

All the main functions and classes in the module are available in `vae/core/`. The primary tools include:

*`PD2dataset` : A dataset object to be used with `PyTorches` data loader

*`VAE` : A CNN based Varational Autoencoder model class

*`beta_mse_loss` : A $\beta$ VAE loss function

*`Trainer` : A class used for training models. This object performs the optimization and calculation of training/validataion loss. It also supports loading saved model and optimizer checkpoints

Also available from the base import are a few plotting functions

*`plot_loss` : Plots the loss vs training step

*`plot_recon` : Makes plots of original and reconstructed events

*`plot_latent_2d` : Plots a 2d scatter plot of the latent space variables

Some of the functions in `vae/utils/` may be helpful, particularly

*`reduce_dims` : performs dimensionality reduction (for the latent variables) using either PCA or t-SNE



It is recommended to import as `vae` NOT `from vae import *`!

__Example__
```python
import vae

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


model_path = 'path_to_model'

# which normalization to use
file_mapping = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v6/file_mapping.h5'
tracelength = 1625
epochs = 25

scale_factor = 2.5e-6
offset = 0.25
learning_rate = 1e-4
z_hidden = 15
beta = 1

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
```

                                       
                                

