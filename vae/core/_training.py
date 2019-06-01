###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains a Training object and checkpoint loader
# for training the CNN based VAE. Both are able to be
# imported from the base level of the module. 
###############################################################

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F
import vae

__all__ = ["Trainer", "load_checkpoint"]



class Trainer(object):
    """
    Object to train and test a Variational Autoencoder model with 
    PyTorch, and store the loss function as a function of training
    step. 
    """
    def __init__(self, 
                 model, 
                 train_dl, 
                 test_dl, 
                 loss_func, 
                 nepochs, 
                 starting_epoch=0,
                 optimizer_type='Adam',
                 lr=1e-3,
                 optim_kwargs={},
                 loss_kwargs={},
                 savemodel=False,
                 savename='',
                 ncheckpoints=1,
                ):
        """
        Initilization of Trainer object. 
        
        Parameters
        ----------
        model : pytorch model
            Initialized VAE model (Note, it is already assumed
            that the model has already been moved to the GPU)
        train_dl : pytorch DataLoader object
            The dataloader for the training data
        test_dl : pytorch DataLoader object
            The dataloader for the test data
        loss_func : function
            Loss function to use for VAE. Note, the input to
            the loss function must be of the form:
            (recon_batch, x, mu, scale,)
        nepochs : int
            Number of epochs to train over
        starting_epoch : int, 
            What epcoh the training should start on
        optimizer_type : str, optional
            Type of optimizer to use, must correspond
            to an available algorithm from torch.optim
        lr : float, optional
            Learning rate for optimizer.
        optim_kwargs : dict, optional
            optional key word args for the
            optimizer.
        loss_kwargs : dict, optional
            optional key word aregs for the
            loss function.
        savename : str, optional
            Path + file name for model and
            optimizer settings to be saved
        ncheckpoints : int, str, array-like, optional
            How many check points to save model at. 
            If 1 (default), the model is only saved
            at the end of training. If 'all', it is 
            saved every epoch. If a list or array is passed,
            it is saved for the corresponding indices. 
        
        """
        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        
        try:
            optimizer = eval(f'optim.{optimizer_type}(model.parameters(), lr=lr, **optim_kwargs)')
        except:
            raise ValueError('Invalid optimizer type or kwargs')
        
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.nepochs = nepochs
        self.starting_epoch =  starting_epoch
        self.training_loss = []
        self.testing_loss = []
        self.device = device
        stepsize = len(train_dl)//10
        if stepsize == 0:
            stepsize = len(train_dl)
        self._step_size = stepsize
        self.savemodel=savemodel
        self.savename=savename
        if ncheckpoints is 1:
            self.ncheckpoints=[self.nepochs]
        elif ncheckpoints is 'all':
            self.ncheckpoints=np.arange(self.nepochs)
        else:
            self.ncheckpoints = ncheckpoints
        
        
    def train(self, verbose=False, calc_test_loss=False):
        """
        Function to train model.
        
        Parameters
        ----------
        verbose : Bool, optional
            If True, training progress is displayed
        calc_test_loss : Bool, optional
            If True, the loss function is calculated
            for the testing data every epoch. Defaults
            to False (i.e. only the loss for training
            data is calculated)
        """        
        
        for epoch in range(self.starting_epoch, self.nepochs):
            train_loss = 0
            self.model.train()
            for batch_idx, (x, _) in enumerate(self.train_dl):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, scale = self.model(x)
                loss = self.loss_func(recon_batch, x, mu, scale, **self.loss_kwargs)
                loss.backward()
                train_loss += loss.item()
                self.training_loss.append(loss)
                self.optimizer.step()
                if verbose:
                    if batch_idx % self._step_size == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(x), len(self.train_dl.dataset),
                            100. * batch_idx / len(self.train_dl),
                            loss.item() / len(x)))
            if verbose:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                      epoch, train_loss / len(self.train_dl.dataset)))
            if calc_test_loss:
                self.testing_loss.append(self.test(verbose=verbose))
            if self.savemodel:
                if epoch in self.ncheckpoints:
                    state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'loss_train' : self.training_loss,
                             'loss_val' : self.testing_loss}
                    torch.save(state, f'{self.savename}_epoch{str(epoch).zfill(4)}.pt')
                
    def test(self, verbose=False):
        """
        Function to calculate the loss function for the test data
        
        Parameters
        ----------
        verbose : Bool, optional
            If True, training progress is displayed
        
        Returns
        -------
        test_loss : float
            The average loss for all the batches in the 
            testing data
        """
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for ii, (x, _) in enumerate(self.test_dl):
                x = x.to(self.device)
                recon_batch, mu, scale = self.model(x)
                test_loss += self.loss_func(recon_batch, x, mu, 
                                            scale, **self.loss_kwargs).item()
            test_loss /= len(self.test_dl.dataset)
            if verbose:
                print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss
    
    
def load_checkpoint(path,  
                    z_dims,  
                    train_dl, 
                    test_dl, 
                    loss_func, 
                    nepochs, 
                    optimizer_type='Adam',
                    lr=1e-3,
                    optim_kwargs={},
                    loss_kwargs={},
                    savemodel=False,
                    savename='',
                    ncheckpoints=1,
                    extra_model_params={}):
    """
    Function to load a previously saved model for continued training. 
    This function initalizes and returns a Trainer object. Note, the
    saved checkpoint is assumed to have saved the state_dict for both
    the model, AND the optimizer. Also the lists for the training and 
    testing loss. The settings must be saved in a dictionary with keys:
    
    dict_keys(['epoch', 'state_dict', 'optimizer', 'loss_train', 'loss_val'])
    where 'sate_dict' is the model state dict
          'optimizer' is the optimizer state dict
    
    Parameters
    ----------
    path : str,
        Absolute path to checkpoint to load
    z_dims : int
        The number of latend dimensions in the saved 
        model checkpoint
    train_dl : pytorch DataLoader object
        The dataloader for the training data
    test_dl : pytorch DataLoader object
        The dataloader for the test data
    loss_func : function
        Loss function to use for VAE. Note, the input to
        the loss function must be of the form:
        (recon_batch, x, mu, scale,)
    nepochs : int
        Number of epochs to train over
    starting_epoch : int, 
         What epcoh the training should start on
    optimizer_type : str, optional
        Type of optimizer to use, must correspond
        to an available algorithm from torch.optim
    lr : float, optional
        Learning rate for optimizer.
    optim_kwargs : dict, optional
        optional key word args for the
        optimizer.
    loss_kwargs : dict, optional
        optional key word aregs for the
        loss function.
    savename : str, optional
        Path + file name for model and
        optimizer settings to be saved
    ncheckpoints : int, str, array-like, optional
        How many check points to save model at. 
        If 1 (default), the model is only saved
        at the end of training. If 'all', it is 
        saved every epoch. If a list or array is passed,
        it is saved for the corresponding indices. 
    extra_model_params : dict, optional
        Any additional optional parameters for the model
        when initialized, to ensure the shapes of the 
        layers are correct
    
    Returns
    -------
    trainer : vae.core.Trainer 
        Trainer object to continue training
        from saved checkpoint.
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    model = vae.VAE(z_dims, **extra_model_params).to(device)
    try:
        checkpoint = torch.load(path)
    except:
        raise FileNotFoundError('Unable to load file.')
        
    try:
        optimizer = eval(f'optim.{optimizer_type}(model.parameters())')
    except:
        raise ValueError('Invalid optimizer type or kwargs')
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
    #optimizer.to(device)
    epoch = checkpoint['epoch']
    training_loss = checkpoint['loss_train']
    testing_loss = checkpoint['loss_val']
    
    trainer = Trainer(model=model, 
                      train_dl=train_dl, 
                      test_dl=test_dl, 
                      loss_func=loss_func, 
                      nepochs=nepochs, 
                      starting_epoch=epoch,
                      optimizer_type=optimizer_type,
                      lr=lr,
                      optim_kwargs=optim_kwargs,
                      loss_kwargs=loss_kwargs,
                      savemodel=savemodel,
                      savename=savename,
                      ncheckpoints=ncheckpoints)
    
    trainer.optimizer = optimizer
    trainer.training_loss = training_loss
    trainer.testing_loss = testing_loss
    
    return trainer
     