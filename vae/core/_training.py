
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F

__all__ = ["Trainer"]



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
                 optimizer_type='Adam',
                 lr=1e-3,
                 optim_kwargs={},
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
            Loss function to use for VAE
        nepochs : int
            Number of epochs to train over
        optimizer_type : str, optional
            Type of optimizer to use, must correspond
            to an available algorithm from torch.optim
        lr : float, optional
            Learning rate for optimizer.
        optim_kwargs : dict, optional
            
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
        self.nepochs = nepochs
        self.training_loss = []
        self.testing_loss = []
        self.device = device
        
        self._step_size = len(train_dl)//10
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
        loss_train_list = []
        
        
        for epoch in range(self.nepochs):
            train_loss = 0
            self.model.train()
            for batch_idx, (x, _) in enumerate(self.train_dl):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, scale = self.model(x)
                loss = self.loss_func(recon_batch, x, mu, scale)
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
                    torch.save(state, f'{self.savename}_epoch{epoch}.pt')
                
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
        #test_loss_batch = []
        with torch.no_grad():
            for ii, (x, _) in enumerate(self.test_dl):
                x = x.to(self.device)
                recon_batch, mu, scale = self.model(x)
                test_loss += self.loss_func(recon_batch, x, mu, scale).item()
                #test_loss_batch.append(test_loss)
            test_loss /= len(self.test_dl.dataset)
            if verbose:
                print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss
    
    
