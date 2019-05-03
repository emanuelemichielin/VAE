import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch


__all__ = ["plot_loss"]

def plot_loss(trainer=None, training_loss=None, test_loss=None, nper_epoch=None):
    """
    Function to plot the loss during training and testing of model. Must provide either
    the Trianer object, or the training_loss/test_loss explicitly.
    
    Paramters
    ---------
    trainer : vae.Trainer object, optional
        Trainer object used to train the model. This will
        have all the realavent loss information saved internally. 
    training_loss : list, array, optional
        The training loss for each step during training. 
    test_loss : list, array, optional
        The testing loss averaged for each epoch. Note, 
        the testing loss is assumed to calculated at each
        epoch rather than batch. To make the arrays match, the 
        number of batches per epoch (nper_epoch) used during training must be passed. If 
        the test loss was calculated for every batch, set nper_epoch
        to 1. 
    nper_epoch : int, optional
        The number of batches needed to finish an epoch. 
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    if trainer is not None:
        nper_epoch = len(trainer.train_dl)
        training_loss = trainer.training_loss
        if len(trainer.testing_loss) > 0:
            test_loss = trainer.testing_loss
        else:
            test_loss = None
    elif training_loss is None:
        raise ValueError('must provide either Trainer object or training_loss.')
    
    if ((test_loss is not None) & (nper_epoch is None)):
        raise ValueError('must provide nper_epoch to plot test_loss')
        
    
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(training_loss, label = 'Training set')
    if test_loss is not None:
        ax.plot(np.arange(1, len(test_loss)+1)*nper_epoch,
                test_loss, label='Validation set')

    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--')
    ax.tick_params(which = 'both', tickdir = 'in', top = True, 
                       right = True)
    ax.set_title('Loss vs Number of Training Steps')

    for ep in range(1, num_epochs+1):
        if ep == 1:
            ax.axvline(ep*total_step, linestyle = ':', alpha = 0.1, color = 'g',
                      label = 'Epoch')
        else:
            ax.axvline(ep*total_step, linestyle = ':', alpha = 0.1, color = 'g')

    ax.set_yscale('log')
    ax.legend()
        
    return fig, ax

def plot_recon(dataloader, model, nplots='batch'):
    with torch.no_grad():
    for i, (x, _) in enumerate(test_loader):
        x = x.to(device)
        recon_batch, mu, logvar = model(x)
        x = x.cpu().detach().numpy()
        recon_batch = recon_batch.cpu().detach().numpy()
        break