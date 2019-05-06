import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import numpy as np
import torch


__all__ = ["plot_loss", "plot_recon", "plot_latent_2d"]

def plot_loss(trainer=None, training_loss=None, 
              test_loss=None, nper_epoch=None, 
              nepochs=None, savefig=False, filename=None):
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
    nepochs : int, optional
        The number of epochs used for training.
    savefig : Bool, optional
        If True, the figure is saved
    filename : str, optional
        The abosolute path+name for the figure to be
        saved.
        
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
    if nepochs is not None:
        for ep in range(1, nepochs+1):
            if ep == 1:
                ax.axvline(ep*nper_epoch, linestyle = ':', alpha = 0.1, color = 'g',
                          label = 'Epoch')
            else:
                ax.axvline(ep*nper_epoch, linestyle = ':', alpha = 0.1, color = 'g')

    ax.set_yscale('log')
    ax.legend()
    if savefig:
        try:
            fig.savefig(filename+'.png', dpi=300)
        except:
            raise ValueError('Must provide a valid filepath')
    return fig, ax
        
    return fig, ax

def plot_recon(dataloader, model, nplots=10, xlims=None, ylims=None,
              savefig=False, filename=None):
    """
    Function to plot original traces and the equivalent trace
    reconstructed by the model. 
    
    Parameters
    ----------
    dataloader : torch.Dataloader object
        The dataloader corresponding to the 
        data you want to visualize
    model : torch.nn.Model
        VAE model to use for reconstruction
    nplots : int, optional 
        The number of traces to reconstruct
        and plot. Each trace will be drawn in 
        it's own figure. 
    xlims : tuple, NoneType, optional
        The xlims for the figures. If None (default), 
        the limits will be determined by matplotlib
    ylims : tuple, NoneType, optional
        The ylims for the figures. If None (default), 
        the limits will be determined by matplotlib
    savefig : Bool, optional
        If True, the figure is saved
    filename : str, optional
        The abosolute path+name for the figure to be
        saved.
    """
    
    if nplots > 16:
        raise ValueError('Took many figures to open at once. Please call this function in a loop')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    model.eval()
    nworkers = dataloader.num_workers
    dataloader.num_workers = 1
    
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            recon_batch, mu, logvar = model(x)
            x = x.cpu().detach().numpy()
            recon_batch = recon_batch.cpu().detach().numpy()
            break
    for ii in range(nplots):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_xlabel('Time [Arbitraty Units]')
        ax.set_ylabel('Amplitude [Arbitrary Units]')
        ax.grid(True, linestyle='--')
        ax.tick_params(which = 'both', tickdir = 'in', top = True, 
                       right = True)
        ax.plot(x[ii, 0, ], label='original')
        ax.plot(recon_batch[ii, 0, ], label='reconstructed')
        ax.legend()
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        if savefig:
            try:
                fig.savefig(f'{filename}{ii}.png', dpi=300)
            except:
                raise ValueError('Must provide a valid filepath')
    return fig, ax    
    dataloader.num_workers = nworkers
            
    
def plot_latent_2d(latent_vars, labels=None, label_name=None, pltkwargs={}, 
                   savefig=False, filename=None):
    """
    Function to plot latent variables as a scatter plot colored by
    ground truths (optional). If the latent variables are not 2d, 
    first reduce the dimentions using either tSNE, PCA, or random 
    projections + other combo. 
    
    Parameters
    ----------
    latent_vars : array
        Array of events expressed in latent space. 
        Should be 2d. If more than 2 laten dims, reduce
        the diminsion before call this function
    labels : array, optional
        Array corresponding to the ground truth values. 
        If not specified, the data will not be colored.
    label_name : str, optional
        The name of value for the label (i.e. energy, optimum filter, etc)
    pltkwargs : dict, optional
        Optional key word arguments to be passed to 
        matplotlib.pyplot.scatter()
    savefig : Bool, optional
        If True, the figure is saved
    filename : str, optional
        The abosolute path+name for the figure to be
        saved.
        
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title('Latent Space Representation')
    ax.set_xlabel('Z_1 [Arbitrary Units]')
    ax.set_ylabel('Z_2 [Arbitrary Units]')
    ax.grid(True, linestyle='--')
    ax.tick_params(which = 'both', tickdir = 'in', top = True, 
                   right = True)
    if labels is None:
        labels = 'b'
    clr = ax.scatter(latent_vars[:, 0], latent_vars[:, 1], c=labels, **pltkwargs)
    if label_name:
        fig.colorbar(clr, label=label_name)
        
    if savefig:
        try:
            fig.savefig(filename+'.png', dpi=300)
        except:
            raise ValueError('Must provide a valid filepath')
    return fig, ax

        