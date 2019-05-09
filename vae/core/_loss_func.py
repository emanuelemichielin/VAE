###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains 3 differnet loss functions to be used with
# VAE. All are able to be
# imported from the base level of the module. 
###############################################################

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["beta_mse_loss", "mse_loss", "cross_entropy_loss"]


def mse_loss(recon_x, x, z_loc, z_scale):
    """
    Loss function for VAE. The mean squared
    error (MSE) plus the Kullback–Leibler divergence
    (KLD). 
    
    Parameters 
    ----------
    recon_x : tensor
        Reconstructed input data 
        (x -> encoder -> decoder -> recon_x)
    x : tensor
        Input data
    z_loc : tensor
        Latent space variables
    z_scale :t tensor
        Latent space variables

    Returns
    -------
    loss : tensor
        Evaluated Loss function
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_scale - z_loc.pow(2) - z_scale.exp())
    return BCE + KLD

def beta_mse_loss(recon_x, x, z_loc, z_scale, beta=1):
    """
    Loss function for VAE. The mean squared
    error (MSE) plus the Kullback–Leibler divergence
    (KLD), but with a scaling hyper-parameter beta on
    the KLD term
    
    Parameters 
    ----------
    recon_x : tensor
        Reconstructed input data 
        (x -> encoder -> decoder -> recon_x)
    x : tensor
        Input data
    z_loc : tensor
        Latent space variables
    z_scale :t tensor
        Latent space variables

    Returns
    -------
    loss : tensor
        Evaluated Loss function
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_scale - z_loc.pow(2) - z_scale.exp())
    return BCE + beta * KLD
 


def cross_entropy_loss(recon_x, x, z_loc, z_scale):
    """
    Loss function for VAE. The binary cross entropy
    plus the Kullback–Leibler divergence
    (KLD). 
    
    Parameters 
    ----------
    recon_x : tensor
        Reconstructed input data 
        (x -> encoder -> decoder -> recon_x)
    x : tensor
        Input data
    z_loc : tensor
        Latent space variables
    z_scale :t tensor
        Latent space variables

    Returns
    -------
    loss : tensor
        Evaluated Loss function
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_scale - z_loc.pow(2) - z_scale.exp())
    return BCE + KLD

