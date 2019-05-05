import numpy as np
import pandas as pd
from glob import glob
import pickle as pkl
import vae


__all__ = ["get_scale_factor", "get_conv_shape", "get_conv_pad", "get_invconv_pad"]


def get_scale_factor(path):
    """
    Function to calculate the overall scalefactor for all the
    processed data. It opens all the metadata for every file
    and computes the weighted average of the scale factors for
    each dump
    
    Parameters
    ----------
    path : str
        Absolute path to metadata folder
    
    Returns
    -------
    scale : float
        The scale factor for all the processed data
    """
    
    files = glob(path+'*')
    scale_factors = []
    weights = []
    for f in files:
        meta = vae.io._utils._load_preprocessed_meta(f)
        scale_factors.append(meta['scale_factor'])
        weights.append(meta['n_traces'])
    scale = np.average(scale_factors, weights=weights)
    return scale

def get_conv_shape(tracelength, kernel_size, stride, pad, dilation, inverse=False):
    """
    Function to calculate the expected output shape of a convolution 
    and inverted convolution layer.
    
    Paramters 
    ---------
    tracelength : int
        Number of bins in input trace
    kernel_size : int
        Size of 1d kernel
    stride : int
        Size of stride
    pad : int
        Amount of zero padding used
    dilation : int
        Size of dilation
    inverse : Bool, optional
        If False, the shape of a convolution layer is returned
        IF True, the shape of an inverted convolution layer is
        rurtned
    
    Returns
    -------
    output : int
        The shape of a trace after the convolution
    """
    
    if inverse:
        output = (tracelength - 1) * stride - 2 * pad + kernel_size + pad
    else:
        output = (tracelength + (2 * pad) - (dilation * (kernel_size - 1)) - 1)// stride + 1
        
    return output
    

def get_conv_pad(input_size, output_size, kernel_size=1, stride=1, dilation=1):
    """
    Helper Function to calculate the nessesary padding based on desired outputsize 
    for a convolutional layer.
    
    Paramters 
    ---------
    input_size : int
        Number of bins in input trace
    output_size : int
        Number of bins in desired output trace
    kernel_size : int, optional
        Size of 1d kernel
    stride : int
        Size of stride
    dilation : int
        Size of dilation

    Returns
    -------
    pad : int
        the amount of padding needed
    """
    
    pad = ((output_size - 1)*stride - input_size + dilation*(kernel_size-1) + 1)
    
    return math.floor(pad/2)

def get_invconv_pad(input_size, output_size, kernel_size=1, stride=1, dilation=1, out_pad=0):
    """
    Helper Function to calculate the nessesary padding based on desired outputsize 
    for a inverse convolutional layer.
    
    Paramters 
    ---------
    input_size : int
        Number of bins in input trace
    output_size : int
        Number of bins in desired output trace
    kernel_size : int, optional
        Size of 1d kernel
    stride : int, optional
        Size of stride
    dilation : int, optional
        Size of dilation
    out_pad : int, optional
        The amount of output padding.

    Returns
    -------
    pad : int
        the amount of padding needed
    """
    
    pad = -(output_size - 1 - out_pad - dialation*(kernel_size-1) - (input_size - 1)*stride) / 2
    
    return math.floor(pad/2)