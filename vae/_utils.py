import numpy as np
import pandas as pd
from glob import glob
import vae

__all__ = ["get_scale_factor"]


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
        meta = vae.load_preprocessed_meta(f)
        scale_factors.append(meta['scale_factor'])
        weights.append(meta['n_traces'])
    scale = np.average(scale_factors, weights=weights)
    return scale

