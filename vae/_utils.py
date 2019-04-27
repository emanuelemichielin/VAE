import numpy as np
import pandas as pd
from glob import glob
import pickle as pkl
import vae


__all__ = ["get_scale_factor", "store_rq_labels"]


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

def store_rq_labels(traces_path, rq_path, savepath):
    """
    Utility function to store all conventional features 
    from PD2 DM analysis for corresponding events. 
    
    Parameters
    ----------
    traces_path : list of str
        Absolute path to folder of processed traces
    rq_path : str
        Absolute path to RQ dataframe
    savepath : str
        Absolute path where the labels should be saved
    """
    with open(rq_path, 'rb') as rq_file:
        rq = pkl.load(rq_file)
    rq.sort_values('ser_ev', inplace=True)
    
    for p in traces_path: 
        traces, eventnumbers = vae.load_preprocessed_traces(p)
        label = f"{p.split('/')[-1][:20]}labels.h5"
        cuts = np.zeros(rq.shape[0], dtype=bool)
        for ev in eventnumbers:
            cuts = cuts | (rq.ser_ev == ev)
        df_labels = rq[cuts]
        if not np.all(df_labels.ser_ev == eventnumbers):
            raise ValueError('Shape dump and rq labels do not match')
        df_labels.to_hdf(savepath+label, 'labels')
