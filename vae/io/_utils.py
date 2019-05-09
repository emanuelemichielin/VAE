###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains helper io functions for loading 
# dumps of events. These are meant to be hidden from
# the user
###############################################################

import numpy as np
import pandas as pd
import qetpy as qp
import rqpy as rp
import deepdish as dd
import os
from glob import glob
import pickle as pkl
from vae import PD2_LABEL_COLUMNS






def _save_preprocessed(savepath, traces, metadata):
    """
    Function of save the pre processed data
    
    Parameters
    ----------
    savepath : str, optional
        The path where the processed dumps should be saved. The 
        traces will be saved in savepath/traces/ and the metadata in 
        savepath/metadata/
    traces : dict
        The processed trace dict returned from pre_process_PD2
    metadata : dict
        the metadata returned from pre_process_PD2
    """
    savename = metadata['dump']
    if not os.path.isdir(f'{savepath}traces'):
        os.makedirs(f'{savepath}traces')
    if not os.path.isdir(f'{savepath}metadata'):
        os.makedirs(f'{savepath}metadata')
    
    dd.io.save(f'{savepath}traces/{savename}_traces.h5', traces)
    dd.io.save(f'{savepath}metadata/{savename}_metadata.h5', metadata)
    
def _load_preprocessed_traces(path):
    """
    Function to load preprocessed traces
    
    Parameters
    ----------
    path : str
        Absolute path to processed traces
        
    Returns 
    -------
    traces : ndarray
        The preprocessed traces of shape (#traces,#channels,#bins)
    eventnumber : ndarray
        The series number concatenated with event number
        
    """
    trace_dict = dd.io.load(path)
    return trace_dict['traces'], trace_dict['eventnumber']

def _load_preprocessed_meta(path):
    """
    Function to load preprocessed metadata
    
    Parameters
    ----------
    path : str
        Absolute path to metadata
        
    Returns 
    -------
    metadata : dict
        Dictionary containing:
            'scale_factor' : the sum of all traces devided by tracelength
            'n_traces' : the number of traces in the dump
            'dump' : the dump file number
    """
    metadata = dd.io.load(path)
    return metadata

def _load_labels_dump(path):
    """
    Function to labels for PD2 dump
    
    Parameters
    ----------
    path : str
        Absolute path to labels
        
    Returns 
    -------
    labels : pandas.DataFrame
        All the RQ's from the DM analysis for 
        PD2
    """
    labels = pd.read_hdf(path, 'labels')
    return labels


def parse_MC_file(fname,key,flatten=True, mode = '1d'):
    df = pd.read_hdf(fname,key=key,mode='r')
    
    traces= [t.reshape([-1]) if flatten else t for t in df['tm'].values]
    tms   = np.stack([t for t in traces]) 
    epos  = np.stack([[p[1], p[2], p[3], p[4], p[0]] for p in df['epos'].values])
    S     = np.stack([ ([p[-1]] if mode == '1d' else [p[-1]]*11) for p in df['epos'].values])
    return tms,epos,S


