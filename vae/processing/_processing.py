###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains processing functions for pre-processing
# data. Note, this is mostly inteded to be run in parallel
# on a cluster
###############################################################

import numpy as np
import rqpy as rp
import qetpy as qp
import vae
import pickle as pkl
from vae.io._utils import _save_preprocessed, _load_preprocessed_traces


__all__ = ["pre_process_PD2", "partition_data"]



def pre_process_PD2(path, fs, ioffset, rload, rsh, qetbias, chan, det, ds_factor=6, trunc=0,
                    lgcsave=False, savepath = '', convtoamps=None):
    """
    Function preprocess raw traces for PD2 detector. Can be generalized for multichannel detectors. 
    
    The beginning of all traces are chopped off, set by the variable trunc. The traces are downsampled
    by a factor of ds_factor. And the traces are converted to units of power, rather than current, so 
    that they should be invairant of bias point. 
    
    Parameters
    ----------
    path : str
        Absolute path to .mid.gz dump to open
    fs : int
        Sample rate 
    ioffset : float
        The the squid offset (found either from muons or IV). Note, this number is very
        important for this preprocessing method. If unknown, this function is worthless
    rload : float
        The load resistance of the device
    rsh : float
        The value of the shunt resistor
    qetbias : float
        The applied bias current to the TES
    chan : str
        The channel name ie. 'PBS1'
    det : str
        The detector name ie. 'Z1'
    ds_factor : int, optional
        Factor to downsample the traces by
    trunc : int, optional
        The number of bins to remove from the beginning of the trace (only use if the 
        events were not triggered at the beginning of the trace)
    lgcsave : boolean, optional
        If True, the processed dumps are saved in savepath. defaults to False
    savepath : str, optional
        The path where the processed dumps should be saved. Ignored if not lgcsave
    convtoamps : float, Nonetype, optional
        The conversion from raw ADC bins to TES current. Should be in units of 
        [Amps/ADC bins]. Defaults to None, which means that it will just be loaded
        from the .midas file. 
        
    Returns
    -------
    traces : dict
        Dictionary containing:
            'traces' : the preprocessed traces of shape (#traces,#channels,#bins)
            'eventnumber' : the series number concatenated with event number
    metadata : dict
        Dictionary containing:
            'scale_factor' : the sum of all traces devided by tracelength
            'n_traces' : the number of traces in the dump
            'dump' : the dump file number
            
    """
    x, info_dict = rp.io.get_traces_midgz(path, chan, det, lgcskip_empty=True, lgcreturndict=True,
                                         convtoamps=convtoamps)
    x = x[..., trunc:]
    ser_ev = np.zeros(x.shape[0], dtype=int)
    series = path.split('/')[-2]
    for ii, ev in enumerate(info_dict['eventnumber']):
        ser_ev[ii] = f'{series}_{ev}'
    power_traces = qp.utils.powertrace_simple(trace=x[:,0,:], ioffset=ioffset, qetbias=qetbias, rload=rload, rsh=rsh)
    res = rp.downsample_truncate(traces=power_traces, trunc=power_traces.shape[-1], fs=fs, ds=ds_factor)
    power_ds = res['traces_ds']
    scale_factor = np.sum(np.sum(power_ds, axis = 1), axis=0)/power_ds.shape[-1]
    dump_number = path.split('/')[-1].split('.')[0]
    
    metadata = {'scale_factor' : scale_factor, 
                'n_traces' : x.shape[0],
                'dump' : dump_number}
    traces = {'traces' : power_ds[:,np.newaxis,:], 'eventnumber' : ser_ev}
    if lgcsave:
        _save_preprocessed(savepath, traces, metadata)
    return traces, metadata




def partition_data(eventnumbers, pct_val=.2, pct_test=.2, savename=None, lgcsave=False, seed=42):
    """
    Function to randomize and split event numbers into a training/validation/testing set.
    The percentange of training data = 1 - pct_val - pct_test
    
    Parameters
    ----------
    eventnumbers : array, list
        array of event numbers (in format: series_event) to be divided
    pct_val : float, optional
        Pectentage of data to be placed in validation set (must be float
        between 0 and 1)
    pct_test : float, optional
        Pectentage of data to be placed in testing set (must be float
        between 0 and 1)
    savename : str, NoneType, optional
        Absolute path to where partition should be saved
    lgcsave : bool, optional
        If True, the partition will be saved (provided savename
        is not None)
    seed : int, optional
        Seed for the random number generator used to shuffle 
        the data
        
    Returns
    -------
    partition : dict
        The partitioned data dictionary with keys:
            'train' : array of event numbers for training data
            'validation' : array of event numbers for validataion
            'test' : array of event numbers for test data        
    """
    
    if isinstance(eventnumbers, list):
        eventnumbers = np.asarray(eventnumbers)
    
    np.random.seed(42)
    rand_ints = np.random.choice(len(eventnumbers), len(eventnumbers), replace=False)
    rand_events = eventnumbers[rand_ints]
    
    nval = int(pct_val*len(eventnumbers))
    ntest = int(pct_test*len(eventnumbers))

    partition = {}
    partition['validation'] = rand_events[:nval] 
    partition['test'] = rand_events[nval:(nval+ntest)] 
    partition['train'] = rand_events[(nval+ntest):] 
    
    if lgcsave:
        if savename is not None:
            with open(savename+'.pkl','wb') as file:
                pkl.dump(partition, file)
        else:
            raise ValueError('Please provide a valid savename')
            
    return partition

def _store_rq_labels(traces_path, rq_path, savepath):
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
        
    Returns
    -------
    None
    """
    with open(rq_path, 'rb') as rq_file:
        rq = pkl.load(rq_file)
    #rq.sort_values('ser_ev', inplace=True)
    
    for p in sorted(traces_path): 
        traces, eventnumbers = _load_preprocessed_traces(p)
        label = f"{p.split('/')[-1][:20]}labels.h5"
        cuts = np.zeros(rq.shape[0], dtype=bool)
        for ev in eventnumbers:
            cuts = cuts | (rq.ser_ev == ev)
        df_labels = rq[cuts]
        if not np.all(df_labels.ser_ev == eventnumbers):
            raise ValueError('Shape dump and rq labels do not match')
        df_labels.to_hdf(savepath+label, 'labels')