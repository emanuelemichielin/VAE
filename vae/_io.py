import numpy as np
import pandas as pd
import qetpy as qp
import rqpy as rp
import deepdish as dd
import os
from glob import glob
import pickle as pkl
from vae import PD2_LABEL_COLUMNS


#__all__ = ["save_preprocessed", "load_preprocessed_traces", "load_preprocessed_meta", 
#           "parse_MC_file", "get_traces", "load_partition"]
__all__ = ["get_labels", "get_traces", "load_partition"]


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

def get_labels(eventnumber,  
               ev_mapping=None,
               map_path='/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/file_mapping.h5',
              ):
    """
    Function to load known labels based on given event numbers. 
    It assumes that the mapping from event number to 
    file path is already saved somewhere. 
        
    Parameters
    ----------
    eventnumber : int, list of ints
        Event number, or list of event numbers to load.
        Note: must be in the format seriesnumber_originaleventnumber
    ev_mapping : Pandas.DataFrame, optional
        The mapping from event number to filename. If None,
        the mapping will be loaded from map_path.
    map_path : str, optional
        Aboslute path to the dataframe that maps event number to 
        filepath location for the dump of traces
    tracelength : int, optional
       The length of the traces being loaded. This is nessesary
       to determine the initial array size for the returned traces
       
    Returns
    -------
    labels : array
        An array of traces corresponding to the user specified
        event numbers. Will be of shape (#traces, #channels, #bins)
    """
    
    if np.isscalar(eventnumber):
        eventnumber = [eventnumber]
    good_labels = pd.DataFrame(columns=PD2_LABEL_COLUMNS, index=np.arange(0,len(eventnumber)))
    if ev_mapping is None:
        ev_mapping = pd.read_hdf(map_path,'map')
    cut = np.zeros(len(ev_mapping.eventnumber), dtype = bool)
    for ev in eventnumber:
        cut = cut | (ev_mapping.eventnumber == ev)
    filepaths = np.unique(ev_mapping[cut].label_filepath)
    for file in filepaths:
        labels = _load_labels_dump(file)
        evnums = labels.ser_ev.values
        clabels = np.zeros(len(evnums), dtype = bool)
        for ev in eventnumber:
            clabels = clabels | (evnums == ev)
        for ii, label in enumerate(labels[clabels].itertuples()):
            ind = np.where(eventnumber==evnums[clabels][ii])[0][0]
            good_labels.loc[ind] = label[1:]
    return good_labels

def get_traces(eventnumber,  
               ev_mapping=None,
               map_path='/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/file_mapping.h5', 
               tracelength=3084,
              ):
    """
    Function to load traces based on given event numbers. 
    It assumes that the mapping from event number to 
    file path is already saved somewhere. 
        
    Parameters
    ----------
    eventnumber : int, list of ints
        Event number, or list of event numbers to load.
        Note: must be in the format seriesnumber_originaleventnumber
    ev_mapping : Pandas.DataFrame, optional
        The mapping from event number to filename. If None,
        the mapping will be loaded from map_path.
    map_path : str, optional
        Aboslute path to the dataframe that maps event number to 
        filepath location for the dump of traces
    tracelength : int, optional
       The length of the traces being loaded. This is nessesary
       to determine the initial array size for the returned traces
       
    Returns
    -------
    good_traces : array
        An array of traces corresponding to the user specified
        event numbers. Will be of shape (#traces, #channels, #bins)
    """
    
    if np.isscalar(eventnumber):
        eventnumber = [eventnumber]
    good_traces = np.zeros(shape=(len(eventnumber), 1, tracelength))
    if ev_mapping is None:
        ev_mapping = pd.read_hdf(map_path,'map')
    cut = np.zeros(len(ev_mapping.eventnumber), dtype = bool)
    for ev in eventnumber:
        cut = cut | (ev_mapping.eventnumber == ev)
    filepaths = np.unique(ev_mapping[cut].filepath)
    for file in filepaths:
        traces, evnums = _load_preprocessed_traces(file)
        ctraces = np.zeros(len(evnums), dtype = bool)
        for ev in eventnumber:
            ctraces = ctraces | (evnums == ev)
        for ii, trace in enumerate(traces[ctraces]):
            ind = np.where(eventnumber==evnums[ctraces][ii])
            good_traces[ind] = trace
    return good_traces



def load_partition(basepath, file):
    """
    Function to load event numbers for partitioned
    datasets. 
    
    Parameters
    ----------
    basepath : str
        Absolute path to the directory where all 
        the partitions are saved.
    file : str
        The name of the partition to load, i.e 'triggers'
    
    Returns
    -------
    partition : dict
        The partitioned data dictionary with keys:
            'train' : array of event numbers for training data
            'validation' : array of event numbers for validataion
            'test' : array of event numbers for test data  
    """
    
    file = file.split('.')[0] #remove file extentions
    try:
        with open(basepath+file+'.pkl', 'rb') as file:
            partition = pkl.load(file)
        return partition
    except FileNotFoundError:
        files = glob(basepath+'/*')
        available_files = []
        for f in files:
            available_files.append(f.split('/')[-1].split('.')[0])
        raise ValueError(f'File not found: please choose from: {available_files}')
        
    

def parse_MC_file(fname,key,flatten=True, mode = '1d'):
    df = pd.read_hdf(fname,key=key,mode='r')
    
    traces= [t.reshape([-1]) if flatten else t for t in df['tm'].values]
    tms   = np.stack([t for t in traces]) 
    epos  = np.stack([[p[1], p[2], p[3], p[4], p[0]] for p in df['epos'].values])
    S     = np.stack([ ([p[-1]] if mode == '1d' else [p[-1]]*11) for p in df['epos'].values])
    return tms,epos,S


