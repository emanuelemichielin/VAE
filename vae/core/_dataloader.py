import torch
from torch.utils import data
import numpy as np
import pandas as pd
import vae
from vae import PD2_LABEL_COLUMNS
from vae import io


__all__ = ["PD2dataset"]

class PD2dataset(data.Dataset):
    """
    Class to characterize the PD2 dataset for use with
    PyTorch. This class serves as a generator for the 
    vae.get_traces() function. 
    """
    def __init__(self, 
                 eventnumbers, 
                 labels=None, 
                 scaledata=None, 
                 max_normed=False,
                 baseline_sub=False,
                 offset=None,
                 map_path='/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v3/file_mapping.h5',
                 tracelength=925):
        """
        Initialization of data object. eventnumbers
        are stored and will be iterated over and passed
        to vae.get_traces() during the training process. 
        If labels are not needed (ie, training a VAE, then
        leave as none and they will default to the 
        eventnumbers). Labes can either be passed as an 
        array, or a string corresponding to the truth value
        to be loaded from the previously calibrated data. 
        the string must be one of the keys found in 
        vae._globas.PD2_LABEL_COLUMNS.
        
        Parameters
        ----------
        eventnumbers : array
            Array of event numbers to use for training/testing
        labels : array, str, NoneType, optional
            Groud truth values correpsonding to each item in 
            eventnumbers. String must correspond to either a
            label in the saved RQs, or 'full'. If 'full', all
            the RQs will be returned.
        scaledata : float, NoneType, optional
            If a value is given, then all the data
            will be divided by this value.
        max_normed : Bool, optional
            If True, each trace is scaled such that the baseline
            is cenetered at zero and the maximum is set to one. 
        baseline_sub : Bool, optional
            Baseline of every trace is subtracted.
        offset : float, NoneType, optional
            Offset to be added to each trace. This is 
            done after all other processing steps.
        map_path : str, optional
            Absolute path to eventnumber-file mapping.
        tracelength : int, optional
            The length of the traces being loaded. This is nessesary
            to determine the initial array size for the returned traces.
        """

        self.list_IDs = eventnumbers
        if labels is None:
            labels = eventnumbers
        elif isinstance(labels, str):
            if labels != 'full':
                if labels not in PD2_LABEL_COLUMNS:
                    raise ValueError(f'{labels} not in PD2_LABEL_COLUMNS')
        self.labels = labels
        self.scaledata = scaledata
        self.max_normed = max_normed
        self.baseline_sub = baseline_sub
        self.offset = offset
        self.map = pd.read_hdf(map_path,'map')
        self.tracelength = tracelength

    def __len__(self):
        """Length of the datapoints in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Generator for the dataset. Calls vae.get_traces()
        for a given index. Returns the trace, and the label, 
        if labels are not None. If None, the corresponding
        event number is returned
        """
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = io.get_traces(ID, ev_mapping=self.map, tracelength=self.tracelength)[...,:-1]
        
        if self.max_normed:
            X = X - np.mean(X[..., self.tracelength-200:], axis=-1)
            X = X / np.amin(X, axis=-1, keepdims=True)
        elif self.baseline_sub:
            X = X - np.mean(X[..., self.tracelength-200:], axis=-1)
        if self.scaledata is not None:
            X /= self.scaledata
        if self.offset is not None:
            X += self.offset
        X = X.astype(np.float32)
        X = X[:,0,:]
        if isinstance(self.labels, str):
            labels = io.get_labels(ID, ev_mapping=self.map)
            if self.labels == 'full':
                 y = labels.values.astype(float)
            else:
                y = labels[self.labels].values.astype(float)
        else:
            y = self.labels[index]
        return X, y



