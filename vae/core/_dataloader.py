import torch
from torch.utils import data
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
    def __init__(self, eventnumbers, labels=None, scaledata=None):
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
            eventnumbers. 
        scaledata : float, NoneType, optional
            If a value is given, then all the data
            will be divided by this value.
        """

        self.list_IDs = eventnumbers
        if labels is None:
            labels = eventnumbers
        elif isinstance(labels, str):
            if labels not in PD2_LABEL_COLUMNS:
                raise ValueError(f'{labels} not in PD2_LABEL_COLUMNS')
        self.labels = labels
        self.scaledata = scaledata

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
        X = io.get_traces(ID)
        if self.scaledata is not None:
            X /= self.scaledata
        if isinstance(self.labels, str):
            labels = io.get_labels(ID)
            y = labels[self.labels].values
        else:
            y = self.labels[index]
        return X, y



