import torch
from torch.utils import data
import vae

__all__ = ["PD2dataset"]

class PD2dataset(data.Dataset):
    """
    Class to characterize the PD2 dataset for use with
    PyTorch. This class serves as a generator for the 
    vae.get_traces() function. 
    """
    def __init__(self, eventnumbers, labels=None):
        """
        Initialization of data object. eventnumbers
        are stored and will be iterated over and passed
        to vae.get_traces() during the training process. 
        If using during VAE training, labels are not
        needed and will default to using eventnumbers
        for labels. This way the ground truth values
        can be looked up after the fact based on eventnumber.
        
        Parameters
        ----------
        eventnumbers : array
            Array of event numbers to use for training/testing
        labels : array, optional
            Groud truth values correpsonding to each item in 
            eventnumbers
        """

        self.list_IDs = eventnumbers
        if labels is None:
            self.labels = eventnumbers
        else:
            self.labels = labels

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
        X = vae.get_traces(ID)
        y = self.labels[index]
        return X, y



