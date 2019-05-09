###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains a CNN based VAE Model and is able to be
# imported from the base level of the module. 
###############################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from vae.utils import get_conv_shape




__all__ = ["VAE"]

class Encoder(nn.Module):
    """
    Encoder class for variation autoencoder 
    """
    def __init__(self, 
                 z_dim, 
                 usedropout=False, 
                 inputchans = 1,
                 tracelength=1624,
                 kernels=[8,8,8,8], 
                 strides=[2,2,2,2],
                 pads=[1,0,0,0,],
                 chans=[16,16,32,32,64,16]):
        """
        Encoder initializer. Disclaimer: these parameters currently work with
        well with the shape of the data to get the correct shaped output 
        from the decoder. The user might need to tweek the encoder and 
        decoder params a bit if these are changed in order to gauruntee
        the output shape is as expected. 
        
        Paramters
        ---------
        z_dim : int,
            The number of hidden dimensions to use.
        usedropout : Bool, optional
            If True, dropout is used between each layer
        inputchans : int, optional
            Number of input channels. Defaults to one. 
        tracelength : int, optional
            The shape of the last dimension of the input
            data
        kernesl : list of ints, optional
            The shapes of the kernels for the conv1d layers
        strides : list of ints, optional
            The stride for each conv1d layer
        pads : list of ints, optional
            The padding for each conv1d layer
        chans : list of ints, optional
            The number of each channels for each later. 
            The first 4 correspond to the conv1d layers,
            and the last two elements are for the fully
            connected layers.
        
        """
        
        super(Encoder, self).__init__()
        
        self.usedropout = usedropout
        
        # convolution layers
        self.conv1 = nn.Conv1d(inputchans, chans[0], kernels[0], 
                               strides[0], padding=pads[0])
        self.conv2 = nn.Conv1d(chans[0], chans[1], kernels[1], 
                               strides[1], padding=pads[1])
        self.conv3 = nn.Conv1d(chans[1], chans[2], kernels[2], 
                               strides[2], padding=pads[2])
        self.conv4 = nn.Conv1d(chans[2], chans[3], kernels[3], 
                               strides[3], padding=pads[3])
        
        #calculate expecte output shape from conv layers
        convoutshape = tracelength
        for ii in range(4):
            convoutshape = get_conv_shape(convoutshape, kernels[ii], 
                                      strides[ii], pads[ii], 1)
        # store the shape for use in the forward() method    
        self.convoutshape = convoutshape
        
        # Fully connected layers
        self.fc1 = nn.Linear(chans[3]*convoutshape, chans[4])
        self.fc2 = nn.Linear(chans[4], chans[5])
        self.fc21 = nn.Linear(chans[5], z_dim)
        self.fc22 = nn.Linear(chans[5], z_dim)
        
        # Define batch normalization based on channel shapes
        self.bn1 = nn.BatchNorm1d(chans[0])
        self.bn2 = nn.BatchNorm1d(chans[1])
        self.bn3 = nn.BatchNorm1d(chans[2])
        self.bn4 = nn.BatchNorm1d(chans[3])
        self.bn5 = nn.BatchNorm1d(chans[4])

        
        


    def forward(self, x):
        """
        Function to push the data forward through the network
        """
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        if self.usedropout: x = F.dropout(x, 0.3);
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        if self.usedropout: x = F.dropout(x, 0.3);
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        if self.usedropout: x = F.dropout(x, 0.3);
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        if self.usedropout: x = F.dropout(x, 0.3);
        x = x.view(-1, 32*96)
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        if self.usedropout: x = F.dropout(x, 0.3);
        x = F.relu(self.fc2(x))
        
        # connect last layer to mu, sig layers
        z_loc = self.fc21(x)
        z_scale = self.fc22(x)
        return z_loc, z_scale



class Decoder(nn.Module):
    """
    Decoder class for variational autoencoder
    """
    def __init__(self, 
                 z_dim, 
                 usedropout=False, 
                 outchans = 1,
                 tracelength=1624,
                 kernels=[8,8,8,8], 
                 strides=[2,2,2,2],
                 pads=[0,0,0,0,],
                 chans=[32,32,32,16,16],
                 convoutput = 96,
                 final_kernel=7,
                 final_pad=4,
                ):
        """
        Decoder initializer. Disclaimer: these parameters currently work with
        well with the shape of the data to get the correct shaped output 
        from the decoder. The user might need to tweek the encoder and 
        decoder params a bit if these are changed in order to gauruntee
        the output shape is as expected.
        
        TODO : I still need to come up with an automated way to figure
        out the shapes of the final inverse conv layer so that the user
        doesn't need to figure out the shape by trial and error. 
        
        Paramters
        ---------
        z_dim : int,
            The number of hidden dimensions to use.
        usedropout : Bool, optional
            If True, dropout is used between each layer
        outchans : int, optional
            Number of output channels expected. Defaults to one. 
        tracelength : int, optional
            The shape of the last dimension of the input
            data
        kernesl : list of ints, optional
            The shapes of the kernels for the conv1d layers
        strides : list of ints, optional
            The stride for each conv1d layer
        pads : list of ints, optional
            The padding for each conv1d layer
        chans : list of ints, optional
            The number of each channels for each later. 
            The first 4 correspond to the conv1d layers,
            and the last two elements are for the fully
            connected layers.
        convoutput : int, optional
            The output shape from the convolutional layers
            in the encoder
        final_kernel : int, optional
            The kernel size for the final inverse conv layer
        final_pad : int, optional
            the padding for the final inverse conv layer
        """
        
        super(Decoder, self).__init__()
        
        self.convoutput = convoutput
        self.reshaper = (chans[0], convoutput)
        self.usedropout = usedropout
        
        
        self.fc1 = nn.Linear(z_dim, chans[0]*convoutput)
        self.conv1 = nn.ConvTranspose1d(chans[0], chans[1], kernels[0], 
                                        strides[0], padding=pads[0])
        self.conv2 = nn.ConvTranspose1d(chans[1], chans[2], kernels[1], 
                                        strides[1], padding=pads[1])
        self.conv3 = nn.ConvTranspose1d(chans[2], chans[3], kernels[2], 
                                        strides[2], padding=pads[2])
        self.conv4 = nn.ConvTranspose1d(chans[3], chans[4], kernels[3], 
                                        strides[3], padding=pads[3])
        
        self.conv5 = nn.ConvTranspose1d(chans[4], outchans, 7, 1, padding=4)
        
        self.bn1 = nn.BatchNorm1d(chans[1])
        self.bn2 = nn.BatchNorm1d(chans[2])
        self.bn3 = nn.BatchNorm1d(chans[3])
        self.bn4 = nn.BatchNorm1d(chans[4])
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        Function to push the data forward through the network
        """
        z = F.relu(self.fc1(z))
        if self.usedropout: z = F.dropout(z, 0.3);
        z = z.view(-1, *self.reshaper)
        z = F.relu(self.conv1(z))
        z = self.bn1(z)
        if self.usedropout: z = F.dropout(z, 0.3);
        z = F.relu(self.conv2(z))
        z = self.bn2(z)
        if self.usedropout: z = F.dropout(z, 0.3);
        z = F.relu(self.conv3(z))
        z = self.bn3(z)
        if self.usedropout: z = F.dropout(z, 0.3);
        z = F.relu(self.conv4(z))
        z = self.bn4(z)
        if self.usedropout: z = F.dropout(z, 0.3);
        z = self.conv5(z)
        recon = torch.sigmoid(z)
        return recon


class VAE(nn.Module):
    """
    Variational autoencoder class composed of 1d convolution
    layers 4 and 2 fully connected layera. Note, at the moment, 
    this expects an input dimention of 1624. 
    """
    
    def __init__(self, 
                 z_dim=2, 
                 usedropout_encode=False, 
                 usedropout_decode=False,
                 inputchans = 1,
                 tracelength=1624,
                 kernels=[8,8,8,8], 
                 strides=[2,2,2,2],
                 pads=[1,0,0,0,],
                 chans=[16,16,32,32,64,16], 
                 final_kernel=7,
                 final_pad=4,
                 ):
        """
        VAEinitializer. Disclaimer: these parameters currently work with
        well with the shape of the data to get the correct shaped output 
        from the decoder. The user might need to tweek the encoder and 
        decoder params a bit if these are changed in order to gauruntee
        the output shape is as expected.
        
        TODO : I still need to come up with an automated way to figure
        out the shapes of the final inverse conv layer so that the user
        doesn't need to figure out the shape by trial and error. 
        
        Paramters
        ---------
        z_dim : int,
            The number of hidden dimensions to use.
        usedropout_encode : Bool, optional
            If True, dropout is used between each layer
            in Encoder
        usedropout_decode : Bool, optional
            If True, dropout is used between each layer
            in Decoder
        inputchans : int, optional
            Number of input channels. Defaults to one. 
        tracelength : int, optional
            The shape of the last dimension of the input
            data
        kernesl : list of ints, optional
            The shapes of the kernels for the conv1d layers
        strides : list of ints, optional
            The stride for each conv1d layer
        pads : list of ints, optional
            The padding for each conv1d layer
        chans : list of ints, optional
            The number of each channels for each later. 
            The first 4 correspond to the conv1d layers,
            and the last two elements are for the fully
            connected layers.
        convoutput : int, optional
            The output shape from the convolutional layers
            in the encoder
        final_kernel : int, optional
            The kernel size for the final inverse conv layer
        final_pad : int, optional
            the padding for the final inverse conv layer
        """
        
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim=z_dim, 
                               usedropout=usedropout_encode,
                               inputchans=inputchans,
                               kernels=kernels,
                               strides=strides,
                               pads=pads,
                               chans=chans)
        # Need to improve this part of the code. Note sure
        # at the moment of the best way to make sure the 
        # input stage of the decoder is the correct shape
        # the below works, but is not robust
        initial_dim = [chans[3]]
        initial_dim.extend(chans[:4][::-1])
    
        self.decoder = Decoder(z_dim=z_dim, 
                               usedropout=usedropout_decode, 
                               outchans=inputchans,
                               tracelength=tracelength,
                               kernels=kernels, 
                               strides=strides,
                               pads=[0,0,0,0],
                               chans=initial_dim,
                               convoutput = self.encoder.convoutshape,
                               final_kernel=final_kernel,
                               final_pad=final_pad)
        
        self.device = torch.device("cuda:0")
        self.cuda()
        self.z_dim = z_dim

    def reparameterize(self, z_loc, z_scale):
        """
        Reparameterization trick. Indroduces the random sampling
        outside of the network so that backpropegation still works.
        """
        std = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(self.device)
        z = z_loc + std * epsilon
        return z
    
    def forward(self, x):
        """
        Function to push the data forward through the network
        """
        z_loc, z_scale = self.encoder(x)
        z = self.reparameterize(z_loc, z_scale)
        recon = self.decoder(z)
        return recon, z_loc, z_scale
    
    