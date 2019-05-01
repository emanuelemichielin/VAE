import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
#from torch.nn import init
# import argparse
# import os
# from sklearn.model_selection import train_test_split
# import glob
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.manifold import TSNE
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.utils import shuffle
from torchsummary import summary


__all__ = ["CNN_VAE"]

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=4)
        self.conv2 = nn.Conv1d(16, 16, 8, 2, padding=4)
        self.conv3 = nn.Conv1d(16, 32, 8, 2, padding=3)
        self.conv4 = nn.Conv1d(32, 32, 8, 2, padding=3)
        #self.fc1 = nn.Linear(32*21, 64)
        self.fc1 = nn.Linear(32*58, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc21 = nn.Linear(16, z_dim)
        self.fc22 = nn.Linear(16, z_dim)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.dropout(x, 0.3)
        #print(x.size())
        #x = x.view(-1, 672)
        x = x.view(-1, 32*58)
        
        x = self.relu(self.fc1(x))
        x = self.bn5(x)
        x = F.dropout(x, 0.5)
        x = self.relu(self.fc2(x))
        z_loc = self.fc21(x)
        z_scale = self.fc22(x)
        return z_loc, z_scale



class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        #self.fc1 = nn.Linear(z_dim, 672)
        self.fc1 = nn.Linear(z_dim, 32*58)
        self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv2 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv3 = nn.ConvTranspose1d(32, 16, 8, 2, padding=4)
        self.conv4 = nn.ConvTranspose1d(16, 16, 8, 2, padding=3)
        self.conv5 = nn.ConvTranspose1d(16, 1, 7, 1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        #z = F.dropout(z, 0.3)
        #z = z.view(-1, 32, 21)
        z = z.view(-1, 32, 58)
        z = self.relu(self.conv1(z))
        z = self.bn1(z)
        #z = F.dropout(z, 0.3)
        z = self.relu(self.conv2(z))
        z = self.bn2(z)
        #z = F.dropout(z, 0.3)
        z = self.relu(self.conv3(z))
        z = self.bn3(z)
        #z = F.dropout(z, 0.3)
        z = self.relu(self.conv4(z))
        z = self.bn4(z)
        #z = F.dropout(z, 0.3)
        z = self.conv5(z)
        recon = torch.sigmoid(z)
        return recon


class CNN_VAE(nn.Module):
    def __init__(self, z_dim=2):
        super(CNN_VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.cuda()
        self.z_dim = z_dim

    def reparameterize(self, z_loc, z_scale):
        std = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(device)
        z = z_loc + std * epsilon
        return z