import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary

__all__ = ["Linear_VAE"]



class Linear_VAE(nn.Module):
    """
    Variational Autoencoder Class consisting of linear fully connected 
    layers.
    """
    def __init__(self, z_dim=15, input_dims=924, hid_dims=400):
        super(Linear_VAE, self).__init__()
        
        self.z_dim = z_dim
        self.input_dims = input_dims
        self.hid_dims = hid_dims

        self.fc1 = nn.Linear(self.input_dims, self.hid_dims)
        self.fc21 = nn.Linear(self.hid_dims, self.z_dim)
        self.fc22 = nn.Linear(self.hid_dims, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hid_dims)
        self.fc4 = nn.Linear(self.hid_dims, self.input_dims)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dims))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar