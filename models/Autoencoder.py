import numpy as np
import torch
from torch import nn

from collections import OrderedDict


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(f'Shape: {x.shape}')
        return x


class Simple_Autoencoder(nn.Module):

    def __init__(self, in_dimension, latent_size, batch_size=1):
        super(Simple_Autoencoder, self).__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ('Linear', nn.Linear(in_features=in_dimension, out_features=latent_size)),
                ('ReLU1', nn.ReLU()),
                ('Linear2', nn.Linear(in_features=latent_size, out_features=in_dimension)),
                #('Final Activation', nn.ReLU())
            ])
        )

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        return self.layers(x.float())
