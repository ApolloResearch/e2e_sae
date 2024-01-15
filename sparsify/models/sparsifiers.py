"""
Defines a generic MLP.
"""
from typing import List, Optional

import torch
from torch import nn


class SAE(nn.Module):
    """
    Sparse AutoEncoder
    """
    def __init__(self, input_size, n_dict_components):
        super(SAE, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_size, n_dict_components), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        self.n_dict_components = n_dict_components
        self.input_size = input_size

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        c = self.encoder(x)

        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        x_hat = self.decoder(c)
        return x_hat, c

    @property
    def device(self):
        return next(self.parameters()).device