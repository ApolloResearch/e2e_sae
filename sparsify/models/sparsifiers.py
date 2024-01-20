"""
Defines different sparsifiers.
"""
from typing import List, Optional

import torch
from torch import nn


class SAE(nn.Module):
    """
    Sparse AutoEncoder
    """
    def __init__(self, input_size, n_dict_components, use_bias=False):
        super(SAE, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(input_size, n_dict_components), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        nn.init.orthogonal_(self.decoder.weight)
        if use_bias:
            self.b = nn.Parameter(torch.zeros(input_size))

        self.use_bias = use_bias
        self.n_dict_components = n_dict_components
        self.input_size = input_size

    def forward(self, x):
        if self.use_bias:
            x = x - self.b

        c = self.encoder(x)

        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        x_hat = self.decoder(c)

        if self.use_bias:
            x_hat = x_hat + self.b

        return x_hat, c

    @property
    def device(self):
        return next(self.parameters()).device    


class Codebook(nn.Module):
    """
    Codebook from Tamkin et al. (2023)

    It compute the cosine similarity between an input and a dictionary of features of 
    size size n_dict_components. Then it simply takes the top k most similar codebook features 
    and outputs their sum. The output thus has size input_size and consists of a simple sum of 
    the top k codebook features. There is no encoder, just the dictionary of codebook features.
    """
    def __init__(self, input_size, n_dict_components, k):
        super(Codebook, self).__init__()

        self.codebook = nn.Parameter(torch.randn(n_dict_components, input_size)) # (n_dict_components, input_size)
        nn.init.orthogonal_(self.codebook)
        self.n_dict_components = n_dict_components
        self.input_size = input_size 
        self.k = k

    def forward(self, x):
        # Compute cosine similarity between input and codebook features
        cos_sim = nn.functional.cosine_similarity(x.unsqueeze(1), self.codebook, dim=2) # (batch_size, n_dict_components)

        # Take the top k most similar codebook features
        _, topk = torch.topk(cos_sim, self.k, dim=1) # (batch_size, k)

        # Sum the top k codebook features
        x_hat = torch.sum(self.codebook[topk], dim=1) # (batch_size, input_size)

        return x_hat, None

    @property
    def device(self):
        return next(self.parameters()).device

class DecomposedMatrix(nn.Module):
    """
    Decomposed matrix, like in SVD where M=U*S*V^T, 
    where a sparsity penalty will placed on the entries of the 
    diagonal S matrix (in the training loop) and the U and V* features are constrained 
    to have unit norm (but they are not constrained to be orthogonal).

    In practice, it subtracts a bias from the input, then computes the matrix product,
    then adds the bias back to the output. Without this, some L_p penalties
    would be needlessly suboptimal because the directions wouldn't
    be centered around the mean of the data, meaning that the 
    'activations' (i.e. c) would be larger than necessary in some 
    cases and smaller than necessary in others (hence for some L_p norms
    it would be suboptimal).
    """
    def __init__(self, input_size, n_dict_components):
        super(DecomposedMatrix, self).__init__()

        self.U = nn.Parameter(torch.randn(input_size, n_dict_components))
        self.S = nn.Parameter(torch.randn(n_dict_components))
        self.V = nn.Parameter(torch.randn(input_size, n_dict_components))
        self.b = nn.Parameter(torch.zeros(input_size))
        self.n_dict_components = n_dict_components
        self.input_size = input_size

        # Initialize the weights properly
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self, x):
        # Subtract the bias
        x = x - self.b

        # Apply unit norm constraint to the U and V features
        self.U.data = nn.functional.normalize(self.U.data, dim=0)
        self.V.data = nn.functional.normalize(self.V.data, dim=0)

        # Compute the matrix product
        c = torch.mm(self.U, torch.diag(self.S))
        x_hat = torch.mm(c, self.V.t())

        # Add the bias back
        x_hat = x_hat + self.b

        return x_hat, c
