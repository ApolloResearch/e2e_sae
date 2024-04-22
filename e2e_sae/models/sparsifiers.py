"""
Defines a generic MLP.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SAE(nn.Module):
    """
    Sparse AutoEncoder
    """

    def __init__(
        self, input_size: int, n_dict_components: int, init_decoder_orthogonal: bool = True
    ):
        """Initialize the SAE.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
        """

        super().__init__()
        # self.encoder[0].weight has shape: (n_dict_components, input_size)
        # self.decoder.weight has shape:    (input_size, n_dict_components)

        self.encoder = nn.Sequential(nn.Linear(input_size, n_dict_components, bias=True), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, input_size, bias=True)
        self.n_dict_components = n_dict_components
        self.input_size = input_size

        if init_decoder_orthogonal:
            # Initialize so that there are n_dict_components orthonormal vectors
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass input through the encoder and normalized decoder."""
        c = self.encoder(x)
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        return x_hat, c

    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        return F.normalize(self.decoder.weight, dim=0)

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

    def __init__(self, input_size: int, n_dict_components: int, k: int):
        super().__init__()

        self.codebook = nn.Parameter(
            torch.randn(n_dict_components, input_size)
        )  # (n_dict_components, input_size)
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Compute cosine similarity between input and codebook features (batch_size, dict_size)
        cos_sim = F.cosine_similarity(x.unsqueeze(1), self.codebook, dim=2)

        # Take the top k most similar codebook features
        _, topk = torch.topk(cos_sim, self.k, dim=1)  # (batch_size, k)

        # Sum the top k codebook features
        x_hat = torch.sum(self.codebook[topk], dim=1)  # (batch_size, input_size)

        return x_hat, None

    @property
    def device(self):
        return next(self.parameters()).device
