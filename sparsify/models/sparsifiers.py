"""
Defines a generic MLP.
"""

import torch
from torch import nn


class SAE(nn.Module):
    """
    Sparse AutoEncoder
    """

    def __init__(self, input_size: int, n_dict_components: int):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_size, n_dict_components), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        self.n_dict_components = n_dict_components
        self.input_size = input_size

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.encoder(x)

        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        x_hat = self.decoder(c)
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

    def __init__(self, input_size: int, n_dict_components: int, k: int):
        super().__init__()

        self.codebook = nn.Parameter(
            torch.randn(n_dict_components, input_size)
        )  # (n_dict_components, input_size)
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Compute cosine similarity between input and codebook features
        cos_sim = nn.functional.cosine_similarity(
            x.unsqueeze(1), self.codebook, dim=2
        )  # (batch_size, n_dict_components)

        # Take the top k most similar codebook features
        _, topk = torch.topk(cos_sim, self.k, dim=1)  # (batch_size, k)

        # Sum the top k codebook features
        x_hat = torch.sum(self.codebook[topk], dim=1)  # (batch_size, input_size)

        return x_hat, None

    @property
    def device(self):
        return next(self.parameters()).device
