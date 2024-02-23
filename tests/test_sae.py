import torch

from sparsify.models.sparsifiers import SAE
from sparsify.utils import set_seed


def test_orthonormal_initialization():
    """After initialising an SAE, the dictionary components should be orthonormal."""
    set_seed(0)
    input_size = 2
    n_dict_components = 4
    sae = SAE(input_size, n_dict_components)
    assert sae.decoder.weight.shape == (input_size, n_dict_components)
    # If vectors are orthonormal, the gram matrix (X X^T) should be the identity matrix
    assert torch.allclose(
        sae.decoder.weight @ sae.decoder.weight.T, torch.eye(input_size), atol=1e-6
    )
