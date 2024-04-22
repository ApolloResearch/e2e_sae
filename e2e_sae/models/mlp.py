"""
Defines a generic MLP.
"""

import torch
from torch import nn

from e2e_sae.models.sparsifiers import SAE, Codebook


class Layer(nn.Module):
    """
    Neural network layer consisting of a linear layer followed by RELU.

    Args:
        in_features: The size of each input.
        out_features: The size of each output.
        has_activation_fn: Whether to use an activation function. Default is True.
        bias: Whether to add a bias term to the linear transformation. Default is True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_activation_fn: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if has_activation_fn:
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x


class MLP(nn.Module):
    """
    This class defines an MLP with a variable number of hidden layers.

    All layers use a linear transformation followed by RELU, except for the final layer which
    uses a linear transformation followed by no activation function.

    Args:
        hidden_sizes: A list of integers specifying the sizes of the hidden layers.
        input_size: The size of each input sample.
        output_size: The size of each output sample.
        bias: Whether to add a bias term to the linear transformations. Default is True.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        # Size of each layer (including input and output)
        sizes = [input_size] + hidden_sizes + [output_size]

        layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            # No activation for final layer
            has_activation_fn = i < len(sizes) - 2
            layers.append(
                Layer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    has_activation_fn=has_activation_fn,
                    bias=bias,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPMod(nn.Module):
    """
    This class defines an MLP with a variable number of hidden layers.

    All layers use a linear transformation followed by RELU, except for the final layer which
    uses a linear transformation followed by no activation function.

    Args:
        hidden_sizes: A list of integers specifying the sizes of the hidden layers.
        input_size: The size of each input sample.
        output_size: The size of each output sample.
        bias: Whether to add a bias term to the linear transformations. Default is True.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None,
        input_size: int,
        output_size: int,
        bias: bool = True,
        type_of_sparsifier: str = "sae",
        dict_eles_to_input_ratio: float = 2,
        k: int = 0,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        # Size of each layer (including input and output)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleDict()
        self.sparsifiers = nn.ModuleDict()
        self.dict_eles_to_input_ratio = dict_eles_to_input_ratio
        for i in range(len(sizes) - 1):
            has_activation_fn = i < len(sizes) - 2
            # Add layers with custom keys
            self.layers[f"{i}"] = Layer(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                has_activation_fn=has_activation_fn,
                bias=bias,
            )
            if has_activation_fn:
                if type_of_sparsifier == "sae":
                    sparsifier = SAE(
                        input_size=sizes[i + 1],
                        n_dict_components=int(sizes[i + 1] * self.dict_eles_to_input_ratio),
                    )
                elif type_of_sparsifier == "codebook":
                    assert k > 0, "k must be greater than 0"
                    sparsifier = Codebook(
                        input_size=sizes[i + 1],
                        n_dict_components=int(sizes[i + 1] * dict_eles_to_input_ratio),
                        k=k,
                    )
                else:
                    raise ValueError("type_of_sparsifier must be either 'sae' or 'codebook'")
                self.sparsifiers[f"{i}"] = sparsifier

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outs = {}
        cs = {}
        sparsifiers_outs = {}
        for i in range(len(self.layers)):
            x = self.layers[f"{i}"](x)
            outs[f"{i}"] = x
            if f"{i}" in self.sparsifiers:
                x, c = self.sparsifiers[f"{i}"](x)
                cs[f"{i}"] = c
                sparsifiers_outs[f"{i}"] = x
        return outs, cs, sparsifiers_outs
