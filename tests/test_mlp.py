import pytest
from torch import nn

from e2e_sae.models.mlp import MLP, Layer


@pytest.mark.parametrize(
    "hidden_sizes, bias, expected_layer_sizes",
    [
        # 2 hidden layers, bias False
        ([4, 3], False, [(3, 4), (4, 3), (3, 4)]),
        # no hidden layers, bias True
        ([], True, [(3, 4)]),
        # 1 hidden layer, bias True
        ([4], True, [(3, 4), (4, 4)]),
    ],
)
def test_mlp_layers(
    hidden_sizes: list[int],
    bias: bool,
    expected_layer_sizes: list[tuple[int, int]],
) -> None:
    """Test the MLP constructor for fixed input and output sizes.

    Verifies the created layers' types, sizes and bias.

    Args:
        hidden_sizes: A list of hidden layer sizes. If None, no hidden layers are added.
        bias: Whether to add a bias to the Linear layers.
        expected_layer_sizes: A list of tuples where each tuple is a pair of in_features and
            out_features of a layer.
    """
    input_size = 3
    output_size = 4
    model = MLP(
        hidden_sizes,
        input_size,
        output_size,
        bias=bias,
    )

    assert isinstance(model, MLP)

    for i, layer in enumerate(model.layers):
        assert isinstance(layer, Layer)
        assert isinstance(layer.linear, nn.Linear)

        # Check the in/out feature sizes of Linear layers
        assert layer.linear.in_features == expected_layer_sizes[i][0]
        assert layer.linear.out_features == expected_layer_sizes[i][1]
        # Check bias is not None when bias is True, and None otherwise
        assert layer.linear.bias is not None if bias else layer.linear.bias is None

        if i < len(model.layers) - 1:
            # Activation layers at indices before the last layer
            assert isinstance(layer.activation, nn.GELU)
        else:
            # No activation function for the last layer
            assert not hasattr(layer, "activation")
