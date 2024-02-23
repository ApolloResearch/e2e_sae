from typing import TYPE_CHECKING, Any

from torch import Tensor, nn
from transformer_lens import HookedTransformer

from sparsify.models.sparsifiers import SAE
from sparsify.utils import get_hook_shapes

if TYPE_CHECKING:
    from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config


class SAETransformer(nn.Module):
    """A transformer model with SAEs at various positions.

    Args:
        config: The config for the model.
        tlens_model: The transformer model.
        raw_sae_position_names: A list of all the positions in the tlens_mdoel where SAEs are to be
            placed. These positions may have periods in them, which are replaced with hyphens in
            the keys of the `saes` attribute.
    """

    def __init__(
        self, config: "Config", tlens_model: HookedTransformer, raw_sae_position_names: list[str]
    ):
        super().__init__()
        self.tlens_model = tlens_model.eval()
        self.raw_sae_position_names = raw_sae_position_names
        self.hook_shapes: dict[str, list[int]] = get_hook_shapes(
            self.tlens_model, self.raw_sae_position_names
        )
        # ModuleDict keys can't have periods in them, so we replace them with hyphens
        self.all_sae_position_names = [name.replace(".", "-") for name in raw_sae_position_names]

        self.saes = nn.ModuleDict()
        for i in range(len(self.all_sae_position_names)):
            input_size = self.hook_shapes[self.raw_sae_position_names[i]][-1]
            # TODO: allow choosing which activation dimension (or dimensions) to train the SAE on
            self.saes[self.all_sae_position_names[i]] = SAE(
                input_size=input_size,
                n_dict_components=int(config.saes.dict_size_to_input_ratio * input_size),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.tlens_model(x)

    def to(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "SAETransformer":
        """TODO: Fix this. Tlens implementation of to makes this annoying"""

        if len(args) == 1:
            self.tlens_model.to(device_or_dtype=args[0])
        elif len(args) == 2:
            self.tlens_model.to(device_or_dtype=args[0])
            self.tlens_model.to(device_or_dtype=args[1])
        elif len(kwargs) == 1:
            if "device" in kwargs or "dtype" in kwargs:
                arg = kwargs["device"] if "device" in kwargs else kwargs["dtype"]
                self.tlens_model.to(device_or_dtype=arg)
            else:
                raise ValueError("Invalid keyword argument.")
        elif len(kwargs) == 2:
            assert "device" in kwargs and "dtype" in kwargs, "Invalid keyword arguments."
            self.tlens_model.to(device_or_dtype=kwargs["device"])
            self.tlens_model.to(device_or_dtype=kwargs["dtype"])
        else:
            raise ValueError("Invalid arguments.")

        self.saes.to(*args, **kwargs)
        return self
