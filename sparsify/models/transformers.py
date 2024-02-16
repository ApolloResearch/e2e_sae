from typing import TYPE_CHECKING, Any

from torch import Tensor, nn
from transformer_lens import HookedTransformer

from sparsify.models.sparsifiers import SAE

if TYPE_CHECKING:
    from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config


class SAETransformer(nn.Module):
    def __init__(self, tlens_model: HookedTransformer, config: "Config") -> None:
        super().__init__()
        self.tlens_model = tlens_model.eval()
        self.saes = nn.ModuleDict()
        # Expand the sae_position_names into an explicit list of all sae positions
        # (e.g. 'hook_resid_pre' -> [blocks.0.hook_resid_pre, blocks.1.hook_resid_pre, ...])
        self.sae_position_names_explicit = [
            tlens_hook_key
            for tlens_hook_key in tlens_model.hook_dict
            if any(
                sae_pos_name in tlens_hook_key for sae_pos_name in config.saes.sae_position_names
            )
        ]
        self.sae_positions_training_now = self.sae_position_names_explicit

        for sae_position_name in self.sae_position_names_explicit:
            input_size = (
                self.tlens_model.cfg.d_model
            )  # TODO: Make this accommodate not just residual positions
            n_dict_components = int(config.saes.dict_size_to_input_ratio * input_size)
            self.saes[str(sae_position_name).replace(".", "_")] = SAE(
                input_size=input_size,
                n_dict_components=n_dict_components,
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
