from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from jaxtyping import Float, Int
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
            self.saes[self.all_sae_position_names[i]] = SAE(
                input_size=input_size,
                n_dict_components=int(config.saes.dict_size_to_input_ratio * input_size),
            )

    def forward_raw(
        self, tokens: Int[Tensor, "batch pos"], run_entire_model: bool, final_layer: int | None
    ) -> tuple[
        Float[torch.Tensor, "batch pos d_vocab"], dict[str, Float[torch.Tensor, "batch pos dim"]]
    ]:
        """Forward pass through the original transformer without the SAEs.

        Args:
            tokens: The input tokens.
            run_entire_model: Whether to run the entire model or stop at `final_layer`.
            final_layer: The layer to stop at if `run_entire_model` is False.

        Returns:
            - The logits of the original model.
            - The activations of the original model.
        """
        orig_logits, orig_acts = self.tlens_model.run_with_cache(
            tokens,
            names_filter=self.raw_sae_position_names,
            return_cache_object=False,
            stop_at_layer=None if run_entire_model else final_layer,
        )
        assert isinstance(orig_logits, torch.Tensor)
        return orig_logits, orig_acts

    def forward(
        self,
        tokens: Int[Tensor, "batch pos"],
        sae_hook: Callable,  # type: ignore
        hook_names: list[str],
        orig_acts: dict[str, Float[Tensor, "batch pos dim"]] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch pos d_vocab"] | None,
        dict[str, dict[str, Float[torch.Tensor, "batch pos dim"]]],
    ]:
        """Forward pass through the SAE-augmented model.

        Args:
            tokens: The input tokens.
            orig_acts: The activations of the original model. If not None, simply pass them through
                the SAEs. If None, run the entire SAE-augmented model.
            sae_hook: The hook function to use for the SAEs.
            hook_names: The names of the hooks in the original model.

        Returns:
            - The logits of the SAE-augmented model. If `orig_acts` is not None, this will be None
                as the logits are not computed.
            - The activations of the SAE-augmented model.
        """
        # sae_acts will be written into by the sae_hook
        sae_acts = {hook_name: {} for hook_name in hook_names}
        new_logits: Float[Tensor, "batch pos vocab"] | None = None
        if orig_acts is not None:
            # Just run the already-stored activations through the SAEs
            for hook_name in hook_names:
                sae_hook(
                    value=orig_acts[hook_name].detach().clone(),
                    hook=None,
                    sae=self.saes[hook_name.replace(".", "-")],
                    hook_acts=sae_acts[hook_name],
                )
        else:
            # Run the tokens through the whole SAE-augmented model
            fwd_hooks: list[tuple[str, Callable[..., Float[torch.Tensor, "... d_head"]]]] = [
                (
                    hook_name,
                    partial(
                        sae_hook,
                        sae=cast(SAE, self.saes[hook_name.replace(".", "-")]),
                        hook_acts=sae_acts[hook_name],
                    ),
                )
                for hook_name in hook_names
            ]
            new_logits = self.tlens_model.run_with_hooks(
                tokens,
                fwd_hooks=fwd_hooks,  # type: ignore
            )
        return new_logits, sae_acts

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
