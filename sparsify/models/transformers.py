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
            # TODO: allow choosing which activation dimension (or dimensions) to train the SAE on
            self.saes[self.all_sae_position_names[i]] = SAE(
                input_size=input_size,
                n_dict_components=int(config.saes.dict_size_to_input_ratio * input_size),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.tlens_model(x)

    def forward_both(
        self,
        tokens: Int[Tensor, "batch pos"],
        run_entire_model: bool,
        final_layer: int | None,
        sae_hook: Callable,  # type: ignore
    ) -> tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        dict[str, Float[torch.Tensor, "batch pos dim"]],
        Float[Tensor, "batch pos vocab"] | None,
        dict[str, dict[str, Float[torch.Tensor, "batch pos dim"]]],
    ]:
        """Forward pass through both the original tlens_model and the SAE-augmented model.

        If `run_entire_model` is False, the forward pass will stop at `final_layer` in the original
        model and not compute the logits of the SAE-augmented model. Otherwise, the entire model
        will be run.

        Args:
            tokens: The input tokens.
            run_entire_model: Whether to run the entire model or stop at `final_layer`.
            final_layer: The layer to stop at if `run_entire_model` is False.
            sae_hook: The hook function to use for the SAEs.

        Returns:
            - The logits of the original model.
            - The activations of the original model.
            - The logits of the SAE-augmented model, if `run_entire_model` is True.
            - The activations of the SAE-augmented model.
        """
        # Run model without SAEs
        with torch.inference_mode():
            orig_logits, orig_acts = self.tlens_model.run_with_cache(
                tokens,
                names_filter=self.raw_sae_position_names,
                return_cache_object=False,
                stop_at_layer=None if run_entire_model else final_layer,
            )
            assert isinstance(orig_logits, torch.Tensor)

        # Get SAE feature activations
        sae_acts = {hook_name: {} for hook_name in orig_acts}
        new_logits: Float[Tensor, "batch pos vocab"] | None = None
        if not run_entire_model:
            # Just run the already-stored activations through the SAEs
            for hook_name in orig_acts:
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
                for hook_name in orig_acts
            ]
            new_logits = self.tlens_model.run_with_hooks(
                tokens,
                fwd_hooks=fwd_hooks,  # type: ignore
            )
        return orig_logits, orig_acts, new_logits, sae_acts

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
