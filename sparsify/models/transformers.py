from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformer_lens import HookedTransformer

from sparsify.hooks import CacheActs, SAEActs, cache_hook, sae_hook
from sparsify.models.sparsifiers import SAE
from sparsify.utils import get_hook_shapes

if TYPE_CHECKING:
    from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config


class SAETransformer(nn.Module):
    """A transformer model with SAEs at various positions.

    Args:
        config: The config for the model.
        tlens_model: The transformer model.
        raw_sae_positions: A list of all the positions in the tlens_mdoel where SAEs are to be
            placed. These positions may have periods in them, which are replaced with hyphens in
            the keys of the `saes` attribute.
    """

    def __init__(
        self, config: "Config", tlens_model: HookedTransformer, raw_sae_positions: list[str]
    ):
        super().__init__()
        self.tlens_model = tlens_model.eval()
        self.raw_sae_positions = raw_sae_positions
        self.hook_shapes: dict[str, list[int]] = get_hook_shapes(
            self.tlens_model, self.raw_sae_positions
        )
        # ModuleDict keys can't have periods in them, so we replace them with hyphens
        self.all_sae_positions = [name.replace(".", "-") for name in raw_sae_positions]

        self.saes = nn.ModuleDict()
        for i in range(len(self.all_sae_positions)):
            input_size = self.hook_shapes[self.raw_sae_positions[i]][-1]
            self.saes[self.all_sae_positions[i]] = SAE(
                input_size=input_size,
                n_dict_components=int(config.saes.dict_size_to_input_ratio * input_size),
            )

    def forward_raw(
        self,
        tokens: Int[Tensor, "batch pos"],
        run_entire_model: bool,
        final_layer: int | None,
        cache_positions: list[str] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch pos d_vocab"], dict[str, Float[torch.Tensor, "batch pos dim"]]
    ]:
        """Forward pass through the original transformer without the SAEs.

        Args:
            tokens: The input tokens.
            run_entire_model: Whether to run the entire model or stop at `final_layer`.
            final_layer: The layer to stop at if `run_entire_model` is False.
            cache_positions: Hooks to cache activations at in addition to the SAE positions.

        Returns:
            - The logits of the original model.
            - The activations of the original model.
        """
        all_hook_names = self.raw_sae_positions + (cache_positions or [])
        orig_logits, orig_acts = self.tlens_model.run_with_cache(
            tokens,
            names_filter=all_hook_names,
            return_cache_object=False,
            stop_at_layer=None if run_entire_model else final_layer,
        )
        assert isinstance(orig_logits, torch.Tensor)
        return orig_logits, orig_acts

    def forward(
        self,
        tokens: Int[Tensor, "batch pos"],
        sae_positions: list[str],
        cache_positions: list[str] | None = None,
        orig_acts: dict[str, Float[Tensor, "batch pos dim"]] | None = None,
    ) -> tuple[Float[torch.Tensor, "batch pos d_vocab"] | None, dict[str, SAEActs | CacheActs]]:
        """Forward pass through the SAE-augmented model.

        If `orig_acts` is not None, simply pass them through the SAEs. If None, run the entire
        SAE-augmented model by apply sae_hooks and (optionally) cache_hooks to the input tokens.

        The cache_hooks are used to store activations at positions other than the SAE positions.

        Args:
            tokens: The input tokens.
            sae_hook_names: The names of the hooks to run the SAEs on.
            cache_positions: Hooks to cache activations at in addition to the SAE positions.
            orig_acts: The activations of the original model. If not None, simply pass them through
                the SAEs. If None, run the entire SAE-augmented model.

        Returns:
            - The logits of the SAE-augmented model. If `orig_acts` is not None, this will be None
                as the logits are not computed.
            - The activations of the SAE-augmented model.
        """
        # sae_acts and cache_acts will be written into by sae_hook and cache_hook
        new_acts: dict[str, SAEActs | CacheActs] = {}

        new_logits: Float[Tensor, "batch pos vocab"] | None = None
        if orig_acts is not None:
            # Just run the already-stored activations through the SAEs
            for sae_pos in sae_positions:
                sae_hook(
                    x=orig_acts[sae_pos].detach().clone(),
                    hook=None,
                    sae=self.saes[sae_pos.replace(".", "-")],
                    hook_acts=new_acts,
                    hook_key=sae_pos,
                )
        else:
            # Run the tokens through the whole SAE-augmented model
            sae_hooks = [
                (
                    sae_pos,
                    partial(
                        sae_hook,
                        sae=cast(SAE, self.saes[sae_pos.replace(".", "-")]),
                        hook_acts=new_acts,
                        hook_key=sae_pos,
                    ),
                )
                for sae_pos in sae_positions
            ]
            cache_hooks = [
                (cache_pos, partial(cache_hook, hook_acts=new_acts, hook_key=cache_pos))
                for cache_pos in cache_positions or []
                if cache_pos not in sae_positions
            ]

            new_logits = self.tlens_model.run_with_hooks(
                tokens,
                fwd_hooks=sae_hooks + cache_hooks,  # type: ignore
            )
        return new_logits, new_acts

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
