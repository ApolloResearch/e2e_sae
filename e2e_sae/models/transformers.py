import os
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
import tqdm
import wandb
import yaml
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformer_lens import HookedTransformer
from transformer_lens.utils import LocallyOverridenDefaults, sample_logits
from wandb.apis.public import Run

from e2e_sae.hooks import CacheActs, SAEActs, cache_hook, sae_hook
from e2e_sae.loader import load_tlens_model
from e2e_sae.models.sparsifiers import SAE
from e2e_sae.utils import filter_names, get_hook_shapes


class SAETransformer(nn.Module):
    """A transformer model with SAEs at various positions.

    Args:
        tlens_model: The transformer model.
        raw_sae_positions: A list of all the positions in the tlens_mdoel where SAEs are to be
            placed. These positions may have periods in them, which are replaced with hyphens in
            the keys of the `saes` attribute.
        dict_size_to_input_ratio: The ratio of the dictionary size to the input size for the SAEs.
        init_decoder_orthogonal: Whether to initialize the decoder weights of the SAEs to be
            orthonormal. Not needed when e.g. loading pretrained SAEs. Defaults to True.
    """

    def __init__(
        self,
        tlens_model: HookedTransformer,
        raw_sae_positions: list[str],
        dict_size_to_input_ratio: float,
        init_decoder_orthogonal: bool = True,
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
                n_dict_components=int(dict_size_to_input_ratio * input_size),
                init_decoder_orthogonal=init_decoder_orthogonal,
            )

    def forward_raw(
        self,
        tokens: Int[Tensor, "batch pos"],
        run_entire_model: bool,
        final_layer: int | None = None,
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
        assert (
            not run_entire_model or final_layer is None
        ), "Can't specify both run_entire_model and final_layer"
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

    @torch.inference_mode()
    def generate(
        self,
        input: str | Float[torch.Tensor, "batch pos"] = "",
        sae_positions: list[str] | None | Literal["all"] = "all",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        prepend_bos: bool | None = None,
        padding_side: Literal["left", "right"] | None = None,
        return_type: str | None = "input",
        verbose: bool = True,
    ) -> Int[torch.Tensor, "batch pos_plus_new_tokens"] | str:
        """Sample Tokens from the model.

        Adapted from transformer_lens.HookedTransformer.generate()

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
                pos]) or a text string (this will be converted to a batch of tokens with batch size
                1).
            sae_hook_names: (list[str]) The names of the hooks to run the SAEs on.
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            generated sequence of new tokens, or completed prompt string (by default returns same
                type as input).
        """

        with LocallyOverridenDefaults(
            self.tlens_model, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if isinstance(input, str):
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tlens_model.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                tokens = self.tlens_model.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            else:
                tokens = input

            if return_type == "input":
                return_type = "str" if isinstance(input, str) else "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size = tokens.shape[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokens = tokens.to(device)

            stop_tokens = []
            eos_token_for_padding = 0
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tlens_model.tokenizer is not None
                    and self.tlens_model.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert tokenizer_has_eos_token, (
                        "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or "
                        "has no eos_token_id"
                    )
                    assert self.tlens_model.tokenizer is not None
                    eos_token_id = self.tlens_model.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    assert eos_token_id is not None
                    stop_tokens = eos_token_id
                    eos_token_for_padding = eos_token_id[0]

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(
                batch_size, dtype=torch.bool, device=self.tlens_model.cfg.device
            )

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            for _ in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                if sae_positions is None:
                    logits, _ = self.forward_raw(tokens, run_entire_model=True, final_layer=None)
                else:
                    if sae_positions == "all":
                        sae_positions = self.raw_sae_positions
                    logits, _ = self.forward(tokens, sae_positions=sae_positions)
                assert logits is not None
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(device)
                else:
                    sampled_tokens = final_logits.argmax(-1).to(device)

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    if isinstance(eos_token_for_padding, int):
                        sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(device))
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                assert self.tlens_model.tokenizer is not None
                if self.tlens_model.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.tlens_model.tokenizer.decode(tokens[0, 1:])
                else:
                    return self.tlens_model.tokenizer.decode(tokens[0])

            else:
                return tokens

    @classmethod
    def from_wandb(cls, wandb_project_run_id: str) -> "SAETransformer":
        """Instantiate an SAETransformer using the latest checkpoint from a wandb run.

        Args:
            wandb_project_run_id: The wandb project name and run ID separated by a forward slash.
                E.g. "gpt2/2lzle2f0"

        Returns:
            An instance of the SAETransformer class loaded from the specified wandb run.
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        cache_dir = Path(os.environ.get("SAE_CACHE_DIR", "/tmp/"))
        model_cache_dir = cache_dir / wandb_project_run_id

        train_config_file_remote = [
            file for file in run.files() if file.name.endswith("final_config.yaml")
        ][0]

        train_config_file = train_config_file_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name

        checkpoints = [file for file in run.files() if file.name.endswith(".pt")]
        latest_checkpoint_remote = sorted(
            checkpoints, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1])
        )[-1]
        latest_checkpoint_file = latest_checkpoint_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name
        assert latest_checkpoint_file is not None, "Failed to download the latest checkpoint."

        return cls.from_local_path(
            checkpoint_file=latest_checkpoint_file, config_file=train_config_file
        )

    @classmethod
    def from_local_path(
        cls,
        checkpoint_dir: str | Path | None = None,
        checkpoint_file: str | Path | None = None,
        config_file: str | Path | None = None,
    ) -> "SAETransformer":
        """Instantiate an SAETransformer using a checkpoint from a specified directory.

        NOTE: the current implementation restricts us from using the
        e2e_sae/scripts/train_tlens_saes/run_train_tlens_saes.py.Config class for type
        validation due to circular imports. Would need to move the Config class to a separate file
        to use it here.

        Args:
            checkpoint_dir: The directory containing one or more checkpoint files and
                `final_config.yaml`. If multiple checkpoints are present, load the one with the
                highest n_samples number (i.e. the latest checkpoint).
            checkpoint_file: The specific checkpoint file to load. If specified, `checkpoint_dir`
                is ignored and config_file must also be specified.
            config_file: The config file to load. If specified, `checkpoint_dir` is ignored and
                checkpoint_file must also be specified.

        Returns:
            An instance of the SAETransformer class loaded from the specified checkpoint.
        """
        if checkpoint_file is not None:
            checkpoint_file = Path(checkpoint_file)
            assert config_file is not None
            config_file = Path(config_file)
        else:
            assert checkpoint_dir is not None
            checkpoint_dir = Path(checkpoint_dir)
            assert config_file is None
            config_file = checkpoint_dir / "final_config.yaml"

            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            checkpoint_file = sorted(
                checkpoint_files, key=lambda x: int(x.name.split(".pt")[0].split("_")[-1])
            )[-1]

        with open(config_file) as f:
            config = yaml.safe_load(f)

        tlens_model = load_tlens_model(
            tlens_model_name=config["tlens_model_name"],
            tlens_model_path=config["tlens_model_path"],
        )

        raw_sae_positions = filter_names(
            list(tlens_model.hook_dict.keys()), config["saes"]["sae_positions"]
        )

        model = cls(
            tlens_model=tlens_model,
            raw_sae_positions=raw_sae_positions,
            dict_size_to_input_ratio=config["saes"]["dict_size_to_input_ratio"],
            init_decoder_orthogonal=False,
        )

        model.saes.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
        return model
