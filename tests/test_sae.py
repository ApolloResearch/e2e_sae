from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer

from e2e_sae.loader import load_pretrained_saes
from e2e_sae.models.sparsifiers import SAE
from e2e_sae.models.transformers import SAETransformer
from e2e_sae.utils import save_module, set_seed
from tests.utils import get_tinystories_config


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


@pytest.mark.parametrize("retrain_saes", [True, False])
def test_load_single_pretrained_sae(tmp_path: Path, retrain_saes: bool):
    """Test that loading a single pretrained SAE into a tlens model works.

    First create an SAETransformer with a single SAE position. We will save this model to a file
    and then load it back in to a new SAETransformer.

    Checks that:
        - The trainable parameters don't include the pretrained SAE if retrain_saes is False and
            include it if retrain_saes is True
        - The new SAE params are the same as the pretrained SAE for the position that was copied
    """

    # Make a dummy tlens config with two layers
    tlens_config = {
        "n_layers": 2,
        "d_model": 2,
        "n_ctx": 3,
        "d_head": 2,
        "act_fn": "gelu",
        "d_vocab": 2,
    }
    tlens_model = HookedTransformer(tlens_config)

    sae_position = "blocks.0.hook_resid_post"
    pretrained_config = get_tinystories_config({"saes": {"sae_positions": sae_position}})

    model = SAETransformer(
        tlens_model=tlens_model,
        raw_sae_positions=[sae_position],
        dict_size_to_input_ratio=pretrained_config.saes.dict_size_to_input_ratio,
        init_decoder_orthogonal=False,
    )
    # Save the model.saes to a temp file
    save_module(
        config_dict=pretrained_config.model_dump(mode="json"),
        save_dir=tmp_path,
        module=model.saes,
        model_filename="sae.pth",
    )

    # Get a new config that has more than one sae_position (including the one we saved)
    sae_positions = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    new_config = get_tinystories_config(
        {
            "saes": {
                "sae_positions": sae_positions,
                "pretrained_sae_paths": tmp_path / "sae.pth",
            }
        }
    )
    new_tlens_model = HookedTransformer(tlens_config)
    new_model = SAETransformer(
        tlens_model=new_tlens_model,
        raw_sae_positions=sae_positions,
        dict_size_to_input_ratio=new_config.saes.dict_size_to_input_ratio,
        init_decoder_orthogonal=False,
    )

    assert isinstance(new_config.saes.pretrained_sae_paths, list)

    # Now load in the pretrained SAE to the new model
    trainable_param_names = load_pretrained_saes(
        saes=new_model.saes,
        pretrained_sae_paths=new_config.saes.pretrained_sae_paths,
        all_param_names=[name for name, _ in new_model.saes.named_parameters()],
        retrain_saes=retrain_saes,
    )
    suffixes = ["encoder.0.weight", "encoder.0.bias", "decoder.weight", "decoder.bias"]
    block_0_params = [f"blocks-0-hook_resid_post.{suffix}" for suffix in suffixes]
    block_1_params = [f"blocks-1-hook_resid_post.{suffix}" for suffix in suffixes]
    if retrain_saes:
        assert trainable_param_names == block_0_params + block_1_params
    else:
        assert trainable_param_names == block_1_params

    model_named_params = dict(model.saes.named_parameters())
    new_model_named_params = dict(new_model.saes.named_parameters())
    for suffix in suffixes:
        # Check that the params for block 0 are the same as the pretrained SAE
        assert torch.allclose(
            model_named_params[f"blocks-0-hook_resid_post.{suffix}"],
            new_model_named_params[f"blocks-0-hook_resid_post.{suffix}"],
        )


def test_load_multiple_pretrained_sae(tmp_path: Path):
    """Test that loading multiple pretrained SAE into a tlens model works.

    - Creates an SAETransformer with SAEs in blocks 0 and 1 and save to file.
    - Creates another SAETransformer with SAEs in blocks 1 and 2 and save to file.
    - Creates a new SAETransformer with SAEs in blocks 0, 1 and 2 and load in the saved SAEs.

    Checks that we have:
        - Block 0 should have the params from the first saved SAE
        - Blocks 1 and 2 should have the params from the second saved SAE
    """

    # Make a dummy tlens config with two layers
    tlens_config = {
        "n_layers": 4,
        "d_model": 2,
        "n_ctx": 3,
        "d_head": 2,
        "act_fn": "gelu",
        "d_vocab": 2,
    }
    all_positions = [
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
        "blocks.3.hook_resid_post",
    ]

    filenames = ["sae_0.pth", "sae_1.pth"]
    sae_position_lists = [
        ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
        ["blocks.1.hook_resid_post", "blocks.2.hook_resid_post"],
    ]
    sae_params = []
    for filename, sae_positions in zip(filenames, sae_position_lists, strict=True):
        pretrained_config = get_tinystories_config({"saes": {"sae_positions": sae_positions}})
        tlens_model = HookedTransformer(tlens_config)
        model = SAETransformer(
            tlens_model=tlens_model,
            raw_sae_positions=sae_positions,
            dict_size_to_input_ratio=pretrained_config.saes.dict_size_to_input_ratio,
            init_decoder_orthogonal=False,
        )
        # Save the model.saes to a temp file
        save_module(
            config_dict=pretrained_config.model_dump(mode="json"),
            save_dir=tmp_path,
            module=model.saes,
            model_filename=filename,
        )
        sae_params.append(model.saes)

    new_config = get_tinystories_config(
        {
            "saes": {
                "sae_positions": all_positions,
                "pretrained_sae_paths": [tmp_path / filename for filename in filenames],
            }
        }
    )
    # Create a new model to load in the pretrained SAEs
    new_tlens_model = HookedTransformer(tlens_config)
    new_model = SAETransformer(
        tlens_model=new_tlens_model,
        raw_sae_positions=all_positions,
        dict_size_to_input_ratio=new_config.saes.dict_size_to_input_ratio,
        init_decoder_orthogonal=False,
    )

    assert isinstance(new_config.saes.pretrained_sae_paths, list)
    # Now load in the pretrained SAE to the new model
    trainable_param_names = load_pretrained_saes(
        saes=new_model.saes,
        pretrained_sae_paths=new_config.saes.pretrained_sae_paths,
        all_param_names=[name for name, _ in new_model.saes.named_parameters()],
        retrain_saes=False,
    )

    model_named_params = dict(new_model.saes.named_parameters())
    suffixes = ["encoder.0.weight", "encoder.0.bias", "decoder.weight", "decoder.bias"]
    assert trainable_param_names == [
        f"blocks-3-hook_resid_post.{suffix}" for suffix in suffixes
    ], "Only block 2 should be trainable"

    for suffix in suffixes:
        # Check that the params for block 0 are the same as the first pretrained SAE
        assert torch.allclose(
            model_named_params[f"blocks-0-hook_resid_post.{suffix}"],
            sae_params[0].state_dict()[f"blocks-0-hook_resid_post.{suffix}"],
        )
        # Check that the params for blocks 1 and 2 are the same as the second pretrained SAE
        for block in [1, 2]:
            assert torch.allclose(
                model_named_params[f"blocks-{block}-hook_resid_post.{suffix}"],
                sae_params[1].state_dict()[f"blocks-{block}-hook_resid_post.{suffix}"],
            )
