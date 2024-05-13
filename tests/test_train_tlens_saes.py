from pathlib import Path

import pytest

from e2e_sae.data import DatasetConfig
from e2e_sae.losses import (
    LogitsKLLoss,
    LossConfigs,
    OutToInLoss,
    SparsityLoss,
)
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import Config, SAEsConfig
from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import main as run_training


@pytest.mark.cpuslow
def test_train_tiny_gpt():
    """Test training an SAE on a custom tiny 2-layer GPT-style model.

    NOTE: This could be sped up using a a custom dataset stored locally but we don't yet support
    this.
    """
    model_path = Path(
        "e2e_sae/scripts/train_tlens/sample_models/tiny-gpt2_lr-0.001_bs-16_2024-04-21_14-01-14/epoch_1.pt"
    )
    config = Config(
        wandb_project=None,
        wandb_run_name=None,  # If not set, will use a name based on important config values
        wandb_run_name_prefix="",
        seed=0,
        tlens_model_name=None,
        tlens_model_path=model_path,
        save_dir=None,
        n_samples=3,
        save_every_n_samples=None,
        eval_every_n_samples=2,  # Just eval once at start and once during training
        eval_n_samples=2,
        log_every_n_grad_steps=20,
        collect_act_frequency_every_n_samples=2,
        act_frequency_n_tokens=2000,
        batch_size=2,
        effective_batch_size=2,
        lr=5e-4,
        lr_schedule="cosine",
        min_lr_factor=0.1,
        warmup_samples=2,
        max_grad_norm=10.0,
        loss=LossConfigs(
            sparsity=SparsityLoss(p_norm=1.0, coeff=1.5),
            in_to_orig=None,
            out_to_orig=None,
            out_to_in=OutToInLoss(coeff=0.0),
            logits_kl=LogitsKLLoss(coeff=1.0),
        ),
        train_data=DatasetConfig(
            dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            is_tokenized=True,
            tokenizer_name="gpt2",
            streaming=True,
            split="train",
            n_ctx=1024,
        ),
        eval_data=DatasetConfig(
            dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            is_tokenized=True,
            tokenizer_name="gpt2",
            streaming=True,
            split="train",
            n_ctx=1024,
        ),
        saes=SAEsConfig(
            retrain_saes=False,
            pretrained_sae_paths=None,
            sae_positions=[
                "blocks.1.hook_resid_pre",
            ],
            dict_size_to_input_ratio=1.0,
        ),
    )
    run_training(config)
