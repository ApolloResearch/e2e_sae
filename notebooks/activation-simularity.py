# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import transformer_lens as tlens
import yaml

from sparsify.data import create_data_loader
from sparsify.hooks import SAEActs
from sparsify.models.transformers import SAETransformer
from sparsify.scripts.train_tlens_saes.run_train_tlens_saes import Config

# %%

config_str = """
saes:
    sae_positions: blocks.6.hook_resid_pre
    dict_size_to_input_ratio: 60
tlens_model_name: gpt2-small
train_data:
    dataset_name: apollo-research/Skylion007-openwebtext-tokenizer-gpt2
    is_tokenized: true
    tokenizer_name: gpt2
    streaming: true
    split: train
    n_ctx: 1024
    seed: 0
    column_name: input_ids
eval_data:
    dataset_name: apollo-research/Skylion007-openwebtext-tokenizer-gpt2
    is_tokenized: true
    tokenizer_name: gpt2
    streaming: true
    split: train
    n_ctx: 1024
    seed: 0
    column_name: input_ids
save_every_n_samples: 9999
eval_n_samples: 9999
loss:
    sparsity:
        coeff: 1
    logits_kl:
        coeff: 1
    out_to_in: null
    in_to_orig: null
    out_to_orig: null
batch_size: 8
lr: 0.0001
n_samples: 10000
"""
config = Config(**yaml.safe_load(config_str))
tl_model = tlens.HookedTransformer.from_pretrained(config.tlens_model_name)
sae_pos = "blocks.6.hook_resid_pre"
model = SAETransformer(config, tl_model, [sae_pos])
sae_path = Path("/mnt/ssd-interp/nix/sparsify/sparsify/scripts/train_tlens_saes/out/gpt2_sae.pt")
model.saes.load_state_dict(torch.load(sae_path))
model.to("cuda")
# %%
batch_size = 5
test_loader = create_data_loader(config.eval_data, batch_size=batch_size)[0]

tokens = next(iter(test_loader))["input_ids"].to("cuda")


@torch.no_grad()
def get_acts() -> SAEActs:
    raw_logits, raw_acts = model.forward_raw(
        tokens, run_entire_model=True, final_layer=None, cache_positions=[sae_pos]
    )
    sae_logits, sae_acts = model.forward(tokens, [sae_pos])

    print("raw loss: ", tlens.utils.lm_cross_entropy_loss(raw_logits, tokens).item())
    print("sae loss: ", tlens.utils.lm_cross_entropy_loss(sae_logits, tokens).item())

    assert torch.allclose(raw_acts[sae_pos], sae_acts[sae_pos].input)
    return sae_acts[sae_pos]


sae_acts = get_acts()
# %%


in_norms = torch.norm(sae_acts.input.flatten(0, 1), dim=-1)
out_norms = torch.norm(sae_acts.output.flatten(0, 1), dim=-1)
plt.scatter(in_norms.cpu(), out_norms.cpu(), alpha=0.2, s=2, c="k")
plt.xlabel("Input Norm")
plt.ylabel("Output Norm")
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.plot([0, 200], [0, 200], "r-")
plt.gcf().set_size_inches(4, 4)

# %%

sims = torch.nn.functional.cosine_similarity(sae_acts.input, sae_acts.output, dim=-1)

# %%
plt.hist(sims.cpu().flatten(), bins=100)
plt.xlabel("Cosine Similarity")

# %%

ln_input = torch.nn.functional.layer_norm(sae_acts.input, [768])
ln_output = torch.nn.functional.layer_norm(sae_acts.output, [768])

ln_input_norms = torch.norm(ln_input.flatten(0, 1), dim=-1)
ln_output_norms = torch.norm(ln_output.flatten(0, 1), dim=-1)

ln_sims = torch.nn.functional.cosine_similarity(ln_input, ln_output, dim=-1)


ln_sq_err = (ln_input - ln_output).pow(2).sum(-1)
sq_err = (sae_acts.input - sae_acts.output).pow(2).sum(-1)

plt.hist(sq_err.cpu().flatten(), bins=100)
plt.xlabel("Sq err")

# %%

plt.scatter(sq_err.flatten().cpu(), sims.flatten().cpu(), s=2, c="k", alpha=0.2)
plt.xlim(0, 10000)

# %%
is_198 = tokens.cpu() == 198  # `\n`
plt.scatter(sq_err.flatten().cpu(), sims.flatten().cpu(), s=2, c=is_198, alpha=0.5, cmap="cool")
plt.xlim(0, 10000)
plt.ylabel("cosine sim")
plt.xlabel("MSE")

# %%
