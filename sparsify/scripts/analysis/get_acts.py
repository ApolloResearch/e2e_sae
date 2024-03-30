# %%
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
import transformer_lens as tlens
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

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
batch_size = 10
loader, tokenizer = create_data_loader(config.eval_data, batch_size=batch_size)


def test_saes():
    tokens = next(iter(loader))["input_ids"].to("cuda")
    raw_logits, raw_acts = model.forward_raw(
        tokens, run_entire_model=True, final_layer=None, cache_positions=[sae_pos]
    )
    sae_logits, sae_acts = model.forward(tokens, [sae_pos])

    print("raw loss: ", tlens.utils.lm_cross_entropy_loss(raw_logits, tokens).item())
    print("sae loss: ", tlens.utils.lm_cross_entropy_loss(sae_logits, tokens).item())

    assert torch.allclose(raw_acts[sae_pos], sae_acts[sae_pos].input)


ActTensor = Float[torch.Tensor, "batch seq emb"]
LogitTensor = Float[torch.Tensor, "batch seq vocab"]


def kl_div(new_logits: LogitTensor, old_logits: LogitTensor) -> Float[torch.Tensor, "batch seq"]:
    return F.kl_div(
        F.log_softmax(new_logits, dim=-1), F.softmax(old_logits, dim=-1), reduction="none"
    ).sum(-1)


class Acts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, "batch seq"] = torch.empty(0, 1024, dtype=torch.long)
    orig: ActTensor = torch.empty(0, 1024, 768)
    recon: ActTensor = torch.empty(0, 1024, 768)
    orig_logits: LogitTensor = torch.empty(0, 1024, 50257)
    new_logits: LogitTensor = torch.empty(0, 1024, 50257)
    kl: Float[torch.Tensor, "batch seq"] = torch.empty(0, 1024)

    def add(
        self, tokens: torch.Tensor, acts: SAEActs, orig_logits: LogitTensor, new_logits: LogitTensor
    ):
        self.tokens = torch.cat([self.tokens, tokens.cpu()])
        self.orig = torch.cat([self.orig, acts.input.cpu()])
        self.recon = torch.cat([self.recon, acts.output.cpu()])
        self.orig_logits = torch.cat([self.orig_logits, orig_logits.cpu()])
        self.new_logits = torch.cat([self.new_logits, new_logits.cpu()])
        self.kl = torch.cat([self.kl, kl_div(new_logits, orig_logits).cpu()])

    def __len__(self):
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"Acts(len={len(self)})"


@torch.no_grad()
def get_acts(batches=1) -> Acts:
    loader_iter = iter(loader)
    acts = Acts()

    for _ in tqdm.trange(batches, disable=(batches == 1)):
        tokens = next(loader_iter)["input_ids"].to("cuda")
        orig_logits, _ = model.forward_raw(tokens, run_entire_model=True, final_layer=None)
        sae_logits, sae_acts = model.forward(tokens, [sae_pos])
        acts.add(tokens, sae_acts[sae_pos], orig_logits=orig_logits, new_logits=sae_logits)

    return acts
