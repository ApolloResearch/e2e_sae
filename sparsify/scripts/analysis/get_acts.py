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


def load_sae_from_path(sae_path: Path):
    with open(sae_path.parent / "config.yaml") as f:
        config = Config(**yaml.safe_load(f))
    tl_model = tlens.HookedTransformer.from_pretrained(config.tlens_model_name)  # type: ignore
    model = SAETransformer(config, tl_model, config.saes.sae_positions)
    model.saes.load_state_dict(torch.load(sae_path))
    return model, config


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
def get_acts(model: SAETransformer, config: Config, batch_size=5, batches=1) -> Acts:
    assert config.eval_data is not None
    loader, _ = create_data_loader(config.eval_data, batch_size=batch_size)
    loader_iter = iter(loader)
    acts = Acts()
    sae_pos = config.saes.sae_positions[0]

    for _ in tqdm.trange(batches, disable=(batches == 1)):
        tokens = next(loader_iter)["input_ids"].to("cuda")
        orig_logits, _ = model.forward_raw(tokens, run_entire_model=True, final_layer=None)
        sae_logits, sae_cache = model.forward(tokens, [sae_pos])
        sae_acts = sae_cache[sae_pos]
        assert isinstance(sae_acts, SAEActs)
        assert sae_logits is not None
        acts.add(tokens, sae_acts, orig_logits=orig_logits, new_logits=sae_logits)

    return acts
