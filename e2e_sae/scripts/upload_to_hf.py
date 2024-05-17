"""
Process a set of SAEs for uploading to hf. Doesn't actually upload anything.
"""
import os
import shutil
from pathlib import Path

import torch
import tqdm
import wandb

from e2e_sae.models.transformers import SAETransformer

wandb_id_to_hf_name = [
    ("ue3lz0n7", "local_similar_ce_layer_2"),
    ("ovhfts9n", "e2e_similar_ce_layer_2"),
    ("visi12en", "downstream_similar_ce_layer_2"),
    ("1jy3m5j0", "local_similar_ce_layer_6"),
    ("zgdpkafo", "e2e_similar_ce_layer_6"),
    ("2lzle2f0", "downstream_similar_ce_layer_6"),
    ("m2hntlav", "local_similar_ce_layer_10"),
    ("8crnit9h", "e2e_similar_ce_layer_10"),
    ("cvj5um2h", "downstream_similar_ce_layer_10"),
    ("6vtk4k51", "local_similar_l0_layer_2"),
    ("bst0prdd", "e2e_similar_l0_layer_2"),
    ("e26jflpq", "downstream_similar_l0_layer_2"),
    ("jup3glm9", "local_similar_l0_layer_6"),
    ("tvj2owza", "e2e_similar_l0_layer_6"),
    ("2lzle2f0", "downstream_similar_l0_layer_6"),
    ("5vmpdgaz", "local_similar_l0_layer_10"),
    ("8crnit9h", "e2e_similar_l0_layer_10"),
    ("cvj5um2h", "downstream_similar_l0_layer_10"),
]

out_dir = Path(__file__).parent / "out/hf_models"
out_dir.mkdir(parents=True, exist_ok=True)

api = wandb.Api()

for wandb_id, hf_name in tqdm.tqdm(wandb_id_to_hf_name):
    model = SAETransformer.from_wandb(f"gpt2/{wandb_id}")
    sae = list(model.saes.values())[0]
    raw_state_dict = sae.state_dict()
    state_dict = {
        "encoder.weight": raw_state_dict["encoder.0.weight"],
        "encoder.bias": raw_state_dict["encoder.0.bias"],
        "decoder.weight": sae.dict_elements,
        "decoder.bias": raw_state_dict["decoder.bias"],
    }

    torch.save(state_dict, out_dir / f"{hf_name}.pt")

    run = api.run(f"gpt2/{wandb_id}")

    cache_dir = Path(os.environ.get("SAE_CACHE_DIR", "/tmp/"))
    model_cache_dir = cache_dir / "gpt2/{wandb_id}"

    train_config_file_remote = [
        file for file in run.files() if file.name.endswith("final_config.yaml")
    ][0]

    train_config_file = train_config_file_remote.download(
        exist_ok=True, replace=True, root=model_cache_dir
    ).name

    shutil.copy(train_config_file, out_dir / f"{hf_name}.yaml")
