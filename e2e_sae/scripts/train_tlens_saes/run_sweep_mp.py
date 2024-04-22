"""Run the training script with different config params over multiple GPUs.

Usage:
    python run_sweep_mp.py <path/to/base/config.yaml>
"""

import subprocess
from tempfile import NamedTemporaryFile

import yaml
from fire import Fire

from e2e_sae.scripts.train_tlens_saes.run_train_tlens_saes import Config
from e2e_sae.settings import REPO_ROOT
from e2e_sae.utils import replace_pydantic_model

SCRIPT_PATH = f"{REPO_ROOT}/e2e_sae/scripts/train_tlens_saes/run_train_tlens_saes.py"


def main(config_path_str: str) -> None:
    """Run the training script with different sae_position values.

    NOTE: You must specify the GPU indices to use in the `gpu_idxs` list.
    """
    sweep_name = "tinystories-1m_sparsity-coeff"
    values = [0.01, 0.001, 0.0001, 0.00001]
    gpu_idxs = [0, 1, 2, 3]

    assert len(values) == len(
        gpu_idxs
    ), "Currently only supports having the same number of values and gpu_idxs"

    with open(config_path_str) as f:
        base_config = Config(**yaml.safe_load(f))

    for idx, value in zip(gpu_idxs, values, strict=True):
        update_dict = {
            "train": {"loss": {"sparsity": {"coeff": value}}},
            "wandb_project": sweep_name,
        }
        new_config = replace_pydantic_model(base_config, update_dict)
        # Write the config to a temporary file and then call a subprocess to run the training script
        print(new_config)
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(new_config.model_dump(mode="json"), f)
            config_path = f.name

        session_exists = subprocess.run(f"tmux has-session -t {idx}".split(), capture_output=True)
        if session_exists.returncode == 0:
            # Session exists, kill it
            subprocess.run(f"tmux kill-session -t {idx}".split())

        # Create a new tmux session
        subprocess.run(f"tmux new-session -d -s {idx}".split())

        train_command = f"CUDA_VISIBLE_DEVICES={idx} python {SCRIPT_PATH} {config_path}"
        tmux_send_keys_cuda_command = f"tmux send-keys -t {idx} '{train_command}' Enter"
        subprocess.run(tmux_send_keys_cuda_command, shell=True)


if __name__ == "__main__":
    Fire(main)
