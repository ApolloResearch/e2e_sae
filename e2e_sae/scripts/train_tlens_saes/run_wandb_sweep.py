"""Run wandb sweep agents in parallel using tmux sessions.

Usage:
    The script can be invoked from the command line, providing the wandb agent ID and optionally,
    GPU indices separated by a comma. If no GPU indices are provided, the script will automatically
    use all available GPUs.

    Example command to run the script specifying GPUs:
        python run_wandb_sweep.py your_agent_id 0,1,4

    Example command to run the script using all available GPUs:
        python run_wandb_sweep.py your_agent_id

"""
import subprocess

import torch
from fire import Fire


def main(agent_id: str, gpu_idxs: tuple[int, ...] | int | None = None) -> None:
    """Run the training script with specified GPU indices.

    Args:
        agent_id: The wandb agent ID.
        gpu_idxs: The GPU indices to use for training. If None, all available GPUs will be used.
    """
    if isinstance(gpu_idxs, int):
        gpu_idxs = (gpu_idxs,)
    elif gpu_idxs is None:
        gpu_idxs = tuple(range(torch.cuda.device_count()))

    assert isinstance(gpu_idxs, tuple), "gpu_idxs must be a tuple of integers"

    print(f"Running wandb agent {agent_id} on GPUs {gpu_idxs}")
    for idx in gpu_idxs:
        session_exists = subprocess.run(f"tmux has-session -t {idx}".split(), capture_output=True)
        if session_exists.returncode == 0:
            # Session exists, kill it
            subprocess.run(f"tmux kill-session -t {idx}".split())

        # Create a new tmux session
        subprocess.run(f"tmux new-session -d -s {idx}".split())

        train_command = f"CUDA_VISIBLE_DEVICES={idx} wandb agent {agent_id}"
        tmux_send_keys_cuda_command = f"tmux send-keys -t {idx} '{train_command}' Enter"
        subprocess.run(tmux_send_keys_cuda_command, shell=True)


if __name__ == "__main__":
    Fire(main)
