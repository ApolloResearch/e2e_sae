"""Run wandb sweep agents in parallel using tmux sessions.

Usage:
    python run_wandb_sweep.py <wandb_agent_id>
"""
import subprocess

from fire import Fire


def main(agent_id: str) -> None:
    """Run the training script with different GPU indices.
    NOTE: You must specify the GPU indices to use in the `gpu_idxs` list.
    """
    gpu_idxs = [0, 1, 2, 3, 4, 5]

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
