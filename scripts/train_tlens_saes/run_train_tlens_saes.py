"""Script for training SAEs on top of a transformerlens model.

Usage:

    python run_train_tlens_saes.py path/to/config.yaml

    Args:
        path/to/config.yaml: Path to the config file for training the SAEs.
"""
import fire

from sparsify.train_tlens_saes import main

if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
