# Sparsify

This library is used to train and evaluate Sparse Autoencoders (SAEs). It handles the following
training types:
- End-to-end (e2e): Loss function includes sparsity and final model kl_divergence.
- e2e + future reconstruction: Loss function includes sparsity, final model kl_divergence, and MSE
    at future layers.
- Local (i.e. vanilla SAEs): Loss function includes sparsity and MSE at the SAE layer
- Any combination of the above.

See [TODO: add paper] which argues for training SAEs e2e rather than locally.

## Usage
### Installation
```bash
pip install .
```

### Train SAEs on any [TransformerLens](https://github.com/neelnanda-io/TransformerLens) model
If you would like to track your run with Weights and Biases, place your api key and entity name in
a new file called `.env`. An example is provided in [.env.example](.env.example).

Create a config file (see gpt2 configs [here](sparsify/scripts/train_tlens_saes/) for examples).
Then run
```bash
python sparsify/scripts/train_tlens_saes/run_train_tlens_saes.py <path_to_config>
```

If using a Colab notebook, see this example ([TODO: add link]).

Note that the library also contains scripts for training mlps and SAEs on mlps, as well as training
custom transformerlens models and SAEs on these models (see [here](sparsify/scripts/)).

### Load a Pre-trained SAE
You can load any pre-trained SAE (and accompanying TransformerLens model) trained using this library
from Weights and Biases by running
```python
from sparsify import SAETransformer
model = SAETransformer.from_wandb("<entity/project/run_id>")
# or, if training was done locally:
model = SAETransformer.from_checkpoint("/path/to/checkpoint/dir") 
```
Note that all runs in the [TODO: Add e2e paper]() can be loaded this way (e.g.
[sparsify/gpt2/tvj2owza](https://wandb.ai/sparsify/gpt2/runs/tvj2owza)).


This will instantiate a `SAETransformer` class, which contains a TransformerLens model with SAEs
attached. To do a forward pass without SAEs, use the `forward_raw` method, to do a forward pass with
SAEs, use the `forward` method (or simply call the SAETansformer instance).

## Contributing
Developer dependencies are installed with `make install-dev`, which will also install pre-commit
hooks.

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit checks on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```

This library is maintained by [Dan Braun](https://danbraunai.github.io/).

Join the [Open Source Mechanistic Interpretability Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2hk7rcm8g-IIuaxpte_1GHp5joc~1kww)
to chat about this library and other projects in the space!