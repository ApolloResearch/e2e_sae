# e2e_sae

This library is used to train and evaluate Sparse Autoencoders (SAEs). It handles the following
training types:
- e2e (end-to-end): Loss function includes sparsity and final model kl_divergence.
- e2e + downstream reconstruction: Loss function includes sparsity, final model kl_divergence, and MSE
    at downstream layers.
- local (i.e. vanilla SAEs): Loss function includes sparsity and MSE at the SAE layer
- Any combination of the above.

See our [paper](https://publications.apolloresearch.ai/end_to_end_sparse_dictionary_learning) which argues for training SAEs e2e rather than locally. All SAEs presented in the paper can be found at https://wandb.ai/sparsify/gpt2 and can be loaded using this library.

## Usage
### Installation
```bash
pip install e2e_sae
```

### Train SAEs on any [TransformerLens](https://github.com/neelnanda-io/TransformerLens) model
If you would like to track your run with Weights and Biases, place your api key and entity name in
a new file called `.env`. An example is provided in [.env.example](.env.example).

Create a config file (see gpt2 configs [here](e2e_sae/scripts/train_tlens_saes/) for examples).
Then run
```bash
python e2e_sae/scripts/train_tlens_saes/run_train_tlens_saes.py <path_to_config>
```

If using a Colab notebook, see [this example](demos/train_saes.ipynb).

Sample wandb sweep configs are provided in [e2e_sae/scripts/train_tlens_saes/](e2e_sae/scripts/train_tlens_saes/).

The library also contains scripts for training mlps and SAEs on mlps, as well as training
custom transformerlens models and SAEs on these models (see [here](e2e_sae/scripts/)).
### Load a Pre-trained SAE
You can load any pre-trained SAE (and accompanying TransformerLens model) trained using this library
from Weights and Biases or locally by running
```python
from e2e_sae import SAETransformer
model = SAETransformer.from_wandb("<entity/project/run_id>")
# or, if stored locally
model = SAETransformer.from_local_path("/path/to/checkpoint/dir") 
```
All runs in our
[paper](https://publications.apolloresearch.ai/end_to_end_sparse_dictionary_learning)
can be loaded this way (e.g.[sparsify/gpt2/tvj2owza](https://wandb.ai/sparsify/gpt2/runs/tvj2owza)).


This will instantiate a `SAETransformer` class, which contains a TransformerLens model with SAEs
attached. To do a forward pass without SAEs, use the `forward_raw` method, to do a forward pass with
SAEs, use the `forward` method (or simply call the SAETansformer instance).

The dictionary elements of an SAE can be accessed via `SAE.dict_elements`. This is will normalize
the decoder elements to have norm 1.

### Analysis
To reproduce all of the analysis in our
[paper](https://publications.apolloresearch.ai/end_to_end_sparse_dictionary_learning) use the
scripts in `e2e_sae/scripts/analysis/`.

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
