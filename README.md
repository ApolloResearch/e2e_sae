# Sample project

This is a sample project that uses the engineering practices described in [Apollo Engineering Guide](https://docs.google.com/document/d/1LJedFJKrGs7vi-xA1ucQQxj_y85Z9x4lEmf4sFgeX6g/edit).

If you wish to use this repository as a template for your project, simply click `Use this template -> Create a new repository` on the Github UI. Once your repository is created, run

```bash
chmod +x setup_pkg.sh && ./setup_pkg.sh <name_of_your_package>
```

with the name of your package as the argument. This will rename the package. You should also put
your project details in the `[project]` entry in `pyproject.toml`. You may then wish to remove the
files and content that you do not need for your project.

## Privacy level

You must set the privacy level of your repository in `ACCESS.md`, listing all parties that can access the project. See our [privacy levels](https://www.apolloresearch.ai/blog/security) for more information.

## Installation

From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```

## Usage

### MNIST

The repo provides an example of an experiment which trains an MLP on MNIST. To run the experiment,
define a yaml config file (or use the provided `sparisfy/scripts/train_mnist/mnist.yaml`) and run

```bash
python sparisfy/scripts/train_mnist/run_train_mnist.py <path_to_config_file>
```

You may be asked to enter your wandb API key. You can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, to avoid entering your API key on program execution, you can set the environment variable `WANDB_API_KEY` to your API key.
