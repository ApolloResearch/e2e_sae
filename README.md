# Sample project

This is a sample project that uses the engineering practices described in [Apollo Engineering Guide](https://docs.google.com/document/d/1LJedFJKrGs7vi-xA1ucQQxj_y85Z9x4lEmf4sFgeX6g/edit).

If you wish to use this repository as a template for your project, simply click `Use this template -> Create a new repository` on the Github UI. Once your repository is created, run

```bash
chmod +x setup_pkg.sh && ./setup_pkg.sh <name_of_your_package>
```

with the name of your package as the argument. This will rename the package and set up the pre-commit hooks. You should also provide your name and email address in `setup.py`. You may then wish to remove the files and content that you do not need for your project.

## Privacy level

You must set the privacy level of your repository in `ACCESS.md`, listing all parties that can access the project. See our [privacy levels](https://www.apolloresearch.ai/blog/security) for more information.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Development

To install the development dependencies, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

A pre-commit hook that runs black, isort, mypy, and tests is saved in the `.pre-commit` file. If
you ran the `setup_pkg.sh` as instructed earlier, this pre-commit will be automatically setup.
Otherwise, you can copy it to the `.git/hooks/`
dir and make it executable
(`cp .pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`).

## Usage

### MNIST

To train an MLP on MNIST, define a config file (see `configs/mnist.yaml` for an example) and run

```bash
python scripts/train_mnist.py <path_to_config_file>
```

You may be asked to enter your wandb API key. You can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, to avoid entering your API key on program execution, you can set the environment variable `WANDB_API_KEY` to your API key.
