# Sparsify

## Installation

From the root of the repository, run one of

```bash
make install-dev  # (recommended) Installs the package, dev requirements and pre-commit hooks
make install  # Just installs the package (runs `pip install -e .`)
```

## Development

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

## Usage

Weights and Biases is used to track experiments. Place your api key and entity name in `.env`. An
example is provided in `.env.example`.

All entrypoints are in `sparsify/scripts`. Configs, scripts, and results are all stored in this
directory (results are not checked in to git).