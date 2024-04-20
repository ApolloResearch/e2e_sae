"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""
from collections.abc import Iterable

import pytest
import torch
from pytest import Config, Item, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "cpuslow: mark test as slow to run if no gpu available")


def pytest_collection_modifyitems(config: Config, items: Iterable[Item]) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_cpuslow = pytest.mark.skip(reason="need --runslow option to run since no gpu available")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "cpuslow" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cpuslow)
