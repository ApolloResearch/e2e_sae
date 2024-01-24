from pathlib import Path

from setuptools import find_packages, setup

if Path("requirements.txt").exists():
    requirements = Path("requirements.txt").read_text("utf-8").splitlines()
else:
    requirements = []

setup(
    name="sparsify",
    version="0.0.1",
    description="Repo for sparsifying networks end-to-end",
    long_description=Path("README.md").read_text("utf-8"),
    author="Apollo Research",
    author_email="lee@apolloresearch.ai",
    url="https://github.com/ApolloResearch/sparsify",
    packages=find_packages(include=["sparsify", "sparsify.*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "pylint",
            "pytest",
            "types-PyYAML",
            "types-tqdm",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
