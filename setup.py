from pathlib import Path

from setuptools import find_packages, setup

if Path("requirements.txt").exists():
    requirements = Path("requirements.txt").read_text("utf-8").splitlines()
else:
    requirements = []

setup(
    name="mypkg",
    version="0.0.1",
    description="Sample repository using Apollo best engineering practices",
    long_description=Path("README.md").read_text("utf-8"),
    author="Apollo Research",
    author_email="dan@apolloresearch.ai",
    url="https://github.com/ApolloResearch/sample",
    packages=find_packages(),
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
