#!/usr/bin/env python3
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="NeuralHSMM",
    version="0.1.0",
    author="AWA",
    user="awwea",
    branch="master",
    author_email="andre@awwea.com",
    description="Neural Hidden Semi-Markov Models with contextual duration and emission modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awwea/NeuralHSMM",
    license="MIT",

    packages=find_packages(exclude=("tests", "examples", "notebooks", "docs")),
    python_requires=">=3.9",

    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.7.2",
        "polars>=1.34.0",
        "tqdm>=4.66",
        "matplotlib>=3.9",
        "typing_extensions>=4.11",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "black>=24.0",
            "ruff>=0.6",
            "mypy>=1.11",
            "build",
            "twine",
        ],
        "notebooks": [
            "jupyterlab",
            "seaborn",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    keywords="neural-hsmm hidden-semi-markov-model probabilistic deep-learning pytorch regime-detection",
    include_package_data=True,
    zip_safe=False,
)
