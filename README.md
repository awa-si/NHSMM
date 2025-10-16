# NHSMM — Neural Hidden Semi-Markov Models

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**NHSMM** is a modular PyTorch library for hierarchical regime detection and temporal modeling using **Neural Hidden Semi-Markov Models (HSMM's)**.  

It extends classical HSMM's with learnable **emission**, **duration**, and **transition** components — enabling **context-aware sequence modeling** for financial regimes, sensor signals, and other temporal domains.

---

## 🚀 Key Features

- **Neural Emission Models** — Gaussian, Multinomial, or Bernoulli outputs, fully differentiable  
- **Neural Duration Models** — context-dependent discrete duration distributions  
- **Neural Transition Models** — learnable, covariate-aware transition probabilities  
- **Subclassable Distributions** — Duration, Transition, and Emission inherit from standard PyTorch `Distribution` objects for composability  
- **HSMM-HMM Hybrid Inference**
- **GPU-ready Implementation** — leverage PyTorch automatic differentiation  
- **Flexible Contextual Inputs** — external covariates modulate emissions, durations, and transitions  

---

## 📦 Installation

### From source (recommended for development)

```bash
git clone https://github.com/awwea/NeuralHSMM.git
cd NeuralHSMM
pip install -e .
```

### From PyPI (when published)

```bash
pip install nhsmm (TODO)
```

---

## 🧩 Package Structure

```
nhsmm/
├── defaults.py
├── models/
│   ├── hsmm.py
│   ├── neural.py
│   └── __init__.py
├── distributions/
│   ├── neural.py
│   └── __init__.py
├── utilities/
│   ├── utils.py
│   ├── constraints.py
│   ├── seeds.py
│   └── __init__.py
└── __init__.py
```

---

## 🧠 Example Usage

```python
import torch
from nhsmm.models import NeuralHSMM
from nhsmm.distributions import NeuralGaussian, NeuralDuration, NeuralTransition

# Example input sequence: (time, features)
X = torch.randn(256, 32)

# Initialize HSMM
model = NeuralHSMM(
    n_states=4,
    emission=NeuralGaussian(input_dim=32, hidden_dim=64),
    duration=NeuralDuration(input_dim=32, hidden_dim=32),
    transition=NeuralTransition(input_dim=32, hidden_dim=32),
)

# Forward pass
log_prob = model.log_prob(X)
states = model.viterbi(X)

print("Log-likelihood:", log_prob.item())
print("Most likely state sequence:", states.shape)
```

---

## ⚙️ Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest -v

# Code formatting & linting
black nhsmm
ruff check nhsmm
```

---

## 🧾 License

MIT © 2025 AWA
