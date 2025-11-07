# NHSMM — Neural Hidden Semi-Markov Models

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**NHSMM** is a modular PyTorch library for **hierarchical regime detection** and **temporal sequence modeling** using ** Hidden Semi-Markov Models (HSMMs) **.

It extends classical HSMMs with **learnable emission, duration, and transition components**, enabling **context-aware modeling** of temporal sequences in domains such as financial regimes, sensor signals, speech, and biomedical data.

---

## 🚀 Key Features

* **Neural Emission Models** — Gaussian, Multinomial, or Bernoulli outputs, fully differentiable.
* **Neural Duration Models** — explicit discrete duration distributions, context-modulated.
* **Neural Transition Models** — learnable, covariate-aware transitions with gating and temperature scaling.
* **Contextual HSMM** — external covariates dynamically modulate emissions, durations, and transitions.
* **Subclassable Distributions** — Initial, Duration, Transition, and Emission inherit from PyTorch `Distribution`.
* **HSMM-HMM Hybrid Inference** — forward-backward and Viterbi algorithms adapted for neural components.
* **GPU-ready Implementation** — fully batched operations for fast training and inference.
* **EM-style Updates & Initialization** — maximum likelihood or differentiable updates with temperature annealing.

---

## 📦 Installation

### From Source (Recommended for Development)

```bash
git clone https://github.com/awwea/NeuralHSMM.git
cd NeuralHSMM
pip install -e .
```

### From PyPI (Upcoming)

```bash
pip install nhsmm
```

---

## 🧩 Package Structure

```
nhsmm/
├── constants.py           # Default configuration and base classes
├── context.py             # Contextual Encoder
├── models/
│   ├── hsmm.py            # Core HSMM model & inference
│   ├── gaussian.py        # Contextual gaussian components
│   ├── neural.py          # Contextual neural components
│   └── __init__.py
├── distributions/
│   ├── default.py         # Initial, Duration, Transition, Emission
│   └── __init__.py
├── utilities/
│   ├── utils.py
│   ├── constraints.py
│   ├── seeds.py
│   └── __init__.py
└── __init__.py
```

---

## 🧠 Usage Example (Context-Aware)

```python
import torch
from nhsmm.models import NeuralHSMM

# Example input sequence: 256 time steps, 32 features
X = torch.randn(256, 32)

# Optional external context: could be market indicators, sensor readings, or embeddings
context = torch.randn(256, 16)  # 16-dimensional covariates

# Initialize a 4-state Neural HSMM
model = NeuralHSMM(
    n_states=4,
    context_dim=context.shape[1],  # enable context-aware modulation
    hidden_dim=64,                 # hidden dimension for neural adapters
)

# Forward pass: compute log-likelihood
log_prob = model.log_prob(X, context=context)

# Decode most likely state sequence (Viterbi)
states = model.viterbi(X, context=context)

# Sample synthetic sequences conditioned on context
samples = model.sample(context=context)

# Compute expected state durations
expected_durations = model.duration.expected_duration(context=context)

print("Log-likelihood:", log_prob.item())
print("Most likely states:", states.shape)
print("Sampled states:", samples.shape)
print("Expected durations per state:", expected_durations)
```

---

## 🔍 Contextual Flow Diagram (Conceptual)

```
      ┌───────────────┐
      │ External Input │  ← covariates, features, embeddings
      └───────┬───────┘
              │
              ▼
┌─────────────────────────┐
│ Neural Initial State     │  ← context-modulated initial probabilities
│ Distribution             │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Neural Transition        │  ← context-gated, temperature-scaled transitions
│ Distribution             │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Neural Duration          │  ← context-aware duration probabilities per state
│ Distribution             │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Emission Model           │  ← context-modulated observation likelihoods
│ (Gaussian / Multinomial) │
└─────────────────────────┘
```

* Context flows into **Initial**, **Transition**, and **Duration** distributions.
* Each component can be **batch-modulated** and supports **temperature scaling** without affecting argmax modes.
* EM-style updates allow **differentiable learning** of all parameters while preserving stability.

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
