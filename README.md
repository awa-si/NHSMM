
# NHSMM — (Neural) Hidden Semi-Markov Models
- *Documentation: [wiki](Wiki)*
- *Version: [version](alpha)*

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

**Overview:**  
This document serves as a **self-contained guide** for NHSMM, the modular PyTorch library that forms the foundation for **State Aware Engine (SAE)**. It is suitable for **developers, data scientists, and system integrators** to understand, download, and start using the library across multiple domains.

**Highlights HSMM + neural + contextual + hierarchical features**
---

## Overview

**NHSMM** enables **temporal sequence modeling** and **hidden-state detection** using **Hidden Semi-Markov Models (HSMMs)**.

It powers **SAE** State Aware Engine, a **cross-domain platform** for detecting hidden regimes in **IoT, Health, Security, Robotics, and Finance**. Anyway it can serve as solid base for any research project based on hsmm.

**SAE** leverages NHSMM for:  
- **Cloud-first SaaS deployment** for immediate access  
- **On-prem / Edge deployment** for low-latency or privacy-sensitive systems  
- **Quantum/hardware accelerator readiness** for next-generation predictive modeling

---

## 🚀 Key Features

* Duration Models — explicit discrete duration distributions, context-modulated  
* Emission Models — Gaussian, Multinomial, or Bernoulli outputs, fully differentiable  
* Transition Models — learnable, covariate-aware transitions with gating and temperature scaling  
* Contextual HSMM — external covariates dynamically modulate emissions, durations, and transitions  
* HSMM-HMM Hybrid Inference — forward-backward and Viterbi algorithms adapted for neural components  
* Subclassable Distributions — Initial, Duration, Transition, and Emission inherit from PyTorch `Distribution`  
* EM-style Updates & Initialization — maximum likelihood or differentiable updates with temperature annealing  
* Multi-Domain Usage — supports trading, IoT, robotics, wearable health, cybersecurity applications and even more...
* Extensible — foundation for SAE adapters and API integration for multi-domain systems  
* GPU-ready Implementation — fully batched operations for fast training and inference  

---

## 📦 Installation

### From Source (Recommended for Development)

```bash
git clone https://github.com/awa-si/NHSMM.git;
cd NHSMM;
pip install -e .;
```

### From PyPI (Upcoming)

```bash
pip install nhsmm;
```

---

## 🧩 Package Structure

```
nhsmm/
├── context.py             # Contextual Encoder
├── constants.py           # Default configuration
├── models/
│   ├── base.py            # Core HSMM model & inference
│   └── __init__.py
├── distributions/
│   ├── default.py         # Initial, Duration, Transition, Emission
│   └── __init__.py
├── tools/
│   ├── constraints.py
│   ├── convergence.py
│   ├── utils.py
│   ├── seed.py
│   └── __init__.py
└── __init__.py
```

---

## 🧠 Usage Example (Context-Aware)

```python
import torch
from nhsmm.models import HSMM

# Example input sequence: 256 time steps, 32 features
X = torch.randn(256, 32)

# Optional external context: market indicators, sensor readings, embeddings, etc.
context = torch.randn(256, 16)  # 16-dimensional covariates

# Initialize a 4-state Neural HSMM
model = HSMM(
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

# Explore tests and scripts for more examples
```

---

## 🔍 Conceptual Flow Diagram

```
    ┌───────────────┐
    │ External Input│  ← covariates, features, embeddings
    └───────┬───────┘
            │
            ▼
┌─────────────────────────┐
│ Neural Initial State    │  ← context-modulated initial probabilities
│ Distribution            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Neural Transition       │  ← context-gated, temperature-scaled transitions
│ Distribution            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Neural Duration         │  ← context-aware duration probabilities per state
│ Distribution            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Emission Model          │  ← context-modulated observation likelihoods
│ (Gaussian / Multinomial)│
└─────────────────────────┘
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

## 🌐 Multi-Domain Applicability

SAE (built on NHSMM) can be applied to:

* Security / Cyber-Physical Systems — Identify hidden network or operational states  
* Finance / Trading — Market regime detection and adaptive strategy modeling
* IoT / Industrial Systems — Predict machine regime changes for maintenance  
* Health / Wearables — Detect activity and physiological state transitions  
* Robotics / Motion — Monitor robot behavior for unexpected transitions  

---

## 🧾 License

Apache 2.0 © 2025 AWA.SI
*See [LICENSE](LICENSE) for details.*
