
# NHSMM — (Neural) Hidden Semi-Markov Models
- *Documentation: [wiki](Wiki)*
- *Version: 0.0.2-alpha*

> ⚠️ NHSMM is currently released as an **alpha version (0.0.2-alpha)**.
> The public API may change prior to the first stable `1.0.0` release.


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
├── constraints.py
├── convergence.py
├── data.py
├── seed.py
├── models/
│   ├── base.py            # Core HSMM model & inference
│   └── __init__.py
├── distributions/
│   ├── default.py         # Initial, Duration, Transition, Emission
│   └── __init__.py
├── __vesion__.py
└── __init__.py
```
---

## 🧠 Usage Example — Market Regime Detection (HSMM)

This example demonstrates **Hidden Semi-Markov regime detection** on OHLC-style time-series data using **NHSMM**.  
The same pattern applies to **IoT signals, health data, robotics telemetry, or cybersecurity logs**.

---

### 1. Prepare Data

```python
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from nhsmm.models import HSMM
from nhsmm.context import CNN_LSTM_Encoder
from nhsmm.constants import DTYPE

# Synthetic example: [T, F] = time × features
T, F = 512, 4
X = np.random.randn(T, F)

# Scale features
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=DTYPE)
```

### 2. Build Context Encoder (Optional but Recommended)

Context enables non-stationary transitions and durations.

```python
encoder = CNN_LSTM_Encoder(
    n_features=F,
    cnn_channels=4,
    hidden_dim=64,
)
```

### 3. Initialize Neural HSMM

```python
model = HSMM(
    encoder=encoder,
    n_states=3,              # e.g. range / bull / bear
    n_features=F,
    emission_type="gaussian",
    max_duration=30,
    seed=0,
)
```

### 4. Train with EM-Style Optimization

```python
model.fit(
    X,
    n_init=3,
    max_iter=10,
    tol=1e-4,
    verbose=True,
)
```

### 5. Decode Hidden States (Viterbi) / Inspect

```python
states = model.decode(X, algorithm="viterbi")

print("Decoded states shape:", states.shape)
print("Unique states:", torch.unique(states))

# Inspect Learned Durations
with torch.no_grad():
    durations = torch.exp(model.duration_module.log_matrix())
    durations = durations.mean(dim=(0, 1))  # [K, D]

for i, row in enumerate(durations):
    mean_dur = (torch.arange(1, len(row) + 1) * row).sum()
    print(f"State {i}: mean duration ≈ {mean_dur:.2f}")

# Log-Likelihood Scoring
log_likelihood = model.score(X)
print("Sequence log-likelihood:", log_likelihood.item())

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
