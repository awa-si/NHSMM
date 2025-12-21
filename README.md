
# NHSMM — (Neural) Hidden Semi-Markov Models
- *Repository: [NHSMM on GitHub](https://github.com/awa-si/nhsmm)*
- *Documentation: [wiki](https://github.com/awa-si/nhsmm/Wiki)*
- *Version: 0.0.2-alpha*

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

**Overview**  
This document provides a **self-contained guide** to **NHSMM**, a modular PyTorch library that underpins the **State Aware Engine (SAE)**. It is designed for **developers, data scientists, and system integrators**, enabling them to quickly understand, deploy, and extend the library across diverse application domains.

**Highlights**  
- Combines **Hidden Semi-Markov Models (HSMMs)** with **neural parameterization**  
- Supports **contextual modulation** of initial, transition, duration, and emission distributions  
- Enables **hierarchical and flexible model architectures** for sequential data  
- Fully **PyTorch-compatible**, GPU-ready, and extensible for multi-domain use

> ⚠️ **NHSMM is currently in alpha** (v0.0.2-alpha) and provided as a **proof-of-concept**.  
> It **will be enhanced for production**, and the public API is **subject to change** prior to the first stable `1.0.0` release.

---

## Overview

**NHSMM** provides a **modular, time-aware framework** for **temporal sequence modeling** and **hidden-state inference** using **Hidden Semi-Markov Models (HSMMs)**. It explicitly models **state durations** and **context-dependent transitions**, making it suitable for **non-stationary, real-world sequences**.  

It forms the foundation of the **State Aware Engine (SAE)**, a **cross-domain platform** for detecting **hidden regimes** and **temporal patterns** in domains such as **IoT, health, cybersecurity, robotics, and finance**. NHSMM can also serve as a robust base for **research projects and experimental HSMM applications**.

Key deployment modes powered by SAE & NHSMM (Upcoming):  
- **Cloud-first SaaS** — for scalable, multi-user access to sequence analytics  
- **On-prem / Edge** — low-latency, privacy-sensitive inference for real-time systems  
- **Hardware/Accelerator-ready** — GPU, TPU, or quantum-ready operations for high-throughput temporal modeling  

---

## 🚀 Key Features

* **Contextual HSMM** — external covariates dynamically modulate **initial, transition, duration, and emission distributions**, enabling **non-stationary and time-aware sequence modeling**. Supports continuous or discrete context inputs.  
* **Duration Models** — explicit discrete duration distributions per state, optionally **context-modulated**, to capture **state dwell-times** accurately. Supports both short- and long-term temporal dependencies.  
* **Emission Models** — Gaussian, Student-t, or discrete outputs (e.g., Multinomial/Bernoulli), fully differentiable and **context-aware**, allowing neural adaptation to evolving observation patterns.  
* **Transition Models** — learnable, **covariate-aware transitions** with gating and **temperature scaling** to control stochasticity and enforce smooth or sharp transitions. Supports low-rank factorization for large state spaces.  
* **Hybrid HSMM-HMM Inference** — forward-backward and Viterbi algorithms adapted for **neural-parameterized latent states**, providing exact or approximate **sequence likelihoods and most probable paths**.  
* **Subclassable Distributions** — Initial, Duration, Transition, and Emission modules inherit from PyTorch `Distribution`, allowing **custom probabilistic modules** or integration of specialized neural layers.  
* **EM-style Updates & Initialization** — supports **maximum likelihood**, differentiable updates, and **temperature annealing** for stable and adaptive training of neural-HSMM parameters.  
* **Neural Context Encoders** — optional CNN, LSTM, or hybrid encoders to inject **time-varying covariates** into emission, duration, and transition probabilities.  
* **GPU-Ready Implementation** — fully batched operations for **fast training and inference** on modern accelerators (CUDA-ready).  
* **Multi-Domain Usage** — flexible for **finance (market regime detection), IoT (predictive maintenance), robotics (behavior monitoring), wearable health (activity and state detection), cybersecurity (anomaly detection)**, and other sequential applications.  
* **Extensible Architecture** — modular foundation for SAE adapters, API integration, multi-domain extensions, and **future research projects** in sequence modeling.  
* **Robust Initialization & Adaptivity** — supports **spread or random initialization**, learning rate adaptation, and state-wise parameter modulation for stable convergence across domains.  
* **Hybrid Update Modes** — allows **neural gradient-based updates** and **closed-form EM-style updates**, optionally combined in alternating schemes for best performance.  

---

## 📌 Milestones

| Stage                  | Status  | Notes                                                |
|------------------------|---------|----------------------------------------------------- |
| Proof of Concept       | ✅ Done | Alpha release (0.0.2-alpha)                          |
| Enhancement            | ⚠️ Todo | Improve performance, stability, and extend API       |
| Testing                | ⚠️ Todo | Rigorous validation, benchmarking, and QA            |
| Production Release     | ⚠️ Todo | Full 1.0.0 release with documentation and stable API |

---

## 📦 Installation

### From PyPI (now available)

```bash
pip install nhsmm;
```

### From Source (Recommended for Development)

```bash
git clone https://github.com/awa-si/NHSMM.git;
cd NHSMM;
pip install -e .;
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
┌─────────────────────────────┐
│      External Input         │  ← covariates, features, embeddings
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Neural Initial State Module│  ← context-modulated initial probabilities π
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Neural Transition Module   │  ← context-gated, temperature-scaled transitions A
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Neural Duration Module     │  ← context-aware duration probabilities D
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Emission Module        │  ← context-modulated observation likelihoods
│   (Gaussian / Student-t / …)│
└─────────────┬───────────────┘
              │
              ▼
      ┌─────────────┐
      │ Posterior   │  ← gamma, xi, eta (inference)
      └─────────────┘
              ▲
              │
   ┌──────────┴──────────┐
   │ Backprop / EM Update │
   └─────────────────────┘

```

---

## ⚙️ Development

> NHSMM is actively developed and **contributions are welcome**!
> Whether you want to report bugs, suggest features, or improve documentation, your input helps make the library stronger and more versatile.

```bash
# Fork or clone the repository
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm

# Install in development mode with optional dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Code formatting & linting
black nhsmm
ruff check nhsmm
```

---

## 🌐 Multi-Domain Applicability

The **State Aware Engine (SAE)**, powered by **NHSMM**, is designed for **flexible, context-aware temporal modeling** across diverse domains:

* **Security & Cyber-Physical Systems** — Detect hidden operational or network states, anomalies, and potential threats in real time. Supports **log analysis, sensor fusion, and adaptive monitoring**.  
* **Finance & Trading** — Identify market regimes, model **adaptive strategies**, and forecast **state transitions** for portfolios, trading signals, or risk scenarios. Captures both **short-term fluctuations and long-term trends**.  
* **IoT & Industrial Systems** — Monitor machines, sensors, and processes to predict **regime changes**, enabling **predictive maintenance**, fault detection, and operational optimization.  
* **Health & Wearables** — Track activity, physiological signals, and **state transitions** for personalized health monitoring, anomaly detection, or chronic condition management. Supports **multimodal time-series fusion**.  
* **Robotics & Motion Analytics** — Observe robotic behaviors, detect **unexpected transitions**, optimize task performance, and ensure **safe human-robot interaction**.  
* **Telecommunications & Network Analytics** — Monitor network traffic patterns, detect **latent congestion states**, and predict **temporal anomalies** for automated response.  
* **Energy & Smart Grids** — Track operational states of energy systems, detect **load or failure regimes**, and optimize resource allocation over time.  
* **Cross-Domain Research & AI Applications** — Serve as a foundation for **temporal sequence modeling**, hybrid HSMM-HMM experiments, or **research in neural probabilistic models**.  

---


## 🧾 License

This project is released under the **Apache License 2.0** © 2024 **AWA.SI**.  
For full license terms and conditions, please see the [LICENSE](https://github.com/awa-si/nhsmm/LICENSE) file.

---
