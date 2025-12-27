
# NHSMM — Neural Hidden Semi-Markov Models

- *Repository: [NHSMM on GitHub](https://github.com/awa-si/NHSMM)*
- *Documentation: [Wiki on GitHub](https://github.com/awa-si/NHSMM/wiki)*
- *Article on Medium: [Unlocking Hidden Patterns in Time – Meet NHSMM ](https://medium.com/@awa-si/unlocking-hidden-patterns-in-time-meet-nhsmm-the-neural-hidden-semi-markov-model-cd3f1e2428c2)*
- *Current Version (PyPI): 0.0.3-alpha*

---

> ⚠️ **NHSMM is currently in alpha** and provided as a **proof-of-concept**.  
> The library **is actively evolving** for production readiness, and the public API **may change** before the first stable `1.0.0` release.

This guide provides a **self-contained overview** of **NHSMM**, a **modular PyTorch library** forming the foundation of the **State Aware Engine (SAE)**. It is tailored for **developers, data scientists, and system integrators**, enabling rapid understanding, deployment, and extension across **diverse sequential data domains**.

### 🌟 Highlights

- Integrates **Hidden Semi-Markov Models (HSMMs)** with **neural parameterization** for expressive latent dynamics.  
- Supports **context-aware modulation** of **initial, transition, duration, and emission distributions**, capturing **non-stationary temporal patterns**.  
- Enables **flexible, hierarchical model architectures** for complex sequential data.  
- Built on **PyTorch**, fully GPU-ready, and designed for **scalable multi-domain applications**.  
- Provides a **modular foundation** for experimentation, research, and production deployment in domains such as **finance, IoT, robotics, health, and cybersecurity**.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## 🧩 Overview

**NHSMM** is a **modular, context-aware framework** for **temporal sequence modeling** and **latent state inference** using **Hidden Semi-Markov Models (HSMMs)**.  

It explicitly captures:

* **Context-Dependent State Durations** — each hidden state can persist for variable durations influenced by external covariates, enabling accurate modeling of **state dwell-times**.  
* **Context-Dependent Transitions** — transition probabilities between states adapt dynamically to **time-varying covariates**, allowing **non-stationary and evolving sequences** to be modeled faithfully.  

This design makes NHSMM particularly suitable for **real-world, time-sensitive applications**, where sequences are **non-stationary**, **heterogeneous**, and **fully time-aware**.

---

## 🌐 NHSMM & the State Aware Engine (SAE)

**NHSMM** powers the **State Aware Engine (SAE)** — a **cross-domain platform** for uncovering **hidden regimes** and **temporal patterns** in sequential data across **finance, IoT, health, cybersecurity, robotics**, and related applications.  

The framework is designed for dual usage:

* **Research-Ready** — flexible, modular, and extensible for experimentation with **context-aware HSMMs** and novel sequence modeling approaches.  
* **Production-Ready** — optimized for robust, high-throughput deployment in real-world systems, emphasizing **scalability**, **modularity**, and **accelerator-friendly execution**.  

### 🚀 Key Deployment Modes (Beta)

- **Cloud-First SaaS** — scalable, multi-tenant sequence analytics for enterprise applications.  
- **On-Prem / Edge** — low-latency, privacy-preserving inference on local or edge devices.  
- **Accelerator-Ready** — GPU, TPU, and future hardware backends for **high-throughput temporal modeling**.

---

## 🚀 Key Features

* **Contextual HSMM** — external covariates dynamically modulate **initial, transition, duration, and emission distributions**, enabling **non-stationary and time-aware sequence modeling**. Supports continuous or discrete context inputs.  
* **Duration Models** — explicit discrete duration distributions per state, optionally **context-modulated**, to capture **state dwell-times** accurately. Supports both short- and long-term temporal dependencies.  
* **Emission Models** — Gaussian, Student-t, or discrete outputs (e.g., Multinomial/Bernoulli), fully differentiable and **context-aware**, allowing neural adaptation to evolving observation patterns.  
* **Transition Models** — learnable, **covariate-aware transitions** with gating and **temperature scaling** to control stochasticity and enforce smooth or sharp transitions. Supports low-rank factorization for large state spaces.  
* **Hybrid HSMM-HMM Inference** — forward-backward and Viterbi algorithms adapted for **neural-parameterized latent states**, providing exact or approximate **sequence likelihoods and most probable paths**.  
* **Subclassable Distributions** — Initial, Duration, Transition, and Emission modules inherit from PyTorch `Distribution`, allowing **custom probabilistic modules** or integration of specialized neural layers.  
* **Differentiable Training** — supports **gradient-based optimization**, **temperature annealing**, and **neural modulation** for stable training of HSMM parameters.  
* **Neural Context Encoders** — optional CNN, LSTM, or hybrid encoders to inject **time-varying covariates** into emission, duration, and transition probabilities.  
* **GPU-Ready Implementation** — fully batched operations for **fast training and inference** on modern accelerators (CUDA-ready).  
* **Multi-Domain Usage** — flexible for **finance (market regime detection), IoT (predictive maintenance), robotics (behavior monitoring), wearable health (activity and state detection), cybersecurity (anomaly detection)**, and other sequential applications.  
* **Extensible Architecture** — modular foundation for SAE adapters, API integration, multi-domain extensions, and **future research projects** in sequence modeling.  
* **Robust Initialization & Adaptivity** — supports **spread or random initialization**, learning rate adaptation, and state-wise parameter modulation for stable convergence across domains.  
* **Hybrid Update Modes** — allows **neural gradient-based updates**, optionally combined with alternative schemes for best performance.  

---

## ⚡ Performance & Scalability

* **Vectorized Forward-Backward** — fully batched operations across sequences, states, and durations for **efficient likelihood computation**.  
* **Low-Rank Transitions** — optional low-rank factorization reduces memory and compute costs for **large state spaces**.  
* **Long-Sequence Support** — cumulative emission sums and duration masks enable **efficient handling of long sequences** without redundant computation.  
* **Memory-Efficient Viterbi** — dynamic programming optimized for **GPU execution**, supporting batched decoding with minimal overhead.  
* **Flexible Batch Sizes** — seamlessly handles variable-length sequences with **padding and masking**, ensuring **stable performance across heterogeneous datasets**.  

---

## 📌 Milestones

| Stage                  | Status  | Notes                                                |
|------------------------|---------|------------------------------------------------------|
| Proof of Concept       | ✅ Done | Alpha release (0.0.1-alpha)                          |
| Tresting/Enhancement   | ⚠️ Todo | Improve performance, stability, and extend API       |
| Production Release     | ⚠️ Todo | Full 1.0.0 release with documentation and stable API |

---

## 📦 Installation

NHSMM can be installed quickly via **PyPI** for standard usage or built from source for **development and customization**.

### 🔹 Install from PyPI

The easiest way to install NHSMM is through PyPI:

```bash
pip install nhsmm
```

This provides the latest stable release with all core dependencies, suitable for research or production environments.

### 🔹 Install from Source (Recommended for Development)

To contribute, customize, or use the latest development version:

```bash
git clone https://github.com/awa-si/NHSMM.git
cd NHSMM
pip install -e .
```

This installs NHSMM in editable mode, allowing you to modify the source code and immediately test changes without reinstalling. Ideal for experimenting with new models, encoders, or distributions.

---

## 🧩 Package Structure

```
nhsmm/
├── context.py
├── constants.py
├── convergence.py
├── encoder.py
├── data.py
├── models/
│   ├── base.py
│   └── __init__.py
├── distributions/
│   ├── default.py
│   └── __init__.py
├── __version__.py
└── __init__.py
```

---

## 🧠 Usage Example — Market Regime Detection (HSMM)

Please see:
[State Occupancy & Duration/Transition Diagnostics](docs/test.md)

This example demonstrates **Hidden Semi-Markov regime detection** on OHLCV-style time-series data using **NHSMM**.  
The same pattern applies to **IoT signals, health data, robotics telemetry, or cybersecurity logs**.

---

## 🔍 Conceptual Flow Diagram

```
┌─────────────────────────────┐
│       External Input        │  ← covariates, features, embeddings
└─────────────┬───────────────┘
              │
              ▼
┌─────────────┴───────────────┐
│  Neural Initial State Module│  ← context-modulated initial probabilities π
└─────────────┬───────────────┘
              │
              ▼
┌─────────────┴───────────────┐
│   Neural Transition Module  │  ← context-gated, temperature-scaled transitions A
└────────────┬─┬──────────────┘
             │ ▲
             ▼ │
┌────────────┴─┴──────────────┐
│   Neural Duration Module    │  ← context-aware duration probabilities D
└─────────────┬───────────────┘
              │
              ▼
┌─────────────┴───────────────┐
│       Emission Module       │  ← context-modulated observation likelihoods
│  (Gaussian / Student-t / …) │
└─────────────┬───────────────┘
              ▲
              │
┌─────────────┴───────────────┐
│          Backprop           │
└─────────────────────────────┘
```

---

## ⚙️ Development

> NHSMM is actively developed and **contributions are welcome**!
> Whether you want to report bugs, suggest features, or improve documentation, your input helps make the library stronger and more versatile.

For planned features and research directions, see the project roadmap in the GitHub issues and discussions.

```bash
# Fork or clone the repository
git clone https://github.com/awa-si/NHSMM.git
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

## Support This Project

Development and research around **NHSMM** are supported via
**GitHub Sponsors**, **Patreon**, **Medium**.

See [FUNDING.md](./FUNDING.md) for details on how to contribute and what support enables.

---

## 🧾 License

This project is released under the **Apache License 2.0** © 2024 **AWA.SI**.  
For full license terms and conditions, please see the [LICENSE](https://github.com/awa-si/NHSMM/blob/develop/LICENSE) file.
If you use NHSMM in academic work, please cite the repository.

---
