# NHSMM — Neural Hidden Semi-Markov Models

* **Repository:** [NHSMM on GitHub](https://github.com/awa-si/NHSMM)
* **Interfaces:** [NHSMM Interfaces on GitHub](https://github.com/awa-si/nhsmm-interfaces)
* **Documentation:** [NHSMM Wiki](https://github.com/awa-si/NHSMM/wiki)
* **Article:** [Unlocking Hidden Patterns in Time – Meet NHSMM](https://medium.com/@awa-si/unlocking-hidden-patterns-in-time-meet-nhsmm-the-neural-hidden-semi-markov-model-cd3f1e2428c2)

---

> ⚠️ **Alpha stage** — NHSMM is a **proof-of-concept** and actively evolving. Public APIs may change before stable `1.0.0`.

**NHSMM** is a **modular PyTorch library** for **context-aware sequential modeling**, forming the foundation of the **State Aware Engine ([SAE](https://github.com/awa-si/SAE))**.
**`nhsmm-interfaces`** defines domain-level contracts for integrating NHSMM in diverse systems.

Designed for **developers, data scientists, and system integrators**, NHSMM enables rapid understanding, deployment, and extension of **latent state models** for domains such as **finance, IoT, robotics, health, and cybersecurity**.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## 🌟 Highlights

* **Neural HSMM** — integrates Hidden Semi-Markov Models with neural parameterization for expressive latent dynamics.
* **Context-Aware Modulation** — initial, transition, duration, and emission distributions adapt to external covariates.
* **Flexible Architectures** — supports hierarchical and hybrid models.
* **PyTorch & GPU Ready** — scalable multi-domain deployment.
* **Modular Foundation** — for research, experimentation, and production-ready sequence models.

---

## 🧩 Overview

NHSMM explicitly models:

* **Context-Dependent State Durations** — variable dwell-times per hidden state influenced by covariates.
* **Context-Dependent Transitions** — dynamic transition probabilities adapting to time-varying features.

Suitable for **non-stationary, heterogeneous, and time-aware sequences** across real-world applications.

---

## NHSMM Project Covenant

1. **Forever Open Core** — `nhsmm` remains fully open-source and actively maintained.
2. **No Hidden Dependencies** — core library uses only open components; experimental modules (`nhsmm-interfaces`) are optional.
3. **Transparent Evolution** — research previews and pre-release interfaces are clearly marked.
4. **Community Respect** — contributions are acknowledged; experimental previews may close but knowledge remains accessible.
5. **Clear Upgrade Path** — experimental work informs SAE; core NHSMM is stable and independent.

---

## 🚀 Key Features

* **Contextual HSMM** — dynamic modulation of initial, transition, duration, and emission probabilities.
* **Duration Models** — explicit, context-aware state dwell-times.
* **Emission Models** — Gaussian, Student-t, or discrete outputs; differentiable and context-aware.
* **Transition Models** — learnable, covariate-aware with gating and temperature scaling; supports low-rank factorization.
* **Hybrid HSMM-HMM Inference** — forward-backward and Viterbi adapted for neural latent states.
* **Subclassable Distributions** — extend Initial, Duration, Transition, Emission modules.
* **Differentiable Training** — gradient-based optimization, temperature annealing, neural modulation.
* **Neural Context Encoders** — CNN, LSTM, or hybrid encoders for time-varying covariates.
* **GPU-Ready** — fully batched operations.
* **Multi-Domain Applicability** — finance, IoT, robotics, health, cybersecurity.
* **Extensible Architecture** — foundation for SAE interfaces, API integration, and research projects.
* **Hybrid Update Modes** — neural gradient-based updates, optional alternative schemes.

---

## ⚡ Performance & Scalability

* Vectorized forward-backward for **batched likelihood computation**.
* Optional low-rank transitions for **large state spaces**.
* Supports **long sequences** efficiently.
* Memory-efficient Viterbi optimized for GPU.
* Handles **variable-length sequences** with padding and masking.

---

## 📌 Milestones

| Stage               | Status  | Notes                                   |
| ------------------- | ------- | --------------------------------------- |
| Proof of Concept    | ✅ Done  | Alpha release (0.0.1-alpha)            |
| Testing/Enhancement | ⚠️ Todo | Improve performance, extend API         |
| Production Release  | ⚠️ Todo | Stable 1.0.0 release with documentation |

---

## 📦 Installation

### 🔹 From PyPI (alpha stage)

```bash
pip install nhsmm
```

### 🔹 From Source (recommended)

```bash
git clone https://github.com/awa-si/NHSMM.git
cd NHSMM
pip install -e .
```

Editable mode allows modification and testing without reinstalling.

---

## 🧠 Usage Example — Market Regime Detection

See: [State Occupancy & Duration/Transition Diagnostics](docs/test_ohlcv.md)

Works similarly for **IoT signals, health telemetry, robotics, or cybersecurity logs**.

```python
from nhsmm.models import HSMM
model = HSMM(n_states=3, n_features=5)
seq_features, canonical = model.encoder.encode(sequences)
```

---

## 🔍 Conceptual Flow

```text
External Input → Neural Initial Module (π)
                → Neural Transition Module (A)
                → Neural Duration Module (D)
                → Emission Module (Gaussian/Student-t/Discrete)
                → Forward-Backward / Viterbi → Backprop
```

* **External Input:** features, covariates, embeddings.
* **Neural Modules:** context-conditioned initial, transition, duration, and emission distributions.
* **Inference:** latent states inferred via forward-backward and Viterbi.
* **Backpropagation:** updates all neural modules jointly.

---

## 🌐 Multi-Domain Applicability

* **Security & Cyber-Physical Systems:** anomaly and hidden state detection.
* **Finance & Trading:** regime detection, forecasting, adaptive strategies.
* **IoT & Industrial:** predictive maintenance, fault detection.
* **Health & Wearables:** activity and state tracking, multimodal fusion.
* **Robotics:** behavior monitoring, safe human-robot interaction.
* **Telecommunications & Energy:** latent state monitoring, resource optimization.
* **Research & AI:** temporal modeling, neural-probabilistic experiments.

---

## ⚙️ Development

> Contributions welcome! Bug reports, feature suggestions, or documentation improvements strengthen NHSMM.

```bash
git clone https://github.com/awa-si/NHSMM.git
cd NHSMM
pip install -e ".[dev]"
pytest -v
black nhsmm
ruff check nhsmm
```

---

## Support

Development is supported via **GitHub Sponsors, Patreon, Medium**.
See [FUNDING.md](./FUNDING.md) for details.

---

## 🧾 License

Apache License 2.0 © 2024 **AWA.SI**
Full terms: [LICENSE](https://github.com/awa-si/NHSMM/blob/develop/LICENSE)

If used in academic work, please cite the repository.
