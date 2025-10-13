
# NHSMM — Neural Hidden Semi-Markov Models

**NHSMM** is a modular PyTorch-based library for hierarchical regime detection and temporal modeling using **Neural Hidden Semi-Markov Models (HSMMs)**.

It extends classic HSMMs with learnable emission, duration, and transition components — enabling adaptive, context-aware sequence modeling for financial regimes, sensor data, and other temporal domains.

---

## 🚀 Key Features

- **Neural Gaussian & Multinomial Emissions** — differentiable probabilistic outputs
- **Covariate-Dependent Duration Models** — via neural parameterization
- **Dense HSMM / HSMM-HMM Hybrid Inference**
- **Multi-Timescale Regime Modeling** — supports 1W / 1H / 15M / 1M fusion
- **Torch-based, GPU-ready Implementation**
- **Configurable self-loop, trend, and overlay regime maps**

---

## 📦 Installation

### From source (recommended for development)

```bash
git clone https://github.com/awwea/NeuralHSMM.git
cd nhsmm
pip install -e .
```

### From PyPI (when published)

```bash
pip install nhsmm
```

---

## 🧩 Package Structure

```
nhsmm/
├── models/
│   ├── base.py
│   ├── neural.py
├── ditributions/
│   ├── NeuralDuration.py
│   ├── NeuralEmission.py
│   ├── NeuralGaussian.py
│   ├── NeuralMultinomial.py
│   ├── NeuralTransition.py
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
from nhsmm.core import NeuralHSMM, NeuralGaussian, NeuralDuration

# Example input sequence
X = torch.randn(256, 32)  # (time, features)

# Initialize HSMM
model = NeuralHSMM(
    n_states=4,
    emission=NeuralGaussian(input_dim=32, hidden_dim=64),
    duration=NeuralDuration(input_dim=32, hidden_dim=32),
)

# Forward pass
log_prob = model.log_prob(X)
states = model.viterbi(X)

print("Log-likelihood:", log_prob.item())
print("Most likely state sequence:", states.shape)
```

---

## 📊 Regime Detection Pipeline

`RegimeDetector` orchestrates hierarchical modeling:

1. **Macro HSMM** – Coarse trends (`bull`, `bear`, `range`, `uncertain`)
2. **Micro HSMM** – Fine-grained overlays (`pump`, `dump`, `accumulation`, etc.)
3. **Duration conditioning** – Covariate-based persistence modeling
4. **Online reset hooks** – Bayesian Online Change Point Detection (BOCPD-style)

---

## ⚙️ Development

```bash
pip install -e ".[dev]"
pytest -v
black nhsmm
ruff check nhsmm
```

---

## 🧾 License

MIT © 2025 AWA
