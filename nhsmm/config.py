# nhsmm/constants.py

from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn


logger = logging.getLogger("NHSMM")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


EPS: float = 1e-12
DTYPE = torch.float32
NEG_INF: float = torch.finfo(DTYPE).min
MIN_LOGITS: float = NEG_INF
MAX_LOGITS: float = 1e5


@dataclass
class ModelConfig:
    # -----------------------------
    # Model architecture
    # -----------------------------
    n_states: int
    n_features: int
    n_heads: int = 3
    cnn_kernel: int = 3
    cnn_channels: int = 5
    dropout: float = 0.05
    max_duration: int = 35
    pad_value: float = 0.0
    min_covar: float = 1e-6
    temperature: float = 1.0
    modulate_var: bool = False
    hidden_dim: Optional[int] = None

    # -----------------------------
    # Encoder
    # -----------------------------
    context_dim: Optional[int] = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"

    # -----------------------------
    # HMM / Distribution
    # -----------------------------
    emission_init_mode: Literal["randome", "spread"] = "spread"
    initial_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    duration_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    activation: Literal["leaky_relu", "identity", "softplus", "gelu", "relu", "tanh"] = "tanh"
    emission_type: Literal["gaussian", "studentt"] = "gaussian"

    # -----------------------------
    # General
    # -----------------------------
    seed: Optional[int] = None
    debug: bool = False

    # -----------------------------
    # Optimization / Training
    # -----------------------------
    convergence_mode: Literal["delta", "plateau"] = "plateau"
    n_init: int = 3
    lr: float = 1e-2
    tol: float = 1e-4
    max_iter: int = 5
    loss_bias: float = 1e-3
    plateau_window: int = 6
    plateau_tol: float = 1e-4
    use_scheduler: bool = True
    convergence_stop: bool = True
    verbose: bool = True


