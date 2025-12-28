# nhsmm/constants.py

from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from dataclasses import dataclass
import logging
import torch

logger = logging.getLogger("NHSMM")

EPS = 1e-12
MAX_LOGITS = 1e5
DTYPE = torch.float32
NEG_INF = torch.finfo(DTYPE).min
DEBUG = False

if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

@dataclass
class HSMMConfig:
    n_states: int
    n_features: int
    max_duration: int
    n_heads: int = 4
    dropout: float = 0.0
    min_covar: float = 1e-6
    cnn_channels: float = 5
    temperature: float = 1.0
    modulate_var: bool = False
    emission_type: str = "gaussian"
    hidden_dim: Optional[int] = None
    context_dim: Optional[int] = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    seed: Optional[int] = None
    debug: bool = False
