# nhsmm/constants.py

from typing import Optional, List, Tuple, Any, Literal, Dict, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

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
    n_heads: int = 4
    dropout: float = 0.0
    max_duration: int = 35
    min_covar: float = 1e-6
    cnn_channels: float = 5
    temperature: float = 1.0
    modulate_var: bool = False
    hidden_dim: Optional[int] = None
    context_dim: Optional[int] = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    init_mode: Literal["normal", "biased", "dirichlet", "uniform"] = "normal"
    emission_type: Literal["gaussian", "studentt"] = "gaussian"
    seed: Optional[int] = None
    debug: bool = False

class DefaultDistribution(nn.Module):

    def __init__(
        self,
        initial: Optional[nn.Module] = None,
        duration: Optional[nn.Module] = None,
        transition: Optional[nn.Module] = None,
        emission: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.initial = initial
        self.duration = duration
        self.emission = emission
        self.transition = transition

    def initialize(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5, **dist_kwargs) -> Dict[str, Any]:
        return {
            "initial_dist": self.initial.initialize(context=context, temperature=temperature, jitter=jitter, **dist_kwargs),
            "duration_dist": self.duration.initialize(context=context, temperature=temperature, jitter=jitter, **dist_kwargs),
            "transition_dist": self.transition.initialize(context=context, temperature=temperature, jitter=jitter, **dist_kwargs),
            "emission_dist": self.emission.initialize(context=context, temperature=temperature, jitter=jitter, **dist_kwargs),
        }

