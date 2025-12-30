# nhsmm/constants.py

from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

# --------------------------
# Logging configuration
# --------------------------
logger = logging.getLogger("NHSMM")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --------------------------
# Constants
# --------------------------
EPS: float = 1e-12
MAX_LOGITS: float = 1e5
DTYPE = torch.float32
NEG_INF: float = torch.finfo(DTYPE).min


@dataclass
class HSMMConfig:
    n_states: int
    n_features: int
    n_heads: int = 4
    dropout: float = 0.0
    max_duration: int = 35
    min_covar: float = 1e-6
    cnn_channels: int = 5
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
    """
    Convenience container for all HSMM distributions.
    Provides a unified initialization interface.
    """

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
        self.transition = transition
        self.emission = emission

    def initialize(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5,
        **dist_kwargs
    ) -> Dict[str, Any]:
        """
        Initialize all component distributions with optional context and jitter.
        Returns a dictionary of distribution parameters.
        """
        return {
            "initial_dist": self.initial.initialize(
                context=context, temperature=temperature, jitter=jitter, **dist_kwargs
            ),
            "duration_dist": self.duration.initialize(
                context=context, temperature=temperature, jitter=jitter, **dist_kwargs
            ),
            "transition_dist": self.transition.initialize(
                context=context, temperature=temperature, jitter=jitter, **dist_kwargs
            ),
            "emission_dist": self.emission.initialize(
                context=context, temperature=temperature, jitter=jitter, **dist_kwargs
            ),
        }
