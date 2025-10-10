from typing import Optional
import torch
from torch import nn
from torch.distributions import Categorical
from nhsmm.utilities import utils

class NeuralTransition(Categorical):
    """
    Neural/contextual transition distribution for HSMM states.

    Supports:
        - Direct logits initialization (n_states x n_states or batch x n_states x n_states)
        - Neural encoder for context-dependent transitions
        - Posterior-weighted EM updates
        - Multi-batch support
    """

    EPS = 1e-8

    def __init__(
        self,
        logits: torch.Tensor,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none"
    ):
        """
        logits: Tensor of shape [n_states, n_states] or [batch, n_states, n_states]
        encoder: optional nn.Module, must return [batch?, n_states, n_states] logits
        context_mode: "none" | "temporal" | "spatial"
        """
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

        if logits.ndim not in (2, 3):
            raise ValueError("logits must be [n_states, n_states] or [batch, n_states, n_states]")

        super().__init__(logits=logits)
        self.encoder = encoder
        self.context_mode = context_mode

    @property
    def n_states(self) -> int:
        return self.logits.shape[-1]

    def contextual_logits(self, theta: Optional[utils.ContextualVariables] = None) -> torch.Tensor:
        """Compute context-dependent logits using neural encoder."""
        if self.encoder is None or theta is None:
            return self.logits

        # Encoder flexible input
        if isinstance(theta.X, dict):
            out = self.encoder(**theta.X)
        elif isinstance(theta.X, tuple):
            out = self.encoder(*theta.X)
        else:
            out = self.encoder(theta.X)

        # Ensure last dimension = n_states
        if out.shape[-1] != self.n_states:
            raise ValueError(f"Encoder output dim {out.shape[-1]} != n_states {self.n_states}")

        # Log-softmax for probabilities
        return torch.log_softmax(out, dim=-1).clamp(min=self.EPS)

    def update(self, posterior: torch.Tensor, inplace: bool = True) -> "NeuralTransition":
        """
        Posterior-weighted EM update of transition logits.

        posterior: [n_states, n_states] or [batch, n_states, n_states] expected counts
        """
        if posterior.ndim not in (2, 3):
            raise ValueError("posterior must be [n_states, n_states] or [batch, n_states, n_states]")

        if posterior.shape[-2:] != (self.n_states, self.n_states):
            raise ValueError(f"Expected posterior shape ending with {(self.n_states, self.n_states)}, got {posterior.shape}")

        logits = (posterior + self.EPS).log()
        if inplace:
            self.logits = logits
            return self
        else:
            return NeuralTransition(logits=logits, encoder=self.encoder, context_mode=self.context_mode)

    @classmethod
    def initialize(cls, n_states: int, alpha: float = 1.0, batch: int = 1, encoder: Optional[nn.Module] = None) -> "NeuralTransition":
        """
        Initialize transition matrix with Dirichlet prior.
        Returns shape [batch, n_states, n_states] if batch>1, else [n_states, n_states].
        """
        probs = torch.distributions.Dirichlet(torch.ones(n_states) * alpha).sample([batch])
        if batch == 1:
            probs = probs.squeeze(0)
        return cls(logits=probs.clamp_min(cls.EPS).log(), encoder=encoder)
