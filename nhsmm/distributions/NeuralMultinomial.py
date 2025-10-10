from typing import Optional, Union

import torch
from torch import nn
from torch.distributions import Multinomial
from nhsmm.utilities import utils, constraints


class NeuralMultinomial(Multinomial):
    """
    Neural Multinomial emission distribution for discrete HSMM observations.

    Supports:
        - Direct logit initialization
        - Data-driven initialization
        - Posterior-weighted EM updates
        - Optional neural encoder for contextual logits
    """

    EPS = 1e-8

    def __init__(
        self,
        logits: torch.Tensor,
        trials: int = 1,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ):
        super().__init__(total_count=trials, probs=None, logits=logits)
        self.encoder = encoder
        self.context_mode = context_mode
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

    @property
    def dof(self) -> int:
        batch_size = int(torch.prod(torch.tensor(self.batch_shape))) if self.batch_shape else 1
        return batch_size * (self.event_shape[0] - 1)

    @classmethod
    def sample_emission_pdf(
        cls,
        trials: int,
        n_states: int,
        n_categories: int,
        alpha: float = 1.0,
        X: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ) -> "NeuralMultinomial":
        """Sample or initialize emission logits."""
        if X is not None:
            if X.ndim > 1 and X.shape[-1] == n_categories:
                freqs = X.sum(dim=0) / X.sum()
            else:
                freqs = torch.bincount(X.long().flatten(), minlength=n_categories).float()
                freqs /= freqs.sum()
            emission_matrix = freqs.clamp_min(cls.EPS).log().unsqueeze(0).repeat(n_states, 1)
        else:
            probs = constraints.sample_probs(alpha, (n_states, n_categories))
            emission_matrix = probs.clamp_min(cls.EPS).log()

        logits = torch.log_softmax(emission_matrix, dim=-1)
        return cls(logits, trials, encoder, context_mode)

    def estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
        inplace: bool = False,
    ) -> "NeuralMultinomial":
        """Estimate emission probabilities with optional neural context."""
        if self.encoder is not None and theta is not None:
            logits = self._contextual_logits(X, theta)
        else:
            logits = self._compute_B(X, posterior).clamp_min(self.EPS).log()

        logits = torch.log_softmax(logits, dim=-1)
        if inplace:
            self.logits = logits
            return self
        return NeuralMultinomial(logits, self.total_count, self.encoder, self.context_mode)

    def _contextual_logits(self, X: torch.Tensor, theta: utils.ContextualVariables) -> torch.Tensor:
        if not callable(self.encoder):
            raise ValueError("Encoder must be a callable nn.Module.")
        if isinstance(theta.X, dict):
            encoded = self.encoder(**theta.X)
        elif isinstance(theta.X, tuple):
            encoded = self.encoder(*theta.X)
        else:
            encoded = self.encoder(theta.X)
        if encoded.shape[-1] != self.event_shape[0]:
            raise ValueError(f"Encoder output dim {encoded.shape[-1]} != event_shape {self.event_shape[0]}")
        return torch.log_softmax(encoded, dim=-1)

    def _compute_B(self, X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        n_states, _ = posterior.shape
        n_categories = self.event_shape[0]
        if X.ndim == 1 or X.shape[-1] != n_categories:
            X_onehot = torch.nn.functional.one_hot(X.long(), num_classes=n_categories).float()
        else:
            X_onehot = X.float()
        weighted_counts = posterior @ X_onehot
        weighted_counts /= weighted_counts.sum(dim=1, keepdim=True).clamp_min(self.EPS)
        return weighted_counts

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log p(X | state)."""
        if X.ndim == 1 or X.shape[-1] != self.event_shape[0]:
            X = torch.nn.functional.one_hot(X.long(), num_classes=self.event_shape[0]).float()
        log_probs = torch.log_softmax(self.logits, dim=-1)
        return (X @ log_probs.T).T

    def to(self, device: Union[str, torch.device]) -> "NeuralMultinomial":
        self.logits = self.logits.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        return self
