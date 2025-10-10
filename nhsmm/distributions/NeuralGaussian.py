from typing import Optional, Union
import torch
from torch import nn
from torch.distributions import Normal
from nhsmm.utilities import utils, constraints


class NeuralGaussian(Normal):
    """
    Neural Gaussian emission distribution for continuous HSMM observations.

    Supports:
        - Random initialization via Normal-Wishart prior
        - EM-style re-estimation from posterior weights
        - Optional neural encoder for contextual mean/variance
    """

    EPS = 1e-6

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ):
        super().__init__(loc=loc, scale=scale.clamp_min(self.EPS))
        self.encoder = encoder
        self.context_mode = context_mode
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

    @property
    def dof(self) -> int:
        return self.loc.numel() + self.scale.numel()

    @classmethod
    def sample_emission_pdf(
        cls,
        n_states: int,
        n_features: int,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> "NeuralGaussian":
        """Random Gaussian parameters for each state."""
        loc = mu_scale * torch.randn(n_states, n_features)
        scale = sigma_scale * torch.ones(n_states, n_features)
        return cls(loc, scale, encoder, context_mode)

    def estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
        inplace: bool = False,
    ) -> "NeuralGaussian":
        """Posterior-weighted mean and variance update."""
        if self.encoder is not None and theta is not None:
            loc, scale = self._contextual_params(X, theta)
        else:
            weighted_sum = posterior @ X
            norm = posterior.sum(dim=1, keepdim=True).clamp_min(self.EPS)
            loc = weighted_sum / norm
            var = (posterior @ ((X.unsqueeze(0) - loc.unsqueeze(1)) ** 2)) / norm
            scale = var.sqrt().clamp_min(self.EPS)

        if inplace:
            self.loc, self.scale = loc, scale
            return self
        return NeuralGaussian(loc, scale, self.encoder, self.context_mode)

    def _contextual_params(self, X: torch.Tensor, theta: utils.ContextualVariables):
        """Compute contextual mean and scale via neural encoder."""
        if not callable(self.encoder):
            raise ValueError("Encoder must be a callable nn.Module.")

        encoded = (
            self.encoder(**theta.X) if isinstance(theta.X, dict)
            else self.encoder(*theta.X) if isinstance(theta.X, tuple)
            else self.encoder(theta.X)
        )

        if encoded.shape[-1] != 2 * self.loc.shape[-1]:
            raise ValueError("Encoder must output 2 × feature_dim (mean and log_var).")

        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        scale = log_var.exp().sqrt().clamp_min(self.EPS)
        return mu, scale

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Return log p(X | state)."""
        return super().log_prob(X.unsqueeze(0)).sum(-1)  # (n_states, T)

    def to(self, device: Union[str, torch.device]) -> "NeuralGaussian":
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        return self
