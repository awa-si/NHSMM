from typing import Optional, Tuple, Union
import torch
from torch import nn
from nhsmm.utilities import utils


class NeuralEmission:
    """
    Neural/contextual emission distribution for HSMM states.

    Supports:
        - Gaussian (continuous) or Categorical (discrete) emissions
        - Direct parameter initialization
        - Neural encoder for context-dependent emissions
        - Posterior-weighted EM updates
    """

    EPS = 1e-8

    def __init__(
        self,
        emission_type: str,
        params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none"
    ):
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")
        if emission_type not in {"gaussian", "categorical"}:
            raise ValueError(f"Unsupported emission_type '{emission_type}'")

        self.emission_type = emission_type
        self.encoder = encoder
        self.context_mode = context_mode

        if emission_type == "gaussian":
            if not isinstance(params, tuple) or len(params) != 2:
                raise ValueError("Gaussian params must be (mu, cov)")
            self.mu, self.cov = params
            if self.mu.ndim != 2 or self.cov.ndim != 3:
                raise ValueError("mu must be [n_states, n_features], cov must be [n_states, n_features, n_features]")
            self.n_states, self.n_features = self.mu.shape
        else:  # categorical
            self.logits = params
            if self.logits.ndim not in (2, 3):
                raise ValueError("Categorical logits must be [n_states, n_categories] or [batch, n_states, n_categories]")
            self.n_states = self.logits.shape[-2]
            self.n_categories = self.logits.shape[-1]

    def contextual_params(self, theta: Optional[utils.ContextualVariables] = None):
        """Compute context-dependent parameters using neural encoder."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        # Encoder output
        if isinstance(theta.X, dict):
            out = self.encoder(**theta.X)
        elif isinstance(theta.X, tuple):
            out = self.encoder(*theta.X)
        else:
            out = self.encoder(theta.X)

        if self.emission_type == "gaussian":
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError("Gaussian encoder must return (mu, cov)")
            mu, cov = out
            if mu.shape[-1] != self.n_features:
                raise ValueError(f"Encoder mu shape {mu.shape} != n_features {self.n_features}")
            if cov.shape[-2:] != (self.n_features, self.n_features):
                raise ValueError(f"Encoder cov shape {cov.shape} != [n_features, n_features]")
            return mu, cov
        else:
            if out.shape[-1] != self.n_categories:
                raise ValueError(f"Encoder logits shape {out.shape} != n_categories {self.n_categories}")
            return torch.log_softmax(out, dim=-1).clamp_min(self.EPS)

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        """
        Posterior-weighted EM update of emission parameters.
        - Gaussian: updates mu and cov
        - Categorical: updates logits
        """
        if self.emission_type == "gaussian":
            n_states, n_features = self.mu.shape
            weighted_sum = posterior.T @ X
            counts = posterior.sum(dim=0).clamp_min(self.EPS)[:, None]
            mu_new = weighted_sum / counts

            cov_new = []
            for s in range(n_states):
                diff = X - mu_new[s]
                cov_s = (posterior[:, s][:, None] * diff).T @ diff / counts[s, 0]
                cov_new.append(cov_s + torch.eye(n_features, device=X.device) * self.EPS)
            cov_new = torch.stack(cov_new)

            if inplace:
                self.mu, self.cov = mu_new, cov_new
                return self
            return NeuralEmission("gaussian", (mu_new, cov_new), encoder=self.encoder, context_mode=self.context_mode)

        else:  # categorical
            if X.ndim == 1 or X.shape[1] != self.n_categories:
                X_onehot = torch.nn.functional.one_hot(X.long(), num_classes=self.n_categories).float()
            else:
                X_onehot = X.float()
            weighted_counts = posterior.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(self.EPS)).clamp_min(self.EPS).log()
            if inplace:
                self.logits = logits_new
                return self
            return NeuralEmission("categorical", logits_new, encoder=self.encoder, context_mode=self.context_mode)

    @classmethod
    def initialize(
        cls,
        emission_type: str,
        n_states: int,
        n_features: int = None,
        n_categories: int = None,
        alpha: float = 1.0,
        encoder: Optional[nn.Module] = None
    ):
        """Random initialization of emission parameters."""
        if emission_type == "gaussian":
            mu = torch.randn(n_states, n_features)
            cov = torch.stack([torch.eye(n_features) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder)
        else:
            logits = torch.distributions.Dirichlet(torch.ones(n_categories) * alpha).sample([n_states])
            return cls("categorical", logits.clamp_min(cls.EPS).log(), encoder=encoder)

    def __repr__(self):
        if self.emission_type == "gaussian":
            return f"NeuralEmission(Gaussian, n_states={self.n_states}, n_features={self.n_features})"
        return f"NeuralEmission(Categorical, n_states={self.n_states}, n_categories={self.n_categories})"
