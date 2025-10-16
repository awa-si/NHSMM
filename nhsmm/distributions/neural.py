import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Categorical, Poisson, LogNormal
from typing import Optional, Union, Tuple

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.utilities import utils


class NeuralEmission(Emission):
    """Neural/contextual emission distribution for HSMM states."""

    def __init__(
        self,
        emission_type: str,
        params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.context_mode = context_mode.lower()
        if self.context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

        emission_type = emission_type.lower()
        self.emission_type = emission_type

        if emission_type == "gaussian":
            mu, cov = params
            n_states, n_features = mu.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type="gaussian")
            self.mu = nn.Parameter(mu.to(self.device, self.dtype))
            self.cov = nn.Parameter(cov.to(self.device, self.dtype))
        else:
            logits = params
            n_states, n_features = logits.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type=emission_type)
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))

        self.encoder = encoder

    def contextual_params(self, theta: Optional[utils.ContextualVariables] = None):
        """Return context-modulated emission parameters."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        X = getattr(theta, "X", theta)
        out = self.encoder(*X) if isinstance(X, tuple) else self.encoder(**X) if isinstance(X, dict) else self.encoder(X)

        if self.emission_type == "gaussian":
            mu, cov = out
            return mu.to(self.device, self.dtype), cov.to(self.device, self.dtype)
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, self.dtype)
        X = X.to(self.device, self.dtype)

        if self.emission_type == "gaussian":
            counts = posterior.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (posterior.T @ X) / counts
            cov_new = torch.stack([
                ((X - mu_new[s]).T * posterior[:, s]) @ (X - mu_new[s]) / counts[s, 0]
                + torch.eye(self.n_features, device=self.device, dtype=self.dtype) * EPS
                for s in range(self.n_states)
            ])
            if inplace:
                self.mu.data.copy_(mu_new)
                self.cov.data.copy_(cov_new)
                return self
            return NeuralEmission("gaussian", (mu_new, cov_new), encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device, dtype=self.dtype)
        else:
            X_onehot = X if X.ndim == 2 else F.one_hot(X.long(), num_classes=self.n_features).float()
            weighted_counts = posterior.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device, dtype=self.dtype)

    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None, n_categories: int = None,
                   alpha: float = 1.0, encoder: Optional[nn.Module] = None, device: Optional[torch.device] = None,
                   dtype: torch.dtype = DTYPE):
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=dtype) * 0.1
            cov = torch.stack([torch.eye(n_features, device=device, dtype=dtype) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder, device=device, dtype=dtype)
        else:
            logits = torch.distributions.Dirichlet(
                torch.ones(n_categories, device=device, dtype=dtype) * alpha
            ).sample([n_states])
            return cls(emission_type.lower(), logits.clamp_min(EPS).log(), encoder=encoder,
                       device=device, dtype=dtype)

    def __repr__(self):
        if self.emission_type == "gaussian":
            return f"NeuralEmission(Gaussian, n_states={self.n_states}, n_features={self.n_features})"
        return f"NeuralEmission({self.emission_type.capitalize()}, n_states={self.n_states}, n_categories={self.n_features})"


class NeuralDuration(Duration):
    """Neural/contextual duration distribution for HSMM states."""

    def __init__(self, n_states: int, mode: str = "poisson",
                 rate: Optional[torch.Tensor] = None, mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None, encoder: Optional[nn.Module] = None,
                 context_mode: str = "none", max_duration: int = 20,
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1
        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))

    def _contextual_params(self, context=None):
        X = getattr(context, "X", context)
        encoded = self.encoder(X) if self.encoder else None
        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        mean, log_std = torch.chunk(encoded, 2, dim=-1) if encoded is not None else (self.mean, self.std.log())
        return mean.squeeze(-1), log_std.exp().clamp_min(EPS).squeeze(-1)

    def forward(self, context=None, log=False, return_dist=False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=self.dtype)
        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1))**2 - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)
        probs = F.softmax(logits, dim=-1)
        if return_dist:
            return [Poisson(rate[k]) if self.mode=="poisson" else LogNormal(mean[k], std[k]) for k in range(self.n_states)]
        return torch.log(probs) if log else probs

    def sample(self, n_samples=1, context=None, state_indices=None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    def to(self, device):
        super().to(device)
        if self.encoder:
            self.encoder.to(device)
        return self


class NeuralTransition(Transition):
    """Neural/contextual transition distribution for HSMM states."""

    def __init__(self, n_states: int, logits: Optional[torch.Tensor] = None,
                 encoder: Optional[nn.Module] = None, context_mode: str = "none",
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        super().__init__(n_states=n_states, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))

    def contextual_logits(self, theta=None):
        if self.encoder is None or theta is None:
            return self.logits
        X = getattr(theta, "X", theta)
        out = self.encoder(X)
        return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior, inplace=True):
        logits = torch.log((posterior + EPS).to(self.device, self.dtype))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(n_states=self.n_states, logits=logits, encoder=self.encoder,
                                context_mode=self.context_mode, device=self.device, dtype=self.dtype)

    @classmethod
    def initialize(cls, n_states, alpha=1.0, batch=1, encoder=None, device=None, dtype=DTYPE):
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        probs = torch.distributions.Dirichlet(torch.ones(n_states, device=device, dtype=dtype)*alpha).sample([batch])
        if batch == 1:
            probs = probs.squeeze(0)
        return cls(n_states=n_states, logits=torch.log(probs.clamp_min(EPS)), encoder=encoder, device=device, dtype=dtype)
