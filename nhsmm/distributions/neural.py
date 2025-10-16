# distributions/neural.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Categorical, Bernoulli, Poisson, LogNormal, Distribution
from typing import Optional, Union, Tuple, List

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.utilities import utils


# -------------------- Neural Emission -------------------- #
class NeuralEmission(Emission):
    """
    Neural/contextual emission distribution for HSMM states.

    Supports:
        - Gaussian (continuous) or Categorical/Bernoulli (discrete) emissions
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
        if emission_type not in {"gaussian", "categorical", "bernoulli"}:
            raise ValueError(f"Unsupported emission_type '{emission_type}'")

        self.emission_type = emission_type.lower()
        self.encoder = encoder
        self.context_mode = context_mode

        if self.emission_type == "gaussian":
            if not isinstance(params, tuple) or len(params) != 2:
                raise ValueError("Gaussian params must be (mu, cov)")
            self.mu, self.cov = params
            self.n_states, self.n_features = self.mu.shape
        else:
            self.logits = params
            self.n_states = self.logits.shape[-2]
            self.n_categories = self.logits.shape[-1]

    def contextual_params(self, theta: Optional[utils.ContextualVariables] = None):
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        if isinstance(theta.X, dict):
            out = self.encoder(**theta.X)
        elif isinstance(theta.X, tuple):
            out = self.encoder(*theta.X)
        else:
            out = self.encoder(theta.X)

        if self.emission_type == "gaussian":
            mu, cov = out
            return mu, cov
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(self.EPS)

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
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
        else:
            if X.ndim == 1 or X.shape[1] != self.n_categories:
                X_onehot = F.one_hot(X.long(), num_classes=self.n_categories).float()
            else:
                X_onehot = X.float()
            weighted_counts = posterior.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(self.EPS)).clamp_min(self.EPS).log()
            if inplace:
                self.logits = logits_new
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder, context_mode=self.context_mode)

    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None, n_categories: int = None, alpha: float = 1.0, encoder: Optional[nn.Module] = None):
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features)
            cov = torch.stack([torch.eye(n_features) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder)
        else:
            logits = torch.distributions.Dirichlet(torch.ones(n_categories) * alpha).sample([n_states])
            return cls(emission_type.lower(), logits.clamp_min(cls.EPS).log(), encoder=encoder)

    def __repr__(self):
        if self.emission_type == "gaussian":
            return f"NeuralEmission(Gaussian, n_states={self.n_states}, n_features={self.n_features})"
        return f"NeuralEmission({self.emission_type.capitalize()}, n_states={self.n_states}, n_categories={self.n_categories})"


# -------------------- Neural Duration -------------------- #
class NeuralDuration(Duration):
    """
    Neural/contextual duration distribution for HSMM states.
    Supports Poisson or LogNormal durations with optional neural context modulation.
    """
    def __init__(self, n_states: int, mode: str = "poisson", rate: Optional[torch.Tensor] = None, mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None, encoder: Optional[nn.Module] = None, context_mode: str = "none",
                 max_duration: int = 20, device: Optional[torch.device] = None):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device)
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode
        self.scale = 0.1
        if self.mode == "poisson":
            self._assign_params({"rate": rate})
        elif self.mode == "lognormal":
            self._assign_params({"mean": mean, "std": std})

    def forward(self, context=None, log=False, return_dist=False):
        durations = torch.arange(1, self.max_duration + 1, dtype=DTYPE, device=self.device).unsqueeze(0)
        params = self._get_params(context)
        logits = self._compute_logits(durations, params)
        if return_dist:
            return self._build_distributions(params)
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs) if log else probs

    def _get_params(self, context):
        if self.encoder and context:
            return self._contextual_params(context)
        if self.mode == "poisson":
            return {"rate": self.dist.rate}
        return {"mean": self.dist.mean, "std": self.dist.stddev}

    def _compute_logits(self, durations, params):
        if self.mode == "poisson":
            rate = params["rate"].unsqueeze(-1).clamp_min(EPS)
            return durations * torch.log(rate) - rate - torch.lgamma(durations + 1)
        mean = params["mean"].unsqueeze(-1)
        std = params["std"].unsqueeze(-1).clamp_min(EPS)
        return -0.5 * ((durations - mean)/std)**2 - torch.log(std) - 0.5 * torch.log(2*torch.pi)

    def _contextual_params(self, context):
        if isinstance(context, utils.ContextualVariables):
            X = context.X
            encoded = self.encoder(**X) if isinstance(X, dict) else self.encoder(*X) if isinstance(X, tuple) else self.encoder(X)
        else:
            encoded = self.encoder(context)
        if self.mode == "poisson":
            return {"rate": F.softplus(encoded).squeeze(-1)}
        mean, log_std = torch.chunk(encoded, 2, dim=-1)
        return {"mean": mean.squeeze(-1), "std": log_std.exp().clamp_min(EPS).squeeze(-1)}

    def _assign_params(self, params):
        if self.mode == "poisson":
            self.dist = Poisson(params["rate"].clamp_min(EPS))
        else:
            self.dist = LogNormal(params["mean"], params["std"].clamp_min(EPS))

    def _build_distributions(self, params):
        if self.mode == "poisson":
            return [Poisson(rate=params["rate"][k]) for k in range(self.n_states)]
        return [LogNormal(mean=params["mean"][k], std=params["std"][k]) for k in range(self.n_states)]

    def log_probs(self, context=None):
        return self.forward(context=context, log=True, return_dist=False)

    def probs(self, context=None):
        return self.forward(context=context, log=False, return_dist=False)

    def sample(self, n_samples=1, context=None, state_indices=None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    def estimate_duration_pdf(self, durations, posterior, theta=None, inplace=False):
        params = self._contextual_params(theta) if self.encoder and theta else self._compute_mle(durations, posterior)
        if inplace:
            self._assign_params(params)
            return self
        return NeuralDuration(n_states=self.n_states, mode=self.mode, encoder=self.encoder, context_mode=self.context_mode, max_duration=self.max_duration, device=self.device, **params)

    def _compute_mle(self, durations, posterior):
        weights = posterior.sum(dim=1, keepdim=True).clamp_min(EPS)
        weighted_mean = (posterior @ durations.unsqueeze(-1))/weights
        if self.mode=="poisson":
            return {"rate": weighted_mean.squeeze(-1)}
        log_dur = durations.clamp_min(EPS).log().unsqueeze(-1)
        weighted_log_mean = (posterior @ log_dur)/weights
        weighted_log_var = (posterior @ ((log_dur-weighted_log_mean)**2))/weights
        return {"mean": weighted_log_mean.squeeze(-1), "std": weighted_log_var.sqrt().squeeze(-1)}

    def log_prob(self, durations):
        return self.dist.log_prob(durations.unsqueeze(0) if durations.ndim==1 else durations)

    def to(self, device):
        super().to(device)
        if self.encoder:
            self.encoder.to(device)
        return self


# -------------------- Neural Transition -------------------- #
class NeuralTransition(Transition):
    """
    Neural/contextual transition distribution for HSMM states.
    """
    EPS = 1e-8

    def __init__(self, logits: torch.Tensor, encoder: Optional[nn.Module] = None, context_mode: str = "none"):
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")
        if logits.ndim not in (2,3):
            raise ValueError("logits must be [n_states, n_states] or [batch, n_states, n_states]")
        super().__init__(logits=logits)
        self.encoder = encoder
        self.context_mode = context_mode

    @property
    def n_states(self):
        return self.logits.shape[-1]

    def contextual_logits(self, theta=None):
        if self.encoder is None or theta is None:
            return self.logits
        if isinstance(theta.X, dict):
            out = self.encoder(**theta.X)
        elif isinstance(theta.X, tuple):
            out = self.encoder(*theta.X)
        else:
            out = self.encoder(theta.X)
        return torch.log_softmax(out, dim=-1).clamp(min=self.EPS)

    def update(self, posterior, inplace=True):
        logits = (posterior + self.EPS).log()
        if inplace:
            self.logits = logits
            return self
        return NeuralTransition(logits=logits, encoder=self.encoder, context_mode=self.context_mode)

    @classmethod
    def initialize(cls, n_states, alpha=1.0, batch=1, encoder=None):
        probs = torch.distributions.Dirichlet(torch.ones(n_states)*alpha).sample([batch])
        if batch==1:
            probs = probs.squeeze(0)
        return cls(logits=probs.clamp_min(cls.EPS).log(), encoder=encoder)
