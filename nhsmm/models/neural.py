# nhsmm/models/neural.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Independent, Categorical, Poisson, LogNormal, MultivariateNormal, Dirichlet

from typing import Literal, Optional, Union, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import numpy as np

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.models.hsmm import HSMM
from nhsmm import utils


class NeuralEmission(Emission):
    """Neural/contextual emission distribution with dynamic DOF and adapters."""

    def __init__(
        self,
        emission_type: str,
        params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.context_mode = context_mode.lower()
        if self.context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

        emission_type = emission_type.lower()
        if emission_type == "gaussian":
            mu, cov = params
            n_states, n_features = mu.shape
        else:
            logits = params
            n_states, n_features = logits.shape

        super().__init__(n_states=n_states, n_features=n_features, emission_type=emission_type)
        self.encoder = encoder

        if emission_type == "gaussian":
            self.mu = nn.Parameter(mu.to(self.device, DTYPE))
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            self.cov = nn.Parameter(cov.to(self.device, DTYPE))
            self._gaussian_proj = nn.Linear(n_features, 2 * n_states * n_features, bias=False).to(self.device, DTYPE)
        else:
            self.logits = nn.Parameter(params.to(self.device, DTYPE))

        # Contextual adapters
        self.temporal_adapter = (
            nn.Conv1d(n_features, n_features, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if self.context_mode == "temporal" else None
        )
        self.spatial_adapter = (
            nn.Linear(n_features, n_features, bias=False).to(self.device, DTYPE)
            if self.context_mode == "spatial" else None
        )

    @property
    def dof(self) -> int:
        dof = self.mu.numel() + getattr(self, "cov", torch.tensor([])).numel() if self.emission_type == "gaussian" else self.logits.numel()
        if self.encoder:
            dof += sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        if self.temporal_adapter:
            dof += sum(p.numel() for p in self.temporal_adapter.parameters() if p.requires_grad)
        if self.spatial_adapter:
            dof += sum(p.numel() for p in self.spatial_adapter.parameters() if p.requires_grad)
        return dof

    def contextual_params(self, theta: Optional[torch.Tensor] = None):
        out = theta
        if self.encoder and theta is not None:
            out = self.encoder(theta)
        if self.temporal_adapter and out is not None and out.ndim == 3:  # (B,T,F)
            out = out.transpose(1, 2)
            out = self.temporal_adapter(out)
            out = out.transpose(1, 2)
        if self.spatial_adapter and out is not None:
            out = self.spatial_adapter(out)

        if self.emission_type == "gaussian":
            if out is None:
                return self.mu, self.cov
            if isinstance(out, (tuple, list)) and len(out) == 2:
                mu, cov = out
            else:
                tmp = self._gaussian_proj(out).view(-1, 2, self.n_states, self.n_features)
                mu, cov = tmp[:, 0], torch.diag_embed(F.softplus(tmp[:, 1]) + EPS)
            return mu.to(self.device, DTYPE), cov.to(self.device, DTYPE)
        else:
            logits = out if out is not None else self.logits
            if logits.shape[-1] != self.n_features:
                proj = nn.Linear(logits.shape[-1], self.n_features, bias=False).to(self.device, DTYPE)
                logits = proj(logits)
            return F.log_softmax(logits, dim=-1).clamp_min(EPS)

    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, log: bool = False):
        X = X.to(self.device, DTYPE)
        batch_mode = X.ndim == 3
        B, T, Fdim = (X.shape if batch_mode else (1, *X.shape))
        K = self.n_states

        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            if mu.ndim == 2:
                mu = mu.unsqueeze(0).expand(B, K, Fdim)
            if cov.ndim == 3 and cov.shape[0] != B:
                cov = cov.expand(B, K, Fdim, Fdim)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            logp = dist.log_prob(X.unsqueeze(-2))
            return logp if log else logp.exp()
        else:
            logits = self.contextual_params(theta)
            probs = F.softmax(logits, dim=-1)
            X_idx = X.long()
            if (X_idx.max() >= self.n_features) or (X_idx.min() < 0):
                raise ValueError(f"Observation index out of bounds: 0 <= X < {self.n_features}")
            if batch_mode:
                logp = torch.log(probs.gather(-1, X_idx.unsqueeze(-2).expand(-1, K, -1)) + EPS).transpose(1, 2)
            else:
                logp = torch.log(probs[:, X_idx] + EPS).T
            return logp if log else logp.exp()

    def log_prob(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None):
        return self.forward(X, theta=theta, log=True)

    def sample(self, n_samples: int = 1, theta: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            samples = dist.rsample((n_samples,)).transpose(0, 1)  # (K, n_samples, F)
        else:
            logits = self.contextual_params(theta)
            dist = Categorical(logits=logits)
            samples = dist.sample((n_samples,)).transpose(0, 1)  # (K, n_samples)
        return samples if state_indices is None else samples[state_indices]

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, DTYPE)
        X = X.to(self.device, DTYPE)
        X_flat = X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X
        posterior_flat = posterior.reshape(-1, posterior.shape[-1]) if posterior.ndim > 2 else posterior

        if self.emission_type == "gaussian":
            counts = posterior_flat.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (posterior_flat.T @ X_flat) / counts
            cov_new = torch.stack([
                ((X_flat - mu_new[s]).T * posterior_flat[:, s]) @ (X_flat - mu_new[s]) / counts[s, 0]
                + torch.eye(self.n_features, device=self.device, dtype=DTYPE) * EPS
                for s in range(self.n_states)
            ])
            if inplace:
                self.mu.data.copy_(mu_new)
                self.cov.data.copy_(cov_new)
                return self
            return NeuralEmission("gaussian", (mu_new, cov_new), encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device)
        else:
            X_onehot = F.one_hot(X_flat.long(), num_classes=self.n_features).float()
            weighted_counts = posterior_flat.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device)

    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None,
                   n_categories: int = None, alpha: float = 1.0,
                   encoder: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=DTYPE) * 0.1
            cov = torch.stack([torch.ones(n_features, device=device, dtype=DTYPE) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder, device=device)
        else:
            logits = Dirichlet(torch.ones(n_categories, device=device, dtype=DTYPE) * alpha).sample([n_states]).clamp_min(EPS).log()
            return cls(emission_type.lower(), logits, encoder=encoder, device=device)

class NeuralDuration(Duration):
    """Neural/contextual duration distribution with adapters."""

    def __init__(
        self,
        n_states: int,
        mode: str = "poisson",
        rate: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        max_duration: int = 20,
        device: Optional[torch.device] = None,
        scale: float = 0.1,
    ):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device)
        self.device = device or torch.device("cpu")
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = scale

        self.temporal_adapter = None
        self.spatial_adapter = None
        if self.context_mode == "temporal":
            self.temporal_adapter = nn.Conv1d(n_states, n_states, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
        elif self.context_mode == "spatial":
            self.spatial_adapter = nn.Linear(n_states, n_states, bias=False).to(self.device, DTYPE)

        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))

    @property
    def dof(self) -> int:
        dof = self.rate.numel() if self.mode == "poisson" else self.mean.numel() + self.std.numel()
        if self.encoder:
            dof += sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        if self.temporal_adapter:
            dof += sum(p.numel() for p in self.temporal_adapter.parameters() if p.requires_grad)
        if self.spatial_adapter:
            dof += sum(p.numel() for p in self.spatial_adapter.parameters() if p.requires_grad)
        return dof

    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        X = getattr(context, "X", context) if hasattr(context, "X") else context
        if X is not None:
            X = X.to(self.device, DTYPE)
        encoded = self.encoder(X) if self.encoder and X is not None else None
        if encoded is not None:
            if self.temporal_adapter and encoded.ndim == 3:
                encoded = encoded.transpose(1, 2)
                encoded = self.temporal_adapter(encoded)
                encoded = encoded.transpose(1, 2)
            if self.spatial_adapter:
                encoded = self.spatial_adapter(encoded)
        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded, 2, dim=-1)
            else:
                mean, log_std = self.mean, self.std.log()
            return mean, log_std.exp().clamp_min(EPS)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 2 - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits, dim=-1)
        if return_dist:
            if self.mode == "poisson":
                return [Poisson(rate[k]) for k in range(self.n_states)]
            else:
                return [LogNormal(mean[k], std[k]) for k in range(self.n_states)]
        return torch.log(probs) if log else probs

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    def log_prob(self, X: torch.Tensor, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, log=True)

class NeuralTransition(Transition):
    """Neural/contextual transition distribution with safe logits and adapters."""

    def __init__(
        self,
        n_states: int,
        logits: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
        scale: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__(n_states=n_states, device=device)
        self.device = device or torch.device("cpu")
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = scale
        self.temperature = temperature

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            uniform = torch.full((n_states, n_states), 1.0 / n_states, device=self.device, dtype=DTYPE)
            self.logits = nn.Parameter(torch.log(uniform.clamp_min(EPS)))

        self.temporal_adapter = nn.Conv1d(n_states, n_states, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE) if self.context_mode == "temporal" else None
        self.spatial_adapter = nn.Linear(n_states, n_states, bias=False).to(self.device, DTYPE) if self.context_mode == "spatial" else None

    @property
    def dof(self) -> int:
        dof = self.n_states * (self.n_states - 1)
        if self.encoder:
            dof += sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        if self.temporal_adapter:
            dof += sum(p.numel() for p in self.temporal_adapter.parameters() if p.requires_grad)
        if self.spatial_adapter:
            dof += sum(p.numel() for p in self.spatial_adapter.parameters() if p.requires_grad)
        return dof

    def contextual_logits(self, theta: Optional[torch.Tensor] = None):
        base_logits = self.logits.clone()
        if self.encoder is not None and theta is not None:
            X = getattr(theta, "X", theta)
            out = self.encoder(X)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 2 and out.shape[1] == self.n_states ** 2:
                out = out.view(-1, self.n_states, self.n_states)
            elif out.ndim == 3 and out.shape[1:] == (self.n_states, self.n_states):
                pass
            else:
                out = out.view(-1, self.n_states, self.n_states)
            delta = torch.tanh(out) * self.scale
            base_logits = base_logits.unsqueeze(0).expand_as(delta) + delta

        if self.temporal_adapter is not None and theta is not None:
            delta = theta.transpose(1, 2)
            delta = self.temporal_adapter(delta)
            delta = delta.transpose(1, 2)
            base_logits = base_logits + delta

        if self.spatial_adapter is not None and theta is not None:
            delta = self.spatial_adapter(theta)
            base_logits = base_logits + delta

        safe_logits = F.log_softmax(base_logits / self.temperature, dim=-1)
        safe_logits = torch.where(torch.isnan(safe_logits) | torch.isinf(safe_logits),
                                  torch.log(torch.full_like(safe_logits, 1.0 / self.n_states)), safe_logits)
        return safe_logits

    def update(self, posterior: torch.Tensor, inplace: bool = True):
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
        logits = torch.log(posterior.to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(self.n_states, logits=logits, encoder=self.encoder, context_mode=self.context_mode,
                                device=self.device, scale=self.scale, temperature=self.temperature)

    def forward(self, theta: Optional[torch.Tensor] = None):
        return self.contextual_logits(theta)


@dataclass
class NHSMMConfig:
    n_states: int
    n_features: int
    max_duration: int
    alpha: float = 1.0
    seed: Optional[int] = None
    emission_type: str = "gaussian"
    encoder: Optional[nn.Module] = None
    device: Optional[torch.device] = None
    context_dim: Optional[int] = None
    min_covar: float = 1e-3
    emission_scale: float = 0.1
    transition_scale: float = 0.1
    duration_scale: float = 0.1
    emission_temp: float = 1.0
    transition_temp: float = 1.0
    duration_temp: float = 1.0
    n_context_states: Optional[int] = None

class NeuralHSMM(HSMM):
    """ NeuralHSMM with EM, Viterbi, context, and gradient support.
    """

    def __init__(self, config: NHSMMConfig):
        
        self.config = config
        nn.Module.__init__(self)

        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = config.encoder.to(self.device, dtype=DTYPE) if config.encoder else None
        self.n_features = config.n_features
        self.min_covar = config.min_covar

        # Neural modules
        self.emission_module = NeuralEmission.initialize(
            emission_type=config.emission_type,
            n_states=config.n_states,
            n_features=config.n_features,
            encoder=self.encoder,
            device=self.device,
        )
        self.duration_module = NeuralDuration(
            n_states=config.n_states,
            max_duration=config.max_duration,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim else "none",
            scale=config.duration_scale,
            device=self.device,
        )
        self.transition_module = NeuralTransition(
            n_states=config.n_states,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim else "none",
            scale=config.transition_scale,
            device=self.device,
        )

        # Context embeddings
        if config.n_context_states and config.context_dim:
            self.context_embedding = nn.Embedding(config.n_context_states, config.context_dim)
            nn.init.normal_(self.context_embedding.weight, 0.0, 1e-3)

            self.ctx_transition = nn.Linear(config.context_dim, config.n_states**2, device=self.device, dtype=DTYPE)
            self.ctx_duration = nn.Linear(config.context_dim, config.n_states*config.max_duration, device=self.device, dtype=DTYPE)
            self.ctx_emission = nn.Linear(config.context_dim, config.n_states*config.n_features, device=self.device, dtype=DTYPE)

            for m in (self.ctx_transition, self.ctx_duration, self.ctx_emission):
                nn.init.normal_(m.weight, 0.0, 1e-3)
                nn.init.normal_(m.bias, 0.0, 1e-3)
        else:
            self.context_embedding = None
            self.ctx_transition = None
            self.ctx_duration = None
            self.ctx_emission = None

        super().__init__(
            n_states=config.n_states,
            n_features=config.n_features,
            max_duration=config.max_duration,
            context_dim=config.context_dim,
            min_covar=config.min_covar,
            alpha=config.alpha,
            device=self.device,
            seed=config.seed,
        )
        self._params: dict = {"emission_type": config.emission_type}
        self._params["emission_pdf"] = self.sample_emission_pdf(
            temperature=config.emission_temp,
            scale=config.emission_scale,
        )
        self._context = None

    @property
    def emission_type(self) -> str:
        return self._params.get("emission_type", "gaussian")

    @property
    def pdf(self):
        return self._params.get("emission_pdf", None)

    @property
    def dof(self) -> int:
        """
        Compute degrees of freedom (dof) for the HSMM model, including:
          - initial state probabilities
          - transition probabilities
          - duration probabilities
          - emission parameters
        """

        nS, nD, nF = self.n_states, self.max_duration, self.n_features
        dof = 0

        # Initial state probabilities
        dof += nS - 1

        # Transition matrix (row-stochastic)
        dof += nS * (nS - 1)

        # Duration distributions (row-stochastic)
        dof += nS * (nD - 1)

        # Emission parameters
        pdf = getattr(self, "pdf", None)
        if pdf is not None:
            if isinstance(pdf, Categorical):
                # Each state's categorical probs minus one (sum to 1)
                dof += nS * (pdf.logits.shape[-1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                # Mean and diagonal covariance for each state
                dof += nS * (nF + nF)  # mean + diag(cov)
            else:
                # Fallback: warn for unhandled pdf type
                import warnings
                warnings.warn(f"Unknown pdf type {type(pdf)}, emission dof not counted")

        return dof

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        scale: float = None,
    ):
        """Create a batched emission distribution conditioned on optional X and theta.
        
        Returns:
            - Categorical (n_states distributions) or
            - MultivariateNormal (n_states distributions)
        """
        nS, nF = self.n_states, self.n_features
        scale = scale or self.config.emission_scale

        # Flatten X if provided
        X_flat = X.reshape(-1, nF).to(dtype=DTYPE, device=self.device) if X is not None else None

        # Compute context delta
        delta_mean = None
        if theta is not None and self.ctx_emission is not None:
            t = theta.to(dtype=DTYPE, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            delta = self.ctx_emission(t)
            if delta.ndim > 2:
                delta = delta.view(delta.shape[0], -1)
            delta_mean = delta.mean(dim=0)
            if delta_mean.numel() != nS * nF:
                raise ValueError(f"ctx_emission output ({delta_mean.shape}) cannot be reshaped to ({nS}, {nF})")
            delta_mean = delta_mean.view(nS, nF)

        # --- Categorical emission ---
        if self.emission_type == "categorical":
            if X_flat is not None:
                probs = X_flat.mean(dim=0, keepdim=True).repeat(nS, 1)
            else:
                probs = torch.full((nS, nF), 1.0 / nF, dtype=DTYPE, device=self.device)

            if delta_mean is not None:
                probs = probs + scale * torch.tanh(delta_mean)

            probs = (probs / temperature).to(dtype=DTYPE, device=self.device)
            logits = probs - torch.logsumexp(probs, dim=-1, keepdim=True)
            return Categorical(logits=logits)

        # --- Gaussian emission ---
        elif self.emission_type == "gaussian":
            if X_flat is not None:
                mean = X_flat.mean(dim=0, keepdim=True).repeat(nS, 1)
                var = X_flat.var(dim=0, unbiased=False, keepdim=True).repeat(nS, 1).clamp_min(self.min_covar)
            else:
                mean = torch.zeros(nS, nF, dtype=DTYPE, device=self.device)
                var = torch.full((nS, nF), self.min_covar, dtype=DTYPE, device=self.device)

            if delta_mean is not None:
                mean = mean + scale * torch.tanh(delta_mean)

            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean.to(self.device, DTYPE), covariance_matrix=cov.to(self.device, DTYPE))

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        scale: float = None
    ):
        """
        Estimate emission distributions from weighted data (posterior responsibilities),
        optionally modulated by context/encoder outputs (theta).

        Returns:
            - MultivariateNormal (for Gaussian)
            - Categorical (for categorical)
        """
        K, F = self.n_states, self.n_features
        scale = scale or getattr(self.config, "emission_scale", 1.0)

        # Ensure device/dtype consistency
        X = X.to(dtype=DTYPE, device=self.device)
        posterior = posterior.to(dtype=DTYPE, device=self.device)

        # --- Compute delta from context if available ---
        delta = torch.zeros((K, F), dtype=DTYPE, device=self.device)
        if theta is not None:
            t = theta.to(dtype=DTYPE, device=self.device)
            if self.ctx_emission is not None:
                d = self.ctx_emission(t)
                # Collapse batch if needed
                if d.ndim > 2:
                    d = d.mean(dim=0)
                # Reshape to (K, F) safely
                if d.ndim == 2 and d.shape[1] >= K * F:
                    # Take first K*F features and reshape
                    delta = scale * d[:, :K*F].mean(dim=0).view(K, F)
                elif d.shape == (K, F):
                    delta = scale * d
                else:
                    delta = torch.zeros((K, F), dtype=DTYPE, device=self.device)
            else:
                # fallback: extract first K*F features from theta
                if t.ndim == 2 and t.shape[1] >= K * F:
                    delta = scale * t[:, :K*F].mean(dim=0).view(K, F)
                elif t.ndim == 1 and t.numel() >= K * F:
                    delta = scale * t[:K*F].view(K, F)
                else:
                    delta = torch.zeros((K, F), dtype=DTYPE, device=self.device)

        # --- Categorical emission ---
        if self.emission_type == "categorical":
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                X_labels = X.squeeze(-1).long()
                counts = torch.bincount(X_labels, minlength=self.n_features).to(dtype=DTYPE, device=self.device)
            else:
                counts = X.sum(dim=0).to(dtype=DTYPE, device=self.device)
            probs = counts / counts.sum()
            probs = (probs + delta.mean(dim=0)).clamp_min(EPS)
            probs = probs / probs.sum()
            return Categorical(probs=probs.repeat(K, 1).to(dtype=DTYPE, device=self.device))

        # --- Gaussian emission ---
        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0).clamp_min(EPS)  # (K,)
            mean = (posterior.T @ X) / Nk.unsqueeze(1)  # (K, F)
            diff = X.unsqueeze(1) - mean.unsqueeze(0)   # (T, K, F)
            weighted = diff * posterior.unsqueeze(-1)  # weight residuals
            cov = torch.einsum("tkf,tkh->kfh", weighted, diff) / Nk.unsqueeze(1).unsqueeze(-1)
            cov += torch.eye(F, dtype=DTYPE, device=self.device).unsqueeze(0) * getattr(self, "min_covar", 1e-6)
            mean = mean + delta
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def initialize_emissions(self, X, method: str = "moment"):
        """Initialize emission parameters from raw data X with enhanced robustness."""

        # Convert to tensor if needed
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=DTYPE, device=self.device)
        X = X.to(dtype=DTYPE, device=self.device)
        K = self.n_states

        # --- Gaussian emissions ---
        if self.emission_type == "gaussian":
            if X.ndim != 2:
                raise ValueError(f"Gaussian emissions require X shape (T,F), got {X.shape}")
            T, F = X.shape

            if method == "moment":
                mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)
            elif method == "kmeans":
                from sklearn.cluster import KMeans
                # Fit KMeans safely; fallback if cluster empty
                labels = torch.tensor(
                    KMeans(n_clusters=K, n_init=10, random_state=0)
                    .fit_predict(X.cpu().numpy()), device=self.device
                )
                mu_init = torch.stack([
                    X[labels==k].mean(dim=0) if (labels==k).any() else X.mean(dim=0)
                    for k in range(K)
                ])
                var_init = torch.stack([
                    X[labels==k].var(dim=0, unbiased=False) if (labels==k).any() else X.var(dim=0, unbiased=False)
                    for k in range(K)
                ])
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            # Clamp small variances
            var_init = var_init.clamp_min(getattr(self, "min_covar", 1e-6))
            cov = torch.diag_embed(var_init)

            # Assign to emission module
            self.emission_module.mu = nn.Parameter(mu_init)
            self.emission_module.cov = nn.Parameter(cov)
            self._params["emission_pdf"] = MultivariateNormal(mu_init, covariance_matrix=cov)

        # --- Categorical / Bernoulli emissions ---
        elif self.emission_type in ("categorical", "bernoulli"):
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                counts = torch.bincount(X.squeeze(-1).long(), minlength=self.n_features).to(dtype=DTYPE, device=self.device)
            else:
                counts = X.sum(dim=0).to(dtype=DTYPE, device=self.device)

            probs = counts / counts.sum()
            # Safe fallback if counts are all zeros
            if probs.sum() == 0:
                probs = torch.full_like(probs, 1.0 / len(probs))
            logits = probs.clamp_min(EPS).log().unsqueeze(0).repeat(K, 1)

            # Zero placeholders for Gaussian params (not used)
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.cov = nn.Parameter(torch.zeros_like(logits))
            self._params["emission_pdf"] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

        # Ensure everything is on the correct device
        for name in ["mu", "cov"]:
            param = getattr(self.emission_module, name, None)
            if param is not None:
                setattr(self.emission_module, name, nn.Parameter(param.to(self.device, DTYPE)))

    def set_context(self, context: Optional[torch.Tensor]):
        """Store a (B,H) or (H,) context tensor on model device/dtype."""
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(device=self.device, dtype=DTYPE)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def clear_context(self):
        self._context = None

    def _combine_context(self, theta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Safely combine encoder-produced theta with externally set context."""
        theta = theta.to(self.device, DTYPE) if theta is not None else None
        ctx = self._context.to(self.device, DTYPE) if getattr(self, "_context", None) is not None else None

        if theta is None and ctx is None:
            return None
        if theta is None:
            return ctx
        if ctx is None:
            return theta

        # Align batch dimension
        if theta.shape[0] != ctx.shape[0]:
            if theta.shape[0] == 1:
                theta = theta.expand(ctx.shape[0], -1)
            elif ctx.shape[0] == 1:
                ctx = ctx.expand(theta.shape[0], -1)
            else:
                raise ValueError(f"Batch mismatch: theta {theta.shape} vs context {ctx.shape}")

        return torch.cat([theta, ctx], dim=-1)

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, scale: Optional[float] = None):
        """
        Compute context-modulated emission PDF with safe batch/covariance handling.
        """
        scale = scale or getattr(self.config, "emission_scale", 1.0)
        pdf = self.pdf
        if pdf is None:
            return None

        combined = self._combine_context(theta)
        if combined is None:
            return pdf

        try:
            if hasattr(self, "ctx_emission") and self.ctx_emission is not None:
                delta = self.ctx_emission(combined)
            else:
                delta = combined[..., :self.n_states * self.n_features]

            delta = delta.to(self.device, DTYPE)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0)

            B = delta.shape[0]
            expected = self.n_states * self.n_features

            # Reshape safely
            if delta.shape[1] != expected:
                if delta.shape[1] < expected:
                    delta = F.pad(delta, (0, expected - delta.shape[1]))
                else:
                    delta = delta[..., :expected]
            delta = delta.view(B, self.n_states, self.n_features)
            delta = scale * torch.tanh(delta)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            print(f"[contextual_emission_pdf] Failed to compute delta, using base PDF: {e}")
            return pdf

        # --- Apply modulation ---
        if isinstance(pdf, Categorical):
            logits = pdf.logits.to(self.device, DTYPE)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0).expand(B, -1)
            logits = logits + delta.mean(dim=-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return Categorical(logits=logits)

        elif isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(self.device, DTYPE)
            cov = pdf.covariance_matrix.to(self.device, DTYPE)

            # Add jitter to keep PD
            if cov.ndim == 4:
                cov = cov + EPS * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).view(1, 1, cov.shape[-1], cov.shape[-1])
            elif cov.ndim == 3:
                cov = cov + EPS * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).view(1, cov.shape[-1], cov.shape[-1])
            elif cov.ndim == 2:
                cov = cov + EPS * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)

            # Align mean batch shape
            if mean.ndim == 1:
                mean = mean.unsqueeze(0).expand(B, -1)
            elif mean.ndim == 2:
                if B > 1:
                    mean = mean.unsqueeze(0).expand(B, -1, -1)
            elif mean.ndim == 3 and mean.shape[0] != B:
                mean = mean[0].unsqueeze(0).expand(B, -1, -1)

            mean = mean + delta

            # Keep covariance per-state — only expand if singleton
            if cov.ndim == 3 and cov.shape[0] == 1 and B > 1:
                cov = cov.expand(B, *cov.shape[1:])
            elif cov.ndim == 4 and cov.shape[0] == 1 and B > 1:
                cov = cov.expand(B, *cov.shape[1:])

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            print(f"[contextual_emission_pdf] Unsupported PDF type {type(pdf)}, using base PDF")
            return pdf

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor] = None, scale: Optional[float] = None, temperature: Optional[float] = None):
        """
        Compute context-modulated duration probabilities for HSMM states.

        Args:
            theta (torch.Tensor, optional): Context tensor for modulation.
            scale (float, optional): Scaling factor for context modulation.
            temperature (float, optional): Temperature for softmax smoothing.

        Returns:
            torch.Tensor: Duration probabilities, shape (n_states, max_duration) or (B, n_states, max_duration)
        """
        scale = scale or getattr(self.config, "duration_scale", 1.0)
        temperature = max(temperature or getattr(self.config, "duration_temp", 1.0), 1e-6)

        logits = getattr(self.duration_module, "logits", None)
        if logits is None:
            return self.duration_module.forward(log=False)

        combined = self._combine_context(theta)
        if combined is None or not hasattr(self, "ctx_duration") or self.ctx_duration is None:
            return F.softmax(logits.to(self.device, DTYPE) / temperature, dim=-1)

        try:
            delta = self.ctx_duration(combined).to(self.device, DTYPE)
            # reshape delta to (B, n_states, max_duration) if needed
            if delta.ndim == 2 and delta.shape[1] == self.n_states * self.max_duration:
                delta = delta.view(-1, self.n_states, self.max_duration)
                if delta.shape[0] > 1:
                    delta = delta.mean(dim=0, keepdim=True)
            elif delta.ndim == 1:
                delta = delta.view(1, self.n_states, self.max_duration)
            elif delta.ndim == 3 and delta.shape[1:] == (self.n_states, self.max_duration):
                pass
            else:
                print(f"[contextual_duration_pdf] Unexpected delta shape: {delta.shape}, using base logits")
                delta = torch.zeros_like(logits, device=self.device, dtype=DTYPE)
        except Exception as e:
            print(f"[contextual_duration_pdf] Failed to compute delta: {e}, using base logits")
            delta = torch.zeros_like(logits, device=self.device, dtype=DTYPE)

        logits = logits.to(self.device, DTYPE)
        # expand logits if batch delta exists
        if delta.ndim == 3 and delta.shape[0] > 1:
            logits = logits.unsqueeze(0).expand_as(delta)
        else:
            logits = logits.unsqueeze(0)

        combined_logits = logits + scale * torch.tanh(delta)
        combined_logits = torch.nan_to_num(combined_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return F.softmax(combined_logits / temperature, dim=-1)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor] = None, scale: Optional[float] = None, temperature: Optional[float] = None):
        """
        Compute context-modulated transition probabilities for HSMM states.

        Args:
            theta (torch.Tensor, optional): Context tensor for modulation.
            scale (float, optional): Scaling factor for context modulation.
            temperature (float, optional): Temperature for softmax smoothing.

        Returns:
            torch.Tensor: Transition probabilities, shape (n_states, n_states) or (B, n_states, n_states)
        """
        scale = scale or getattr(self.config, "transition_scale", 1.0)
        temperature = max(temperature or getattr(self.config, "transition_temp", 1.0), 1e-6)

        logits = getattr(self.transition_module, "logits", None)
        if logits is None:
            return self.transition_module.forward(log=False)

        combined = self._combine_context(theta)
        if combined is None or not hasattr(self, "ctx_transition") or self.ctx_transition is None:
            return F.softmax(logits.to(self.device, DTYPE) / temperature, dim=-1)

        try:
            delta = self.ctx_transition(combined).to(self.device, DTYPE)
            # reshape delta to (B, n_states, n_states)
            if delta.ndim == 2 and delta.shape[1] == self.n_states * self.n_states:
                delta = delta.view(-1, self.n_states, self.n_states)
                if delta.shape[0] > 1:
                    delta = delta.mean(dim=0, keepdim=True)
            elif delta.ndim == 1:
                delta = delta.view(1, self.n_states, self.n_states)
            elif delta.ndim == 3 and delta.shape[1:] == (self.n_states, self.n_states):
                pass
            else:
                print(f"[contextual_transition_matrix] Unexpected delta shape: {delta.shape}, using base logits")
                delta = torch.zeros_like(logits, device=self.device, dtype=DTYPE)
        except Exception as e:
            print(f"[contextual_transition_matrix] Failed to compute delta: {e}, using base logits")
            delta = torch.zeros_like(logits, device=self.device, dtype=DTYPE)

        logits = logits.to(self.device, DTYPE)
        # expand logits if batch delta exists
        if delta.ndim == 3 and delta.shape[0] > 1:
            logits = logits.unsqueeze(0).expand_as(delta)
        else:
            logits = logits.unsqueeze(0)

        combined_logits = logits + scale * torch.tanh(delta)
        combined_logits = torch.nan_to_num(combined_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return F.softmax(combined_logits / temperature, dim=-1)

    def encode_observations(
        self,
        X: torch.Tensor,
        detach: bool = True,
        keep_sequence: bool = False,
        flatten_batch: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Encode observations using the attached encoder with optional batch flattening.

        Args:
            X: Tensor of shape [T, F] or [B, T, F].
            detach: If True, detaches the output from the computation graph.
            keep_sequence: If True, keep sequence dimension [B, T, H]; otherwise collapse to [B, H].
            flatten_batch: If True and keep_sequence=False, output is flattened to [B*T, H].

        Returns:
            Encoded tensor of shape [B, H], [B, T, H], or [B*T, H] (if flatten_batch),
            or None if no encoder is attached.
        """
        if self.encoder is None:
            return None

        # Ensure batch dimension
        if X.ndim == 2:
            X = X.unsqueeze(0)  # [1, T, F]
        elif X.ndim != 3:
            raise ValueError(f"Input X must be [T,F] or [B,T,F], got {X.shape}")

        B, T, F = X.shape
        inp = X.to(dtype=DTYPE, device=self.device)

        # Encode
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]  # handle encoders returning (output, hidden)

        if detach:
            out = out.detach()

        out = out.to(dtype=DTYPE, device=self.device)

        # Handle sequence dimension
        if out.ndim == 3:  # [B, T, H]
            if not keep_sequence:
                out = out.mean(dim=1)  # collapse to [B, H]
            if flatten_batch:
                out = out.reshape(B * T, -1)  # [B*T, H]
        elif out.ndim == 2:  # [B, H]
            if keep_sequence:
                out = out.unsqueeze(1)  # add singleton sequence dimension [B, 1, H]
            elif flatten_batch:
                # No sequence to flatten; keep as [B, H]
                pass
        else:
            raise RuntimeError(f"Unexpected encoder output shape: {out.shape}")

        return out

    def forward(
        self,
        X: torch.Tensor,
        return_pdf: bool = False,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, "PDFType"]:
        """
        Monolithic forward pass: encode observations, merge context, return theta or contextually-modulated PDF.

        Args:
            X: Observation tensor [T, F] or [B, T, F].
            return_pdf: If True, returns contextually-modulated emission PDF.
            context: Optional raw context features.
            context_ids: Optional IDs to embed and merge with context.

        Returns:
            Encoded theta tensor [B, H] or [B, T, H], or contextually-modulated emission PDF.
        """
        prev_ctx = self._context
        ctx_list = []

        # --- Build context ---
        if context_ids is not None:
            if self.context_embedding is None:
                raise ValueError("context_ids provided but no context_embedding defined.")
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:
                ctx_emb = ctx_emb.mean(dim=1)
            ctx_list.append(ctx_emb.to(dtype=DTYPE, device=self.device))

        if context is not None:
            ctx_list.append(context.to(dtype=DTYPE, device=self.device))

        ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
        if ctx is not None:
            self.set_context(ctx)

        # --- Encode observations ---
        theta = self.encode_observations(X)  # [B, H] or [B, T, H]

        # --- Early return: just theta ---
        if not return_pdf:
            self._context = prev_ctx
            return theta

        # --- Contextual emission PDF ---
        pdf = self.pdf
        if pdf is None:
            self._context = prev_ctx
            return None

        # Merge context + encoder
        combined = self._combine_context(theta)
        delta = None
        try:
            if combined is not None and self.ctx_emission is not None:
                delta = self.ctx_emission(combined)
            elif combined is not None:
                delta = combined[..., :self.n_states * self.n_features]
        except Exception as e:
            print(f"[forward] ctx_emission failed: {e}")
            delta = None

        # --- Safe delta reshaping ---
        if delta is not None:
            delta = delta.to(self.device, DTYPE)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0)
            B = max(1, delta.shape[0])
            expected = self.n_states * self.n_features
            if delta.numel() < expected * B:
                delta = delta.flatten().repeat(B)[:expected*B]
            delta = delta.view(B, self.n_states, self.n_features)
            delta = getattr(self.config, "emission_scale", 1.0) * torch.tanh(delta)

        # --- Apply delta to PDF ---
        if isinstance(pdf, Categorical):
            logits = pdf.logits.to(self.device, DTYPE)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0).expand(B, -1)
            if delta is not None:
                logits = logits + delta.mean(dim=-1)
            # sanitize
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.log(torch.full_like(logits, 1.0 / self.n_features))
            pdf_mod = Categorical(logits=logits)
            # Dynamic DOF
            self.emission_module.dof_dynamic = self.n_states * (pdf_mod.logits.shape[-1] - 1)

        elif isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(self.device, DTYPE)
            cov = pdf.covariance_matrix.to(self.device, DTYPE).clamp_min(1e-6)
            if mean.ndim == 2:
                mean = mean.unsqueeze(0)
            if delta is not None:
                mean = mean + delta
            pdf_mod = MultivariateNormal(loc=mean, covariance_matrix=cov)
            # Dynamic DOF
            nS = mean.shape[-2]
            nF = mean.shape[-1]
            self.emission_module.dof_dynamic = nS * (2 * nF)

        else:
            pdf_mod = pdf

        self._context = prev_ctx
        return pdf_mod

    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:
        """
        Monolithic, batch-safe, context-aware prediction with dynamic DOF handling.

        Args:
            X: Observations [T, F] or [B, T, F].
            lengths: Optional sequence lengths for padded batch.
            algorithm: "viterbi" or "map".
            context: Optional raw context tensor.
            context_ids: Optional context IDs for embedding.
            batch_size: Batch size for HSMM prediction.

        Returns:
            List of predicted state sequences (torch.Tensor).
        """
        with torch.no_grad():
            # ---------------- Device + DType ---------------- #
            X = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=DTYPE)
            X = X.to(dtype=DTYPE, device=self.device)

            # ---------------- Context Handling ---------------- #
            prev_ctx = self._context
            ctx_list = []

            if context_ids is not None:
                if self.context_embedding is None:
                    raise ValueError("context_ids provided but context_embedding is None.")
                ctx_emb = self.context_embedding(context_ids.to(self.device, DTYPE))
                if ctx_emb.ndim == 3:
                    ctx_emb = ctx_emb.mean(dim=1)
                ctx_list.append(ctx_emb)

            if context is not None:
                ctx_list.append(context.to(dtype=DTYPE, device=self.device))

            ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
            if ctx is not None:
                self.set_context(ctx)

            # ---------------- Encode Observations ---------------- #
            theta = self.encode_observations(X)

            # ---------------- Contextual PDFs ---------------- #
            try:
                pdf = self._contextual_emission_pdf(X, theta)
            except Exception as e:
                print(f"[predict] emission PDF failed: {e}")
                pdf = self._params.get("emission_pdf", None)

            try:
                transition = self._contextual_transition_matrix(theta)
            except Exception as e:
                print(f"[predict] transition matrix failed: {e}")
                transition = getattr(self, "_transition_logits", None)

            try:
                duration = self._contextual_duration_pdf(theta)
            except Exception as e:
                print(f"[predict] duration PDF failed: {e}")
                duration = getattr(self, "_duration_logits", None)

            # ---------------- Backup & Assign Contextual Params ---------------- #
            prev_pdf = self._params.get("emission_pdf", None)
            prev_transition = getattr(self, "_transition_logits", None)
            prev_duration = getattr(self, "_duration_logits", None)

            if pdf is not None:
                self._params["emission_pdf"] = pdf
                if hasattr(pdf, "dof"):
                    self._dynamic_dof = pdf.dof  # update dynamic DOF if available

            if transition is not None:
                self._transition_logits = transition

            if duration is not None:
                self._duration_logits = duration

            # ---------------- Run Base HSMM Prediction ---------------- #
            preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

            # ---------------- Restore Previous Params ---------------- #
            self._params["emission_pdf"] = prev_pdf
            if prev_transition is not None:
                self._transition_logits = prev_transition
            elif hasattr(self, "_transition_logits"):
                del self._transition_logits
            if prev_duration is not None:
                self._duration_logits = prev_duration
            elif hasattr(self, "_duration_logits"):
                del self._duration_logits

            self._context = prev_ctx

            # ---------------- Ensure List[Tensor] ---------------- #
            if isinstance(preds, torch.Tensor):
                return [preds]
            if isinstance(preds, list):
                return [p if torch.is_tensor(p) else torch.as_tensor(p, device=self.device) for p in preds]
            return [torch.as_tensor(preds, device=self.device)]

    def decode(
        self,
        X: torch.Tensor,
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        algorithm: Literal["viterbi", "map"] = "viterbi",
    ) -> np.ndarray:
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        # Encode observations/context
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # Backup original parameters
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        # Temporarily assign context-aware parameters
        self._params["emission_pdf"] = self._contextual_emission_pdf(X, theta)
        self._transition_logits = self._contextual_transition_matrix(theta)
        self._duration_logits = self._contextual_duration_pdf(theta, scale=duration_weight)

        # Run prediction
        preds_list = self.predict(X, algorithm=algorithm)

        # Restore original parameters
        self._params["emission_pdf"] = prev_pdf
        if prev_transition is not None:
            self._transition_logits = prev_transition
        else:
            del self._transition_logits
        if prev_duration is not None:
            self._duration_logits = prev_duration
        else:
            del self._duration_logits

        # Flatten/concatenate batch
        if len(preds_list) == 1:
            out = preds_list[0]
        else:
            out = torch.cat(preds_list, dim=0)

        return out.detach().cpu().numpy()

    def tune(
        self,
        X: torch.Tensor,
        lengths: list[int] = None,
        configs: list[dict] = None,
        score_metric: str = "log_likelihood",
        verbose: bool = False
    ) -> dict:
        """
        GPU-batched hyperparameter tuning with vectorized normalization and type/shape checks.
        """

        if not configs:
            raise ValueError("configs must be provided as a list of dicts.")

        X = X.to(device=self.device, dtype=DTYPE)

        # --- sequence handling ---
        if hasattr(self, "to_observations"):
            obs_X = self.to_observations(X, lengths)
        else:
            obs_X = type("Obs", (), {"sequence": [X], "lengths": [len(X)], "log_probs": None})()

        B = len(obs_X.sequence)
        n_configs = len(configs)
        K, Dmax = self.n_states, self.max_duration
        scores: dict[int, float] = {}

        # --- prepare parameter stack ---
        param_keys = set().union(*[cfg.keys() for cfg in configs])
        param_stack = {k: [] for k in param_keys if k != "encoder_params"}

        for i, cfg in enumerate(configs):
            for k in list(param_stack.keys()):
                try:
                    val = cfg.get(k, getattr(self, k, None) or getattr(self, "_params", {}).get(k, None))
                    if val is None:
                        if verbose:
                            print(f"[Config {i}] skipping field '{k}' (unhandled)")
                        continue

                    if not isinstance(val, torch.Tensor):
                        val = torch.as_tensor(val, device=self.device, dtype=DTYPE)
                    else:
                        val = val.to(device=self.device, dtype=DTYPE)

                    param_stack[k].append(val)
                except Exception as e:
                    if verbose:
                        print(f"[Config {i}] skipping field '{k}' due to: {e}")
                    continue

        # --- stack tensors ---
        for k in list(param_stack.keys()):
            try:
                param_stack[k] = torch.stack(param_stack[k], dim=0).to(dtype=DTYPE, device=self.device)
            except Exception as e:
                if verbose:
                    print(f"[tune] skipping field '{k}' (stacking failed: {e})")
                param_stack.pop(k, None)

        # --- vectorized normalization for logits ---
        if "transition_logits" in param_stack:
            vals = param_stack["transition_logits"]
            if vals.ndim == 3 and vals.shape[1:] == (K, K):
                param_stack["transition_logits"] = vals - torch.logsumexp(vals, dim=-1, keepdim=True)

        if "duration_logits" in param_stack:
            vals = param_stack["duration_logits"]
            if vals.ndim == 3 and vals.shape[1:] == (K, Dmax):
                param_stack["duration_logits"] = vals - torch.logsumexp(vals, dim=-1, keepdim=True)

        # --- emission PDF check ---
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized for batched tuning.")

        seq_tensor = torch.cat(obs_X.sequence, dim=0).to(dtype=DTYPE, device=self.device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(obs_X.lengths, device=self.device, dtype=torch.int64), dim=0).tolist())

        try:
            lp_base = pdf.log_prob(seq_tensor.unsqueeze(1)).to(dtype=DTYPE)
            if lp_base.ndim > 2:
                lp_base = lp_base.sum(dim=list(range(2, lp_base.ndim)))
            all_log_probs = lp_base.unsqueeze(0).repeat(n_configs, 1, 1)
        except Exception as e:
            raise RuntimeError(f"Emission log-prob computation failed: {e}")

        # --- assign fields to model and compute scores ---
        with torch.no_grad():
            for i in range(n_configs):
                last_key, last_val = None, None
                try:
                    for k, vals in param_stack.items():
                        last_key, last_val = k, vals[i]
                        val = last_val

                        if hasattr(self, k):
                            try:
                                setattr(self, k, val)
                            except:
                                if verbose:
                                    print(f"[Config {i}] skipping field '{k}' (cannot set attribute)")
                        elif k in getattr(self, "_params", {}):
                            self._params[k] = val
                        else:
                            if verbose:
                                print(f"[Config {i}] skipping field '{k}' (unknown)")

                    obs_X.log_probs = [all_log_probs[i, seq_offsets[j]:seq_offsets[j+1]] for j in range(B)]
                    score_tensor = self._compute_log_likelihood(obs_X)
                    score = float(torch.nan_to_num(score_tensor, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")).sum().item())
                    scores[i] = score

                    if verbose:
                        print(f"[Config {i}] score={score:.4f}")

                except Exception as e:
                    print(f"[Config {i}] Internal tuning failed in field '{last_key or 'unknown'}': {e}")
                    if last_val is not None:
                        try:
                            print(f"  param '{last_key}' shape={tuple(last_val.shape)}, dtype={last_val.dtype}, device={last_val.device}")
                        except:
                            pass
                    scores[i] = float('-inf')

        return scores
