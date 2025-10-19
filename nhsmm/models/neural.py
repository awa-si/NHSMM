# nhsmm/models/neural.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Independent, Categorical, Poisson, LogNormal, MultivariateNormal

from typing import Literal, Optional, Union, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import numpy as np

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.models.hsmm import HSMM
from nhsmm.utilities import utils

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

class NeuralEmission(Emission):
    """Neural/contextual emission distribution for HSMM states."""

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
            self.mu = nn.Parameter(mu.to(device, DTYPE))
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            self.cov = nn.Parameter(cov.to(device, DTYPE))
        else:
            self.logits = nn.Parameter(params.to(device, DTYPE))

    # ---------------- Contextual Params ---------------- #
    def contextual_params(self, theta: Optional[torch.Tensor] = None):
        """Return context-modulated parameters."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        out = self.encoder(theta)
        if self.emission_type == "gaussian":
            mu, cov = out
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            return mu.to(self.device, DTYPE), cov.to(self.device, DTYPE)
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    # ---------------- Forward / Log-Prob ---------------- #
    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, log: bool = False):
        X = X.to(self.device, DTYPE)
        batch_mode = X.ndim == 3
        B, T, F = (X.shape if batch_mode else (1, *X.shape))
        K = self.n_states

        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            dist = MultivariateNormal(mu, covariance_matrix=cov)
            logp = dist.log_prob(X.unsqueeze(-2))  # broadcast over states
            return logp if log else logp.exp()
        else:
            logits = self.contextual_params(theta)
            probs = F.softmax(logits, dim=-1)
            if batch_mode:
                X_idx = X.long().view(B * T)
                logp = torch.log(probs[:, X_idx % self.n_features] + EPS).T.view(B, T, K)
            else:
                X_idx = X.long()
                logp = torch.log(probs[:, X_idx] + EPS).T
            return logp if log else logp.exp()

    def log_prob(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None):
        return self.forward(X, theta=theta, log=True)

    # ---------------- EM Update ---------------- #
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
                                  context_mode=self.context_mode, device=self.device, dtype=DTYPE)
        else:
            X_onehot = F.one_hot(X_flat.long(), num_classes=self.n_features).float()
            weighted_counts = posterior_flat.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder, context_mode=self.context_mode, device=self.device, dtype=DTYPE)

    # ---------------- Initialization ---------------- #
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
            logits = torch.distributions.Dirichlet(
                torch.ones(n_categories, device=device, dtype=DTYPE) * alpha
            ).sample([n_states]).clamp_min(EPS).log()
            return cls(emission_type.lower(), logits, encoder=encoder, device=device)

    def __repr__(self):
        if self.emission_type == "gaussian":
            return f"NeuralEmission(Gaussian, n_states={self.n_states}, n_features={self.n_features})"
        return f"NeuralEmission({self.emission_type.capitalize()}, n_states={self.n_states}, n_categories={self.n_features})"

class NeuralDuration(Duration):
    """Neural/contextual duration distribution for HSMM states."""

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

        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))

    # ---------------- Contextual Parameters ---------------- #
    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        X = getattr(context, "X", context)
        encoded = self.encoder(X) if self.encoder and X is not None else None

        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded, 2, dim=-1)
            else:
                mean, log_std = self.mean, self.std.log()
            return mean.squeeze(-1), log_std.exp().clamp_min(EPS).squeeze(-1)

    # ---------------- Forward / Probabilities ---------------- #
    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)

        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) \
                     - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 2 \
                     - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)

        probs = F.softmax(logits, dim=-1)

        if return_dist:
            if self.mode == "poisson":
                return [Poisson(rate[k]) for k in range(self.n_states)]
            else:
                return [LogNormal(mean[k], std[k]) for k in range(self.n_states)]

        return torch.log(probs) if log else probs

    # ---------------- Sampling ---------------- #
    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    # ---------------- Device Management ---------------- #
    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        if hasattr(self, "rate"):
            self.rate = nn.Parameter(self.rate.to(device))
        if hasattr(self, "mean"):
            self.mean = nn.Parameter(self.mean.to(device))
        if hasattr(self, "std"):
            self.std = nn.Parameter(self.std.to(device))
        if self.encoder:
            self.encoder.to(device)
        return self

class NeuralTransition(Transition):
    """Neural/contextual transition distribution for HSMM states."""

    def __init__(
        self,
        n_states: int,
        logits: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
        scale: float = 0.1,
    ):
        super().__init__(n_states=n_states, device=device)
        self.device = device or torch.device("cpu")
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = scale

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            self.logits = nn.Parameter(torch.zeros(n_states, n_states, device=self.device, dtype=DTYPE))

    # ---------------- Contextual Logits ---------------- #
    def contextual_logits(self, theta: Optional[torch.Tensor] = None):
        if self.encoder is None or theta is None:
            return self.logits

        X = getattr(theta, "X", theta)
        out = self.encoder(X)
        if isinstance(out, tuple):
            out = out[0]  # handle encoder returning (output, hidden)

        if out.ndim > 2:
            out = out.view(-1, self.n_states, self.n_states)

        return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    # ---------------- EM Update ---------------- #
    def update(self, posterior: torch.Tensor, inplace: bool = True):
        logits = torch.log((posterior + EPS).to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(
            n_states=self.n_states,
            logits=logits,
            encoder=self.encoder,
            context_mode=self.context_mode,
            device=self.device,
        )

    # ---------------- Initialization ---------------- #
    @classmethod
    def initialize(
        cls,
        n_states: int,
        alpha: float = 1.0,
        batch: int = 1,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        probs = torch.distributions.Dirichlet(
            torch.ones(n_states, device=device, dtype=DTYPE) * alpha
        ).sample([batch])

        if batch == 1:
            probs = probs.squeeze(0)

        return cls(
            n_states=n_states,
            logits=torch.log(probs.clamp_min(EPS)),
            encoder=encoder,
            device=device,
        )

    # ---------------- Device Management ---------------- #
    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        if hasattr(self, "logits"):
            self.logits = nn.Parameter(self.logits.to(device))
        if self.encoder:
            self.encoder.to(device)
        return self

class NeuralHSMM(HSMM, nn.Module):
    """ NeuralHSMM with EM, Viterbi, context, and gradient support.
    """

    def __init__(self, config: NHSMMConfig):
        
        self.config = config
        nn.Module.__init__(self)

        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = config.encoder.to(self.device, dtype=torch.float64) if config.encoder else None
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
            self.ctx_transition = nn.Linear(config.context_dim, config.n_states**2, device=self.device, dtype=torch.float64)
            self.ctx_duration = nn.Linear(config.context_dim, config.n_states*config.max_duration, device=self.device, dtype=torch.float64)
            self.ctx_emission = nn.Linear(config.context_dim, config.n_states*config.n_features, device=self.device, dtype=torch.float64)
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
        nS, nD, nF = self.n_states, self.max_duration, self.n_features
        dof = (nS-1) + nS*(nS-1) + nS*(nD-1)
        pdf = self.pdf
        if pdf is not None:
            if isinstance(pdf, Categorical): dof += nS*(pdf.logits.shape[1]-1)
            elif isinstance(pdf, MultivariateNormal): dof += nS*(2*nF)
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
        scale = scale or self.config.emission_scale

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
                if d.ndim == 2 and d.shape[1] == K * F:
                    d = d.view(-1, K, F)
                    delta = scale * torch.tanh(d.mean(dim=0))
                elif d.shape == (K, F):
                    delta = scale * torch.tanh(d)
                else:
                    raise ValueError(f"Unexpected ctx_emission output shape {d.shape}")
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
            probs = (probs + delta.mean(dim=0)).clamp_min(EPS)  # delta is additive adjustment
            probs = probs / probs.sum()
            return Categorical(probs=probs.repeat(K, 1).to(dtype=DTYPE, device=self.device))

        # --- Gaussian emission ---
        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0).clamp_min(EPS)  # (K,)
            mean = (posterior.T @ X) / Nk.unsqueeze(1)  # (K, F)
            diff = X.unsqueeze(1) - mean.unsqueeze(0)   # (T, K, F)
            weighted = diff * posterior.unsqueeze(-1)  # weight residuals
            cov = torch.einsum("tkf,tkh->kfh", weighted, diff) / Nk.unsqueeze(1).unsqueeze(-1)
            cov += torch.eye(F, dtype=DTYPE, device=self.device).unsqueeze(0) * self.min_covar
            mean = mean + delta
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def initialize_emissions(self, X, method: str = "moment"):
        """Initialize emission parameters from raw data X."""
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=DTYPE, device=self.device)
        X = X.to(dtype=DTYPE, device=self.device)
        K = self.n_states

        if self.emission_type == "gaussian":
            if X.ndim != 2:
                raise ValueError(f"Gaussian emissions require X shape (T,F), got {X.shape}")
            T, F = X.shape

            if method == "moment":
                mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)
            elif method == "kmeans":
                from sklearn.cluster import KMeans
                labels = torch.tensor(KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X.cpu().numpy()), device=self.device)
                mu_init = torch.stack([X[labels==k].mean(dim=0) if (labels==k).any() else X.mean(dim=0) for k in range(K)])
                var_init = torch.stack([X[labels==k].var(dim=0, unbiased=False) if (labels==k).any() else X.var(dim=0, unbiased=False) for k in range(K)])
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            var_init = var_init.clamp_min(self.min_covar)
            cov = torch.diag_embed(var_init)

            self.emission_module.mu = nn.Parameter(mu_init)
            self.emission_module.cov = nn.Parameter(cov)
            self._params["emission_pdf"] = MultivariateNormal(mu_init, covariance_matrix=cov)

        elif self.emission_type in ("categorical", "bernoulli"):
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                counts = torch.bincount(X.squeeze(-1).long(), minlength=self.n_features).to(dtype=DTYPE, device=self.device)
            else:
                counts = X.sum(dim=0).to(dtype=DTYPE, device=self.device)
            probs = counts / counts.sum()
            logits = probs.clamp_min(EPS).log().unsqueeze(0).repeat(K, 1)

            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.cov = nn.Parameter(torch.zeros_like(logits))
            self._params["emission_pdf"] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

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
        if theta is None and self._context is None:
            return None

        theta = theta.to(self.device, DTYPE) if theta is not None else None
        ctx = self._context.to(self.device, DTYPE) if self._context is not None else None

        if theta is None:
            return ctx
        if ctx is None:
            return theta

        # Align batch dimensions
        if theta.shape[0] != ctx.shape[0]:
            if theta.shape[0] == 1:
                theta = theta.expand(ctx.shape[0], -1)
            elif ctx.shape[0] == 1:
                ctx = ctx.expand(theta.shape[0], -1)
            else:
                raise ValueError(f"Batch mismatch: theta {theta.shape} vs context {ctx.shape}")

        return torch.cat([theta, ctx], dim=-1)

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor], scale: float = None):
        """Return contextually-shifted emission distribution."""
        scale = scale or self.config.emission_scale
        pdf = self.pdf
        if pdf is None:
            return None

        combined = self._combine_context(theta)
        if combined is None:
            return pdf

        # Determine delta shift
        delta = self.ctx_emission(combined) if self.ctx_emission is not None else combined[:, :self.n_states * self.n_features]
        if delta.ndim == 2:
            delta = delta.view(delta.shape[0], self.n_states, self.n_features)
            delta_mean = delta.mean(dim=0)
        else:
            delta_mean = delta.view(self.n_states, self.n_features) if delta.ndim == 1 else delta.mean(dim=0)

        # Apply delta
        if isinstance(pdf, Categorical):
            logits = getattr(pdf, "logits", None)
            if logits is None:
                raise RuntimeError("Categorical pdf missing logits.")
            return Categorical(logits=(logits + scale * torch.tanh(delta_mean)).to(self.device, DTYPE))

        if isinstance(pdf, MultivariateNormal):
            new_mean = (pdf.mean + scale * torch.tanh(delta_mean)).to(self.device, DTYPE)
            cov = pdf.covariance_matrix.to(self.device, DTYPE)
            return MultivariateNormal(loc=new_mean, covariance_matrix=cov)

        raise TypeError(f"Unsupported PDF type: {type(pdf)}")

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor], scale: float = None, temperature: float = None):
        """Return (n_states, max_duration) duration probabilities modulated by context."""
        scale = scale or self.config.duration_scale
        temperature = temperature or self.config.duration_temp
        logits = getattr(self.duration_module, "logits", None)

        if logits is None:
            return self.duration_module.forward(log=False)

        combined = self._combine_context(theta)
        if combined is None or self.ctx_duration is None:
            return F.softmax(logits.to(self.device, DTYPE) / temperature, dim=-1)

        delta = self.ctx_duration(combined)
        if delta.ndim == 2 and delta.shape[1] == self.n_states * self.max_duration:
            delta = delta.view(-1, self.n_states, self.max_duration).mean(dim=0)
        else:
            delta = delta.view(self.n_states, self.max_duration)

        return F.softmax((logits.to(self.device, DTYPE) + scale * torch.tanh(delta)) / temperature, dim=-1)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor], scale: float = None, temperature: float = None):
        """Return (n_states, n_states) transition matrix modulated by context."""
        scale = scale or self.config.transition_scale
        temperature = temperature or self.config.transition_temp
        logits = getattr(self.transition_module, "logits", None)

        if logits is None:
            return self.transition_module.forward(log=False)

        combined = self._combine_context(theta)
        if combined is None or self.ctx_transition is None:
            return F.softmax(logits.to(self.device, DTYPE) / temperature, dim=-1)

        delta = self.ctx_transition(combined)
        if delta.ndim == 2 and delta.shape[1] == self.n_states * self.n_states:
            delta = delta.view(-1, self.n_states, self.n_states).mean(dim=0)
        else:
            delta = delta.view(self.n_states, self.n_states)

        return F.softmax((logits.to(self.device, DTYPE) + scale * torch.tanh(delta)) / temperature, dim=-1)

    def encode_observations(
        self,
        X: torch.Tensor,
        detach: bool = True,
        keep_sequence: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Encode observations using the attached encoder.

        Args:
            X: Tensor of shape [T, F] or [B, T, F].
            detach: If True, detaches the output from the computation graph.
            keep_sequence: If True, keep sequence dimension [B, T, H]; otherwise collapse to [B, H].

        Returns:
            Encoded tensor of shape [B, H] or [B, T, H], or None if no encoder is attached.
        """
        if self.encoder is None:
            return None

        # Ensure batch dimension
        if X.ndim == 2:
            X = X.unsqueeze(0)  # [1, T, F]

        inp = X.to(dtype=DTYPE, device=self.device)

        # Encode
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]

        if detach:
            out = out.detach()
        out = out.to(dtype=DTYPE, device=self.device)

        # Handle sequence dimension
        if out.ndim == 3:  # [B, T, H]
            if not keep_sequence:
                out = out.mean(dim=1)  # collapse to [B, H]
        elif out.ndim == 2:  # [B, H], keep as-is
            if keep_sequence:
                out = out.unsqueeze(1)  # add singleton sequence dimension [B, 1, H]
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
        Top-level forward pass: merge context, encode observations, optionally return emission PDF.

        Args:
            X: Observation tensor [T, n_features] or [batch, T, n_features].
            return_pdf: If True, returns the contextually-modulated emission PDF.
            context: Optional raw context features.
            context_ids: Optional context IDs to embed and merge with raw context.

        Returns:
            Theta tensor or emission PDF, depending on return_pdf.
        """
        prev_ctx = self._context
        ctx_list = []

        # Compute context embedding if applicable
        if context_ids is not None:
            if self.context_embedding is None:
                raise ValueError("context_ids provided but no context_embedding defined.")
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:
                ctx_emb = ctx_emb.mean(dim=1)
            ctx_list.append(ctx_emb.to(dtype=DTYPE))

        # Include raw context if provided
        if context is not None:
            ctx_list.append(context.to(device=self.device, dtype=DTYPE))

        # Merge contexts if any
        ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
        if ctx is not None:
            self.set_context(ctx)

        # Encode observations
        theta = self.encode_observations(X)

        if return_pdf:
            pdf = self._contextual_emission_pdf(X, theta)
            self._context = prev_ctx
            return pdf

        self._context = prev_ctx
        return theta

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
        Context-aware prediction wrapper.

        Args:
            X: Observations (T x features) or batch of sequences (B x T x features)
            lengths: Optional sequence lengths for batched sequences
            algorithm: "viterbi" or "map"
            context: Optional context tensor (B x T x C)
            context_ids: Optional context IDs (for embedding lookups)
            batch_size: Batch size for prediction

        Returns:
            List of predicted state sequences (torch.Tensor)
        """
        # Ensure tensor on correct device
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        # Encode context if present
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # Compute context-aware parameters
        pdf = self._contextual_emission_pdf(X, theta)
        transition = self._contextual_transition_matrix(theta)
        duration = self._contextual_duration_pdf(theta)

        # Backup original HSMM internals
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition_logits = getattr(self, "_transition_logits", None)
        prev_duration_logits = getattr(self, "_duration_logits", None)

        # Assign temporary parameters for prediction
        if pdf is not None:
            self._params["emission_pdf"] = pdf

        if transition is not None:
            if isinstance(transition, torch.Tensor) and transition.shape == (self.n_states, self.n_states):
                self._transition_logits = transition
            else:
                print(f"[predict] Warning: transition shape {transition.shape} unexpected, skipping assignment.")

        if duration is not None:
            if isinstance(duration, torch.Tensor) and duration.shape == (self.n_states, self.max_duration):
                self._duration_logits = duration
            else:
                print(f"[predict] Warning: duration shape {duration.shape} unexpected, skipping assignment.")

        # Run base HSMM prediction
        preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

        # Restore previous parameters
        self._params["emission_pdf"] = prev_pdf
        if prev_transition_logits is not None:
            self._transition_logits = prev_transition_logits
        else:
            if hasattr(self, "_transition_logits"):
                del self._transition_logits
        if prev_duration_logits is not None:
            self._duration_logits = prev_duration_logits
        else:
            if hasattr(self, "_duration_logits"):
                del self._duration_logits

        # Ensure output is always a list of tensors
        if isinstance(preds, torch.Tensor):
            return [preds]
        elif isinstance(preds, list):
            return [p if torch.is_tensor(p) else torch.as_tensor(p, device=self.device) for p in preds]
        else:
            return [torch.as_tensor(preds, device=self.device)]

    def decode(
        self,
        X: torch.Tensor,
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        algorithm: Literal["viterbi", "map"] = "viterbi",
    ) -> np.ndarray:
        """
        Context-aware decoding wrapper.
        Encodes observations, applies duration weighting, and returns numpy predictions.

        Args:
            X: Observations (T x features) or batch (B x T x features)
            duration_weight: Weighting factor for duration probabilities
            context: Optional context tensor
            context_ids: Optional context ID tensor
            algorithm: "viterbi" or "map"

        Returns:
            numpy.ndarray of predicted states
        """
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        # Forward pass to get encoded context
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # Apply duration weighting
        _ = self._contextual_transition_matrix(theta, scale=duration_weight)
        _ = self._contextual_duration_pdf(theta, scale=duration_weight)
        _ = self._contextual_emission_pdf(X, theta)

        # Use improved predict
        preds_list = self.predict(X, algorithm=algorithm, context=theta, context_ids=context_ids)

        # Flatten or unify output to numpy
        if len(preds_list) == 1:
            out = preds_list[0]
        else:
            out = torch.stack(preds_list)

        return out.detach().cpu().numpy()
