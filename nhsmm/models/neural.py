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
            self._categorical_proj = None

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
        base = self.mu.numel() + getattr(self, "cov", torch.tensor([])).numel() if self.emission_type == "gaussian" else self.logits.numel()
        adapters = sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in
                       [self.encoder, self.temporal_adapter, self.spatial_adapter] if m)
        return base + adapters

    def _apply_context(self, theta: Optional[torch.Tensor]):
        if theta is None:
            return None
        out = theta
        if self.encoder:
            out = self.encoder(out)
            if isinstance(out, tuple):
                out = out[0]
        if self.temporal_adapter and out.ndim == 3:  # (B,T,F)
            out = self.temporal_adapter(out.transpose(1, 2)).transpose(1, 2)
        if self.spatial_adapter:
            out = self.spatial_adapter(out)
        return out

    def contextual_params(self, theta: Optional[torch.Tensor] = None):
        out = self._apply_context(theta)
        if self.emission_type == "gaussian":
            if out is None:
                mu, cov = self.mu, self.cov
            else:
                proj = self._gaussian_proj(out)
                mu_param, cov_param = proj.chunk(2, dim=-1)
                B = mu_param.shape[0]
                mu = mu_param.view(B, self.n_states, self.n_features)
                cov = F.softplus(cov_param.view(B, self.n_states, self.n_features)) + EPS
                cov = torch.diag_embed(cov)
            return mu.to(self.device, DTYPE), cov.to(self.device, DTYPE)
        else:
            logits = out if out is not None else self.logits
            if logits.ndim == 2 and theta is not None:
                logits = logits.unsqueeze(0).expand(theta.shape[0], -1, -1)
            return logits

    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, log: bool = False):
        X = X.to(self.device, DTYPE)
        batch_mode = X.ndim == 3
        B, T, Fdim = (X.shape if batch_mode else (1, *X.shape))
        K = self.n_states

        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            cov = cov + EPS * torch.eye(Fdim, device=self.device, dtype=DTYPE).unsqueeze(0).expand(mu.shape[0], -1, -1, -1)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            logp = dist.log_prob(X.unsqueeze(-2))
        else:
            logits = self.contextual_params(theta)
            logp = F.log_softmax(logits, dim=-1)
            X_idx = X.long()
            if batch_mode:
                idx = X_idx.unsqueeze(-2).expand(-1, K, -1)
                logp = torch.gather(logp.exp(), -1, idx).clamp_min(EPS).log().transpose(1, 2)
            else:
                logp = logp[:, X_idx].clamp_min(EPS).log().T

        return logp if log else logp.exp()

    def log_prob(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None):
        return self.forward(X, theta=theta, log=True)

    def sample(self, n_samples: int = 1, theta: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            cov = cov + EPS * torch.eye(mu.shape[-1], device=self.device, dtype=DTYPE).unsqueeze(0)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            samples = dist.rsample((n_samples,)).transpose(0, 1)
        else:
            logits = self.contextual_params(theta)
            dist = Categorical(logits=logits)
            samples = dist.sample((n_samples,)).transpose(0, 1)

        if state_indices is not None:
            samples = samples[:, state_indices] if samples.ndim == 3 else samples[state_indices]
        return samples

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, DTYPE)
        X = X.to(self.device, DTYPE)
        X_flat = X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X
        posterior_flat = posterior.reshape(-1, posterior.shape[-1]) if posterior.ndim > 2 else posterior

        if self.emission_type == "gaussian":
            counts = posterior_flat.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (posterior_flat.T @ X_flat) / counts
            diff = X_flat.unsqueeze(1) - mu_new.unsqueeze(0)
            cov_new = torch.einsum('tkf,tkh->kfh', diff * posterior_flat.unsqueeze(-1), diff) / counts.squeeze(-1)
            cov_new += torch.eye(self.n_features, device=self.device, dtype=DTYPE).unsqueeze(0) * EPS
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
    """Neural/contextual duration distribution with batch support and adapters."""

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

        self.temporal_adapter = (
            nn.Conv1d(n_states, n_states, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if self.context_mode == "temporal" else None
        )
        self.spatial_adapter = (
            nn.Linear(n_states, n_states, bias=False).to(self.device, DTYPE)
            if self.context_mode == "spatial" else None
        )

        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))

        self.register_buffer("_durations", torch.arange(1, max_duration + 1, device=self.device, dtype=DTYPE))

    @property
    def dof(self) -> int:
        base = self.rate.numel() if self.mode == "poisson" else self.mean.numel() + self.std.numel()
        adapters = sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in
                       [self.encoder, self.temporal_adapter, self.spatial_adapter] if m)
        return base + adapters

    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        X = context.to(self.device, DTYPE) if context is not None else None
        encoded = self.encoder(X) if self.encoder and X is not None else None
        if encoded is not None:
            if self.temporal_adapter and encoded.ndim == 3:
                encoded = encoded.transpose(1, 2)
                encoded = self.temporal_adapter(encoded)
                encoded = encoded.transpose(1, 2)
            if self.spatial_adapter:
                encoded = self.spatial_adapter(encoded)

        if self.mode == "poisson":
            rate = F.softplus(encoded).view(-1, self.n_states) if encoded is not None else self.rate.view(1, -1)
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded, 2, dim=-1)
            else:
                mean, log_std = self.mean.view(1, -1), self.std.view(1, -1).log()
            return mean.to(self.device, DTYPE), log_std.exp().clamp_min(EPS)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        durations = self._durations
        D = durations.shape[0]
        K = self.n_states

        if self.mode == "poisson":
            rate = self._contextual_params(context)
            B = rate.shape[0]
            logits = rate.unsqueeze(-1) * torch.log(durations.unsqueeze(0)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            B = mean.shape[0]
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 2
            logits = logits - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)

        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits, dim=-1)

        if return_dist:
            if self.mode == "poisson":
                return torch.poisson(probs * D)
            else:
                return LogNormal(mean.unsqueeze(-1), std.unsqueeze(-1))

        return torch.log(probs) if log else probs

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        if self.mode == "poisson":
            rate = self._contextual_params(context)
            B, K = rate.shape
            samples = torch.poisson(rate.unsqueeze(-1).expand(B, K, n_samples))
        else:
            mean, std = self._contextual_params(context)
            B, K = mean.shape
            dist = LogNormal(mean.unsqueeze(-1), std.unsqueeze(-1))
            samples = dist.rsample((n_samples,)).transpose(0, 1)
        if state_indices is not None:
            samples = samples[:, state_indices] if samples.ndim == 3 else samples[state_indices]
        return samples

    def log_prob(self, X: torch.Tensor, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, log=True)


class NeuralTransition(Transition):
    """Neural/contextual transition distribution with batch-safe adapters."""

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
        self.temperature = max(temperature, 1e-6)

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            uniform = torch.full((n_states, n_states), 1.0 / n_states, device=self.device, dtype=DTYPE)
            self.logits = nn.Parameter(torch.log(uniform.clamp_min(EPS)))

        self.temporal_adapter = (
            nn.Conv1d(n_states, n_states, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if self.context_mode == "temporal" else None
        )
        self.spatial_adapter = (
            nn.Linear(n_states, n_states, bias=False).to(self.device, DTYPE)
            if self.context_mode == "spatial" else None
        )

    @property
    def dof(self) -> int:
        dof = self.n_states * (self.n_states - 1)
        adapters = sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in
                       [self.encoder, self.temporal_adapter, self.spatial_adapter] if m)
        return dof + adapters

    def contextual_logits(self, theta: Optional[torch.Tensor] = None):
        B = theta.shape[0] if theta is not None else 1
        base_logits = self.logits.unsqueeze(0).expand(B, self.n_states, self.n_states)

        if self.encoder and theta is not None:
            out = self.encoder(theta)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 2:
                assert out.shape[1] == self.n_states ** 2
                out = out.view(B, self.n_states, self.n_states)
            elif out.ndim == 3 and out.shape[1:] != (self.n_states, self.n_states):
                out = out.view(B, self.n_states, self.n_states)
            base_logits = base_logits + torch.tanh(out) * self.scale

        if self.temporal_adapter and theta is not None and theta.ndim == 3:
            delta = self.temporal_adapter(theta.transpose(1, 2)).transpose(1, 2)
            delta = delta.mean(dim=1) if delta.shape[1] != self.n_states else delta
            base_logits = base_logits + delta

        if self.spatial_adapter and theta is not None:
            delta = self.spatial_adapter(theta)
            if delta.ndim == 2:
                delta = delta.unsqueeze(-1).expand(-1, -1, self.n_states)
            base_logits = base_logits + delta

        return F.log_softmax(torch.nan_to_num(base_logits) / self.temperature, dim=-1)

    def forward(self, theta: Optional[torch.Tensor] = None):
        return self.contextual_logits(theta)

    def sample(self, n_samples: int = 1, theta: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        logits = self.contextual_logits(theta)
        B, K, _ = logits.shape
        dist = Categorical(logits=logits)
        samples = dist.sample((n_samples,)).transpose(0, 1).transpose(1, 2)
        if state_indices is not None:
            samples = samples[:, state_indices, :]
        return samples

    def update(self, posterior: torch.Tensor, inplace: bool = True):
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
        logits = torch.log(posterior.to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(
            n_states=self.n_states,
            logits=logits,
            encoder=self.encoder,
            context_mode=self.context_mode,
            device=self.device,
            scale=self.scale,
            temperature=self.temperature,
        )


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
    n_context_states: Optional[int] = 0

class NeuralHSMM(HSMM):
    """ NeuralHSMM with EM, Viterbi, context, and gradient support.
    """

    def __init__(self, config: NHSMMConfig):
        self.config = config
        self.device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = DTYPE
        nn.Module.__init__(self)

        # Optional encoder
        self.encoder = config.encoder.to(self.device, dtype=self.dtype) if config.encoder else None
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
            context_mode="temporal" if config.context_dim > 0 else "none",
            scale=config.duration_scale,
            device=self.device,
        )
        self.transition_module = NeuralTransition(
            n_states=config.n_states,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim > 0 else "none",
            scale=config.transition_scale,
            device=self.device,
        )

        # Helper to init context linear layers
        def _init_ctx_linear(out_dim: int) -> nn.Linear:
            lin = nn.Linear(config.context_dim, out_dim, device=self.device, dtype=self.dtype)
            nn.init.normal_(lin.weight, 0.0, 1e-3)
            nn.init.normal_(lin.bias, 0.0, 1e-3)
            return lin

        # Context embeddings
        if config.n_context_states > 0 and config.context_dim > 0:
            self.context_embedding = nn.Embedding(config.n_context_states, config.context_dim)
            nn.init.normal_(self.context_embedding.weight, 0.0, 1e-3)

            self.ctx_transition = _init_ctx_linear(config.n_states ** 2)
            self.ctx_duration = _init_ctx_linear(config.n_states * config.max_duration)
            self.ctx_emission = _init_ctx_linear(config.n_states * config.n_features)
        else:
            self.context_embedding = None
            self.ctx_transition = None
            self.ctx_duration = None
            self.ctx_emission = None

        # Params dictionary
        self._params: dict = {"emission_type": config.emission_type}

        # Initialize HSMM base
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

        # Sample initial emission PDF
        self._params["emission_pdf"] = self.sample_emission_pdf(
            temperature=config.emission_temp,
            scale=config.emission_scale,
        )

    @property
    def dof(self) -> int:
        """
        Compute degrees of freedom (dof) for the full HSMM model, including:
          - initial state probabilities
          - transition probabilities
          - duration probabilities
          - emission parameters
          - context embeddings and neural context modulators
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
                dof += nS * (pdf.logits.shape[-1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                dof += nS * (nF + nF)  # mean + diagonal cov
            else:
                import warnings
                warnings.warn(f"Unknown pdf type {type(pdf)}, emission dof not counted")

        # Context embeddings
        if getattr(self, "context_embedding", None):
            ctx_emb: nn.Embedding = self.context_embedding
            dof += ctx_emb.weight.numel()

        # Context linear modulators
        for ctx_layer_name in ("ctx_emission", "ctx_transition", "ctx_duration"):
            ctx_layer = getattr(self, ctx_layer_name, None)
            if ctx_layer is not None:
                dof += ctx_layer.weight.numel() + ctx_layer.bias.numel()

        # Neural modules
        for module_name in ("emission_module", "transition_module", "duration_module"):
            module = getattr(self, module_name, None)
            if module is not None:
                dof += sum(p.numel() for p in module.parameters() if p.requires_grad)

        return dof

    def _prepare_delta(
        self,
        delta: Optional[torch.Tensor],
        shape: Tuple[int, int, int],
        scale: float = 0.1,
        broadcast: bool = True
    ) -> torch.Tensor:
        """
        Vectorized preparation of context delta for emission modulation.

        Args:
            delta: Optional input tensor [B, ...] or flattened.
            shape: Target shape (B_target, n_states, feature_dim)
            scale: Scaling factor
            broadcast: Whether to broadcast along batch dimension if needed.

        Returns:
            Tensor of shape [B_target, n_states, feature_dim]
        """

        B_target, n_states, F_dim = shape
        device, dtype = self.device, DTYPE

        # Default zero delta if None
        if delta is None:
            return torch.zeros((B_target, n_states, F_dim), dtype=dtype, device=device)

        delta = delta.to(device=device, dtype=dtype)

        # Flatten all but batch dimension
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        elif delta.ndim > 2:
            delta = delta.reshape(delta.shape[0], -1)

        B, total_features = delta.shape
        expected_features = n_states * F_dim

        # Pad or truncate to match expected features
        if total_features < expected_features:
            delta = F.pad(delta, (0, expected_features - total_features))
        elif total_features > expected_features:
            delta = delta[..., :expected_features]

        # Reshape to [B, n_states, F_dim] safely
        delta = delta.reshape(B, n_states, F_dim)

        # Clamp NaNs/Infs and scale
        delta = scale * torch.tanh(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0))

        # Broadcast batch dimension if needed
        if broadcast:
            if B == 1:
                delta = delta.expand(B_target, -1, -1)
            elif B < B_target:
                # Repeat instead of truncating silently
                repeats = (B_target + B - 1) // B
                delta = delta.repeat(repeats, 1, 1)[:B_target]

        return delta

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        scale: float = None,
    ):
        """
        Sample emission distribution conditioned on optional observations X [B, T, F] and context theta [B, H].

        Args:
            X: Optional observations [B, T, F].
            theta: Optional context tensor [B, H] or [H].
            temperature: Softmax temperature for categorical emissions.
            scale: Context modulation scale.

        Returns:
            Batched Categorical or MultivariateNormal distribution over states.
        """
        nS, nF = self.n_states, self.n_features
        scale = scale or getattr(self.config, "emission_scale", 1.0)
        device, dtype = self.device, DTYPE

        # Determine batch size
        B = 1
        if X is not None:
            B = X.shape[0]
            X = X.to(dtype=dtype, device=device)
        if theta is not None:
            t = theta.to(dtype=dtype, device=device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            B = max(B, t.shape[0])
        else:
            t = None

        # Compute context delta: [B, nS, nF]
        delta = None
        if t is not None:
            if getattr(self, "ctx_emission", None):
                delta = self.ctx_emission(t)
            else:
                delta = t[..., : nS * nF]
        delta = self._prepare_delta(delta, shape=(B, nS, nF), scale=scale)

        # Categorical emission
        if self.emission_type.lower() == "categorical":
            if X is not None:
                # Compute mean over T for each feature, broadcast to states
                probs = X.mean(dim=1, keepdim=True).expand(B, nS, nF)
            else:
                probs = torch.full((B, nS, nF), 1.0 / nF, dtype=dtype, device=device)

            probs = probs + delta
            probs = probs / max(temperature, 1e-6)
            logits = probs - torch.logsumexp(probs, dim=-1, keepdim=True)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return Categorical(logits=logits)

        # Gaussian emission
        elif self.emission_type.lower() == "gaussian":
            if X is not None:
                mean = X.mean(dim=1, keepdim=True).expand(B, nS, nF)
                var = X.var(dim=1, unbiased=False, keepdim=True).expand(B, nS, nF).clamp_min(self.min_covar)
            else:
                mean = torch.zeros(B, nS, nF, dtype=dtype, device=device)
                var = torch.full((B, nS, nF), self.min_covar, dtype=dtype, device=device)

            mean = mean + delta
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def initialize_emissions(self, X, method: str = "moment", theta: Optional[torch.Tensor] = None, scale: float = None):
        """
        Initialize emission parameters for Gaussian or Categorical/Bernoulli data,
        optionally modulated by context theta.

        Args:
            X: Observations ([T,F], [B,T,F], [T], [B,T], or multi-hot [B,T,F])
            method: "moment" or "kmeans" for Gaussian
            theta: Optional context tensor [H] or [B,H]
            scale: Modulation scale for context
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=DTYPE, device=self.device)
        X = X.to(dtype=DTYPE, device=self.device)

        K, F = self.n_states, self.n_features
        scale = scale or getattr(self.config, "emission_scale", 1.0)

        # --- Prepare context delta ---
        delta = None
        if theta is not None:
            t = theta.to(dtype=DTYPE, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            B = t.shape[0]
            if getattr(self, "ctx_emission", None):
                delta = self.ctx_emission(t)
            else:
                delta = t[..., : K * F]
            delta = self._prepare_delta(delta, shape=(B, K, F), scale=scale)
        else:
            B = 1

        # --- Gaussian ---
        if self.emission_type == "gaussian":
            if X.ndim == 3:
                X_flat = X.reshape(-1, F)
            elif X.ndim == 2:
                X_flat = X
            else:
                raise ValueError(f"Gaussian emissions require [T,F] or [B,T,F], got {X.shape}")

            if method == "moment":
                mu_init = X_flat.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X_flat.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)
            elif method == "kmeans":
                from sklearn.cluster import KMeans
                labels = torch.tensor(
                    KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X_flat.cpu().numpy()),
                    device=self.device
                )
                counts = torch.bincount(labels, minlength=K).clamp_min(1).to(dtype=DTYPE, device=self.device)
                mu_init = torch.zeros(K, F, dtype=DTYPE, device=self.device)
                mu_init.scatter_add_(0, labels.unsqueeze(-1).expand(-1, F), X_flat)
                mu_init /= counts.unsqueeze(-1)
                diff = X_flat - mu_init[labels]
                var_init = torch.zeros(K, F, dtype=DTYPE, device=self.device)
                var_init.scatter_add_(0, labels.unsqueeze(-1).expand(-1, F), diff**2)
                var_init /= counts.unsqueeze(-1)
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            var_init = var_init.clamp_min(getattr(self, "min_covar", 1e-6))
            cov = torch.diag_embed(var_init)

            # Apply context delta if available
            if delta is not None:
                if B == 1:
                    mu_init = mu_init + delta[0]
                else:
                    # Expand mu_init to match batch
                    mu_init = mu_init.unsqueeze(0).expand(B, K, F) + delta
                cov = cov.unsqueeze(0).expand(B, K, F, F)  # keep cov consistent with batch

            self.emission_module.mu = nn.Parameter(mu_init.to(self.device, DTYPE))
            self.emission_module.cov = nn.Parameter(cov.to(self.device, DTYPE))
            self._params["emission_pdf"] = MultivariateNormal(mu_init, covariance_matrix=cov)

        # --- Categorical / Bernoulli ---
        elif self.emission_type in ("categorical", "bernoulli"):
            if X.ndim == 1:
                counts = torch.bincount(X.long(), minlength=F).to(dtype=DTYPE, device=self.device)
            elif X.ndim == 2:
                counts = X.sum(dim=0).to(dtype=DTYPE, device=self.device)
            elif X.ndim == 3:
                counts = X.sum(dim=(0, 1)).to(dtype=DTYPE, device=self.device)
            else:
                raise ValueError(f"Unsupported input shape {X.shape} for categorical emissions")

            probs = counts / counts.sum() if counts.sum() > 0 else torch.full((F,), 1.0 / F, device=self.device, dtype=DTYPE)
            logits = probs.clamp_min(EPS).log().unsqueeze(0).repeat(K, 1)

            # Apply context delta as additive logits
            if delta is not None:
                if B == 1:
                    logits = logits + delta[0]
                else:
                    logits = logits.unsqueeze(0).expand(B, K, F) + delta

            self._params["emission_pdf"] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> Distribution:
        """
        Estimate emission distribution from weighted data (posterior responsibilities),
        modulated per sequence by context theta.

        Args:
            X: Observations [B*T, F] or [B, T, F].
            posterior: Responsibilities [B*T, K] or [B, T, K].
            theta: Optional context [B, H].
            scale: Context modulation scale.

        Returns:
            MultivariateNormal or Categorical, batch-aware.
        """
        K, F = self.n_states, self.n_features
        device, dtype = self.device, DTYPE
        scale = scale or getattr(self.config, "emission_scale", 0.1)

        # Ensure X/posterior are tensors on the correct device
        X = X.to(dtype=dtype, device=device)
        posterior = posterior.to(dtype=dtype, device=device)

        # Determine batch size
        B = 1
        if X.ndim == 3:  # [B, T, F]
            B, T, _ = X.shape
            X_flat = X.reshape(B*T, F)
            posterior_flat = posterior.reshape(B*T, K)
        else:
            X_flat = X
            posterior_flat = posterior
            T = X_flat.shape[0]

        # --- Context delta per batch ---
        delta = None
        if theta is not None:
            combined = self.combine_context(theta)  # [B, H]
            if combined is not None:
                delta = self.ctx_emission(combined) if getattr(self, "ctx_emission", None) else combined[..., :K*F]
        delta = self._prepare_delta(delta, shape=(B, K, F), scale=scale, broadcast=False)  # keep batch dimension

        # --- Categorical emission ---
        if self.emission_type == "categorical":
            # Flatten batch/time to compute counts per sequence
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                X_flat = X_flat.squeeze(-1).long()
            else:
                X_flat = X_flat.long()

            counts = torch.zeros(B, K, F, device=device, dtype=dtype)
            # Compute weighted counts using posterior
            for b in range(B):
                start = b*T
                end = (b+1)*T
                counts[b] = (posterior_flat[start:end].T @ F.one_hot(X_flat[start:end], num_classes=F).to(dtype=dtype, device=device))

            probs = counts + delta  # apply context per batch
            probs = probs.clamp_min(EPS)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return Categorical(probs=probs)

        # --- Gaussian emission ---
        elif self.emission_type == "gaussian":
            mean = torch.zeros(B, K, F, device=device, dtype=dtype)
            cov = torch.zeros(B, K, F, F, device=device, dtype=dtype)

            for b in range(B):
                start = b*T
                end = (b+1)*T
                Nk = posterior_flat[start:end].sum(dim=0).clamp_min(EPS)  # [K]
                mean[b] = (posterior_flat[start:end].T @ X_flat[start:end]) / Nk.unsqueeze(1)
                mean[b] = mean[b] + delta[b]  # apply context

                diff = X_flat[start:end].unsqueeze(1) - mean[b].unsqueeze(0)  # [T,K,F]
                weighted_diff = diff * posterior_flat[start:end].unsqueeze(-1)
                cov[b] = torch.einsum("tkf,tkh->kfh", weighted_diff, diff) / Nk.unsqueeze(-1).unsqueeze(-1)

                # Add min_covar to diagonal
                eye = torch.eye(F, device=device, dtype=dtype).unsqueeze(0)  # [1,F,F]
                cov[b] = cov[b] + getattr(self, "min_covar", 1e-6) * eye

                # Ensure positive definiteness
                eigvals, eigvecs = torch.linalg.eigh(cov[b])
                eigvals = torch.clamp(eigvals, min=getattr(self, "min_covar", 1e-6))
                cov[b] = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _contextual_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> Distribution:
        """
        Vectorized context-modulated emission PDF for batch-safe computation.

        Args:
            X: [T, F] or [B, T, F] observations (unused here, can guide fallback).
            theta: Latent encoding [B, latent_dim] or None.
            scale: Modulation scaling factor.

        Returns:
            Categorical or MultivariateNormal distribution.
        """
        scale = scale or getattr(self.config, "emission_scale", 1.0)
        pdf = self.pdf
        if pdf is None:
            return None

        # Determine batch size
        B = theta.shape[0] if theta is not None else 1
        nS, nF = self.n_states, self.n_features

        # --- Compute delta from context ---
        delta = None
        if theta is not None:
            combined = self.combine_context(theta)
            if combined is not None:
                ctx_layer = getattr(self, "ctx_emission", None)
                delta = ctx_layer(combined) if ctx_layer is not None else combined[..., :nS * nF]

        delta = self._prepare_delta(delta, shape=(B, nS, nF), scale=scale)

        # --- Categorical emission ---
        if isinstance(pdf, Categorical):
            logits = pdf.logits.to(self.device, DTYPE)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0).expand(nS, -1)  # [n_states, n_features]
            logits = logits.unsqueeze(0).expand(B, -1, -1)  # [B, n_states, n_features]
            logits = logits + delta
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            # Flatten last two dims if needed for Categorical
            logits = logits.view(B, -1)
            return Categorical(logits=logits)

        # --- Gaussian emission ---
        elif isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(self.device, DTYPE)
            cov = pdf.covariance_matrix.to(self.device, DTYPE)

            # Expand mean to batch
            if mean.ndim == 1:
                mean = mean.unsqueeze(0).expand(B, nS, nF)
            elif mean.ndim == 2:
                mean = mean.unsqueeze(0).expand(B, -1, -1)
            elif mean.ndim == 3 and mean.shape[0] != B:
                mean = mean[0].unsqueeze(0).expand(B, -1, -1)

            mean = mean + delta  # apply context

            # Expand covariance to batch
            if cov.ndim == 2:  # [F, F] shared
                cov = cov.unsqueeze(0).unsqueeze(0).expand(B, nS, nF, nF)
            elif cov.ndim == 3:  # [nS, F, F]
                cov = cov.unsqueeze(0).expand(B, -1, -1, -1)
            elif cov.ndim == 4 and cov.shape[0] != B:
                cov = cov[0].unsqueeze(0).expand(B, -1, -1, -1)

            # Add jitter for numerical stability
            eps_eye = EPS * torch.eye(nF, device=self.device, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
            cov = cov + eps_eye  # broadcast to [B, nS, F, F]

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            import warnings
            warnings.warn(f"[contextual_emission_pdf] Unsupported PDF type {type(pdf)}, using base PDF")
            return pdf

    def _contextual_duration_pdf(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> Categorical:
        """Return context-modulated duration distribution as a Categorical PDF, batch-safe."""
        scale = scale or getattr(self.config, "duration_scale", 0.1)
        temperature = max(temperature or getattr(self.config, "duration_temp", 1.0), 1e-6)

        # Base logits from duration module
        logits = getattr(self.duration_module, "logits", None)
        if logits is None:
            base_probs = self.duration_module.forward(log=False)
            B = 1
            if theta is not None:
                B = theta.shape[0]
            return Categorical(probs=base_probs.unsqueeze(0).expand(B, -1, -1))

        # Determine batch size
        B = 1
        if theta is not None:
            B = theta.shape[0]

        # Context modulation
        delta = None
        combined = self.combine_context(theta)
        if combined is not None and getattr(self, "ctx_duration", None):
            delta = self.ctx_duration(combined)

        # Prepare delta safely
        delta = self._prepare_delta(delta, shape=(B, self.n_states, self.max_duration), scale=scale)

        # Align logits to batch
        logits = self._align_logits(logits, B)  # ensures [B, n_states, max_duration]

        # Apply context delta and handle numeric issues
        combined_logits = torch.nan_to_num(logits + delta, nan=0.0, posinf=0.0, neginf=0.0)

        # Temperature-scaled softmax
        probs = F.softmax(combined_logits / temperature, dim=-1)

        return probs

    def _contextual_transition_matrix(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> Categorical:
        """Return context-modulated transition matrix as a Categorical distribution."""
        scale = scale or getattr(self.config, "transition_scale", 1.0)
        temperature = max(temperature or getattr(self.config, "transition_temp", 1.0), 1e-6)

        # Base logits from transition module
        logits = getattr(self.transition_module, "logits", None)
        if logits is None:
            logits = self.transition_module.forward(log=False)  # [n_states, n_states]

        # Determine batch size
        B = theta.shape[0] if theta is not None else 1

        # Context modulation
        delta = None
        combined = self.combine_context(theta)
        if combined is not None and getattr(self, "ctx_transition", None):
            delta = self.ctx_transition(combined)

        # Batch-safe delta
        delta = self._prepare_delta(delta, shape=(B, self.n_states, self.n_states), scale=scale)

        # Align logits to batch
        logits = self._align_logits(logits, B)
        combined_logits = logits + delta

        # Softmax over target states with temperature and numeric safety
        combined_logits = torch.nan_to_num(combined_logits)
        probs = F.softmax(combined_logits / temperature, dim=-1)

        return probs

    def encode_observations(
        self, 
        X: torch.Tensor, 
        pool: Optional[str] = None, 
        store: bool = True
    ) -> Optional[torch.Tensor]:
        if self.encoder is None:
            if store:
                self._context = None
            return None

        # Ensure tensor, device, dtype
        X = X.to(device=self.device, dtype=DTYPE)

        # Normalize input to [B, T, F]
        if X.ndim == 1:
            X = X.unsqueeze(0).unsqueeze(0)  # [1,1,F]
        elif X.ndim == 2:
            X = X.unsqueeze(0)               # [1,T,F]
        elif X.ndim != 3:
            raise ValueError(f"Expected input shape (F,), (T,F) or (B,T,F), got {X.shape}")

        # Temporarily override pooling
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None:
            self.encoder.pool = pool

        try:
            _ = self.encoder(X, return_context=True)
            vec = self.encoder.get_context()
        except Exception as e:
            raise RuntimeError(f"Encoder forward() failed for input shape {X.shape}: {e}")
        finally:
            self.encoder.pool = original_pool  # restore

        if vec is not None:
            vec = vec.detach().to(device=self.device, dtype=DTYPE)

        if store:
            self._context = vec

        return vec

    def _forward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized forward algorithm for NeuralHSMM with optional context θ.
        Supports batched or singleton sequences and context-modulated emissions.
        Returns a list of log-alpha tensors of shape (T_i, K_i, Dmax) per sequence.
        """
        neg_inf = -torch.inf
        device, dtype = self.device, DTYPE
        Dmax = self.max_duration

        # --- Prepare batch sequences ---
        if not X.is_batch:
            seq_batch = [seq.to(device=device, dtype=dtype) for seq in X.log_probs]
            lengths = X.lengths
        else:
            batch_out = X.to_batch(return_mask=False, include_log_probs=True)
            if isinstance(batch_out, tuple):
                seq_batch = batch_out[1].to(device=device, dtype=dtype)
            else:
                seq_batch = batch_out.to(device=device, dtype=dtype)
            lengths = X.lengths

        alpha_list = []

        for b, seq_len in enumerate(lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, self.n_states, Dmax), neg_inf, device=device, dtype=dtype))
                continue

            log_probs = seq_batch[b][:seq_len]

            # Flatten extra dimensions if any
            if log_probs.ndim > 2:
                log_probs = log_probs.flatten(start_dim=1)
            K_b = log_probs.shape[1]

            # --- Context modulation ---
            if theta is not None and hasattr(self, "_contextual_emission_pdf"):
                theta_seq = theta if theta.shape[0] == seq_len else theta.repeat_interleave(seq_len // theta.shape[0], dim=0)
                log_probs = self._contextual_emission_pdf(log_probs, theta_seq)
                if log_probs.ndim > 2:
                    log_probs = log_probs.flatten(start_dim=1)
                K_b = log_probs.shape[1]

            # --- Prepare cumulative sums for durations ---
            cumsum_emit = torch.vstack((torch.zeros((1, K_b), device=device, dtype=dtype),
                                        torch.cumsum(log_probs, dim=0)))

            # --- Initialize log-alpha ---
            log_alpha = torch.full((seq_len, K_b, Dmax), neg_inf, device=device, dtype=dtype)
            max_d0 = min(Dmax, seq_len)
            durations0 = torch.arange(1, max_d0 + 1, device=device)
            emit_sums0 = (cumsum_emit[durations0] - cumsum_emit[0]).T
            log_alpha[0, :, :max_d0] = self.init_logits[:K_b].unsqueeze(1) + self.duration_logits[:K_b, :max_d0] + emit_sums0

            # --- Recursion ---
            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                durations = torch.arange(1, max_dt + 1, device=device)
                starts = t - durations + 1

                emit_sums_t = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T

                idx = torch.clamp(starts - 1, min=0)
                prev_alpha_first = log_alpha[idx, :, 0]

                mask = (starts == 0).unsqueeze(1)
                prev_alpha_first = torch.where(mask, self.init_logits[:K_b].unsqueeze(0), prev_alpha_first)

                prev_alpha_exp = prev_alpha_first.unsqueeze(2) + self.transition_logits[:K_b, :K_b].unsqueeze(0)
                log_alpha_t = torch.logsumexp(prev_alpha_exp, dim=1).T

                log_alpha[t, :, :max_dt] = log_alpha_t + self.duration_logits[:K_b, :max_dt] + emit_sums_t

            alpha_list.append(log_alpha)

        return alpha_list

    @torch.no_grad()
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
        Context-aware HSMM prediction using base HSMM DP/backtrace.
        Only overrides emissions, transitions, and durations with context-modulated versions.
        """
        # --- Backup previous state ---
        prev_ctx = getattr(self, "_context", None)
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        # --- Build merged context ---
        ctx_list = []
        if context_ids is not None:
            if self.context_embedding is None:
                raise ValueError("context_ids provided but no context_embedding defined.")
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:
                ctx_emb = ctx_emb.mean(dim=1)
            ctx_list.append(ctx_emb.to(dtype=DTYPE, device=self.device))

        if context is not None:
            ctx_list.append(context.to(dtype=DTYPE, device=self.device))

        merged_ctx = torch.cat(ctx_list, dim=-1) if ctx_list else None
        if merged_ctx is not None:
            self.set_context(merged_ctx)

        # --- Encode observations and compute context-modulated distributions ---
        theta = self.encode_observations(X)
        try:
            pdf = self._contextual_emission_pdf(X, theta)
        except Exception as e:
            print(f"[predict] emission PDF creation failed: {e}")
            pdf = prev_pdf

        transition_logits = self._contextual_transition_matrix(theta)
        duration_logits = self._contextual_duration_pdf(theta)

        # --- Align logits batch ---
        B = X.shape[0] if X.ndim == 3 else 1
        def broadcast(logits):
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[0] != B:
                logits = self._prepare_delta(logits, shape=(B, *logits.shape[1:]), scale=1.0, broadcast=True)
            return logits

        transition_logits = broadcast(transition_logits)
        duration_logits = broadcast(duration_logits)

        # --- Temporarily override dynamic parameters ---
        if pdf is not None:
            self._params["emission_pdf"] = pdf
            if hasattr(pdf, "dof"):
                self._dynamic_dof = pdf.dof

        self._transition_logits = transition_logits
        self._duration_logits = duration_logits

        # --- Delegate to base HSMM predict ---
        preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

        # --- Restore previous state ---
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

        # Ensure list[Tensor] output
        if isinstance(preds, torch.Tensor):
            return [preds]
        return [torch.as_tensor(p, device=self.device) if not torch.is_tensor(p) else p for p in preds]

    @torch.no_grad()
    def decode(
        self,
        X: torch.Tensor,
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        algorithm: Literal["viterbi", "map"] = "viterbi",
    ) -> list[np.ndarray]:
        """
        Fully tensorized, batch-safe HSMM decoding with context-modulated emissions,
        transitions, and durations. Returns a list of numpy arrays, one per sequence.
        """
        # --- Ensure tensor on correct device ---
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        # --- Encode observations and merge context ---
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # --- Backup original parameters ---
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        try:
            # --- Assign context-aware emission PDF ---
            pdf = self._contextual_emission_pdf(X, theta)
            self._params["emission_pdf"] = pdf
            if hasattr(pdf, "dof"):
                self._dynamic_dof = pdf.dof

            # --- Contextual transition and duration logits ---
            transition = self._contextual_transition_matrix(theta)
            duration = self._contextual_duration_pdf(theta, scale=duration_weight)

            # --- Align logits for batch size ---
            B = theta.shape[0] if theta.ndim > 1 else 1
            self._transition_logits = self._align_logits(transition, B=B)
            self._duration_logits = self._align_logits(duration, B=B)

            # --- Base HSMM prediction ---
            preds_list = super().predict(X, algorithm=algorithm)

        finally:
            # --- Restore original parameters safely ---
            if prev_pdf is not None:
                self._params["emission_pdf"] = prev_pdf
            else:
                self._params.pop("emission_pdf", None)

            if prev_transition is not None:
                self._transition_logits = prev_transition
            elif hasattr(self, "_transition_logits"):
                del self._transition_logits

            if prev_duration is not None:
                self._duration_logits = prev_duration
            elif hasattr(self, "_duration_logits"):
                del self._duration_logits

        # --- Convert outputs to list of numpy arrays, one per sequence ---
        out_list: list[np.ndarray] = []
        for p in preds_list:
            if torch.is_tensor(p):
                out_list.append(p.detach().cpu().numpy())
            else:
                out_list.append(np.asarray(p, dtype=np.int64))

        return out_list

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