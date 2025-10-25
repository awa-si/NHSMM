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

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition, Contextual
from nhsmm.models.hsmm import HSMM
from nhsmm import utils


class NeuralEmission(Emission):
    """Vectorized contextual neural emission distribution supporting Gaussian and Categorical outputs."""

    def __init__(
        self,
        emission_type: str,
        params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        context_mode: str = None,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            n_states=params[0].shape[0] if emission_type == "gaussian" else params.shape[0],
            n_features=params[0].shape[1] if emission_type == "gaussian" else params.shape[1],
            emission_type=emission_type,
        )
        self.context_mode = context_mode
        self.device = device or torch.device("cpu")
        if self.context_mode not in {None, "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode: {context_mode}")
        self.encoder = encoder

        # Core parameters
        if emission_type == "gaussian":
            mu, cov = params
            self.mu = nn.Parameter(mu.to(self.device, DTYPE))
            self.cov = nn.Parameter(torch.diag_embed(cov) if cov.ndim == 2 else cov.to(self.device, DTYPE))
            self._proj = nn.Linear(self.n_features, 2 * self.n_states * self.n_features, bias=False).to(self.device, DTYPE)
        else:
            self.logits = nn.Parameter(params.to(self.device, DTYPE))
            self._proj = None

        # Contextual adapter
        self.contextual = Contextual(
            context_dim=None if self.encoder is None else getattr(self.encoder, "output_dim", None),
            target_dim=self.n_features,
            device=self.device,
            temporal_adapter=self.context_mode == "temporal",
            spatial_adapter=self.context_mode == "spatial",
            allow_projection=True
        )

    @property
    def dof(self) -> int:
        base = self.mu.numel() + self.cov.numel() if self.emission_type == "gaussian" else self.logits.numel()
        adapters = sum(
            p.numel() 
            for m in [self.encoder, self.contextual.temporal_adapter, self.contextual.spatial_adapter] if m 
            for p in m.parameters() if p.requires_grad
        )
        if self._proj is not None:
            adapters += sum(p.numel() for p in self._proj.parameters() if p.requires_grad)
        return base + adapters

    def _contextual_params(self, theta: Optional[torch.Tensor] = None):
        # Encode context if available
        ctx = theta
        if self.encoder and ctx is not None:
            ctx = self.encoder(ctx)
            if isinstance(ctx, tuple):
                ctx = ctx[0]

        # Apply contextual adapters
        param_base = self.mu if self.emission_type == "gaussian" else self.logits
        ctx = self.contextual._apply_context(param_base, ctx) if ctx is not None else param_base

        if self.emission_type == "gaussian":
            proj = self._proj(ctx)
            mu_param, cov_param = proj.chunk(2, dim=-1)
            B = mu_param.shape[0]
            mu = mu_param.view(B, self.n_states, self.n_features)
            cov = torch.diag_embed(F.softplus(cov_param.view(B, self.n_states, self.n_features)) + EPS)
            return mu, cov

        # Categorical logits, expanded batch dimension
        logits = ctx
        if logits.ndim == 2:
            batch_size = theta.shape[0] if theta is not None else 1
            logits = logits.unsqueeze(0).expand(batch_size, -1, -1)
        return logits

    contextual_params = _contextual_params

    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, log: bool = False):
        X = X.to(self.device, DTYPE)
        batch = X.ndim == 3
        B, T, Fdim = (X.shape if batch else (1, *X.shape))
        K = self.n_states

        if self.emission_type == "gaussian":
            mu, cov = self._contextual_params(theta)
            # Covariance already includes EPS
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            X_exp = X.unsqueeze(-2) if not batch else X
            logp = dist.log_prob(X_exp)
        else:
            logits = self._contextual_params(theta)
            X_idx = X.long()
            # Vectorized batch indexing
            if batch:
                B = X_idx.shape[0]
                K = self.n_states
                logp_full = F.log_softmax(logits, dim=-1)
                X_idx_exp = X_idx.unsqueeze(1).expand(-1, K, -1)
                logp = logp_full.gather(-1, X_idx_exp).transpose(1, 2)
            else:
                logp = F.log_softmax(logits, dim=-1)[0, X_idx].T
        return logp if log else logp.exp()

    log_prob = forward

    def sample(self, n_samples: int = 1, theta: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        if self.emission_type == "gaussian":
            mu, cov = self._contextual_params(theta)
            dist = Independent(MultivariateNormal(mu, covariance_matrix=cov), 1)
            samples = dist.rsample((n_samples,)).transpose(0, 1)
        else:
            logits = self._contextual_params(theta)
            dist = Categorical(logits=logits)
            samples = dist.sample((n_samples,)).transpose(0, 1)
        if state_indices is not None:
            samples = samples[:, state_indices] if samples.ndim == 3 else samples[state_indices]
        return samples

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior, X = posterior.to(self.device, DTYPE), X.to(self.device, DTYPE)
        Xf = X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X
        Pf = posterior.reshape(-1, posterior.shape[-1]) if posterior.ndim > 2 else posterior

        if self.emission_type == "gaussian":
            counts = Pf.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (Pf.T @ Xf) / counts
            diff = Xf.unsqueeze(1) - mu_new.unsqueeze(0)
            cov_new = torch.einsum("tkf,tkh->kfh", diff * Pf.unsqueeze(-1), diff) / counts.squeeze(-1)
            cov_new += EPS * torch.eye(self.n_features, device=self.device, dtype=DTYPE).unsqueeze(0)
            if inplace:
                self.mu.data.copy_(mu_new)
                self.cov.data.copy_(cov_new)
                return self
            return NeuralEmission("gaussian", (mu_new, cov_new), encoder=self.encoder, context_mode=self.context_mode, device=self.device)

        # Categorical update (fully vectorized)
        X_onehot = F.one_hot(Xf.long(), num_classes=self.n_features).float()
        weighted = Pf.T @ X_onehot
        logits_new = (weighted / weighted.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
        if inplace:
            self.logits.data.copy_(logits_new)
            return self
        return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder, context_mode=self.context_mode, device=self.device)

    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None, n_categories: int = None,
                   alpha: float = 1.0, encoder: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=DTYPE) * 0.1
            cov = torch.ones(n_states, n_features, device=device, dtype=DTYPE)
            return cls("gaussian", (mu, cov), encoder=encoder, device=device)
        logits = Dirichlet(torch.ones(n_categories, device=device, dtype=DTYPE) * alpha).sample([n_states]).clamp_min(EPS).log()
        return cls(emission_type.lower(), logits, encoder=encoder, device=device)


class NeuralDuration(Duration):
    """Contextual neural duration distribution with batch support and adapters."""

    def __init__(
        self,
        n_states: int,
        mode: str = "poisson",
        rate: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        context_mode: str = None,
        max_duration: int = 20,
        scale: float = 0.1,
    ):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device)
        self.device = device or torch.device("cpu")
        self.context_mode = context_mode
        self.encoder = encoder
        self.scale = scale
        self.mode = mode

        # Contextual adapters
        self.contextual = Contextual(
            context_dim=None if self.encoder is None else getattr(self.encoder, "output_dim", None),
            target_dim=self.n_states,
            device=self.device,
            temporal_adapter=self.context_mode == "temporal",
            spatial_adapter=self.context_mode == "spatial",
            allow_projection=True
        )

        # Core parameters
        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))

        self.register_buffer("_durations", torch.arange(1, max_duration + 1, device=self.device, dtype=DTYPE))

    @property
    def dof(self) -> int:
        base = self.rate.numel() if self.mode == "poisson" else self.mean.numel() + self.std.numel()
        adapters = sum(p.numel() for m in [self.encoder, self.contextual.temporal_adapter, self.contextual.spatial_adapter]
                       if m for p in m.parameters() if p.requires_grad)
        return base + adapters

    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        ctx = context
        if self.encoder and ctx is not None:
            ctx = self.encoder(ctx)
            if isinstance(ctx, tuple):
                ctx = ctx[0]

        base_param = self.rate if self.mode == "poisson" else torch.cat([self.mean, self.std.log()], dim=-1)
        ctx = self.contextual._apply_context(base_param, ctx) if ctx is not None else base_param

        if self.mode == "poisson":
            rate = F.softplus(ctx).view(-1, self.n_states)
            return rate.clamp_min(EPS)
        else:
            if ctx is not None:
                mean, log_std = torch.chunk(ctx, 2, dim=-1)
            else:
                mean, log_std = self.mean.view(1, -1), self.std.view(1, -1).log()
            return mean.to(self.device, DTYPE), log_std.exp().clamp_min(EPS)

    contextual_params = _contextual_params

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False):
        durations = self._durations
        K = self.n_states

        if self.mode == "poisson":
            rate = self._contextual_params(context)  # [B, K]
            B = rate.shape[0]
            logits = rate.unsqueeze(-1) * torch.log(durations.unsqueeze(0)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            B = mean.shape[0]
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 2
            logits = logits - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)

        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs) if log else probs

    log_prob = forward

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        if self.mode == "poisson":
            rate = self._contextual_params(context)  # [B, K]
            B, K = rate.shape
            samples = torch.poisson(rate.unsqueeze(-1).repeat(1, 1, n_samples))
        else:
            mean, std = self._contextual_params(context)
            B, K = mean.shape
            dist = LogNormal(mean.unsqueeze(-1), std.unsqueeze(-1))
            samples = dist.rsample((n_samples,)).transpose(0, 1)

        if state_indices is not None:
            samples = samples[:, state_indices] if samples.ndim == 3 else samples[state_indices]
        return samples

    def update(self, posterior: torch.Tensor, context: Optional[torch.Tensor] = None, inplace: bool = True):
        posterior = posterior.to(self.device, DTYPE)

        if self.mode == "poisson":
            # Weighted mean of durations
            counts = posterior.sum(dim=-1).clamp_min(EPS)  # sum over duration axis
            rate_new = (posterior * self._durations.unsqueeze(0)).sum(dim=-1) / counts
            rate_new = rate_new.clamp_min(EPS)
            if inplace:
                self.rate.data.copy_(rate_new)
                return self
            return NeuralDuration(self.n_states, mode="poisson", rate=rate_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device)

        else:
            counts = posterior.sum(dim=-1).clamp_min(EPS)
            mean_new = (posterior * self._durations.unsqueeze(0)).sum(dim=-1) / counts
            std_new = torch.sqrt((posterior * (self._durations.unsqueeze(0) - mean_new.unsqueeze(-1))**2).sum(dim=-1) / counts)
            std_new = std_new.clamp_min(EPS)
            if inplace:
                self.mean.data.copy_(mean_new)
                self.std.data.copy_(std_new)
                return self
            return NeuralDuration(self.n_states, mode="normal", mean=mean_new, std=std_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device)

    @classmethod
    def initialize(cls, n_states: int, mode: str = "poisson", max_duration: int = 20,
                   encoder: Optional[nn.Module] = None, context_mode: str = "none", device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        if mode == "poisson":
            rate = torch.ones(n_states, device=device, dtype=DTYPE)
            return cls(n_states, mode="poisson", rate=rate, encoder=encoder, context_mode=context_mode, device=device,
                       max_duration=max_duration)
        mean = torch.ones(n_states, device=device, dtype=DTYPE)
        std = torch.ones(n_states, device=device, dtype=DTYPE)
        return cls(n_states, mode="normal", mean=mean, std=std, encoder=encoder, context_mode=context_mode, device=device,
                   max_duration=max_duration)


class NeuralTransition(Transition):
    """Contextual neural transition distribution with batch-safe adapters and encoder projection."""

    def __init__(
        self,
        n_states: int,
        logits: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        context_mode: str = None,
        scale: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__(n_states=n_states, device=device)
        self.device = device or torch.device("cpu")
        self.encoder = encoder
        self.context_mode = context_mode
        self.scale = scale
        self.temperature = max(temperature, 1e-6)
        self.n_states = n_states

        # Core logits
        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            uniform = torch.full((n_states, n_states), 1.0 / n_states, device=self.device, dtype=DTYPE)
            self.logits = nn.Parameter(torch.log(uniform.clamp_min(EPS)))

        # Encoder projection to n_states^2
        if self.encoder is not None:
            self.proj_encoder = nn.Linear(getattr(encoder, "output_dim", None) or n_states, n_states * n_states, bias=True).to(self.device, DTYPE)

        # Contextual adapter
        self.contextual = Contextual(
            context_dim=None if self.encoder is None else getattr(self.encoder, "output_dim", None),
            target_dim=n_states,
            device=self.device,
            temporal_adapter=self.context_mode == "temporal",
            spatial_adapter=self.context_mode == "spatial",
            allow_projection=True
        )

    @property
    def dof(self) -> int:
        base = self.n_states * self.n_states  # actual logits parameters
        adapters = sum(
            p.numel()
            for m in [self.encoder, getattr(self, "proj_encoder", None),
                      self.contextual.temporal_adapter, self.contextual.spatial_adapter]
            if m for p in m.parameters() if p.requires_grad
        )
        return base + adapters

    def contextual_logits(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = theta.shape[0] if theta is not None else 1
        base_logits = self.logits.unsqueeze(0).expand(B, self.n_states, self.n_states)

        # Encoder contribution
        if self.encoder and theta is not None:
            out = self.encoder(theta)
            if isinstance(out, tuple):
                out = out[0]
            out = self.proj_encoder(out)  # [B, n_states^2]
            out = out.view(B, self.n_states, self.n_states)
            base_logits = base_logits + torch.tanh(out) * self.scale

        # Contextual adapter
        base_logits = self.contextual._apply_context(base_logits, theta)

        # Safe softmax with temperature
        base_logits = torch.nan_to_num(base_logits, nan=0.0, posinf=0.0, neginf=0.0)
        return F.log_softmax(base_logits / self.temperature, dim=-1)

    forward = contextual_logits

    def sample(self, n_samples: int = 1, theta: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        logits = self.contextual_logits(theta)  # [B, n_states, n_states]
        dist = Categorical(logits=logits)
        samples = dist.sample((n_samples,)).permute(1, 0, 2)  # [B, n_samples, n_states]
        if state_indices is not None:
            samples = samples[:, :, state_indices]  # select specific states
        return samples

    def update(self, posterior: torch.Tensor, inplace: bool = True):
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
        logits_new = torch.log(posterior.to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits_new)
            return self
        return NeuralTransition(
            n_states=self.n_states,
            logits=logits_new,
            encoder=self.encoder,
            context_mode=self.context_mode,
            device=self.device,
            scale=self.scale,
            temperature=self.temperature,
        )

    @classmethod
    def initialize(cls, n_states: int, encoder: Optional[nn.Module] = None,
                   context_mode: str = "none", device: Optional[torch.device] = None,
                   scale: float = 0.1, temperature: float = 1.0):
        device = device or torch.device("cpu")
        uniform_logits = torch.log(torch.full((n_states, n_states), 1.0 / n_states, device=device, dtype=DTYPE).clamp_min(EPS))
        return cls(n_states=n_states, logits=uniform_logits, encoder=encoder,
                   context_mode=context_mode, device=device, scale=scale, temperature=temperature)


@dataclass
class NHSMMConfig:
    n_states: int
    n_features: int
    max_duration: int
    alpha: float = 1.0
    seed: Optional[int] = None
    duration_mode: str = "normal"
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
        self.duration_module = NeuralDuration.initialize(
            n_states=config.n_states,
            mode="poisson" if config.duration_mode == "poisson" else "normal",
            max_duration=config.max_duration,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim > 0 else "none",
            device=self.device,
        )
        self.transition_module = NeuralTransition.initialize(
            n_states=config.n_states,
            encoder=self.encoder,
            context_mode="temporal" if config.context_dim > 0 else "none",
            device=self.device,
            scale=config.transition_scale,
            temperature=config.transition_temp,
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
        dof += nS - 1
        dof += nS * (nS - 1)
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

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        scale: Optional[float] = None,
    ) -> Union[Categorical, MultivariateNormal]:
        nS, nF = self.n_states, self.n_features
        scale = scale or getattr(self.config, "emission_scale", 1.0)
        device, dtype = self.device, DTYPE

        # --- Determine batch size ---
        B_X = X.shape[0] if X is not None and X.ndim > 1 else 0
        B_theta = theta.shape[0] if theta is not None and theta.ndim > 1 else 0
        B = max(1, B_X, B_theta)

        # --- Context delta ---
        delta = None
        if theta is not None:
            t = theta.unsqueeze(0) if theta.ndim == 1 else theta
            delta = getattr(self, "ctx_emission", lambda x: x[:, :nS*nF])(t)
        delta = self._prepare_delta(delta, shape=(B, nS, nF), scale=scale)

        # --- Categorical emission ---
        if self.emission_type.lower() == "categorical":
            if X is not None:
                if X.ndim == 2:  # [T, F]
                    logits = X.mean(dim=0, keepdim=True).expand(B, nS, nF)
                elif X.ndim == 3:  # [B, T, F]
                    logits = X.mean(dim=1, keepdim=True).expand(B, nS, nF)
                else:
                    logits = torch.full((B, nS, nF), 1.0 / nF, device=device, dtype=dtype)
            else:
                logits = torch.full((B, nS, nF), 1.0 / nF, device=device, dtype=dtype)

            logits = (logits + delta) / max(temperature, 1e-6)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            return Categorical(logits=logits.reshape(B * nS, nF))

        # --- Vectorized Gaussian emission ---
        elif self.emission_type.lower() == "gaussian":
            # Base mean/variance
            if X is not None:
                if X.ndim == 2:  # [T, F]
                    mean = X.mean(dim=0, keepdim=True).expand(B, nS, nF)
                    var = X.var(dim=0, unbiased=False, keepdim=True).expand(B, nS, nF)
                elif X.ndim == 3:  # [B, T, F]
                    mean = X.mean(dim=1, keepdim=True).expand(B, nS, nF)
                    var = X.var(dim=1, unbiased=False, keepdim=True).expand(B, nS, nF)
            else:
                mean = torch.zeros(B, nS, nF, device=device, dtype=dtype)
                var = torch.full((B, nS, nF), self.min_covar, device=device, dtype=dtype)

            var = var.clamp_min(self.min_covar)

            # Apply context delta
            mean = mean + delta

            # Construct diagonal covariance matrix vectorized
            cov = torch.diag_embed(var)  # [B, nS, nF, nF]

            # Ensure positive definiteness by adding jitter
            cov = cov + EPS * torch.eye(nF, device=device, dtype=dtype).view(1, 1, nF, nF)

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

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

        # Ensure tensors on correct device/dtype
        X = X.to(dtype=dtype, device=device)
        posterior = posterior.to(dtype=dtype, device=device)

        # Flatten batch/time if needed
        if X.ndim == 3:  # [B, T, F]
            B, T, _ = X.shape
            X_flat = X.reshape(B*T, F)
            posterior_flat = posterior.reshape(B*T, K)
        else:
            X_flat = X
            posterior_flat = posterior
            B = 1
            T = X_flat.shape[0]

        # --- Context delta ---
        delta = None
        if theta is not None and getattr(self, "ctx_emission", None):
            combined = self.combine_context(theta)  # [B, H]
            delta = self.ctx_emission(combined)
        delta = self._prepare_delta(delta, shape=(B, K, F), scale=scale, broadcast=False)

        # --- Categorical emission ---
        if self.emission_type.lower() == "categorical":
            if X_flat.ndim == 1 or (X_flat.ndim == 2 and X_flat.shape[1] == 1):
                X_flat = X_flat.squeeze(-1).long()
            else:
                X_flat = X_flat.long()

            # Weighted one-hot counts, vectorized
            X_onehot = F.one_hot(X_flat, num_classes=F).to(dtype=dtype, device=device)  # [B*T, F]
            counts = posterior_flat.T @ X_onehot  # [K, F]
            counts = counts.reshape(B, K, F)  # [B, K, F]

            probs = counts + delta
            probs = probs.clamp_min(EPS)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return Categorical(probs=probs)

        # --- Gaussian emission ---
        elif self.emission_type.lower() == "gaussian":
            # Compute weighted mean vectorized
            Nk = posterior_flat.reshape(B, T, K).sum(dim=1).clamp_min(EPS)  # [B, K]
            X_exp = X_flat.reshape(B, T, F)[:, :, None, :]  # [B, T, 1, F]
            post_exp = posterior_flat.reshape(B, T, K)[:, :, :, None]  # [B, T, K, 1]

            mean = (post_exp * X_exp).sum(dim=1) / Nk.unsqueeze(-1)  # [B, K, F]
            mean = mean + delta  # apply context

            # Compute diagonal covariance matrix
            diff = X_exp - mean[:, None, :, :]  # [B, T, K, F]
            weighted_diff = diff * post_exp  # [B, T, K, F]
            cov_diag = (weighted_diff * diff).sum(dim=1) / Nk.unsqueeze(-1)  # [B, K, F]

            # Add min_covar for numerical stability
            cov_diag = cov_diag.clamp_min(getattr(self, "min_covar", 1e-6))
            cov = torch.diag_embed(cov_diag)  # [B, K, F, F]

            # Ensure positive definiteness
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = eigvals.clamp_min(getattr(self, "min_covar", 1e-6))
            cov = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

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
        Vectorized, context-modulated emission PDF for batch-safe computation.

        Args:
            X: [T, F] or [B, T, F] observations (unused, for potential fallback)
            theta: [B, latent_dim] latent encoding
            scale: modulation factor

        Returns:
            Categorical or MultivariateNormal distribution.
        """
        scale = scale or getattr(self.config, "emission_scale", 1.0)
        pdf = self.pdf
        if pdf is None:
            import warnings
            warnings.warn("[contextual_emission_pdf] PDF is None, returning uniform fallback.")
            nS, nF = self.n_states, self.n_features
            if self.emission_type == "categorical":
                logits = torch.full((1, nS, nF), 1.0 / nF, device=self.device, dtype=DTYPE)
                return Categorical(logits=logits)
            else:
                mean = torch.zeros(1, nS, nF, device=self.device, dtype=DTYPE)
                cov = torch.eye(nF, device=self.device, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
                return MultivariateNormal(loc=mean, covariance_matrix=cov)

        # Determine batch size
        if theta is not None:
            B = theta.shape[0]
        elif X is not None and X.ndim == 3:
            B = X.shape[0]
        else:
            B = 1

        nS, nF = self.n_states, self.n_features

        # --- Context delta ---
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
            if logits.ndim == 1:           # [F]
                logits = logits.view(1, 1, nF).expand(B, nS, nF)
            elif logits.ndim == 2:         # [nS, F]
                logits = logits.unsqueeze(0).expand(B, -1, -1)
            elif logits.ndim == 3 and logits.shape[0] != B:
                logits = logits[0].unsqueeze(0).expand(B, -1, -1)
            logits = logits + delta
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return Categorical(logits=logits)

        # --- Gaussian emission ---
        elif isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(self.device, DTYPE)
            cov = pdf.covariance_matrix.to(self.device, DTYPE)

            # Batch-safe mean
            if mean.ndim == 1:           # [F]
                mean = mean.view(1, 1, nF).expand(B, nS, nF)
            elif mean.ndim == 2:         # [nS, F]
                mean = mean.unsqueeze(0).expand(B, -1, -1)
            elif mean.ndim == 3 and mean.shape[0] != B:
                mean = mean[0].unsqueeze(0).expand(B, -1, -1)
            mean = mean + delta

            # Batch-safe covariance
            if cov.ndim == 2:            # [F, F]
                cov = cov.view(1, 1, nF, nF).expand(B, nS, nF, nF)
            elif cov.ndim == 3:          # [nS, F, F]
                cov = cov.unsqueeze(0).expand(B, -1, -1, -1)
            elif cov.ndim == 4 and cov.shape[0] != B:
                cov = cov[0].unsqueeze(0).expand(B, -1, -1, -1)

            # Add jitter for numerical stability
            jitter = EPS * torch.eye(nF, device=self.device, dtype=DTYPE).view(1, 1, nF, nF)
            cov = cov + jitter

            # Ensure positive definiteness via eigenvalue clamping
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=getattr(self, "min_covar", 1e-6))
            cov = torch.einsum("...ij,...j,...jk->...ik", eigvecs, eigvals, eigvecs.transpose(-1, -2))

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            import warnings
            warnings.warn(f"[contextual_emission_pdf] Unsupported PDF type {type(pdf)}, returning base PDF")
            return pdf

    def _contextual_duration_pdf(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """Return context-modulated duration distribution as a Categorical PDF, batch-safe."""
        
        scale = scale or getattr(self.config, "duration_scale", 0.1)
        temperature = max(temperature or getattr(self.config, "duration_temp", 1.0), 1e-6)
        
        B = theta.shape[0] if theta is not None else 1
        device, dtype = self.device, DTYPE
        
        # Base logits
        logits = getattr(self.duration_module, "logits", None)
        if logits is None:
            base_probs = self.duration_module.forward(log=False).clamp_min(EPS)
            logits = torch.log(base_probs)
        
        # Align logits to batch [B, n_states, max_duration]
        if logits.ndim == 2:
            logits = logits.unsqueeze(0).expand(B, -1, -1)
        elif logits.ndim == 1:
            logits = logits.unsqueeze(0).unsqueeze(0).expand(B, self.n_states, -1)
        else:
            logits = logits.expand(B, -1, -1)
        
        # Context delta
        if theta is not None and getattr(self, "ctx_duration", None):
            combined = self.combine_context(theta)
            if combined is not None:
                delta = self.ctx_duration(combined).view(B, self.n_states, self.max_duration)
                logits = logits + delta * scale
        
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return F.softmax(logits / temperature, dim=-1)

    def _contextual_transition_matrix(
        self,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Return context-modulated transition matrix as a batched probability tensor.

        Args:
            theta: [B, H] context embeddings
            scale: context modulation scale
            temperature: softmax temperature

        Returns:
            Tensor [B, n_states, n_states] representing transition probabilities
        """
        scale = scale or getattr(self.config, "transition_scale", 1.0)
        temperature = max(temperature or getattr(self.config, "transition_temp", 1.0), 1e-6)
        B = theta.shape[0] if theta is not None else 1
        nS = self.n_states
        device, dtype = self.device, DTYPE

        # --- Base logits ---
        logits = getattr(self.transition_module, "logits", None)
        if logits is None:
            base_probs = self.transition_module.forward(log=False).clamp_min(EPS)
            logits = torch.log(base_probs)

        # Align logits to batch [B, n_states, n_states]
        if logits.ndim == 2:  # [nS, nS]
            logits = logits.unsqueeze(0).expand(B, -1, -1)
        elif logits.ndim == 1:  # unlikely, [nS]
            logits = logits.unsqueeze(0).unsqueeze(0).expand(B, nS, nS)
        else:  # assume already [B, nS, nS]
            logits = logits.expand(B, -1, -1)

        # --- Context modulation ---
        delta = torch.zeros(B, nS, nS, device=device, dtype=dtype)
        if theta is not None and getattr(self, "ctx_transition", None):
            combined = self.combine_context(theta)
            if combined is not None:
                delta = self.ctx_transition(combined)
                delta = self._prepare_delta(delta, shape=(B, nS, nS), scale=scale)

        # --- Combine and softmax ---
        logits = torch.nan_to_num(logits + delta, nan=0.0, posinf=50.0, neginf=-50.0)
        return F.softmax(logits / temperature, dim=-1)

    def __forward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
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
        Overrides emissions, transitions, and durations with context-modulated versions.
        """
        # --- Backup previous state ---
        prev_ctx = getattr(self, "_context", None)
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        # --- Merge context embeddings ---
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

        # --- Encode observations ---
        theta = self.encode_observations(X, store=False)

        # --- Compute context-modulated distributions ---
        try:
            pdf = self._contextual_emission_pdf(X, theta)
        except Exception as e:
            print(f"[predict] emission PDF creation failed: {e}")
            pdf = prev_pdf

        transition_logits = self._contextual_transition_matrix(theta)
        duration_logits = self._contextual_duration_pdf(theta)

        # --- Align logits for batch ---
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

        # --- Call base HSMM predict ---
        preds = super().predict(
            X, lengths=lengths, algorithm=algorithm, batch_size=batch_size
        )

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

        # --- Ensure list[Tensor] output on CPU ---
        if isinstance(preds, torch.Tensor):
            return [preds.cpu()]
        return [p.cpu() if torch.is_tensor(p) else torch.as_tensor(p, device="cpu") for p in preds]

    @torch.no_grad()
    def decode(
        self,
        X: torch.Tensor,
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        batch_size: int = 256,
    ) -> list[np.ndarray]:
        """
        Fully tensorized, batch-safe HSMM decoding with context-modulated emissions,
        transitions, and durations. Returns a list of numpy arrays, one per sequence.

        Optimizations:
        - Batched emission evaluation to reduce GPU memory usage.
        - Cleaner restoration of previous parameters.
        """
        # --- Ensure tensor on correct device ---
        X = torch.as_tensor(X, dtype=DTYPE, device=self.device) if not torch.is_tensor(X) else X.to(dtype=DTYPE, device=self.device)

        # --- Encode observations and merge context ---
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # --- Backup original parameters ---
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        try:
            # --- Context-aware emission PDF ---
            B = theta.shape[0] if theta.ndim > 1 else 1
            pdf_chunks = []
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                X_chunk = X[start:end]
                theta_chunk = theta[start:end] if theta.shape[0] == X.shape[0] else theta
                pdf_chunks.append(self._contextual_emission_pdf(X_chunk, theta_chunk))
            # Merge PDFs if needed
            pdf = pdf_chunks[0] if len(pdf_chunks) == 1 else pdf_chunks

            self._params["emission_pdf"] = pdf
            if hasattr(pdf, "dof"):
                self._dynamic_dof = pdf.dof

            # --- Contextual transition and duration logits ---
            transition = self._contextual_transition_matrix(theta)
            duration = self._contextual_duration_pdf(theta, scale=duration_weight)

            # --- Align logits for batch size ---
            self._transition_logits = self._align_logits(transition, B=B)
            self._duration_logits = self._align_logits(duration, B=B)

            # --- Delegate to base HSMM prediction ---
            preds_list = super().predict(X, algorithm=algorithm)

        finally:
            # --- Restore previous parameters safely ---
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

        # --- Convert outputs to list of numpy arrays ---
        return [p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p, dtype=np.int64) for p in preds_list]

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