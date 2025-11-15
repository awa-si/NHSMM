# nhsmm/distributions/default.py
import math
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Bernoulli, Laplace, MultivariateNormal, Normal, Independent, Poisson, StudentT
)

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm.tools import constraints


class Categorical(Distribution):
    """
    Enhanced Categorical distribution with:
    - Differentiable Gumbel-Softmax sampling (hard and relaxed)
    - Batch/time-safe handling
    - Dynamic temperature annealing
    - Proper support, logits/probs setters
    - Fully compatible with torch.distributions API
    """

    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }
    has_rsample = True

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        tau: float = 1.0,
        validate_args=None,
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs")
        if probs is not None:
            logits = torch.log(probs.clamp_min(EPS))
        self.logits = logits
        self.tau = tau
        batch_shape = logits.shape[:-1]
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([]), validate_args=validate_args)

    # ---------------- Properties ----------------
    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    @probs.setter
    def probs(self, p: torch.Tensor):
        if p.shape != self.logits.shape:
            raise ValueError(f"Expected probs shape {self.logits.shape}, got {p.shape}")
        self.logits = torch.log(p.clamp_min(EPS))

    @property
    def logits_(self):
        return self.logits

    @logits_.setter
    def logits_(self, l):
        self.logits = l

    @property
    def support(self):
        K = self.logits.size(-1)
        return torch.distributions.constraints.integer_interval(0, K - 1)

    # ---------------- Log probability ----------------
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.long()
        flat_logits = self.logits.reshape(-1, self.logits.size(-1))
        flat_value = value.reshape(-1)
        logp = F.log_softmax(flat_logits, dim=-1)
        return logp.gather(-1, flat_value.unsqueeze(-1)).squeeze(-1).reshape(value.shape)

    # ---------------- Internal Gumbel ----------------
    def _gumbel_logits(self, sample_shape=torch.Size(), generator=None):
        shape = sample_shape + self.logits.shape
        g = -torch.log(-torch.log(
            torch.rand(shape, device=self.logits.device, generator=generator).clamp_min(EPS)
        ))
        return (self.logits.expand(shape) + g) / self.tau

    # ---------------- Sampling ----------------
    def _sample_gumbel_softmax(self, sample_shape=torch.Size(), hard=True, generator=None):
        y_soft = F.softmax(self._gumbel_logits(sample_shape, generator), dim=-1)
        if not hard:
            return y_soft
        y_hard = F.one_hot(y_soft.argmax(-1), num_classes=y_soft.size(-1)).to(y_soft.dtype)
        return (y_hard - y_soft).detach() + y_soft

    def sample(self, sample_shape=torch.Size(), hard=True, tau=None, generator=None):
        old_tau = self.tau
        if tau is not None:
            self.tau = tau
        y = self._sample_gumbel_softmax(sample_shape, hard=hard, generator=generator)
        self.tau = old_tau
        return y

    # ---------------- Reparameterized sample ----------------
    def rsample(self, sample_shape=torch.Size(), tau=None, generator=None):
        return self.sample(sample_shape, hard=False, tau=tau, generator=generator)

torch.distributions.Categorical = Categorical

class DistributionBase(nn.Module):
    """
    Base class for context-modulated HSMM parameters.

    Features:
    - Supports 1D/2D/3D context
    - Optional temporal Conv1d + spatial Linear adapters
    - Deterministic caching with versioning
    - Safe broadcasting
    - Optional learnable delta scaling
    - Normalization: l2, LayerNorm, BatchNorm
    - Fully device/DTYPE aware
    - Subclass hooks: gating, residuals, low-rank, constraints
    """
    _dist_type: type[Distribution] = Categorical  # override in subclasses

    def __init__(
        self,
        target_dim: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        activation: str = "tanh",
        final_activation: str = "tanh",
        learnable_scale: bool = False,
        max_delta: float = 0.5,
        cache_enabled: bool = True,
        cache_limit: int = 32,
        cache_grad_safe: bool = False,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = DTYPE
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.max_delta = max_delta
        self.debug = debug

        # ---------------- Activations ----------------
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        # ---------------- Hidden dimension ----------------
        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        # ---------------- Context encoder ----------------
        self.context_net: Optional[nn.Module] = None
        self._proj: Optional[nn.Linear] = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim, dtype=DTYPE),
            ).to(self.device, DTYPE)
            self._init_weights(self.context_net)
        elif allow_projection:
            # projection will be dynamically created on first context mismatch
            self._proj = None

        # ---------------- Adapters ----------------
        self.temporal_adapter = nn.Conv1d(target_dim, target_dim, 3, padding=1, bias=False).to(self.device, DTYPE) if temporal_adapter else None
        self.spatial_adapter = nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE) if spatial_adapter else None
        for a in [self.temporal_adapter, self.spatial_adapter]:
            if a is not None:
                nn.init.xavier_uniform_(a.weight)

        # ---------------- Delta scaling ----------------
        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE, device=self.device))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE))

        # ---------------- Normalization ----------------
        self._batchnorm: Optional[nn.BatchNorm1d] = None
        self.log_temperature = nn.Parameter(torch.tensor(0.0, dtype=DTYPE, device=self.device))

        # ---------------- Cache ----------------
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int32))
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.cache_grad_safe = cache_grad_safe

        # ---------------- Base logits ----------------
        self.logits = nn.Parameter(torch.zeros(target_dim, dtype=DTYPE, device=self.device))

    @property
    def dist_type(self):
        return getattr(self, "_dist_type", self.__class__._dist_type)

    # ---------------- Utilities ----------------
    def _get_activation(self, name: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }.get(name.lower(), nn.Identity())

    def _init_weights(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------- Context validation ----------------
    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        context = context.to(self.device, self.dtype)
        if context.ndim not in (1, 2, 3):
            raise ValueError(f"Unsupported context shape {context.shape}")

        in_dim = context.shape[-1]
        if self.context_dim is None:
            self.context_dim = in_dim

        if in_dim != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"context_dim mismatch: expected {self.context_dim}, got {in_dim}")
            if self._proj is None or self._proj.in_features != in_dim:
                self._proj = nn.Linear(in_dim, self.context_dim, device=self.device, dtype=self.dtype)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
                self._invalidate_cache()
            context = self._proj(context)
        return context

    # ---------------- Subclass hooks ----------------
    def _context_gate(self, context: torch.Tensor) -> torch.Tensor: return 0.0
    def _context_residual(self, context: torch.Tensor) -> torch.Tensor: return 0.0
    def _context_low_rank(self, context: torch.Tensor) -> torch.Tensor: return 0.0
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor: return logits
    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain non-finite values.")
        return logits

    # ---------------- Deterministic hash ----------------
    @torch.no_grad()
    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        if context is None:
            return f"none-v{int(self._param_version)}"
        c = context.detach().float()
        h = (float(c.mean()), float(c.std()), int(c.numel()), int(self._param_version))
        return str(h)

    # ---------------- Cache utils ----------------
    @torch.no_grad()
    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        if not self.cache_enabled: return None
        out = self._cache.get(key)
        if out is not None: self._cache.move_to_end(key)
        return out

    @torch.no_grad()
    def _cache_set(self, key: str, value: torch.Tensor):
        if not self.cache_enabled: return
        self._cache[key] = value if self.cache_grad_safe else value.detach()
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _invalidate_cache(self):
        self._cache.clear()
        self._param_version += 1

    # ---------------- Delta preparation ----------------
    def _prepare_delta(self, delta: torch.Tensor, l2_normalize=False, layer_norm=False,
                       batch_norm=False, skip_adapters=False) -> torch.Tensor:
        delta = delta.to(self.device, self.dtype)
        if l2_normalize: delta = F.normalize(delta, dim=-1, eps=EPS)
        if layer_norm: delta = F.layer_norm(delta, delta.shape[-1:])
        if batch_norm:
            flat = delta.flatten(0, -2)
            if self._batchnorm is None or self._batchnorm.num_features != flat.shape[-1]:
                self._batchnorm = nn.BatchNorm1d(flat.shape[-1], affine=True, track_running_stats=False, eps=EPS).to(self.device, self.dtype)
            delta = self._batchnorm(flat).view(delta.shape)

        if not skip_adapters:
            if self.temporal_adapter is not None:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        delta = self.final_activation_fn(delta) * self.delta_scale
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=self.max_delta, neginf=-self.max_delta)
        if self.debug:
            print("Delta stats:", delta.mean().item(), delta.std().item())
        return delta

    # ---------------- Apply context ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor],
                       grad_scale: Optional[float] = None, skip_adapters=False,
                       l2_normalize=False) -> torch.Tensor:

        context = self._validate_context(context)
        if context is None:
            return base.unsqueeze(0) if base.ndim == 1 else base

        batch_size, seq_len = (context.shape[0], None) if context.ndim == 2 else (context.shape[0], context.shape[1])
        context_flat = context.reshape(-1, context.shape[-1]) if context.ndim == 3 else context

        delta = self.context_net(context_flat) if self.context_net is not None else \
                self._proj(context_flat) if self._proj is not None else \
                torch.zeros((context_flat.shape[0], self.target_dim), device=self.device, dtype=self.dtype)

        delta += self._context_gate(context)
        delta += self._context_residual(context)
        delta += self._context_low_rank(context)
        delta = self._prepare_delta(delta, l2_normalize=l2_normalize, skip_adapters=skip_adapters)

        if seq_len is not None:
            delta = delta.view(batch_size, seq_len, self.target_dim)

        # Broadcast base
        base_exp = base
        if base.ndim == 1:
            base_exp = base.unsqueeze(0).expand(delta.shape[0], -1) if seq_len is None else base.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        elif base.ndim == 2 and seq_len is not None and base.shape[0] == batch_size:
            base_exp = base.unsqueeze(1).expand(batch_size, seq_len, self.target_dim)

        if grad_scale is not None:
            delta = delta * grad_scale

        return self._apply_constraints(base_exp + delta)

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(self.logits, context)

        temp = torch.exp(getattr(self, "log_temperature", torch.tensor(0.0, device=self.device, dtype=self.dtype))) if temperature is None else torch.as_tensor(temperature, device=self.device, dtype=self.dtype)
        mod = mod / max(temp, 1e-6)
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)

        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Forward / Distribution ----------------
    def forward(self, context=None, log=False, return_dist=False, temperature=None):
        mod = self._modulate(context, temperature)
        if return_dist: return self._dist_type(logits=mod)
        return F.log_softmax(mod, dim=-1) if log else F.softmax(mod, dim=-1)

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context=None, temperature=None):
        return self.forward(context=context, return_dist=True, temperature=temperature).sample()

    def log_prob(self, x, context=None, temperature=None):
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        return dist.log_prob(x.view(-1).to(torch.long)).view(*x.shape)

    # ---------------- Log matrix / Expected probs ----------------
    def log_matrix(self, context=None, temperature=None):
        return F.log_softmax(self._modulate(context, temperature), dim=-1)

    def expected_probs(self, context=None, temperature=None):
        return self.forward(context=context, log=False, temperature=temperature)

    def mode(self, context=None, temperature=None):
        return torch.argmax(self.forward(context=context, log=False, temperature=temperature), dim=-1)

    # ---------------- EM / update ----------------
    @torch.no_grad()
    def update(self, *args, **kwargs):
        # placeholder for subclass EM updates
        pass

    # ---------------- Initialize ----------------
    def initialize(self, mode="uniform", **_):
        # subclasses may override for proper initialization
        return self


class Emission(DistributionBase):
    """DistributionBase emission distribution supporting Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-6,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
        emission_type: str = "gaussian",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        debug: bool = False,
        scale: float = 0.1,
        dof: float = 5.0,
        seed: int = 0,
    ):
        target_dim = n_states * n_features
        super().__init__(
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            spatial_adapter=spatial_adapter,
            temporal_adapter=temporal_adapter,
            allow_projection=allow_projection,
            debug=debug,
        )

        self.n_states = n_states
        self.n_features = n_features
        self.emission_type = emission_type.lower()
        self.adaptive_scale = adaptive_scale
        self.modulate_var = modulate_var
        self.min_covar = min_covar
        self.scale = scale
        self.seed = seed
        self.dof = dof

        # Buffers
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        self.register_buffer("_emission_covs", torch.eye(n_features, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states,1,1))
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"categorical","bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"laplace","studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    @property
    def dist_type(self):
        """Return the torch distribution class corresponding to `emission_type`."""
        type_map = {
            "gaussian": torch.distributions.Independent,
            "laplace": torch.distributions.Independent,
            "normal": torch.distributions.Independent,
            "categorical": torch.distributions.Categorical,
            "bernoulli": torch.distributions.Bernoulli,
            # add other mappings as needed
        }

        base_map = {
            "normal": torch.distributions.Normal,
            "gaussian": torch.distributions.Normal,
            "laplace": torch.distributions.Laplace,
        }

        key = self.emission_type.lower()
        if key in type_map:
            base = base_map.get(key, None)
            if base is not None:
                # Wrap in Independent if multivariate
                return lambda *args, **kwargs: type_map[key](base(*args, **kwargs), 1)
            return type_map[key]
        raise ValueError(f"Unknown emission_type: {self.emission_type}")

    # ---------------- Utility / Context Methods ----------------
    @torch.no_grad()
    def _spread_means(
        self,
        means: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        n_iter: int = 10,
        min_dist: float = 1e-3,
    ) -> torch.Tensor:
        """Vectorized jitter of means with optional context modulation to ensure minimal separation."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        K, F = means.shape

        # initial jitter
        candidate = means + scale * torch.randn_like(means)
        last_mod = candidate

        for i in range(n_iter):

            # --- FIX: recompute context delta for the *current* candidate ---
            if context is not None:
                delta = self._prepare_delta(candidate, context)
            else:
                delta = 0.0

            candidate_mod = candidate + delta

            # pairwise squared distances (vectorized)
            norms = candidate_mod.pow(2).sum(dim=1, keepdim=True)
            dist_sq = norms + norms.T - 2.0 * (candidate_mod @ candidate_mod.T)
            dist_sq.fill_diagonal_(float("inf"))

            # all states sufficiently separated?
            if torch.all(dist_sq.min(dim=1).values > min_dist):
                return candidate_mod

            # incremental jitter
            jitter_scale = scale * 0.1 * (1 - i / max(1, n_iter - 1))
            candidate = candidate + jitter_scale * torch.randn_like(means)
            last_mod = candidate_mod

        return last_mod

    # ---------------- Distribution Estimation ----------------
    @torch.no_grad()
    def _estimate_dist(
        self,
        X: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        init_spread: float = 1.0,
        max_jitter: int = 5,
    ):
        """Estimate emission distribution parameters with proper batch handling."""

        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std() < EPS:
                X = X + 1e-3 * torch.randn_like(X)

        # ---------------- Continuous Distributions ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self._emission_means.clone()

            # Weighted mean
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)          # [T,K]
                w_sum = w.sum(dim=0) + EPS            # [K]
                means = (w.T @ X) / w_sum.unsqueeze(1)  # [K,F]

            # Theta adjustment
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                means = means + theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # Context modulation (modulate each state independently)
            means = self._modulate(means, context)  # [K,F]

            if etype == "gaussian":
                if X is not None and posterior is not None:
                    # diff: [T,K,F], broadcasting is automatic
                    diff = X[:, None, :] - means[None, :, :]
                    weighted = diff * w[:, :, None]
                    covs = torch.einsum("tkf,tkd->kfd", weighted, diff) / w_sum[:, None, None]
                else:
                    covs = self._emission_covs.clone()

                # Ensure positive-definite
                I = torch.eye(F, device=self.device)
                covs = covs + self.min_covar * I[None, :]
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0:
                            break
                        covs[k] = covs[k] + jitter * I
                        jitter *= 2

                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covariance_matrix=covs)

            else:  # Laplace / StudentT
                if X is not None and posterior is not None:
                    diff = (X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]
                    scales = diff.sum(dim=0) / w_sum[:, None]
                else:
                    scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()

                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales ** 2))

                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=scales), 1)

        # ---------------- Discrete Distributions ----------------
        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)  # [T,K]
                w_sum = w.sum(dim=0) + EPS     # [K]

                if etype == "categorical":
                    logits = torch.zeros((K, F), device=self.device)
                    for k in range(K):
                        counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                        logits[k] = torch.log(counts / counts.sum() + EPS)
                else:  # Bernoulli / Poisson
                    rate = (w.T @ X.float()) / w_sum[:, None]
                    logits = torch.log(rate.clamp_min(EPS))
            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            # Theta adjustment
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                logits = logits + theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # Context modulation
            logits = self._modulate(logits, context)

            # Save parameters
            if etype == "categorical":
                self._emission_params.copy_(torch.softmax(logits, dim=-1))
                return Categorical(logits=logits)
            elif etype == "bernoulli":
                self._emission_params.copy_(logits)
                return Independent(Bernoulli(logits=logits), 1)
            else:  # Poisson
                self._emission_params.copy_(logits)
                return Independent(Poisson(logits=logits), 1)

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ---------------- Forward / Distribution ----------------
    def _modulate(self, tensor: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply context modulation and optional adaptive scaling to a parameter tensor.

        Args:
            tensor: Parameter tensor of shape [n_states, n_features] or similar.
            context: Optional context tensor of shape [batch_size, context_dim] or [context_dim].

        Returns:
            Modulated tensor with the same shape as input.
        """
        if context is None:
            return tensor

        # Apply context via DistributionBase method
        modulated = self._apply_context(tensor, context)

        # Adaptive scaling based on context norm
        if self.adaptive_scale:
            if context.ndim > 1:
                norm = context.norm(dim=-1).mean() + EPS
            else:
                norm = context.norm() + EPS
            modulated = modulated * (self.scale / norm)

        return modulated

    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        etype = self.emission_type

        if etype == "gaussian":
            mu = self._modulate(self.mu, context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                var += self._modulate(var, context).abs()
            cov = torch.diag_embed(var)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(cov)
            dist = Independent(Normal(loc=mu, scale=var.sqrt()), 1)

        elif etype in {"laplace", "studentt"}:
            loc = self._modulate(self.loc, context)
            scale = torch.clamp(self.scale_param, min=self.min_covar)
            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale ** 2))
            dist_cls = Laplace if etype == "laplace" else StudentT
            if etype == "laplace":
                dist = Independent(dist_cls(loc=loc, scale=scale), 1)
            else:
                dist = Independent(StudentT(df=self.dof, loc=loc, scale=scale), 1)

        else:
            base_param = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._modulate(base_param, context)
            self._emission_params.copy_(out)

            if etype == "categorical":
                dist = Categorical(logits=out)
            elif etype == "bernoulli":
                dist = Independent(Bernoulli(logits=out), 1)
            else:  # poisson
                dist = Independent(Poisson(rate=torch.exp(out)), 1)

        return dist if return_dist else (self._emission_means, self._emission_covs) if etype in {"gaussian","laplace","studentt"} else self._emission_params

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(dtype=DTYPE, device=self.device)

    def log_prob(self, x, context=None):
        dist = self.forward(context=context, return_dist=True)
        if isinstance(dist, Independent) and x.ndim == 2 and x.shape[1] == self.n_features:
            x = x.unsqueeze(1)
        return dist.log_prob(x)

    def log_matrix(self, context=None, temperature=None):
        return super().log_matrix(context=context, temperature=temperature)

    def expected_probs(self, context=None, temperature=None):
        return super().expected_probs(context=context, temperature=temperature)

    def parameters_tensor(self):
        if self.emission_type in {"gaussian","laplace","studentt"}:
            return self._emission_means,self._emission_covs
        return self._emission_params

    @torch.no_grad()
    def update(self, X=None, posterior=None, theta=None, context=None, theta_scale=0.1, update_rate=0.5, init_spread=0.1, max_jitter=5, rank=None):
        new_dist = self._estimate_dist(
            X=X, theta=theta, context=context, posterior=posterior,
            theta_scale=theta_scale, init_spread=init_spread, max_jitter=max_jitter
        )
        etype = self.emission_type

        if etype == "gaussian":
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.loc)
            self._emission_covs.mul_(1 - update_rate).add_(update_rate * new_dist.covariance_matrix)
            self.mu.copy_(self._emission_means)
            self.log_var.copy_(torch.log(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS)))

        elif etype in {"laplace", "studentt"}:
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.base_dist.loc)
            scale_sq = new_dist.base_dist.scale ** 2
            diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale_sq
            self._emission_covs.copy_(torch.diag_embed(diag))
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(diag, min=EPS)))

        else:
            if etype == "categorical":
                new_logits = new_dist.logits
            elif etype == "bernoulli":
                new_logits = new_dist.base_dist.logits
            else:
                new_logits = torch.log(torch.clamp(new_dist.base_dist.rate, min=EPS))

            self._emission_params.mul_(1 - update_rate).add_(update_rate * new_logits)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    @torch.no_grad()
    def initialize(self, X=None, context=None, mode="data", iters=15):
        if X is None or mode == "default":
            return self.forward(context=context, return_dist=True)

        K, F = self.n_states, self.n_features
        Xf = X.reshape(-1, F).to(dtype=DTYPE) if X.ndim == 3 else X.to(dtype=DTYPE)
        device = Xf.device
        N = Xf.shape[0]

        def run_kmeans(Xflat, K, iters):
            idx = torch.randperm(Xflat.shape[0], device=device)[:K]
            centers = Xflat[idx].clone()
            for _ in range(iters):
                dist = torch.cdist(Xflat, centers)
                labels = dist.argmin(dim=1)
                for k in range(K):
                    pts = Xflat[labels == k]
                    if pts.shape[0] >= 2:
                        centers[k] = pts.mean(dim=0)
            return centers, labels

        if mode == "data":
            mean = Xf.mean(0)
            cov = (Xf - mean).T @ (Xf - mean) / max(N - 1, 1) + self.min_covar * torch.eye(F, device=device)
            means = mean.expand(K, F).clone()
            covs = cov.unsqueeze(0).expand(K, F, F).clone()
            if self.emission_type == "gaussian":
                self.mu.copy_(means)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1)))
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covs)
            elif self.emission_type in {"laplace", "studentt"}:
                sc = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                self.loc.copy_(means)
                self.scale_param.copy_(sc)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(sc ** 2))
                dist_cls = Laplace if self.emission_type == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=sc), 1)
            else:
                logits = torch.log_softmax(Xf.mean(0).expand(K, F), dim=-1)
                param = getattr(self, "logits", getattr(self, "log_rate", None))
                param.copy_(logits)
                self._emission_params.copy_(logits)
                if self.emission_type == "categorical":
                    return Categorical(logits=logits)
                elif self.emission_type == "bernoulli":
                    return Independent(Bernoulli(logits=logits), 1)
                else:
                    return Independent(Poisson(logits), 1)

        if mode == "kmeans":
            centers, labels = run_kmeans(Xf, K, iters)
            means = centers
            covs = torch.stack([torch.cov(Xf[labels==k].T) + self.min_covar * torch.eye(F, device=device) for k in range(K)])
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)
            if self.emission_type == "gaussian":
                self.mu.copy_(means)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1)))
                return MultivariateNormal(means, covs)
            elif self.emission_type in {"laplace", "studentt"}:
                sc = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                self.loc.copy_(means)
                self.scale_param.copy_(sc)
                return Independent(Laplace(loc=means, scale=sc) if self.emission_type=="laplace" else StudentT(df=self.dof, loc=means, scale=sc), 1)
            else:
                logits = means
                param = getattr(self, "logits", getattr(self, "log_rate", None))
                param.copy_(logits)
                self._emission_params.copy_(logits)
                if self.emission_type=="categorical":
                    return Categorical(logits=logits)
                elif self.emission_type=="bernoulli":
                    return Independent(Bernoulli(logits=logits),1)
                else:
                    return Independent(Poisson(logits),1)


class Initial(DistributionBase):
    """Initial-state distribution for HSMMs with neural/contextual modulation.

    Features:
    - Neural/contextual modulation via MLP, residual, and optional low-rank
    - Learnable temperature scaling
    - EM-style or gradient-based updates
    - Deterministic caching
    - Fully differentiable via Categorical
    """
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: int | None = None,
        hidden_dim: int | None = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        scale: float = 1.0,
        rank: int | None = None,
        cache_limit: int = 32,
        debug: bool = False,
    ):
        super().__init__(
            target_dim=n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.scale = scale
        self.rank = rank

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Optional neural/residual/low-rank gating
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, dtype=DTYPE),
            )
            self._residual_gate = nn.Sequential(
                nn.Linear(context_dim, h, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, dtype=DTYPE),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE)
        else:
            self._context_gate = None

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states,), -math.log(n_states), dtype=DTYPE)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, n_states, dtype=DTYPE)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode: {mode}")
        return self._validate_logits(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self.dist_type(logits=logits)

    # ---------------- Validation ----------------
    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain non-finite values.")
        return logits

    # ---------------- Context modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None:
            return logits

        # Ensure context is 3D: (B, T, D)
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:
            context = context.unsqueeze(1)
        elif context.ndim != 3:
            raise ValueError(f"Unsupported context shape: {context.shape}")

        B, T, D = context.shape
        n_states = self.n_states

        # Base logits expansion
        if logits.ndim == 1:
            base = logits.view(1, 1, n_states).expand(B, T, n_states)
        else:
            base = logits.unsqueeze(0).expand(B, T, n_states)

        mod = super()._apply_context(base, context)

        # Flatten for neural/residual gates
        ctx_flat = context.reshape(B * T, D)

        if self._context_gate is not None:
            gate = self._context_gate(ctx_flat).view(B, T, n_states)
            residual = self._residual_gate(ctx_flat).view(B, T, n_states)
            mod = mod + gate + residual

            # Optional low-rank
            if self.rank is not None:
                U = self._U(ctx_flat).view(B, T, n_states, self.rank)
                V = self._V(ctx_flat).view(B, T, 1, self.rank)
                mod = mod + (U * V).sum(-1)

        mod = self._validate_logits(mod)
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)) if mod.ndim > 1 else mod)
        return mod

    # ---------------- Modulate / caching ----------------
    def _modulate(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        temp = torch.exp(self.log_temperature) if temperature is None else max(temperature, 1e-6)
        mod = self._apply_context(self.logits, context) / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)
        self._cache_set(key, mod.detach())
        return mod

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context=None, temperature=None):
        return super().sample(context=context, temperature=temperature)

    def log_prob(self, x, context=None, temperature=None):
        return super().log_prob(x, context=context, temperature=temperature)

    def log_matrix(self, context=None, temperature=None):
        return super().log_matrix(context=context, temperature=temperature)

    def expected_probs(self, context=None, temperature=None):
        return super().expected_probs(context=context, temperature=temperature)

    def mode(self, context=None, temperature=None):
        return super().mode(context=context, temperature=temperature)

    # ---------------- EM / gradient update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        context: Optional[torch.Tensor] = None,
    ):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = self._validate_logits(new_logits)
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            lr = update_rate or 1.0
            mod_logits = self._apply_context(self.logits.unsqueeze(0), context)
            temp = torch.exp(self.log_temperature) if temperature is None else max(temperature, 1e-6)
            log_probs = F.log_softmax(mod_logits / temp, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(lr * p.grad)
            self._invalidate_cache()


class Duration(DistributionBase):
    """Categorical duration distribution per state for HSMMs with neural/contextual modulation."""

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        scale: float = 1.0,
        gate_factor: float = 0.5,
        temperature: float = 1.0,
        min_temperature: float = 1e-6,
        smooth_factor: float = 0.01,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        cache_limit: int = 32,
        debug: bool = False,
        init_mode: str = "uniform",
    ):
        target_dim = n_states * max_duration
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.max_duration = max_duration
        self.scale = scale
        self.gate_factor = gate_factor
        self.smooth_factor = float(smooth_factor)
        self.min_temperature = min_temperature
        self.temperature = temperature

        self.register_buffer("_durations", torch.arange(1, max_duration + 1, dtype=DTYPE))

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        self._context_gate: Optional[nn.Module] = None
        if context_dim is not None:
            hidden = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden, n_states, dtype=DTYPE),
            ).to(self.device, self.dtype)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE, device=self.device)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states, 1)
            w /= w.sum(dim=1, keepdim=True)
            logits = torch.log(w)
        elif mode == "normal":
            logits = torch.randn(n_states, max_duration, dtype=DTYPE, device=self.device) * 0.1
            logits -= torch.arange(max_duration, dtype=DTYPE, device=self.device) * 0.05
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        return self._validate_logits(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        self.log_temperature.data.fill_(torch.log(torch.tensor(max(self.log_temperature.exp().item(), self.min_temperature), dtype=DTYPE)))
        return self.dist_type(logits=logits)

    # ---------------- Context modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None:
            if context.ndim == 1:
                context = context.view(1, 1, -1)
            elif context.ndim == 2:
                context = context.unsqueeze(1)
            elif context.ndim != 3:
                raise ValueError(f"Unsupported context shape {context.shape}")

            B, T, D = context.shape
            base = logits.view(1, 1, self.n_states, self.max_duration).expand(B, T, -1, -1)
            mod = super()._apply_context(base, context)
        else:
            mod = super()._apply_context(logits, context)

        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context).unsqueeze(-1)
            mod = mod + gate * self.gate_factor

        if self.smooth_factor > 0:
            mod = torch.log((1 - self.smooth_factor) * mod.exp() + self.smooth_factor)

        mod = self._validate_logits(mod)

        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)) if context is not None else mod)
        return mod

    # ---------------- Unified modulate ----------------
    def _modulate(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.log_temperature.exp().item(), self.min_temperature)
        mod = self._apply_context(self.logits, context) / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)
        self._cache_set(key, mod.detach())
        return mod

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context=None, temperature=None):
        return super().sample(context=context, temperature=temperature)

    def log_prob(self, x, context=None, temperature=None):
        return super().log_prob(x, context=context, temperature=temperature)

    def log_matrix(self, context=None, temperature=None):
        return super().log_matrix(context=context, temperature=temperature)

    def expected_probs(self, context=None, temperature=None):
        return super().expected_probs(context=context, temperature=temperature)

    def mode(self, context=None, temperature=None):
        return super().mode(context=context, temperature=temperature)

    # ---------------- EM-style / gradient update ----------------
    @torch.no_grad()
    def update(self, new_logits: Optional[torch.Tensor] = None, posterior: Optional[torch.Tensor] = None,
               from_probs: bool = False, update_rate: Optional[float] = None, temperature: Optional[float] = None):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = self._validate_logits(new_logits)
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            mod_logits = self._modulate(temperature=temperature)
            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()


class Transition(DistributionBase):
    """Transition distribution for HSMMs with neural/contextual modulation."""

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
        gate_factor: float = 0.5,
        smooth_factor: float = 0.0,
        cache_limit: int = 32,
        scale: float = 1.0,
        debug: bool = False,
    ):
        target_dim = n_states * n_states
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self.gate_factor = gate_factor
        self.smooth_factor = smooth_factor
        self.temperature = temperature
        self.scale = scale

        # Initialize logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Optional neural gating
        self._context_gate: Optional[nn.Module] = None
        if context_dim is not None:
            hidden = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden, n_states, dtype=DTYPE),
            ).to(self.device, self.dtype)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, n_states), -math.log(n_states), dtype=DTYPE, device=self.device)
        elif mode == "diag_bias":
            m = torch.full((n_states, n_states), 0.1, dtype=DTYPE, device=self.device)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=1, keepdim=True)
            logits = torch.log(m)
        elif mode == "normal":
            logits = torch.randn(n_states, n_states, dtype=DTYPE, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        return self._apply_constraints(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(self._validate_logits(logits))
        self._logits_buffer.copy_(self.logits)
        self._mod_logits_buffer.copy_(self.logits)
        self._invalidate_cache()
        return self.dist_type(logits=self.logits)

    # ---------------- Context modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        mod = super()._apply_context(logits, context)

        # Neural gating
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            while gate.ndim < mod.ndim:
                gate = gate.unsqueeze(-1)
            mod = mod + gate * self.gate_factor

        # Structural constraints & smoothing
        mod = self._apply_constraints(mod)
        if self.smooth_factor > 0:
            mod = torch.log((1 - self.smooth_factor) * mod.exp() + self.smooth_factor / self.n_states)

        mod = self._validate_logits(mod)

        # Cache buffer
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=tuple(range(mod.ndim - 2))) if mod.ndim > 2 else mod)
        return mod

    # ---------------- Structural constraints ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        n = self.n_states
        if self.transition_type == "semi":
            mask = torch.eye(n, dtype=torch.bool, device=logits.device)
            logits[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            tril_mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=logits.device), -1)
            logits[..., tril_mask] = -float("inf")
        return logits

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context=None, temperature=None):
        return super().sample(context=context, temperature=temperature)

    def log_prob(self, x, context=None, temperature=None):
        return super().log_prob(x, context=context, temperature=temperature)

    def log_matrix(self, context=None, temperature=None):
        return super().log_matrix(context=context, temperature=temperature)

    def expected_probs(self, context=None, temperature=None):
        return super().expected_probs(context=context, temperature=temperature)

    def mode(self, context=None, temperature=None):
        return super().mode(context=context, temperature=temperature)

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(self, new_logits=None, posterior=None, from_probs=False, update_rate=None, temperature=None):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = self._apply_constraints(new_logits)
            self.logits.data.copy_(self._validate_logits(new_logits))
            self._logits_buffer.copy_(self.logits)
            self._mod_logits_buffer.copy_(self.logits)
            self._invalidate_cache()
            return

        # EM-like posterior update
        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            logits = self.logits.unsqueeze(0) if posterior.ndim == 2 else self.logits.unsqueeze(0).unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or self.temperature), dim=-1)
            loss = -(posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            for p in self.parameters():
                if p.grad is not None:
                    p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()

