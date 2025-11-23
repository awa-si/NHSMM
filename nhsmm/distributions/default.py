# nhsmm/distributions/default.py

import math
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Bernoulli, Laplace, MultivariateNormal,
    Normal, Independent, Poisson, StudentT
)

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm.tools import constraints


class Categorical(Distribution):
    """
    Differentiable Categorical distribution with:
    - Gumbel-Softmax sampling (hard and relaxed)
    - Batch/time-safe handling
    - Temperature scaling
    - Compatible with torch.distributions API
    """

    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }
    has_rsample = True

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        validate_args=None,
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs")
        if probs is not None:
            logits = torch.log(probs.clamp_min(EPS))
        self._logits = logits
        self.tau = tau
        batch_shape = logits.shape[:-1]
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([]), validate_args=validate_args)

    # ---------------- Properties ----------------
    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @logits.setter
    def logits(self, value: torch.Tensor):
        self._logits = value

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self._logits, dim=-1)

    @probs.setter
    def probs(self, p: torch.Tensor):
        if p.shape != self._logits.shape:
            raise ValueError(f"Expected probs shape {self._logits.shape}, got {p.shape}")
        self._logits = torch.log(p.clamp_min(EPS))

    @property
    def support(self):
        K = self._logits.size(-1)
        return torch.distributions.constraints.integer_interval(0, K - 1)

    # ---------------- Log probability ----------------
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.long()
        flat_logits = self._logits.reshape(-1, self._logits.size(-1))
        flat_value = value.reshape(-1)
        logp = F.log_softmax(flat_logits, dim=-1)
        return logp.gather(-1, flat_value.unsqueeze(-1)).squeeze(-1).reshape(value.shape)

    # ---------------- Internal Gumbel ----------------
    def _gumbel_logits(self, sample_shape=torch.Size(), generator=None):
        shape = sample_shape + self._logits.shape
        g = -torch.log(-torch.log(
            torch.rand(shape, device=self._logits.device, generator=generator).clamp_min(EPS)
        ))
        return (self._logits.expand(shape).contiguous() + g) / self.tau

    # ---------------- Gumbel-Softmax Sampling ----------------
    def _sample_gumbel_softmax(self, sample_shape=torch.Size(), hard: bool = True, generator=None):
        y_soft = F.softmax(self._gumbel_logits(sample_shape, generator), dim=-1)
        if not hard:
            return y_soft
        # Hard one-hot with gradient trick
        y_hard = F.one_hot(y_soft.argmax(-1), num_classes=y_soft.size(-1)).to(y_soft.dtype)
        return (y_hard - y_soft).detach() + y_soft

    def sample(self, sample_shape=torch.Size(), hard: bool = True, tau: Optional[float] = None, generator=None):
        old_tau = self.tau
        if tau is not None:
            self.tau = tau
        y = self._sample_gumbel_softmax(sample_shape, hard=hard, generator=generator)
        self.tau = old_tau
        return y

    def rsample(self, sample_shape=torch.Size(), tau: Optional[float] = None, generator=None):
        return self.sample(sample_shape, hard=False, tau=tau, generator=generator)


class DistributionBase(nn.Module):
    """Base class for context-modulated HSMM parameters with batch/sequence support."""
    _dist_type: type = None  # To be set in subclass, e.g., Categorical

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
        layer_norm: bool = False,
        batch_norm: bool = False,
        debug: bool = False,
    ):
        super().__init__()

        # Core dimensions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # only for compatibility, will be removed
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_delta = max_delta
        self.debug = debug

        # Activations
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)
        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        # Context network
        self.context_net: Optional[nn.Module] = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim, dtype=DTYPE),
            )
            self._init_weights(self.context_net)

        # Projection layer (lazy initialization)
        self._proj: Optional[nn.Linear] = None
        if allow_projection and context_dim is not None:
            self._proj = nn.Linear(context_dim, context_dim, dtype=DTYPE)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)

        # Adapters
        self.temporal_adapter = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False, dtype=DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter = (
            nn.Linear(target_dim, target_dim, bias=False, dtype=DTYPE)
            if spatial_adapter else None
        )
        for adapter in (self.temporal_adapter, self.spatial_adapter):
            if adapter is not None:
                nn.init.xavier_uniform_(adapter.weight)

        # Delta scale
        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE))

        # BatchNorm
        self._batchnorm: Optional[nn.BatchNorm1d] = None

        # Temperature
        self.log_temperature = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        # Cache
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int64))
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.cache_grad_safe = cache_grad_safe

        # Base logits
        self.logits = nn.Parameter(torch.zeros(target_dim, dtype=DTYPE))

    # ---------------- Helpers ----------------
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

    # ---------------- Context modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        if context is None:
            delta = torch.zeros_like(base)
            return self._apply_constraints(base + delta)

        # Standardize to [B, S, H]
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:
            context = context.unsqueeze(1)
        elif context.ndim != 3:
            raise ValueError(f"Unsupported context ndim {context.ndim}")
        B, S, H = context.shape

        # Determine context_dim dynamically
        if self.context_dim is None:
            self.context_dim = H

        # Lazy projection if shape mismatch
        if H != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"Context dim mismatch: expected {self.context_dim}, got {H}")
            if self._proj is None or self._proj.in_features != H:
                self._proj = nn.Linear(H, self.context_dim, dtype=DTYPE)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
                self._invalidate_cache()
            context = self._proj(context)

        # Compute delta
        delta = torch.zeros((B, S, base.shape[-1]), dtype=base.dtype, device=base.device)
        if self.context_net is not None:
            delta += self.context_net(context)

        # L2 normalization
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)

        # Norm layers
        if self.layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if self.batch_norm:
            flat = delta.flatten(0, -2)
            if self._batchnorm is None or self._batchnorm.num_features != flat.shape[-1]:
                self._batchnorm = nn.BatchNorm1d(flat.shape[-1], affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(flat).view(delta.shape)

        # Adapters
        if not skip_adapters:
            if self.temporal_adapter:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter:
                delta = self.spatial_adapter(delta)

        # Activation, scaling, clamping
        delta = self.final_activation_fn(delta) * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", 1e6), getattr(self, "max_delta", 1e6))

        # Broadcast base to [B, S, K]
        if base.ndim == 1:
            base = base.view(1, 1, -1)
        elif base.ndim == 2:
            base = base.unsqueeze(0)

        if grad_scale is not None:
            delta = delta * grad_scale

        return self._apply_constraints(base + delta)

    # ---------------- Modulation ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False,
    ) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(self.logits, context)
        temp = torch.exp(self.log_temperature) if temperature is None else torch.as_tensor(temperature, dtype=mod.dtype)
        mod = mod / temp.clamp_min(EPS)
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)

        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Distribution parameters ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        tau_val = tau or torch.exp(self.log_temperature)
        mod_logits = logits / tau_val.clamp_min(EPS)
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = self._validate_logits(mod_logits)
        return {"logits": mod_logits, **kwargs}

    # ---------------- Distribution API ----------------
    def forward(self, log=False, return_dist=False, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, tau=temperature, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod, dim=-1) if log else F.softmax(mod, dim=-1)

    def log_prob(self, x: torch.Tensor, context=None, temperature=None, **dist_kwargs):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), tau=temperature, **dist_kwargs))
        if hasattr(dist, "log_prob"):
            return dist.log_prob(x)
        return F.log_softmax(self._modulate(context, temperature), dim=-1).gather(-1, x.long())

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs: Any):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, tau=temperature, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod, dim=-1)

    def sample(self, context=None, temperature=None, **dist_kwargs):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), tau=temperature, **dist_kwargs))
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

    def expected_probs(self, context=None, temperature=None, return_dist=False, **dist_kwargs):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, tau=temperature, **dist_kwargs))
        return dist if return_dist else F.softmax(mod, dim=-1)

    def mode(self, context=None, temperature=None, return_dist=False, **dist_kwargs):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), tau=temperature, **dist_kwargs))
        if return_dist: return dist
        if hasattr(dist, "mode"): return dist.mode
        if hasattr(dist, "probs"): return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    # ---------------- Utilities ----------------
    @torch.no_grad()
    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        c = context.detach().float()
        return str((float(c.mean().item()), float(c.std().item()), int(c.numel()), int(self._param_version.item())))

    @torch.no_grad()
    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        if not self.cache_enabled: return None
        out = self._cache.get(key)
        if out is not None:
            self._cache.move_to_end(key)
        return out

    @torch.no_grad()
    def _cache_set(self, key: str, value: torch.Tensor):
        if not self.cache_enabled: return
        self._cache[key] = value.detach().clone()
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _invalidate_cache(self):
        self._cache.clear()
        self._param_version += 1

    # ---------------- Placeholder methods ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain non-finite values.")
        return logits

    @property
    def dist_type(self):
        return self._dist_type

    @dist_type.setter
    def dist_type(self, value):
        self._dist_type = value

    @torch.no_grad()
    def update(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def initialize(self, mode="uniform", **_):
        return self


class Initial(DistributionBase):
    """
    Context-aware, differentiable Initial-state distribution for HSMMs.
    Supports context gates, residuals, and low-rank factors.
    """

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: int = 64,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        rank: Optional[int] = None,
        cache_limit: int = 64,
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
        self.rank = rank
        self.init_mode = init_mode
        self.temperature = temperature

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits.data.copy_(init_logits)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context-specific gates / low-rank factors
        self.context_gate: Optional[nn.Module] = None
        self.residual_gate: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None

        if context_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE)

        # Validate low-rank configuration
        if (self._U is not None or self._V is not None) and self.rank is None:
            raise ValueError("Rank must be specified if using low-rank factors.")

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
            raise ValueError(f"Unknown init_mode '{mode}'")
        return self._validate_logits(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._dist_type(logits=logits)

    # ---------------- Context modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if context is None or not any([self.context_gate, self.residual_gate, self._U]):
            return super()._apply_context(base, context=context, grad_scale=grad_scale)

        K = self.n_states
        dtype, device = base.dtype, base.device

        # Standardize context to [B,T,H]
        if context.ndim == 1:          # [H]
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:        # [T,H]
            context = context.unsqueeze(0)
        B, T, H = context.shape
        ctx_flat = context.reshape(B * T, H)

        # Compute delta [B*T, K]
        delta = torch.zeros(B * T, K, dtype=dtype, device=device)
        if self.context_gate:
            delta += self.context_gate(ctx_flat)
        if self.residual_gate:
            delta += self.residual_gate(ctx_flat)
        if self._U is not None and self._V is not None:
            U = self._U(ctx_flat).view(B * T, K, self.rank)
            V = self._V(ctx_flat).view(B * T, 1, self.rank)
            delta += (U * V).sum(-1)

        if grad_scale is not None:
            delta = delta * grad_scale

        delta = delta.view(B, T, K)

        # Broadcast base to [B,T,K]
        if base.ndim == 1:
            base = base.view(1, 1, K)
        elif base.ndim == 2:
            base = base.unsqueeze(0)

        return self._apply_constraints(base + delta)

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, **kwargs) -> dict:
        tau = kwargs.get("tau") or kwargs.get("temperature") or self.temperature
        return {"logits": logits, "tau": tau}

    def _get_dist(self, context=None, **kwargs) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=kwargs.get("temperature"))
        return self._dist_type(**self._dist_params(mod_logits, **kwargs))

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).sample(**kwargs)

    def log_prob(self, x, context=None, temperature=None):
        return self._get_dist(context=context, temperature=temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None):
        return self._get_dist(context=context, temperature=temperature).probs

    def mode(self, context=None, temperature=None):
        dist = self._get_dist(context=context, temperature=temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    # ---------------- EM-style / Learnable update ----------------
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
            mod_logits = self._modulate(context=context, temperature=temperature)
            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(lr * p.grad)
            self._invalidate_cache()


class Duration(DistributionBase):
    """
    Context-aware categorical duration distribution per state for HSMMs.
    Compatible with Initial for alpha_tensor computations.
    Internal shape: [K, Dmax].
    """

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        context_dim: Optional[int] = None,
        hidden_dim: int = 64,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        rank: Optional[int] = None,
        cache_limit: int = 32,
        debug: bool = False,
    ):
        super().__init__(
            target_dim=n_states * max_duration,
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
        self.rank = rank
        self.init_mode = init_mode
        self.temperature = temperature

        # Base logits: [K, Dmax]
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / residual / low-rank
        self.context_gate: Optional[nn.Module] = None
        self.residual_gate: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None

        if context_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE)
            w = w.unsqueeze(0).repeat(n_states, 1)
            w /= w.sum(dim=1, keepdim=True)
            logits = torch.log(w)
        elif mode == "normal":
            logits = torch.randn(n_states, max_duration, dtype=DTYPE) * 0.1
            logits -= torch.arange(max_duration, dtype=DTYPE) * 0.05
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
        return self._get_dist()

    # ---------------- Context modulation ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, grad_scale: Optional[float] = None) -> torch.Tensor:
        K, Dmax = self.n_states, self.max_duration
        if context is None or not any([self.context_gate, self.residual_gate, self._U]):
            return base

        B, T, H = (1, 1, context.shape[0]) if context.ndim == 1 else ((1, context.shape[0], context.shape[1]) if context.ndim == 2 else context.shape)
        ctx_flat = context.reshape(B * T, H) if context.ndim > 1 else context.unsqueeze(0)

        delta = torch.zeros(B * T, K, Dmax, dtype=DTYPE)

        if self.context_gate:
            delta += self.context_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, Dmax)
        if self.residual_gate:
            delta += self.residual_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, Dmax)
        if self._U is not None and self._V is not None:
            U = self._U(ctx_flat).view(B * T, K, self.rank)
            V = self._V(ctx_flat).view(B * T, 1, self.rank)
            delta += (U * V).sum(-1).unsqueeze(-1).expand(-1, -1, Dmax)

        if grad_scale is not None:
            delta = delta * grad_scale

        return base.unsqueeze(0).unsqueeze(0) + delta.view(B, T, K, Dmax)

    # ---------------- Modulate ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(self.logits, context)
        tau = temperature if temperature is not None else self.temperature
        mod = mod / max(tau, EPS)
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)

        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        tau_val = tau if tau is not None else self.temperature
        if not torch.is_tensor(tau_val):
            tau_val = torch.tensor(tau_val, dtype=logits.dtype)
        mod_logits = logits / tau_val.clamp_min(EPS)
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = self._validate_logits(mod_logits)
        return {"logits": mod_logits, **kwargs}

    def _get_dist(self, context=None, temperature=None) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=temperature)
        tau = temperature or self.temperature
        return self._dist_type(**self._dist_params(mod_logits, tau=tau))

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).sample(**kwargs)

    def rsample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).rsample()

    def log_prob(self, x, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).probs

    def mode(self, context=None, temperature=None, **kwargs):
        dist = self._get_dist(context=context, temperature=temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    # ---------------- EM-style / Learnable update ----------------
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
            mod_logits = self._modulate(context=context, temperature=temperature)
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
    """
    Context-aware categorical transition distribution for HSMMs.

    - Supports context gates, residuals, low-rank factors
    - Structural transition constraints
    - Batch / sequence aware: returns [K,K] or [B,T,K,K]
    - Temperature scaling + caching; does not forward `tau` into Categorical
    """
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: int = 64,
        rank: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
        gate_factor: float = 0.5,
        smooth_factor: float = 0.0,
        cache_limit: int = 32,
        debug: bool = False,
        allow_projection: bool = True,
    ):
        super().__init__(
            target_dim=n_states * n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.rank = rank
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self.gate_factor = gate_factor
        self.smooth_factor = smooth_factor
        self.temperature = temperature
        self.allow_projection = allow_projection

        # base logits [K,K]
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # context modules
        self.context_gate: Optional[nn.Module] = None
        self.residual_gate: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None
        self._proj: Optional[nn.Linear] = None

        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, n_states), -math.log(n_states), dtype=DTYPE)
        elif mode == "diag_bias":
            m = torch.full((n_states, n_states), 0.1, dtype=DTYPE)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=1, keepdim=True)
            logits = torch.log(m)
        elif mode == "normal":
            logits = torch.randn(n_states, n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        return self._apply_constraints(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._get_dist()

    # ---------------- Context modulation ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        n = self.n_states
        dtype = base.dtype

        # ensure base [K,K]
        if base.ndim == 1:
            base = base.view(n, n)
        elif base.ndim == 2 and base.shape == (n, n):
            pass
        else:
            raise ValueError(f"Unexpected base shape {base.shape}")

        if context is None or not any([self.context_gate, self.residual_gate, self._U]):
            return self._apply_constraints(base)

        # standardize context to [B,T,H]
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:
            context = context.unsqueeze(0)
        B, T, H = context.shape
        ctx_flat = context.reshape(B * T, H)

        # optional projection
        if self.context_dim is None:
            self.context_dim = H
        if H != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"Context dim mismatch: expected {self.context_dim}, got {H}")
            if self._proj is None or self._proj.in_features != H:
                self._proj = nn.Linear(H, self.context_dim, dtype=DTYPE)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
            ctx_flat = self._proj(ctx_flat)
            context = ctx_flat.view(B, T, self.context_dim)
            ctx_flat = ctx_flat.view(B * T, self.context_dim)

        # additive delta [B*T, K*K]
        delta = torch.zeros(B * T, n * n, dtype=dtype)

        if self.context_gate is not None:
            row = self.context_gate(ctx_flat)
            delta += row.unsqueeze(2).expand(-1, -1, n).reshape(B * T, n * n)
        if self.residual_gate is not None:
            row = self.residual_gate(ctx_flat)
            delta += row.unsqueeze(2).expand(-1, -1, n).reshape(B * T, n * n)
        if self.rank is not None and self._U is not None and self._V is not None:
            U = self._U(ctx_flat).view(B * T, n, self.rank)
            V = self._V(ctx_flat).view(B * T, 1, self.rank)
            delta += (U * V).sum(-1).reshape(B * T, n * n)

        # reshape to [B,T,K,K] and add base
        mod = base.view(1, 1, n, n) + delta.view(B, T, n, n)
        mod = self._apply_constraints(mod)

        # squeeze singleton dims
        if B == 1 and T == 1:
            return mod.squeeze(0).squeeze(0)
        if B == 1:
            return mod.squeeze(0)
        return mod

    # ---------------- Modulate ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        temp_val = temperature or self.temperature
        temp_tensor = torch.as_tensor(float(temp_val), dtype=DTYPE)
        key = f"{self._context_hash(context)}-T{float(temp_tensor.mean()):.6e}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(self.logits, context)
        mod = mod / temp_tensor.clamp_min(EPS)
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)
        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, **kwargs) -> dict:
        return {"logits": self._validate_logits(logits)}

    def _get_dist(self, context=None, temperature=None):
        return self._dist_type(**self._dist_params(self._modulate(context, temperature)))

    # ---------------- Sampling / log-prob / mode ----------------
    def sample(self, context=None, temperature=None, hard=True, **kwargs):
        return self._get_dist(context, temperature).sample(hard=hard)

    def log_prob(self, x, context=None, temperature=None, **kwargs):
        return self._get_dist(context, temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None):
        return self._get_dist(context, temperature).probs

    def mode(self, context=None, temperature=None, **kwargs):
        dist = self._get_dist(context, temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    # ---------------- Structural constraints ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        n = self.n_states
        out = logits.clone()
        if self.transition_type == "semi":
            mask = torch.eye(n, dtype=torch.bool)
            out[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            tril_mask = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
            out[..., tril_mask] = -float("inf")
        return out

    # ---------------- EM / learnable update ----------------
    @torch.no_grad()
    def update(self, new_logits=None, posterior=None, from_probs=False, update_rate=None, temperature=None, context=None):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = self._apply_constraints(new_logits)
            validated = self._validate_logits(new_logits)
            self.logits.data.copy_(validated)
            self._logits_buffer.copy_(validated)
            self._mod_logits_buffer.copy_(validated)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            mod_logits = self._modulate(context, temperature)
            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()


class Emission(DistributionBase):
    """Emission distribution supporting Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson."""

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
        temperature: float = 1.0,
        debug: bool = False,
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
        self.temperature = temperature
        self.min_covar = min_covar
        self.seed = seed
        self.dof = dof

        # Buffers
        self.register_buffer(
            "_emission_means",
            torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device),
        )
        self.register_buffer(
            "_emission_covs",
            torch.eye(n_features, dtype=DTYPE, device=self.device)
            .unsqueeze(0)
            .repeat(n_states, 1, 1),
        )
        self.register_buffer(
            "_emission_params",
            torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device),
        )

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"categorical", "bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"laplace", "studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

        # Optional MLP context modulation
        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states * n_features),
            )
        else:
            self.context_gate = None

    def _dist_params(self, tensor: torch.Tensor, temperature: Optional[float] = None, context: Optional[torch.Tensor] = None, grad_safe: bool = False, **kwargs) -> dict:
        mod_tensor = self._modulate(tensor, context=context, temperature=temperature, grad_safe=grad_safe)
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            return {"logits": mod_tensor, **kwargs}
        else:
            if self.emission_type == "gaussian":
                var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
                cov = torch.diag_embed(var)
            else:
                scale = torch.clamp(self.scale_param, min=self.min_covar)
                cov = torch.diag_embed(scale**2)
            return {"means": mod_tensor, "cov": cov, **kwargs}

    def _get_dist(
        self,
        X: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
        max_jitter: int = 5,
    ):
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std().item() < EPS:
                X = X + 1e-3 * torch.randn_like(X)

        if posterior is not None and X is not None:
            w = posterior.clamp_min(EPS)
            w_sum = w.sum(dim=0) + EPS
        else:
            w = w_sum = None

        theta_vec = theta.mean(dim=0) if theta is not None and theta.ndim > 1 else theta

        # Continuous
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self._emission_means.clone()
            if X is not None and w is not None:
                means = (w.T @ X) / w_sum.unsqueeze(1)
            if theta_vec is not None:
                means += theta_scale * theta_vec.unsqueeze(0)
            means = self._modulate(means, context=context, temperature=temperature)

            if etype == "gaussian":
                if X is not None and w is not None:
                    diff = X[:, None, :] - means[None, :, :]
                    covs = torch.einsum("nkf,nkd->kfd", diff * w[:, :, None], diff) / w_sum[:, None, None]
                else:
                    covs = self._emission_covs.clone()
                covs += self.min_covar * torch.eye(F, device=self.device)[None, :]
                # Ensure positive-definite
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0:
                            break
                        covs[k] += jitter * torch.eye(F, device=self.device)
                        jitter *= 2
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                dist = Independent(MultivariateNormal(means, covariance_matrix=covs), 1)
                return dist

            else:  # Laplace / StudentT
                if X is not None and w is not None:
                    scales = ((X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]).sum(dim=0) / w_sum[:, None]
                else:
                    scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()
                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                dist = Independent(dist_cls(loc=means, scale=scales), 1)
                return dist

        # Discrete
        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None and w is not None:
                if etype == "categorical":
                    logits = torch.zeros((K, F), device=self.device)
                    for k in range(K):
                        counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                        logits[k] = torch.log(counts.clamp_min(EPS) / counts.sum().clamp_min(EPS))
                else:
                    rate = (w.T @ X.float()) / w_sum[:, None]
                    logits = torch.log(rate.clamp_min(EPS))
            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            if theta_vec is not None:
                logits += theta_scale * theta_vec.unsqueeze(0)
            logits = self._modulate(logits, context=context, temperature=temperature)
            self._emission_params.copy_(logits)

            if etype == "categorical":
                return Categorical(logits=logits)
            elif etype == "bernoulli":
                return Independent(Bernoulli(logits=logits), 1)
            else:  # Poisson
                return Independent(Poisson(rate=torch.exp(logits)), 1)

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ------------------- Context / Modulation -------------------
    def _apply_context(self, tensor: torch.Tensor, context: Optional[torch.Tensor] = None, grad_scale: Optional[float] = None):
        """
        Apply context modulation to a tensor of shape [K,F] or [N,K,F] (or compatible shapes).
        """
        K, F = self.n_states, self.n_features
        original_shape = tensor.shape
        # Flatten last two dimensions for base context application
        flat_tensor = tensor.reshape(-1, K * F)
        modulated = super()._apply_context(flat_tensor, context, grad_scale=grad_scale)
        # Restore original shape
        return modulated.reshape(*original_shape)

    def _modulate(self, tensor: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float | torch.Tensor] = None, grad_safe: bool = False):
        """
        Apply context gate, adaptive scaling, and temperature to tensor.
        Supports [K,F] and [N,K,F].
        """
        K, F = self.n_states, self.n_features
        original_shape = tensor.shape
        flat_tensor = tensor.reshape(-1, K * F)

        # Base context modulation
        mod_base = super()._apply_context(flat_tensor, context)

        # Apply emission-specific context_gate
        if context is not None and self.context_gate is not None:
            if mod_base.shape[0] > 1:
                ctx = context.expand(mod_base.shape[0], -1) if context.ndim == 2 and context.size(0) == 1 else context
            else:
                ctx = context
            gate_out = self.context_gate(ctx).reshape_as(mod_base)
            mod_base = mod_base + gate_out
            if self.adaptive_scale:
                norm = ctx.norm(dim=-1).mean() if ctx.ndim > 1 else ctx.norm()
                norm = max(norm.item(), EPS)
                mod_base = mod_base * (self.scale / norm)

        # Temperature scaling
        if temperature is not None:
            temp_val = float(temperature) if isinstance(temperature, (float, int)) else temperature.clamp_min(EPS).mean().item()
            mod_base = mod_base / max(temp_val, EPS)

        # Gradient-safe option
        if grad_safe:
            mod_base = mod_base.detach() + tensor.reshape(-1, K * F) - tensor.reshape(-1, K * F).detach()

        return mod_base.reshape(*original_shape)

    # ------------------- Distribution Construction -------------------
    def _get_dist(
        self,
        X: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
        max_jitter: int = 5,
    ):
        """
        Compute torch.distributions object with context and temperature modulation.
        Automatically sets self.dist_type.
        """
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        # Prepare data
        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std().item() < EPS:
                X = X + 1e-3 * torch.randn_like(X)

        # Posterior weighting
        if posterior is not None and X is not None:
            w = posterior.clamp_min(EPS)
            w_sum = w.sum(dim=0) + EPS
        else:
            w = w_sum = None

        # Theta modulation
        theta_vec = theta.mean(dim=0) if theta is not None and theta.ndim > 1 else theta

        # ---------------- Continuous ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self._emission_means.clone()
            if X is not None and w is not None:
                means = (w.T @ X) / w_sum.unsqueeze(1)
            if theta_vec is not None:
                means += theta_scale * theta_vec.unsqueeze(0)
            means = self._modulate(means, context=context, temperature=temperature)

            if etype == "gaussian":
                if X is not None and w is not None:
                    diff = X[:, None, :] - means[None, :, :]
                    covs = torch.einsum("nkf,nkd->kfd", diff * w[:, :, None], diff) / w_sum[:, None, None]
                else:
                    covs = self._emission_covs.clone()
                I = torch.eye(F, device=self.device)
                covs += self.min_covar * I[None, :]
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0: break
                        covs[k] += jitter * I
                        jitter *= 2
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                dist = MultivariateNormal(means, covariance_matrix=covs)
                self.dist_type = lambda *a, **kw: Independent(MultivariateNormal(*a, **kw), 1)
                return dist

            else:  # Laplace / StudentT
                if X is not None and w is not None:
                    scales = ((X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]).sum(dim=0) / w_sum[:, None]
                else:
                    scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()
                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                dist = Independent(dist_cls(loc=means, scale=scales), 1)
                self.dist_type = lambda *a, **kw: Independent(dist_cls(*a, **kw), 1)
                return dist

        # ---------------- Discrete ----------------
        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None and w is not None:
                if etype == "categorical":
                    logits = torch.zeros((K, F), device=self.device)
                    for k in range(K):
                        counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                        logits[k] = torch.log(counts.clamp_min(EPS) / counts.sum().clamp_min(EPS))
                else:
                    rate = (w.T @ X.float()) / w_sum[:, None]
                    logits = torch.log(rate.clamp_min(EPS))
            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            if theta_vec is not None:
                logits += theta_scale * theta_vec.unsqueeze(0)
            logits = self._modulate(logits, context=context, temperature=temperature)
            self._emission_params.copy_(logits)

            if etype == "categorical":
                dist = Categorical(logits=logits)
                self.dist_type = Categorical
            elif etype == "bernoulli":
                dist = Independent(Bernoulli(logits=logits), 1)
                self.dist_type = lambda *a, **kw: Independent(Bernoulli(*a, **kw), 1)
            else:  # Poisson
                dist = Independent(Poisson(rate=torch.exp(logits)), 1)
                self.dist_type = lambda *a, **kw: Independent(Poisson(*a, **kw), 1)

            return dist

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ------------------- Forward / Distribution -------------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
    ):
        """
        Produce distribution or parameters for emissions.
        Context, temperature, and theta modulation are applied once.
        """
        dist = self._get_dist(
            X=None,
            context=context,
            theta=None,
            theta_scale=0.1,
            temperature=temperature,
            max_jitter=5,
        )

        # Update buffers for consistency based on emission type
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # Continuous distributions
            if self.emission_type == "gaussian":
                self._emission_covs.copy_(self._emission_covs)
            else:  # laplace / studentt
                scale = self.scale_param.clamp_min(self.min_covar)
                self._emission_covs.copy_(torch.diag_embed(scale**2))
        else:
            # Discrete distributions: categorical / bernoulli / poisson
            self._emission_params.copy_(self._emission_params)
            # Sync learnable attribute if exists
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        # Return either distribution object or parameters
        if return_dist:
            return dist
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        return self._emission_params

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        samples = dist.rsample((n_samples,)) if getattr(dist, "has_rsample", False) else dist.sample((n_samples,))
        return samples.to(dtype=DTYPE, device=self.device)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        N, K, F = x.shape[0], self.n_states, self.n_features
        etype = self.emission_type

        if etype in {"gaussian", "laplace", "studentt"}:
            x_exp = x.unsqueeze(1).expand(-1, K, -1) if x.ndim == 2 else x
            return dist.log_prob(x_exp)
        elif etype == "categorical":
            logits = getattr(self, "logits", self._emission_params)
            x_exp = x.long().unsqueeze(1).expand(-1, K, -1) if x.ndim == 2 else x.long()
            log_probs = F.log_softmax(logits, dim=-1).unsqueeze(0)
            return torch.gather(log_probs.expand(N, K, F), -1, x_exp).squeeze(-1)
        elif etype == "bernoulli":
            logits = getattr(self, "logits", self._emission_params)
            x_exp = x.unsqueeze(1).expand(-1, K, F)
            return -F.binary_cross_entropy_with_logits(logits.unsqueeze(0).expand(N, K, F), x_exp, reduction="none").sum(-1)
        elif etype == "poisson":
            rate = torch.exp(getattr(self, "log_rate", self._emission_params))
            x_exp = x.unsqueeze(1).expand(-1, K, F)
            log_probs = -rate.unsqueeze(0).expand(N, K, F) + x_exp * torch.log(rate.unsqueeze(0).expand(N, K, F).clamp_min(EPS)) - torch.lgamma(x_exp + 1)
            return log_probs.sum(-1)
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Returns log-likelihood matrix for all states and observations.
        Can optionally incorporate temperature scaling.
        """
        log_probs = super().log_matrix(context=context, temperature=temperature)
        if temperature is not None and temperature != 1.0:
            log_probs = log_probs / temperature
        return log_probs

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Returns expected probabilities for discrete distributions.
        For continuous distributions, this returns means as proxy.
        """
        probs = super().expected_probs(context=context, temperature=temperature)
        return probs

    def parameters_tensor(self) -> torch.Tensor:
        """
        Return the core learnable parameters for the emission distribution.
        For continuous: (means, covariances)
        For discrete: logits or rates
        """
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        return self._emission_params

    @torch.no_grad()
    def update(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        update_rate: float = 0.5,
        temperature: Optional[float] = None,
        init_spread: float = 0.1,
        max_jitter: int = 5,
        rank=None
    ):
        """
        Compute new distribution parameters via _get_dist and update buffers via EMA.
        Fully relies on self.dist_type and internal buffers; no base_dist usage.
        """
        new_dist = self._get_dist(
            X=X,
            posterior=posterior,
            theta=theta,
            context=context,
            theta_scale=theta_scale,
            temperature=temperature,
            max_jitter=max_jitter,
        )

        etype = self.emission_type

        if etype == "gaussian":
            # EMA update for continuous mean and covariance
            self._emission_means.mul_(1 - update_rate).add_(update_rate * self._emission_means)
            self._emission_covs.mul_(1 - update_rate).add_(update_rate * self._emission_covs)
            self.mu.copy_(self._emission_means)
            cov_diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)
            self.log_var.copy_(torch.log(torch.clamp(cov_diag, min=EPS)))

        elif etype in {"laplace", "studentt"}:
            # EMA update for loc and scale
            self._emission_means.mul_(1 - update_rate).add_(update_rate * self.loc)
            diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * self.scale_param**2
            self._emission_covs.copy_(torch.diag_embed(diag))
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(diag, min=EPS)))

        else:  # categorical / bernoulli / poisson
            self._emission_params.mul_(1 - update_rate).add_(update_rate * getattr(self, "logits", getattr(self, "log_rate", self._emission_params)))
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    @torch.no_grad()
    def initialize(self, X: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, mode: str = "data", iters: int = 15, theta: Optional[torch.Tensor] = None, theta_scale: float = 0.1, temperature: Optional[float] = None):
        if X is None or mode == "default":
            return self.forward(context=context, temperature=temperature, return_dist=True)

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

        # Initialize means/covariances
        if mode == "data":
            mean = Xf.mean(dim=0)
            cov = (Xf - mean).T @ (Xf - mean) / max(N - 1, 1) + self.min_covar * torch.eye(F, device=device)
            means = mean.expand(K, F).clone()
            covs = cov.unsqueeze(0).expand(K, F, F).clone()
        elif mode == "kmeans":
            centers, labels = run_kmeans(Xf, K, iters)
            means = centers
            covs = torch.stack([torch.cov(Xf[labels == k].T) + self.min_covar * torch.eye(F, device=device) for k in range(K)])
        else:
            raise ValueError(f"Unsupported initialization mode: {mode}")

        # Theta & context modulation
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            means += theta_scale * theta_vec.unsqueeze(0)
        means = self._modulate(means, context=context, temperature=temperature)

        self._emission_means.copy_(means)
        self._emission_covs.copy_(covs)

        # Construct distribution
        if self.emission_type == "gaussian":
            self.mu.copy_(means)
            cov_diag = torch.diagonal(covs, dim1=-2, dim2=-1)
            cov_diag = cov_diag[:, :F] if cov_diag.shape != self.log_var.shape else cov_diag
            self.log_var.copy_(torch.log(torch.clamp(cov_diag, min=EPS)))
            return Independent(MultivariateNormal(means, covariance_matrix=covs), 1)
        elif self.emission_type in {"laplace", "studentt"}:
            scale = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
            self.loc.copy_(means)
            self.scale_param.copy_(scale)
            dist_cls = Laplace if self.emission_type == "laplace" else StudentT
            return Independent(dist_cls(loc=means, scale=scale), 1)
        else:  # Discrete
            if mode == "data":
                logits = torch.log_softmax(Xf.mean(dim=0).expand(K, F), dim=-1)
            else:
                logits = means
            logits = self._modulate(logits, context=context, temperature=temperature)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(logits)
            self._emission_params.copy_(logits)
            if self.emission_type == "categorical":
                return Categorical(logits=logits)
            elif self.emission_type == "bernoulli":
                return Independent(Bernoulli(logits=logits), 1)
            else:
                return Independent(Poisson(rate=torch.exp(logits)), 1)

