# nhsmm/distributions/base.py
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
    """Base class for context-modulated HSMM parameters."""
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
        layer_norm: bool = False,
        batch_norm: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_delta = max_delta
        self.debug = debug

        # Activation functions
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        # Context network
        self.context_net: Optional[nn.Module] = None
        self._proj: Optional[nn.Linear] = None
        if context_dim is not None and hidden_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.LayerNorm(hidden_dim, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim, dtype=DTYPE, device=self.device),
            )
            self._init_weights(self.context_net)

        # Temporal & spatial adapters
        self.temporal_adapter = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter = (
            nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE)
            if spatial_adapter else None
        )
        for adapter in (self.temporal_adapter, self.spatial_adapter):
            if adapter is not None:
                nn.init.xavier_uniform_(adapter.weight)

        # Delta scale
        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE, device=self.device))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE, device=self.device))

        # BatchNorm placeholder
        self._batchnorm: Optional[nn.BatchNorm1d] = None

        # Temperature parameter
        self.log_temperature = nn.Parameter(torch.tensor(0.0, dtype=DTYPE, device=self.device))

        # Cache
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int64))
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.cache_grad_safe = cache_grad_safe

        # Base logits
        self.logits = nn.Parameter(torch.zeros(target_dim, dtype=DTYPE, device=self.device))

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
                if m.bias is not None: nn.init.zeros_(m.bias)

    # ---------------- Context hooks ----------------
    def _context_gate(self, context_flat: torch.Tensor) -> torch.Tensor:
        return torch.zeros((context_flat.shape[0], self.target_dim), dtype=DTYPE, device=self.device)

    def _context_residual(self, context_flat: torch.Tensor) -> torch.Tensor:
        return self._context_gate(context_flat)

    def _context_low_rank(self, context_flat: torch.Tensor) -> torch.Tensor:
        return self._context_gate(context_flat)

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain non-finite values.")
        return logits

    # ---------------- Cache ----------------
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

    # ---------------- Context modulation ----------------
    # Context shapes:
    # [B, 1, F]   # pooled context
    # [B, S, F]   # sequence-expanded context
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        device, dtype = base.device, base.dtype

        # ------------------ Handle None context ------------------
        if context is None:
            delta = torch.zeros_like(base, device=device, dtype=dtype)
            return self._apply_constraints(base + delta)

        # ------------------ Normalize context to [B, S, H] ------------------
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)  # [1, 1, H]
        elif context.ndim == 2:
            context = context.unsqueeze(1)  # [B, 1, H]
        elif context.ndim != 3:
            raise ValueError(f"Unsupported context ndim {context.ndim}")
        B, S, H = context.shape

        # ------------------ Optional projection ------------------
        if self.context_dim is None:
            self.context_dim = H
        if H != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"Context dim mismatch: expected {self.context_dim}, got {H}")
            if self._proj is None or self._proj.in_features != H:
                self._proj = nn.Linear(H, self.context_dim, device=device, dtype=dtype)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
                self._invalidate_cache()
            context = self._proj(context)

        # ------------------ Compute delta ------------------
        delta = torch.zeros((B, S, base.shape[-1]), device=device, dtype=dtype)
        if self.context_net:
            delta += self.context_net(context)
        delta += self._context_gate(context)
        delta += self._context_residual(context)
        delta += self._context_low_rank(context)

        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)

        # ------------------ Norm layers ------------------
        if self.layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if self.batch_norm:
            flat = delta.flatten(0, -2)
            if self._batchnorm is None or self._batchnorm.num_features != flat.shape[-1]:
                self._batchnorm = nn.BatchNorm1d(flat.shape[-1], affine=True, track_running_stats=False, eps=EPS).to(device, dtype)
            delta = self._batchnorm(flat).view(delta.shape)

        # ------------------ Adapters ------------------
        if not skip_adapters:
            if self.temporal_adapter:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter:
                delta = self.spatial_adapter(delta)

        # ------------------ Activation, scaling, clamping ------------------
        delta = self.final_activation_fn(delta) * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", 1e6), getattr(self, "max_delta", 1e6))

        # ------------------ Broadcast delta to match base ------------------
        if delta.shape != base.shape:
            # Expand safely using expand_as
            try:
                delta = delta.expand_as(base)
            except RuntimeError:
                # fallback: broadcast with broadcasting semantics
                delta = delta + torch.zeros_like(base)

        # ------------------ Apply grad scaling ------------------
        if grad_scale is not None:
            delta = delta * grad_scale

        return self._apply_constraints(base + delta)

    # ---------------- Modulation w/ Temperature ----------------
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

        temp = torch.exp(self.log_temperature) if temperature is None else torch.as_tensor(temperature, device=self.device, dtype=DTYPE)
        mod = mod / temp.clamp_min(EPS)
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)

        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Forward / Distribution ----------------
    def forward(self, log: bool = False, return_dist: bool = False, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs: Any):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod, dim=-1) if log else F.softmax(mod, dim=-1)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs: Any):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), **dist_kwargs))
        if isinstance(dist, Categorical):
            return dist.log_prob(x.view(-1).long()).view(*x.shape)
        return dist.log_prob(x)

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs: Any):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod, dim=-1)

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs: Any):
        mod = self._modulate(context, temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        return dist if return_dist else F.softmax(mod, dim=-1)

    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs: Any):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), **dist_kwargs))
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs: Any):
        dist = self.dist_type(**self._dist_params(self._modulate(context, temperature), **dist_kwargs))
        if return_dist: return dist
        if hasattr(dist, "mode"): return dist.mode
        if hasattr(dist, "probs"): return torch.argmax(dist.probs, dim=-1)
        if hasattr(dist, "logits"): return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)
        raise TypeError(f"Cannot compute mode for distribution type {type(dist)}")

    @property
    def dist_type(self): return self._dist_type

    @torch.no_grad()
    def update(self, *args, **kwargs): pass

    def initialize(self, mode="uniform", **_): return self


class Initial(DistributionBase):
    """
    Context-aware, differentiable Initial-state distribution for HSMMs.

    Features:
    - Compatible with ContextEncoder outputs ([B, F] or [B, 1, F])
    - Supports adapters, layer/batch norm, L2 normalization
    - Differentiable with gradient-safe caching
    - EM-style updates via posterior or new_logits
    - Temperature scaling
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
        self.rank = rank
        self.n_states = n_states
        self.init_mode = init_mode
        self.temperature = temperature

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits.data.copy_(init_logits)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context-specific gates / low-rank factors
        self.context_gate_net: Optional[nn.Module] = None
        self.residual_gate_net: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None

        if context_dim is not None and hidden_dim is not None:
            self.context_gate_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            self.residual_gate_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE, device=self.device)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE, device=self.device)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states,), -math.log(n_states), dtype=DTYPE, device=self.device)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, n_states, dtype=DTYPE, device=self.device)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(n_states, dtype=DTYPE, device=self.device) * 0.1
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

    # ---------------- Context hooks ----------------
    def _context_gate(self, context_flat: torch.Tensor) -> torch.Tensor:
        if self.context_gate_net is None:
            return torch.zeros((context_flat.shape[0], self.n_states), device=self.device, dtype=DTYPE)
        return self.context_gate_net(context_flat)

    def _context_residual(self, context_flat: torch.Tensor) -> torch.Tensor:
        if self.residual_gate_net is None:
            return torch.zeros((context_flat.shape[0], self.n_states), device=self.device, dtype=DTYPE)
        return self.residual_gate_net(context_flat)

    def _context_low_rank(self, context_flat: torch.Tensor) -> torch.Tensor:
        if self._U is None or self._V is None:
            return torch.zeros((context_flat.shape[0], self.n_states), device=self.device, dtype=DTYPE)
        B_T = context_flat.shape[0]
        U = self._U(context_flat).view(B_T, self.n_states, self.rank)
        V = self._V(context_flat).view(B_T, 1, self.rank)
        return (U * V).sum(-1)

    # ---------------- Modulate ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False,
    ) -> torch.Tensor:
        # Compute modulated logits using base class
        mod = super()._modulate(context=context, temperature=temperature, grad_safe=grad_safe)
        return mod

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None) -> dict:
        return {"logits": logits, "tau": tau or self.temperature}

    def _get_dist(self, context=None, temperature=None) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=temperature)
        tau = temperature or self.temperature
        return self._dist_type(**self._dist_params(mod_logits, tau=tau))

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).sample(**kwargs)

    def log_prob(self, x, context=None, temperature=None):
        return self._get_dist(context=context, temperature=temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None):
        dist = self._get_dist(context=context, temperature=temperature)
        return dist.probs

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

    Features:
    - Differentiable, context-modulated logits
    - Supports ContextEncoder outputs ([B, F] or [B, 1, F])
    - Rank decomposition for low-rank context influence
    - Gradient-safe caching and temperature scaling
    - EM-style or learnable updates
    """
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        context_dim: Optional[int] = None,
        hidden_dim: int = 64,
        min_temperature: float = 1e-6,
        smooth_factor: float = 0.01,
        rank: Optional[int] = None,
        gate_factor: float = 0.5,
        temperature: float = 1.0,
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
        self.rank = rank
        self.gate_factor = gate_factor
        self.smooth_factor = smooth_factor
        self.temperature = temperature
        self.min_temperature = min_temperature

        self.register_buffer("_durations", torch.arange(1, max_duration + 1, dtype=DTYPE, device=self.device))

        # Initialize logits
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / residuals / low-rank
        self.context_gate: Optional[nn.Module] = None
        self.residual_gate: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None

        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE, device=self.device)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE, device=self.device)

        # Optional low-rank context matrices
        if context_dim is not None and rank is not None:
            self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE, device=self.device)
            self._V = nn.Linear(context_dim, rank, dtype=DTYPE, device=self.device)

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE, device=self.device)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE, device=self.device)
            w = w.unsqueeze(0).repeat(n_states, 1)
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
        return self._get_dist()

    # ---------------- Context modulation ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        n, Dmax = self.n_states, self.max_duration
        device, dtype = base.device, base.dtype

        # Base [K, Dmax] if 2D
        if base.ndim == 1:
            base = base.view(n, Dmax)
        elif base.ndim == 2 and base.shape[0] == n and base.shape[1] == Dmax:
            base = base
        else:
            raise ValueError(f"Unexpected base shape {base.shape}")

        if context is None:
            return base  # [K, Dmax]

        # Context [B,T,H]
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:
            context = context.unsqueeze(1)
        B, T, H = context.shape
        ctx_flat = context.reshape(B*T, H)

        # Compute delta: [B*T, K, Dmax]
        delta = torch.zeros(B*T, n, Dmax, device=device, dtype=dtype)
        if self.context_gate:
            gate = self.context_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, Dmax)
            delta += gate
        if self.residual_gate:
            residual = self.residual_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, Dmax)
            delta += residual
        if self.rank and self._U and self._V:
            U = self._U(ctx_flat).view(B*T, n, self.rank)
            V = self._V(ctx_flat).view(B*T, 1, self.rank)
            delta += (U * V).sum(-1).unsqueeze(-1).expand(-1, -1, Dmax)

        delta = delta.view(B, T, n, Dmax)
        base_exp = base.unsqueeze(0).unsqueeze(0).expand(B, T, n, Dmax)

        mod = base_exp + delta
        if self.smooth_factor > 0:
            mod = torch.log((1 - self.smooth_factor) * mod.exp() + self.smooth_factor / Dmax)

        return self._validate_logits(mod)

    def _modulate(self, context: Optional[torch.Tensor] = None, temperature: Optional[float | torch.Tensor] = None, grad_safe: bool = False) -> torch.Tensor:
        temp = (
            torch.full_like(self.logits, self.temperature)
            if temperature is None
            else (
                torch.full_like(self.logits, temperature)
                if isinstance(temperature, (float, int))
                else temperature.clamp_min(self.min_temperature)
            )
        )

        key = f"{self._context_hash(context)}-temp{float(temp.mean()):.6f}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        mod = self._apply_context(self.logits, context)
        mod = mod / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)

        self._cache_set(key, mod.detach() if not grad_safe else mod)
        return mod

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None) -> dict:
        return {"logits": logits, "tau": tau or self.temperature}

    def _get_dist(self, context=None, temperature=None) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=temperature)
        tau = temperature or self.temperature
        return self._dist_type(**self._dist_params(mod_logits, tau=tau))

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, hard=True, **kwargs):
        return self._get_dist(context=context, temperature=temperature).sample(hard=hard)

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

    Features:
    - Differentiable, context-modulated logits
    - Supports ContextEncoder outputs ([B, F] or [B, 1, F])
    - Rank decomposition for low-rank context influence
    - Gradient-safe caching and temperature scaling
    - Structural transition constraints (ergodic, semi, left-to-right)
    - EM-style or learnable updates
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

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / residuals / low-rank
        self.context_gate: Optional[nn.Module] = None
        self.residual_gate: Optional[nn.Module] = None
        self._U: Optional[nn.Linear] = None
        self._V: Optional[nn.Linear] = None

        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE, device=self.device),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states, dtype=DTYPE, device=self.device),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE, device=self.device)
                self._V = nn.Linear(context_dim, rank, dtype=DTYPE, device=self.device)

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
        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._get_dist()

    # ---------------- Context modulation ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Applies context modulation safely for Transition logits.
        Reshapes [K*K] → [K,K] or [B,T,K,K] dynamically.
        """
        n = self.n_states
        device, dtype = base.device, base.dtype

        # Flatten base if necessary
        base_flat = base.view(-1) if base.ndim == 2 else base

        # If no context, just return base logits
        if context is None:
            return self._apply_constraints(base_flat.view(n, n) if base_flat.numel() == n*n else base_flat)

        # Ensure context is [B, T, H]
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(1)
        elif context.ndim == 2:
            context = context.unsqueeze(1)
        B, T, H = context.shape
        ctx_flat = context.reshape(B*T, H)

        # Compute delta
        delta = torch.zeros_like(base_flat, device=device, dtype=dtype).unsqueeze(0).expand(B*T, -1)
        if self.context_gate:
            delta += self.context_gate(ctx_flat)
        if self.residual_gate:
            delta += self.residual_gate(ctx_flat)
        if self.rank is not None and self._U is not None and self._V is not None:
            U = self._U(ctx_flat).view(B*T, n, self.rank)
            V = self._V(ctx_flat).view(B*T, 1, self.rank)
            delta += (U * V).sum(-1)

        # Reshape delta back
        target_dim_per_state = n
        delta = delta.view(B, T, n, target_dim_per_state)
        base_exp = base_flat.view(1, 1, n, n).expand(B, T, n, n)

        mod = self._apply_constraints(base_exp + delta)
        return mod

    # ---------------- Unified modulation ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float | torch.Tensor] = None,
        grad_safe: bool = False,
    ) -> torch.Tensor:
        temp = (
            torch.full_like(self.logits, self.temperature)
            if temperature is None
            else torch.full_like(self.logits, temperature)
            if isinstance(temperature, (float, int))
            else temperature.clamp_min(1e-6)
        )
        key = f"{self._context_hash(context)}-temp{float(temp.mean()):.6f}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        mod = self._apply_context(self.logits, context)
        mod = mod / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = self._validate_logits(mod)
        self._cache_set(key, mod.detach() if not grad_safe else mod)
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

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None) -> dict:
        return {"logits": logits, "tau": tau or self.temperature}

    def _get_dist(self, context=None, temperature=None):
        mod_logits = self._modulate(context=context, temperature=temperature)
        tau = temperature or self.temperature
        return self._dist_type(**self._dist_params(mod_logits, tau=tau))

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, hard=True, **kwargs):
        return self._get_dist(context=context, temperature=temperature).sample(hard=hard)

    def log_prob(self, x, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None):
        return self._get_dist(context=context, temperature=temperature).probs

    def mode(self, context=None, temperature=None, **kwargs):
        dist = self._get_dist(context=context, temperature=temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits=None,
        posterior=None,
        from_probs=False,
        update_rate=None,
        temperature=None,
        context=None,
    ):
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

        # ------------------- MLP context modulation -------------------
        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states * n_features),
            )
        else:
            self.context_gate = None

    @property
    def dist_type(self):
        type_map = {
            "gaussian": torch.distributions.Independent,
            "laplace": torch.distributions.Independent,
            "normal": torch.distributions.Independent,
            "categorical": torch.distributions.Categorical,
            "bernoulli": torch.distributions.Bernoulli,
        }
        base_map = {
            "normal": torch.distributions.Normal,
            "gaussian": torch.distributions.Normal,
            "laplace": torch.distributions.Laplace,
        }
        key = self.emission_type.lower()
        if key in type_map:
            base = base_map.get(key, None)
            return (lambda *args, **kwargs: type_map[key](base(*args, **kwargs), 1)) if base else type_map[key]
        raise ValueError(f"Unknown emission_type: {self.emission_type}")

    @torch.no_grad()
    def _get_dist(
        self,
        X: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
        init_spread: float = 1.0,
        max_jitter: int = 5,
    ):
        """
        Estimate emission distribution parameters, optionally using data, posterior weights,
        theta modulation, context, and temperature scaling.

        Supports continuous (gaussian, laplace, studentt) and discrete
        (categorical, bernoulli, poisson) distributions.
        """
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        # --- Input preprocessing ---
        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std().item() < EPS:
                X = X + 1e-3 * torch.randn_like(X)

        # --- Posterior weighting ---
        if posterior is not None and X is not None:
            w = posterior.clamp_min(EPS)  # [N, K]
            w_sum = w.sum(dim=0) + EPS     # [K]
        else:
            w = None
            w_sum = None

        # --- Theta modulation ---
        theta_vec = theta.mean(dim=0) if theta is not None and theta.ndim > 1 else theta

        # ---------------- Continuous distributions ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self._emission_means.clone()

            # Weighted means from data
            if X is not None and w is not None:
                means = (w.T @ X) / w_sum.unsqueeze(1)

            # Apply theta modulation
            if theta_vec is not None:
                means += theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # Context + temperature modulation
            means = self._modulate(means, context, temperature)

            if etype == "gaussian":
                # Covariance estimation
                if X is not None and w is not None:
                    diff = X[:, None, :] - means[None, :, :]
                    weighted = diff * w[:, :, None]
                    covs = torch.einsum("nkf,nkd->kfd", weighted, diff) / w_sum[:, None, None]
                else:
                    covs = self._emission_covs.clone()

                # Regularize covs with jitter if necessary
                I = torch.eye(F, device=self.device)
                covs += self.min_covar * I[None, :]
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0:
                            break
                        covs[k] += jitter * I
                        jitter *= 2

                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covariance_matrix=covs)

            else:  # Laplace / StudentT
                if X is not None and w is not None:
                    scales = ((X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]).sum(dim=0) / w_sum[:, None]
                else:
                    scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()

                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))

                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=scales), 1)

        # ---------------- Discrete distributions ----------------
        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None and w is not None:
                if etype == "categorical":
                    logits = torch.zeros((K, F), device=self.device)
                    for k in range(K):
                        counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                        logits[k] = torch.log(counts.clamp_min(EPS) / counts.sum().clamp_min(EPS))
                else:  # Bernoulli / Poisson
                    rate = (w.T @ X.float()) / w_sum[:, None]
                    logits = torch.log(rate.clamp_min(EPS))
            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            # Apply theta modulation
            if theta_vec is not None:
                logits += theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # Context + temperature
            logits = self._modulate(logits, context, temperature)
            self._emission_params.copy_(logits)

            if etype == "categorical":
                return Categorical(logits=logits)
            elif etype == "bernoulli":
                return Independent(Bernoulli(logits=logits), 1)
            else:  # Poisson
                return Independent(Poisson(rate=torch.exp(logits)), 1)

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    def _modulate(
        self,
        tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float | torch.Tensor] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        """
        Modulate tensor with optional context MLP, adaptive scaling, and temperature.
        Works for batch or single-state tensors.

        Args:
            tensor: Input tensor [K, F] or [N, K, F].
            context: Optional context tensor [C] or [N, C].
            temperature: Optional temperature scalar or tensor.
            grad_safe: If True, prevents gradients from flowing through modulation.

        Returns:
            Modulated tensor with same shape as input.
        """
        mod = tensor

        # --- Context modulation ---
        if context is not None and self.context_gate is not None:
            # Expand batch if tensor has batch dimension
            if tensor.ndim == 3:  # [N, K, F]
                N, K, F = tensor.shape
                context_out = self.context_gate(context)  # [N, K*F] or [N, F*K]
                context_out = context_out.view(N, K, F)
            else:  # [K, F]
                context_out = self.context_gate(context)  # [K*F]
                context_out = context_out.view(tensor.shape)

            mod = mod + context_out

            # Adaptive scaling
            if self.adaptive_scale:
                norm = context.norm(dim=-1).mean() if context.ndim > 1 else context.norm()
                norm = max(norm.item(), EPS)
                mod = mod * (self.scale / norm)

        # --- Temperature modulation ---
        if temperature is not None:
            if isinstance(temperature, (float, int)):
                temp_tensor = torch.full_like(mod, float(temperature))
            else:
                temp_tensor = temperature.clamp_min(EPS)
            mod = mod / temp_tensor

        # --- Gradient-safe option ---
        if grad_safe:
            mod = mod.detach() + tensor - tensor.detach()

        return mod

    # ------------------- Forward / Distribution -------------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
    ):
        etype = self.emission_type

        # ---------------- Continuous distributions ----------------
        if etype == "gaussian":
            mu = self._modulate(self.mu, context, temperature, grad_safe=True)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)

            # Optional adaptive variance modulation
            if self.modulate_var:
                var += self._modulate(var, context, temperature, grad_safe=True).abs()

            # Add small jitter to prevent singular covariance
            jitter = torch.full_like(var, self.min_covar)
            var = var + jitter

            cov = torch.diag_embed(var)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(cov)

            dist = Independent(Normal(loc=mu, scale=var.sqrt()), 1)

        elif etype in {"laplace", "studentt"}:
            loc = self._modulate(self.loc, context, temperature, grad_safe=True)
            scale = torch.clamp(self.scale_param, min=self.min_covar)

            # Optional adaptive scaling
            if self.adaptive_scale and context is not None:
                norm = context.norm(dim=-1).mean() if context.ndim > 1 else context.norm()
                norm = max(norm.item(), EPS)
                scale = scale * (self.scale / norm)

            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale**2))

            dist_cls = Laplace if etype == "laplace" else StudentT
            dist_args = {"loc": loc, "scale": scale}
            if etype == "studentt":
                dist_args["df"] = self.dof
            dist = Independent(dist_cls(**dist_args), 1)

        # ---------------- Discrete distributions ----------------
        else:
            base_param = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._modulate(base_param, context, temperature, grad_safe=True)

            # Apply small jitter for exploration (optional for Poisson/Bernoulli)
            if etype in {"bernoulli", "poisson"}:
                out += 1e-6 * torch.randn_like(out)

            self._emission_params.copy_(out)

            if etype == "categorical":
                dist = Categorical(logits=out)
            elif etype == "bernoulli":
                dist = Independent(Bernoulli(logits=out), 1)
            else:  # Poisson
                rate = torch.clamp(torch.exp(out), max=1e6)
                dist = Independent(Poisson(rate=rate), 1)
                if hasattr(self, "log_rate"):
                    self.log_rate.copy_(out)

        if return_dist:
            return dist
        if etype in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        return self._emission_params

    # ------------------- Sample / Log Prob -------------------
    def sample(
        self,
        n_samples: int = 1,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        samples = dist.rsample((n_samples,)) if isinstance(dist, Independent) else dist.sample((n_samples,))
        return samples.to(dtype=DTYPE, device=self.device)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        dist = self.forward(context=context, temperature=temperature, return_dist=True)

        if isinstance(dist, Independent):
            if x.ndim == 2 and x.shape[1] == self.n_features:
                x = x.unsqueeze(1)
            elif x.ndim == 1 and self.n_features == 1:
                x = x.unsqueeze(0).unsqueeze(1)

        return dist.log_prob(x)

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
        Update emission parameters using optional data, posterior weights,
        theta modulation, context, and temperature.

        Args:
            X: Observed data [N, F] or [N, ...].
            posterior: Optional posterior probabilities [N, K].
            theta: Optional modulation vector or tensor.
            context: Optional context tensor [N, C] or [C].
            theta_scale: Scale for theta modulation.
            update_rate: Exponential moving average rate.
            temperature: Optional temperature for scaling emission parameters.
            init_spread: Initial spread for stochastic updates.
            max_jitter: Maximum Cholesky jitter attempts.
            rank: Optional rank info (not used here).

        Returns:
            Updated torch.distribution instance for emissions.
        """
        # Compute new distribution parameters
        new_dist = self._get_dist(
            X=X,
            theta=theta,
            context=context,
            posterior=posterior,
            theta_scale=theta_scale,
            init_spread=init_spread,
            max_jitter=max_jitter,
            temperature=temperature
        )
        etype = self.emission_type

        # ---------- Continuous distributions ----------
        if etype == "gaussian":
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.loc)
            self._emission_covs.mul_(1 - update_rate).add_(update_rate * new_dist.covariance_matrix)
            self.mu.copy_(self._emission_means)

            cov_diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)
            if cov_diag.shape != self.log_var.shape:
                cov_diag = cov_diag[:, :self.n_features]
            self.log_var.copy_(torch.log(torch.clamp(cov_diag, min=EPS)))

        elif etype in {"laplace", "studentt"}:
            loc = new_dist.base_dist.loc
            scale = getattr(new_dist.base_dist, "scale", None) or getattr(new_dist.base_dist, "scale_param")
            self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)

            diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale**2
            self._emission_covs.copy_(torch.diag_embed(diag))

            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(diag, min=EPS)))

        # ---------- Discrete distributions ----------
        else:
            if etype == "categorical":
                new_logits = new_dist.logits
            elif etype == "bernoulli":
                new_logits = new_dist.base_dist.logits
            else:  # Poisson
                new_logits = torch.log(torch.clamp(new_dist.base_dist.rate, min=EPS))

            # EMA update
            self._emission_params.mul_(1 - update_rate).add_(update_rate * new_logits)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        mode: str = "data",
        iters: int = 15,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
    ):
        """
        Initialize emission parameters from data, k-means, or default.
        Supports context, theta modulation, and temperature scaling.

        Args:
            X: Optional data tensor [N, F] or [N, T, F].
            context: Optional context tensor [N, C] or [C].
            mode: "data", "kmeans", or "default".
            iters: K-means iterations if mode="kmeans".
            theta: Optional modulation vector or tensor.
            theta_scale: Scale for theta modulation.
            temperature: Optional temperature for scaling emission parameters.

        Returns:
            Initialized torch.distribution instance for the emission.
        """
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

        # --- Initialize means and covariances ---
        if mode == "data":
            mean = Xf.mean(dim=0)
            cov = (Xf - mean).T @ (Xf - mean) / max(N - 1, 1) + self.min_covar * torch.eye(F, device=device)
            means = mean.expand(K, F).clone()
            covs = cov.unsqueeze(0).expand(K, F, F).clone()

        elif mode == "kmeans":
            centers, labels = run_kmeans(Xf, K, iters)
            means = centers
            covs = torch.stack([
                torch.cov(Xf[labels == k].T) + self.min_covar * torch.eye(F, device=device)
                for k in range(K)
            ])
        else:
            raise ValueError(f"Unsupported initialization mode: {mode}")

        # --- Apply theta modulation ---
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            means += theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

        # --- Context & temperature modulation ---
        means = self._modulate(means, context=context, temperature=temperature)

        # --- Assign to buffers and parameters ---
        self._emission_means.copy_(means)
        self._emission_covs.copy_(covs)

        if self.emission_type == "gaussian":
            self.mu.copy_(means)
            cov_diag = torch.diagonal(covs, dim1=-2, dim2=-1)
            if cov_diag.shape != self.log_var.shape:
                cov_diag = cov_diag[:, :self.n_features]
            self.log_var.copy_(torch.log(torch.clamp(cov_diag, min=EPS)))
            return MultivariateNormal(means, covariance_matrix=covs)

        elif self.emission_type in {"laplace", "studentt"}:
            scale = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
            self.loc.copy_(means)
            self.scale_param.copy_(scale)
            dist_cls = Laplace if self.emission_type == "laplace" else StudentT
            dist = dist_cls(loc=means, scale=scale) if self.emission_type == "laplace" else StudentT(df=self.dof, loc=means, scale=scale)
            return Independent(dist, 1)

        else:  # Categorical / Bernoulli / Poisson
            if mode == "data":
                logits = torch.log_softmax(Xf.mean(dim=0).expand(K, F), dim=-1)
            else:  # kmeans
                logits = means

            # Apply context & temperature modulation for discrete
            logits = self._modulate(logits, context=context, temperature=temperature)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(logits)
            self._emission_params.copy_(logits)

            if self.emission_type == "categorical":
                return Categorical(logits=logits)
            elif self.emission_type == "bernoulli":
                return Independent(Bernoulli(logits=logits), 1)
            else:  # Poisson
                return Independent(Poisson(rate=torch.exp(logits)), 1)

