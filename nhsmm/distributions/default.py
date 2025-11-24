# nhsmm/distributions/default.py
# per timestep vary

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
        return F.softmax(self._logits / self.tau, dim=-1)

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
    def log_prob(self, value: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        tau_val = tau or self.tau
        value = value.long()
        flat_logits = (self._logits / tau_val).reshape(-1, self._logits.size(-1))
        flat_value = value.reshape(-1)
        logp = F.log_softmax(flat_logits, dim=-1)
        return logp.gather(-1, flat_value.unsqueeze(-1)).squeeze(-1).reshape(value.shape)

    # ---------------- Internal Gumbel ----------------
    def _gumbel_logits(self, sample_shape=torch.Size(), tau: Optional[float] = None, generator=None):
        tau_val = tau or self.tau
        shape = sample_shape + self._logits.shape
        g = -torch.log(-torch.log(
            torch.rand(shape, device=self._logits.device, generator=generator).clamp_min(EPS)
        ))
        return (self._logits.expand(shape).contiguous() + g) / tau_val

    # ---------------- Gumbel-Softmax Sampling ----------------
    def _sample_gumbel_softmax(self, sample_shape=torch.Size(), hard: bool = True, tau: Optional[float] = None, generator=None):
        y_soft = F.softmax(self._gumbel_logits(sample_shape, tau=tau, generator=generator), dim=-1)
        if not hard:
            return y_soft
        # Hard one-hot with gradient trick
        y_hard = F.one_hot(y_soft.argmax(-1), num_classes=y_soft.size(-1)).to(y_soft.dtype)
        return (y_hard - y_soft).detach() + y_soft

    def sample(self, sample_shape=torch.Size(), hard: bool = True, tau: Optional[float] = None, generator=None):
        return self._sample_gumbel_softmax(sample_shape, hard=hard, tau=tau, generator=generator)

    def rsample(self, sample_shape=torch.Size(), tau: Optional[float] = None, generator=None):
        return self.sample(sample_shape, hard=False, tau=tau, generator=generator)


class DistributionBase(nn.Module):
    """Base class for context-modulated HSMM parameters with batch/sequence support."""
    _dist_type: type = None

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

        # Core dims
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

        # Projection layer
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
        self.log_temperature = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        # Cache
        self.register_buffer("_logits_buffer", torch.zeros(target_dim, dtype=DTYPE))
        self.register_buffer("_mod_logits_buffer", torch.zeros(target_dim, dtype=DTYPE))
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
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Fully batch-shape agnostic context application with optional gates/residuals/low-rank factors.
        Works for base shapes [K], [B,K], [B,T,K], [B,T,...,K] or [K,Dmax], etc.
        """

        orig_shape = base.shape
        K = base.shape[-1]  # last dimension is always the "feature/state" dim

        # ---------------- Flatten context ----------------
        if context is None:
            ctx_flat = None
        else:
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B * T, H)

            # Lazy projection if needed
            if self.context_dim is None:
                self.context_dim = H
            if H != self.context_dim:
                if not self.allow_projection:
                    raise ValueError(f"Context dim mismatch: expected {self.context_dim}, got {H}")
                if self._proj is None or self._proj.in_features != H:
                    self._proj = nn.Linear(H, self.context_dim, dtype=base.dtype)
                    nn.init.xavier_uniform_(self._proj.weight)
                    nn.init.zeros_(self._proj.bias)
                ctx_flat = self._proj(ctx_flat)

        # ---------------- Flatten base ----------------
        base_flat = base.reshape(-1, K)
        delta = torch.zeros_like(base_flat, dtype=base.dtype, device=base.device)

        # ---------------- Context gates/residuals/low-rank ----------------
        if ctx_flat is not None:
            if hasattr(self, "context_gate") and self.context_gate is not None:
                delta += self.context_gate(ctx_flat)
            if hasattr(self, "residual_gate") and self.residual_gate is not None:
                delta += self.residual_gate(ctx_flat)
            if hasattr(self, "_U") and hasattr(self, "_V") and self._U is not None and self._V is not None:
                U = self._U(ctx_flat).view(ctx_flat.size(0), -1, getattr(self, "rank", 1))
                V = self._V(ctx_flat).view(ctx_flat.size(0), 1, getattr(self, "rank", 1))
                delta += (U * V).sum(-1)

        # ---------------- Optional normalization ----------------
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if getattr(self, "layer_norm", False):
            delta = F.layer_norm(delta, delta.shape[-1:])
        if getattr(self, "batch_norm", False):
            if not hasattr(self, "_batchnorm") or self._batchnorm is None or self._batchnorm.num_features != delta.shape[-1]:
                self._batchnorm = nn.BatchNorm1d(delta.shape[-1], affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(delta)

        # ---------------- Adapters ----------------
        if not skip_adapters:
            if getattr(self, "temporal_adapter", None):
                x = delta.unsqueeze(-1)
                x = self.temporal_adapter(x)
                delta = x.squeeze(-1)
            if getattr(self, "spatial_adapter", None):
                delta = self.spatial_adapter(delta)

        # ---------------- Activation, scaling, clamping ----------------
        delta = getattr(self, "final_activation_fn", nn.Identity())(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", 1e6), getattr(self, "max_delta", 1e6))
        if grad_scale is not None:
            delta *= grad_scale

        # ---------------- Restore original shape ----------------
        delta = delta.reshape(orig_shape)
        base_expanded = base
        if base.ndim < delta.ndim:
            base_expanded = base.view((1,) * (delta.ndim - base.ndim) + base.shape)

        return self._apply_constraints(base_expanded + delta)

    # ---------------- Unified modulate with caching ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        tau = temperature if temperature is not None else getattr(self, "temperature", 1.0)
        tau_tensor = torch.as_tensor(float(tau), dtype=DTYPE, device=getattr(self, "logits", torch.tensor(0.0)).device)
        key = f"{self._context_hash(context)}-T{float(tau_tensor):.6e}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(getattr(self, "logits", torch.zeros(1, dtype=DTYPE)), context)
        mod = mod / tau_tensor.clamp_min(EPS)
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
    def initialize(self, mode="uniform", **_):
        return self

    # --- Unified EM / learnable update ---
    @torch.no_grad()
    def update(self, new_logits: Optional[torch.Tensor] = None,
               posterior: Optional[torch.Tensor] = None,
               from_probs: bool = False,
               update_rate: Optional[float] = None,
               temperature: Optional[float] = None,
               context: Optional[torch.Tensor] = None):
        # --- Direct replacement ---
        if new_logits is not None:
            while new_logits.ndim > 2: new_logits = new_logits.squeeze(0)
            if from_probs: new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None: new_logits = new_logits / max(temperature, 1e-6)

            if hasattr(self, "_apply_constraints"):
                new_logits = self._apply_constraints(new_logits)
            new_logits = self._validate_logits(new_logits)

            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        # --- EM-style update ---
        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            mod_logits = self._modulate(context=context, temperature=temperature)
            if posterior.ndim < mod_logits.ndim:
                posterior = posterior.view(*posterior.shape, *[1]*(mod_logits.ndim - posterior.ndim)).expand_as(mod_logits)
            elif posterior.shape != mod_logits.shape:
                raise ValueError(f"Posterior shape {posterior.shape} does not match modulated logits {mod_logits.shape}")

            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)

            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None: p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()


class Initial(DistributionBase):
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

        # Context gates / low-rank factors are handled automatically by DistributionBase
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

        if (getattr(self, "_U", None) or getattr(self, "_V", None)) and self.rank is None:
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

    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, hard: Optional[bool] = None, timestep: Optional[int] = None, **kwargs) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=temperature)

        # Select timestep if requested
        if timestep is not None:
            if mod_logits.ndim == 3:
                mod_logits = mod_logits[:, timestep, :]
            else:
                raise ValueError(f"Cannot select timestep {timestep}, logits shape={mod_logits.shape}")

        dist = self._dist_type(**self._dist_params(mod_logits, **kwargs))

        if hard is not None:
            dist.hard = hard
        return dist

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, hard: bool = True, timestep: Optional[int] = None, **kwargs):
        dist = self._get_dist(context=context, temperature=temperature, hard=hard, timestep=timestep)
        return dist.sample(**kwargs)

    def log_prob(self, x, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.log_prob(x)

    def expected_probs(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.probs

    def mode(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        if hasattr(dist, "mode"):
            return dist.mode
        return dist.probs.argmax(-1)


class Duration(DistributionBase):
    """
    Context-aware categorical duration distribution per state for HSMMs.
    Internal shape: [K, Dmax].
    Supports batching: returns [B, T, K, Dmax] for multiple sequences.
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

        self.rank = rank
        self.n_states = n_states
        self.max_duration = max_duration
        self.init_mode = init_mode
        self.temperature = temperature

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / residual / low-rank handled automatically
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

        if (getattr(self, "_U", None) or getattr(self, "_V", None)) and self.rank is None:
            raise ValueError("Rank must be specified if using low-rank factors.")

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE).unsqueeze(0).repeat(n_states, 1)
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

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        tau_val = tau if tau is not None else self.temperature
        # Use max to support float tau
        mod_logits = logits / max(tau_val, EPS)
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
        return self._get_dist(context=context, temperature=temperature).rsample(**kwargs)

    def log_prob(self, x, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context=context, temperature=temperature).probs

    def mode(self, context=None, temperature=None, **kwargs):
        dist = self._get_dist(context=context, temperature=temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        return dist.probs.argmax(-1)


class Transition(DistributionBase):
    """
    Context-aware categorical transition distribution for HSMMs.

    Supports:
    - Context gates, residuals, low-rank factors
    - Structural constraints (ergodic, semi, left-to-right)
    - Batch/sequence aware: [K,K] or [B,T,K,K]
    - Temperature scaling + caching
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
        cache_limit: int = 32,
        debug: bool = False,
    ):
        super().__init__(
            target_dim=n_states * n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            cache_limit=cache_limit,
            cache_enabled=True,
            debug=debug,
        )

        self.n_states = n_states
        self.rank = rank
        self.temperature = temperature
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)

        # Base logits [K,K]
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context modules
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

        if (getattr(self, "_U", None) or getattr(self, "_V", None)) and self.rank is None:
            raise ValueError("Rank must be specified if using low-rank factors.")

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
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, grad_scale: Optional[float] = None) -> torch.Tensor:
        n = self.n_states
        # Flatten base
        base_flat = base.view(-1) if base.ndim == 2 else base
        mod_flat = super()._apply_context(base_flat, context=context, grad_scale=grad_scale)
        mod = mod_flat.view(n, n)
        return self._apply_constraints(mod)

    # ---------------- Modulate ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        tau = temperature if temperature is not None else self.temperature
        key = f"{self._context_hash(context)}-T{float(tau):.6e}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        mod = self._apply_context(self.logits, context)
        mod = mod / max(tau, EPS)
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
    def sample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context, temperature).sample(**kwargs)

    def rsample(self, context=None, temperature=None, **kwargs):
        return self._get_dist(context, temperature).rsample(**kwargs)

    def log_prob(self, x, context=None, temperature=None, **kwargs):
        return self._get_dist(context, temperature).log_prob(x)

    def expected_probs(self, context=None, temperature=None):
        return self._get_dist(context, temperature).probs

    def mode(self, context=None, temperature=None, **kwargs):
        dist = self._get_dist(context, temperature)
        if hasattr(dist, "mode"):
            return dist.mode
        return dist.probs.argmax(-1)

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
        scale: float = 1.0,
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
        self.scale = scale
        self.seed = seed
        self.dof = dof

        # Buffers
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE))
        self.register_buffer("_emission_covs", torch.eye(n_features, dtype=DTYPE).unsqueeze(0).repeat(n_states, 1, 1))
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE))

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, dtype=DTYPE))
        elif self.emission_type in {"categorical", "bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        elif self.emission_type in {"laplace", "studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, dtype=DTYPE))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

        # Context projection
        if context_dim is not None and hidden_dim is not None:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states * n_features),
            )
            self.context_proj_layer = nn.Linear(context_dim, n_features) if context_dim != n_features else None
        else:
            self.context_gate = None
            self.context_proj_layer = None

    # ------------------- Context / Modulation -------------------
    def _modulate(self, params: torch.Tensor, context: Optional[torch.Tensor] = None,
                  temperature: Optional[float] = None, grad_safe: bool = False) -> torch.Tensor:
        tau = max(temperature or self.temperature, EPS)

        if self.emission_type in {"gaussian", "laplace", "studentt"} and context is not None:
            K, F = params.shape
            if context.ndim == 1:
                context = context.view(1, 1, -1)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape

            if self.context_gate:
                context_mod = self.context_gate(context.reshape(B*T, H)).reshape(B, T, K, F)
            elif self.context_proj_layer:
                context_mod = self.context_proj_layer(context).unsqueeze(2).expand(B, T, K, F)
            else:
                if H != F:
                    raise ValueError(f"Context dim {H} != feature dim {F}")
                context_mod = context.unsqueeze(2).expand(B, T, K, F)

            out = params.unsqueeze(0).unsqueeze(0).expand(B, T, K, F) + context_mod
            out = out / tau
        else:
            out = params / tau if self.emission_type in {"categorical", "bernoulli", "poisson"} else params

        return out if grad_safe else out.detach()

    # ------------------- Distribution Construction -------------------
    def _dist_params(self, tensor: torch.Tensor, temperature: Optional[float] = None,
                     context: Optional[torch.Tensor] = None, grad_safe: bool = False, **kwargs) -> dict:
        mod_tensor = self._modulate(tensor, context=context, temperature=temperature, grad_safe=grad_safe)
        etype = self.emission_type

        if etype in {"categorical", "bernoulli", "poisson"}:
            return {"logits": mod_tensor, **kwargs}

        if etype == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)
        else:
            scale = self.scale_param.clamp_min(self.min_covar)
            cov = torch.diag_embed(scale ** 2)

        return {"means": mod_tensor, "cov": cov, **kwargs}

    def _get_continuous_dist(self, X=None, theta=None, context=None, posterior=None,
                             theta_scale=0.1, temperature=None, max_jitter=5):
        K, F = self.n_states, self.n_features
        means = self._emission_means.clone()

        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)
            w_sum = w.sum(dim=0) + EPS
            means = (w.T @ X) / w_sum.unsqueeze(1)

        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            means = means + theta_scale * theta_vec.unsqueeze(0)

        means = self._modulate(means, context=context, temperature=temperature)

        if self.emission_type == "gaussian":
            covs = self._emission_covs.clone()
            I = torch.eye(F)
            for k in range(K):
                jitter = self.min_covar
                for _ in range(max_jitter):
                    _, info = torch.linalg.cholesky_ex(covs[k])
                    if info == 0: break
                    covs[k] += jitter * I
                    jitter *= 2
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)
            self.dist_type = lambda *a, **kw: Independent(MultivariateNormal(*a, **kw), 1)
            return MultivariateNormal(means, covariance_matrix=covs)

        if X is not None and posterior is not None:
            scales = ((X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]).sum(dim=0) / w_sum[:, None]
        else:
            scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()
        scales = scales.clamp_min(self.min_covar)
        self._emission_means.copy_(means)
        self._emission_covs.copy_(torch.diag_embed(scales ** 2))
        dist_cls = Laplace if self.emission_type == "laplace" else lambda **kw: StudentT(df=self.dof, **kw)
        self.dist_type = lambda *a, **kw: Independent(dist_cls(*a, **kw), 1)
        return Independent(dist_cls(loc=means, scale=scales), 1)

    def _get_discrete_dist(self, X=None, theta=None, context=None, posterior=None,
                           theta_scale=0.1, temperature=None):
        K, F = self.n_states, self.n_features
        etype = self.emission_type

        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)
            w_sum = w.sum(dim=0) + EPS
            if etype == "categorical":
                logits = torch.zeros(K, F)
                for k in range(K):
                    counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                    logits[k] = torch.log(counts.clamp_min(EPS) / counts.sum().clamp_min(EPS))
            else:
                rate = (w.T @ X.float()) / w_sum.unsqueeze(1)
                logits = torch.log(rate.clamp_min(EPS))
        else:
            logits = torch.full((K, F), -math.log(F), dtype=DTYPE)

        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            logits = logits + theta_scale * theta_vec.unsqueeze(0)

        logits = self._modulate(logits, context=context, temperature=temperature)
        self._emission_params.copy_(logits)

        if etype == "categorical":
            self.dist_type = Categorical
            return Categorical(logits=logits)
        elif etype == "bernoulli":
            self.dist_type = lambda *a, **kw: Independent(Bernoulli(*a, **kw), 1)
            return Independent(Bernoulli(logits=logits), 1)
        else:
            self.dist_type = lambda *a, **kw: Independent(Poisson(*a, **kw), 1)
            return Independent(Poisson(rate=torch.exp(logits)), 1)

    def _get_dist(self, X=None, emission_type=None, theta=None, context=None,
                  posterior=None, theta_scale=0.1, temperature=None, max_jitter=5):
        etype = emission_type or self.emission_type
        if etype in {"gaussian", "laplace", "studentt"}:
            return self._get_continuous_dist(X=X, theta=theta, context=context,
                                             posterior=posterior, theta_scale=theta_scale,
                                             temperature=temperature, max_jitter=max_jitter)
        elif etype in {"categorical", "bernoulli", "poisson"}:
            return self._get_discrete_dist(X=X, theta=theta, context=context,
                                           posterior=posterior, theta_scale=theta_scale,
                                           temperature=temperature)
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ------------------- Forward / Distribution -------------------
    def forward(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False):
        """
        Produce emission distribution or parameters.
        Context and temperature modulation applied once.
        """

        # 1) Produce distribution (also updates _emission_* via _get_dist)
        dist = self._get_dist(
            X=None,
            context=context,
            theta=None,
            theta_scale=0.1,
            temperature=temperature,
            max_jitter=5,
        )

        # ------------------------------------------------------------------
        # 2) SAFE BUFFER SYNCHRONIZATION (NO IN-PLACE ON LEAF GRAD TENSORS)
        # ------------------------------------------------------------------
        with torch.no_grad():

            if self.emission_type in {"gaussian", "laplace", "studentt"}:

                if self.emission_type == "gaussian":
                    # mean
                    if hasattr(self, "mu"):
                        self.mu.copy_(self._emission_means.detach())

                    # covariance -> log_var
                    cov_diag = torch.diagonal(
                        self._emission_covs, dim1=-2, dim2=-1
                    ).clamp_min(self.min_covar)
                    logv = torch.log(cov_diag)

                    if hasattr(self, "log_var"):
                        self.log_var.copy_(logv.detach())

                else:
                    # laplace / studentt
                    scale = self.scale_param.clamp_min(self.min_covar)

                    if hasattr(self, "loc"):
                        self.loc.copy_(self._emission_means.detach())

                    if hasattr(self, "scale_param"):
                        self.scale_param.copy_(scale.detach())

                    # maintain consistent covariance buffer
                    if hasattr(self, "_emission_covs"):
                        cov = torch.diag_embed((scale.detach()) ** 2)
                        self._emission_covs.copy_(cov)

            else:
                # Discrete: categorical / bernoulli / poisson
                if hasattr(self, "_emission_params"):
                    self._emission_params.copy_(self._emission_params.detach())

                # Update associated buffer (logits or log_rate)
                if hasattr(self, "logits"):
                    self.logits.copy_(self._emission_params.detach())
                elif hasattr(self, "log_rate"):
                    self.log_rate.copy_(self._emission_params.detach())

        # ------------------------------------------------------------------
        # 3) RETURN VALUES
        # ------------------------------------------------------------------
        if return_dist:
            return dist

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
        max_jitter: int = 5,
    ):
        """
        Update emission parameters via EMA based on new data/posterior.
        Ensures numerical stability and clamps variances/scales.
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
            new_means = self._emission_means.clone()
            new_covs = self._emission_covs.clone()
            # EMA update
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_means)
            self._emission_covs.mul_(1 - update_rate).add_(update_rate * new_covs)
            # Sync learnable parameters with clamping
            self.mu.copy_(self._emission_means)
            cov_diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)
            self.log_var.copy_(torch.log(torch.clamp(cov_diag, min=EPS)))

        elif etype in {"laplace", "studentt"}:
            new_loc = self.loc.clone()
            new_scale = self.scale_param.clone()
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_loc)
            diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * new_scale**2
            diag = diag.clamp_min(EPS)  # ensure numerical stability
            self._emission_covs.copy_(torch.diag_embed(diag))
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(diag))

        else:  # categorical / bernoulli / poisson
            new_params = getattr(self, "logits", getattr(self, "log_rate", self._emission_params)).clone()
            self._emission_params.mul_(1 - update_rate).add_(update_rate * new_params)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, temperature=temperature, return_dist=True)

        if getattr(dist, "has_rsample", False):
            samples = dist.rsample((n_samples,))
        else:
            samples = dist.sample((n_samples,))

        return samples.to(dtype=DTYPE)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:

        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        N, K, D = x.shape[0], self.n_states, self.n_features
        etype = self.emission_type

        # ---- unified expand helper ----
        def expand_x(x):
            return x.unsqueeze(1).expand(-1, K, -1)

        # -------------------------------
        if etype in {"gaussian", "laplace", "studentt"}:
            x_exp = expand_x(x) if x.ndim == 2 else x
            return dist.log_prob(x_exp)

        elif etype == "categorical":
            # prefer logits from the actual dist if provided
            logits = getattr(dist, "logits", getattr(self, "logits", self._emission_params))
            logits = logits.unsqueeze(0).expand(N, K, -1, D) if logits.ndim == 3 else logits

            x_exp = expand_x(x.long()) if x.ndim == 2 else x.long()
            log_probs = F.log_softmax(logits, dim=-1)

            return torch.gather(log_probs, -1, x_exp.unsqueeze(-1)).squeeze(-1)

        elif etype == "bernoulli":
            logits = getattr(dist, "logits", getattr(self, "logits", self._emission_params))
            x_exp = expand_x(x)
            logits_exp = logits.unsqueeze(0).expand(N, K, D)
            return -F.binary_cross_entropy_with_logits(
                logits_exp, x_exp, reduction="none"
            ).sum(-1)

        elif etype == "poisson":
            log_rate = getattr(dist, "log_rate", getattr(self, "log_rate", self._emission_params))
            rate = torch.exp(log_rate)

            x_exp = expand_x(x)
            rate_exp = rate.unsqueeze(0).expand(N, K, D)

            # x * log(rate) is undefined for x=0 unless masked
            log_probs = (
                -rate_exp
                + torch.where(
                    x_exp > 0,
                    x_exp * torch.log(rate_exp.clamp_min(EPS)),
                    torch.zeros_like(x_exp),
                )
                - torch.lgamma(x_exp + 1)
            )
            return log_probs.sum(-1)

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        log_probs = super().log_matrix(context=context, temperature=temperature)
        if temperature and temperature != 1.0:
            log_probs = log_probs / temperature
        return log_probs

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return super().expected_probs(context=context, temperature=temperature)

    def parameters_tensor(self) -> torch.Tensor:
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        return self._emission_params

    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        theta_scale: float = 0.1,
        mode: str = "data",
        iters: int = 15,
    ):
        K, F = self.n_states, self.n_features

        # Flatten input
        if X is not None:
            Xf = X.reshape(-1, F).to(dtype=DTYPE)
            N = Xf.shape[0]
        else:
            Xf = None
            N = 0

        # KMeans helper
        def run_kmeans(Xflat, K, iters):
            idx = torch.randperm(Xflat.shape[0])[:K]
            centers = Xflat[idx].clone()
            for _ in range(iters):
                dist = torch.cdist(Xflat, centers)
                labels = dist.argmin(dim=1)
                for k in range(K):
                    pts = Xflat[labels == k]
                    if pts.numel() > 0:
                        centers[k] = pts.mean(0)
            return centers, labels

        # Base initialization
        if Xf is not None:
            if mode == "data":
                mean = Xf.mean(0)
                cov = (Xf - mean).T @ (Xf - mean) / max(N - 1, 1)
                cov = cov + self.min_covar * torch.eye(F, dtype=DTYPE)
                means = mean.expand(K, F).clone()
                covs = cov.unsqueeze(0).expand(K, F, F).clone()
            elif mode == "kmeans":
                centers, labels = run_kmeans(Xf, K, iters)
                means = centers
                covs = torch.stack([
                    torch.cov(Xf[labels == k].T) + self.min_covar * torch.eye(F, dtype=DTYPE)
                    if (labels == k).sum() > 1 else torch.eye(F, dtype=DTYPE) * self.min_covar
                    for k in range(K)
                ])
            else:
                means = torch.zeros(K, F, dtype=DTYPE)
                covs = torch.eye(F, dtype=DTYPE).unsqueeze(0).repeat(K, 1, 1) * self.min_covar
        else:
            means = torch.zeros(K, F, dtype=DTYPE)
            covs = torch.eye(F, dtype=DTYPE).unsqueeze(0).repeat(K, 1, 1) * self.min_covar

        # Theta modulation
        if theta is not None:
            theta_vec = theta.mean(0) if theta.ndim > 1 else theta
            means = means + theta_scale * theta_vec.to(dtype=DTYPE).unsqueeze(0)

        # Context modulation
        means_mod = self._modulate(means, context=context, temperature=temperature)
        if means_mod.ndim == 4 and means_mod.shape[0] == 1 and means_mod.shape[1] == 1:
            means_mod = means_mod.squeeze(0).squeeze(0)

        # Store buffers safely
        self._emission_means.copy_(means_mod)
        self._emission_covs.copy_(covs)

        # Continuous emissions
        if self.emission_type == "gaussian":
            self.mu.copy_(means_mod)
            cov_diag = torch.diagonal(covs, dim1=-2, dim2=-1)
            self.log_var.copy_(torch.log(cov_diag.clamp_min(EPS)))
            return Independent(MultivariateNormal(means_mod, covariance_matrix=covs), 1)

        if self.emission_type in {"laplace", "studentt"}:
            scale = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
            self.loc.copy_(means_mod)
            self.scale_param.copy_(scale)
            dist_cls = Laplace if self.emission_type == "laplace" else StudentT
            return Independent(dist_cls(loc=means_mod, scale=scale), 1)

        # Discrete emissions
        if Xf is not None and mode == "data":
            logits = torch.log_softmax(Xf.mean(0).expand(K, F), -1)
        else:
            logits = means_mod

        logits = self._modulate(logits, context=context, temperature=temperature)

        param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
        if param_attr is not None:
            param_attr.copy_(logits)
        self._emission_params.copy_(logits)

        if self.emission_type == "categorical":
            return Categorical(logits=logits)
        if self.emission_type == "bernoulli":
            return Independent(Bernoulli(logits=logits), 1)
        return Independent(Poisson(rate=torch.exp(logits)), 1)

