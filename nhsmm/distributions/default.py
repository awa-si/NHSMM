# nhsmm/distributions/default.py

from __future__ import annotations
from typing import Optional, Union, Literal, Tuple, Dict, Any
from collections import OrderedDict
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Bernoulli, Laplace, MultivariateNormal,
    Normal, Independent, Poisson, StudentT
)

from nhsmm.tools import constraints
from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger


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
            raise ValueError("Specify exactly one of logits or probs.")
        if probs is not None:
            logits = torch.log(probs.clamp_min(EPS))
        self._logits = logits
        self.tau = tau
        batch_shape = logits.shape[:-1]
        super().__init__(
            batch_shape=batch_shape, event_shape=torch.Size([]), validate_args=validate_args
        )

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
        tau_val = tau if tau is not None else self.tau
        value = value.long()
        logits_scaled = self._logits / tau_val
        log_probs = F.log_softmax(logits_scaled, dim=-1)
        return log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    # ---------------- Internal Gumbel ----------------
    def _gumbel_logits(
        self, sample_shape: torch.Size = torch.Size(), tau: Optional[float] = None, generator=None
    ) -> torch.Tensor:
        tau_val = tau if tau is not None else self.tau
        shape = sample_shape + self._logits.shape
        # Uniform in [EPS, 1-EPS] for stability
        u = torch.rand(shape, device=self._logits.device, generator=generator).clamp(EPS, 1.0 - EPS)
        g = -torch.log(-torch.log(u))
        return (self._logits.expand(shape) + g) / tau_val

    # ---------------- Gumbel-Softmax Sampling ----------------
    def _sample_gumbel_softmax(
        self, sample_shape: torch.Size = torch.Size(), hard: bool = True, tau: Optional[float] = None, generator=None
    ) -> torch.Tensor:
        y_soft = F.softmax(self._gumbel_logits(sample_shape, tau=tau, generator=generator), dim=-1)
        if not hard:
            return y_soft
        y_hard = F.one_hot(y_soft.argmax(-1), num_classes=y_soft.size(-1)).to(y_soft.dtype)
        return (y_hard - y_soft).detach() + y_soft

    def sample(
        self, sample_shape: torch.Size = torch.Size(), hard: bool = True, tau: Optional[float] = None, generator=None
    ) -> torch.Tensor:
        return self._sample_gumbel_softmax(sample_shape, hard=hard, tau=tau, generator=generator)

    def rsample(
        self, sample_shape: torch.Size = torch.Size(), tau: Optional[float] = None, generator=None
    ) -> torch.Tensor:
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

        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_delta = max_delta
        self.debug = debug
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.cache_grad_safe = cache_grad_safe
        self.temperature = 1.0

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

        # Base logits
        self.logits = nn.Parameter(torch.zeros(target_dim, dtype=DTYPE))

        # Canonical shape for tensor alignment
        self._shape = self._infer_shape()

    # ---------------- Shape Handling ----------------
    def _infer_shape(self) -> Tuple[int, ...]:
        if hasattr(self, "_shape") and self._shape is not None:
            return tuple(self._shape)
        if hasattr(self.dist_type, "event_shape"):
            es = getattr(self.dist_type, "event_shape")
            if isinstance(es, tuple) and len(es) > 0:
                return es
        return (self.target_dim,)

    @torch.no_grad()
    def _tensor_shape(self, x: Optional[torch.Tensor], name="tensor", atol=0) -> Optional[torch.Tensor]:
        if x is None:
            return None
        x = x.clone()
        target_shape = tuple(self._shape)
        n_trailing = len(target_shape)
        leading_shape = x.shape[:-n_trailing] if x.ndim > n_trailing else ()

        # Collapse extra dims
        while x.ndim > len(leading_shape) + n_trailing:
            x = x.mean(dim=-1)

        # Expand missing dims
        while x.ndim < len(leading_shape) + n_trailing:
            x = x.unsqueeze(-1)

        # Align trailing dims
        for i, target_size in enumerate(target_shape):
            dim = -n_trailing + i
            current_size = x.shape[dim]
            if current_size == target_size:
                continue
            elif current_size == 1:
                expand_sizes = list(x.shape)
                expand_sizes[dim] = target_size
                x = x.expand(*expand_sizes)
            elif atol > 0 and abs(current_size - target_size) <= atol:
                expand_sizes = list(x.shape)
                expand_sizes[dim] = target_size
                if current_size < target_size:
                    x = x.expand(*expand_sizes)
                else:
                    x = x.narrow(dim, 0, target_size)
            else:
                raise ValueError(f"[{name}] Cannot reshape {x.shape} to target {target_shape}")
        return x

    # ---------------- Activation & Weight Utilities ----------------
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

    # ---------------- Context & Modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        base_ndim = base.ndim
        mod = base.clone()

        # Align context
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(1)
            B, T, H = context.shape

            expected_H = getattr(self, "context_dim", H)
            if H != expected_H:
                if not getattr(self, "allow_projection", True):
                    raise ValueError(f"Context dim mismatch: expected {expected_H}, got {H}")
                if getattr(self, "_proj", None) is None or self._proj.in_features != H:
                    self._proj = nn.Linear(H, expected_H, dtype=base.dtype)
                    nn.init.xavier_uniform_(self._proj.weight)
                    nn.init.zeros_(self._proj.bias)
                context = self._proj(context.reshape(B*T, H)).view(B, T, expected_H)

        delta = torch.zeros_like(mod)

        # Normalization
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if self.layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if self.batch_norm:
            nf = delta.shape[-1]
            if getattr(self, "_batchnorm", None) is None or self._batchnorm.num_features != nf:
                self._batchnorm = nn.BatchNorm1d(nf, affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(delta.view(-1, nf)).view(*delta.shape)

        # Adapters
        if not skip_adapters:
            if self.temporal_adapter is not None and delta.shape[-2] >= 3:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        # Activation & clamp
        delta = self.final_activation_fn(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", float("inf")), getattr(self, "max_delta", float("inf")))
        if grad_scale is not None:
            delta = delta * grad_scale

        # Align dimensions
        while delta.ndim > base_ndim:
            delta = delta.squeeze(0)
        if hasattr(self, "_shape"):
            delta = self._tensor_shape(delta, name="context_delta")

        return self._apply_constraints(mod + delta)

    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        tau = max(temperature if temperature is not None else getattr(self, "temperature", 1.0), EPS)
        tau_tensor = torch.as_tensor(tau, dtype=DTYPE, device=self.logits.device)
        key = f"{self._context_hash(context)}-T{float(tau):.6g}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.clone().detach()

        mod = self._apply_context(self.logits, context)
        mod = self._validate_logits(mod)
        mod = mod / tau_tensor

        self._cache_set(key, mod.clone())
        return mod

    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        logits = self._validate_logits(logits)
        return {"logits": logits, **kwargs}

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        if context is None and getattr(self, "context_dim", None) is not None:
            warnings.warn("Context is None but `context_dim` is set. Distribution will be computed without context.")
        mod_logits = self._modulate(context=context, temperature=temperature)
        return self.dist_type(logits=mod_logits)

    # ---------------- Forward / Sampling ----------------
    def forward(self, log=False, return_dist=False, context=None, temperature=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    def log_prob(self, x, context=None, temperature=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if hasattr(dist, "log_prob"):
            return dist.log_prob(x)
        return F.log_softmax(mod_logits, dim=-1).gather(-1, x.long())

    def __log_matrix(self, context=None, temperature=None, return_dist=False, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if return_dist: return dist
        return F.log_softmax(mod_logits, dim=-1)

    def log_matrix(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
        **dist_kwargs
    ) -> torch.Tensor:
        """
        Compute the log-probabilities (or distribution object) with proper
        alignment to [B, T, ...] if context is provided.

        Args:
            context: optional [T,D] or [B,T,D] context tensor
            temperature: optional scaling of logits
            return_dist: if True, returns distribution object
            dist_kwargs: extra kwargs for the distribution constructor

        Returns:
            Tensor of shape [B, T, *event_shape] or a distribution object
        """
        # Step 1: get modulated logits
        mod_logits = self._modulate(context=context, temperature=temperature)

        # Step 2: determine batch (B) and seq_len (T)
        if context is None:
            B, T = 1, 1
        elif context.ndim == 2:
            T, _ = context.shape
            B = 1
        elif context.ndim == 3:
            B, T, _ = context.shape
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

        # Step 3: expand logits to [B, T, ...] if needed
        while mod_logits.ndim < 3:
            mod_logits = mod_logits.unsqueeze(0)
        mod_logits = mod_logits.expand(B, T, *mod_logits.shape[2:])

        # Step 4: optionally return distribution object
        if return_dist:
            return self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))

        # Step 5: return log-probabilities
        return F.log_softmax(mod_logits, dim=-1)

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

    # ---------------- Cache & Hash ----------------
    @torch.no_grad()
    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        c = context.detach().float()
        return str((float(c.mean().item()), float(c.std().item()), int(c.numel()), int(self._param_version.item())))

    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        if hasattr(self, "_cache") and key in self._cache:
            cached = self._cache[key]
            if isinstance(cached, torch.Tensor): return cached
            return getattr(cached, "logits", None)
        return None

    def _cache_set(self, key: str, value: torch.Tensor):
        if not hasattr(self, "_cache"):
            self._cache = dict()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Cannot cache object of type {type(value)}. Must be logits tensor.")
        if len(self._cache) > self.cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value.detach()

    @torch.no_grad()
    def _invalidate_cache(self):
        self._param_version += 1
        self._cache.clear()

    # ---------------- Constraints / Validation ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=True)
            bad_vals = logits[bad_idx]
            raise ValueError(f"Non-finite logits at {bad_idx}: {bad_vals}")
        return logits

    # ---------------- Type property ----------------
    @property
    def dist_type(self):
        return self._dist_type

    @dist_type.setter
    def dist_type(self, value):
        self._dist_type = value

    # ---------------- Initialize & Update ----------------
    @torch.no_grad()
    def initialize(self, mode="uniform", **_):
        return self

    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        from_probs: bool = False,
        grad_safe: bool = False
    ):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)
            new_logits = self._apply_constraints(new_logits)
            new_logits = self._validate_logits(new_logits)
            if new_logits.shape != self.logits.shape:
                raise ValueError(f"Shape mismatch: {new_logits.shape} vs {self.logits.shape}")
            self.logits.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            mod_logits = self._modulate(context=context, temperature=temperature)
            target_shape = mod_logits.shape
            while posterior.ndim < mod_logits.ndim:
                posterior = posterior.unsqueeze(-1)
            if posterior.shape != target_shape:
                posterior = posterior.expand_as(mod_logits)
            posterior_sum = posterior.sum(dim=-1)
            if not torch.allclose(posterior_sum, torch.ones_like(posterior_sum), atol=1e-6):
                raise ValueError(f"Posterior sums !=1 over last dim: {posterior_sum}")
            log_probs = F.log_softmax(mod_logits, dim=-1)
            nll = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            grad_params = [p for p in self.parameters() if p.requires_grad]
            if grad_safe:
                grads = torch.autograd.grad(nll, grad_params, retain_graph=False, allow_unused=True)
                with torch.no_grad():
                    for p, g in zip(grad_params, grads):
                        if g is not None:
                            p.add_((update_rate or 1.0) * g)
            else:
                self.zero_grad()
                nll.backward()
                with torch.no_grad():
                    for p in grad_params:
                        if p.grad is not None:
                            p.add_((update_rate or 1.0) * p.grad)
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
        self.init_mode = init_mode
        self.temperature = temperature

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits.data.copy_(init_logits)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates
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

    # ---------------- Modulation ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False,
    ) -> torch.Tensor:
        # Base logits expanded to [1,1,K] for HSMM
        base = self._tensor_shape(self.logits, name="base_logits")  # [K] -> [1,1,K]
        mod = base

        if context is not None and hasattr(self, "context_gate"):
            # Standardize to [B,T,H]
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(1)
            B, T, H = context.shape

            # Apply context gates
            delta = self.context_gate(context.reshape(B * T, H)).view(B, T, -1)
            if hasattr(self, "residual_gate"):
                delta += self.residual_gate(context.reshape(B * T, H)).view(B, T, -1)

            mod = base.expand(B, T, -1) + delta

            # Temporal adapter if present
            if getattr(self, "temporal_adapter", None) is not None:
                mod = self.temporal_adapter(mod.transpose(1, 2)).transpose(1, 2)

        # Temperature scaling
        tau = max(temperature or self.temperature, 1e-6)
        mod = mod / tau

        if grad_safe:
            mod = mod.detach()

        # Ensure trailing shape matches canonical [K] and maintain batch/time dims
        return self._tensor_shape(mod, name="mod_logits")

    # ---------------- Distribution helpers ----------------
    def _get_dist(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **kwargs
    ) -> Distribution:
        mod_logits = self._modulate(context=context, temperature=temperature)

        # Slice timestep if requested
        if timestep is not None:
            if mod_logits.ndim == 3:
                mod_logits = mod_logits[:, timestep, :]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {mod_logits.shape}")

        # Enforce trailing shape
        mod_logits = self._tensor_shape(mod_logits, name="dist_logits")
        return self._dist_type(logits=mod_logits, **kwargs)

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, timestep: Optional[int] = None):
        return self._get_dist(context, temperature, timestep).sample()

    def log_prob(self, x, context=None, temperature=None, timestep: Optional[int] = None):
        return self._get_dist(context, temperature, timestep).log_prob(x)

    # ---------------- Log-probability matrix ----------------
    def log_matrix(
        self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None
    ) -> torch.Tensor:
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return F.log_softmax(dist.logits, dim=-1)

    def expected_probs(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context, temperature, timestep)
        return dist.probs

    def mode(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context, temperature, timestep)
        return dist.mode if hasattr(dist, "mode") else dist.probs.argmax(-1)

    # ---------------- Robust update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: float = 1.0,
        temperature: Optional[float] = None,
        grad_safe: bool = True,
    ):
        # Direct replacement
        if new_logits is not None:
            if new_logits.ndim > 1:
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))
            new_logits = self._tensor_shape(new_logits, "new_logits")
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)
            if grad_safe:
                new_logits = new_logits.detach()
            self.logits.data.mul_(1 - update_rate).add_(update_rate * new_logits)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        # EM-style posterior update
        if posterior is not None:
            mod_logits = self._modulate(context=context, temperature=temperature)
            while posterior.ndim < mod_logits.ndim:
                posterior = posterior.unsqueeze(-1)
            if posterior.shape != mod_logits.shape:
                posterior = posterior.expand_as(mod_logits)
            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(update_rate * p.grad)
            self._invalidate_cache()


class Duration(DistributionBase):
    """
    Context-aware categorical duration distribution per state for HSMMs.
    Supports batching, sequence context, and optional timestep selection.
    """

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
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
        self.init_mode = init_mode
        self.temperature = max(temperature, 1e-6)
        self._shape = (n_states, max_duration)

        # Base logits
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context/residual gates
        if context_dim is not None:
            hidden_dim = hidden_dim or max(16, n_states)
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

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states, max_duration, mode: str) -> torch.Tensor:
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
    def initialize(self, mode="uniform") -> Distribution:
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._get_dist()

    # ---------------- Tensor shape helper ----------------
    def _tensor_shape(self, x: Optional[torch.Tensor], name="tensor") -> Optional[torch.Tensor]:
        if x is None:
            return None
        x = x.clone()
        target_shape = self._shape  # (n_states, max_duration)
        target_ndim = len(target_shape)

        # Preserve leading batch/time dims
        while x.ndim < target_ndim:
            x = x.unsqueeze(0)

        for i, target in enumerate(target_shape, start=-target_ndim):
            if x.shape[i] == target:
                continue
            elif x.shape[i] == 1:
                expand_sizes = list(x.shape)
                expand_sizes[i] = target
                x = x.expand(*expand_sizes)
            else:
                raise ValueError(f"[{name}] Cannot reshape tensor {x.shape} -> {target_shape}")
        return x

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        base = self._tensor_shape(self.logits, "base_logits")
        mod = base

        if context is not None:
            # Standardize context to [B, T, H]
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape

            # Expand base logits to [B, T, n_states, max_duration]
            mod = base.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

            # Context gate
            if hasattr(self, "context_gate") and self.context_gate is not None:
                delta_ctx = self.context_gate(context.reshape(B*T, H))
                delta_ctx = delta_ctx.unsqueeze(-1).expand(-1, -1, self.max_duration)
                delta_ctx = delta_ctx.view(B, T, self.n_states, self.max_duration)
                mod = mod + delta_ctx

            # Residual gate
            if hasattr(self, "residual_gate") and self.residual_gate is not None:
                delta_res = self.residual_gate(context.reshape(B*T, H))
                delta_res = delta_res.unsqueeze(-1).expand(-1, -1, self.max_duration)
                delta_res = delta_res.view(B, T, self.n_states, self.max_duration)
                mod = mod + delta_res

        tau = max(temperature or self.temperature, 1e-6)
        mod = mod / tau
        if grad_safe:
            mod = mod.detach()

        # Enforce trailing shape strictly
        mod = self._tensor_shape(mod, "mod_logits")
        return mod

    # ---------------- Log-matrix ----------------
    def log_matrix(self, context=None, temperature=None, timestep: Optional[int] = None) -> torch.Tensor:
        mod_logits = self._modulate(context, temperature)
        log_probs = F.log_softmax(mod_logits, dim=-1)
        if timestep is not None:
            if log_probs.ndim == 4:
                log_probs = log_probs[:, timestep, :, :]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {log_probs.shape}")
        return log_probs

    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, timestep=None) -> Distribution:
        log_probs = self.log_matrix(context=context, temperature=temperature, timestep=timestep)
        log_probs = self._tensor_shape(log_probs, "dist_logits")
        return Categorical(logits=log_probs)

    # ---------------- Sampling ----------------
    def sample(self, context=None, temperature=None, timestep=None):
        return self._get_dist(context, temperature, timestep).sample()

    def rsample(self, context=None, temperature=None):
        return self._get_dist(context, temperature).rsample()

    def mode(self, context=None, temperature=None, timestep=None, return_dist=False):
        dist = self._get_dist(context, temperature, timestep)
        return dist if return_dist else dist.probs.argmax(-1)

    # ---------------- Update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits=None,
        posterior=None,
        context=None,
        from_probs=False,
        update_rate=1.0,
        temperature=None,
        grad_safe=True,
    ):
        if new_logits is not None:
            while new_logits.ndim > len(self._shape):
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))
            new_logits = self._tensor_shape(new_logits, "new_logits")
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)
            if grad_safe:
                new_logits = new_logits.detach()
            self.logits.data.mul_(1 - update_rate).add_(update_rate * new_logits)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        if posterior is not None:
            log_probs = self.log_matrix(context=context, temperature=temperature)
            while posterior.ndim < log_probs.ndim:
                posterior = posterior.unsqueeze(-1)
            if posterior.shape != log_probs.shape:
                posterior = posterior.expand_as(log_probs)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(update_rate * p.grad)
            self._invalidate_cache()


class Transition(DistributionBase):
    """
    Context-aware categorical transition distribution for HSMMs.
    Supports batch/time context, optional low-rank factorization, 
    and timestep selection.
    """

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = 64,
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
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.rank = rank
        self.temperature = max(temperature, 1e-6)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self._shape = (n_states, n_states)

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context/residual/low-rank gates
        if context_dim is not None:
            hidden_dim = hidden_dim or max(16, n_states)
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states * n_states, dtype=DTYPE),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states * n_states, dtype=DTYPE),
            )
            if rank is not None:
                self._U = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)
                self._V = nn.Linear(context_dim, n_states * rank, dtype=DTYPE)

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
    def initialize(self, mode="diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._get_dist()

    # ---------------- Tensor shape helper ----------------
    def _tensor_shape(self, x: torch.Tensor, name="tensor") -> torch.Tensor:
        target_shape = self._shape
        target_ndim = len(target_shape)
        leading_ndim = x.ndim - target_ndim
        if leading_ndim < 0:
            x = x.view((1,) * (-leading_ndim) + x.shape)
            leading_ndim = 0
        for i, t in enumerate(target_shape, start=-target_ndim):
            if x.shape[i] == t:
                continue
            elif x.shape[i] == 1:
                sz = list(x.shape)
                sz[i] = t
                x = x.expand(*sz)
            else:
                raise ValueError(f"[{name}] cannot reshape {x.shape} -> {target_shape}")
        return x

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        tau = max(temperature or self.temperature, 1e-6)
        key = f"{self._context_hash(context)}-T{float(tau):.6g}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        base = self._tensor_shape(self.logits, "base_logits")
        if context is None:
            mod_logits = base
        else:
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B*T, H)
            mod_logits = base.unsqueeze(0).unsqueeze(0).expand(B, T, *self._shape)
            delta_total = torch.zeros_like(mod_logits)

            if getattr(self, "context_gate", None):
                delta_total += self.context_gate(ctx_flat).view(B, T, *self._shape)
            if getattr(self, "residual_gate", None):
                delta_total += self.residual_gate(ctx_flat).view(B, T, *self._shape)
            if getattr(self, "_U", None) and getattr(self, "_V", None):
                r = self.rank
                U = self._U(ctx_flat).view(B*T, self.n_states, r)
                V = self._V(ctx_flat).view(B*T, self.n_states, r)
                delta_total += torch.einsum("bik,bjk->bij", U, V).view(B, T, *self._shape)

            mod_logits = mod_logits + delta_total

        mod_logits = mod_logits / tau
        if grad_safe:
            mod_logits = mod_logits.detach()
        mod_logits = self._tensor_shape(mod_logits, "modulated_logits")
        self._cache_set(key, mod_logits if grad_safe else mod_logits.detach())
        return mod_logits

    # ---------------- Log-matrix ----------------
    def log_matrix(self, context=None, temperature=None, timestep: Optional[int] = None) -> torch.Tensor:
        mod_logits = self._modulate(context, temperature)
        log_probs = F.log_softmax(mod_logits, dim=-1)

        if timestep is not None:
            if log_probs.ndim == len(self._shape) + 2:  # [B,T,K,K]
                log_probs = log_probs[:, timestep, ...]
            elif log_probs.ndim == len(self._shape) + 1:  # [T,K,K]
                log_probs = log_probs[timestep, ...]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {log_probs.shape}")
        return log_probs

    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, timestep=None) -> Categorical:
        log_probs = self.log_matrix(context, temperature, timestep)
        log_probs = self._tensor_shape(log_probs, "dist_logits")
        return Categorical(logits=log_probs)

    # ---------------- Sampling / mode ----------------
    def sample(self, context=None, temperature=None, timestep=None, **kwargs):
        return self._get_dist(context, temperature, timestep).sample(**kwargs)

    def rsample(self, context=None, temperature=None, timestep=None, **kwargs):
        return self._get_dist(context, temperature, timestep).rsample(**kwargs)

    def mode(self, context=None, temperature=None, timestep=None, return_dist=False):
        dist = self._get_dist(context, temperature, timestep)
        return dist if return_dist else dist.probs.argmax(-1)

    # ---------------- Update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        context=None,
        temperature=None,
        update_rate: float = 1.0,
        grad_safe: bool = True,
    ):
        tau = max(temperature or self.temperature, 1e-6)

        if new_logits is not None:
            new_logits = self._tensor_shape(new_logits, "new_logits")
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            new_logits = self._apply_constraints(new_logits)
            new_logits = self._validate_logits(new_logits)
            canonical = new_logits.view(-1, *self._shape).mean(dim=0)
            self.logits.data.mul_(1 - update_rate).add_(update_rate * canonical)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        if posterior is not None:
            posterior = self._tensor_shape(posterior, "posterior")
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
            new_logits = torch.log(posterior.clamp_min(EPS))
            canonical = new_logits.view(-1, *self._shape).mean(dim=0)
            self.logits.data.mul_(1 - update_rate).add_(update_rate * canonical)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        super().update(
            new_logits=new_logits,
            posterior=posterior,
            from_probs=from_probs,
            context=context,
            temperature=temperature,
            update_rate=update_rate,
            grad_safe=grad_safe,
        )


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

        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            spatial_adapter=spatial_adapter,
            target_dim=n_states * n_features,
            temporal_adapter=temporal_adapter,
            allow_projection=allow_projection,
            debug=debug,
        )

        self._shape = (n_states, n_features)

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
        else:
            self.context_gate = None

    @torch.no_grad()
    def _tensor_shape(self, x: Optional[torch.Tensor], name="tensor") -> Optional[torch.Tensor]:
        """Align any tensor to [B, T, K, F] (or [K, F] for base params)."""
        if x is None:
            return None

        x = x.clone()
        target_shape = tuple(self._shape)  # (K, F)
        n_trailing = len(target_shape)

        # Leading dims = anything before K,F
        leading_shape = x.shape[:-n_trailing] if x.ndim > n_trailing else ()

        # Collapse extra trailing dims (mean over extras)
        while x.ndim > len(leading_shape) + n_trailing:
            x = x.mean(dim=-1)

        # Expand missing trailing dims
        while x.ndim < len(leading_shape) + n_trailing:
            x = x.unsqueeze(-1)

        # Match trailing dims [K, F] safely
        for i, ts in enumerate(target_shape):
            dim = -n_trailing + i
            if x.shape[dim] == ts:
                continue
            elif x.shape[dim] == 1:
                x = x.expand(*x.shape[:dim], ts, *x.shape[dim+1:])
            else:
                raise ValueError(f"[{name}] Cannot reshape {x.shape} to target {target_shape}")

        return x

    def _modulate(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        """Apply context modulation and temperature scaling for discrete distributions."""
        mod = self._apply_context(base, context) if context is not None else base
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            tau = max(temperature or getattr(self, "temperature", 1.0), 1e-6)
            mod = mod / tau

        # Clamp to avoid NaN for discrete distributions
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            mod = torch.nan_to_num(mod, nan=0.0, posinf=1e6, neginf=-1e6)

        return self._tensor_shape(mod, name="modulated")

    def _dist_params(self, tensor: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        """Return distribution parameters with consistent shapes and safe values."""
        mod = self._modulate(tensor, context=context, temperature=temperature)

        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)
            return {"means": mod, "cov": cov}

        elif self.emission_type in {"laplace", "studentt"}:
            scale = self.scale_param.clamp_min(self.min_covar)
            return {"loc": mod, "scale": scale}

        elif self.emission_type in {"categorical", "bernoulli"}:
            # Ensure finite logits
            mod = torch.nan_to_num(mod, nan=0.0)
            return {"logits": mod}

        elif self.emission_type == "poisson":
            # Poisson expects logits (log-rate); clamp to avoid NaN
            mod = torch.nan_to_num(mod, nan=0.0)
            return {"logits": mod}

        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    def _get_continuous_dist(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
        max_jitter: int = 5,
    ):
        K, F = self.n_states, self.n_features
        device, dtype = self._emission_means.device, DTYPE

        # ---------------- Base means ----------------
        means = self.mu if self.emission_type == "gaussian" else self.loc
        means = means.clone()

        # Weighted EM update
        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)  # [N, K]
            w_sum = w.sum(dim=0, keepdim=True) + EPS  # [1, K]
            means = (w.T @ X) / w_sum.T  # [K, F]

        # Theta modulation
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            means += theta_scale * theta_vec.unsqueeze(0)

        # Context + temperature modulation
        means_mod = self._modulate(means, context=context, temperature=temperature)
        means_mod = self._tensor_shape(means_mod, name="means_mod")

        if self.emission_type == "gaussian":
            covs = self._emission_covs.clone()
            I = torch.eye(F, dtype=dtype, device=device)
            for k in range(K):
                jitter = self.min_covar
                for _ in range(max_jitter):
                    _, info = torch.linalg.cholesky_ex(covs[k])
                    if info == 0:
                        break
                    covs[k] += jitter * I
                    jitter *= 2

            with torch.no_grad():
                self._emission_means.copy_(means_mod)
                self._emission_covs.copy_(covs)
                self.mu.copy_(means_mod)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1).clamp_min(EPS)))

            return MultivariateNormal(means_mod, covariance_matrix=covs)

        # Laplace / StudentT
        scale = self.scale_param.clamp_min(self.min_covar)
        scale_mod = self._tensor_shape(scale, name="scale")
        dist_cls = Laplace if self.emission_type == "laplace" else StudentT

        with torch.no_grad():
            self._emission_means.copy_(means_mod)
            self._emission_covs.copy_(torch.diag_embed(scale_mod**2))
            self.loc.copy_(means_mod)
            self.scale_param.copy_(scale_mod)

        if self.emission_type == "studentt":
            return Independent(dist_cls(df=self.dof, loc=means_mod, scale=scale_mod), 1)
        return Independent(dist_cls(loc=means_mod, scale=scale_mod), 1)

    def _get_discrete_dist(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
    ):
        """
        Return a discrete emission distribution with proper [B, T, K, F] shape.
        """
        K, F = self.n_states, self.n_features
        device, dtype = self._emission_means.device, DTYPE

        # Base logits
        logits = torch.zeros(K, F, dtype=dtype, device=device)
        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)  # [N, K]
            w_sum = w.sum(dim=0, keepdim=True) + EPS  # [1, K]

            if self.emission_type == "categorical":
                X_long = X.long()
                for f in range(F):
                    idx = X_long[:, f]
                    logits[:, f] = (w * (idx[:, None] == torch.arange(K, device=device))).sum(dim=0)
                logits = torch.log((logits / logits.sum(dim=-1, keepdim=True)).clamp_min(EPS))
            else:  # Bernoulli / Poisson
                rate = (w.T @ X.float()) / w_sum.T
                logits = torch.log(rate.clamp_min(EPS))

        # Theta modulation
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            logits += theta_scale * theta_vec.unsqueeze(0)

        # Context + temperature modulation
        logits_mod = self._modulate(logits, context=context, temperature=temperature)
        logits_mod = self._tensor_shape(logits_mod, name="logits_mod")  # [B, T, K, F]

        # Update buffers
        with torch.no_grad():
            self._emission_params.copy_(logits_mod)
            if self.emission_type in {"categorical", "bernoulli"}:
                self.logits.copy_(logits_mod)
            elif self.emission_type == "poisson":
                self.log_rate.copy_(logits_mod)

        # Construct distribution
        dist_params = self._dist_params(logits_mod, context=context, temperature=temperature)
        if self.emission_type == "categorical":
            self.dist_type = Categorical
            return Categorical(logits=dist_params["logits"])
        elif self.emission_type == "bernoulli":
            self.dist_type = lambda *a, **kw: Independent(Bernoulli(*a, **kw), 1)
            return Independent(Bernoulli(logits=dist_params["logits"]), 1)
        elif self.emission_type == "poisson":
            self.dist_type = lambda *a, **kw: Independent(Poisson(*a, **kw), 1)
            return Independent(Poisson(rate=torch.exp(dist_params["logits"])), 1)

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
        if etype in {"gaussian", "laplace", "studentt"}:
            return self._get_continuous_dist(
                X=X, theta=theta, context=context, posterior=posterior,
                theta_scale=theta_scale, temperature=temperature, max_jitter=max_jitter
            )
        elif etype in {"categorical", "bernoulli", "poisson"}:
            return self._get_discrete_dist(
                X=X, theta=theta, context=context, posterior=posterior,
                theta_scale=theta_scale, temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    def forward(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist=False):
        """Return distribution or tensor of parameters aligned for HSMM [B, T, K, F]."""
        dist = self._get_dist(context=context, temperature=temperature)
        if return_dist:
            return dist

        if self.emission_type == "gaussian":
            return dist.mean, dist.covariance_matrix
        if self.emission_type in {"laplace", "studentt"}:
            return dist.loc, dist.scale
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            tensor = self.logits if self.emission_type != "poisson" else self.log_rate
            return self._dist_params(tensor, context=context, temperature=temperature)["logits"]

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        """Sample n_samples from emission distribution with HSMM batch awareness."""
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        samples = dist.rsample((n_samples,)) if getattr(dist, "has_rsample", False) else dist.sample((n_samples,))
        if isinstance(dist, Independent):
            # collapse last two dims for [K, F] shape
            samples = samples.view(n_samples, *samples.shape[-2:])
        return samples.to(dtype=DTYPE)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        """Compute log-probabilities [B, T, K, F] for HSMM sequences."""
        K, F = self.n_states, self.n_features
        tau = max(temperature or getattr(self, "temperature", 1.0), EPS)
        B, T = x.shape[:2] if x.ndim >= 3 else (x.shape[0], 1)

        # Expand input x to [B, T, K, F]
        def expand_x(x_tensor):
            if x_tensor.ndim == 4:
                return x_tensor
            if x_tensor.ndim == 2:  # [B*T, F] -> [B*T, 1, K, F]
                return x_tensor.unsqueeze(1).unsqueeze(2).expand(-1, 1, K, F)
            if x_tensor.ndim == 3:  # [B, K, F] -> [B, 1, K, F]
                return x_tensor.unsqueeze(1).expand(-1, 1, -1, -1)
            raise ValueError(f"Cannot expand tensor of shape {x_tensor.shape}")

        x_exp = expand_x(x)

        if self.emission_type in {"categorical", "bernoulli"}:
            logits = self._modulate(self.logits, context=context, temperature=tau)
            log_probs = F.log_softmax(logits, dim=-1).unsqueeze(0).unsqueeze(0).expand(B, T, K, F)
            return torch.gather(log_probs, -1, x_exp.long().unsqueeze(-1)).squeeze(-1)

        if self.emission_type == "poisson":
            logits = self._modulate(self.log_rate, context=context, temperature=tau)
            log_probs = x_exp * logits.exp().clamp_min(EPS).log() - logits.exp().clamp_min(EPS)
            return log_probs

        # Continuous distributions
        base = self.mu if self.emission_type == "gaussian" else self.loc
        loc = self._modulate(base, context=context, temperature=tau)
        loc_exp = loc.unsqueeze(0) if loc.ndim == 2 else loc

        if self.emission_type == "gaussian":
            cov_exp = self._emission_covs.unsqueeze(0)  # [1, K, F, F]
            diff = x_exp - loc_exp
            L = torch.linalg.cholesky(cov_exp)
            sol = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
            log_det = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            return -0.5 * (sol.squeeze(-1)**2).sum(-1) - 0.5 * F * math.log(2 * math.pi) - 0.5 * log_det

        # Laplace / StudentT not vectorized yet
        if self.emission_type in {"laplace", "studentt"}:
            scale = self.scale_param.clamp_min(EPS)
            return -(x_exp - loc_exp).abs() / scale - scale.log()  # simplified approximation

        raise NotImplementedError(f"Emission type {self.emission_type} not implemented for log_prob.")

    def parameters_tensor(self) -> torch.Tensor:
        """Return base parameters for HSMM usage (K, F)."""
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
        mode: str = "kmeans",
        iters: int = 25,
    ):
        """
        Initialize emission parameters for continuous and discrete distributions.
        Uses theta/context modulation and ensures shapes are consistent via _tensor_shape.
        """
        K, F = self.n_states, self.n_features
        device = X.device if X is not None else self._emission_means.device

        Xf = X.reshape(-1, F).to(dtype=DTYPE) if X is not None else None

        # ---------------- Continuous emissions ----------------
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # Means
            if Xf is not None:
                if mode == "kmeans" and Xf.shape[0] >= K:
                    idx = torch.randperm(Xf.shape[0], device=device)[:K]
                    means = Xf[idx].clone()
                    for _ in range(iters):
                        dist = torch.cdist(Xf, means)
                        labels = dist.argmin(dim=1)
                        for k in range(K):
                            pts = Xf[labels == k]
                            if pts.numel() > 0:
                                means[k] = pts.mean(0)
                else:
                    means = Xf.mean(0).expand(K, F)
                cov_base = torch.cov(Xf.T) + self.min_covar * torch.eye(F, dtype=DTYPE, device=device)
                covs = cov_base.unsqueeze(0).repeat(K, 1, 1)
            else:
                means = torch.zeros(K, F, dtype=DTYPE, device=device)
                covs = torch.eye(F, dtype=DTYPE, device=device).unsqueeze(0).repeat(K, 1, 1) * self.min_covar

            # Theta modulation
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                means += theta_scale * theta_vec.unsqueeze(0)

            # Context + temperature modulation
            means_mod = self._modulate(means, context=context, temperature=temperature)
            means_mod = self._tensor_shape(means_mod, name="means_mod")

            # Update buffers
            self._emission_means.copy_(means_mod)
            self._emission_covs.copy_(covs)
            if self.emission_type == "gaussian":
                self.mu.copy_(means_mod)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1).clamp_min(EPS)))
            else:
                self.loc.copy_(means_mod)
                self.scale_param.copy_(torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar))

            return self._get_dist(context=context, temperature=temperature)

        # ---------------- Discrete emissions ----------------
        elif self.emission_type in {"categorical", "bernoulli", "poisson"}:
            if Xf is not None and mode == "data":
                if self.emission_type == "categorical":
                    counts = torch.stack([torch.bincount(Xf[:, f].long(), minlength=K) for f in range(F)], dim=1)
                    logits = torch.log((counts / counts.sum(dim=0, keepdim=True)).clamp_min(EPS))
                else:  # Bernoulli / Poisson
                    logits = torch.log(Xf.mean(0).expand(K, F).clamp_min(EPS))
            else:
                logits = torch.zeros(K, F, dtype=DTYPE, device=device)

            # Theta/context modulation
            logits_mod = self._modulate(logits, context=context, temperature=temperature)
            logits_mod = self._tensor_shape(logits_mod, name="logits_mod")

            # Update learnable buffers
            if hasattr(self, "logits"):
                self.logits.copy_(logits_mod)
            elif hasattr(self, "log_rate"):
                self.log_rate.copy_(logits_mod)
            self._emission_params.copy_(logits_mod)

            return self._get_dist(context=context, temperature=temperature)

        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

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
        Update emission parameters via EMA using new data/posterior/context.
        """
        etype = self.emission_type
        K, F = self.n_states, self.n_features
        device = self._emission_means.device

        # ---------------- Get new distribution ----------------
        new_dist = self._get_dist(
            X=X,
            posterior=posterior,
            theta=theta,
            context=context,
            theta_scale=theta_scale,
            temperature=temperature,
            max_jitter=max_jitter,
        )

        # ---------------- Continuous emissions ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            loc = getattr(new_dist, "loc", getattr(new_dist, "mean", None)).to(dtype=DTYPE, device=device)
            if etype == "gaussian":
                cov = new_dist.covariance_matrix.to(dtype=DTYPE, device=device)
                self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)
                self._emission_covs.mul_(1 - update_rate).add_(update_rate * cov)
                self.mu.copy_(self._emission_means)
                diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1).clamp_min(EPS)
                self.log_var.copy_(torch.log(diag))
            else:
                scale = getattr(new_dist, "scale", None).to(dtype=DTYPE, device=device)
                self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)
                diag = ((1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1)
                        + update_rate * scale**2).clamp_min(EPS)
                self._emission_covs.copy_(torch.diag_embed(diag))
                self.loc.copy_(self._emission_means)
                self.scale_param.copy_(torch.sqrt(diag))

        # ---------------- Discrete emissions ----------------
        else:
            params = getattr(new_dist, "logits", getattr(new_dist, "rate", None))
            params = params.to(dtype=DTYPE, device=device)
            params_mod = self._tensor_shape(params, name="discrete_params")
            if hasattr(self, "_emission_params"):
                self._emission_params.mul_(1 - update_rate).add_(update_rate * params_mod)
            if etype in {"categorical", "bernoulli"}:
                self.logits.copy_(self._emission_params)
            elif etype == "poisson":
                self.log_rate.copy_(self._emission_params)

        return new_dist

