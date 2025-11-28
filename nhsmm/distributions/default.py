# nhsmm/distributions/default.py

from __future__ import annotations
from typing import Optional, Union, Literal, Tuple, Dict, Any
from collections import OrderedDict
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
      - Gumbel-Softmax sampling (hard + relaxed)
      - Stable logits/probs handling
      - Temperature scaling
      - Compatible with torch.distributions API
      - Broadcast-safe batch/time handling
    """

    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }
    has_rsample = True

    def __init__(self, logits=None, probs=None, validate_args=False, dim=-1):
        super().__init__(validate_args=validate_args)
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs.")

        self.dim = dim

        if logits is not None:
            logits = logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)
            self._logits = logits
            self._probs = F.softmax(logits, dim=dim)
        else:
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            self._probs = probs
            self._logits = probs.clamp(min=EPS).log()

        self._logits = self._logits.to(DTYPE)
        self._probs = self._probs.to(DTYPE)

    @property
    def logits(self):
        return self._logits

    @property
    def probs(self):
        return self._probs

    @property
    def batch_shape(self):
        return self._logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape
        flat = self._probs.reshape(-1, self._probs.size(self.dim))
        idx = torch.multinomial(flat, 1, replacement=True)
        idx = idx.reshape(*shape)
        return idx

    def rsample(self, sample_shape=torch.Size(), temperature=1.0, hard=False):
        temperature = max(float(temperature), EPS)
        shape = sample_shape + self._logits.shape
        logits = self._logits.expand(shape)

        y = F.gumbel_softmax(
            logits,
            tau=temperature,
            hard=hard,
            dim=self.dim,
        )
        return y

    def log_prob(self, value):
        if value.dim() == 0:
            value = value.unsqueeze(0)
        lp = F.log_softmax(self._logits, dim=self.dim)
        return lp.gather(self.dim, value.long().unsqueeze(self.dim)).squeeze(self.dim)

    def entropy(self):
        p = self._probs
        return -(p * p.clamp_min(EPS).log()).sum(dim=self.dim)

    def mode(self):
        return self._logits.argmax(dim=self.dim)


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
        self.cache_grad_safe = cache_grad_safe
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.max_delta = max_delta
        self.temperature = 1.0
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
            self._proj = nn.Linear(context_dim, context_dim, bias=True, dtype=DTYPE)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)
        else:
            self._proj = None

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
        self._shape = tuple(self._infer_shape())

    # ---------------- Shape ----------------
    def _infer_shape(self) -> Tuple[int, ...]:
        if getattr(self, "_shape", None) is not None:
            shape = tuple(self._shape)
            if len(shape) == 0:
                raise ValueError("Event shape cannot be empty.")
            return shape

        es = getattr(self.dist_type, "event_shape", None)
        if isinstance(es, (tuple, list)) and len(es) > 0:
            return tuple(int(s) for s in es)

        if not hasattr(self, "target_dim"):
            raise RuntimeError("Cannot infer shape: no _shape, no event_shape, no target_dim.")
        return (int(self.target_dim),)

    @torch.no_grad()
    def _tensor_shape(self, x: Optional[torch.Tensor], name="tensor", atol=0) -> Optional[torch.Tensor]:
        """Safely reshape/broadcast a tensor to match the distribution's trailing shape.

        - Preserves leading dimensions
        - Enforces target trailing shape
        - Supports tolerant matching via `atol`
        - Avoids ambiguous squeezing/unsqueezing behavior
        """
        if x is None:
            return None

        # small optimization: only clone if we’ll modify it
        if not x.is_contiguous():
            x = x.contiguous()

        target = tuple(self._shape)
        n_target = len(target)
        x_shape = x.shape

        # Extract leading shape
        leading = x_shape[:-n_target] if x.ndim >= n_target else x_shape[:0]

        # 1) If x has TOO MANY trailing dims: collapse only the excess part
        #    (avoids destructive "while" loops and preserves ordering)
        if x.ndim > len(leading) + n_target:
            extra = x.ndim - (len(leading) + n_target)
            # Collapse all extra dims into the last retained leading dimension
            collapse_dims = tuple(range(len(leading), len(leading) + extra))
            x = x.mean(dim=collapse_dims)

            # make shape consistent
            x = x.reshape(*leading, *x.shape[-n_target:])

        # 2) If x has TOO FEW trailing dims: append singleton dims
        missing = (len(leading) + n_target) - x.ndim
        if missing > 0:
            x = x.view(*x.shape, *([1] * missing))

        # 3) Align trailing dims
        for i, tgt in enumerate(target):
            dim = -n_target + i
            cur = x.shape[dim]

            if cur == tgt:
                continue

            # allow `1 → tgt` expansion
            if cur == 1:
                expand = list(x.shape)
                expand[dim] = tgt
                x = x.expand(*expand)
                continue

            # tolerant mode
            if atol > 0 and abs(cur - tgt) <= atol:
                if cur < tgt:
                    expand = list(x.shape)
                    expand[dim] = tgt
                    x = x.expand(*expand)
                else:
                    # cur > tgt and within tolerance: slice
                    x = x.narrow(dim, 0, tgt)
                continue

            raise ValueError(
                f"[{name}] Cannot reshape {tuple(x_shape)} → {target}. "
                f"Failed at dim {dim}: {cur} vs {tgt}."
            )

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

    def _apply_temperature(self, logits: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Scale logits by temperature. If temperature is None, uses learnable log_temperature.
        
        Args:
            logits: tensor of logits
            temperature: optional scalar; overrides learnable temperature

        Returns:
            logits scaled by temperature
        """
        tau = self.log_temperature.exp() if temperature is None else max(temperature, EPS)
        tau_tensor = torch.as_tensor(tau, dtype=logits.dtype, device=logits.device)
        return logits / tau_tensor

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

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        """
        Modulate base logits with context, apply constraints, validate, and scale by temperature.

        Caching ensures repeated contexts with same temperature reuse results.

        Args:
            context: optional context tensor
            temperature: overrides learnable temperature if provided
            grad_safe: return detached logits if True
        """
        tau_val = float(self.log_temperature.exp()) if temperature is None else float(max(temperature, EPS))
        key = f"{self._context_hash(context)}-T{tau_val:.6g}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.clone().detach()

        # Context modulation
        mod = self._apply_context(self.logits, context)

        # Validation
        mod = self._validate_logits(mod)

        # Temperature scaling
        mod = self._apply_temperature(mod, temperature)

        # Cache result
        self._cache_set(key, mod.clone())

        return mod

    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        logits = self._validate_logits(logits)
        return {"logits": logits, **kwargs}

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        if context is None and getattr(self, "context_dim", None) is not None:
            logger.warning("Context is None but `context_dim` is set. Distribution will be computed without context.")
        mod_logits = self._modulate(context=context, temperature=temperature)
        return self.dist_type(logits=mod_logits)

    def forward(self, log=False, return_dist=False, context=None, temperature=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if return_dist: return dist
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    def log_prob(self, x, context=None, temperature=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if hasattr(dist, "log_prob"): return dist.log_prob(x)
        return F.log_softmax(mod_logits, dim=-1).gather(-1, x.long())

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None,
                   return_dist: bool = False, **dist_kwargs) -> torch.Tensor:
        mod_logits = self._modulate(context=context, temperature=temperature)

        # Determine batch (B) and seq_len (T)
        if context is None:
            B, T = 1, 1
        elif context.ndim == 2:
            T, _ = context.shape
            B = 1
        elif context.ndim == 3:
            B, T, _ = context.shape
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

        # Expand logits to [B, T, ...] if needed
        while mod_logits.ndim < 3:
            mod_logits = mod_logits.unsqueeze(0)
        mod_logits = mod_logits.expand(B, T, *mod_logits.shape[2:])

        if return_dist:
            return self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        return F.log_softmax(mod_logits, dim=-1)

    def sample(self, context=None, temperature=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

    def expected_probs(self, context=None, temperature=None, return_dist=False, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        return dist if return_dist else F.softmax(mod_logits, dim=-1)

    def mode(self, context=None, temperature=None, return_dist=False, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod_logits, tau=temperature, **dist_kwargs))
        if return_dist: return dist
        if hasattr(dist, "mode"): return dist.mode
        if hasattr(dist, "probs"): return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

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

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=True)
            bad_vals = logits[bad_idx]
            raise ValueError(f"Non-finite logits at {bad_idx}: {bad_vals}")
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
        hidden_dim: Optional[int] = None,
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
        self._logits_buffer.copy_(init_logits)
        self._mod_logits_buffer.copy_(init_logits)

        # Context gates
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

    # ---------------- Context handling ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Extend DistributionBase `_apply_context`:
        - Applies context_gate and residual_gate if present
        - Preserves batch/time handling
        """
        # Base logic: adapters, normalization, activation
        mod = super()._apply_context(base, context=context, skip_adapters=skip_adapters, l2_normalize=l2_normalize)

        if context is not None and hasattr(self, "context_gate"):
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(1)
            B, T, H = context.shape

            delta = self.context_gate(context.reshape(B * T, H)).view(B, T, -1)
            if hasattr(self, "residual_gate"):
                delta += self.residual_gate(context.reshape(B * T, H)).view(B, T, -1)

            # Add gate delta to modulated logits
            mod = mod.expand(B, T, -1) + delta

        return self._tensor_shape(mod, name="context_mod_logits")

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False):
        base = self._tensor_shape(self.logits, name="base_logits")
        tau_val = float(self.temperature if temperature is None else max(temperature, 1e-6))
        context_key = self._context_hash(context)
        key = f"modulate-{base.mean().item():.6g}-{base.std().item():.6g}-{context_key}-T{tau_val:.6g}"

        cached = self._cache_get(key)
        if cached is not None:
            if grad_safe:
                return cached.clone()
            else:
                return cached.clone().requires_grad_(True)  # restore grad

        mod = self._apply_context(base, context=context)
        mod = self._apply_temperature(mod, tau_val)
        mod = self._tensor_shape(mod, "modulated")

        self._cache_set(key, mod.clone().detach())
        return mod.detach() if grad_safe else mod

    # ---------------- Distribution helpers ----------------
    def _get_dist(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **kwargs
    ) -> Distribution:
        mod_logits = self._modulate(context=context, temperature=temperature)
        if timestep is not None:
            if mod_logits.ndim == 3:
                mod_logits = mod_logits[:, timestep, :]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {mod_logits.shape}")
        mod_logits = self._tensor_shape(mod_logits, name="dist_logits")
        return self._dist_type(logits=mod_logits, **kwargs)

    # ---------------- Standard API ----------------
    def sample(self, context=None, temperature=None, timestep: Optional[int] = None, **kwargs):
        return self._get_dist(context, temperature, timestep, **kwargs).sample()

    def log_prob(self, x, context=None, temperature=None, timestep: Optional[int] = None, **kwargs):
        return self._get_dist(context, temperature, timestep, **kwargs).log_prob(x)

    def log_matrix(self, context=None, temperature=None, timestep: Optional[int] = None, **kwargs):
        return F.log_softmax(self._get_dist(context, temperature, timestep, **kwargs).logits, dim=-1)

    def expected_probs(self, context=None, temperature=None, timestep: Optional[int] = None, **kwargs):
        return self._get_dist(context, temperature, timestep, **kwargs).probs

    def mode(self, context=None, temperature=None, timestep: Optional[int] = None, **kwargs):
        dist = self._get_dist(context, temperature, timestep, **kwargs)
        return getattr(dist, "mode", None) or dist.probs.argmax(-1)

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
        if new_logits is not None and new_logits.ndim > 1:
            new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))
        super().update(
            new_logits=new_logits,
            posterior=posterior,
            context=context,
            from_probs=from_probs,
            update_rate=update_rate,
            temperature=temperature,
            grad_safe=grad_safe
        )


class Duration(DistributionBase):
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
        self._shape = (n_states, max_duration)
        self.temperature = max(temperature, EPS)

        # Base logits
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Learnable duration bias (effectively "soft max_duration")
        self.log_duration = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE))

        # Context/residual gates
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

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        base = self._tensor_shape(self.logits, "base_logits")

        # Add learnable duration bias
        base = base + self.log_duration

        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        context_key = self._context_hash(context)
        key = f"duration-{base.mean().item():.6g}-{base.std().item():.6g}-{context_key}-T{tau_val:.6g}"

        cached = self._cache_get(key)
        if cached is not None:
            return cached.clone().detach() if grad_safe else cached

        # Apply context
        mod = base
        if context is not None and hasattr(self, "context_gate"):
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B * T, H)

            mod = base.unsqueeze(0).unsqueeze(0).expand(B, T, self.n_states, self.max_duration)

            delta_ctx = self.context_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, self.max_duration)
            delta_res = self.residual_gate(ctx_flat).unsqueeze(-1).expand(-1, -1, self.max_duration)

            mod = mod + delta_ctx.view(B, T, self.n_states, self.max_duration)
            mod = mod + delta_res.view(B, T, self.n_states, self.max_duration)

        # Apply temperature
        mod = mod / tau_val
        mod = self._tensor_shape(mod, "modulated")
        self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

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
        mod_logits = self._modulate(context, temperature)
        if timestep is not None:
            if mod_logits.ndim == 4:
                mod_logits = mod_logits[:, timestep, :, :]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {mod_logits.shape}")
        mod_logits = self._tensor_shape(mod_logits, "dist_logits")
        return self._dist_type(logits=mod_logits)

    # ---------------- Sampling / mode ----------------
    sample = lambda self, *a, **k: self._get_dist(*a, **k).sample()
    rsample = lambda self, *a, **k: self._get_dist(*a, **k).rsample() if hasattr(self._get_dist(*a, **k), "rsample") else self._get_dist(*a, **k).sample()
    mode = lambda self, *a, return_dist=False, **k: self._get_dist(*a, **k) if return_dist else self._get_dist(*a, **k).probs.argmax(-1)
    expected_probs = lambda self, *a, **k: self._get_dist(*a, **k).probs

    # ---------------- Robust vectorized update ----------------
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
            mod_logits = self._modulate(context=context, temperature=temperature)
            while posterior.ndim < mod_logits.ndim:
                posterior = posterior.unsqueeze(-1)
            posterior = posterior.expand_as(mod_logits)

            # Vectorized EMA update using expected log-probs
            log_probs = F.log_softmax(mod_logits, dim=-1)
            expected_logits = torch.log((posterior * log_probs.exp()).sum(dim=0).clamp_min(EPS))
            self.logits.data.mul_(1 - update_rate).add_(update_rate * expected_logits)
            self._invalidate_cache()


class Transition(DistributionBase):
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        rank: Optional[int] = None,
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

        self.rank = rank
        self.n_states = n_states
        self._shape = (n_states, n_states)
        self.temperature = max(temperature, EPS)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context/residual/low-rank gates
        if context_dim is not None and hidden_dim is not None:
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

    @torch.no_grad()
    def _tensor_shape(self, x: torch.Tensor, name="tensor") -> torch.Tensor:
        target_ndim = len(self._shape)
        leading_ndim = x.ndim - target_ndim
        if leading_ndim < 0:
            x = x.view((1,) * (-leading_ndim) + x.shape)
        for i, t in enumerate(self._shape, start=-target_ndim):
            if x.shape[i] == t:
                continue
            elif x.shape[i] == 1:
                sz = list(x.shape)
                sz[i] = t
                x = x.expand(*sz)
            else:
                raise ValueError(f"[{name}] cannot reshape {x.shape} -> {self._shape}")
        return x

    # ---------------- Modulation ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        base = self._tensor_shape(self.logits, "base_logits")
        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        context_key = self._context_hash(context)
        key = f"transition-{base.mean().item():.6g}-{base.std().item():.6g}-{context_key}-T{tau_val:.6g}"

        cached = self._cache_get(key)
        if cached is not None:
            return cached.clone().detach() if grad_safe else cached

        # ---------------- Apply context gates ----------------
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B * T, H)

            mod_logits = base.unsqueeze(0).unsqueeze(0).expand(B, T, *self._shape)
            delta_total = torch.zeros_like(mod_logits)

            if getattr(self, "context_gate", None):
                delta_total += self.context_gate(ctx_flat).view(B, T, *self._shape)
            if getattr(self, "residual_gate", None):
                delta_total += self.residual_gate(ctx_flat).view(B, T, *self._shape)
            if getattr(self, "_U", None) and getattr(self, "_V", None):
                r = self.rank
                U = self._U(ctx_flat).view(B * T, self.n_states, r)
                V = self._V(ctx_flat).view(B * T, self.n_states, r)
                delta_total += torch.einsum("bik,bjk->bij", U, V).view(B, T, *self._shape)

            base = mod_logits + delta_total

        # Temperature scaling
        mod = base / tau_val
        mod = self._tensor_shape(mod, name="transition")
        self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

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
    sample = lambda self, *a, **k: self._get_dist(*a, **k).sample(**k)
    rsample = lambda self, *a, **k: self._get_dist(*a, **k).rsample(**k)
    expected_probs = lambda self, *a, **k: self._get_dist(*a, **k).probs
    mode = lambda self, *a, return_dist=False, **k: self._get_dist(*a, **k) if return_dist else self._get_dist(*a, **k).probs.argmax(-1)

    # ---------------- Robust vectorized update ----------------
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
            mod_logits = self._modulate(context=context, temperature=temperature)
            while posterior.ndim < mod_logits.ndim:
                posterior = posterior.unsqueeze(-1)
            posterior = posterior.expand_as(mod_logits)

            # Vectorized EMA update
            log_probs = F.log_softmax(mod_logits, dim=-1)
            expected_logits = torch.log((posterior * log_probs.exp()).sum(dim=0).clamp_min(EPS))
            self.logits.data.mul_(1 - update_rate).add_(update_rate * expected_logits)
            self._invalidate_cache()
            return


class Emission(DistributionBase):

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
        """
        Align any tensor to trailing shape [K, F] if base, or [B, T, K, F] if batched.
        Expands singleton dims safely and raises an error for incompatible sizes.
        """
        if x is None:
            return None

        x_shape = x.shape
        target_shape = tuple(self._shape)  # (K, F)
        n_trailing = len(target_shape)

        if x.ndim < n_trailing:
            raise ValueError(f"[{name}] Tensor has too few dims {x_shape}, expected at least {n_trailing}")

        # Expand or squeeze trailing dims
        trailing_dims = x_shape[-n_trailing:]
        expand_sizes = []
        for actual, target in zip(trailing_dims, target_shape):
            if actual == target:
                expand_sizes.append(target)
            elif actual == 1:
                expand_sizes.append(target)
            else:
                raise ValueError(f"[{name}] Cannot reshape trailing dims {trailing_dims} -> {target_shape}")

        # Expand singleton trailing dims
        expand_shape = x_shape[:-n_trailing] + tuple(expand_sizes)
        x = x.expand(*expand_shape)

        return x

    def _expand_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert any input [F], [B, F], [B, T, F], or [B, K, F] to [B, T, K, F].
        Safe for log_prob computation.
        """
        K, F = self.n_states, self.n_features

        if x.ndim == 1:  # [F]
            x_exp = x.view(1, 1, 1, F).expand(1, 1, K, F)
        elif x.ndim == 2:  # [B, F]
            B = x.shape[0]
            x_exp = x.view(B, 1, 1, F).expand(B, 1, K, F)
        elif x.ndim == 3:
            B, M, F_in = x.shape
            if M == K:  # [B, K, F]
                x_exp = x.unsqueeze(1)  # [B, 1, K, F]
            else:  # [B, T, F]
                x_exp = x.unsqueeze(2).expand(B, M, K, F_in)
        elif x.ndim == 4:  # Already [B, T, K, F]
            x_exp = x
        else:
            raise ValueError(f"_expand_input: Cannot handle tensor with shape {x.shape}")

        # Validate feature dimension
        if x_exp.shape[-1] != F:
            raise ValueError(f"_expand_input: Feature dimension mismatch {x_exp.shape[-1]} != {F}")

        return x_exp

    def _modulate(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        """
        Modulate base emission parameters with context and temperature.
        Ensures resulting tensor aligns to [K, F] or [B, T, K, F] as appropriate.
        """

        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        context_key = self._context_hash(context)
        key = f"emission-{base.mean().item():.6g}-{base.std().item():.6g}-{context_key}-T{tau_val:.6g}"

        # Check cache
        cached = self._cache_get(key)
        if cached is not None:
            return cached.clone().detach() if grad_safe else cached

        # Apply context & temperature
        mod = super()._apply_context(base, context=context, grad_scale=None)
        mod = super()._apply_temperature(mod, temperature=tau_val)

        # Align trailing shape safely
        mod = self._tensor_shape(mod, name="modulated")

        # Cache and return
        self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

    def _dist_params(
        self,
        tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return emission distribution parameters safely aligned with [K, F] or [B, T, K, F].
        Applies context modulation and temperature scaling.
        """

        mod = self._modulate(tensor, context=context, temperature=temperature)

        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)
            return {"means": mod, "cov": cov}

        elif self.emission_type in {"laplace", "studentt"}:
            scale = self.scale_param.clamp_min(self.min_covar)
            scale_mod = self._tensor_shape(scale, "scale")  # [K, F] or broadcasted
            return {"loc": mod, "scale": scale_mod}

        elif self.emission_type in {"categorical", "bernoulli"}:
            mod = torch.nan_to_num(mod, nan=0.0)
            return {"logits": mod}

        elif self.emission_type == "poisson":
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

    @torch.no_grad()
    def _get_dist(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
        mode: str = "kmeans",
        iters: int = 20,
        max_jitter: int = 5,
    ):
        """
        Compute emission distribution given optional data, posterior, theta, and context.
        Handles continuous and discrete emissions uniformly via _modulate.
        Updates internal buffers consistently.
        """
        K, F = self.n_states, self.n_features
        device = self._emission_means.device
        tau = temperature or getattr(self, "temperature", 1.0)
        
        Xf = X.reshape(-1, F).to(dtype=DTYPE) if X is not None else None
        
        # ---------------- Continuous emissions ----------------
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # Initialize means
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
            means_mod = self._modulate(means, context=context, temperature=tau)
            means_mod = self._tensor_shape(means_mod, name="means_mod")

            # Ensure positive definite covariance for Gaussian
            if self.emission_type == "gaussian":
                I = torch.eye(F, dtype=DTYPE, device=device)
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0:
                            break
                        covs[k] += jitter * I
                        jitter *= 2

            # Update buffers
            self._emission_means.copy_(means_mod)
            self._emission_covs.copy_(covs)
            if self.emission_type == "gaussian":
                self.mu.copy_(means_mod)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1).clamp_min(EPS)))
                return MultivariateNormal(means_mod, covariance_matrix=covs)
            else:
                self.loc.copy_(means_mod)
                scale = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                self.scale_param.copy_(scale)
                dist_cls = Laplace if self.emission_type == "laplace" else StudentT
                return Independent(dist_cls(loc=means_mod, scale=scale), 1)

        # ---------------- Discrete emissions ----------------
        else:
            if Xf is not None and mode == "data":
                if self.emission_type == "categorical":
                    counts = torch.stack([torch.bincount(Xf[:, f].long(), minlength=K) for f in range(F)], dim=1)
                    base = torch.log((counts / counts.sum(dim=0, keepdim=True)).clamp_min(EPS))
                else:  # Bernoulli / Poisson
                    base = torch.log(Xf.mean(0).expand(K, F).clamp_min(EPS))
            else:
                if self.emission_type in {"categorical", "bernoulli"}:
                    base = torch.zeros(K, F, dtype=DTYPE, device=device)
                elif self.emission_type == "poisson":
                    base = torch.zeros(K, F, dtype=DTYPE, device=device)

            # Theta modulation
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                base += theta_scale * theta_vec.unsqueeze(0)

            # Context + temperature modulation
            base_mod = self._modulate(base, context=context, temperature=tau)
            base_mod = self._tensor_shape(base_mod, name="discrete_mod")

            # Update buffers
            self._emission_params.copy_(base_mod)
            if self.emission_type in {"categorical", "bernoulli"}:
                self.logits.copy_(base_mod)
            elif self.emission_type == "poisson":
                self.log_rate.copy_(base_mod)

            # Return distribution
            if self.emission_type == "categorical":
                return Categorical(logits=base_mod)
            elif self.emission_type == "bernoulli":
                return Independent(Bernoulli(logits=base_mod), 1)
            elif self.emission_type == "poisson":
                return Independent(Poisson(rate=base_mod.exp()), 1)

    def forward(self, context=None, temperature=None, return_dist=False):
        dist = self._get_dist(context=context, temperature=temperature)
        if return_dist:
            return dist
        if self.emission_type == "gaussian":
            return dist.mean, dist.covariance_matrix
        if self.emission_type in {"laplace", "studentt"}:
            return dist.loc, dist.scale
        return self._emission_params

    def sample(self, n_samples=1, context=None, temperature=None):
        dist = self._get_dist(context=context, temperature=temperature)
        samples = dist.rsample((n_samples,)) if getattr(dist, "has_rsample", False) else dist.sample((n_samples,))
        if isinstance(dist, Independent):
            samples = samples.view(n_samples, *samples.shape[-2:])
        return samples.to(dtype=DTYPE)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute log-probabilities [B, T, K, F] for HSMM sequences with temperature and context modulation.
        Uses _modulate for consistent context and temperature application.
        Supports continuous (gaussian, laplace, studentt) and discrete (categorical, bernoulli, poisson) emissions.
        """
        K, F = self.n_states, self.n_features

        # Expand input to [B, T, K, F]
        x_exp = self._expand_input(x)

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # Base parameters
            base = self.mu if self.emission_type == "gaussian" else self.loc
            params_mod = self._modulate(base, context=context, temperature=temperature)

            if self.emission_type == "gaussian":
                cov_exp = self._emission_covs.unsqueeze(0)  # [1, K, F, F]
                diff = x_exp - params_mod  # Broadcast [B, T, K, F] - [K, F]
                L = torch.linalg.cholesky(cov_exp)
                sol = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
                return -0.5 * (sol.squeeze(-1) ** 2).sum(-1) - 0.5 * F * math.log(2 * math.pi) - 0.5 * log_det

            else:  # Laplace / StudentT
                scale = self.scale_param.clamp_min(EPS)
                scale_mod = self._modulate(scale, context=context, temperature=temperature)
                scale_mod = self._tensor_shape(scale_mod, name="scale_mod")
                dist_cls = Laplace if self.emission_type == "laplace" else StudentT
                dist = Independent(dist_cls(df=self.dof if self.emission_type=="studentt" else None, loc=params_mod, scale=scale_mod), 1)
                return dist.log_prob(x_exp)

        elif self.emission_type in {"categorical", "bernoulli", "poisson"}:
            # Base logits/rates
            base = None
            if self.emission_type == "categorical":
                base = self.logits
            elif self.emission_type == "bernoulli":
                base = self.logits
            elif self.emission_type == "poisson":
                base = self.log_rate

            base_mod = self._modulate(base, context=context, temperature=temperature)

            if self.emission_type == "categorical":
                log_probs = F.log_softmax(base_mod, dim=-1)
                return torch.gather(
                    log_probs.unsqueeze(0).unsqueeze(0).expand(*x_exp.shape, -1),
                    -1,
                    x_exp.long().unsqueeze(-1)
                ).squeeze(-1)

            elif self.emission_type == "bernoulli":
                return Bernoulli(logits=base_mod).log_prob(x_exp)

            elif self.emission_type == "poisson":
                rate = base_mod.exp().clamp_min(EPS)
                return Poisson(rate).log_prob(x_exp)
        else:
            raise NotImplementedError(f"Emission type {self.emission_type} not implemented in log_prob")

    def parameters_tensor(self) -> torch.Tensor:
        """Return base parameters for HSMM usage (K, F)."""
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        return self._emission_params

    @torch.no_grad()
    def initialize(self, X=None, context=None, theta=None, temperature=None, theta_scale=0.1, mode="kmeans", iters=20):
        return self._get_dist(X=X, theta=theta, context=context, theta_scale=theta_scale, temperature=temperature, mode=mode, iters=iters)

    @torch.no_grad()
    def update(self, X=None, posterior=None, theta=None, context=None, theta_scale=0.1, update_rate=0.5, temperature=None):
        new_dist = self._get_dist(X=X, theta=theta, context=context, theta_scale=theta_scale, temperature=temperature)

        # EMA updates
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            loc = getattr(new_dist, "loc", getattr(new_dist, "mean", None)).to(dtype=DTYPE, device=self._emission_means.device)
            self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)
            if self.emission_type == "gaussian":
                cov = new_dist.covariance_matrix.to(dtype=DTYPE, device=self._emission_means.device)
                self._emission_covs.mul_(1 - update_rate).add_(update_rate * cov)
                self.mu.copy_(self._emission_means)
                diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1).clamp_min(EPS)
                self.log_var.copy_(torch.log(diag))
            else:
                scale = getattr(new_dist, "scale", None)
                diag = ((1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale**2).clamp_min(EPS)
                self._emission_covs.copy_(torch.diag_embed(diag))
                self.loc.copy_(self._emission_means)
                self.scale_param.copy_(torch.sqrt(diag))
        else:
            params = getattr(new_dist, "logits", getattr(new_dist, "rate", None)).to(dtype=DTYPE, device=self._emission_means.device)
            self._emission_params.mul_(1 - update_rate).add_(update_rate * params)
            if self.emission_type in {"categorical", "bernoulli"}:
                self.logits.copy_(self._emission_params)
            elif self.emission_type == "poisson":
                self.log_rate.copy_(self._emission_params)

        return new_dist

