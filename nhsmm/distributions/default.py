# nhsmm/distributions/default.py

from __future__ import annotations
from typing import Optional, Union, Literal, Tuple, Dict, Any
from collections import OrderedDict
from abc import ABC, abstractmethod
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Bernoulli, Laplace, MultivariateNormal,
    Normal, Independent, Poisson, StudentT
)

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm import constraints


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

        self.target_dim = int(target_dim)
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.cache_grad_safe = cache_grad_safe
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_delta = max_delta
        self.temperature = 1.0
        self.debug = debug

        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        self._proj: Optional[nn.Linear] = None
        if allow_projection and context_dim is not None:
            self._proj = nn.Linear(context_dim, context_dim, bias=True, dtype=DTYPE)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)

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

        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE))

        self._batchnorm: Optional[nn.BatchNorm1d] = None

        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int64))
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        self._infer_shape()
        prod_shape = int(math.prod(self._shape))
        if prod_shape != self.target_dim:
            raise ValueError(f"target_dim ({self.target_dim}) != prod(_shape) ({prod_shape})")

        logits_init = torch.zeros(*self._shape, dtype=DTYPE)
        self.logits = nn.Parameter(logits_init)

        self.register_buffer("_mod_logits_buffer", self.logits.data.clone())
        self.register_buffer("_logits_buffer", self.logits.data.clone())

        if context_dim is not None and hidden_dim is not None:
            out_dim = prod_shape
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, out_dim, dtype=DTYPE),
            )
            self._init_weights(self.context_net)
        else:
            self.context_net = None

        self.log_temperature = self._build_log_temperature(self._shape)

    def reset_buffers(self):
        self._logits_buffer.copy_(self.logits.data)
        self._mod_logits_buffer.copy_(self.logits.data)

    def _infer_shape(self) -> None:
        if hasattr(self, "_shape") and self._shape is not None:
            if len(self._shape) == 0:
                raise ValueError("_shape cannot be empty")
            return
        self._shape = (int(self.target_dim),)

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _build_log_temperature(self, shape: Optional[Tuple[int, ...]] = None, init: float = 0.0) -> nn.Parameter:
        shape = shape or tuple(self._shape)
        return nn.Parameter(torch.full(shape, fill_value=float(init), dtype=DTYPE))

    def _get_activation(self, name: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }.get(name.lower(), nn.Identity())

    def _init_weights(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tensor_shape(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """
        Ensure `tensor` has trailing dimensions compatible with self._shape.
        Leading dimensions are preserved; trailing dims are broadcasted if needed.

        Args:
            tensor: input tensor to reshape/expand
            name: for error messages

        Returns:
            tensor expanded to (..., *self._shape)
        """
        if tensor is None:
            return None

        k = len(self._shape)
        if tensor.ndim < k:
            raise ValueError(f"{name} has fewer dims ({tensor.ndim}) than required trailing dims ({k})")

        # Check trailing dimensions
        trailing = tensor.shape[-k:]
        for td, ts in zip(trailing, self._shape):
            if td != ts and td != 1:
                raise ValueError(f"{name} trailing dim {trailing} incompatible with target {self._shape}")

        # Leading dims stay as-is; trailing dims expand to self._shape
        leading = tensor.shape[:-k]
        return tensor.expand(*leading, *self._shape)

    def _prepare_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Standardize context to (B, T, H) shape for internal processing."""
        if context is None:
            return None

        if context.ndim == 1:  # single vector -> (1, 1, H)
            ctx = context.view(1, 1, -1)
        elif context.ndim == 2:  # (T, H) -> (1, T, H)
            ctx = context.unsqueeze(0)
        elif context.ndim == 3:  # already (B, T, H)
            ctx = context
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

        if self.context_dim is not None and ctx.shape[-1] != self.context_dim:
            raise ValueError(f"context_dim mismatch: got {ctx.shape[-1]}, expected {self.context_dim}")

        return ctx

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False) -> torch.Tensor:
        """Compute context-dependent delta for logits."""
        base_struct = self._tensor_shape(base, "base")
        if context is None or self.context_net is None:
            return torch.zeros_like(base_struct)

        ctx = self._prepare_context(context)
        B, T, H = ctx.shape
        flat_out = int(math.prod(self._shape))

        # optional projection
        if self._proj is not None and H != getattr(self, "context_dim", H):
            ctx = self._proj(ctx.reshape(B * T, H)).view(B, T, -1)

        delta_flat = self.context_net(ctx.reshape(B * T, -1))
        if delta_flat.shape[-1] != flat_out:
            raise ValueError(f"context_net output {delta_flat.shape[-1]} != expected {flat_out}")

        delta = delta_flat.view(B, T, *self._shape)

        # Optional normalization
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if self.layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if self.batch_norm:
            nf = delta.shape[-1]
            if getattr(self, "_batchnorm", None) is None or self._batchnorm.num_features != nf:
                self._batchnorm = nn.BatchNorm1d(nf, affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(delta.view(-1, nf)).view(*delta.shape)

        # Optional adapters
        if not skip_adapters:
            if self.temporal_adapter is not None and delta.shape[-2] >= 3:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        delta = self.final_activation_fn(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", float("inf")), getattr(self, "max_delta", float("inf")))
        if grad_scale is not None:
            delta = delta * grad_scale

        return delta

    def _apply_temperature(self, logits: torch.Tensor, temperature: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Scale logits by temperature, supporting both scalar and tensor temps."""
        tau: torch.Tensor
        if temperature is None:
            tau = self.log_temperature.exp()
        else:
            tau = torch.tensor(float(temperature), device=logits.device, dtype=DTYPE)
        tau = tau.clamp_min(EPS)

        # expand tau to match leading dims of logits
        if isinstance(tau, torch.Tensor) and tau.ndim == len(self._shape):
            tau = tau.view(*([1] * (logits.ndim - tau.ndim)), *tau.shape)

        return logits / tau

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[Union[float, torch.Tensor]] = None,
        grad_safe: bool = False) -> torch.Tensor:
        """Return context- and temperature-modulated logits."""

        context_key = self._context_hash(context)
        try:
            temp_hash = f"scalar-{float(temperature):.6g}"
        except Exception:
            temp_hash = "scalar-err"
        key = f"{context_key}-T{temp_hash}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached

        base = self.logits
        delta = self._apply_context(base, context=context)
        delta = self._apply_constraints(delta)

        # broadcasting: base + delta
        if context is None:
            mod = base
        else:
            # align trailing dims
            base_exp = base.view((1,) * (delta.ndim - len(self._shape)) + tuple(self._shape))
            mod = base_exp + delta

        mod = self._apply_temperature(mod, temperature)
        mod = self._validate_logits(mod)

        if self.cache_enabled:
            self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        logits = self._validate_logits(logits)
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature)
        return self.dist_type(**self._dist_params(mod_logits, **dist_kwargs))

    def forward(self, log: bool = False, return_dist: bool = False, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        if return_dist:
            return dist
        return F.log_softmax(mod, dim=-1) if log else F.softmax(mod, dim=-1)

    def log_prob(self, x, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        if hasattr(dist, "log_prob"):
            return dist.log_prob(x)
        return F.log_softmax(mod, dim=-1).gather(-1, x.long())

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        if return_dist:
            return self.dist_type(**self._dist_params(mod, **dist_kwargs))
        return F.log_softmax(mod, dim=-1)

    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        return dist if return_dist else F.softmax(mod, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature)
        dist = self.dist_type(**self._dist_params(mod, **dist_kwargs))
        if return_dist:
            return dist
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        from_probs: bool = False,
        grad_safe: bool = False):
        lr = update_rate or 1.0

        if new_logits is not None:
            # Convert probabilities to logits if needed
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            # Apply temperature scaling if given
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)
            # Match trailing shape to self._shape
            new_logits = self._tensor_shape(new_logits, "new_logits")
            # Reduce any leading dimensions to match the parameter shape
            if new_logits.shape != self.logits.shape:
                # average all leading dims, keep only trailing dims
                lead_dims = new_logits.ndim - len(self._shape)
                if lead_dims > 0:
                    new_logits = new_logits.mean(dim=tuple(range(lead_dims)))
            new_logits = self._validate_logits(new_logits)
            # Weighted update
            self.logits.data.mul_(1 - lr).add_(lr * new_logits)
            self._logits_buffer.data.copy_(self.logits.data)
            self._mod_logits_buffer.data.copy_(self.logits.data)
            self._invalidate_cache()
            return


        if posterior is None:
            return

        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return

        mod_logits = self._modulate(context=context, temperature=temperature)

        # Normalize posterior if needed
        if posterior.ndim >= 1:
            psum = posterior.sum(dim=-1, keepdim=True)
            if (torch.abs(psum - 1.0) > 1e-3).any() and self.debug:
                logger.debug("posterior sums deviate from 1; renormalizing")
            posterior = posterior / psum.clamp_min(EPS)

        # Expand posterior to match mod_logits
        posterior = self._tensor_shape(posterior, "posterior").expand_as(mod_logits)

        # Compute loss
        log_probs = F.log_softmax(mod_logits, dim=-1)
        loss = -(posterior * log_probs).sum() / (posterior.sum().clamp_min(EPS))

        if grad_safe:
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None:
                    p.add_(lr * g)
        else:
            self.zero_grad(set_to_none=True)
            loss.backward()
            for p in params:
                if p.grad is not None:
                    p.add_(lr * p.grad)

        self._invalidate_cache()

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=True)
            bad_vals = logits[bad_idx]
            raise ValueError(f"Non-finite logits at {bad_idx}: {bad_vals}")
        return logits

    @property
    def dist_type(self) -> type:
        if self._dist_type is None:
            raise TypeError(f"{self.__class__.__name__} must define `_dist_type`.")
        return self._dist_type

    @dist_type.setter
    def dist_type(self, value: type) -> None:
        if not isinstance(value, type):
            raise TypeError("dist_type must be a distribution *class*, not an instance.")
        self._dist_type = value

    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        """Generate a stable hash for context and current parameter version."""
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        
        # Flatten a small sample of the context for hashing to save memory
        c = context.detach().float()
        sample = c.flatten()[::max(1, c.numel() // 1024)]  # take at most 1024 elements
        sample_bytes = sample.cpu().numpy().tobytes()
        ctx_hash = hashlib.sha256(sample_bytes).hexdigest()[:12]  # short hash
        return f"{ctx_hash}-v{int(self._param_version.item())}"

    def _cache_set(self, key: str, value: torch.Tensor) -> None:
        """Set cache entry with LRU eviction."""
        self._cache[key] = value.detach()
        self._cache.move_to_end(key)  # mark as recently used
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        value = self._cache.get(key, None)
        if value is not None:
            self._cache.move_to_end(key)
            return value.to(dtype=self.logits.dtype, device=self.logits.device)
        return None

    def _invalidate_cache(self) -> None:
        """Clear cache and bump parameter version."""
        self._param_version += 1
        self._cache.clear()


class Initial(DistributionBase):
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        cache_limit: int = 64,
        debug: bool = False,
    ):
        self._shape = (n_states,)

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

        # Initialize logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits.data.copy_(init_logits)
        self._logits_buffer.data.copy_(init_logits)
        self._mod_logits_buffer.data.copy_(init_logits)

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
        self.n_states = n_states
        self.max_duration = max_duration
        self._shape = (n_states, max_duration)
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

        # Base temperature
        self.temperature = max(temperature, EPS)
        # Optional learnable log-duration offsets
        self.log_duration = nn.Parameter(torch.zeros(self._shape, dtype=DTYPE))

        # Initialize logits
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits.data.copy_(init_logits)
        self._logits_buffer.copy_(self.logits.data)
        self._mod_logits_buffer.copy_(self.logits.data)

        # Context network (optional)
        if context_dim is not None:
            out_dim = n_states * max_duration
            if hidden_dim is not None:
                self.context_net = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim, dtype=DTYPE),
                )
            else:
                self.context_net = nn.Linear(context_dim, out_dim, dtype=DTYPE)
                nn.init.xavier_uniform_(self.context_net.weight)
                nn.init.zeros_(self.context_net.bias)
        else:
            self.context_net = None

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
    def initialize(self, mode: str = "uniform"):
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return self._get_dist()

    # ---------------- Modulation ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        # Mask duration < 1
        mask = torch.arange(self.max_duration, dtype=logits.dtype, device=logits.device) < 1
        if logits.ndim >= 2:
            mask = mask.view(*([1] * (logits.ndim - 2)), 1, self.max_duration)
        return logits.masked_fill(mask, -float("inf"))

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        mod = base
        if context is not None and self.context_net is not None:
            ctx = self._prepare_context(context)
            B, T, H = ctx.shape
            delta = self.context_net(ctx.reshape(B * T, H)).view(B, T, *self._shape)
            base_exp = base.view((1, 1) + self._shape).expand(B, T, *self._shape)
            mod = base_exp + delta

            if l2_normalize:
                mod = F.normalize(mod, dim=-1, eps=EPS)
            if self.layer_norm:
                mod = F.layer_norm(mod, self._shape)
            if self.batch_norm:
                nf = mod.shape[-1]
                if getattr(self, "_batchnorm", None) is None or self._batchnorm.num_features != nf:
                    self._batchnorm = nn.BatchNorm1d(nf, affine=True, track_running_stats=False, eps=EPS)
                mod = self._batchnorm(mod.view(-1, nf)).view(*mod.shape)

            if grad_scale is not None:
                mod = mod * grad_scale
        return base

    def _modulate(self, context=None, temperature=None, grad_safe=False, timestep=None):
        base = self._tensor_shape(self.logits + self.log_duration, "duration")
        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        key = f"duration-{self._context_hash(context)}-T{tau_val:.6g}"
        if timestep is not None:
            key += f"-step{timestep}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached

        # Context, temperature, constraints
        mod = self._apply_context(base, context=context)
        mod = self._apply_temperature(mod, tau_val)
        mod = self._apply_constraints(mod)

        # Select timestep if sequence-level
        if timestep is not None:
            if mod.ndim == 4:
                if timestep >= mod.shape[1]:
                    raise IndexError(f"Timestep {timestep} out of bounds for shape {mod.shape}")
                mod = mod[:, timestep, :, :]
            elif mod.ndim == 2 and timestep != 0:
                raise IndexError(f"Timestep {timestep} invalid for shape {mod.shape}")

        if self.cache_enabled:
            self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, timestep=None):
        mod_logits = self._modulate(context, temperature, timestep=timestep)
        mod_logits = self._tensor_shape(mod_logits, "dist_logits")
        return self._dist_type(logits=mod_logits)

    def sample(self, context=None, temperature=None, timestep=None, **kwargs):
        return self._get_dist(context, temperature, timestep, **kwargs).sample()

    def rsample(self, context=None, temperature=None, timestep=None, **kwargs):
        dist = self._get_dist(context, temperature, timestep, **kwargs)
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

    def mode(self, context=None, temperature=None, timestep=None, return_dist=False, **kwargs):
        dist = self._get_dist(context, temperature, timestep, **kwargs)
        if return_dist:
            return dist
        return dist.probs.argmax(-1)

    def expected_probs(self, context=None, temperature=None, timestep=None, **kwargs):
        dist = self._get_dist(context, temperature, timestep, **kwargs)
        return dist.probs

    def log_matrix(self, context=None, temperature=None, timestep=None):
        mod_logits = self._modulate(context, temperature)
        if timestep is not None:
            if mod_logits.ndim == 4:
                mod_logits = mod_logits[:, timestep, :, :]
            elif mod_logits.ndim == 2 and timestep != 0:
                raise IndexError(f"Timestep {timestep} invalid for shape {mod_logits.shape}")
        return F.log_softmax(mod_logits, dim=-1)

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
        lr = update_rate or 1.0

        if new_logits is not None:
            # Collapse extra leading dims
            while new_logits.ndim > len(self._shape):
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))

            # Handle reversed trailing dims automatically
            trailing = new_logits.shape[-len(self._shape):]
            if trailing != self._shape:
                if trailing == self._shape[::-1] and len(self._shape) == 2:
                    new_logits = new_logits.transpose(-2, -1)
                else:
                    raise ValueError(
                        f"new_logits trailing dims {trailing} incompatible with target {self._shape}"
                    )

            # Align to self._shape
            new_logits = self._tensor_shape(new_logits, "new_logits")

            # From probabilities if needed
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))

            # Apply temperature scaling
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)

            # Detach for grad-safe
            if grad_safe:
                new_logits = new_logits.detach()

            # Update parameters
            self.logits.data.mul_(1 - lr).add_(lr * new_logits)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        # Fallback to DistributionBase update (posterior updates)
        super().update(
            new_logits=None,
            posterior=posterior,
            context=context,
            from_probs=from_probs,
            update_rate=update_rate,
            temperature=temperature,
            grad_safe=grad_safe,
        )


class Transition(DistributionBase):
    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        n_features: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
        cache_limit: int = 32,
        debug: bool = False,
    ):
        self.n_states = n_states
        self._shape = (n_states, n_states)
        self.temperature = max(temperature, EPS)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)

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

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Optional context network
        if context_dim is not None:
            out_dim = n_states * n_states
            if hidden_dim is not None:
                self.context_net = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim, dtype=DTYPE)
                )
            else:
                self.context_net = nn.Linear(context_dim, out_dim, dtype=DTYPE)
                nn.init.xavier_uniform_(self.context_net.weight)
                nn.init.zeros_(self.context_net.bias)
        else:
            self.context_net = None

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

    # ---------------- Context & Constraints ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        mod = base
        if context is not None and self.context_net is not None:
            ctx = self._prepare_context(context)
            B, T, H = ctx.shape
            delta = self.context_net(ctx.reshape(B * T, H)).view(B, T, *self._shape)
            base_exp = base.view(1, 1, *self._shape).expand(B, T, *self._shape)
            mod = base_exp + delta
        return base

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        n = self.n_states
        if self.transition_type == "semi":
            mask = torch.eye(n, dtype=torch.bool, device=logits.device)
            mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
            out[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=logits.device), -1)
            mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
            out[..., mask] = -float("inf")
        return out

    # ---------------- Modulation ----------------
    def _modulate(self, context=None, temperature=None, timestep: Optional[int] = None, grad_safe=False):
        base = self._tensor_shape(self.logits, "transition")
        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        key = f"transition-{base.mean().item():.6g}-{base.std().item():.6g}-{self._context_hash(context)}-T{tau_val:.6g}"
        if timestep is not None:
            key += f"-step{timestep}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached

        mod = self._apply_context(base, context=context)
        mod = self._apply_constraints(mod)
        mod = self._apply_temperature(mod, tau_val)

        if timestep is not None:
            if mod.ndim == 4:  # [B, T, K, K]
                if timestep >= mod.shape[1]:
                    raise IndexError(f"Timestep {timestep} out of bounds for shape {mod.shape}")
                mod = mod[:, timestep, :, :]
            elif mod.ndim == 3:  # [T, K, K]
                mod = mod[timestep, :, :]
            elif mod.ndim == 2 and timestep != 0:
                raise IndexError(f"Timestep {timestep} invalid for shape {mod.shape}")

        if self.cache_enabled:
            self._cache_set(key, mod.clone())

        return mod.detach() if grad_safe else mod

    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, timestep=None) -> Categorical:
        mod_logits = self._modulate(context, temperature, timestep)
        mod_logits = self._tensor_shape(mod_logits, "dist_logits")
        return self._dist_type(logits=mod_logits)

    def sample(self, context=None, temperature=None, timestep=None, **kwargs):
        return self._get_dist(context, temperature, timestep).sample(**kwargs)

    def rsample(self, context=None, temperature=None, timestep=None, **kwargs):
        dist = self._get_dist(context, temperature, timestep)
        return dist.rsample(**kwargs) if getattr(dist, "has_rsample", False) else dist.sample(**kwargs)

    def expected_probs(self, context=None, temperature=None, timestep=None):
        return self._get_dist(context, temperature, timestep).probs

    def mode(self, context=None, temperature=None, timestep=None, return_dist=False):
        dist = self._get_dist(context, temperature, timestep)
        return dist if return_dist else dist.probs.argmax(-1)

    def log_matrix(self, context=None, temperature=None, timestep=None):
        mod_logits = self._modulate(context, temperature, timestep)
        return F.log_softmax(mod_logits, dim=-1)

    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: float = 1.0,
        temperature: Optional[float] = None,
        grad_safe: bool = True):
        lr = update_rate or 1.0

        if new_logits is not None:
            # Collapse extra leading dims
            while new_logits.ndim > len(self._shape):
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))

            # Handle reversed trailing dims automatically
            trailing = new_logits.shape[-len(self._shape):]
            if trailing != self._shape:
                if trailing == self._shape[::-1] and len(self._shape) == 2:
                    new_logits = new_logits.transpose(-2, -1)
                else:
                    raise ValueError(
                        f"new_logits trailing dims {trailing} incompatible with target {self._shape}"
                    )

            # Align to self._shape
            new_logits = self._tensor_shape(new_logits, "new_logits")

            # From probabilities if needed
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))

            # Apply temperature scaling
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)

            # Detach for grad-safe
            if grad_safe:
                new_logits = new_logits.detach()

            # Update parameters
            self.logits.data.mul_(1 - lr).add_(lr * new_logits)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        # Fallback to DistributionBase update (posterior updates)
        super().update(
            new_logits=None,
            posterior=posterior,
            context=context,
            from_probs=from_probs,
            update_rate=update_rate,
            temperature=temperature,
            grad_safe=grad_safe,
        )


class Emission(DistributionBase):

    _dist_type = MultivariateNormal

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
        self._shape = (n_states, n_features)

        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            spatial_adapter=spatial_adapter,
            target_dim=n_states * n_features,
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
        self.register_buffer("_emission_covs", torch.eye(n_features, dtype=DTYPE).unsqueeze(0).repeat(n_states, 1, 1))
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE))
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

        # Context network placeholder
        self.context_net: Optional[nn.Sequential] = None

    def _tensor_shape(self, x: Optional[torch.Tensor], name: str = "tensor") -> Optional[torch.Tensor]:
        if x is None: return None
        target = tuple(self._shape)
        n_trail = len(target)
        if x.ndim < n_trail:
            raise ValueError(f"[{name}] Tensor has too few dims {tuple(x.shape)} (< {n_trail})")
        leading, trailing = x.shape[:-n_trail], x.shape[-n_trail:]
        for t_val, tgt_val in zip(trailing, target):
            if t_val != tgt_val and t_val != 1:
                raise ValueError(f"[{name}] cannot align trailing dims {trailing} -> {target}")
        return x.expand(*leading, *target)

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, grad_scale: Optional[float] = None, skip_adapters: bool = False, l2_normalize: bool = False) -> torch.Tensor:
        return base

    def _apply_constraints(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply emission-specific constraints:
        - Ensure positive variance/scale for continuous distributions
        - Clamp discrete logits to avoid NaN / extreme values
        """
        if tensor is None: return None

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # For Gaussian: variance must be >= min_covar
            return tensor  # the tensor itself (mean/loc) doesn't need clipping here
        elif self.emission_type in {"categorical", "bernoulli", "poisson"}:
            # Prevent NaNs or extreme logits
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    def _modulate(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, grad_safe: bool = False) -> torch.Tensor:
        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        key = f"emission-{base.mean().item():.6g}-{base.std().item():.6g}-{self._context_hash(context)}-T{tau:.6g}"
        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached

        base = self._tensor_shape(base, name="transition")
        delta = self._apply_context(base, context=context)

        mod = base + delta if delta is not None else base
        mod = self._apply_temperature(mod, temperature=tau)
        mod = self._apply_constraints(mod)

        mod = self._tensor_shape(mod, name="emission")

        if self.cache_enabled:
            self._cache_set(key, mod.clone())
        return mod.detach() if grad_safe else mod

    def _dist_params(self, tensor: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> Dict[str, torch.Tensor]:

        mod = self._modulate(tensor, context=context, temperature=temperature)
        mod = self._validate_logits(mod)

        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            return {"loc": mod, "cov": torch.diag_embed(var)}
        elif self.emission_type in {"laplace", "studentt"}:
            return {"loc": mod, "scale": self.scale_param.clamp_min(self.min_covar)}
        elif self.emission_type in {"categorical", "bernoulli", "poisson"}:
            return {"logits": torch.nan_to_num(mod, nan=0.0)}
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    @torch.no_grad()
    def _get_dist(self, X: Optional[torch.Tensor] = None,
                  posterior: Optional[torch.Tensor] = None,
                  theta: Optional[torch.Tensor] = None,
                  context: Optional[torch.Tensor] = None,
                  temperature: Optional[float] = None,
                  theta_scale: float = 0.5,
                  mode: str = "data",
                  max_jitter: int = 5,
                  **dist_kwargs):

        K, F = self.n_states, self.n_features
        device, dtype = self._emission_means.device, DTYPE
        tau = temperature or self.temperature
        Xf = X.reshape(-1, F).to(dtype=dtype, device=device) if X is not None else None

        # Continuous distributions
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            base = self.mu if self.emission_type == "gaussian" else self.loc
            if Xf is not None and posterior is not None:
                w = posterior.clamp_min(EPS)
                base = (w.T @ Xf) / (w.sum(dim=0, keepdim=True).T + EPS)
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                base = base + theta_scale * theta_vec.unsqueeze(0)
            params = self._dist_params(base, context=context, temperature=tau)

            loc = self._tensor_shape(params["loc"], name="loc_mod")
            if self.emission_type == "gaussian":
                cov = torch.diag_embed(torch.nn.functional.softplus(self.log_var).clamp_min(self.min_covar))
                I = torch.eye(F, dtype=dtype, device=device)
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(cov[k])
                        if info == 0: break
                        cov[k] += jitter * I
                        jitter *= 2
                dist = MultivariateNormal(loc, covariance_matrix=cov)
                self._emission_covs.copy_(cov)
                self.log_var.copy_(torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(EPS)))
                self.mu.copy_(loc)
            else:
                scale = self._tensor_shape(params["scale"], name="scale")
                dist_cls = Laplace if self.emission_type == "laplace" else StudentT
                dist = Independent(dist_cls(loc=loc, scale=scale, df=self.dof if self.emission_type=="studentt" else None), 1)
                self.scale_param.copy_(scale)
                self.loc.copy_(loc)
                self._emission_covs.copy_(torch.diag_embed(scale**2))
            self._emission_means.copy_(loc)
            return dist

        # Discrete distributions
        else:
            base = self.logits if self.emission_type != "poisson" else self.log_rate
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                base = base + theta_scale * theta_vec.unsqueeze(0)
            params = self._dist_params(base, context=context, temperature=tau)
            logits_mod = self._tensor_shape(params["logits"], name="logits_mod")

            with torch.no_grad():
                self._emission_params.copy_(logits_mod)
                if self.emission_type in {"categorical", "bernoulli"}:
                    self.logits.copy_(logits_mod)
                elif self.emission_type == "poisson":
                    self.log_rate.copy_(logits_mod)

            if self.emission_type == "categorical":
                return Categorical(logits=logits_mod)
            elif self.emission_type == "bernoulli":
                return Independent(Bernoulli(logits=logits_mod), 1)
            elif self.emission_type == "poisson":
                return Independent(Poisson(rate=logits_mod.exp()), 1)

    def forward(self, x: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False):
        dist = self._get_dist(context=context, temperature=temperature)
        if return_dist:
            return dist
        if self.emission_type == "gaussian":
            return dist.mean, dist.covariance_matrix
        elif self.emission_type in {"laplace", "studentt"}:
            return dist.loc, dist.scale
        else:
            return self._emission_params

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        dist = self._get_dist(context=context, temperature=temperature)
        sampler = getattr(dist, "rsample", dist.sample)
        return sampler((n_samples,)).to(dtype=DTYPE)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute log-probabilities for the given inputs, broadcasting properly
        for all emission types: Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson.
        """
        K, F = self.n_states, self.n_features

        # Ensure x has shape [B, T, F]
        if x.ndim == 1:
            x = x.view(1, 1, F)
        elif x.ndim == 2:
            x = x.view(x.shape[0], 1, F)
        B, T, _ = x.shape

        # Get modulated distribution parameters
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            base_tensor = self.mu if self.emission_type == "gaussian" else self.loc
            params = self._dist_params(base_tensor, context=context, temperature=temperature)

            loc = params["loc"].view(1, 1, K, F).expand(B, T, K, F)
            if self.emission_type == "gaussian":
                scale = torch.nn.functional.softplus(self.log_var).clamp_min(self.min_covar).view(1, 1, K, F).expand_as(loc)
                diff = x.unsqueeze(2) - loc  # [B,T,K,F]
                log_prob = -0.5 * (diff ** 2 / scale).sum(-1) \
                           - 0.5 * scale.log().sum(-1) \
                           - 0.5 * F * math.log(2 * math.pi)
                return log_prob

            else:  # Laplace or StudentT
                scale = params["scale"].view(1, 1, K, F).expand(B, T, K, F)
                dist_cls = Laplace if self.emission_type == "laplace" else StudentT
                dist = Independent(dist_cls(loc=loc, scale=scale,
                                            df=self.dof if self.emission_type == "studentt" else None), 1)
                return dist.log_prob(x.unsqueeze(2).expand(B, T, K, F))

        else:  # categorical, bernoulli, poisson
            base_tensor = self.logits if self.emission_type != "poisson" else self.log_rate
            params = self._dist_params(base_tensor, context=context, temperature=temperature)
            logits = params["logits"].view(1, 1, K, F).expand(B, T, K, F)

            if self.emission_type == "categorical":
                x_exp = x.long().unsqueeze(2).expand(B, T, K)
                log_probs = F.log_softmax(logits, dim=2)
                return torch.gather(log_probs, 2, x_exp).squeeze(-1).sum(-1)

            elif self.emission_type == "bernoulli":
                x_exp = x.unsqueeze(2).expand(B, T, K, F)
                dist = Independent(Bernoulli(logits=logits), 1)
                return dist.log_prob(x_exp)

            elif self.emission_type == "poisson":
                x_exp = x.unsqueeze(2).expand(B, T, K, F)
                rate = logits.exp().clamp_min(EPS)
                dist = Independent(Poisson(rate=rate), 1)
                return dist.log_prob(x_exp)

    @torch.no_grad()
    def initialize(self, X=None, posterior=None, theta=None, context=None, temperature=None, theta_scale=0.1, mode="kmeans", iters=20):

        init_dist = self._get_dist(
            X=X,
            posterior=posterior,
            theta=theta,
            context=context,
            theta_scale=theta_scale,
            temperature=temperature,
            mode=mode,
            iters=iters
        )

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            loc = getattr(init_dist, "loc", getattr(init_dist, "mean", None))
            self._emission_means.copy_(loc)

            if self.emission_type == "gaussian":
                cov = init_dist.covariance_matrix
                self._emission_covs.copy_(cov)
                self.mu.copy_(loc)
                diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(EPS)
                self.log_var.copy_(torch.log(diag))
            else:  # Laplace / StudentT
                scale = getattr(init_dist, "scale")
                diag = scale**2
                self._emission_covs.copy_(torch.diag_embed(diag))
                self.loc.copy_(loc)
                self.scale_param.copy_(scale)
        else:
            if self.emission_type in {"categorical", "bernoulli"}:
                params = init_dist.logits
                self.logits.copy_(params)
            elif self.emission_type == "poisson":
                params = init_dist.rate
                self.log_rate.copy_(params)
            self._emission_params.copy_(params)

        return init_dist

    @torch.no_grad()
    def update(self, X=None, posterior=None, theta=None, context=None, theta_scale=0.1, update_rate=0.5, temperature=None):
        new_dist = self._get_dist(X=X, posterior=posterior, theta=theta, context=context, theta_scale=theta_scale, temperature=temperature)

        K, F = self.n_states, self.n_features
        device, dtype = self._emission_means.device, DTYPE

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            loc = getattr(new_dist, "loc", getattr(new_dist, "mean", None)).to(dtype=dtype, device=device)
            self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)

            if self.emission_type == "gaussian":
                cov = new_dist.covariance_matrix.to(dtype=dtype, device=device)
                self._emission_covs.mul_(1 - update_rate).add_(update_rate * cov)

                self.mu.copy_(self._emission_means)
                diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1).clamp_min(EPS)
                self.log_var.copy_(torch.log(diag))
            else:  # Laplace / StudentT
                scale = getattr(new_dist, "scale", None).to(dtype=dtype, device=device)
                diag = ((1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale**2).clamp_min(EPS)
                self._emission_covs.copy_(torch.diag_embed(diag))
                self.loc.copy_(self._emission_means)
                self.scale_param.copy_(torch.sqrt(diag))

        else:
            if self.emission_type in {"categorical", "bernoulli"}:
                params = getattr(new_dist, "logits").to(dtype=dtype, device=device)
            elif self.emission_type == "poisson":
                params = getattr(new_dist, "rate").to(dtype=dtype, device=device)

            self._emission_params.mul_(1 - update_rate).add_(update_rate * params)
            if self.emission_type in {"categorical", "bernoulli"}:
                self.logits.copy_(self._emission_params)
            elif self.emission_type == "poisson":
                self.log_rate.copy_(self._emission_params)

        return new_dist

