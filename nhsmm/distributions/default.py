# nhsmm/distributions/default.py

from __future__ import annotations
from typing import Optional, Union, Literal, Tuple, Dict, Any
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Bernoulli, Laplace, MultivariateNormal,
    Normal, Independent, Poisson, StudentT
)

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm import constraints


class Categorical(Distribution):
    """
    HSMM-friendly, differentiable, broadcast-safe Categorical.

    Features:
      - Stores probs and log_probs at init to avoid repeated softmax.
      - Integer sampling via torch.multinomial (broadcast-safe).
      - Differentiable rsample via Gumbel-Softmax (soft / hard).
      - log_prob supports arbitrary leading dims and negative `dim`.
      - Fully compatible with batch/time leading dimensions.
    """

    has_rsample = True
    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }

    def __init__(self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        validate_args: bool = False,
        dim: int = -1):
        super().__init__(validate_args=validate_args)

        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs.")
        self.dim = dim

        if logits is not None:
            logits = logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS).to(dtype=DTYPE)
            self._logits = logits
            self._log_probs = F.log_softmax(logits, dim=self.dim)
            self._probs = self._log_probs.exp()
        else:
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            self._probs = probs.to(dtype=DTYPE)
            self._log_probs = probs.clamp_min(EPS).log()
            self._logits = self._log_probs.clone()

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        return self._probs

    @property
    def log_probs(self) -> torch.Tensor:
        return self._log_probs

    @property
    def mean(self) -> torch.Tensor:
        return self._probs

    @property
    def batch_shape(self):
        return self._logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        flattened = self._probs.reshape(-1, self._probs.shape[-1])  # [L, C]
        idx = torch.multinomial(flattened, 1, replacement=True)  # [L,1]
        idx = idx.reshape(*self.batch_shape)
        if len(sample_shape) > 0:
            idx = idx.unsqueeze(0).expand(*sample_shape, *idx.shape)
        return idx.long()

    def rsample(self, sample_shape=torch.Size(), temperature: float = 1.0, hard: bool = False):
        sample_shape = torch.Size(sample_shape)
        temperature = max(float(temperature), EPS)

        logits = self._logits
        n_unsqueeze = len(sample_shape)
        for _ in range(n_unsqueeze):
            logits = logits.unsqueeze(0)

        out_shape = tuple(sample_shape) + tuple(self._logits.shape)
        logits = logits.expand(*out_shape)
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=self.dim)

    def log_prob(self, value):
        value = torch.as_tensor(value, device=self._logits.device)
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            value = value.long()
        lp = self._log_probs
        gather_dim = self.dim if self.dim >= 0 else lp.ndim + self.dim
        expanded_value = value.unsqueeze(gather_dim)
        return lp.gather(gather_dim, expanded_value).squeeze(gather_dim)

    def entropy(self):
        p = self._probs.clamp_min(EPS)
        return -(p * p.log()).sum(dim=self.dim)

    def mode(self):
        return self._logits.argmax(dim=self.dim)


class Neural(nn.Module, ABC):

    _dist_factory: Distribution = None

    def __init__(
        self,
        target_dim: int,
        activation: str = "tanh",
        final_activation: str = "tanh",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        learnable_scale: bool = True,
        cache_enabled: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = False,
        max_delta: float = 0.5,
        cache_limit: int = 32,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.target_dim = int(target_dim)
        self.allow_projection = allow_projection
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_delta = max_delta
        self.temperature = 1.0

        self._process_mode: str = "scalar"
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        self._proj: Optional[nn.Linear] = None
        if allow_projection and context_dim is not None:
            self._proj = nn.Linear(context_dim, context_dim, bias=True, dtype=DTYPE)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)

        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE))

        self._batchnorm: Optional[nn.BatchNorm1d] = None

        self._infer_shape()
        prod_shape = int(math.prod(self._shape))
        if prod_shape != self.target_dim:
            raise ValueError(f"target_dim ({self.target_dim}) != prod(_shape) ({prod_shape})")

        self.context_net: Optional[nn.Sequential] = None
        if context_dim is not None and hidden_dim is not None:
            out_dim = prod_shape
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, out_dim, dtype=DTYPE),
            )
            self._init_weights(self.context_net)

        self.temporal_adapter: Optional[nn.Linear] = None
        self.spatial_adapter: Optional[nn.Linear] = None
        self.log_temperature = self._log_temperature(self._shape)

        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int64))
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        logits_init = torch.zeros(*self._shape, dtype=DTYPE)
        self.logits = nn.Parameter(logits_init, requires_grad=True)

        self.register_buffer("_mod_logits_buffer", self.logits.data.clone())
        self.register_buffer("_logits_buffer", self.logits.data.clone())

    def _infer_shape(self) -> None:
        if hasattr(self, "_shape") and self._shape is not None:
            if len(self._shape) == 0:
                raise ValueError("_shape cannot be empty")
            return
        self._shape = (int(self.target_dim),)

    def _log_temperature(self, shape: Optional[Tuple[int, ...]] = None, init: float = 0.0) -> nn.Parameter:
        shape = shape or tuple(self._shape)
        return nn.Parameter(torch.full(shape, fill_value=float(init), dtype=DTYPE))

    def _reset_buffers(self):
        self._logits_buffer.data.copy_(self.logits.data)
        self._mod_logits_buffer.data.copy_(self.logits.data)

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

    def _apply_temperature(self, logits: torch.Tensor, temperature: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Scale logits by temperature, supporting both scalar and tensor temps."""

        if temperature is None:
            tau = self.log_temperature.exp()
        else:
            tau = torch.tensor(float(temperature), device=logits.device, dtype=DTYPE)
        tau = tau.clamp_min(EPS)

        if isinstance(tau, torch.Tensor):
            if tau.ndim > len(self._shape):
                raise ValueError("temperature tensor ndim incompatible with parameter shape")
            if tau.ndim == len(self._shape):
                tau = tau.view(*([1] * (logits.ndim - tau.ndim)), *tau.shape)
        return logits / tau

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
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
        if context is None:
            self._process_mode = "scalar"
            return None
        if context.ndim == 1:  # (H,)
            self._process_mode = "scalar"
            return context.view(1, 1, -1)
        elif context.ndim == 2:  # (B,H)
            self._process_mode = "batch"
            return context.unsqueeze(1)  # (B,1,H)
        elif context.ndim == 3:  # (B,T,H)
            self._process_mode = "batch"
            return context
        elif context.ndim == 4:  # (S,B,T,H)
            self._process_mode = "sample_batch"
            return context
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False) -> torch.Tensor:
        """
        Compute delta logits to add to base logits from context.
        Supports scalar, batch, and sequence contexts with optional timestep selection.
        """
        base = self._tensor_shape(base)

        # No context or context_net -> zero delta
        if context is None or context.numel() == 0 or self.context_net is None:
            return torch.zeros_like(base, requires_grad=base.requires_grad)

        # Canonicalize context to (S,B,T,H)
        ctx = self._prepare_context(context)
        S, B, T, H = (ctx.shape if ctx.ndim == 4 else (1, *ctx.shape[:2], ctx.shape[-1]))

        # Timestep selection
        if timestep is not None:
            if torch.is_tensor(timestep) and timestep.numel() > 1:
                if timestep.numel() != B:
                    raise IndexError("timestep vector length mismatch")
                idx = torch.arange(B, device=ctx.device)
                ctx = ctx[:, idx, timestep.view(-1), :]
            else:
                t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
                ctx = ctx[:, :, t:t+1, :] if T > 1 else ctx

        # Optional projection
        if self._proj is not None and self.allow_projection:
            if ctx.numel() == 0: return ctx  # skip projection
            ctx = self._proj(ctx.reshape(-1, ctx.shape[-1])).view(*ctx.shape[:-1], -1)
        elif self.context_dim is not None and ctx.shape[-1] != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: got {ctx.shape[-1]}, expected {self.context_dim}. "
                "Set allow_projection=True to apply linear projection."
            )

        # Compute delta via context_net
        delta = self.context_net(ctx.reshape(-1, ctx.shape[-1]))
        delta = delta.view(*ctx.shape[:-1], *self._shape)  # (S,B,T,*shape) or (B,T,*shape)

        # Normalization / adapters
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if self.layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if self.batch_norm:
            nf = delta.shape[-1]
            if getattr(self, "_batchnorm", None) is None or self._batchnorm.num_features != nf:
                self._batchnorm = nn.BatchNorm1d(nf, affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(delta.view(-1, nf)).view(*delta.shape)

        if not skip_adapters:
            if self.temporal_adapter is not None and delta.shape[-2] >= 3:
                delta = self.temporal_adapter(delta.transpose(-2, -1)).transpose(-2, -1)
            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        # Activation, scaling, clamping, optional grad scaling
        delta = self.final_activation_fn(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", float("inf")), getattr(self, "max_delta", float("inf")))
        if grad_scale is not None:
            delta = delta * grad_scale

        # Squeeze timestep dim if single
        if timestep is not None and delta.shape[2] == 1:
            delta = delta.squeeze(2)  # (S,B,*shape) or (B,*shape)

        return delta

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        """
        Return modulated logits: base + delta, after constraints and temperature scaling.
        Supports caching and [S,B,T,...] contexts.
        """
        # Cache key
        context_key = self._context_hash(context)
        timestep_key = f"T{timestep}" if timestep is not None else "TNone"
        temp_key = f"{float(temperature):.6g}" if temperature is not None else "None"
        key = f"{context_key}-{timestep_key}-Temp{temp_key}"

        # Check cache
        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone() if not grad_safe else cached.clone().detach()

        # Base + delta
        delta = self._apply_context(self.logits, context=context, timestep=timestep, grad_scale=None)
        mod = self._validate_logits(self.logits + delta)
        mod = self._apply_temperature(mod, temperature)

        # Ensure trailing dims match logits shape
        mod = self._tensor_shape(mod)

        # Expand leading dims based on mode
        leading_dims_map = {"scalar": 1, "batch": 2, "sequence": 2, "sample_batch": 3}
        leading_dims = leading_dims_map.get(self._process_mode, 1)
        while mod.ndim < leading_dims + len(self._shape):
            mod = mod.unsqueeze(0)

        # Cache
        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    @abstractmethod
    def _init_params(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs) -> Distribution:
        pass

    @abstractmethod
    def _apply_constraints(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def _dist_params(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def _get_dist(self, context=None, temperature=None, timestep=None, **dist_kwargs):
        """Return the distribution object for the given context/temperature/timestep."""
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(**self._dist_params(mod_logits, **dist_kwargs))

    def forward(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
        **dist_kwargs) -> torch.Tensor:
        """
        Forward pass through the distribution.

        Args:
            context: Optional context tensor for modulation.
            temperature: Optional temperature for scaling logits.
            return_dist: If True, return the distribution object instead of logits.
            **dist_kwargs: Extra keyword arguments for the distribution.

        Returns:
            Modulated logits tensor, or distribution object if `return_dist=True`.
        """
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=None)
        dist = self._dist(**self._dist_params(mod_logits, **dist_kwargs))
        if return_dist: return dist
        return mod_logits

    def log_prob(self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **dist_kwargs) -> torch.Tensor:
        """
        Compute log-probabilities for scalar, batch, or sequence inputs.
        Supports extra leading sample dimension [S,B,T,C] while preserving backward compatibility.
        """
        x_tensor = x.long()
        n_states = self._shape[-1]

        # Modulate logits
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)

        # Ensure last dim is states
        if mod.shape[-1] != n_states:
            raise ValueError(f"Last dim of modulated logits ({mod.shape[-1]}) != n_states ({n_states})")

        # Unsqueeze mod to match x leading dims
        while mod.ndim < x_tensor.ndim:
            mod = mod.unsqueeze(0)

        # Broadcast leading singleton dims
        expand_shape = []
        for xd, md in zip(x_tensor.shape, mod.shape):
            expand_shape.append(xd if md == 1 else md)
        expand_shape.append(n_states)  # last dim is states
        mod = mod.expand(*expand_shape)

        # Flatten for gather
        mod_flat = mod.reshape(-1, n_states)
        x_flat = x_tensor.reshape(-1, 1)

        # Compute log-probabilities
        log_probs_flat = F.log_softmax(mod_flat, dim=-1).gather(-1, x_flat).squeeze(-1)

        # Reshape back to original x shape
        return log_probs_flat.view(*x_tensor.shape)

    def log_matrix(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):
        """
        Produce initial-state logits.

        Shape rules:
            logits: [B, S]
            dist: categorical-style distribution over S

        If return_dist=True:
            returns the constructed distribution.
        """
        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)

        # Validate shape early (helps debugging)
        if logits.dim() != 2:
            raise ValueError(f"Initial.log_matrix expected logits shape [B, S], got {tuple(logits.shape)}")

        if return_dist:
            dist = self._dist(**self._dist_params(logits, **dist_kwargs))
            return dist

        return logits

    def sample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):
        """
        Draw a sample from the distribution.

        - Supports scalar, batch, and sequence inputs.
        - Converts integer indices to one-hot vectors with last dim = n_states.
        - Maintains leading dimensions for multiple samples [S,B,T,...].
        """
        n_states = self._shape[-1]
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep, **dist_kwargs)
        s = dist.sample()  # may be integer indices or already one-hot

        # Convert integer indices to one-hot
        if s.dtype in [torch.int64, torch.long] or s.ndim == 0:
            s = F.one_hot(s, num_classes=n_states)

        # Ensure trailing dim = n_states
        if s.shape[-1] != n_states:
            s = s.view(*s.shape[:-1], n_states)

        if return_dist: return dist
        return s

    def rsample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs) -> torch.Tensor:
        """
        Draw a reparameterized sample if supported, otherwise fallback to regular sample.

        Supports:
            - Scalar, batch, and sequence inputs ([B], [B,T], [S,B,T,...])
            - Automatic one-hot conversion if distribution returns integer indices
            - Preserves leading dimensions for multiple samples
        """
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)

        # Draw sample using rsample if supported, else fallback
        s = dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()
        n_states = self._shape[-1]

        # Convert integer indices to one-hot
        if s.dtype in [torch.int64, torch.long] or s.ndim == 0:
            s = F.one_hot(s, num_classes=n_states)

        # Ensure trailing dim matches number of states
        if s.shape[-1] != n_states:
            s = s.view(*s.shape[:-1], n_states)

        if return_dist:
            return dist

        return s

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, return_dist: bool = False, **dist_kwargs):
        """Return expected probabilities or the distribution itself."""
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(mod_logits, **dist_kwargs))
        return dist if return_dist else F.softmax(mod_logits, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, return_dist: bool = False, **dist_kwargs):
        """Return the mode of the distribution."""
        dist = self._get_dist(context, temperature, timestep=timestep, **dist_kwargs)
        if return_dist:
            return dist
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(F.softmax(dist.logits, dim=-1), dim=-1)

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=True)
            bad_vals = logits[bad_idx]
            raise ValueError(f"Non-finite logits at {bad_idx}: {bad_vals}")
        return logits

    @property
    def _dist(self) -> type:
        if self._dist_factory is None:
            raise TypeError(f"{self.__class__.__name__} must define `_dist_factory`.")
        return self._dist_factory

    @_dist.setter
    def _dist(self, value: type) -> None:
        if not isinstance(value, type):
            raise TypeError("_dist must be a distribution *class*, not an instance.")
        self._dist_factory = value

    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        """Generate a stable hash for context and current parameter version."""
        if context is None:
            return f"none-v{int(self._param_version.item())}"

        c = context.detach().float()
        sample = c.flatten()[::max(1, c.numel() // 1024)]  # take at most 1024 elements
        sample_bytes = sample.cpu().numpy().tobytes()
        ctx_hash = hashlib.sha256(sample_bytes).hexdigest()[:12]  # short hash
        return f"{ctx_hash}-v{int(self._param_version.item())}"

    def _cache_set(self, key: str, value: torch.Tensor, grad_safe: bool = True) -> None:
        self._cache[key] = value.detach() if grad_safe else value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        value = self._cache.get(key, None)
        if value is not None:
            self._cache.move_to_end(key)
            return value
        return None

    def _invalidate_cache(self) -> None:
        self._param_version += 1
        self._cache.clear()

    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        optimizer_cls: type = torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):
        """
        Update logits (and optionally log_temperature) using new_logits or posterior.
        Safe for any context shape using _tensor_shape.
        """
        lr = update_rate or 1.0
        optimizer_kwargs = optimizer_kwargs or {}

        # -------------------- Direct new_logits update --------------------
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)

            new_logits = self._tensor_shape(new_logits)
            if new_logits.shape != self.logits.shape:
                lead_dims = new_logits.ndim - len(self._shape)
                if lead_dims > 0:
                    new_logits = new_logits.mean(dim=tuple(range(lead_dims)))

            with torch.no_grad():
                self.logits.data.mul_(1 - lr).add_(lr * new_logits)
            self._reset_buffers()
            self._invalidate_cache()
            return

        # -------------------- Posterior-based update --------------------
        if posterior is None:
            return

        # Enable gradients
        if not self.logits.requires_grad:
            self.logits.requires_grad_(True)
        if update_temperature and not self.log_temperature.requires_grad:
            self.log_temperature.requires_grad_(True)

        # Compute modulated logits
        mod_logits = self._modulate(context=context, temperature=temperature, grad_safe=False)

        # Normalize posterior safely
        psum = posterior.sum(dim=-1, keepdim=True)
        posterior = posterior / psum.clamp_min(EPS)

        # Ensure trailing dims match logits
        posterior = self._tensor_shape(posterior).expand_as(mod_logits)

        # Compute cross-entropy loss
        log_probs = F.log_softmax(mod_logits, dim=-1)
        loss = -(posterior * log_probs).sum() / posterior.sum().clamp_min(EPS)

        # Persistent optimizer
        if not hasattr(self, "_optimizer") or self._optimizer is None:
            params = [self.logits]
            if update_temperature:
                params.append(self.log_temperature)
            self._optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
        else:
            for pg in self._optimizer.param_groups:
                pg['lr'] = lr

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        # Update buffers and cache
        self._reset_buffers()
        self._invalidate_cache()


class Initial(Neural):

    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
    ):
        self._shape = (n_states,)
        self.n_states = n_states
        self.init_mode = init_mode

        super().__init__(
            target_dim=n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
        )

        # Initialize logits
        self._init_params(mode=init_mode)

    def _init_params(self, mode: str, jitter: float = 1e-3) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((self.n_states,), -math.log(self.n_states), dtype=DTYPE)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, self.n_states, dtype=DTYPE)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(self.n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        # Add small jitter to break exact ties
        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._reset_buffers()
        self._invalidate_cache()
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "uniform", context: Optional[torch.Tensor] = None, **dist_kwargs) -> Distribution:
        logits = self._init_params(mode)
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _apply_constraints(self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = torch.clamp(logits, min=-1e6, max=1e6)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits - logits.logsumexp(dim=-1, keepdim=True)

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:
        return super()._apply_context(
            base,
            context=context,
            timestep=timestep,
            grad_scale=grad_scale,
            skip_adapters=True
        )

    def _modulate(self, context=None, temperature=None, timestep=None, grad_safe=False):
        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        logits = super()._modulate(context=context, temperature=temperature, timestep=timestep, grad_safe=grad_safe)
        logits = logits / tau  # scale logits by temperature
        return logits

    def log_matrix(self, context=None, temperature=None, timestep=None, return_dist=False, T=None, **dist_kwargs):
        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        
        # Ensure batch dimension
        if logits.ndim == 1:          # [K]
            logits = logits.unsqueeze(0)  # [1, K]
        
        # Ensure batch + timestep dimension
        if logits.ndim == 2:          # [B, K]
            logits = logits.unsqueeze(1)  # [B, 1, K]

        # Broadcast T if provided
        if T is not None and logits.shape[1] == 1:
            logits = logits.expand(-1, T, -1)  # [B, T, K]

        if return_dist:
            return self._dist(**self._dist_params(logits, **dist_kwargs))
        return logits  # [B, T, K]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(mod, **dist_kwargs))
        if return_dist:
            return dist

        probs = F.softmax(mod, dim=-1)

        # Canonicalize shape for consistency: always [B,T,K]
        # (Initial has no duration dim)
        if probs.ndim == 1:          # (K,)
            probs = probs.unsqueeze(0).unsqueeze(0)
        elif probs.ndim == 2:        # (B,K)
            probs = probs.unsqueeze(1)

        return probs

    def update(self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        optimizer_cls: type = torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):
        """
        Wrapper around Neural.update() ensuring safe tensor shapes for Initial.
        """
        # Ensure trailing dims match _shape before delegating
        if new_logits is not None:
            new_logits = self._tensor_shape(new_logits)
        if posterior is not None:
            posterior = self._tensor_shape(posterior)

        super().update(
            new_logits=new_logits,
            posterior=posterior,
            context=context,
            update_rate=update_rate,
            temperature=temperature,
            from_probs=from_probs,
            grad_safe=grad_safe,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            update_temperature=update_temperature
        )


class Duration(Neural):

    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 40,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
    ):
        self.n_states = int(n_states)
        self.max_duration = int(max_duration)
        self._shape = (self.n_states, self.max_duration)
        self.temperature = max(float(temperature), EPS)

        super().__init__(
            target_dim=self.n_states * self.max_duration,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            temporal_adapter=False,
            spatial_adapter=False,
            allow_projection=True,
            activation="tanh",
            final_activation="tanh",
        )

        self.log_duration = nn.Parameter(torch.zeros(self._shape, dtype=DTYPE))
        self._init_params(mode=init_mode)

    @torch.no_grad()
    def _init_params(self, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.max_duration), dtype=DTYPE)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, self.max_duration, dtype=DTYPE)
            w = w.unsqueeze(0).expand(self.n_states, -1)
            w = (w / w.sum(dim=1, keepdim=True)).clamp_min(EPS)
            logits = w.log()
        elif mode == "normal":
            logits = torch.randn(*self._shape, dtype=DTYPE) * 0.1
            decay = torch.arange(self.max_duration, dtype=DTYPE) * 0.05
            logits = logits - decay
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._reset_buffers()
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "uniform", context: Optional[torch.Tensor] = None, **dist_kwargs) -> Distribution:
        logits = self._init_params(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # All durations valid by default; placeholder for future per-state masking
        return logits

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:
        """
        Returns only the delta to add to the base (which already includes log_duration).
        Avoids double-counting log_duration in _modulate.
        """
        if context is None or self.context_net is None:
            return torch.zeros_like(base, requires_grad=base.requires_grad)
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale, skip_adapters=True)

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:

        # Base tensor includes log_duration exactly once
        base = self._tensor_shape(self.logits + self.log_duration)
        tau = float(self.temperature if temperature is None else max(temperature, EPS))

        # Cache key
        key = f"duration-{self._context_hash(context)}-T{tau:.6g}"
        if timestep is not None:
            key += f"-step{int(timestep)}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        # Compute delta from context
        delta = self._apply_context(base, context=context, timestep=timestep)
        mod = base + delta
        mod = self._apply_constraints(mod)
        mod = self._validate_logits(mod)
        mod = self._apply_temperature(mod, tau)

        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(**self._dist_params(mod, **dist_kwargs))

    def log_matrix(self, context=None, temperature=None, timestep=None, return_dist=False, T=None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        logp = F.log_softmax(mod, dim=-1)  # [B, K, Dmax] or [K, Dmax]

        # Ensure batch dimension
        if logp.ndim == 2:  # [K, Dmax]
            logp = logp.unsqueeze(0)  # [1, K, Dmax]
        
        # Add timestep dimension for forward
        logp = logp.unsqueeze(1) if logp.ndim == 3 else logp  # [B, 1, K, Dmax]

        # Broadcast T if provided
        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, -1, -1)  # [B, T, K, Dmax]

        if return_dist:
            dist = self._dist(**self._dist_params(mod, **dist_kwargs))
            return dist

        return logp  # [B, T, K, Dmax]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):
        """
        Return expected probabilities for durations, with optional distribution object.

        Shapes are canonicalized to [S,B,T,K,D] where possible.
        """
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(mod, **dist_kwargs))
        
        if return_dist:
            return dist

        probs = F.softmax(mod, dim=-1)

        # Squeeze leading singleton dims intelligently
        if self._process_mode == "scalar":
            probs = probs.squeeze(0).squeeze(0)        # -> (K,D)
        elif self._process_mode in ("batch", "sequence"):
            while probs.ndim > 3 and probs.shape[1] == 1:
                probs = probs.squeeze(1)               # remove extra timestep dim
        elif self._process_mode == "sample_batch":
            while probs.ndim > 4 and probs.shape[2] == 1:
                probs = probs.squeeze(2)               # remove timestep dim

        return probs

    def sample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)
        s = dist.sample()  # categorical sample (int indices)

        # convert to one-hot with trailing dim = max_duration
        s_onehot = F.one_hot(s.long(), num_classes=self.max_duration)
        s_onehot = s_onehot.to(dtype=DTYPE)

        if return_dist:
            return dist
        return s_onehot

    def update(self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        optimizer_cls: type = torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):
        """
        Wrapper around Neural.update() ensuring safe tensor shapes for Initial.
        """
        # Ensure trailing dims match _shape before delegating
        if new_logits is not None:
            new_logits = self._tensor_shape(new_logits)
        if posterior is not None:
            posterior = self._tensor_shape(posterior)

        super().update(
            new_logits=new_logits,
            posterior=posterior,
            context=context,
            update_rate=update_rate,
            temperature=temperature,
            from_probs=from_probs,
            grad_safe=grad_safe,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            update_temperature=update_temperature
        )


class Transition(Neural):

    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        n_features: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
    ):
        self.n_states = int(n_states)
        self._shape = (n_states, n_states)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self.temperature = max(float(temperature), EPS)

        super().__init__(
            target_dim=n_states * n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
        )
        self._init_params(mode=init_mode)

    @torch.no_grad()
    def _init_params(self, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.n_states), dtype=DTYPE)
        elif mode == "diag_bias":
            m = torch.full(self._shape, 0.1, dtype=DTYPE)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=-1, keepdim=True)
            logits = m.log()
        elif mode == "normal":
            logits = torch.randn(*self._shape, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._reset_buffers()
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias", context: Optional[torch.Tensor] = None, **dist_kwargs):
        logits = self._init_params(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        n = self.n_states
        out = logits.clone()
        if self.transition_type == "semi":
            mask = torch.eye(n, dtype=torch.bool, device=logits.device)
            mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
            out[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=logits.device), -1)
            mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
            out[..., mask] = -float("inf")
        return out

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:
        # Align with Neural API, skip adapters for Transition
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale, skip_adapters=True)

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        key = f"trans-{self._context_hash(context)}"
        if timestep is not None:
            key += f"-step{int(timestep)}"
        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        key += f"-T{tau:.6g}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        mod = self.logits + self._apply_context(self.logits, context=context, timestep=timestep)
        mod = self._apply_constraints(mod)
        mod = self._apply_temperature(mod, tau)
        mod = self._validate_logits(mod)

        if self.cache_enabled:
            self._cache_set(key, mod.clone().detach(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def log_matrix(self, context=None, temperature=None, timestep=None, return_dist=False, T=None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        logp = F.log_softmax(mod, dim=-1)

        # Ensure batch + timestep dims
        if logp.ndim == 2:       # [K, K]
            logp = logp.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        elif logp.ndim == 3:     # [B, K, K]
            logp = logp.unsqueeze(1)               # [B, 1, K, K]

        # Broadcast T if provided
        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, -1, -1)      # [B, T, K, K]

        if return_dist:
            return self._dist(**self._dist_params(mod, **dist_kwargs))
        return logp  # [B, T, K, K]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        return_dist: bool = False,
        **dist_kwargs):
        """
        Return expected transition probabilities, optionally the distribution object.

        Canonicalizes output to [S,B,T,K,D] or squeezes leading singleton dims intelligently.
        """
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(mod, **dist_kwargs))
        
        if return_dist:
            return dist

        probs = F.softmax(mod, dim=-1)

        # Squeeze leading singleton dims intelligently
        if self._process_mode == "scalar":
            probs = probs.squeeze(0).squeeze(0)        # -> (K,D)
        elif self._process_mode in ("batch", "sequence"):
            while probs.ndim > 3 and probs.shape[1] == 1:
                probs = probs.squeeze(1)               # remove extra timestep dim
        elif self._process_mode == "sample_batch":
            while probs.ndim > 4 and probs.shape[2] == 1:
                probs = probs.squeeze(2)               # remove timestep dim

        return probs

    def sample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **dist_kwargs,) -> torch.Tensor:
        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(logits, **dist_kwargs))
        s = dist.sample()
        return F.one_hot(s.long(), num_classes=self.n_states).to(dtype=DTYPE)

    def update(self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        optimizer_cls: type = torch.optim.SGD,
        optimizer_kwargs: Optional[dict] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):
        if new_logits is not None:
            new_logits = self._tensor_shape(new_logits)
        if posterior is not None:
            posterior = self._tensor_shape(posterior)

        super().update(
            new_logits=new_logits,
            posterior=posterior,
            context=context,
            update_rate=update_rate,
            temperature=temperature,
            from_probs=from_probs,
            grad_safe=grad_safe,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            update_temperature=update_temperature
        )


class Emission(Neural):

    _dist_factory = None

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-6,
        init_mode: str = "spread",
        modulate_var: bool = False,
        emission_type: str = "gaussian",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        allow_projection: bool = True,
        temperature: float = 1.0,
    ):

        self.n_states = n_states
        self.n_features = n_features
        self._shape = (n_states, n_features)
        self.emission_type = emission_type
        self.modulate_var = modulate_var
        self.temperature = temperature
        self.min_covar = min_covar
        self.init_mode = init_mode
        self.dof = 5.0

        if self.emission_type == "gaussian":
            self._dist_factory = MultivariateNormal
        elif self.emission_type == "studentt":
            self._dist_factory = lambda **kwargs: Independent(StudentT(df=self.dof, **kwargs), 1)
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            target_dim=n_states * n_features,
            allow_projection=allow_projection,
        )

        self._init_params(mode=self.emission_type)
        self._invalidate_cache()

    def _init_params(self, mode: Optional[str] = None):
        """
        Legacy initializer retained for API compatibility.
        Delegates to _init_params and ignores mode except for validation.
        """
        if mode is not None and mode != self.emission_type:
            raise ValueError(f"Emission initialized with emission_type={self.emission_type}, got mode={mode}")

        K, F = self.n_states, self.n_features

        self.register_buffer("_emission_covs", torch.zeros(K, F, F, dtype=DTYPE))
        self.register_buffer("_emission_means", torch.zeros(K, F, dtype=DTYPE))

        if self.emission_type == "gaussian":
            self.log_var = nn.Parameter(torch.full((K, F), -1.0, dtype=DTYPE))
            self.mu = nn.Parameter(torch.randn(K, F, dtype=DTYPE) * 0.1)

        else:
            self.scale_param = nn.Parameter(torch.full((K, F), 0.1, dtype=DTYPE))
            self.loc = nn.Parameter(torch.randn(K, F, dtype=DTYPE) * 0.1)

        self._reset_buffers()

    def _reset_buffers(self) -> None:
        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            self._emission_covs.copy_(torch.diag_embed(var))
            self._emission_means.copy_(self.mu)
        else:  # studentt
            # Use softplus for positive scale and squared for covariance
            scale = F.softplus(self.scale_param).clamp_min(self.min_covar)
            self._emission_covs.copy_(torch.diag_embed(scale ** 2))
            self._emission_means.copy_(self.loc)
        super()._reset_buffers()

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wrap super()._tensor_shape to ensure broadcasting for HSMM EM.

        Ensures tensor has shape [1, 1, K, F] if currently [K, F],
        so it can be safely expanded to [B, T, K, F].
        """
        tensor = super()._tensor_shape(tensor)

        if tensor.ndim == 2:  # [K, F] -> [1, 1, K, F]
            K, F = tensor.shape
            tensor = tensor.view(1, 1, K, F)
        elif tensor.ndim == 3:  # [K, F, something] unexpected but handled
            tensor = tensor.unsqueeze(0)  # prepend batch dim

        return tensor

    @torch.no_grad()
    def initialize(self, context: Optional[torch.Tensor] = None, mode: Optional[str] = None):
        K, F = self.n_states, self.n_features
        mode = mode or getattr(self, "init_mode", "spread")

        if mode == "random":
            init_mean = torch.randn(K, F, dtype=DTYPE) * 0.1
        elif mode == "spread":
            init_mean = torch.zeros(K, F, dtype=DTYPE)
            for f in range(F):
                linspace = torch.linspace(-1.0, 1.0, steps=K, dtype=DTYPE)
                init_mean[:, f] = linspace[torch.randperm(K)]
            init_mean += torch.randn_like(init_mean) * 0.05
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if context is not None:
            init_mean = self._modulate(base=init_mean, context=context, grad_safe=True)

        mean_spread = init_mean.std(dim=0, keepdim=True).clamp_min(1e-2)
        init_var = mean_spread ** 2 + torch.rand(K, F, dtype=DTYPE) * 0.01

        if self.emission_type == "gaussian":
            self.mu.copy_(init_mean)
            self.log_var.copy_(torch.log(init_var))
        else:  # student-t
            self.loc.copy_(init_mean)
            self.scale_param.copy_(init_var.sqrt())

        self._reset_buffers(), self._invalidate_cache()
        base = self.mu if self.emission_type == "gaussian" else self.loc
        return self._get_dist(base)

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:

        delta = super()._apply_context(
            base,
            context=context,
            timestep=timestep,
            grad_scale=grad_scale,
            skip_adapters=True
        )
        if delta.ndim < base.ndim:
            delta = delta.unsqueeze(1)  # [B,1,...] -> [B,n_states,...] via broadcasting later
        return delta

    def _modulate(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, grad_safe: bool = False) -> torch.Tensor:

        if context is None or not self.modulate_var:
            return base

        if context.ndim == 3 and context.shape[1] == 1:
            context = context.squeeze(1)

        delta = self._apply_context(base=base, context=context, timestep=timestep)
        while delta.ndim < base.ndim:
            delta = delta.unsqueeze(1)

        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        mod = self._apply_constraints(base + delta)
        mod = self._apply_temperature(mod, tau)

        return mod.detach() if grad_safe else mod

    def _apply_constraints(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """No-op constraint hook kept for API compatibility."""
        return tensor

    def _dist_params(self, loc: torch.Tensor, **dist_kwargs) -> dict:
        """Return distribution parameters compatible with _dist factory."""
        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)
            return {"loc": loc, "covariance_matrix": cov, **dist_kwargs}

        # Student-t
        scale = F.softplus(self.scale_param).clamp_min(self.min_covar)
        return {"loc": loc, "scale": scale, **dist_kwargs}

    def _get_dist(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):
        base = self.mu if self.emission_type == "gaussian" else self.loc
        base = self._modulate(base=base, context=context, timestep=timestep)
        base = self._tensor_shape(base)
        return self._dist(**self._dist_params(base, **dist_kwargs))

    def forward(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False, **dist_kwargs):

        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        if return_dist: return dist

        if self.emission_type == "gaussian":
            return self._tensor_shape(dist.mean), self._tensor_shape(dist.covariance_matrix)

        return self._tensor_shape(dist.loc), self._tensor_shape(dist.scale)

    def log_prob(self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None, **dist_kwargs) -> torch.Tensor:
        K, F = self.n_states, self.n_features

        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)

        if x.ndim == 1:
            x = x.view(1, 1, F)
        elif x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.unsqueeze(2).expand(-1, -1, K, -1)
        logp = dist.log_prob(x)
        if logp.ndim == 4:
            logp = logp.sum(-1)
        return logp

    def expect_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None, **dist_kwargs) -> torch.Tensor:

        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        loc = dist.mean if self.emission_type == "gaussian" else dist.base_dist.loc
        loc = self._tensor_shape(loc)
        if loc.ndim == 4:
            loc = loc.mean(-1)
        return torch.softmax(loc, dim=-1)

    @torch.no_grad()
    def update(self,
        X: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        update_rate: Optional[float] = None):

        if X is None or posterior is None: return

        lr = 1.0 if update_rate is None else update_rate
        K, F = self.n_states, self.n_features

        Xf = X.reshape(-1, F)
        w = posterior.reshape(-1, K).clamp_min(EPS)
        Nk = w.sum(dim=0)

        mask = Nk > self.min_covar
        print("Nk per state:", Nk)
        print("Update mask:", mask)

        if not mask.any(): return
        mean = (w.T @ Xf) / Nk.unsqueeze(-1)
        print("Computed weighted means:\n", mean)

        if self.emission_type == "gaussian":
            print("Before update mu:\n", self.mu)
            self.mu[mask] = self.mu[mask].mul(1 - lr).add_(lr * mean[mask])
            print("After update mu:\n", self.mu)

            Xc = Xf.unsqueeze(1) - mean.unsqueeze(0)
            var = (w.unsqueeze(-1) * Xc.pow(2)).sum(dim=0) / Nk.unsqueeze(-1)
            var = var.clamp_min(self.min_covar)
            print("Computed weighted variances:\n", var)
            self.log_var[mask] = self.log_var[mask].mul(1 - lr).add_(lr * torch.log(var[mask]))
            print("After update log_var:\n", self.log_var)

        else:  # student-t
            print("Before update loc:\n", self.loc)
            self.loc[mask] = self.loc[mask].mul(1 - lr).add_(lr * mean[mask])
            print("After update loc:\n", self.loc)

            Xc = Xf.unsqueeze(1) - mean.unsqueeze(0)
            scale = (w.unsqueeze(-1) * Xc.abs()).sum(dim=0) / Nk.unsqueeze(-1)
            scale = scale.clamp_min(self.min_covar)
            print("Computed weighted scales:\n", scale)
            self.scale_param[mask] = self.scale_param[mask].mul(1 - lr).add_(lr * scale[mask])
            print("After update scale_param:\n", self.scale_param)

        self._reset_buffers()
        self._invalidate_cache()

