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
    Differentiable, broadcast-safe Categorical distribution:
      - Supports integer sampling and differentiable Gumbel-Softmax (soft/hard)
      - Stable logits/probs handling
      - Temperature scaling
      - Fully compatible with batch/time/timestep leading dims
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
            self._logits = logits.to(DTYPE)
            self._probs = F.softmax(logits, dim=dim).to(DTYPE)
        else:
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            self._probs = probs.to(DTYPE)
            self._logits = (probs.clamp_min(EPS).log()).to(DTYPE)

    @property
    def logits(self):
        return self._logits

    @property
    def probs(self):
        return self._probs

    @property
    def mean(self):
        return self._probs

    @property
    def batch_shape(self):
        return self._logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        """
        Integer samples. Fully broadcast-safe for [B, T, ...] contexts.
        """
        sample_shape = torch.Size(sample_shape)
        flat_probs = self._probs.flatten(start_dim=0, end_dim=-2)  # flatten leading dims
        idx = torch.multinomial(flat_probs, 1, replacement=True)  # [B_flat, 1]

        if len(sample_shape) > 0:
            idx = idx.unsqueeze(0).expand(sample_shape + idx.shape)

        final_shape = sample_shape + self.batch_shape
        return idx.reshape(*final_shape)

    def rsample(self, sample_shape=torch.Size(), temperature=1.0, hard=False):
        """
        Differentiable Gumbel-Softmax sample.
        Returns probabilities if soft, one-hot if hard=True.
        Fully broadcast-safe.
        """
        sample_shape = torch.Size(sample_shape)
        temperature = max(float(temperature), EPS)
        shape = sample_shape + self._logits.shape
        logits = self._logits.expand(shape)
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=self.dim)

    def log_prob(self, value):
        """
        Compute log-probabilities for integer inputs.
        Supports arbitrary leading batch/time dims.
        """
        value = torch.as_tensor(value, device=self._logits.device, dtype=torch.long)

        # Broadcast logits to match value shape
        if value.shape[:-1] != self._logits.shape[:-1]:
            logits = self._logits.expand(*value.shape[:-1], self._logits.shape[-1])
        else:
            logits = self._logits

        lp = F.log_softmax(logits, dim=self.dim)
        gather_dim = self.dim if self.dim >= 0 else lp.ndim + self.dim
        return lp.gather(gather_dim, value.unsqueeze(gather_dim)).squeeze(gather_dim)

    def entropy(self):
        """
        Returns entropy with proper EPS clamp for stability.
        """
        p = self._probs.clamp_min(EPS)
        return -(p * p.log()).sum(dim=self.dim)

    def mode(self):
        """
        Returns argmax along category dim, fully broadcast-safe.
        """
        return self._logits.argmax(dim=self.dim)


class Neural(nn.Module, ABC):

    _dist_type: type = None

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
        learnable_scale: bool = False,
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

        self._mode: str = "scalar"
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        self._proj: Optional[nn.Linear] = None
        if allow_projection and context_dim is not None:
            self._proj = nn.Linear(context_dim, context_dim, bias=True, dtype=DTYPE)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)

        self.temporal_adapter = None
        self.spatial_adapter = None

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
        self.logits = nn.Parameter(logits_init, requires_grad=True)

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

    def _infer_shape(self) -> None:
        if hasattr(self, "_shape") and self._shape is not None:
            if len(self._shape) == 0:
                raise ValueError("_shape cannot be empty")
            return
        self._shape = (int(self.target_dim),)

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
        """
        Standardize context to (B, T, H) shape for internal processing and
        update self._mode for downstream logic.

        Modes:
            - "scalar"         : no context or single vector (H,)
            - "sequence"       : (T, H) per-sequence context
            - "batch"          : (B, T, H) batched sequence context
        """
        if context is None:
            self._mode = "scalar"
            return None

        if context.ndim == 1:  # (H,)
            ctx = context.view(1, 1, -1)
            self._mode = "scalar"
        elif context.ndim == 2:  # (T, H)
            ctx = context.unsqueeze(0)  # (1, T, H)
            self._mode = "sequence"
        elif context.ndim == 3:  # (B, T, H)
            ctx = context
            self._mode = "batch"
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

        # Validate context dim if specified
        if self.context_dim is not None and ctx.shape[-1] != self.context_dim:
            raise ValueError(f"context_dim mismatch: got {ctx.shape[-1]}, expected {self.context_dim}")

        return ctx

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False) -> torch.Tensor:
        """
        Return the *delta* to add to base logits.
        - If no context or no context_net: returns zeros_like(base).
        - Supports scalar, sequence, batch context and optional timestep (scalar int or per-batch vector).
        """
        base = self._tensor_shape(base, "base")

        # No context => zero delta (keeps mod = base + delta consistent)
        if context is None or self.context_net is None:
            return torch.zeros_like(base, requires_grad=base.requires_grad)

        # canonicalize context -> (B, T, H)
        ctx = self._prepare_context(context)   # updates self._mode
        B, T, H = ctx.shape

        # timestep selection: accept scalar int/0-dim tensor or per-batch vector
        if timestep is not None:
            if torch.is_tensor(timestep) and timestep.numel() > 1:
                tvec = timestep.view(-1).long()
                if tvec.numel() != B:
                    raise IndexError("timestep vector length mismatch")
                # pick per-batch row -> shape (B, H), then (B,1,H)
                idx = torch.arange(B, device=ctx.device)
                ctx_slice = ctx[idx, tvec, :].unsqueeze(1)  # (B,1,H)
            else:
                t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
                # for sequence/batch with T>1 pick t, else keep whole ctx (scalar-like)
                ctx_slice = ctx[:, t:t+1, :] if (self._mode in ("sequence", "batch") and T > 1) else ctx
        else:
            ctx_slice = ctx  # (B, T, H)

        # optional projection
        if self._proj is not None and self.allow_projection:
            ctx_flat = ctx_slice.reshape(-1, ctx_slice.shape[-1])
            ctx_slice = self._proj(ctx_flat).view(ctx_slice.shape[0], ctx_slice.shape[1], -1)
        elif self.context_dim is not None and ctx_slice.shape[-1] != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: got {ctx_slice.shape[-1]}, expected {self.context_dim}. "
                "Set allow_projection=True to apply linear projection."
            )

        # compute delta via context_net
        delta = self.context_net(ctx_slice.reshape(ctx_slice.shape[0] * ctx_slice.shape[1], -1))
        delta = delta.view(ctx_slice.shape[0], ctx_slice.shape[1], *self._shape)  # (B, T', *shape)

        # optional normalization / adapters / activation / scaling
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

        delta = self.final_activation_fn(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        delta = torch.clamp(delta, -getattr(self, "max_delta", float("inf")), getattr(self, "max_delta", float("inf")))
        if grad_scale is not None:
            delta = delta * grad_scale

        # squeeze timestep dim back if requested (T'=1)
        if timestep is not None and delta.shape[1] == 1:
            delta = delta.squeeze(1)   # becomes (B, *shape)

        # return delta (to be used as base + delta)
        return delta

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        """
        Compute logits modulated by context, adapters, constraints, and temperature,
        optionally at a specific timestep. Supports caching.
        """
        # Standardize context and update mode
        _ = self._prepare_context(context)  # updates self._mode
        ndim = 0 if context is None else context.ndim

        # Build cache key
        context_key = self._context_hash(context)
        timestep_key = f"T{timestep}" if timestep is not None else "TNone"
        temp_key = f"{float(temperature):.6g}" if temperature is not None else "None"
        key = f"{context_key}-{timestep_key}-Temp{temp_key}"

        # Check cache
        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone() if not grad_safe else cached.clone().detach()

        # Base logits
        base = self.logits
        delta = self._apply_context(base, context=context, timestep=timestep, grad_scale=None)
        mod = base + delta

        mod = self._apply_constraints(mod)
        mod = self._apply_temperature(mod, temperature)
        mod = self._validate_logits(mod)

        mod = self._tensor_shape(mod, "modulated_logits")

        # Expand leading dims based on mode
        leading_dims_map = {"scalar": 1, "sequence": 2, "batch": 3}
        leading_dims = leading_dims_map.get(self._mode, 1)
        while mod.ndim < leading_dims + len(self._shape):
            mod = mod.unsqueeze(0)

        # Cache result
        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    @abstractmethod
    def _init_logits(self, *args, **kwargs) -> torch.Tensor:
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

    def forward(
        self,
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
        if return_dist:
            return dist
        return mod_logits

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs) -> torch.Tensor:
        """
        Compute log-probabilities for scalar, batch, or sequence inputs.
        Handles leading singleton dims and expands modulated logits to match x shape.
        """
        x_tensor = x.long()
        n_states = self._shape[-1]

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)

        # Ensure at least 2 dims for gather
        if mod.ndim == 1:
            mod = mod.unsqueeze(0)

        # Squeeze extra leading singleton dims
        while mod.shape[0] == 1 and mod.ndim > x_tensor.ndim:
            mod = mod.squeeze(0)

        # Align batch/sequence dimensions
        if x_tensor.ndim == 2 and mod.ndim == 2 and mod.shape[0] != x_tensor.shape[0]:
            mod = mod.expand(x_tensor.shape[0], n_states)
        elif x_tensor.ndim == 2 and mod.ndim == 3:
            if mod.shape[1] == 1:
                mod = mod.expand(-1, x_tensor.shape[1], -1)

        mod_flat = mod.reshape(-1, n_states)
        x_flat = x_tensor.reshape(-1, 1)
        log_probs_flat = F.log_softmax(mod_flat, dim=-1).gather(-1, x_flat).squeeze(-1)
        return log_probs_flat.view(*x_tensor.shape)

    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, return_dist: bool = False, **dist_kwargs):
        """
        Return full probability matrix (softmax of logits) or distribution object.
        """
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(mod_logits, **dist_kwargs))
        return dist if return_dist else F.softmax(mod_logits, dim=-1)

    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, return_dist: bool = False, **dist_kwargs):
        """
        Draw a sample (one-hot for categorical-like distributions).
        Generic fallback: use _dist.sample().
        """
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)
        s = dist.sample()
        # Convert scalar/int indices to one-hot vectors
        if s.ndim == 0 or s.dtype in [torch.int64, torch.long]:
            s = F.one_hot(s, num_classes=self._shape[-1])
        return s

    def rsample(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs) -> torch.Tensor:
        """Draw a reparameterized sample if supported, otherwise a regular sample."""
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)
        return dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

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
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)
            new_logits = self._tensor_shape(new_logits, "new_logits")
            if new_logits.shape != self.logits.shape:
                lead_dims = new_logits.ndim - len(self._shape)
                if lead_dims > 0:
                    new_logits = new_logits.mean(dim=tuple(range(lead_dims)))
            with torch.no_grad():
                self.logits.data.mul_(1 - lr).add_(lr * new_logits)
            self.reset_buffers()
            self._invalidate_cache()
            return

        if posterior is None:
            return

        # Ensure logits requires gradients
        if not self.logits.requires_grad:
            self.logits.requires_grad_(True)

        # Compute modulated logits with gradients
        mod_logits = self._modulate(context=context, temperature=temperature, grad_safe=False)

        # Normalize posterior
        psum = posterior.sum(dim=-1, keepdim=True)
        posterior = posterior / psum.clamp_min(EPS)
        posterior = self._tensor_shape(posterior, "posterior").expand_as(mod_logits)

        # Compute loss
        log_probs = F.log_softmax(mod_logits, dim=-1)
        loss = -(posterior * log_probs).sum() / posterior.sum().clamp_min(EPS)

        # Backpropagate gradients
        self.zero_grad(set_to_none=True)
        loss.backward()

        # Apply update safely
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p.add_(lr * p.grad)
                    p.grad.zero_()

        self._invalidate_cache()

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(logits).all():
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=True)
            bad_vals = logits[bad_idx]
            raise ValueError(f"Non-finite logits at {bad_idx}: {bad_vals}")
        return logits

    @property
    def _dist(self) -> type:
        if self._dist_type is None:
            raise TypeError(f"{self.__class__.__name__} must define `_dist_type`.")
        return self._dist_type

    @_dist.setter
    def _dist(self, value: type) -> None:
        if not isinstance(value, type):
            raise TypeError("_dist must be a distribution *class*, not an instance.")
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

    def reset_buffers(self):
        self._logits_buffer.copy_(self.logits.data)
        self._mod_logits_buffer.copy_(self.logits.data)

    @abstractmethod
    def initialize(self, mode: str = "uniform") -> Distribution:
        pass


class Initial(Neural):
    _dist_type = Categorical

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
        self._init_logits(mode=init_mode)

    # ---------------- Initialization ----------------
    def _init_logits(self, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((self.n_states,), -math.log(self.n_states), dtype=DTYPE)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, self.n_states, dtype=DTYPE)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(self.n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._logits_buffer.data.copy_(self.logits)
        self._mod_logits_buffer.data.copy_(self.logits)
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "uniform", **dist_kwargs) -> Distribution:
        logits = self._init_logits(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply constraints to logits before producing a distribution.

        Args:
            logits: Raw logits tensor (..., n_states)
            mask: Optional boolean mask where False indicates invalid states

        Returns:
            Constrained logits tensor
        """
        logits = torch.clamp(logits, min=-1e6, max=1e6)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)  # effectively zero probability
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        return logits


class Duration(Neural):
    """
    Semi-Markov per-state duration module.

    - Each state k has a categorical distribution over durations d=1..max_duration.
    - Base logits shape: [K, D]
    - Modulated logits shape: [K, D] (scalar), [T, K, D] (sequence), or [B, T, K, D] (batch).
    """
    _dist_type = Categorical

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
            final_activation="tanh"
        )

        self.log_duration = nn.Parameter(torch.zeros(self._shape, dtype=DTYPE))
        self._init_logits(mode=init_mode)

    @torch.no_grad()
    def _init_logits(self, mode: str) -> torch.Tensor:
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
        self._logits_buffer.copy_(self.logits)
        self._mod_logits_buffer.copy_(self.logits)
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "uniform", **dist_kwargs) -> Distribution:
        logits = self._init_logits(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Currently all durations valid; ready for future masking per-state
        return logits

    def _apply_context(
        self, base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:
        if context is None or self.context_net is None:
            return base
        # Delegate almost entirely to Neural
        base = (base + self.log_duration)
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale)

    def _modulate(
        self, context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        # Delegate to Neural, adding temperature and caching
        base = self._tensor_shape(self.logits + self.log_duration, "duration_base")
        tau = float(self.temperature if temperature is None else max(temperature, EPS))

        key = f"duration-{self._context_hash(context)}-T{tau:.6g}"
        if timestep is not None:
            key += f"-step{int(timestep)}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        mod = self._apply_context(base, context=context, timestep=timestep)
        mod = self._apply_temperature(mod, tau)
        mod = self._apply_constraints(mod)

        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(logits=mod, **dist_kwargs)

    def log_matrix(self, context=None, temperature=None, timestep=None, return_dist=False):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        logp = F.log_softmax(mod, dim=-1)
        if return_dist:
            return self._dist(logits=mod)
        # canonicalize to [B,T,K,D]
        if logp.ndim == 2:
            return logp.unsqueeze(0).unsqueeze(0)
        elif logp.ndim == 3:
            return logp.unsqueeze(0)
        return logp


class Transition(Neural):
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
    ):
        self.n_states = n_states
        self._shape = (n_states, n_states)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self.temperature = max(temperature, EPS)

        super().__init__(
            target_dim=n_states * n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
        )
        self._init_logits(mode=init_mode)

    def _init_logits(self, mode: str) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((self.n_states, self.n_states), -math.log(self.n_states), dtype=DTYPE)
        elif mode == "diag_bias":
            m = torch.full((self.n_states, self.n_states), 0.1, dtype=DTYPE)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=-1, keepdim=True)
            logits = torch.log(m)
        elif mode == "normal":
            logits = torch.randn(self.n_states, self.n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(self.logits)
        self._mod_logits_buffer.copy_(self.logits)
        return logits

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias", **dist_kwargs):
        logits = self._init_logits(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        n = self.n_states
        if self.transition_type == "semi":
            diag_mask = torch.eye(n, dtype=torch.bool, device=logits.device)
            diag_mask = diag_mask.view((1,) * (logits.ndim - 2) + diag_mask.shape)
            out[..., diag_mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            ltr_mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=logits.device), -1)
            ltr_mask = ltr_mask.view((1,) * (logits.ndim - 2) + ltr_mask.shape)
            out[..., ltr_mask] = -float("inf")
        return out

    def _apply_context(self, base: torch.Tensor, context=None, timestep=None, grad_scale=None) -> torch.Tensor:
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale, skip_adapters=True)

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        context_key = self._context_hash(context)
        timestep_key = f"T{timestep}" if timestep is not None else "TNone"
        temp_key = f"{float(temperature):.6g}" if temperature is not None else "None"
        key = f"{context_key}-{timestep_key}-Temp{temp_key}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone() if not grad_safe else cached.clone().detach()

        delta = self._apply_context(self.logits, context=context, timestep=timestep)
        mod = self.logits + delta
        mod = self._apply_constraints(mod)
        tau_val = float(self.temperature if temperature is None else max(temperature, EPS))
        mod = self._apply_temperature(mod, tau_val)
        mod = self._validate_logits(mod)

        # unify shapes: scalar -> [K,K], batch -> [B,K,K]
        if mod.ndim == 2:  # scalar context
            mod_out = mod
        else:
            # merge leading dims except last two as batch
            batch_dims = mod.shape[:-2]
            mod_out = mod.reshape(-1, self.n_states, self.n_states)

        if self.cache_enabled:
            self._cache_set(key, mod.clone().detach(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def sample(self, context=None, temperature=None, timestep=None, **dist_kwargs):
        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        probs = F.softmax(logits, dim=-1)
        dist = self._dist(probs=probs)
        s = dist.sample()
        return F.one_hot(s, num_classes=self.n_states).float()

    def log_matrix(self, context=None, temperature=None):
        logits = self._modulate(context=context, grad_safe=True)
        masked = self._apply_constraints(logits)
        return F.log_softmax(masked, dim=-1)


class Emission(Neural):
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
        allow_projection: bool = True,
        temperature: float = 1.0,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self._shape = (n_states, n_features)
        self.emission_type = emission_type.lower()
        self.min_covar = min_covar
        self.modulate_var = modulate_var
        self.adaptive_scale = adaptive_scale
        self.temperature = temperature
        self.dof = 5.0

        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            target_dim=n_states * n_features,
            allow_projection=allow_projection,
        )

        # self.context_net: Optional[nn.Sequential] = None
        self._init_logits(mode=self.emission_type)

    @torch.no_grad()
    def _init_logits(self, mode: str) -> "Emission":
        K, F = self.n_states, self.n_features

        if mode == "gaussian":
            self.mu = nn.Parameter(torch.randn(K, F, dtype=DTYPE) * 0.1)
            self.log_var = nn.Parameter(torch.full((K, F), -1.0, dtype=DTYPE))
        elif mode in {"categorical", "bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(K, F, dtype=DTYPE))
        elif mode == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(K, F, dtype=DTYPE))
        elif mode in {"laplace", "studentt"}:
            self.loc = nn.Parameter(torch.randn(K, F, dtype=DTYPE) * 0.1)
            self.scale_param = nn.Parameter(torch.full((K, F), 0.1, dtype=DTYPE))
        else:
            raise ValueError(f"Unsupported emission_type: {mode}")

        self.register_buffer("_emission_covs", torch.eye(F, dtype=DTYPE).unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("_emission_means", torch.zeros(K, F, dtype=DTYPE))
        self.register_buffer("_emission_params", torch.zeros(K, F, dtype=DTYPE))
        return self

    @torch.no_grad()
    def initialize(self, X=None, posterior=None, theta=None, context=None, temperature=None, theta_scale=0.1, mode="kmeans", iters=20):
        init_dist = self._get_dist(X=X, posterior=posterior, theta=theta, context=context, theta_scale=theta_scale, temperature=temperature, mode=mode, iters=iters)
        base_dist = getattr(init_dist, "base_dist", init_dist)

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            loc = getattr(base_dist, "loc", getattr(init_dist, "mean", None))
            self._emission_means.copy_(loc)

            if self.emission_type == "gaussian":
                cov = getattr(init_dist, "covariance_matrix", torch.diag_embed(F.softplus(self.log_var)))
                self._emission_covs.copy_(cov)
                self.mu.copy_(loc)
                self.log_var.copy_(torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(EPS)))
            else:
                scale = getattr(base_dist, "scale")
                self._emission_covs.copy_(torch.diag_embed(scale**2))
                self.loc.copy_(loc)
                self.scale_param.copy_(scale)

        else:
            if self.emission_type in {"categorical", "bernoulli"}:
                params = getattr(base_dist, "logits")
                self.logits.copy_(params)
            elif self.emission_type == "poisson":
                params = getattr(base_dist, "rate")
                self.log_rate.copy_(params)
            self._emission_params.copy_(params)

        return init_dist

    def _tensor_shape(self, x: Optional[torch.Tensor], name: str = "tensor") -> Optional[torch.Tensor]:
        """
        Expands a tensor to match the trailing emission shape (n_states, n_features),
        broadcasting leading batch/timestep dimensions if needed.
        """
        if x is None:
            return None

        target = self._shape
        if x.ndim < len(target):
            raise ValueError(f"[{name}] Tensor has too few dims {tuple(x.shape)} (< {len(target)})")

        leading, trailing = x.shape[:-len(target)], x.shape[-len(target):]
        for t_val, tgt_val in zip(trailing, target):
            if t_val != tgt_val and t_val != 1:
                raise ValueError(f"[{name}] cannot align trailing dims {trailing} -> {target}")

        # Expand without copying if possible
        return x.expand(*leading, *target)

    def _apply_constraints(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply emission-specific constraints, e.g., clamp discrete logits.
        """
        if tensor is None:
            return None

        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return tensor

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        """
        Modulate base parameters with context network.

        Args:
            base: Tensor of shape (n_states, n_features)
            context: Optional context tensor, shape (B, T, H) or (B, H)
            timestep: Optional timestep (currently unused, reserved)
            grad_safe: Detach gradients if True

        Returns:
            Tensor of shape (B, T, n_states, n_features) or (n_states, n_features) if no context
        """
        mod = base

        if context is not None and self.context_net is not None:
            ctx = self._prepare_context(context)  # Ensure shape (B, T, H)
            B, T, H = ctx.shape

            # Flatten batch and timestep for network input
            delta_flat = self.context_net(ctx.reshape(B * T, H))  # (B*T, n_states*n_features)
            delta = delta_flat.view(B, T, *self._shape)           # (B, T, n_states, n_features)

            # Broadcast base to match emission dims
            base_exp = base.view((1, 1) + self._shape).expand(B, T, *self._shape)
            mod = base_exp + delta

            if grad_safe:
                mod = mod.detach()

        return mod

    def _modulate(
        self,
        base: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[Union[float, torch.Tensor]] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:
        """
        Modulate emission parameters with optional context and temperature.

        Returns:
            Tensor of shape (B, T, n_states, n_features) or (n_states, n_features) if no context.
        """
        # Default base selection
        if base is None:
            if self.emission_type == "gaussian":
                base = self.mu
            elif self.emission_type in {"laplace", "studentt"}:
                base = self.loc
            else:
                base = self.logits if self.emission_type != "poisson" else self.log_rate

        tau = self.temperature if temperature is None else float(max(temperature, EPS))
        key = f"emission-{self._context_hash(context)}-T{tau:.6g}"
        if timestep is not None:
            key += f"-step{int(timestep)}"

        # Return cached if available
        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        # Apply context modulation
        mod = self._apply_context(base, context=context, timestep=timestep, grad_safe=grad_safe)

        # Apply temperature scaling (only emission dims affected)
        mod = self._apply_temperature(mod, tau)

        # Validate logits for discrete emissions
        mod = self._validate_logits(mod)

        # Apply emission-specific constraints
        mod = self._apply_constraints(mod)

        # Ensure trailing emission dims match
        if mod.shape[-len(self._shape):] != self._shape:
            raise RuntimeError(f"Emission modulated shape mismatch: {mod.shape[-len(self._shape):]} vs {self._shape}")

        # Cache the result
        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod

    def _dist_params(
        self,
        tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Return distribution parameters after context modulation and temperature scaling.
        Shape: (B, T, n_states, n_features)
        """
        mod = self._modulate(tensor, context=context, temperature=temperature)
        n_features = self.n_features

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
    def _get_dist(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        theta_scale: float = 0.5,
        mode: str = "data",
        max_jitter: int = 5,
        **dist_kwargs):
        K, n_features = self.n_states, self.n_features
        device, dtype = self._emission_means.device, DTYPE
        tau = temperature or self.temperature
        Xf = X.reshape(-1, n_features).to(dtype=dtype, device=device) if X is not None else None

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            base = self.mu if self.emission_type == "gaussian" else self.loc

            if Xf is not None and posterior is not None:
                w = posterior.clamp_min(EPS)
                base = (w.T @ Xf) / (w.sum(dim=0, keepdim=True).T + EPS)

            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                base = base + theta_scale * theta_vec.unsqueeze(0)

            params = self._dist_params(base, context=context, temperature=tau)
            loc = self._tensor_shape(params["loc"], "loc_mod")

            if self.emission_type == "gaussian":
                cov = torch.diag_embed(F.softplus(self.log_var).clamp_min(self.min_covar))
                I = torch.eye(n_features, dtype=dtype, device=device)
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(cov[k])
                        if info == 0:
                            break
                        cov[k] += jitter * I
                        jitter *= 2
                dist = MultivariateNormal(loc, covariance_matrix=cov)
                self._emission_covs.copy_(cov)
                self.log_var.copy_(torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(EPS)))
                self.mu.copy_(loc)

            elif self.emission_type == "laplace":
                scale = self._tensor_shape(params["scale"], "scale")
                dist = Independent(Laplace(loc=loc, scale=scale), 1)
                self.scale_param.copy_(scale)
                self.loc.copy_(loc)
                self._emission_covs.copy_(torch.diag_embed(scale**2))

            else:  # studentt
                scale = self._tensor_shape(params["scale"], "scale")
                dist = Independent(StudentT(loc=loc, scale=scale, df=self.dof), 1)
                self.scale_param.copy_(scale)
                self.loc.copy_(loc)
                self._emission_covs.copy_(torch.diag_embed(scale**2))

            self._emission_means.copy_(loc)
            return dist

        else:  # discrete
            base = self.logits if self.emission_type != "poisson" else self.log_rate

            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                base = base + theta_scale * theta_vec.unsqueeze(0)

            params = self._dist_params(base, context=context, temperature=tau)
            logits_mod = self._tensor_shape(params["logits"], "logits_mod")
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

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
        **dist_kwargs):
        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        if return_dist:
            return dist

        n_features = self.n_features
        if self.emission_type == "gaussian":
            return dist.mean, dist.covariance_matrix
        elif self.emission_type in {"laplace", "studentt"}:
            return dist.loc, dist.scale
        else:
            return self._emission_params

    def sample(
        self,
        n_samples: int = 1,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        **dist_kwargs):
        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return dist.rsample((n_samples,)).to(dtype=DTYPE)

        s = dist.sample((n_samples,))
        if self.emission_type == "categorical":
            s = s.squeeze(-1)
        return s.to(dtype=DTYPE)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        **dist_kwargs) -> torch.Tensor:
        K, n_features = self.n_states, self.n_features

        # Ensure input has batch and time dims
        if x.ndim == 1:
            x = x.view(1, 1, n_features)
        elif x.ndim == 2:
            x = x.view(x.shape[0], 1, n_features)
        B, T, _ = x.shape

        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            base = self.mu if self.emission_type == "gaussian" else self.loc
            params = self._dist_params(base, context=context, temperature=temperature)
            loc = self._tensor_shape(params["loc"], "loc_mod").view(1, 1, K, n_features).expand(B, T, K, n_features)

            if self.emission_type == "gaussian":
                scale = F.softplus(self.log_var).clamp_min(self.min_covar).view(1, 1, K, n_features).expand_as(loc)
                diff = x.unsqueeze(2) - loc
                return -0.5 * (diff**2 / scale).sum(-1) - 0.5 * scale.log().sum(-1) - 0.5 * n_features * math.log(2 * math.pi)

            scale = self._tensor_shape(params["scale"], "scale_mod").view(1, 1, K, n_features).expand_as(loc)
            dist_obj = Independent(
                Laplace(loc=loc, scale=scale), 1
            ) if self.emission_type == "laplace" else \
                       Independent(StudentT(loc=loc, scale=scale, df=self.dof), 1)
            return dist_obj.log_prob(x.unsqueeze(2).expand(B, T, K, n_features))

        # Discrete emissions
        base = self.logits if self.emission_type != "poisson" else self.log_rate
        params = self._dist_params(base, context=context, temperature=temperature)
        logits = self._tensor_shape(params["logits"], "logits_mod").view(1, 1, K, n_features).expand(B, T, K, n_features)

        if self.emission_type == "categorical":
            if n_features != 1:
                raise ValueError(f"Categorical emission expects n_features=1, got {n_features}")
            logits = logits.squeeze(-1)
            x_int = x.long().squeeze(-1)
            return F.log_softmax(logits, dim=-1).gather(-1, x_int.unsqueeze(-1)).squeeze(-1)

        elif self.emission_type == "bernoulli":
            x_bin = (x > 0).float()
            x_exp = x_bin.unsqueeze(2).expand(B, T, K, n_features)
            return Independent(Bernoulli(logits=logits), 1).log_prob(x_exp)

        elif self.emission_type == "poisson":
            x_int = x.clamp_min(0).round().long()
            x_exp = x_int.unsqueeze(2).expand(B, T, K, n_features)
            return Independent(Poisson(rate=logits.exp().clamp_min(EPS)), 1).log_prob(x_exp)

    def update(
        self,
        new_base: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: float = 1.0,
        temperature: Optional[float] = None,
        grad_safe: bool = True):
        """
        Update emission parameters with optional new base values or posterior statistics.

        Args:
            new_base: Optional tensor providing new emission values.
            posterior: Optional posterior responsibilities for update.
            context: Optional context tensor for modulation.
            from_probs: Whether `new_base` is given as probabilities (applies log transform).
            update_rate: Learning rate for incremental update (0..1).
            temperature: Optional temperature for scaling updates.
            grad_safe: If True, detach tensors to prevent gradient flow.
        """
        lr = update_rate or 1.0

        if new_base is not None:
            # Align shape with emission dims
            new_base = self._tensor_shape(new_base, "new_base")

            # Convert from probabilities if requested
            if from_probs:
                new_base = torch.log(new_base.clamp_min(EPS))

            # Apply temperature scaling
            if temperature is not None:
                new_base = new_base / max(temperature, EPS)

            # Detach for grad-safe updates
            if grad_safe:
                new_base = new_base.detach()

            # Update parameters by emission type
            if self.emission_type == "gaussian":
                self.mu.data.mul_(1 - lr).add_(lr * new_base)
                # Recompute covariance with current log_var
                var = F.softplus(self.log_var).clamp_min(self.min_covar)
                self._emission_covs.copy_(torch.diag_embed(var))

            elif self.emission_type in {"laplace", "studentt"}:
                self.loc.data.mul_(1 - lr).add_(lr * new_base)
                var = F.softplus(self.scale_param).clamp_min(self.min_covar)
                self._emission_covs.copy_(torch.diag_embed(var**2))

            else:  # discrete types
                if self.emission_type in {"categorical", "bernoulli"}:
                    self.logits.data.mul_(1 - lr).add_(lr * new_base)
                elif self.emission_type == "poisson":
                    self.log_rate.data.mul_(1 - lr).add_(lr * new_base)

            # Clear cached modulated tensors
            self._invalidate_cache()
            return

        # Fallback: delegate to parent update if no new_base provided
        super().update(
            new_logits=None,
            posterior=posterior,
            context=context,
            from_probs=from_probs,
            update_rate=update_rate,
            temperature=temperature,
            grad_safe=grad_safe,
        )


