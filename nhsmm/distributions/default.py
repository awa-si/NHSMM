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
    Distribution, Bernoulli, Laplace, MultivariateNormal, Normal, Independent, Poisson, StudentT
)

from nhsmm.constants import DEBUG, DTYPE, EPS, logger, MAX_LOGITS, NEG_INF
from nhsmm import constraints


class Categorical(Distribution):
    """
    A numerically stable categorical distribution supporting logits or probabilities,
    with optional differentiable Gumbel-Softmax sampling.
    """

    has_rsample = True
    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        validate_args: bool = False,
        dim: int = -1,
    ):
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
        """
        Draw samples [sample_shape, batch_shape].
        """
        sample_shape = torch.Size(sample_shape)
        batch_flat = self._probs.reshape(-1, self._probs.shape[-1])
        n_samples = int(torch.prod(torch.tensor(sample_shape, dtype=torch.int64)))
        idx = torch.multinomial(batch_flat, n_samples, replacement=True)
        # Reshape to [sample_shape, batch_shape]
        idx = idx.view(*sample_shape, *self.batch_shape)
        return idx.long()

    def rsample(self, sample_shape=torch.Size(), temperature: float = 1.0, hard: bool = False):
        """
        Differentiable Gumbel-Softmax sampling.
        """
        sample_shape = torch.Size(sample_shape)
        temperature = max(float(temperature), EPS)

        logits = self._logits
        expand_shape = tuple(sample_shape) + logits.shape
        logits_exp = logits.expand(*expand_shape)
        return F.gumbel_softmax(logits_exp, tau=temperature, hard=hard, dim=-1)

    def log_prob(self, value: torch.Tensor):
        """
        Compute log-probabilities of indices (integers).
        """
        value = torch.as_tensor(value, device=self._logits.device)
        if not torch.is_integral(value):
            raise ValueError("Categorical indices must be integers.")

        gather_dim = self.dim if self.dim >= 0 else self._log_probs.ndim + self.dim
        value_exp = value.unsqueeze(gather_dim)
        return self._log_probs.gather(gather_dim, value_exp).squeeze(gather_dim)

    def entropy(self):
        """
        Shannon entropy of the categorical distribution.
        """
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
        if temperature is None:
            tau = self.log_temperature.exp()
        else:
            tau = torch.tensor(float(temperature), dtype=DTYPE)
        tau = tau.clamp_min(EPS)

        if isinstance(tau, torch.Tensor):
            if tau.ndim > len(self._shape):
                raise ValueError("temperature tensor ndim incompatible with parameter shape")
            if tau.ndim == len(self._shape):
                tau = tau.view(*([1] * (logits.ndim - tau.ndim)), *tau.shape)
        return logits / tau

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None: return None

        k = len(self._shape)
        if tensor.ndim < k:
            raise ValueError(f"{name} has fewer dims ({tensor.ndim}) than required trailing dims ({k})")

        # Check trailing dimensions
        trailing = tensor.shape[-k:]
        for td, ts in zip(trailing, self._shape):
            if td != ts and td != 1:
                raise ValueError(f"Trailing dim {trailing} incompatible with target {self._shape}")

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

        base = self._tensor_shape(base)
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

        context_key = self._context_hash(context)
        timestep_key = f"T{timestep}" if timestep is not None else "TNone"
        temp_key = f"{float(temperature):.6g}" if temperature is not None else "None"
        key = f"{context_key}-{timestep_key}-Temp{temp_key}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone() if not grad_safe else cached.clone().detach()

        # Base + delta
        delta = self._apply_context(self.logits, context=context, timestep=timestep, grad_scale=None)
        mod = self._validate_logits(self.logits + delta)
        mod = self._apply_temperature(mod, temperature)
        mod = self._tensor_shape(mod)

        # Expand leading dims based on mode
        leading_dims_map = {"scalar": 1, "batch": 2, "sequence": 2, "sample_batch": 3}
        leading_dims = leading_dims_map.get(self._process_mode, 1)
        while mod.ndim < leading_dims + len(self._shape):
            mod = mod.unsqueeze(0)

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

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context=None, temperature=None, timestep=None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(**self._dist_params(mod_logits, **dist_kwargs))

    def forward(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False, **dist_kwargs) -> torch.Tensor:

        mod_logits = self._modulate(context=context, temperature=temperature, timestep=None)
        dist = self._dist(**self._dist_params(mod_logits, **dist_kwargs))
        if return_dist: return dist
        return mod_logits

    def log_prob(self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs) -> torch.Tensor:
        x_tensor = x.long()
        n_states = self._shape[-1]

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        if mod.shape[-1] != n_states:
            raise ValueError(f"Last dim of modulated logits ({mod.shape[-1]}) != n_states ({n_states})")

        while mod.ndim < x_tensor.ndim:
            mod = mod.unsqueeze(0)

        expand_shape = []
        for xd, md in zip(x_tensor.shape, mod.shape):
            expand_shape.append(xd if md == 1 else md)
        expand_shape.append(n_states)  # last dim is states
        mod = mod.expand(*expand_shape)

        mod_flat = mod.reshape(-1, n_states)
        x_flat = x_tensor.reshape(-1, 1)

        log_probs_flat = F.log_softmax(mod_flat, dim=-1).gather(-1, x_flat).squeeze(-1)
        return log_probs_flat.view(*x_tensor.shape)

    def log_matrix(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        if logits.dim() != 2:
            raise ValueError(f"Initial.log_matrix expected logits shape [B, S], got {tuple(logits.shape)}")
        return logits

    def sample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        n_states = self._shape[-1]
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep, **dist_kwargs)
        s = dist.sample()

        if s.dtype in [torch.int64, torch.long] or s.ndim == 0:
            s = F.one_hot(s, num_classes=n_states)
        if s.shape[-1] != n_states:
            s = s.view(*s.shape[:-1], n_states)
        return s

    def rsample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs) -> torch.Tensor:

        n_states = self._shape[-1]
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)
        s = dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

        if s.dtype in [torch.int64, torch.long] or s.ndim == 0:
            s = F.one_hot(s, num_classes=n_states)
        if s.shape[-1] != n_states:
            s = s.view(*s.shape[:-1], n_states)
        return s

    def expected_probs(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs):
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return F.softmax(mod_logits, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs):

        dist = self._get_dist(context, temperature, timestep=timestep, **dist_kwargs)
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
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):

        lr = float(update_rate) if update_rate is not None else 1.0
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = self._apply_temperature(new_logits, temperature)

            new_logits = self._tensor_shape(new_logits)
            if new_logits.ndim > len(self._shape):
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - len(self._shape))))

            self.logits.data.mul_(1.0 - lr).add_(lr * new_logits)
            self._reset_buffers()
            self._invalidate_cache()
            return

        if posterior is None:
            return

        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
        if context is not None and self.context_net is not None:
            posterior = self._tensor_shape(posterior)
            log_post = torch.log(posterior.clamp_min(EPS))
            base = self.logits
            while base.ndim < log_post.ndim:
                base = base.unsqueeze(0)

            delta_target = log_post - base
            ctx = self._prepare_context(context)
            SBT = delta_target.shape[:-len(self._shape)]
            H = ctx.shape[-1]

            ctx_flat = ctx.reshape(-1, H)
            tgt_flat = delta_target.reshape(-1, *self._shape)

            # Forward context_net
            delta_pred = self.context_net(ctx_flat)
            delta_pred = delta_pred.view_as(tgt_flat)

            delta_mean = tgt_flat.mean(dim=0)
            self.logits.data.mul_(1.0 - lr).add_(lr * delta_mean)

            self._reset_buffers()
            self._invalidate_cache()
            return

        posterior = self._tensor_shape(posterior)
        if posterior.ndim > len(self._shape):
            posterior = posterior.mean(dim=tuple(range(posterior.ndim - len(self._shape))))

        new_logits = torch.log(posterior.clamp_min(EPS))
        if temperature is not None:
            new_logits = self._apply_temperature(new_logits, temperature)

        self.logits.data.mul_(1.0 - lr).add_(lr * new_logits)

        if update_temperature:
            entropy = -(posterior * new_logits).sum(dim=-1).mean()
            self.log_temperature.data.copy_(entropy.clamp_min(EPS).log())

        self._reset_buffers()
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
        if self._dist_factory is None:
            raise TypeError(f"{self.__class__.__name__} must define `_dist_factory`.")
        return self._dist_factory

    @_dist.setter
    def _dist(self, value: type) -> None:
        if not isinstance(value, type):
            raise TypeError("_dist must be a distribution *class*, not an instance.")
        self._dist_factory = value

    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
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

    def load_state_dict(self, state_dict, strict: bool = True):
        state = super().load_state_dict(state_dict, strict=strict)
        self._reset_buffers()
        self._invalidate_cache()
        return state

    def _reset_buffers(self):
        self._logits_buffer.data.copy_(self.logits.data)
        self._mod_logits_buffer.data.copy_(self.logits.data)

    @torch.no_grad()
    def reset_parameters(self, mode: Optional[str] = None):
        mode = mode or getattr(self, "init_mode", None)
        if not hasattr(self, "_init_params") or not mode:
            raise RuntimeError(f"{self.__class__.__name__} does not support parameter reset")

        self._init_params(mode)
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

        self._init_params(mode=init_mode)

    def _init_params(self, mode: str, context: Optional[torch.Tensor] = None, jitter: float = 1e-5) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full((self.n_states,), -math.log(self.n_states), dtype=DTYPE)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, self.n_states, dtype=DTYPE)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(self.n_states, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        if context is not None and self.context_net is not None:
            delta = self.context_net(context.mean(dim=0))
            logits = logits + delta.view_as(logits)

        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)

        self._reset_buffers()
        self._invalidate_cache()

    @torch.no_grad()
    def initialize(self, mode: str = "normal", context: Optional[torch.Tensor] = None, **dist_kwargs) -> Distribution:
        self._init_params(mode)
        self._invalidate_cache()
        return self._dist(**self._dist_params(self.logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        logits = super()._modulate(context=context, temperature=temperature, timestep=timestep, grad_safe=grad_safe)
        logits = self._apply_constraints(logits)
        return logits

    def log_matrix(self, context=None, temperature=None, timestep=None, T=None, **dist_kwargs):
        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)

        if logits.ndim == 1:          # [K]
            logits = logits.unsqueeze(0)  # [1, K]
        if logits.ndim == 2:          # [B, K]
            logits = logits.unsqueeze(1)  # [B, 1, K]
        if T is not None and logits.shape[1] == 1:
            logits = logits.expand(-1, T, -1)  # [B, T, K]
        return logits  # [B, T, K]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        probs = F.softmax(mod, dim=-1)
        if probs.ndim == 1:          # (K,)
            probs = probs.unsqueeze(0).unsqueeze(0)
        elif probs.ndim == 2:        # (B,K)
            probs = probs.unsqueeze(1)

        return probs

    @torch.no_grad()
    def update(self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        update_temperature: bool = False,
        from_probs: bool = False,
        grad_safe: bool = False):

        if new_logits is not None:
            new_logits = self._tensor_shape(new_logits)

            if new_logits.ndim > 1:
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - 1)))

            super().update(
                new_logits=new_logits,
                context=None,
                update_rate=update_rate,
                temperature=temperature,
                from_probs=from_probs,
                update_temperature=update_temperature,
                grad_safe=grad_safe
            )
            return

        if posterior is None: return
        posterior = self._tensor_shape(posterior)

        # Accept [B,K], [B,T,K], [S,B,T,K]
        if posterior.ndim >= 3:
            posterior = posterior[..., 0, :]

        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)
        if context is not None and self.context_net is not None:
            super().update(
                posterior=posterior,   # shape [B,K] (or [S,B,K])
                context=context,       # canonical context
                update_rate=update_rate,
                update_temperature=update_temperature,
                grad_safe=grad_safe
            )
            return

        posterior = posterior.mean(dim=tuple(range(posterior.ndim - 1)))
        super().update(
            posterior=posterior,
            context=None,
            update_rate=update_rate,
            update_temperature=update_temperature,
            grad_safe=grad_safe
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
        self.init_mode = init_mode
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
        self._init_params(mode=init_mode)

    @torch.no_grad()
    def _init_params(self, mode: str, context: Optional[torch.Tensor] = None, jitter: float = 1e-5) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.max_duration), dtype=DTYPE)
        elif mode == "biased":
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

        if context is not None and self.context_net is not None:
            delta = self.context_net(context.mean(dim=0))
            logits = logits + delta.view_as(logits)

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)

        self._reset_buffers()
        self._invalidate_cache()

    @torch.no_grad()
    def initialize(self, mode: str = "normal", **dist_kwargs) -> Distribution:
        self._init_params(mode)
        return self._dist(**self._dist_params(self.logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return logits - logits.logsumexp(dim=-1, keepdim=True)

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale, skip_adapters=True)

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:

        base = self._tensor_shape(self.logits)
        tau = float(self.temperature if temperature is None else max(temperature, EPS))

        key = f"duration-{self._context_hash(context)}-T{tau:.6g}"
        if timestep is not None:
            key += f"-step{int(timestep)}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        delta = self._apply_context(base, context=context, timestep=timestep)
        mod = self._apply_constraints(base + delta)
        mod = self._validate_logits(mod)
        mod = self._apply_temperature(mod, tau)

        if self.cache_enabled:
            self._cache_set(key, mod.clone(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def _get_dist(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, timestep: Optional[int] = None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(**self._dist_params(mod, **dist_kwargs))

    def log_matrix(self, context=None, temperature=None, timestep=None, T=None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        logp = F.log_softmax(mod, dim=-1)  # [B, K, Dmax] or [K, Dmax]
        if logp.ndim == 2:  # [K, Dmax]
            logp = logp.unsqueeze(0)  # [1, K, Dmax]
        logp = logp.unsqueeze(1) if logp.ndim == 3 else logp  # [B, 1, K, Dmax]
        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, -1, -1)  # [B, T, K, Dmax]
        return logp  # [B, T, K, Dmax]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        probs = F.softmax(mod, dim=-1)
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
        timestep: Optional[int] = None, **dist_kwargs):
        dist = self._get_dist(context, temperature, timestep, **dist_kwargs)
        s = dist.sample()  # categorical sample (int indices)
        s_onehot = F.one_hot(s.long(), num_classes=self.max_duration)
        s_onehot = s_onehot.to(dtype=DTYPE)
        return s_onehot

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

        lr = float(update_rate) if update_rate is not None else 1.0
        lr = max(lr, 0.0)
        if new_logits is not None:
            logits = self._tensor_shape(new_logits)

            # Reduce non-structural dimensions only
            if logits.ndim > 2:
                logits = logits.mean(dim=tuple(range(logits.ndim - 2)))

            if from_probs:
                logits = torch.log(logits.clamp_min(EPS))

            if temperature is not None:
                logits = self._apply_temperature(logits, temperature)

            self.logits.data.lerp_(logits, lr)
            self._reset_buffers()
            self._invalidate_cache()
            return

        if posterior is None or lr == 0.0: return
        posterior = self._tensor_shape(posterior)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)

        if context is not None and self.context_net is not None:
            super().update(
                posterior=posterior,
                context=context,
                update_rate=lr,
                temperature=temperature,
                grad_safe=grad_safe,
            )
            return

        # Reduce batch / time dimensions, preserve [K, D]
        if posterior.ndim > 2:
            posterior = posterior.mean(dim=tuple(range(posterior.ndim - 2)))

        posterior = posterior.clamp_min(EPS)

        # Optional temperature smoothing of the *posterior itself*
        # (important to prevent early collapse)
        if temperature is not None:
            posterior = posterior.pow(1.0 / max(float(temperature), EPS))
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)

        target_logits = torch.log(posterior)

        # EMA-style M-step
        self.logits.data.lerp_(target_logits, lr)

        self._reset_buffers()
        self._invalidate_cache()


class Transition(Neural):

    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        n_features: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "uniform",
        temperature: float = 1.0,
    ):
        self.init_mode = init_mode
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
    def _init_params(self, mode: str, context: Optional[torch.Tensor] = None, jitter: float = 1e-5) -> torch.Tensor:
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.n_states), dtype=DTYPE)
        elif mode == "biased":
            m = torch.full(self._shape, 0.1, dtype=DTYPE)
            m.fill_diagonal_(0.7)
            m = m.clamp_min(jitter)
            m /= m.sum(dim=-1, keepdim=True)
            logits = m.log()
        elif mode == "normal":
            logits = torch.randn(*self._shape, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        if context is not None and self.context_net is not None:
            delta = self.context_net(context.mean(dim=0))
            logits = logits + delta.view_as(logits)

        logits = self._validate_logits(logits)
        self.logits.data.copy_(logits)

        self._reset_buffers()
        self._invalidate_cache()

    @torch.no_grad()
    def initialize(self, mode: str = "normal", **dist_kwargs):
        self._init_params(mode)
        return self._dist(**self._dist_params(self.logits, **dist_kwargs))

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale, skip_adapters=True)

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:

        key = f"transition-{self._context_hash(context)}"
        if timestep is not None:
            key += f"-step{int(timestep)}"
        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        key += f"-T{tau:.6g}"

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.clone().detach() if grad_safe else cached.clone()

        base = self.logits
        delta = self._apply_context(base, context=context, timestep=timestep)
        mod = self._apply_constraints(base + delta)
        mod = self._apply_temperature(mod, tau)
        mod = self._validate_logits(mod)

        if self.cache_enabled:
            self._cache_set(key, mod.clone().detach(), grad_safe=grad_safe)

        return mod.detach() if grad_safe else mod

    def log_matrix(self, context=None, temperature=None, timestep=None, T=None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        logp = F.log_softmax(mod, dim=-1)
        if logp.ndim == 2:       # [K, K]
            logp = logp.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        elif logp.ndim == 3:     # [B, K, K]
            logp = logp.unsqueeze(1)               # [B, 1, K, K]

        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, -1, -1)      # [B, T, K, K]
        return logp  # [B, T, K, K]

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        probs = F.softmax(mod, dim=-1)
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
        timestep: Optional[int] = None, **dist_kwargs,) -> torch.Tensor:

        logits = self._modulate(context=context, temperature=temperature, timestep=timestep)
        dist = self._dist(**self._dist_params(logits, **dist_kwargs))
        s = dist.sample()
        return F.one_hot(s.long(), num_classes=self.n_states).to(dtype=DTYPE)

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

        lr = float(update_rate) if update_rate is not None else 1.0
        if new_logits is not None:
            logits = self._tensor_shape(new_logits)

            if logits.ndim > 2:
                logits = logits.mean(dim=tuple(range(logits.ndim - 2)))

            if from_probs:
                logits = torch.log(logits.clamp_min(EPS))
            if temperature is not None:
                logits = self._apply_temperature(logits, temperature)

            self.logits.data.mul_(1.0 - lr).add_(lr * logits)
            self._reset_buffers()
            self._invalidate_cache()
            return

        if posterior is None: return
        posterior = self._tensor_shape(posterior)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS)

        if context is not None and self.context_net is not None:
            super().update(
                posterior=posterior,   # [B,T,K,K] or [B,K,K]
                context=context,
                update_rate=update_rate,
                temperature=temperature,
                grad_safe=grad_safe
            )
            return

        if posterior.ndim > 2:
            posterior = posterior.mean(dim=tuple(range(posterior.ndim - 2)))

        posterior = posterior.clamp_min(EPS)
        target_logits = torch.log(posterior)

        self.logits.data.mul_(1.0 - lr).add_(lr * target_logits)
        self._reset_buffers()
        self._invalidate_cache()


class Emission(Neural):

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
        temperature: float = 1.0
    ):
        self._shape = (n_states, n_features)

        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            target_dim=n_states * n_features,
            allow_projection=allow_projection,
        )

        self.n_states = n_states
        self.n_features = n_features
        self.emission_type = emission_type
        self.modulate_var = modulate_var
        self.temperature = temperature
        self.min_covar = min_covar
        self.init_mode = init_mode
        self.dof = 5.0  # for student-t

        # Setup distribution factory
        if emission_type == "gaussian":
            self._dist_factory = MultivariateNormal
        elif emission_type == "studentt":
            self._dist_factory = lambda **kwargs: Independent(StudentT(df=self.dof, **kwargs), 1)
        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")

        # Buffers for fast access
        self.register_buffer("_emission_covs", torch.zeros(n_states, n_features, n_features, dtype=DTYPE))
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE))

        if self.emission_type == "gaussian":
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -1.0, dtype=DTYPE))
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features, dtype=DTYPE) * 0.1)
        else:
            self.scale_param = nn.Parameter(torch.full((self.n_states, self.n_features), 0.1, dtype=DTYPE))
            self.loc = nn.Parameter(torch.randn(self.n_states, self.n_features, dtype=DTYPE) * 0.1)

        self._init_params(self.init_mode)
        self._invalidate_cache()

    @torch.no_grad()
    def _init_params(self, mode: str, context: Optional[str] = None, jitter: float = 1e-5) -> None:
        if mode == "random":
            init_mean = torch.randn(self.n_states, self.n_features, dtype=DTYPE) * 0.1
        elif mode == "spread":
            linspace = torch.linspace(-1.0, 1.0, steps=self.n_states, dtype=DTYPE)
            init_mean = torch.stack([linspace[torch.randperm(self.n_states)] for _ in range(self.n_features)], dim=1)
            init_mean = init_mean + torch.randn_like(init_mean) * 0.05
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if context is not None and self.context_net is not None:
            delta = self.context_net(context.mean(dim=0))
            init_mean = init_mean + delta.view_as(init_mean)

        mean_spread = init_mean.std(dim=0, keepdim=True).clamp_min(self.min_covar)
        init_var = mean_spread ** 2 + torch.rand(self.n_states, self.n_features, dtype=DTYPE) * 0.01

        if self.emission_type == "gaussian":
            self.mu.copy_(init_mean)
            self.log_var.copy_(torch.log(init_var))
        else:  # student-t
            self.loc.copy_(init_mean)
            self.scale_param.copy_(init_var.sqrt())

        self._reset_buffers()
        self._invalidate_cache()

    @torch.no_grad()
    def initialize(self, mode: str = "spread", **dist_kwargs):
        self._init_params(mode)
        base = self.mu if self.emission_type == "gaussian" else self.loc
        return self._dist(**self._dist_params(base, **dist_kwargs))

    def _reset_buffers(self) -> None:
        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            self._emission_covs.copy_(torch.diag_embed(var))
            self._emission_means.copy_(self.mu)
        else:  # student-t
            scale = F.softplus(self.scale_param).clamp_min(self.min_covar)
            self._emission_covs.copy_(torch.diag_embed(scale ** 2))
            self._emission_means.copy_(self.loc)

        super()._reset_buffers()

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures tensor is in canonical shape [B, T, K, F] for emissions.
        - 2D tensors [K, F] -> [1, 1, K, F]
        - 3D tensors [B, K, F] -> [B, 1, K, F]
        """
        tensor = super()._tensor_shape(tensor)
        if tensor.ndim == 2:  # [K, F]
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, K, F]
        elif tensor.ndim == 3:  # [B, K, F]
            tensor = tensor.unsqueeze(1)  # [B, 1, K, F]
        return tensor

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
            pad_dims = base.ndim - delta.ndim
            delta = delta.view(delta.shape[0], *([1] * pad_dims), *delta.shape[1:])

        if grad_scale is not None:
            delta = delta * grad_scale

        return delta

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        grad_safe: bool = False) -> torch.Tensor:

        base = self.mu if self.emission_type == "gaussian" else self.loc

        if context is None or not self.modulate_var:
            return base.detach() if grad_safe else base
        if context.ndim == 3 and context.shape[1] == 1:
            context = context.squeeze(1)

        delta = self._apply_context(base=base, context=context, timestep=timestep)
        if delta.ndim < base.ndim:
            shape_pad = [1] * (base.ndim - delta.ndim)
            delta = delta.view(delta.shape[0], *shape_pad, *delta.shape[1:])

        tau = float(self.temperature if temperature is None else max(temperature, EPS))
        mod = base + delta
        mod = self._apply_constraints(mod)
        mod = self._apply_temperature(mod, tau)
        mod = self._tensor_shape(mod)
        return mod.detach() if grad_safe else mod

    def _apply_constraints(self, tensor: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return tensor

    def _dist_params(self, loc: torch.Tensor, **dist_kwargs) -> dict:
        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)  # [K, F, F]
            return {"loc": loc, "covariance_matrix": cov, **dist_kwargs}
        elif self.emission_type == "studentt":
            scale = F.softplus(self.scale_param).clamp_min(self.min_covar)
            return {"loc": loc, "scale": scale, **dist_kwargs}
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    def _get_dist(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):
        base = self._modulate(context=context, temperature=temperature, timestep=timestep)
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
        if x.ndim == 1: x = x.view(1, 1, F)
        elif x.ndim == 2: x = x.unsqueeze(0)
        x = x.unsqueeze(2).expand(-1, -1, K, -1)
        logp = dist.log_prob(x)
        if logp.ndim == 4: logp = logp.sum(-1)
        return logp

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

