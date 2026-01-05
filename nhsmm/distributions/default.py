from __future__ import annotations
from typing import Optional, Union, Literal, Tuple, Dict, Any
from collections import OrderedDict
from abc import ABC, abstractmethod
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.distributions import (
    Distribution, MultivariateNormal, Independent, StudentT
)

from nhsmm.config import EPS, logger, MAX_LOGITS, NEG_INF


class Categorical(Distribution):

    has_rsample = True
    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }

    def __init__(self, logits=None, probs=None, validate_args=False, dim=-1):
        super().__init__(validate_args=validate_args)
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs.")

        self.dim = dim

        if logits is not None:
            logits = logits.clamp(min=-MAX_LOGITS, max=MAX_LOGITS)
            self._logits = logits
            self._log_probs = nnF.log_softmax(logits, dim=self.dim)
            self._probs = self._log_probs.exp()
        else:
            self._probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            self._log_probs = self._probs.clamp_min(EPS).log()
            self._logits = torch.log(self._probs)

    @property
    def logits(self): return self._logits
    @property
    def probs(self): return self._probs
    @property
    def log_probs(self): return self._log_probs
    @property
    def mean(self): return self._probs
    @property
    def batch_shape(self): return self._logits.shape[:-1]
    @property
    def event_shape(self): return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        # Flatten batch for multinomial
        batch_flat = self._probs.reshape(-1, self._probs.shape[-1])
        n_samples = int(torch.prod(torch.tensor(sample_shape, dtype=torch.int64)))
        idx = torch.multinomial(batch_flat, n_samples, replacement=True)
        return idx.view(*sample_shape, *self.batch_shape).long()

    def rsample(self, sample_shape=torch.Size(), temperature: Optional[float] = None, hard: bool = False):
        tau = 1.0 if temperature is None else temperature
        logits_exp = self._logits.unsqueeze(0).expand(*sample_shape, *self.batch_shape, self._logits.shape[-1])
        return nnF.gumbel_softmax(logits_exp, tau=tau, hard=hard, dim=-1)

    def log_prob(self, value: torch.Tensor):
        value = value.long()
        gather_dim = self.dim if self.dim >= 0 else self._log_probs.ndim + self.dim
        return self._log_probs.gather(gather_dim, value.unsqueeze(gather_dim)).squeeze(gather_dim)

    def entropy(self):
        p = self._probs.clamp_min(EPS)
        return -(p * p.log()).sum(dim=self.dim)

    def mode(self):
        return self._logits.argmax(dim=self.dim)


class IndependentStudentT(Distribution):
    has_rsample = True
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
        "df": torch.distributions.constraints.positive
    }

    def __init__(self, loc, scale, df, event_dim=1, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.loc = loc
        self.scale = scale
        self.df = df
        self.event_dim = event_dim  # number of dimensions to treat as event

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(self.loc.shape, self.scale.shape, self.df.shape)[:-self.event_dim]

    @property
    def event_shape(self):
        return torch.Size(self.loc.shape[-self.event_dim:])

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2 * self.df / (self.df - 2)

    @property
    def stddev(self):
        return self.variance.sqrt()

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + torch.broadcast_shapes(self.loc.shape, self.scale.shape, self.df.shape)
        z = torch.randn(shape, device=self.loc.device, dtype=self.loc.dtype)
        v = torch.distributions.Chi2(self.df).rsample(shape)
        y = self.loc + self.scale * z / torch.sqrt(v / self.df)
        return y

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        t = (value - self.loc) / self.scale
        df = self.df
        log_unnormalized = -(df + 1) / 2 * torch.log1p(t ** 2 / df)
        log_normalization = torch.lgamma((df + 1)/2) - torch.lgamma(df/2)
        log_normalization -= 0.5 * torch.log(df * torch.pi) + torch.log(self.scale)
        return log_unnormalized + log_normalization

    def entropy(self):
        df = self.df
        return 0.5 * (df + 1) * (torch.digamma((df + 1)/2) - torch.digamma(df/2)) + \
               torch.lgamma(df/2) - torch.lgamma((df+1)/2) + \
               0.5 * torch.log(df * torch.pi) + torch.log(self.scale)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(IndependentStudentT, _instance)
        new.loc = self.loc.expand(batch_shape + self.event_shape)
        new.scale = self.scale.expand(batch_shape + self.event_shape)
        new.df = self.df.expand(batch_shape + self.event_shape)
        new.event_dim = self.event_dim
        return new


class Neural(nn.Module, ABC):

    _dist_factory: type[Distribution] = None

    def __init__(
        self,
        target_dim: int,
        activation: str = "tanh",
        final_activation: str = "tanh",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        allow_projection: bool = True,
        max_delta: float = 0.5,
        alpha: float = 1.0
    ):
        super().__init__()

        self.alpha = alpha
        self.max_delta = max_delta
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection

        self.activation_fn = self._get_activation(activation)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.final_activation_fn = self._get_activation(final_activation)

        self._proj: Optional[nn.Linear] = None
        if allow_projection and context_dim is not None:
            self._proj = nn.Linear(context_dim, context_dim, bias=True)
            nn.init.xavier_uniform_(self._proj.weight)
            nn.init.zeros_(self._proj.bias)

        prod_shape = hasattr(self, '_shape') and int(math.prod(self._shape))
        if prod_shape != self.target_dim:
            raise ValueError(f"target_dim ({self.target_dim}) != prod(_shape) ({prod_shape})")

        self.context_net: Optional[nn.Sequential] = None
        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)
        if context_dim is not None and hidden_dim is not None:
            out_dim = int(math.prod(self._shape))
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Linear(hidden_dim, out_dim),
            )
            self._init_weights(self.context_net)

        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.logits = nn.Parameter(torch.zeros(*self._shape), requires_grad=True)

    def _get_activation(self, name: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
            "leaky_relu": nn.LeakyReLU(0.01),
        }.get(name.lower(), nn.Identity())

    def _init_weights(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def base(self) -> torch.Tensor:
        return self.logits

    @base.setter
    def base(self, value: torch.Tensor) -> None:
        if value.shape != self._shape:
            raise ValueError(f"base shape {value.shape} does not match expected {self._shape}")
        self.logits.copy_(self._validate_base(value))
        self._invalidate_cache()

    @property
    def _dist(self) -> type:
        if self._dist_factory is None:
            raise TypeError(f"{self.__class__.__name__} must define `_dist_factory`.")
        return self._dist_factory

    @_dist.setter
    def _dist(self, value: type) -> None:
        if not isinstance(value, type):
            raise TypeError("_dist must be a distribution *class*")
        self._dist_factory = value

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        k = len(self._shape)
        if tensor.ndim < k:
            raise ValueError(f"Tensor ndim={tensor.ndim} < required trailing dims={k}")
        if tensor.shape[-k:] != self._shape:
            tensor = tensor.view(*tensor.shape[:-k], *self._shape)
        return tensor

    def _apply_temperature(self,
        logits: torch.Tensor,
        temperature: Optional[float] = None) -> torch.Tensor:
        tau = torch.as_tensor(temperature, device=logits.device) if temperature is not None else self.log_temperature.exp()
        tau = tau.clamp_min(1e-6)
        return logits / tau

    def _prepare_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if self.context_dim is not None and context.shape[-1] != self.context_dim:
            raise ValueError(f"Unsupported context ndim={context.ndim}")
        if context.ndim == 1:  # (H,)
            return context.view(1, 1, -1)
        elif context.ndim == 2:  # (B,H)
            return context.unsqueeze(1)  # (B,1,H)
        elif context.ndim == 3:  # (B,T,H)
            return context
        elif context.ndim == 4:  # (S,B,T,H)
            return context
        else:
            raise ValueError(f"Unsupported context ndim={context.ndim}")

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:

        if context is None or self.context_net is None:
            return torch.zeros_like(base)

        ctx = self._prepare_context(context)
        S, B, T, H = (ctx.shape if ctx.ndim == 4 else (1, *ctx.shape[:2], ctx.shape[-1]))

        # Select timestep if provided
        if timestep is not None:
            t_idx = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
            ctx = ctx[:, :, t_idx:t_idx+1, :] if T > 1 else ctx

        # Optional projection
        if self._proj is not None and self.allow_projection:
            ctx = self._proj(ctx.reshape(-1, ctx.shape[-1])).view(*ctx.shape[:-1], -1)
        elif self.context_dim is not None and ctx.shape[-1] != self.context_dim:
            raise ValueError(f"context_dim mismatch: got {ctx.shape[-1]}, expected {self.context_dim}")

        # Deep context network
        delta = self.context_net(ctx.reshape(-1, ctx.shape[-1]))

        expected = int(torch.prod(torch.tensor(self._shape)))
        if delta.shape[-1] != expected:
            raise RuntimeError("context_net output mismatch")

        delta = delta.view(*ctx.shape[:-1], *self._shape)
        delta = self.final_activation_fn(delta)

        if grad_scale is not None: delta = delta * grad_scale

        delta = torch.clamp(delta * self.delta_scale, -self.max_delta, self.max_delta)

        if timestep is not None and delta.shape[2] == 1:
            delta = delta.squeeze(2)
        return delta

    def _modulate(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **kwargs) -> torch.Tensor:

        base = self._tensor_shape(self.base)
        delta = self._apply_context(base, context, timestep)
        mod = self._apply_constraints(base + delta, mask=kwargs.get("mask", None))
        mod = self._apply_temperature(mod, temperature)
        mod = self._validate_base(mod)
        return mod

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
    def log_matrix(self, *args, **kwargs) -> torch.Tensor:
        pass

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context=None, temperature=None, timestep=None, **dist_kwargs):
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep)
        return self._dist(**self._dist_params(mod, **dist_kwargs))

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
        timestep: Optional[int] = None, **kwargs) -> torch.Tensor:

        x_tensor = x.long()
        n_states = self._shape[-1]

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        if mod.shape[-1] != n_states:
            raise ValueError(f"Last dim of modulated logits ({mod.shape[-1]}) != n_states ({n_states})")

        # Normalize logits shape to (*x.shape, n_states)
        while mod.ndim > x_tensor.ndim + 1:
            mod = mod.squeeze(0)
        while mod.ndim < x_tensor.ndim + 1:
            mod = mod.unsqueeze(0)

        mod = mod.expand(*x_tensor.shape, n_states)

        mod_flat = mod.reshape(-1, n_states)
        x_flat = x_tensor.reshape(-1, 1)

        logp = nnF.log_softmax(mod_flat, dim=-1).gather(-1, x_flat).squeeze(-1)
        return logp.view(*x_tensor.shape)

    def sample(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs) -> torch.Tensor:

        n_states = self._shape[-1]
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep, **dist_kwargs)
        s = dist.sample()

        if s.dtype in [torch.int64, torch.long] or s.ndim == 0:
            s = nnF.one_hot(s, num_classes=n_states)
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
            s = nnF.one_hot(s, num_classes=n_states)
        if s.shape[-1] != n_states:
            s = s.view(*s.shape[:-1], n_states)
        return s

    def expected_probs(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **kwargs) -> torch.Tensor:
        mod_logits = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        return nnF.softmax(mod_logits, dim=-1)

    def mode(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, **dist_kwargs):

        dist = self._get_dist(context, temperature, timestep=timestep, **dist_kwargs)
        if hasattr(dist, "mode"):
            return dist.mode
        if hasattr(dist, "probs"):
            return dist.probs.argmax(-1)
        return torch.argmax(nnF.softmax(dist.logits, dim=-1), dim=-1)

    def _validate_base(self, base: torch.Tensor) -> torch.Tensor:
        base = torch.clamp(base, -MAX_LOGITS, MAX_LOGITS)
        if not torch.isfinite(base).all():
            bad_idx = (~torch.isfinite(base)).nonzero(as_tuple=True)
            raise ValueError(f"Non-finite base at {bad_idx}")
        return base


class Initial(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        self._shape = (n_states,)

        super().__init__(
            target_dim=n_states,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            allow_projection=allow_projection,
            final_activation="tanh",
            activation="tanh",
        )

        self.n_states = n_states
        self.init_mode = init_mode
        self._init_params(mode=init_mode)

    def _init_params(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5) -> torch.Tensor:

        mode = mode or self.init_mode

        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.n_states))
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, self.n_states)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(self._shape) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        if context is not None and self.context_net is not None:
            ctx = context.mean(dim=tuple(range(context.ndim - 1))) if context.ndim > 1 else context
            delta = self.context_net(ctx)
            logits = logits + delta.view_as(logits)

        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter

        return logits

    def initialize(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5, **dist_kwargs) -> Distribution:

        init_logits = self._init_params(mode=mode, context=context, jitter=jitter)
        self.logits = nn.Parameter(init_logits, requires_grad=True)
        return self._get_dist(context=context, temperature=temperature, timestep=None, **dist_kwargs)

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None: return logits
        mask = mask.view((1,) * (logits.ndim - 1) + mask.shape)
        return logits.masked_fill(~mask, NEG_INF)

    def log_matrix(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, T: Optional[int] = None, **kwargs) -> torch.Tensor:

        logits = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)

        # ensure 3 dims [B, T, K]
        if logits.ndim == 1:       # [K] → [1,1,K]
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 2:     # [T,K] → [1,T,K]
            logits = logits.unsqueeze(0)
        # else [B,T,K], do nothing

        # if T is not None and logits.shape[1] == 1:
            # logits = logits.expand(-1, T, -1)
        return logits


class Duration(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        self._shape = (n_states, max_duration)
        super().__init__(
            target_dim=n_states * max_duration,
            allow_projection=allow_projection,
            context_dim=context_dim,
            final_activation="tanh",
            hidden_dim=hidden_dim,
            activation="tanh",
        )
        self.n_states = n_states
        self.init_mode = init_mode
        self.max_duration = max_duration
        self._init_params(mode=init_mode)

    def _init_params(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5) -> torch.Tensor:

        mode = mode or self.init_mode

        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.max_duration))
        elif mode == "biased":
            w = torch.linspace(0.7, 0.3, self.max_duration)
            w = w.unsqueeze(0).expand(self.n_states, -1)
            w = (w / w.sum(dim=1, keepdim=True)).clamp_min(EPS)
            logits = w.log()
        elif mode == "normal":
            logits = torch.randn(*self._shape) * 0.1
            decay = torch.arange(self.max_duration) * 0.05
            logits = logits - decay.unsqueeze(0)
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        if context is not None and self.context_net is not None:
            ctx_delta = self.context_net(context.mean(dim=0))
            logits = logits + ctx_delta.view_as(logits)

        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter

        return logits

    def initialize(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5, **dist_kwargs) -> Distribution:

        init_logits = self._init_params(mode)
        self.logits = nn.Parameter(init_logits, requires_grad=True)
        return self._get_dist(context=context, temperature=temperature, timestep=None, **dist_kwargs)

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = logits.clamp(min=NEG_INF, max=MAX_LOGITS)
        if mask is None: return logits
        mask = mask.to(device=logits.device, dtype=torch.bool)
        mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
        return logits.masked_fill(~mask, NEG_INF)

    def log_matrix(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        T: Optional[int] = None, **kwargs) -> torch.Tensor:

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        # mod: [B, T, K, D]   (duration)
        # or   [B, T, K, K]   (transition)

        soft_dmax = kwargs.get("soft_dmax", None)

        if soft_dmax is not None:
            gate = torch.sigmoid(soft_dmax).clamp_min(EPS)

            if gate.ndim == 1:
                # [D] → global duration gate
                mod = mod + gate.log().view(1, 1, 1, -1)

            elif gate.ndim == 2:
                # [K, D] → state-dependent duration gate
                mod = mod + gate.log().view(1, 1, *gate.shape)

            else:
                raise ValueError("soft_dmax must have shape [D] or [K, D]")

        logp = nnF.log_softmax(mod, dim=-1)

        while logp.ndim < 4:
            logp = logp.unsqueeze(0)

        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, *logp.shape[2:])

        return logp


class Transition(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transition_type: str,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        max_duration: Optional[int] = None,
    ):

        if max_duration is None:
            self._shape = (n_states, n_states)
            self.target_dim = n_states * n_states
        else:
            self._shape = (n_states, max_duration, n_states)
            self.target_dim = n_states * max_duration * n_states

        super().__init__(
            allow_projection=allow_projection,
            target_dim=self.target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation="tanh",
        )
        self.n_states = n_states
        self.init_mode = init_mode
        self.max_duration = max_duration
        self.transition_type = transition_type
        # self._init_params(mode=init_mode)

    def _init_params(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5) -> torch.Tensor:

        K = self.n_states
        mode = mode or self.init_mode
        D = self.max_duration
        shape = (K, D, K) if D > 1 else (K, K)

        if mode == "uniform":
            logits = torch.full(shape, -math.log(K))
        elif mode == "biased":
            if D == 1:
                m = torch.full((K, K), 0.1)
                m.fill_diagonal_(0.7)
                m /= m.sum(dim=-1, keepdim=True)
                logits = m.log()
            else:  # duration-dependent: shape (K, D, K)
                m = torch.full(shape, 0.1)
                for k in range(K):
                    m[k, :, k] = 0.7  # bias self-transitions across durations
                    m[k] /= m[k].sum(dim=-1, keepdim=True)
                logits = m.log()
        elif mode == "normal":
            logits = torch.randn(*shape) * 0.1
            if D > 1:
                decay = torch.arange(D) * 0.05  # decay over duration
                logits = logits - decay.view(1, D, 1)
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

        if context is not None and self.context_net is not None:
            ctx_delta = self.context_net(context.mean(dim=0))
            logits = logits + ctx_delta.view(*([1]*(logits.ndim - ctx_delta.ndim)), *ctx_delta.shape)

        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter
        return logits

    def initialize(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5, **dist_kwargs) -> Distribution:

        init_logits = self._init_params(mode)
        self.logits = nn.Parameter(init_logits, requires_grad=True)
        return self._get_dist(context=context, temperature=temperature, timestep=None, **dist_kwargs)

    def _apply_constraints(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.n_states
        D = getattr(self, "max_duration", None)
        device = logits.device

        if self.transition_type == "ergodic":
            constraint = None  # no restriction
        elif self.transition_type == "semi":
            if D is None:
                constraint = torch.eye(n, device=device, dtype=torch.bool)  # [K,K]
            else:
                constraint = torch.zeros(n, D, n, device=device, dtype=torch.bool)
                for k in range(n):
                    constraint[k, :, k] = True  # allow self-transitions across all durations
        elif self.transition_type == "left-to-right":
            if D is None:
                constraint = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=0)
            else:
                constraint = torch.zeros(n, D, n, device=device, dtype=torch.bool)
                for k in range(n):
                    for to_state in range(k+1):
                        constraint[k, :, to_state] = True
        else:
            raise ValueError(f"Unsupported transition_type: {self.transition_type}")

        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            while mask.ndim < logits.ndim:
                mask = mask.unsqueeze(0)
            constraint = mask if constraint is None else (constraint & mask)

        if constraint is None: return logits
        constraint = constraint.view((1,) * (logits.ndim - constraint.ndim) + constraint.shape)
        return logits.masked_fill(~constraint, NEG_INF)

    def log_matrix(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None, T: Optional[int] = None, **kwargs) -> torch.Tensor:

        mod = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        # mod shape:
        #   duration=None      -> [B, T, K, K]
        #   duration!=None     -> [B, T, K, D, K]

        soft_dmax = kwargs.get("soft_dmax", None)

        if soft_dmax is not None and self.max_duration is not None:
            # soft_dmax: [K, D]
            gate = torch.sigmoid(soft_dmax)  # [K, D]
            gate = gate.clamp_min(EPS)

            # Broadcast to [1, 1, K, D, 1]
            mod = mod + gate.log().view(1, 1, *gate.shape, 1)

        logp = nnF.log_softmax(mod, dim=-1)

        expected_ndim = 4 if self.max_duration is None else 5
        while logp.ndim < expected_ndim:
            logp = logp.unsqueeze(0)

        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, *logp.shape[2:])

        return logp


class Emission(Neural):

    def __init__(
        self,
        n_states: int,
        n_features: int,
        emission_type: str,
        min_covar: float = 1e-6,
        init_mode: str = "spread",
        allow_projection: bool = True,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        self._shape = (n_states, n_features)

        if emission_type == "gaussian":
            self._dist_factory = MultivariateNormal
        elif emission_type == "studentt":
            self.dof = nn.Parameter(torch.full((n_states,), 5.0))
            self._dist_factory = IndependentStudentT
        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")

        super().__init__(
            allow_projection=allow_projection,
            target_dim=n_states * n_features,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        )
        self.n_states = n_states
        self.init_mode = init_mode
        self.min_covar = min_covar
        self.n_features = n_features
        self.emission_type = emission_type

        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features) * 0.1)
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -1.0))
        else:
            self.loc = nn.Parameter(torch.randn(self.n_states, self.n_features) * 0.1)
            self.scale_param = nn.Parameter(torch.full((self.n_states, self.n_features), 0.1))

    @property
    def base(self):
        base = self.mu if self.emission_type == "gaussian" else self.loc
        base = self._validate_base(base)
        return base

    def _init_params(self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5) -> torch.Tensor:
        mode = mode or self.init_mode

        if mode == "random":
            init_mean = torch.randn(self.n_states, self.n_features) * 0.1
        elif mode == "spread":
            linspace = torch.linspace(-1.0, 1.0, steps=self.n_states)
            perm_idx = torch.stack([torch.randperm(self.n_states) for _ in range(self.n_features)], dim=0)  # [F, K]
            init_mean = linspace[perm_idx].T  # [K, F]
            init_mean += torch.randn_like(init_mean) * 0.05  # jitter
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if context is not None and self.context_net is not None:
            if context.ndim in (2, 3):
                ctx = context.mean(dim=0)
            else:
                ctx = context
            delta = self.context_net(ctx)
            if delta.numel() == self.n_states * self.n_features:
                delta = delta.view(self.n_states, self.n_features)
            init_mean = init_mean + delta

            if self.emission_type == "studentt":
                df_delta = self.context_net(ctx)
                df = nnF.softplus(self.dof + df_delta) + 2.0
                self.dof.copy_(df)

        mean_spread = init_mean.std(dim=0, keepdim=True).clamp_min(self.min_covar)
        init_var = mean_spread**2 + torch.rand(self.n_states, self.n_features) * jitter

        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(init_mean, requires_grad=True)
            self.log_var = nn.Parameter(torch.log(init_var), requires_grad=True)
        else:
            self.loc = nn.Parameter(init_mean, requires_grad=True)
            self.scale_param = nn.Parameter(init_var.sqrt(), requires_grad=True)

        return self.base

    @torch.no_grad()
    def initialize(self,
        mode: Optional[str] = "spread",
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        jitter: float = 1e-5, **dist_kwargs) -> Distribution:
        base = self._init_params(mode, context, jitter)
        return self._dist(**self._dist_params(base, **dist_kwargs))

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = super()._tensor_shape(tensor)
        if tensor.ndim == 2:  # [K, F]
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, K, F]
        elif tensor.ndim == 3:  # [B, K, F]
            tensor = tensor.unsqueeze(1)  # [B, 1, K, F]
        return tensor

    def _apply_constraints(self,
        tensor: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:

        if tensor is None: return None
        if mask is not None:
            tensor = tensor.clone()
            tensor[~mask] = 0.0
        return tensor

    def _apply_context(self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None) -> torch.Tensor:

        delta = super()._apply_context(base, context=context, timestep=timestep, grad_scale=grad_scale)
        if grad_scale is not None: delta = delta * grad_scale
        return delta

    def _dist_params(self, loc: torch.Tensor, **dist_kwargs) -> dict:
        if loc.ndim == 2:  # [K, F] -> [1, 1, K, F]
            loc = loc.unsqueeze(0).unsqueeze(0)

        B, T, K, n_feat = loc.shape
        if self.emission_type == "gaussian":
            var = nnF.softplus(self.log_var).clamp_min(self.min_covar)  # [K, F]
            cov = torch.diag_embed(var)                                                  # [K, F, F]
            cov = cov[None, None, :, :, :].expand(B, T, K, n_feat, n_feat)              # [B, T, K, F, F]
            return {
                "loc": loc,
                "covariance_matrix": cov,
                # "scale_tril": torch.diag_embed(var.sqrt()),
                **dist_kwargs
            }

        elif self.emission_type == "studentt":
            scale = nnF.softplus(self.scale_param).clamp_min(self.min_covar)  # [K, F]
            df = nnF.softplus(self.dof) + 2.0                                 # [K]
            scale = scale[None, None, :, :].expand(B, T, K, n_feat)                            # [B, T, K, F]
            df = df[None, None, :, None].expand(B, T, K, n_feat)                                # [B, T, K, F]
            return {"loc": loc, "scale": scale, "df": df, **dist_kwargs}

        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    def forward(self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False, **dist_kwargs) -> Tuple:

        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        if return_dist: return dist
        if self.emission_type == "gaussian":
            if hasattr(dist, 'covariance_matrix'):
                cov = dist.covariance_matrix
            elif hasattr(dist, 'scale_tril'):
                cov = dist.scale_tril @ dist.scale_tril.transpose(-1, -2)
            return self._tensor_shape(dist.mean), self._tensor_shape(cov)
        return self._tensor_shape(dist.loc), self._tensor_shape(dist.scale)

    def log_prob(self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None, **dist_kwargs) -> torch.Tensor:
        # Ensure canonical shape [B, T, F]
        if x.ndim == 1: x = x.view(1, 1, -1)
        elif x.ndim == 2: x = x.unsqueeze(0)

        B, T, F = x.shape
        K = self.n_states

        dist = self._get_dist(context=context, temperature=temperature, **dist_kwargs)
        if F != self.n_features:
            raise ValueError(f"Feature mismatch: input F={F}, expected {self.n_features}")

        # Generate [B, T, K, F] tensor for log_prob evaluation
        if x.shape[-1] == self.n_features:
            x_exp = x[..., None, :]           # [B, T, 1, F]
            x_exp = x_exp.expand(-1, -1, K, -1)  # [B, T, K, F]
        else:
            x_exp = x

        logp = dist.log_prob(x_exp)          # [B, T, K] or [B, T, K, F]
        if logp.ndim == 4: logp = logp.sum(-1)              # [B, T, K]
        return logp

    def log_matrix(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Emission distributions don't have a log_matrix representation")

