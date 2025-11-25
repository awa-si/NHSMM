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

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        """
        Fully batch-shape agnostic context modulation with gates/residuals/low-rank factors.
        Supports:
          - base: [K], [K,Dmax], [L,K], [L,K,K], ...
          - context: [H], [T,H], [B,T,H]
        Avoids in-place ops and broadcasting errors.
        """
        orig_shape = base.shape
        base_ndim = base.ndim
        K = base.shape[-2] if base_ndim >= 2 else base.shape[-1]  # last-but-one is states if >=2

        # -------- Align context --------
        if context is None:
            ctx_flat = None
            B, T = 1, 1
        else:
            # ensure context is [B,T,H]
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B*T, H)

            # Optional projection
            expected = getattr(self, "context_dim", H)
            if H != expected:
                if not getattr(self, "allow_projection", True):
                    raise ValueError(f"Context dim mismatch: expected {expected}, got {H}")
                if not hasattr(self, "_proj") or self._proj.in_features != H:
                    self._proj = nn.Linear(H, expected, dtype=base.dtype)
                    nn.init.xavier_uniform_(self._proj.weight)
                    nn.init.zeros_(self._proj.bias)
                ctx_flat = self._proj(ctx_flat)

        # -------- Expand base to match context if needed --------
        USE_GATES = False
        if ctx_flat is not None and USE_GATES:
            bshape = (B*T,) + base.shape[-base_ndim:]
            base_exp = base.unsqueeze(0).expand(bshape) if base_ndim < 3 else base.reshape(-1, *base.shape[-2:])
            delta = torch.zeros_like(base_exp)

            if getattr(self, "context_gate", None):
                gate = self.context_gate(ctx_flat)
                gate = gate.reshape_as(delta)
                delta = delta + gate
            if getattr(self, "residual_gate", None):
                res = self.residual_gate(ctx_flat)
                res = res.reshape_as(delta)
                delta = delta + res
            if getattr(self, "_U", None) and getattr(self, "_V", None):
                r = getattr(self, "rank", 1)
                U = self._U(ctx_flat).view(ctx_flat.size(0), -1, r)
                V = self._V(ctx_flat).view(ctx_flat.size(0), -1, r)
                delta = delta + (U * V).sum(-1)

            # Restore batch/sequence shape
            delta = delta.reshape((B, T) + base.shape[-base_ndim:])

            # Squeeze leading batch dim if base had no batch
            if base_ndim < 3:
                delta = delta.squeeze(0)
                if base_ndim == 1:
                    delta = delta.squeeze(0)

        else:
            delta = torch.zeros_like(base)

        # -------- Normalization --------
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if getattr(self, "layer_norm", False):
            delta = F.layer_norm(delta, delta.shape[-1:])
        if getattr(self, "batch_norm", False):
            nf = delta.shape[-1]
            if not hasattr(self, "_batchnorm") or self._batchnorm is None or self._batchnorm.num_features != nf:
                self._batchnorm = nn.BatchNorm1d(nf, affine=True, track_running_stats=False, eps=EPS)
            delta = self._batchnorm(delta)

        # -------- Adapters --------
        if not skip_adapters:
            if getattr(self, "temporal_adapter", None):
                delta = self.temporal_adapter(delta.unsqueeze(-1)).squeeze(-1)
            if getattr(self, "spatial_adapter", None):
                delta = self.spatial_adapter(delta)

        # -------- Activation, scale, clamp, grad --------
        delta = getattr(self, "final_activation_fn", nn.Identity())(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        mx = getattr(self, "max_delta", 1e6)
        delta = torch.clamp(delta, -mx, mx)
        if grad_scale is not None:
            delta = delta * grad_scale

        # Remove unnecessary singleton dimensions if base was lower-rank
        while delta.ndim > base.ndim:
            delta = delta.squeeze(1)

        # -------- Combine with base and apply constraints --------
        return self._apply_constraints(base + delta)

    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        # Unified temperature handling
        tau = temperature if temperature is not None else getattr(self, "temperature", 1.0)
        tau_tensor = torch.as_tensor(tau, dtype=DTYPE, device=self.logits.device)
        tau_tensor = tau_tensor.clamp_min(EPS)

        # Cache
        key = f"{self._context_hash(context)}-T{float(tau):.6g}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()

        # Apply context
        base_logits = getattr(self, "logits", torch.zeros(1, dtype=DTYPE, device=tau_tensor.device))
        mod = self._apply_context(base_logits, context)

        # Temperature + logsoftmax
        mod = mod / tau_tensor
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)

        # Last validation pass
        mod = self._validate_logits(mod)

        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        tau_val = tau or torch.exp(self.log_temperature)
        mod_logits = logits / tau_val.clamp_min(EPS)
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = self._validate_logits(mod_logits)
        return {"logits": mod_logits, **kwargs}

    def forward(self, log=False, return_dist=False, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, **dist_kwargs):
        mod = self._modulate(context, temperature, grad_safe=False)
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
        self._param_version += 1
        self._cache.clear()

    # ---------------- Placeholder methods ----------------
    def _apply_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _validate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp large values without changing sign
        logits = torch.where(
            logits > MAX_LOGITS, torch.full_like(logits, MAX_LOGITS), logits
        )
        logits = torch.where(
            logits < -MAX_LOGITS, torch.full_like(logits, -MAX_LOGITS), logits
        )

        # Check for non-finite values
        if not torch.isfinite(logits).all():
            # Identify offending entries
            finite_mask = torch.isfinite(logits)
            bad_indices = (~finite_mask).nonzero(as_tuple=True)
            bad_vals = logits[bad_indices]
            raise ValueError(
                f"Logits contain non-finite values at indices {bad_indices}: {bad_vals}"
            )
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
        from_probs: bool = False,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        context: Optional[torch.Tensor] = None,
        grad_safe: bool = False
    ):
        """
        Unified EM / learnable update for logits.
        
        - new_logits: Direct replacement (logits or probabilities if from_probs=True)
        - posterior: Soft counts for EM-style update
        - update_rate: Scaling for gradient or EM update
        - temperature: Optional scaling for logits
        - context: Optional for neural modulated logits
        - grad_safe: Use grad-safe additive update
        """
        # ---------------- Direct replacement ----------------
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))

            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)

            # Apply module-specific constraints if available
            new_logits = self._apply_constraints(new_logits) if hasattr(self, "_apply_constraints") else new_logits
            new_logits = self._validate_logits(new_logits)

            # Ensure shape matches self.logits exactly
            if new_logits.shape != self.logits.shape:
                raise ValueError(f"Shape mismatch: new_logits {new_logits.shape} vs self.logits {self.logits.shape}")

            # Replace safely
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        # ---------------- Posterior-based EM update ----------------
        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            mod_logits = self._modulate(context=context, temperature=temperature)

            # Broadcast posterior safely to match mod_logits
            if posterior.ndim < mod_logits.ndim:
                posterior = posterior.view(*posterior.shape, *[1]*(mod_logits.ndim - posterior.ndim))
            if posterior.shape != mod_logits.shape:
                try:
                    posterior = posterior.expand_as(mod_logits)
                except RuntimeError as e:
                    raise ValueError(f"Cannot expand posterior {posterior.shape} to modulated logits {mod_logits.shape}") from e

            # Compute negative log-likelihood loss
            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)

            if grad_safe:
                # Grad-safe: manual gradient update
                grads = torch.autograd.grad(
                    loss,
                    [p for p in self.parameters() if p.requires_grad],
                    retain_graph=False,
                    allow_unused=True
                )
                with torch.no_grad():
                    for p, g in zip([p for p in self.parameters() if p.requires_grad], grads):
                        if g is not None:
                            p.add_((update_rate or 1.0) * g)
            else:
                # Standard gradient update
                self.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in self.parameters():
                        if p.grad is not None:
                            p.data.add_((update_rate or 1.0) * p.grad)

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

        self.n_states = n_states
        self.rank = rank
        self.init_mode = init_mode
        self.temperature = temperature

        # Base logits
        init_logits = self._init_logits(n_states, init_mode)
        self.logits.data.copy_(init_logits)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / low-rank factors
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
            elif hasattr(self, "_U") or hasattr(self, "_V"):
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

    # ---------------- Modulation ----------------
    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        grad_safe: bool = False
    ) -> torch.Tensor:
        """
        Modulate base logits with optional context and temperature.
        If grad_safe=True, returns a detached tensor for safe EMA-style updates.
        
        Returns shape [batch, n_states] if context is provided, else [n_states].
        """
        base_logits = self.logits  # [K]

        # Expand to batch if context is given
        if context is not None:
            batch_size = context.shape[0]
            if base_logits.ndim == 1:
                base_logits = base_logits.unsqueeze(0).expand(batch_size, -1)  # [B, K]

            # Apply context gate
            if hasattr(self, "context_gate") and self.context_gate is not None:
                delta = self.context_gate(context)
                mod_logits = base_logits + delta
            else:
                mod_logits = base_logits
        else:
            mod_logits = base_logits

        # Apply temperature scaling
        tau = max(temperature or self.temperature, 1e-6)
        mod_logits = mod_logits / tau

        # Detach for grad-safe updates
        if grad_safe:
            mod_logits = mod_logits.detach()

        return mod_logits


    # ---------------- Distribution helpers ----------------
    def _get_dist(self, context=None, temperature=None, timestep: Optional[int] = None, **kwargs) -> Categorical:
        mod_logits = self._modulate(context=context, temperature=temperature)

        # Timestep handling
        if timestep is not None:
            if mod_logits.ndim == 3:  # [B, T, K]
                mod_logits = mod_logits[:, timestep, :]
            else:
                raise ValueError(f"Timestep {timestep} incompatible with shape {mod_logits.shape}")

        return self._dist_type(logits=mod_logits, **kwargs)

    # ---------------- Sampling / log-prob ----------------
    def sample(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.sample()

    def log_prob(self, x, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.log_prob(x)

    def expected_probs(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.probs

    def mode(self, context=None, temperature=None, timestep: Optional[int] = None):
        dist = self._get_dist(context=context, temperature=temperature, timestep=timestep)
        return dist.probs.argmax(-1) if not hasattr(dist, "mode") else dist.mode

    # ---------------- Robust update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: float = 1.0,
        temperature: Optional[float] = None,
        grad_safe: bool = True,
    ):
        if new_logits is not None:
            # Collapse batch and sequence dims
            if new_logits.ndim > 1:
                new_logits = new_logits.mean(dim=tuple(range(new_logits.ndim - 1)))

            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))

            if temperature is not None:
                new_logits = new_logits / max(temperature, EPS)

            # Detach if grad_safe
            if grad_safe:
                new_logits = new_logits.detach()

            # EMA update
            self.logits.data.mul_(1 - update_rate).add_(update_rate * new_logits)
            self._logits_buffer.copy_(self.logits.data)
            self._mod_logits_buffer.copy_(self.logits.data)
            self._invalidate_cache()
            return

        # Optional posterior-based EM update
        if posterior is not None:
            mod_logits = self._modulate(context=context, temperature=temperature)

            # Broadcast posterior if needed
            while posterior.ndim < mod_logits.ndim:
                posterior = posterior.unsqueeze(-1)
            if posterior.shape != mod_logits.shape:
                posterior = posterior.expand_as(mod_logits)

            log_probs = F.log_softmax(mod_logits, dim=-1)
            loss = -(posterior * log_probs).sum() / (posterior.sum() + EPS)

            # Standard autograd update
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
    """

    _dist_type = Categorical

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        rank: Optional[int] = None,
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

        self.rank = rank
        self.n_states = n_states
        self.max_duration = max_duration
        self.init_mode = init_mode
        self.temperature = max(temperature, 1e-6)  # ensure positive non-zero

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context gates / residual / low-rank handled automatically
        if context_dim is not None:
            hidden_dim = hidden_dim or max(16, n_states)  # ensure valid hidden_dim
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
            # Bias shorter durations; document
            logits = torch.randn(n_states, max_duration, dtype=DTYPE) * 0.1
            logits -= torch.arange(max_duration, dtype=DTYPE) * 0.05
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        return self._validate_logits(logits)

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, tau: Optional[float] = None, **kwargs) -> dict:
        tau_val = max(tau if tau is not None else self.temperature, 1e-6)  # prevent tau=0
        mod_logits = logits / tau_val
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = self._validate_logits(mod_logits)
        return {"logits": mod_logits, **kwargs}

    def _get_dist(self, context=None, temperature=None) -> Categorical:
        tau = max(temperature or self.temperature, 1e-6)
        key = self._context_hash(context) + f"-T{tau:.6g}"

        # Return cached distribution if available
        if key in self._cache:
            return self._cache[key]

        # Modulate logits
        mod_logits = self._modulate(context=context, temperature=tau)

        # Prepare logits with temperature scaling
        mod_logits = mod_logits / tau
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = self._validate_logits(mod_logits)

        dist = Categorical(logits=mod_logits)
        self._cache[key] = dist
        return dist

    # ---------------- Optimized Sampling ----------------
    def sample(self, context=None, temperature=None, hard: bool = True, generator=None):
        dist = self._get_dist(context, temperature)
        return dist.sample(hard=hard, generator=generator)

    def rsample(self, context=None, temperature=None, generator=None):
        dist = self._get_dist(context, temperature)
        return dist.rsample(generator=generator)

    def mode(self, context=None, temperature=None, return_dist=False):
        dist = self._get_dist(context, temperature)
        if return_dist: 
            return dist
        return dist.probs.argmax(-1)


class Transition(DistributionBase):
    """
    Context-aware categorical transition distribution for HSMMs.
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
        self.temperature = max(temperature, 1e-6)
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)

        # Base logits [K,K]
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Context modules
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
        """
        Context-modulated transition logits, batch-shape safe.
        """
        K = self.n_states

        # Determine target shape
        if context is None:
            target_shape = base.shape  # [K,K]
            delta = torch.zeros_like(base)
        else:
            # Ensure context is [B,T,H]
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(0)
            B, T, H = context.shape
            ctx_flat = context.reshape(B*T, H)

            # Broadcast base to [B,T,K,K]
            if base.ndim == 2:
                base_exp = base.unsqueeze(0).unsqueeze(0).expand(B, T, K, K)
            elif base.ndim == 3:  # [T,K,K]
                base_exp = base.unsqueeze(0).expand(B, T, K, K)
            else:  # [B,T,K,K]
                base_exp = base

            # Compute delta
            delta = torch.zeros_like(base_exp)

            if getattr(self, "context_gate", None):
                gate = self.context_gate(ctx_flat).view(B, T, K, K)
                delta = delta + gate
            if getattr(self, "residual_gate", None):
                res = self.residual_gate(ctx_flat).view(B, T, K, K)
                delta = delta + res
            if getattr(self, "_U", None) and getattr(self, "_V", None):
                r = self.rank
                U = self._U(ctx_flat).view(B*T, K, r)
                V = self._V(ctx_flat).view(B*T, K, r)
                delta = delta + torch.einsum("bik,bjk->bij", U, V).view(B, T, K, K)

            target_shape = base_exp.shape
            base = base_exp

        # Apply final activation and scaling
        if getattr(self, "final_activation_fn", None):
            delta = getattr(self, "final_activation_fn")(delta)
        delta = delta * getattr(self, "delta_scale", 1.0)
        if grad_scale is not None:
            delta = delta * grad_scale

        # Add to base
        mod = base + delta

        return self._apply_constraints(mod)

    # ---------------- Modulate ----------------
    def _modulate(self, context=None, temperature=None, grad_safe=False) -> torch.Tensor:
        """
        Compute modulated logits with context and temperature.
        Supports caching, batch/time dimensions, and structural constraints.
        """
        tau = temperature if temperature is not None else self.temperature
        tau = max(tau, EPS)
        
        # Cache key
        key = f"{self._context_hash(context)}-T{float(tau):.6g}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached if grad_safe else cached.detach()
        
        # Apply context directly to [K,K] or [B,T,K,K]
        mod = self._apply_context(self.logits, context)
        
        # Apply temperature and logsoftmax over last dimension (next-state axis)
        mod = mod / tau
        if mod.ndim == 2:
            mod = mod - mod.logsumexp(dim=-1, keepdim=True)  # [K,K]
        else:
            mod = mod - mod.logsumexp(dim=-1, keepdim=True)  # [B,T,K,K]
        
        # Validate logits
        mod = self._validate_logits(mod)
        
        # Cache
        self._cache_set(key, mod if grad_safe else mod.detach())
        return mod

    # ---------------- Distribution helpers ----------------
    def _dist_params(self, logits: torch.Tensor, **kwargs) -> dict:
        return {"logits": self._validate_logits(logits)}

    def _get_dist(self, context=None, temperature=None):
        """
        Returns a Categorical distribution for the transition matrix.
        Batch-shape safe for [K,K], [T,K,K], or [B,T,K,K].
        """
        mod = self._modulate(context, temperature)

        # Flatten for torch.distributions.Categorical only if sampling
        if mod.ndim == 2:
            return Categorical(logits=mod)  # [K,K]
        elif mod.ndim == 3:
            # [T,K,K] -> flatten last axis for Categorical
            T, K1, K2 = mod.shape
            return Categorical(logits=mod.view(T*K1, K2))
        elif mod.ndim == 4:
            # [B,T,K,K] -> flatten B*T*K1
            B, T, K1, K2 = mod.shape
            return Categorical(logits=mod.view(B*T*K1, K2))
        else:
            raise ValueError(f"Unexpected modulated logits shape: {mod.shape}")

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
            mask = torch.eye(n, dtype=torch.bool, device=logits.device)
            # Broadcast mask to match batch dims
            mask = mask.view((1,) * (logits.ndim - 2) + mask.shape)
            out[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            tril_mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=logits.device), -1)
            tril_mask = tril_mask.view((1,) * (logits.ndim - 2) + tril_mask.shape)
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
        else:
            self.context_gate = None

    # ------------------- Context / Modulation -------------------
    def _modulate(self, base: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        """Use DistributionBase._apply_context for all context modulation."""
        mod = self._apply_context(base, context)
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            tau = temperature or self.temperature
            mod = mod / max(tau, 1e-6)
        return mod

    # ------------------- Distribution Construction -------------------
    def _dist_params(self, tensor: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None):
        mod = self._modulate(tensor, context=context, temperature=temperature)
        if self.emission_type == "gaussian":
            var = F.softplus(self.log_var).clamp_min(self.min_covar)
            cov = torch.diag_embed(var)
            return {"means": mod, "cov": cov}
        elif self.emission_type in {"laplace", "studentt"}:
            scale = self.scale_param.clamp_min(self.min_covar)
            return {"loc": mod, "scale": scale}
        elif self.emission_type in {"categorical", "bernoulli"}:
            return {"logits": mod}
        elif self.emission_type == "poisson":
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
        dtype, device = DTYPE, self._emission_means.device

        # ---------------- Base means ----------------
        means = self._emission_means.clone()

        # Weighted EM update
        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)  # [N, K]
            w_sum = w.sum(dim=0) + EPS    # [K]
            means = (w.T @ X) / w_sum.unsqueeze(1)  # [K, F]

        # Theta modulation
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            means += theta_scale * theta_vec.unsqueeze(0)

        # Context + temperature modulation
        means_mod = self._modulate(means, context=context, temperature=temperature)

        # ---------------- Gaussian ----------------
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

            self._emission_means.copy_(means_mod)
            self._emission_covs.copy_(covs)

            dist_params = self._dist_params(tensor=means_mod, context=context, temperature=temperature)
            return MultivariateNormal(dist_params["means"], covariance_matrix=dist_params["cov"])

        # ---------------- Laplace / StudentT ----------------
        # Compute robust scale estimate
        if X is not None and posterior is not None:
            scales = ((X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]).sum(dim=0) / w_sum[:, None]
        else:
            scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()

        scales = scales.clamp_min(self.min_covar)

        self._emission_means.copy_(means_mod)
        self._emission_covs.copy_(torch.diag_embed(scales**2))

        dist_params = self._dist_params(tensor=means_mod, context=context, temperature=temperature)
        dist_cls = Laplace if self.emission_type == "laplace" else StudentT

        # For StudentT, ensure correct degrees of freedom
        if self.emission_type == "studentt":
            return Independent(dist_cls(df=self.dof, loc=dist_params["loc"], scale=dist_params["scale"]), 1)
        
        return Independent(dist_cls(loc=dist_params["loc"], scale=dist_params["scale"]), 1)

    def _get_discrete_dist(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        temperature: Optional[float] = None,
    ):
        K, F = self.n_states, self.n_features
        etype = self.emission_type
        device = self._emission_means.device if hasattr(self, "_emission_means") else DTYPE

        # ---------------- Base logits ----------------
        if X is not None and posterior is not None:
            w = posterior.clamp_min(EPS)  # [N, K]
            w_sum = w.sum(dim=0) + EPS     # [K]

            if etype == "categorical":
                # Vectorized bincount per state
                logits = torch.zeros((K, F), dtype=DTYPE, device=device)
                for k in range(K):
                    for f in range(F):
                        logits[k, f] = (w[:, k] * (X[:, f] == f).float()).sum()
                logits = torch.log((logits / logits.sum(dim=-1, keepdim=True)).clamp_min(EPS))

            elif etype in {"bernoulli", "poisson"}:
                rate = (w.T @ X.float()) / w_sum.unsqueeze(1)
                logits = torch.log(rate.clamp_min(EPS))

        else:
            # fallback: uniform / small negative logits
            if etype == "categorical":
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=device)
            else:
                logits = torch.zeros((K, F), dtype=DTYPE, device=device)

        # ---------------- Theta modulation ----------------
        if theta is not None:
            theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
            logits += theta_scale * theta_vec.unsqueeze(0)

        # ---------------- Context + temperature ----------------
        logits_mod = self._modulate(logits, context=context, temperature=temperature)

        # ---------------- Update emission buffer ----------------
        if hasattr(self, "_emission_params"):
            self._emission_params.copy_(logits_mod)
        if etype == "categorical" or etype == "bernoulli":
            self.logits.copy_(logits_mod)
        elif etype == "poisson":
            self.log_rate.copy_(logits_mod)

        # ---------------- Distribution construction ----------------
        dist_params = self._dist_params(tensor=logits_mod, context=context, temperature=temperature)

        if etype == "categorical":
            self.dist_type = Categorical
            return Categorical(logits=dist_params["logits"])

        if etype == "bernoulli":
            self.dist_type = lambda *a, **kw: Independent(Bernoulli(*a, **kw), 1)
            return Independent(Bernoulli(logits=dist_params["logits"]), 1)

        if etype == "poisson":
            self.dist_type = lambda *a, **kw: Independent(Poisson(*a, **kw), 1)
            return Independent(Poisson(rate=torch.exp(dist_params["logits"])), 1)

        raise ValueError(f"Unsupported discrete emission_type: {etype}")

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

    def forward(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None, return_dist: bool = False):
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
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        if getattr(dist, "has_rsample", False):
            samples = dist.rsample((n_samples,))
        else:
            samples = dist.sample((n_samples,))
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
        Compute log-probability of input `x` under the emission distribution,
        optionally conditioned on `context` and scaled by `temperature`.
        Fully vectorized; avoids constructing full distribution objects.
        Returns tensor of shape [N, K] (batch x states).
        """
        etype = self.emission_type
        N = x.shape[0]
        K, D = self.n_states, self.n_features
        tau = max(temperature or getattr(self, "temperature", 1.0), EPS)

        # ---------------- Helper: expand x to [N, K, D] ----------------
        def expand_x(x_tensor):
            if x_tensor.ndim == 2:  # [N, D] -> [N, K, D]
                return x_tensor.unsqueeze(1).expand(-1, K, -1)
            return x_tensor

        # ---------------- Base tensor + modulation ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            base = self.mu if etype == "gaussian" else self.loc
            loc = self._modulate(base, context=context, temperature=tau)
            loc_exp = loc.unsqueeze(0) if loc.ndim == 2 else loc  # ensure [1, K, D] or batch-shape
            x_exp = expand_x(x)
            if etype == "gaussian":
                cov = self._emission_covs
                cov_exp = cov.unsqueeze(0) if cov.ndim == 3 else cov
                diff = x_exp - loc_exp
                # Mahalanobis distance for log_prob
                L = torch.linalg.cholesky(cov_exp)  # [1,K,D,D] or [N,K,D,D]
                sol = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                log_det = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
                log_prob = -0.5 * (sol.squeeze(-1)**2).sum(-1) - 0.5*D*math.log(2*math.pi) - 0.5*log_det
                return log_prob

            else:  # laplace / studentt
                scale = self.scale_param.clamp_min(EPS)
                scale_exp = scale.unsqueeze(0) if scale.ndim == 2 else scale
                x_exp = expand_x(x)
                if etype == "laplace":
                    log_prob = -torch.abs(x_exp - loc_exp) / scale_exp - torch.log(2*scale_exp)
                else:  # studentt
                    nu = getattr(self, "nu", 1.0)
                    log_prob = (
                        torch.lgamma((nu + 1)/2) - torch.lgamma(nu/2)
                        - 0.5*math.log(math.pi*nu) - torch.log(scale_exp)
                        - ((nu+1)/2) * torch.log(1 + ((x_exp - loc_exp)/scale_exp)**2 / nu)
                    )
                return log_prob.sum(-1)  # sum over feature dim

        else:  # Discrete emissions
            if etype in {"categorical", "bernoulli"}:
                logits = self._modulate(self.logits, context=context, temperature=tau)
            elif etype == "poisson":
                logits = self._modulate(self.log_rate, context=context, temperature=tau)
            else:
                raise ValueError(f"Unsupported emission_type: {etype}")

            x_exp = expand_x(x)
            if etype == "categorical":
                log_probs = F.log_softmax(logits, dim=-1)
                if log_probs.ndim < x_exp.ndim + 1:
                    log_probs = log_probs.unsqueeze(0).expand(N, K, -1, -1)
                return torch.gather(log_probs, -1, x_exp.unsqueeze(-1)).squeeze(-1)

            elif etype == "bernoulli":
                logits_exp = logits.unsqueeze(0).expand(N, K, D)
                return -F.binary_cross_entropy_with_logits(logits_exp, x_exp.float(), reduction="none").sum(-1)

            elif etype == "poisson":
                rate_exp = logits.unsqueeze(0).exp().clamp_min(EPS).expand(N, K, D)
                x_exp = x_exp.float()
                log_probs = -rate_exp + x_exp * torch.log(rate_exp) - torch.lgamma(x_exp + 1)
                return log_probs.sum(-1)

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
        mode: str = "kmeans",
        iters: int = 15,
    ):
        K, F = self.n_states, self.n_features
        dtype = DTYPE
        device = X.device if X is not None else self._emission_means.device

        # Flatten input
        Xf = X.reshape(-1, F).to(dtype=dtype) if X is not None else None

        # ---------------- Continuous emissions ----------------
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            # Initialize means
            if Xf is not None:
                if mode == "kmeans" and Xf.shape[0] >= K:
                    # Simple KMeans initialization
                    idx = torch.randperm(Xf.shape[0])[:K]
                    means = Xf[idx].clone()
                    for _ in range(iters):
                        dist = torch.cdist(Xf, means)
                        labels = dist.argmin(dim=1)
                        for k in range(K):
                            pts = Xf[labels == k]
                            if pts.numel() > 0:
                                means[k] = pts.mean(0)
                else:
                    # Fallback: global mean
                    means = Xf.mean(0).expand(K, F)

                # Shared covariance initialization
                cov_base = torch.cov(Xf.T) + self.min_covar * torch.eye(F, dtype=dtype, device=device)
                covs = cov_base.unsqueeze(0).repeat(K, 1, 1)
            else:
                means = torch.zeros(K, F, dtype=dtype, device=device)
                covs = torch.eye(F, dtype=dtype, device=device).unsqueeze(0).repeat(K, 1, 1) * self.min_covar

            # Apply theta modulation
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                means += theta_scale * theta_vec.unsqueeze(0)

            # Context + temperature modulation
            means_mod = self._modulate(means, context=context, temperature=temperature)

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
        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            if Xf is not None and mode == "data":
                if self.emission_type == "categorical":
                    # Vectorized bincount per feature
                    counts = torch.stack([
                        torch.bincount(Xf[:, f].long(), minlength=F) for f in range(F)
                    ], dim=1)
                    logits = torch.log((counts / counts.sum(dim=0, keepdim=True)).clamp_min(EPS))
                else:
                    logits = torch.log(Xf.mean(0).expand(K, F).clamp_min(EPS))
            else:
                logits = torch.zeros(K, F, dtype=dtype, device=device)

            # Apply theta/context modulation
            logits_mod = self._modulate(logits, context=context, temperature=temperature)

            # Update learnable buffers
            if hasattr(self, "logits"):
                self.logits.copy_(logits_mod)
            elif hasattr(self, "log_rate"):
                self.log_rate.copy_(logits_mod)
            self._emission_params.copy_(logits_mod)

            return self._get_dist(context=context, temperature=temperature)

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
        Supports Gaussian, Laplace, StudentT, and discrete distributions.
        """
        etype = self.emission_type
        K, F = self.n_states, self.n_features
        device = X.device if X is not None else self._emission_means.device

        # ---------------- Continuous emissions ----------------
        if etype in {"gaussian", "laplace", "studentt"}:
            new_dist = self._get_dist(
                X=X, posterior=posterior, theta=theta, context=context,
                theta_scale=theta_scale, temperature=temperature, max_jitter=max_jitter
            )

            if etype == "gaussian":
                # EMA update of mean and covariance
                self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.mean)
                self._emission_covs.mul_(1 - update_rate).add_(update_rate * new_dist.covariance_matrix)
                self.mu.copy_(self._emission_means)
                diag = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1).clamp_min(EPS)
                self.log_var.copy_(torch.log(diag))

            else:  # laplace/studentt
                loc = getattr(new_dist, "loc", new_dist.mean)
                scale = getattr(new_dist, "scale", None)
                self._emission_means.mul_(1 - update_rate).add_(update_rate * loc)
                diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale**2
                diag = diag.clamp_min(EPS)
                self._emission_covs.copy_(torch.diag_embed(diag))
                self.loc.copy_(self._emission_means)
                self.scale_param.copy_(torch.sqrt(diag))

        # ---------------- Discrete emissions ----------------
        else:
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)  # [N, K]
                w_sum = w.sum(dim=0) + EPS  # [K]

                if etype == "categorical":
                    # Vectorized per-feature count
                    X_long = X.long()
                    counts = torch.zeros((K, F), dtype=DTYPE, device=device)
                    for k in range(K):
                        for f in range(F):
                            counts[k, f] = (w[:, k] * (X_long[:, f] == f).float()).sum()
                    new_params = torch.log((counts / counts.sum(dim=-1, keepdim=True)).clamp_min(EPS))

                elif etype in {"bernoulli", "poisson"}:
                    rate = (w.T @ X.float()) / w_sum.unsqueeze(1)
                    new_params = torch.log(rate.clamp_min(EPS))
            else:
                # fallback to current parameters if no data
                new_params = getattr(self, "logits", None) if etype != "poisson" else getattr(self, "log_rate", None)

            # Apply theta modulation
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                new_params = new_params + theta_scale * theta_vec.unsqueeze(0)

            # Apply context + temperature modulation
            new_params_mod = self._modulate(new_params, context=context, temperature=temperature)

            # EMA update
            if hasattr(self, "_emission_params"):
                self._emission_params.mul_(1 - update_rate).add_(update_rate * new_params_mod)
            if etype in {"categorical", "bernoulli"}:
                self.logits.copy_(self._emission_params)
            elif etype == "poisson":
                self.log_rate.copy_(self._emission_params)

        # Return updated distribution
        return self._get_dist(
            X=X,
            posterior=posterior,
            theta=theta,
            context=context,
            theta_scale=theta_scale,
            temperature=temperature,
            max_jitter=max_jitter,
        )

