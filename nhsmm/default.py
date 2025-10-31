# nhsmm/defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Categorical, Normal, Bernoulli,
    MultivariateNormal, Laplace, StudentT, Independent
)

from sklearn.cluster import KMeans
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

from nhsmm.constants import DTYPE, EPS, MAX_LOGITS, logger


class Contextual(nn.Module):
    """
    Context-modulated parameter adapter with caching and safe delta handling.

    Supports:
    - Single or batch contexts ([D] or [T, D])
    - Optional projection for dimension mismatch
    - Temporal/spatial adapters
    - Activation, scaling, clamping, gradient scaling
    - Gradient-safe caching
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        final_activation: str = "tanh",
        activation: str = "tanh",
        max_delta: float = 0.5,
        cache_enabled: bool = True,
        cache_limit: int = 32,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.aggregate_context = aggregate_context
        self.aggregate_method = aggregate_method
        self.allow_projection = allow_projection
        self.final_activation_name = final_activation
        self.max_delta = max_delta
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.debug = debug

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        # Context network
        self.context_net: Optional[nn.Module] = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=DTYPE),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=DTYPE),
            )
            self._init_weights(self.context_net)

        # Optional adapters
        self.temporal_adapter = (
            nn.Conv1d(target_dim, target_dim, 3, padding=1, bias=False).to(self.device, DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter = (
            nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE)
            if spatial_adapter else None
        )
        if self.temporal_adapter:
            nn.init.xavier_uniform_(self.temporal_adapter.weight)
        if self.spatial_adapter:
            nn.init.xavier_uniform_(self.spatial_adapter.weight)

        # Lazy projection
        self._proj: Optional[nn.Linear] = None

        # Gradient-safe LRU cache
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._param_version = 0

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_projection(self, in_dim: int):
        if not self.allow_projection:
            raise ValueError(f"Context last dimension must be {self.context_dim}, got {in_dim}")
        if self._proj is None:
            out_dim = self.context_dim or self.target_dim
            self._proj = nn.Linear(in_dim, out_dim, device=self.device, dtype=DTYPE)
            nn.init.normal_(self._proj.weight, 0.0, 1e-3)
            nn.init.zeros_(self._proj.bias)
            if self.debug:
                print(f"[Contextual] Added projection {in_dim} → {out_dim}")

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if context.ndim not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D context, got {context.shape}")
        context = context.to(self.device, DTYPE)
        in_dim = context.shape[-1]
        if self.context_dim is None:
            self.context_dim = in_dim
        if in_dim != self.context_dim:
            self._ensure_projection(in_dim)
            context = self._proj(context)
        return context

    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        if context is None:
            return "none"
        ctx = torch.round(context.detach().cpu() * 1e6) / 1e6
        flat = ctx.flatten()
        sample = flat[torch.linspace(0, len(flat) - 1, min(len(flat), 1024), dtype=torch.long)]
        return f"{tuple(ctx.shape)}-{hash(tuple(sample.tolist())) & 0xffffffff}"

    # ---------------- Caching ----------------
    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        if not self.cache_enabled:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_set(self, key: str, value: torch.Tensor):
        if not self.cache_enabled:
            return
        self._cache[key] = value.to(self.device, DTYPE)
        if len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    def _invalidate_cache(self):
        self._cache.clear()
        self._param_version += 1
        if self.debug:
            print(f"[Contextual] cache invalidated (v={self._param_version})")

    # ---------------- Core ----------------
    def _prepare_delta(self, delta: torch.Tensor, scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)

        if self.temporal_adapter:
            delta = self.temporal_adapter(delta.view(1, -1, 1)).view(-1)
        if self.spatial_adapter:
            delta = self.spatial_adapter(delta)

        delta = self._get_activation(self.final_activation_name)(delta) * scale
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        if grad_scale is not None:
            delta = delta * grad_scale
        return delta

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        context = self._validate_context(context)
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if context is None:
            result = base
        else:
            delta = self.context_net(context) if self.context_net else self._proj(context)
            delta = self._prepare_delta(delta, scale=scale, grad_scale=grad_scale)
            result = base + torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
            result = torch.clamp(result, -MAX_LOGITS, MAX_LOGITS)

        self._cache_set(key, result)
        return result

    # ---------------- Public ----------------
    def get_config(self) -> Dict[str, Any]:
        return dict(
            target_dim=self.target_dim,
            context_dim=self.context_dim,
            aggregate_context=self.aggregate_context,
            aggregate_method=self.aggregate_method,
            device=str(self.device),
            dtype=str(DTYPE),
            max_delta=self.max_delta,
            cache_enabled=self.cache_enabled,
            cache_limit=self.cache_limit,
            debug=self.debug,
            temporal_adapter=self.temporal_adapter is not None,
            spatial_adapter=self.spatial_adapter is not None,
            allow_projection=self.allow_projection,
        )

    def to(self, device, **kwargs):
        self.device = torch.device(device)
        super().to(device, **kwargs)
        for m in [self.context_net, self.temporal_adapter, self.spatial_adapter, self._proj]:
            if m is not None:
                m.to(device, **kwargs)
        return self


class Emission(Contextual):
    """Contextual emission distribution supporting Gaussian, Categorical, Bernoulli, Poisson, Laplace, StudentT."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        k_means: bool = True,
        min_covar: float = 1e-6,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
        emission_type: str = "gaussian",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        debug: bool = False,
        scale: float = 0.1,
        dof: float = 5.0,
        seed: int = 0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            target_dim=n_states * n_features,
            aggregate_context=aggregate_context,
            aggregate_method=aggregate_method,
            temporal_adapter=temporal_adapter,
            spatial_adapter=spatial_adapter,
            allow_projection=allow_projection,
            debug=debug,
        )

        self.n_states = n_states
        self.n_features = n_features
        self.k_means = k_means
        self.min_covar = min_covar
        self.modulate_var = modulate_var
        self.adaptive_scale = adaptive_scale
        self.emission_type = emission_type.lower()
        self.dof = dof
        self.seed = seed
        self.scale = scale
        self.device = self.device

        # State gating for context modulation
        self.state_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Sigmoid(),
            )
            if context_dim is not None
            else None
        )

        # Buffers for emission statistics
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        self.register_buffer(
            "_emission_covs",
            torch.eye(n_features, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states, 1, 1),
        )
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, device=self.device, dtype=DTYPE) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, device=self.device, dtype=DTYPE))
        elif self.emission_type in {"categorical", "bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, device=self.device, dtype=DTYPE))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, device=self.device, dtype=DTYPE))
        elif self.emission_type in {"laplace", "studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, device=self.device, dtype=DTYPE) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, device=self.device, dtype=DTYPE))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    # ---------------- Context Modulation ----------------
    def _modulate_per_state(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None or self.state_gate is None:
            return base
        gate = self.state_gate(context)
        if self.adaptive_scale:
            norm = context.norm(dim=-1, keepdim=True) + EPS
            gate = gate * (self.scale / norm)
        if gate.ndim < base.ndim:
            gate = gate.unsqueeze(-1)
        return base * gate

    # ---------------- Mean Spreading ----------------
    @torch.no_grad()
    def _spread_means(self, means: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        jitter = scale * torch.randn_like(means)
        for _ in range(3):
            dists = torch.cdist(means + jitter, means + jitter)
            if torch.all(dists + torch.eye(dists.size(0), device=dists.device) > 1e-3):
                break
            jitter += scale * 0.1 * torch.randn_like(jitter)
        return means + jitter

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta_scale: float = 0.1,
        init_spread: float = 1.0,
        k_means: Optional[bool] = None,
        context: Optional[torch.Tensor] = None,
    ):
        k_means = k_means if k_means is not None else self.k_means
        emission_type = (emission_type or self.emission_type).lower()
        K, F = self.n_states, self.n_features
        device, dtype = self.device, DTYPE

        if X is not None:
            X = X.to(dtype=dtype, device=device)
            if X.std() < EPS:
                X += 1e-3 * torch.randn_like(X)

        # Continuous emissions
        if emission_type in {"gaussian", "laplace", "studentt"}:
            if X is not None and posterior is not None:
                weights_sum = posterior.sum(dim=0, keepdim=True).clamp_min(EPS)
                means = (posterior.T @ X) / weights_sum.T
                if emission_type == "gaussian":
                    covs = torch.zeros(K, F, F, dtype=dtype, device=device)
                    eye_F = torch.eye(F, dtype=dtype, device=device)
                    for k in range(K):
                        w = posterior[:, k:k+1].clamp_min(EPS)
                        diff = X - means[k:k+1]
                        denom = w.sum().clamp_min(EPS)
                        C = ((diff * w).T @ diff) / denom
                        covs[k] = 0.5 * (C + C.T) + self.min_covar * eye_F
                else:
                    scales = torch.zeros(K, F, dtype=dtype, device=device)
                    for k in range(K):
                        w = posterior[:, k:k+1].clamp_min(EPS)
                        diff = (X - means[k:k+1]).abs()
                        scales[k] = ((diff * w).sum(dim=0) / weights_sum[0, k]).clamp_min(self.min_covar)
            else:
                means = self._emission_means.clone()
                if emission_type == "gaussian":
                    covs = self._emission_covs.clone()
                else:
                    scales = torch.sqrt(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)

            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                means += theta_scale * theta_tensor.expand(K, -1)

            if context is not None:
                means = self._modulate_per_state(means, context)

            if emission_type == "gaussian":
                means = self._spread_means(means, scale=init_spread)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(loc=means, covariance_matrix=covs)

            else:
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))
                if emission_type == "laplace":
                    return Independent(Laplace(means, scales), 1)
                else:
                    return Independent(StudentT(df=self.dof, loc=means, scale=scales), 1)

        # Discrete emissions
        elif emission_type in {"categorical", "bernoulli", "poisson"}:
            if X is not None:
                if emission_type == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    params = counts / counts.sum(-1, keepdim=True)
                else:
                    params = X.float().mean(0, keepdim=True).expand(K, -1)
            else:
                params = torch.full((K, F), 1 / F, dtype=dtype, device=device)

            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                params = params + theta_scale * theta_tensor.expand(K, -1)
            if context is not None:
                params = self._modulate_per_state(params, context)

            if emission_type == "categorical":
                params = params.clamp_min(EPS)
                params = params / params.sum(dim=-1, keepdim=True)

            self._emission_params.copy_(params)

            if emission_type == "categorical":
                return Categorical(probs=params)
            elif emission_type == "bernoulli":
                return Independent(Bernoulli(probs=params), 1)
            else:
                return Independent(Poisson(params), 1)

        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")

    # ---------------- Forward ----------------
    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        if self.emission_type == "gaussian":
            mu = self._modulate_per_state(self._apply_context(self.mu, context, self.scale), context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                delta_var = self._modulate_per_state(self._apply_context(var, context, self.scale).abs(), context)
                var = torch.clamp(var + delta_var, min=self.min_covar)
            dist = Independent(Normal(mu, var.sqrt()), 1)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(torch.diag_embed(var))

        elif self.emission_type in {"laplace", "studentt"}:
            loc = self._modulate_per_state(self.loc, context)
            scale = torch.clamp(self.scale_param, min=self.min_covar)
            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale**2))
            dist = Independent(Laplace(loc, scale), 1) if self.emission_type == "laplace" else Independent(StudentT(df=self.dof, loc=loc, scale=scale), 1)

        elif self.emission_type in {"categorical", "bernoulli", "poisson"}:
            base = getattr(self, "logits", getattr(self, "log_rate", None))
            out = F.softplus(base) if self.emission_type == "poisson" else base
            out = self._modulate_per_state(self._apply_context(out, context, self.scale), context)
            self._emission_params.copy_(out)
            if return_dist:
                if self.emission_type == "categorical":
                    dist = Categorical(logits=out)
                elif self.emission_type == "bernoulli":
                    dist = Independent(Bernoulli(logits=out), 1)
                else:
                    dist = Independent(Poisson(out), 1)
            else:
                return out

        if return_dist:
            return dist
        return (self._emission_means, self._emission_covs) if self.emission_type in {"gaussian", "laplace", "studentt"} else self._emission_params

    # ---------------- Log-Prob and Sampling ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, return_dist=True).log_prob(x.to(self.device, DTYPE))

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, return_dist=True).sample((n_samples,)).to(self.device, DTYPE)


class Initial(Contextual):
    """
    Contextual initial state distribution with LRU caching and multiple init modes.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        scale: float = 1.0,
        cache_size: int = 8,
        debug: bool = False,
        init_mode: str = "uniform",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            target_dim=n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation="tanh",
            debug=debug,
            device=device,
            cache_enabled=True,
            cache_limit=cache_size,
        )

        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.debug = debug
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_limit = cache_size

        # ---------------- Initialize logits ----------------
        if init_mode == "uniform":
            logits_init = torch.log(torch.ones(n_states) / n_states)
        elif init_mode == "biased":
            weights = torch.linspace(0.8, 0.2, n_states)
            logits_init = torch.log(weights / weights.sum())
        elif init_mode == "normal":
            logits_init = torch.randn(n_states) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'")

        self.logits = nn.Parameter(logits_init.to(self.device, dtype=DTYPE))

        # ---------------- Cache & param hash ----------------
        self._param_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._last_param_hash = torch.tensor([float(self.logits.detach().sum())], device=self.device, dtype=DTYPE)

        if not hasattr(self, "_cache"):
            self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        """Reset initial state logits and return Categorical distribution."""
        n = self.n_states
        if mode == "uniform":
            logits = torch.log(torch.ones(n, device=self.device, dtype=DTYPE) / n)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, n, device=self.device, dtype=DTYPE)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(n, device=self.device, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.logits.copy_(logits)
        self._cache.clear()
        self._param_cache.clear()
        self._last_param_hash.fill_(logits.sum())
        return Categorical(logits=logits)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _context_hash(self, context: Optional[torch.Tensor]) -> int:
        if context is None:
            return 0
        ctx_flat = torch.round(context.flatten() * 1e6).to(torch.int64)
        return hash(tuple(ctx_flat.tolist()))

    def _params_changed(self) -> bool:
        param_sum = float(self.logits.detach().sum())
        changed = abs(param_sum - float(self._last_param_hash)) > 1e-6
        if changed:
            self._last_param_hash.fill_(param_sum)
            self._param_cache.clear()
            if self.debug:
                logger.debug("[Initial] Parameter change detected, cache cleared")
        return changed

    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor], scale: float) -> torch.Tensor:
        mod = super()._apply_context(logits, context, scale)
        return torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

    # ----------------------------------------------------------------------
    # Core
    # ----------------------------------------------------------------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        log: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor | Categorical:
        self._params_changed()
        ctx_hash = self._context_hash(context)

        # LRU cache lookup
        cached = self._param_cache.get(ctx_hash)
        if cached is not None:
            self._param_cache.move_to_end(ctx_hash)
            return cached if log else cached.exp()

        # Compute fresh logits
        mod_logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        log_probs = F.log_softmax(mod_logits, dim=-1)

        # Maintain bounded LRU cache
        if len(self._param_cache) >= self.cache_limit:
            self._param_cache.popitem(last=False)
        self._param_cache[ctx_hash] = log_probs.detach()

        if return_dist:
            return Categorical(logits=mod_logits)
        return log_probs if log else log_probs.exp()

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True)

    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        return dist.sample().to(self.device)


class Duration(Contextual):
    """
    Contextual categorical duration distribution per state with full LRU caching.
    """

    def __init__(
        self,
        n_states: int,
        max_duration: int = 25,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        scale: float = 1.0,
        debug: bool = False,
        init_mode: str = "uniform",
        cache_limit: int = 32,
        device: Optional[torch.device] = None,
    ):
        target_dim = n_states * max_duration
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation="tanh",
            device=device,
            debug=debug,
            cache_enabled=True,
            cache_limit=cache_limit,
        )

        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_limit = cache_limit
        self.debug = debug

        # ---------------- Initialize logits ----------------
        if init_mode == "uniform":
            logits_init = torch.log(torch.ones(n_states, max_duration) / max_duration)
        elif init_mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration).unsqueeze(0).repeat(n_states, 1)
            logits_init = torch.log(w / w.sum(dim=1, keepdim=True))
        elif init_mode == "normal":
            logits_init = torch.randn(n_states, max_duration) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'")

        self.logits = nn.Parameter(logits_init.to(self.device, dtype=DTYPE))

        # ---------------- Cache & hashes ----------------
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._last_param_hash = torch.tensor([float(self.logits.detach().sum())], device=self.device, dtype=DTYPE)
        self._last_temp_hash = torch.tensor([self.temperature], device=self.device, dtype=DTYPE)
        self._last_scale_hash = torch.tensor([self.scale], device=self.device, dtype=DTYPE)

        self._durations = torch.arange(1, max_duration + 1, device=self.device, dtype=DTYPE)

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        """Reset per-state duration logits and return Categorical distribution."""
        n, d = self.n_states, self.max_duration
        if mode == "uniform":
            logits = torch.log(torch.ones(n, d, device=self.device, dtype=DTYPE) / d)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, d, device=self.device, dtype=DTYPE).unsqueeze(0).repeat(n, 1)
            logits = torch.log(w / w.sum(dim=1, keepdim=True))
        elif mode == "normal":
            logits = torch.randn(n, d, device=self.device, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.logits.copy_(logits)
        self._cache.clear()
        self._last_param_hash.fill_(logits.sum())
        self._last_temp_hash.fill_(self.temperature)
        self._last_scale_hash.fill_(self.scale)
        return Categorical(logits=logits)

    # ---------------- Helpers ----------------
    def _context_hash(self, context: Optional[torch.Tensor]) -> int:
        if context is None:
            return 0
        ctx_flat = torch.round(context.flatten() * 1e6).to(torch.int64)
        return hash(tuple(ctx_flat.tolist()))

    def _params_changed(self) -> bool:
        """Invalidate cache if logits, temperature, or scale changed."""
        param_sum = float(self.logits.detach().sum())
        temp_val = float(self.temperature)
        scale_val = float(self.scale)
        changed = (
            abs(param_sum - float(self._last_param_hash)) > 1e-6
            or abs(temp_val - float(self._last_temp_hash)) > 1e-6
            or abs(scale_val - float(self._last_scale_hash)) > 1e-6
        )
        if changed:
            self._last_param_hash.fill_(param_sum)
            self._last_temp_hash.fill_(temp_val)
            self._last_scale_hash.fill_(scale_val)
            self._cache.clear()
            if self.debug:
                logger.debug("[Duration] Cache invalidated due to parameter change")
        return changed

    # ---------------- Core ----------------
    def _mod_logits(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Return context-modulated logits with LRU caching."""
        self._params_changed()
        ctx_hash = self._context_hash(context)

        cached = self._cache.get(ctx_hash)
        if cached is not None:
            self._cache.move_to_end(ctx_hash)
            return cached

        mod_logits = self._apply_context(self.logits.flatten(), context, self.scale)
        mod_logits = mod_logits.view(self.n_states, self.max_duration) / self.temperature
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        # Maintain bounded LRU cache
        self._cache[ctx_hash] = mod_logits.detach()
        if len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

        return mod_logits

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        log: bool = False,
        return_dist: bool = False
    ) -> Union[torch.Tensor, Categorical]:
        mod_logits = self._mod_logits(context)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.log_softmax(self._mod_logits(context), dim=-1)

    def expected_duration(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False)
        return (probs * self._durations).sum(dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False)
        return torch.argmax(probs, dim=-1) + 1  # durations are 1-based

    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        return dist.sample().to(self.device)


class Transition(Contextual):
    """
    Contextual transition distribution per state with full LRU caching.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        scale: float = 1.0,
        debug: bool = False,
        init_mode: str = "uniform",
        cache_limit: int = 32,
        device: Optional[torch.device] = None,
    ):
        target_dim = n_states * n_states
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation="tanh",
            device=device,
            debug=debug,
            cache_enabled=True,
            cache_limit=cache_limit,
        )

        self.n_states = n_states
        self.cache_limit = cache_limit
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        # ---------------- Initialize logits ----------------
        if init_mode == "uniform":
            logits_init = torch.log(torch.ones(n_states, n_states) / n_states)
        elif init_mode == "diag_bias":
            logits_init = torch.full((n_states, n_states), 0.1, dtype=DTYPE)
            logits_init.fill_diagonal_(0.7)
            logits_init /= logits_init.sum(dim=1, keepdim=True)
            logits_init = torch.log(logits_init)
        elif init_mode == "normal":
            logits_init = torch.randn(n_states, n_states) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'")

        self.logits = nn.Parameter(logits_init.to(self.device, dtype=DTYPE))

        # ---------------- Cache & hashes ----------------
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._last_param_hash = torch.tensor([float(self.logits.detach().sum())], device=self.device, dtype=DTYPE)
        self._last_temp_hash = torch.tensor([self.temperature], device=self.device, dtype=DTYPE)
        self._last_scale_hash = torch.tensor([self.scale], device=self.device, dtype=DTYPE)

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        n = self.n_states
        if mode == "uniform":
            logits = torch.log(torch.ones(n, n, device=self.device, dtype=DTYPE) / n)
        elif mode == "diag_bias":
            logits = torch.full((n, n), 0.1, device=self.device, dtype=DTYPE)
            logits.fill_diagonal_(0.7)
            logits /= logits.sum(dim=1, keepdim=True)
            logits = torch.log(logits)
        elif mode == "normal":
            logits = torch.randn(n, n, device=self.device, dtype=DTYPE) * 0.1
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.logits.copy_(logits)
        self._cache.clear()
        self._last_param_hash.fill_(logits.sum())
        self._last_temp_hash.fill_(self.temperature)
        self._last_scale_hash.fill_(self.scale)

        return Categorical(logits=logits)

    # ---------------- Helpers ----------------
    def _context_hash(self, context: Optional[torch.Tensor]) -> int:
        if context is None:
            return 0
        ctx_flat = torch.round(context.flatten() * 1e6).to(torch.int64)
        return hash(tuple(ctx_flat.tolist()))

    def _params_changed(self) -> bool:
        """Invalidate cache if logits, temperature, or scale changed."""
        param_sum = float(self.logits.detach().sum())
        temp_val = float(self.temperature)
        scale_val = float(self.scale)
        changed = (
            abs(param_sum - float(self._last_param_hash)) > 1e-6
            or abs(temp_val - float(self._last_temp_hash)) > 1e-6
            or abs(scale_val - float(self._last_scale_hash)) > 1e-6
        )
        if changed:
            self._last_param_hash.fill_(param_sum)
            self._last_temp_hash.fill_(temp_val)
            self._last_scale_hash.fill_(scale_val)
            self._cache.clear()
            if self.debug:
                logger.debug("[Transition] Cache invalidated due to parameter change")
        return changed

    # ---------------- Core ----------------
    def _mod_logits(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Return context-modulated logits with LRU caching."""
        self._params_changed()
        ctx_hash = self._context_hash(context)

        cached = self._cache.get(ctx_hash)
        if cached is not None:
            self._cache.move_to_end(ctx_hash)
            return cached

        mod_logits = self._apply_context(self.logits.flatten(), context, self.scale)
        mod_logits = mod_logits.view(self.n_states, self.n_states) / self.temperature
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        self._cache[ctx_hash] = mod_logits.detach()
        if len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

        return mod_logits

    # ---------------- Forward ----------------
    def forward(
        self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False
    ) -> Union[torch.Tensor, Categorical]:
        mod_logits = self._mod_logits(context)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Utility Methods ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.log_softmax(self._mod_logits(context), dim=-1)

    def expected_transitions(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=False)

    def mode(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False)
        return torch.argmax(probs, dim=-1)

    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        return dist.sample()  # LongTensor for state indices

