# nhsmm/distributions/default.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Categorical, Normal, Bernoulli,
    MultivariateNormal, Laplace, StudentT, Independent
)

import math
import hashlib
from sklearn.cluster import KMeans
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger


class Contextual(nn.Module):
    """
    Context-modulated parameter adapter for HSMMs.

    Features:
    - Optional context encoder or linear projection.
    - Optional temporal and spatial adapters.
    - Per-context caching with LRU behavior.
    - Safe GPU and dtype handling.
    - Delta scaling with final activation.
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
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
        self.allow_projection = allow_projection
        self.max_delta = max_delta
        self.cache_enabled = cache_enabled
        self.cache_limit = cache_limit
        self.debug = debug

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)

        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        # ---------------- Context encoder ----------------
        self.context_net: Optional[nn.Module] = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=DTYPE),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=DTYPE),
            )
            self._init_weights(self.context_net)

        # ---------------- Optional adapters ----------------
        self.temporal_adapter: Optional[nn.Conv1d] = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter: Optional[nn.Linear] = (
            nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE)
            if spatial_adapter else None
        )
        for a in [self.temporal_adapter, self.spatial_adapter]:
            if a is not None:
                nn.init.xavier_uniform_(a.weight)

        # ---------------- Projection ----------------
        self._proj: Optional[nn.Linear] = None

        # ---------------- Caching ----------------
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_last_param_sum", torch.tensor(0.0, dtype=DTYPE))

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

    # ---------------- Context Handling ----------------
    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Ensure context is correct shape and projected if needed."""
        if context is None:
            return None
        if context.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D context, got {context.shape}")

        context = context.to(self.device, DTYPE)
        in_dim = context.shape[-1]

        if self.context_dim is None:
            self.context_dim = in_dim

        if in_dim != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"Expected context_dim={self.context_dim}, got {in_dim}")
            if self._proj is None or self._proj.in_features != in_dim:
                self._proj = nn.Linear(in_dim, self.context_dim or self.target_dim, device=self.device, dtype=DTYPE)
                nn.init.normal_(self._proj.weight, 0.0, 1e-3)
                nn.init.zeros_(self._proj.bias)
                self._invalidate_cache()
                if self.debug:
                    logger.debug(f"[Contextual] Added projection {in_dim} → {self._proj.out_features}")
            context = self._proj(context)
        return context

    # ---------------- Caching ----------------
    @torch.no_grad()
    def _context_hash(self, context: Optional[torch.Tensor]) -> str:
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        ctx_bytes = context.detach().cpu().numpy().tobytes()
        return f"{hashlib.md5(ctx_bytes).hexdigest()}-v{int(self._param_version.item())}"

    @torch.no_grad()
    def _cache_get(self, key: str) -> Any:
        if not self.cache_enabled:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    @torch.no_grad()
    def _cache_set(self, key: str, value: Any):
        if not self.cache_enabled:
            return
        if isinstance(value, torch.Tensor):
            self._cache[key] = value.detach().to(self.device, DTYPE)
        else:
            self._cache[key] = value
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _invalidate_cache(self):
        self._cache.clear()
        self._param_version += 1
        if self.debug:
            logger.debug(f"[Contextual] Cache invalidated (v={int(self._param_version.item())})")

    @torch.no_grad()
    def _params_changed(self, param: torch.Tensor) -> bool:
        """Invalidate cache if parameters changed."""
        current_sum = float(param.detach().sum())
        if abs(current_sum - float(self._last_param_sum.item())) > 1e-7:
            self._last_param_sum.copy_(torch.tensor(current_sum, dtype=DTYPE, device=self.device))
            self._invalidate_cache()
            return True
        return False

    # ---------------- Delta Preparation ----------------
    def _prepare_delta(
        self,
        delta: torch.Tensor,
        scale: float = 0.1,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False
    ) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)

        # Apply adapters
        if not skip_adapters:
            if self.temporal_adapter is not None:
                x = delta.unsqueeze(0).transpose(1, 2) if delta.ndim == 2 else delta[None, :, None]
                delta = self.temporal_adapter(x).transpose(1, 2).squeeze(0)
            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        delta = self.final_activation_fn(delta) * scale
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_scale is not None:
            delta = delta * grad_scale
        return delta

    # ---------------- Core Context Modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        scale: float = 0.1,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
    ) -> torch.Tensor:
        """
        Apply context modulation:
        base + delta(context) -> adapters -> final activation -> scaling
        """
        context = self._validate_context(context)
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if context is None:
            result = base
        else:
            delta = self.context_net(context) if self.context_net else self._proj(context)
            # Expand delta for broadcasting if needed
            while delta.ndim < base.ndim:
                delta = delta.unsqueeze(0)
            delta = self._prepare_delta(delta, scale=scale, grad_scale=grad_scale, skip_adapters=skip_adapters)
            result = base + delta

        self._cache_set(key, result)
        return result

    # ---------------- API ----------------
    def initialize(self, mode: str = "uniform", **kwargs):
        """No-op by default. Override in derived classes."""
        if self.debug:
            logger.debug(f"[Contextual.initialize] mode={mode} (noop)")
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
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        debug: bool = False,
        scale: float = 0.1,
        dof: float = 5.0,
        seed: int = 0,
    ):
        target_dim = n_states * n_features
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
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

        # Cached buffers
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        self.register_buffer("_emission_covs", torch.eye(n_features, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states,1,1))
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"categorical", "bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"laplace", "studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    @torch.no_grad()
    def _spread_means(self, means: torch.Tensor, scale: float = 1.0, n_iter: int = 5) -> torch.Tensor:
        if self.seed is not None: torch.manual_seed(self.seed)
        K, F = means.shape
        candidate = means + scale * torch.randn_like(means)
        for _ in range(n_iter):
            diff = candidate.unsqueeze(0) - candidate.unsqueeze(1)
            dist_sq = (diff ** 2).sum(-1)
            dist_sq.fill_diagonal_(float('inf'))
            if torch.all(dist_sq.min(dim=1).values > 1e-3):
                return candidate
            candidate += 0.1 * scale * torch.randn_like(means)
        return candidate

    def _adapt(self, tensor: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and self.adaptive_scale:
            return tensor * self.scale / (context.norm(dim=-1, keepdim=True) + EPS)
        return tensor

    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        init_spread: float = 1.0,
    ):
        etype = (emission_type or self.emission_type).lower()
        K, F = self.n_states, self.n_features
        X = X.to(dtype=DTYPE, device=self.device) if X is not None else None
        if X is not None and X.std() < EPS: X += 1e-3 * torch.randn_like(X)

        if etype in {"gaussian", "laplace", "studentt"}:
            # Weighted initialization
            if X is not None and posterior is not None:
                weights = posterior.clamp_min(EPS)
                weights_sum = weights.sum(dim=0, keepdim=True)
                means = (weights.T @ X) / weights_sum.T
                if etype == "gaussian":
                    diff = X.unsqueeze(1) - means.unsqueeze(0)
                    weighted = diff * weights.unsqueeze(-1)
                    covs = torch.einsum("tkf,tkd->kfd", weighted, diff) / weights_sum.T.unsqueeze(-1)
                    covs = 0.5 * (covs + covs.transpose(-1, -2)) + self.min_covar * torch.eye(F, device=self.device)
                else:
                    scales = ((X.unsqueeze(1) - means.unsqueeze(0)).abs() * weights.unsqueeze(-1)).sum(dim=0) / weights_sum.T
                    scales = scales.clamp_min(self.min_covar)
            else:
                means = self._emission_means.clone()
                covs = self._emission_covs.clone() if etype == "gaussian" else torch.diag_embed(torch.sqrt(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar))

            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                means += theta_scale * theta_tensor.expand(K, -1)
            if context is not None:
                means = self._apply_context(means, context, self.scale)

            if etype == "gaussian":
                means = self._spread_means(means, scale=init_spread)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(loc=means, covariance_matrix=covs)
            else:
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales ** 2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=scales) if etype=="laplace" else Independent(StudentT(df=self.dof, loc=means, scale=scales), 1), 1)

        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None:
                if etype == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    params = counts / counts.sum(-1, keepdim=True)
                else:
                    params = X.float().mean(0, keepdim=True).expand(K, -1)
            else:
                params = torch.full((K, F), 1 / F, dtype=DTYPE, device=self.device)

            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                params += theta_scale * theta_tensor.expand(K, -1)
            if context is not None:
                params = self._apply_context(params, context, self.scale)

            if etype == "categorical":
                params = params.clamp_min(EPS)
                params /= params.sum(dim=-1, keepdim=True)
            self._emission_params.copy_(params)

            if etype == "categorical":
                return Categorical(probs=params)
            elif etype == "bernoulli":
                return Independent(Bernoulli(probs=params), 1)
            else:
                return Independent(Poisson(params), 1)

        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        """Always returns a distribution if return_dist=True, else returns raw parameters."""
        etype = self.emission_type

        if etype == "gaussian":
            mu = self._adapt(self._apply_context(self.mu, context, self.scale), context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var: var = torch.clamp(var + self._apply_context(var, context, self.scale).abs(), min=self.min_covar)
            cov = torch.diag_embed(var)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(cov)
            dist = Independent(Normal(mu, var.sqrt()), 1)

        elif etype in {"laplace", "studentt"}:
            loc = self._adapt(self._apply_context(self.loc, context, self.scale), context)
            scale = torch.clamp(self.scale_param, min=self.min_covar)
            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale**2))
            dist = Independent(Laplace(loc, scale), 1) if etype == "laplace" else Independent(StudentT(df=self.dof, loc=loc, scale=scale), 1)

        else:  # categorical, bernoulli, poisson
            base = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._adapt(self._apply_context(base, context, self.scale), context)
            self._emission_params.copy_(out)
            if etype == "categorical":
                dist = Categorical(logits=out)
            elif etype == "bernoulli":
                dist = Independent(Bernoulli(logits=out), 1)
            else:
                dist = Independent(Poisson(out), 1)

        if return_dist:
            return dist
        else:
            if etype in {"gaussian", "laplace", "studentt"}:
                return self._emission_means, self._emission_covs
            else:
                return self._emission_params

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        if x.ndim == 2 and isinstance(dist, Independent):
            x = x.unsqueeze(1)  # match [L, 1, F]
        return dist.log_prob(x)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)

    def parameters_tensor(self):
        """Returns raw emission parameters as tensor for DP fallback."""
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        else:
            return self._emission_params


class Initial(Contextual):
    """
    Contextual initial state distribution for HSMMs with caching and context modulation.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        scale: float = 1.0,
        cache_size: int = 32,
        debug: bool = False,
    ):
        super().__init__(
            target_dim=n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            cache_limit=cache_size,
            activation="tanh",
            cache_enabled=True,
            debug=debug,
        )

        self.scale = scale
        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)

        # ---------------- Logits as parameter and buffer ----------------
        init_logits = self._init_logits(n_states, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        # Optional context gate
        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh(),
            )
            if context_dim is not None
            else None
        )

    # ---------------- Initialization modes ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            return torch.full((n_states,), -math.log(n_states), dtype=DTYPE, device=self.device)
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, n_states, dtype=DTYPE, device=self.device)
            return torch.log(w / w.sum())
        elif mode == "normal":
            return torch.randn(n_states, dtype=DTYPE, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        """Reset logits buffer and parameter."""
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply context modulation with optional gate and clamping."""
        mod = super()._apply_context(logits, context, scale=self.scale)
        if self._context_gate is not None and context is not None:
            mod = mod + self._context_gate(context)
        return torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

    def _mod_logits(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Return context-modulated logits with caching and temperature scaling."""
        self._params_changed(self.logits)
        ctx_hash = self._context_hash(context)
        cached = self._cache.get(ctx_hash)
        if cached is not None:
            self._cache.move_to_end(ctx_hash)
            return cached

        mod_logits = self._apply_context(self.logits, context) / self.temperature
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        self._cache_set(ctx_hash, mod_logits.detach())
        return mod_logits

    # ---------------- Forward / Distribution ----------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        log: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling ----------------
    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, return_dist=True).sample().to(self.device)

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True)


class Duration(Contextual):
    """
    Contextual categorical duration distribution per state for HSMMs.
    Supports batch-compatible context modulation, per-state gates, and caching.
    """

    def __init__(
        self,
        n_states: int,
        max_duration: int = 30,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        cache_limit: int = 32,
        scale: float = 1.0,
        debug: bool = False,
    ):
        target_dim = n_states * max_duration
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            cache_limit=cache_limit,
            cache_enabled=True,
            activation="tanh",
            debug=debug,
        )

        self.scale = scale
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)

        # ---------------- Logits as parameter and buffer ----------------
        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        # Optional per-state context gate
        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh(),
            )
            if context_dim is not None
            else None
        )

        # Duration indices buffer for expected duration calculation
        self.register_buffer("_durations", torch.arange(1, max_duration + 1, device=self.device, dtype=DTYPE))

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        """Reset logits to initialization mode and clear cache."""
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        """Generate initial logits per mode."""
        if mode == "uniform":
            return torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE, device=self.device)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states, 1)
            w /= w.sum(dim=1, keepdim=True)
            return torch.log(w)
        elif mode == "normal":
            return torch.randn(n_states, max_duration, dtype=DTYPE, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply context modulation with optional per-state gating and clamping."""
        mod = super()._apply_context(logits.flatten(), context, scale=self.scale)
        mod = mod.view(self.n_states, self.max_duration)

        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0)
            mod = mod + gate.unsqueeze(-1) * 0.5  # stabilized per-state gating

        return torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

    def _mod_logits(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Return context-modulated logits with caching and normalization."""
        ctx_hash = self._context_hash(context)
        cached = self._cache.get(ctx_hash)
        if cached is not None:
            self._cache.move_to_end(ctx_hash)
            return cached

        mod_logits = self._apply_context(self.logits, context) / self.temperature
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        self._cache_set(ctx_hash, mod_logits.detach())
        return mod_logits

    # ---------------- Forward / Distribution ----------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        log: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Diagnostics ----------------
    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw a sample from the context-modulated duration distribution."""
        return self.forward(context=context, return_dist=True).sample().to(self.device)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log-probabilities of observed durations x."""
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return log-probabilities for all durations and states."""
        return F.log_softmax(self._mod_logits(context), dim=-1)

    def expected_duration(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute expected duration per state under the modulated distribution."""
        probs = self.forward(context=context, log=False)
        return torch.sum(probs * self._durations, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the most likely duration per state."""
        return torch.argmax(self.forward(context=context, log=False), dim=-1) + 1


class Transition(Contextual):
    """
    Contextual transition distribution per state for HSMMs.
    Supports batch-compatible context modulation, per-row gates, and caching.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
        cache_limit: int = 32,
        scale: float = 1.0,
        debug: bool = False,
    ):
        target_dim = n_states * n_states
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            cache_limit=cache_limit,
            cache_enabled=True,
            activation="tanh",
            debug=debug,
        )

        self.scale = scale
        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)

        # ---------------- Logits as parameter and buffer ----------------
        init_logits = self._init_logits(n_states, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        # Optional per-row context gate
        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh(),
            )
            if context_dim is not None
            else None
        )

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        """Reset logits and return a categorical transition distribution."""
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        """Generate initial logits per mode."""
        if mode == "uniform":
            return torch.full((n_states, n_states), -math.log(n_states), dtype=DTYPE, device=self.device)
        elif mode == "diag_bias":
            m = torch.full((n_states, n_states), 0.1, dtype=DTYPE, device=self.device)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=1, keepdim=True)
            return torch.log(m)
        elif mode == "normal":
            return torch.randn(n_states, n_states, dtype=DTYPE, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply context modulation with optional per-row gating and clamping."""
        mod = super()._apply_context(logits.flatten(), context, scale=self.scale)
        mod = mod.view(self.n_states, self.n_states)

        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0)
            mod = mod + gate.unsqueeze(-1) * 0.5  # stabilized per-row gating

        return torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

    def _mod_logits(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Return context-modulated logits with caching and normalization."""
        ctx_hash = self._context_hash(context)
        cached = self._cache.get(ctx_hash)
        if cached is not None:
            self._cache.move_to_end(ctx_hash)
            return cached

        mod_logits = self._apply_context(self.logits, context) / self.temperature
        mod_logits = mod_logits - mod_logits.logsumexp(dim=-1, keepdim=True)
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        self._cache_set(ctx_hash, mod_logits.detach())
        return mod_logits

    # ---------------- Forward / Distribution ----------------
    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        log: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Diagnostics ----------------
    def sample(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw a sample from the context-modulated transition distribution."""
        return self.forward(context=context, return_dist=True).sample().to(self.device)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log-probabilities of observed transitions x."""
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    def log_matrix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return log-probabilities for all transitions."""
        return F.log_softmax(self._mod_logits(context), dim=-1)

    def expected_transitions(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute expected transition probabilities per state under modulated distribution."""
        return self.forward(context=context, log=False)

    def mode(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the most likely next state per row."""
        return torch.argmax(self.forward(context=context, log=False), dim=-1)

