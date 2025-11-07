# nhsmm/distributions/default.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Categorical, Normal, Bernoulli,
    MultivariateNormal, Laplace, StudentT, Independent, Poisson
)

import math
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger


class Contextual(nn.Module):
    """
    Context-modulated parameter adapter for HSMMs.

    Supports:
    - Optional context encoder or linear projection.
    - Optional temporal (Conv1d) and spatial (Linear) adapters.
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
        cache_enabled: bool = True,
        activation: str = "tanh",
        max_delta: float = 0.5,
        cache_limit: int = 32,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        """Initialize context adapter with optional encoder, adapters, caching, and device/dtype control."""
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

        # Context encoder
        self.context_net: Optional[nn.Module] = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim),
            ).to(self.device, DTYPE)
            self._init_weights(self.context_net)

        # Optional adapters
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

        # Projection fallback
        self._proj: Optional[nn.Linear] = None

        # Caching
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_last_param_sum", torch.tensor(0.0, dtype=DTYPE))

    # ---------------- Utilities ----------------
    def _get_activation(self, name: str) -> nn.Module:
        """Return the activation function for the given name."""
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }.get(name.lower(), nn.Identity())

    def _init_weights(self, module: nn.Module):
        """Initialize Linear layers with Xavier uniform weights and zero bias."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------- Context Handling ----------------
    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Validate and project context tensor.

        Converts context to proper device/dtype and applies linear projection if needed.
        """
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
        """Generate a fast hash string for caching based on context tensor and parameter version."""
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        flat = context.detach().float().sum().item()
        return f"{hash(int(flat * 1e6))}-v{int(self._param_version.item())}"

    @torch.no_grad()
    def _cache_get(self, key: str) -> Any:
        """Retrieve a cached result if available."""
        if not self.cache_enabled:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    @torch.no_grad()
    def _cache_set(self, key: str, value: Any):
        """Store a result in the cache and enforce cache limit."""
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
        """Clear the cache and increment the parameter version."""
        self._cache.clear()
        self._param_version += 1
        if self.debug:
            logger.debug(f"[Contextual] Cache invalidated (v={int(self._param_version.item())})")

    @torch.no_grad()
    def _params_changed(self, param: torch.Tensor, rtol: float = 1e-6) -> bool:
        """
        Check if parameters changed relative to last sum and invalidate cache if so.
        Returns True if cache was invalidated.
        """
        current_sum = float(param.detach().sum())
        last_sum = float(self._last_param_sum.item())
        if abs(current_sum - last_sum) > max(rtol * abs(last_sum), 1e-7):
            self._last_param_sum.copy_(torch.tensor(current_sum, dtype=DTYPE, device=self.device))
            self._invalidate_cache()
            return True
        return False

    # ---------------- Delta Preparation ----------------
    def _prepare_delta(
        self,
        delta: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        scale: float = 0.1,
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        residual: bool = False,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Fully vectorized delta preparation with optional caching of adapters.

        Args:
            delta: (..., F) input tensor
            base: optional tensor to add residual to
            scale: final scaling factor
            grad_scale: optional gradient scaling
            skip_adapters: skip temporal/spatial adapters
            l2_normalize, layer_norm, batch_norm: normalization options
            residual: return delta + base
            use_cache: whether to cache adapter outputs
            cache_key: key to store/retrieve cached adapters

        Returns:
            Processed delta tensor, safely broadcasted.
        """
        delta = delta.to(self.device, DTYPE)

        # ---------------- Normalization ----------------
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if batch_norm:
            flattened = delta.flatten(0, -2)
            bn = nn.BatchNorm1d(flattened.shape[-1], affine=False, eps=EPS).to(self.device, DTYPE)
            delta = bn(flattened).view(delta.shape)

        # ---------------- Adapter Caching ----------------
        if not skip_adapters:
            if use_cache and cache_key is not None:
                if cache_key in self._adapter_cache:
                    delta = self._adapter_cache[cache_key]
                else:
                    if self.temporal_adapter is not None:
                        delta = self.temporal_adapter(delta)
                    if self.spatial_adapter is not None:
                        delta = self.spatial_adapter(delta)
                    self._adapter_cache[cache_key] = delta
            else:
                if self.temporal_adapter is not None:
                    delta = self.temporal_adapter(delta)
                if self.spatial_adapter is not None:
                    delta = self.spatial_adapter(delta)

        # ---------------- Activation & Scaling ----------------
        act_fn = self.final_activation_fn if callable(self.final_activation_fn) else self.final_activation_fn
        delta = act_fn(delta) * scale
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_scale is not None:
            delta = delta * grad_scale

        # ---------------- Optional residual ----------------
        if residual and base is not None:
            delta = delta.expand_as(base)
            delta = delta + base

        # ---------------- Clamp extreme values ----------------
        if self.max_delta is not None:
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)

        return delta

    # ---------------- Core Context Modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        grad_scale: Optional[float] = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
        scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply context modulation to base parameters.

        Returns: base + delta(context), safely broadcasted, optionally cached.
        """
        context = self._validate_context(context)
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if context is None:
            result = base
        else:
            delta = self.context_net(context) if self.context_net else (self._proj(context) if self._proj else 0)
            # Safe broadcasting to base shape
            if delta.shape != base.shape:
                for _ in range(base.ndim - delta.ndim):
                    delta = delta.unsqueeze(0)
                delta = delta.expand_as(base)
            delta = self._prepare_delta(delta, scale=scale, grad_scale=grad_scale,
                                        skip_adapters=skip_adapters, l2_normalize=l2_normalize)
            result = base + delta

        self._cache_set(key, result)
        return result

    # ---------------- API ----------------
    def initialize(self, mode: str = "uniform", **kwargs):
        """Placeholder for parameter initialization; override in subclasses."""
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
        self.dof = dof
        self.seed = seed
        self.scale = scale
        self.k_means = k_means
        self.min_covar = min_covar
        self.modulate_var = modulate_var
        self.adaptive_scale = adaptive_scale

        # Buffers
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

    # ---------------- Utility / Context Methods ----------------
    @torch.no_grad()
    def _spread_means(self, means: torch.Tensor, context: Optional[torch.Tensor] = None,
                      scale: float = 1.0, n_iter: int = 10, min_dist: float = 1e-3) -> torch.Tensor:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        candidate = means + scale * torch.randn_like(means)
        for i in range(n_iter):
            candidate_mod = candidate + (self._prepare_delta(candidate, context) if context is not None else 0)
            dist_sq = torch.cdist(candidate_mod, candidate_mod, p=2)**2
            dist_sq.fill_diagonal_(float('inf'))
            min_pairwise = dist_sq.min(dim=1).values
            if torch.all(min_pairwise > min_dist):
                return candidate_mod
            candidate += scale * 0.1 * (1 - i / n_iter) * torch.randn_like(means)
        return candidate_mod

    def _adapt(self, tensor: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and self.adaptive_scale:
            return tensor * self.scale / (context.norm(dim=-1, keepdim=True) + EPS)
        return tensor

    # ---------------- Distribution Estimation ----------------
    @torch.no_grad()
    def _estimate_dist(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        emission_type: Optional[str] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        init_spread: float = 1.0,
        max_jitter: int = 5,
    ):
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features
        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std() < EPS:
                X += 1e-3 * torch.randn_like(X)

        # Continuous distributions
        if etype in {"gaussian", "laplace", "studentt"}:
            if X is not None and posterior is not None:
                weights = posterior.clamp_min(EPS)
                weights_sum = weights.sum(dim=0, keepdim=True)
                means = (weights.T @ X) / weights_sum.T
                if etype == "gaussian":
                    diff = X.unsqueeze(1) - means.unsqueeze(0)
                    weighted = diff * weights.unsqueeze(-1)
                    covs = torch.einsum("tkf,tkd->kfd", weighted, diff) / weights_sum.T.unsqueeze(-1)
                    covs = covs + self.min_covar * torch.eye(F, device=self.device)
                    for k in range(K):
                        jitter = self.min_covar
                        for _ in range(max_jitter):
                            _, info = torch.linalg.cholesky_ex(covs[k])
                            if info == 0: break
                            covs[k] += jitter * torch.eye(F, device=self.device)
                            jitter *= 2
                    scales = None
                else:
                    scales = ((X.unsqueeze(1) - means.unsqueeze(0)).abs() * weights.unsqueeze(-1)).sum(dim=0) / weights_sum.T
                    scales = scales.clamp_min(self.min_covar)
                    covs = None
            else:
                means = self._emission_means.clone()
                covs = self._emission_covs.clone() if etype == "gaussian" else None
                scales = torch.diagonal(self._emission_covs, dim1=-2, dim2=-1).sqrt() if etype != "gaussian" else None

            if context is not None:
                means = self._apply_context(means, context, self.scale)
            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                means += theta_scale * theta_tensor.expand(K, -1)
            if etype == "gaussian":
                means = self._spread_means(means, scale=init_spread)
                self._emission_means.copy_(means.detach())
                self._emission_covs.copy_(covs.detach())
                return MultivariateNormal(loc=means, covariance_matrix=covs)
            else:
                self._emission_means.copy_(means.detach())
                self._emission_covs.copy_(torch.diag_embed(scales**2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=scales), 1)

        # Discrete distributions
        elif etype in {"categorical", "bernoulli", "poisson"}:
            if X is not None:
                if etype == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    logits = torch.log((counts / counts.sum(-1, keepdim=True)) + EPS)
                else:
                    logits = torch.log(X.float().mean(0, keepdim=True) + EPS).expand(K, -1)
            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                logits += theta_scale * theta_tensor.expand(K, -1)
            if context is not None:
                logits = self._apply_context(logits, context, self.scale)

            self._emission_params.copy_(torch.softmax(logits, dim=-1) if etype == "categorical" else logits)

            if etype == "categorical": return Categorical(logits=logits)
            elif etype == "bernoulli": return Independent(Bernoulli(logits=logits), 1)
            else: return Independent(Poisson(logits), 1)
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, X=None, posterior=None, context=None, theta=None,
                   theta_scale=0.1, init_spread=0.1, max_jitter=5):
        X = X.to(self.device, DTYPE) if X is not None else None
        posterior = posterior.to(self.device, DTYPE) if posterior is not None else None
        context = context.to(self.device, DTYPE) if context is not None else None
        theta = theta.to(self.device, DTYPE) if theta is not None else None

        dist = self._estimate_dist(X=X, posterior=posterior, context=context,
                                   theta=theta, theta_scale=theta_scale,
                                   init_spread=init_spread, max_jitter=max_jitter)

        etype = self.emission_type
        loc = getattr(dist, "loc", None)
        if loc is None and hasattr(dist, "base_dist"):
            loc = getattr(dist.base_dist, "loc", None)
        cov = getattr(dist, "covariance_matrix", None)
        if cov is None and hasattr(dist, "base_dist") and hasattr(dist.base_dist, "scale"):
            cov = dist.base_dist.scale**2

        if loc is not None:
            self._emission_means.copy_(loc)
            if hasattr(self, "mu"): self.mu.copy_(loc)
            elif hasattr(self, "loc"): self.loc.copy_(loc)
        if cov is not None:
            if etype == "gaussian":
                self._emission_covs.copy_(cov)
                if hasattr(self, "log_var"):
                    self.log_var.copy_(torch.log(torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=EPS)))
            else:
                self._emission_covs.copy_(torch.diag_embed(cov))
                if hasattr(self, "scale_param"):
                    self.scale_param.copy_(torch.sqrt(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS)))
        else:
            logits = getattr(dist, "logits", None)
            if logits is None and hasattr(dist, "base_dist"):
                logits = getattr(dist.base_dist, "logits", None)
            if logits is not None:
                self._emission_params.copy_(logits)
                param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
                if param_attr is not None:
                    param_attr.copy_(self._emission_params)
        return dist

    # ---------------- Incremental Update ----------------
    @torch.no_grad()
    def update(
        self,
        X: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        update_rate: float = 0.5,
        init_spread: float = 0.1,
        max_jitter: int = 5,
        rank: Optional[int] = None,
    ):
        if X is None and posterior is None:
            return None

        new_dist = self._estimate_dist(
            X=X, posterior=posterior, theta=theta, context=context,
            theta_scale=theta_scale, init_spread=init_spread, max_jitter=max_jitter
        )

        etype = self.emission_type

        # Continuous
        if etype == "gaussian":
            new_mu = new_dist.loc
            new_cov = new_dist.covariance_matrix
            if rank is not None and rank < new_cov.size(-1):
                eigvals, eigvecs = torch.linalg.eigh(new_cov)
                top_vecs = eigvecs[..., -rank:]
                top_vals = eigvals[..., -rank:]
                new_cov = top_vecs @ torch.diag_embed(top_vals) @ top_vecs.transpose(-2, -1)
                new_cov += EPS * torch.eye(new_cov.size(-1), device=self.device, dtype=DTYPE)
            self._emission_means.copy_((1 - update_rate) * self._emission_means + update_rate * new_mu)
            self._emission_covs.copy_((1 - update_rate) * self._emission_covs + update_rate * new_cov)
            self.mu.copy_(self._emission_means)
            self.log_var.copy_(torch.log(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS)))

        elif etype in {"laplace", "studentt"}:
            new_loc = new_dist.base_dist.loc
            new_scale = new_dist.base_dist.scale
            self._emission_means.copy_((1 - update_rate) * self._emission_means + update_rate * new_loc)
            self._emission_covs.copy_(
                torch.diag_embed((1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * new_scale**2)
            )
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS)))

        # Discrete
        else:
            if etype == "categorical":
                new_logits = new_dist.logits
            elif etype == "bernoulli":
                new_logits = new_dist.base_dist.logits
            else:  # poisson
                new_logits = torch.log(torch.clamp(new_dist.base_dist.rate, min=EPS))
            self._emission_params.copy_((1 - update_rate) * self._emission_params + update_rate * new_logits)
            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    # ---------------- Batched Refit ----------------
    @torch.no_grad()
    def refit(
        self,
        X: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        rank: Optional[int] = None
    ):
        X = X.to(self.device, DTYPE)
        B, T, D = X.shape
        K = self.n_states

        gamma = torch.ones(B, T, K, device=self.device, dtype=DTYPE) if gamma is None else gamma.to(self.device, DTYPE)
        if mask is not None:
            mask = mask.to(self.device, DTYPE)
            gamma = gamma * mask.unsqueeze(-1)

        gamma_sum = gamma.sum(dim=(0, 1), keepdim=True) + EPS
        gamma_norm = gamma / gamma_sum

        X_mod = self.contextual(X, context) if context is not None and hasattr(self, 'contextual') and self.contextual else X

        weighted_X = gamma_norm.unsqueeze(-1) * X_mod.unsqueeze(-2)
        new_mean = weighted_X.sum(dim=(0, 1))
        new_mean = self._spread_means(new_mean, context=context)

        X_centered = X_mod.unsqueeze(-2) - new_mean.unsqueeze(0).unsqueeze(0)
        weighted_centered = X_centered * gamma_norm.unsqueeze(-1)

        if rank is None or rank >= D:
            new_cov = torch.matmul(weighted_centered.transpose(-2, -1), X_centered).sum(dim=0) / gamma_sum.squeeze(0)
        else:
            U = weighted_centered.reshape(B*T, K, D).transpose(0,1)
            cov_lowrank = []
            for k in range(K):
                Y = U[k]
                Q, _ = torch.linalg.qr(Y[:rank].T)
                S = (Y @ Q) / gamma_sum[0,0,k]
                cov_lowrank.append(Q @ S.T + EPS * torch.eye(D, device=self.device, dtype=DTYPE))
            new_cov = torch.stack(cov_lowrank, dim=0)

        new_mean = torch.nan_to_num(new_mean, nan=0.0, posinf=0.0, neginf=0.0)
        new_cov = torch.nan_to_num(new_cov, nan=1.0, posinf=1.0, neginf=1.0)

        if hasattr(self, 'mu'):
            self.mu.data.copy_(new_mean)
            self._emission_means.copy_(new_mean)
        elif hasattr(self, 'loc'):
            self.loc.data.copy_(new_mean)
            self._emission_means.copy_(new_mean)
        self._emission_covs.data.copy_(new_cov)

    # ---------------- Forward / Distribution ----------------
    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        etype = self.emission_type
        if etype == "gaussian":
            mu = self._adapt(self._apply_context(self.mu, context, self.scale), context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                var += self._apply_context(var, context, self.scale).abs()
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
        else:
            base_param = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._adapt(self._apply_context(base_param, context, self.scale), context)
            self._emission_params.copy_(out)
            if etype == "categorical": dist = Categorical(logits=out)
            elif etype == "bernoulli": dist = Independent(Bernoulli(logits=out), 1)
            else: dist = Independent(Poisson(out), 1)

        return dist if return_dist else (self._emission_means, self._emission_covs) if etype in {"gaussian","laplace","studentt"} else self._emission_params

    # ---------------- Log Probability ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        if x.ndim == 2 and isinstance(dist, Independent):
            x = x.unsqueeze(1)
        return dist.log_prob(x)

    # ---------------- Sampling ----------------
    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)

    # ---------------- Raw Parameters ----------------
    def parameters_tensor(self):
        if self.emission_type in {"gaussian", "laplace", "studentt"}:
            return self._emission_means, self._emission_covs
        else:
            return self._emission_params


class Initial(Contextual):
    """
    Contextual initial state distribution for HSMMs.
    Supports differentiable temperature annealing, batch modulation,
    gating, caching, and EM-style updates.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        scale: float = 1.0,
        cache_limit: int = 32,
        debug: bool = False,
    ):
        super().__init__(
            target_dim=n_states,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            cache_limit=cache_limit,
            cache_enabled=True,
            activation="tanh",
            debug=debug,
        )

        self.n_states = n_states
        self.scale = scale
        self.temperature = max(temperature, 1e-6)

        init_logits = self._init_logits(n_states, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        # Optional learnable context gate
        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh()
            ) if context_dim is not None else None
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
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and context.device != self.device:
            context = context.to(self.device)

        batch_mode = context is not None and context.ndim > 1
        mod_logits = logits.unsqueeze(0) if batch_mode else logits
        mod_logits = super()._apply_context(mod_logits, context, scale=self.scale)

        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0)
            mod_logits = mod_logits + gate

        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod_logits.mean(dim=0) if batch_mode else mod_logits)

        return mod_logits

    # ---------------- Temperature-annealed logits ----------------
    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        ctx_hash = self._context_hash(context)
        cached = self._cache_get(ctx_hash)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, 1e-6)
        mod_logits = self._apply_context(self.logits, context)
        # differentiable annealing: logits / temperature
        mod_logits = mod_logits / temp
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
        temperature: Optional[float] = None,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context, temperature)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 2:  # batch mode
            return torch.stack([Categorical(logits=dist.logits[i]).sample() for i in range(dist.logits.shape[0])], dim=0).to(self.device)
        return dist.sample().to(self.device)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 2:
            return torch.stack([Categorical(logits=dist.logits[i]).log_prob(x[i].to(torch.long)) for i in range(dist.logits.shape[0])], dim=0)
        return dist.log_prob(x.to(torch.long))

    # ---------------- Log Matrix ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return F.log_softmax(self._mod_logits(context, temperature), dim=-1)

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: torch.Tensor,
        from_probs: bool = False,
        clamp: bool = True,
        temperature: Optional[float] = None,
    ):
        if new_logits.ndim == 2:  # batch reduction
            new_logits = new_logits.mean(dim=0)

        if from_probs:
            new_logits = torch.log(new_logits + EPS)

        if temperature is not None:
            temp = max(temperature, 1e-6)
            new_logits = new_logits / temp

        if clamp:
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)

        self.logits.data.copy_(new_logits)
        self._logits_buffer.copy_(new_logits)
        self._mod_logits_buffer.copy_(new_logits)
        self._invalidate_cache()


class Duration(Contextual):
    """
    Contextual categorical duration distribution per state for HSMMs.
    Supports batch-compatible modulation, gating, smoothing, caching,
    differentiable temperature annealing, and EM-style updates.
    """

    def __init__(
        self,
        n_states: int,
        scale: float = 1.0,
        cache_limit: int = 32,
        max_duration: int = 30,
        gate_factor: float = 0.5,
        temperature: float = 1.0,
        init_mode: str = "uniform",
        smooth_factor: float = 0.01,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        debug: bool = False,
    ):
        target_dim = n_states * max_duration
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            cache_limit=cache_limit,
            cache_enabled=False,
            activation="tanh",
            debug=debug,
        )

        self.scale = scale
        self.n_states = n_states
        self.max_duration = max_duration
        self.gate_factor = gate_factor
        self.min_temperature = 1e-6
        self.smooth_factor = smooth_factor
        self.temperature = max(temperature, self.min_temperature)

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh()
            ) if context_dim is not None else None
        )

        self.register_buffer("_durations", torch.arange(1, max_duration + 1, dtype=DTYPE, device=self.device))

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            return torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE)
        elif mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE).unsqueeze(0).repeat(n_states, 1)
            w /= w.sum(dim=1, keepdim=True)
            return torch.log(w)
        elif mode == "normal":
            x = torch.randn(n_states, max_duration, dtype=DTYPE) * 0.1
            x = x - torch.arange(max_duration, dtype=DTYPE) * 0.05
            return x
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and context.device != self.device:
            context = context.to(self.device)

        batch_mode = context is not None and context.ndim > 1
        mod_logits = logits.unsqueeze(0) if batch_mode else logits
        mod_logits = super()._apply_context(mod_logits, context, scale=self.scale)

        # reshape to [B, n_states, max_duration] or [n_states, max_duration]
        mod_logits = mod_logits.view(context.shape[0], self.n_states, self.max_duration) if batch_mode else mod_logits.view(self.n_states, self.max_duration)

        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0)
            mod_logits = mod_logits + gate.unsqueeze(-1) * self.gate_factor

        # optional smoothing for stability
        if self.smooth_factor > 0:
            smooth_logits = torch.log(torch.ones_like(mod_logits) * self.smooth_factor)
            mod_logits = torch.logsumexp(torch.stack([mod_logits, smooth_logits], dim=-1), dim=-1)

        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod_logits.mean(dim=0) if batch_mode else mod_logits)

        return mod_logits

    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        ctx_hash = self._context_hash(context)
        cached = self._cache_get(ctx_hash)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, self.min_temperature)
        mod_logits = self._apply_context(self.logits, context) / temp
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
        temperature: Optional[float] = None,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context, temperature)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Log-prob ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 3:
            return torch.stack([Categorical(logits=dist.logits[i]).log_prob(x[i].to(torch.long)) for i in range(dist.logits.shape[0])], dim=0)
        return dist.log_prob(x.to(torch.long))

    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 3:
            return torch.stack([Categorical(logits=dist.logits[i]).sample() for i in range(dist.logits.shape[0])], dim=0).to(self.device)
        return dist.sample().to(self.device)

    # ---------------- Log Matrix / Expected Duration ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return F.log_softmax(self._mod_logits(context, temperature), dim=-1)

    def expected_duration(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, temperature=temperature)
        return torch.sum(probs * self._durations, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return torch.argmax(self.forward(context=context, log=False, temperature=temperature), dim=-1) + 1

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: torch.Tensor,
        from_probs: bool = False,
        clamp: bool = True,
        temperature: Optional[float] = None,
    ):
        if new_logits.ndim == 3:  # batch reduction
            new_logits = new_logits.mean(dim=0)

        if from_probs:
            new_logits = torch.log(new_logits + EPS)

        if temperature is not None:
            temp = max(temperature, self.min_temperature)
            new_logits = new_logits / temp

        if clamp:
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)

        self.logits.data.copy_(new_logits)
        self._logits_buffer.copy_(new_logits)
        self._mod_logits_buffer.copy_(new_logits)
        self._invalidate_cache()


class Transition(Contextual):
    """
    Contextual transition distribution per state for HSMMs.
    Key enhancements:
        - preserves argmax / best-permutation accuracy
        - batch-safe context gating
        - temperature scaling without changing modes
        - EM-style updates with batch reduction
        - caching for repeated contexts
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
        gate_factor: float = 0.5,
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

        self.n_states = n_states
        self.scale = scale
        self.temperature = max(temperature, 1e-6)
        self.gate_factor = gate_factor

        init_logits = self._init_logits(n_states, init_mode)
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())
        self.logits = nn.Parameter(init_logits.clone())

        self._context_gate = (
            nn.Sequential(
                nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
                nn.Tanh()
            ) if context_dim is not None else None
        )

    # ---------------- Initialization ----------------
    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
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
        if context is not None and context.device != self.device:
            context = context.to(self.device)

        batch_mode = context is not None and context.ndim > 1
        mod_logits = logits.unsqueeze(0) if batch_mode else logits
        mod_logits = super()._apply_context(mod_logits, context, scale=self.scale)

        # reshape to [B, n_states, n_states] or [n_states, n_states]
        mod_logits = mod_logits.view(context.shape[0], self.n_states, self.n_states) if batch_mode else mod_logits.view(self.n_states, self.n_states)

        # Context gating (batch-safe)
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0)
            mod_logits = mod_logits + gate.unsqueeze(-1) * self.gate_factor

        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod_logits.mean(dim=0) if batch_mode else mod_logits)

        return mod_logits

    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        ctx_hash = self._context_hash(context)
        cached = self._cache_get(ctx_hash)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, 1e-6)
        mod_logits = self._apply_context(self.logits, context) / temp
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
        temperature: Optional[float] = None,
    ) -> torch.Tensor | Categorical:
        mod_logits = self._mod_logits(context, temperature)
        if return_dist:
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Log-prob ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 3:
            return torch.stack([Categorical(logits=dist.logits[i]).log_prob(x[i].to(torch.long)) for i in range(dist.logits.shape[0])], dim=0)
        return dist.log_prob(x.to(torch.long))

    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        if dist.logits.ndim == 3:
            return torch.stack([Categorical(logits=dist.logits[i]).sample() for i in range(dist.logits.shape[0])], dim=0).to(self.device)
        return dist.sample().to(self.device)

    # ---------------- Log Matrix / Expected Transitions ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return F.log_softmax(self._mod_logits(context, temperature), dim=-1)

    def expected_transitions(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return self.forward(context=context, log=False, temperature=temperature)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return torch.argmax(self.forward(context=context, log=False, temperature=temperature), dim=-1)

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: torch.Tensor,
        from_probs: bool = False,
        clamp: bool = True,
        temperature: Optional[float] = None,
    ):
        # Reduce batch if needed
        if new_logits.ndim == 3:  # [B, n_states, n_states]
            new_logits = new_logits.mean(dim=0)

        if from_probs:
            new_logits = torch.log(new_logits + EPS)

        if temperature is not None:
            temp = max(temperature, 1e-6)
            new_logits = new_logits / temp

        if clamp:
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)

        self.logits.data.copy_(new_logits)
        self._logits_buffer.copy_(new_logits)
        self._mod_logits_buffer.copy_(new_logits)
        self._invalidate_cache()

