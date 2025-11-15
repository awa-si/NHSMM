# nhsmm/distributions/default.py
import math
from collections import OrderedDict
from typing import Optional, Union, Literal, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    # Categorical,
    Normal,
    Bernoulli,
    MultivariateNormal,
    Laplace,
    StudentT,
    Independent,
    Poisson,
)

from nhsmm.constants import DEBUG, DTYPE, EPS, MAX_LOGITS, logger
from nhsmm.tools import constraints


class Categorical(Distribution):
    """
    Drop-in replacement for torch.distributions.Categorical
    with differentiable Gumbel-Softmax sampling.

    Key Guarantees:
    - sample() returns indices with correct shape [...batch]
    - rsample() / sample(hard=False) gives relaxed differentiable sample
    - fully compatible with torch.distributions API
    """

    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }
    support = torch.distributions.constraints.integer_interval(0, 1e12)  # dummy (not enforced)
    has_rsample = True

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        tau: float = 1.0,
        validate_args=None,
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs")

        if probs is not None:
            logits = torch.log(probs.clamp_min(EPS))

        # logits shape [..., K] → batch_shape=[...,], event_shape=[]
        self.logits = logits
        self.tau = tau

        batch_shape = logits.shape[:-1]
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([]), validate_args=validate_args)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    @probs.setter
    def probs(self, p: torch.Tensor):
        self.logits = torch.log(p.clamp_min(EPS))

    @property
    def logits_(self):
        return self.logits

    @logits_.setter
    def logits_(self, l):
        self.logits = l

    # ------------------------------------------------------------------
    # Log prob
    # ------------------------------------------------------------------
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        value shape: [...batch]
        returns: log probs [...batch]
        """
        value = value.long()
        logp = torch.log_softmax(self.logits, dim=-1)
        return logp.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------
    # Internal Gumbel sampling
    # ------------------------------------------------------------------
    def _gumbel_logits(self, sample_shape):
        """
        Returns logits expanded to sample_shape, then adds Gumbel noise.
        """
        batch = self.logits.shape[:-1]
        K = self.logits.shape[-1]

        # correct shape: sample_shape + batch_shape + (K,)
        shape = sample_shape + batch + (K,)

        g = -torch.log(-torch.log(torch.rand(shape, device=self.logits.device).clamp_min(EPS)))
        logits_exp = self.logits.expand(sample_shape + batch + (K,))
        return (logits_exp + g) / self.tau

    # ------------------------------------------------------------------
    # Non-differentiable sample → indices only
    # ------------------------------------------------------------------
    def _sample(self, sample_shape=torch.Size()):
        logits_g = self._gumbel_logits(sample_shape)
        return logits_g.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Differentiable Gumbel-Softmax sample
    # ------------------------------------------------------------------
    def sample(self, sample_shape=torch.Size(), hard: bool = True):
        """
        If hard=True: straight-through estimator (indices but grad flows)
        If hard=False: relaxed Gumbel-Softmax (soft assignment)
        """
        logits_g = self._gumbel_logits(sample_shape)
        y = F.softmax(logits_g, dim=-1)

        if not hard:
            return y  # relaxed, differentiable

        # straight-through hard sample
        y_hard = torch.zeros_like(y)
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard.scatter_(-1, idx, 1.0)

        return (y_hard - y).detach() + y  # ST estimator

    # ------------------------------------------------------------------
    # Reparameterized sample alias (relaxed)
    # ------------------------------------------------------------------
    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape, hard=False)


class Contextual(nn.Module):
    """
    Context-modulated parameter adapter for HSMMs.

    - Supports 1D/2D/3D context shapes
    - Optional temporal Conv1d + spatial Linear adapters
    - Fast O(1) hashing for caching
    - Safe broadcasting for all base/context shapes
    - Optional learnable delta scaling
    - Persistent but stabilized BatchNorm
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: int | None = None,
        hidden_dim: int | None = None,
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
        device: torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.max_delta = max_delta
        self.debug = debug

        # ------------------ Activations ------------------
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        # ------------------ Hidden size ------------------
        hidden_dim = hidden_dim or max(16, target_dim // 2, (context_dim or target_dim))

        # ------------------ Context Encoder ------------------
        self.context_net: nn.Module | None = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim),
            ).to(self.device, DTYPE)
            self._init_weights(self.context_net)

        # ------------------ Projection ------------------
        self._proj: nn.Linear | None = None

        # ------------------ Adapters ------------------
        self.temporal_adapter = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter = (
            nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE)
            if spatial_adapter else None
        )

        for a in [self.temporal_adapter, self.spatial_adapter]:
            if a is not None:
                nn.init.xavier_uniform_(a.weight)

        # ------------------ Learnable scale ------------------
        if learnable_scale:
            self.delta_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE, device=self.device))
        else:
            self.register_buffer("delta_scale", torch.tensor(0.1, dtype=DTYPE))

        # ------------------ Normalization ------------------
        self._batchnorm: nn.BatchNorm1d | None = None

        # ------------------ Caching ------------------
        self.cache_enabled = cache_enabled
        self.cache_grad_safe = cache_grad_safe
        self.cache_limit = cache_limit
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # versioning
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int32))

    # =====================================================================
    # Utility
    # =====================================================================

    def _get_activation(self, name):
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }.get(name.lower(), nn.Identity())

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # =====================================================================
    # Context handling + projection
    # =====================================================================

    def _validate_context(self, context: torch.Tensor | None):
        if context is None:
            return None
        context = context.to(self.device, DTYPE)

        if context.ndim not in (1, 2, 3):
            raise ValueError(f"Expected context dims 1,2,3; got {context.shape}")

        in_dim = context.shape[-1]
        if self.context_dim is None:
            self.context_dim = in_dim

        if in_dim != self.context_dim:
            if not self.allow_projection:
                raise ValueError(f"context_dim mismatch: expected {self.context_dim}, got {in_dim}")

            # build / rebuild projection
            if self._proj is None or self._proj.in_features != in_dim:
                self._proj = nn.Linear(in_dim, self.context_dim, device=self.device, dtype=DTYPE)
                nn.init.xavier_uniform_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
                self._invalidate_cache()

            context = self._proj(context)

        return context

    # =====================================================================
    # Fast stable hashing (O(1))
    # =====================================================================

    @torch.no_grad()
    def _context_hash(self, context):
        """
        Hash = (mean, std, numel, version)
        Fast, stable, independent of tensor size.
        """
        if context is None:
            return f"none-v{int(self._param_version)}"

        c = context.detach().float()
        h = (
            float(c.mean()),
            float(c.std()),
            context.numel(),
            int(self._param_version),
        )
        return str(h)

    # =====================================================================
    # Cache
    # =====================================================================

    @torch.no_grad()
    def _cache_get(self, key):
        if not self.cache_enabled:
            return None
        out = self._cache.get(key)
        if out is not None:
            self._cache.move_to_end(key)
        return out

    @torch.no_grad()
    def _cache_set(self, key, value):
        if not self.cache_enabled:
            return
        self._cache[key] = value if self.cache_grad_safe else value.detach()
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _invalidate_cache(self):
        self._cache.clear()
        self._param_version += 1

    # =====================================================================
    # Δ preparation
    # =====================================================================

    def _prepare_delta(
        self,
        delta: torch.Tensor,
        l2_normalize: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        skip_adapters: bool = False,
    ):
        delta = delta.to(self.device, DTYPE)

        # ------------------ normalizations ------------------
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)

        if layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])

        if batch_norm:
            flat = delta.flatten(0, -2)
            if self._batchnorm is None or self._batchnorm.num_features != flat.shape[-1]:
                # no running stats → stabilizes training for variable-length sequences
                self._batchnorm = nn.BatchNorm1d(flat.shape[-1], affine=True, track_running_stats=False, eps=EPS).to(self.device, DTYPE)
            delta = self._batchnorm(flat).view(delta.shape)

        # ------------------ adapters ------------------
        if not skip_adapters:
            # TEMPORAL adapter expects (B,D,T)
            if self.temporal_adapter is not None:
                delta = delta.transpose(-2, -1)     # (..., D, T)
                delta = self.temporal_adapter(delta)
                delta = delta.transpose(-2, -1)     # back

            if self.spatial_adapter is not None:
                delta = self.spatial_adapter(delta)

        # ------------------ final activation & scaling ------------------
        delta = self.final_activation_fn(delta)
        delta = delta * self.delta_scale
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        delta = torch.nan_to_num(delta, 0.0, 0.0, 0.0)

        return delta

    # =====================================================================
    # Context application
    # =====================================================================

    def _apply_context(
        self,
        base: torch.Tensor,
        context: torch.Tensor | None,
        grad_scale: float | None = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
    ):
        context = self._validate_context(context)
        cache_key = self._context_hash(context)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ------------------ no-context path ------------------
        if context is None:
            out = base
            self._cache_set(cache_key, out)
            return out

        # ------------------ encode ------------------
        if self.context_net:
            delta = self.context_net(context)
        else:
            delta = self._proj(context) if self._proj else 0

        # safe broadcasting against base tensor
        while delta.ndim < base.ndim:
            delta = delta.unsqueeze(0)
        delta = delta.expand_as(base)

        # ------------------ prepare Δ ------------------
        delta = self._prepare_delta(
            delta,
            l2_normalize=l2_normalize,
            skip_adapters=skip_adapters,
        )

        if grad_scale is not None:
            delta = delta * grad_scale

        out = base + delta
        self._cache_set(cache_key, out)
        return out

    # =====================================================================
    # API
    # =====================================================================

    def initialize(self, mode="uniform", **_):
        return self


class Emission(Contextual):
    """Contextual emission distribution supporting Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson."""

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
        self.adaptive_scale = adaptive_scale
        self.modulate_var = modulate_var
        self.min_covar = min_covar
        self.scale = scale
        self.seed = seed
        self.dof = dof

        # Buffers
        self.register_buffer("_emission_means", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        self.register_buffer("_emission_covs", torch.eye(n_features, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states,1,1))
        self.register_buffer("_emission_params", torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))

        # Learnable parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -1.0, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"categorical","bernoulli"}:
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type == "poisson":
            self.log_rate = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        elif self.emission_type in {"laplace","studentt"}:
            self.loc = nn.Parameter(torch.randn(n_states, n_features, dtype=DTYPE, device=self.device) * 0.1)
            self.scale_param = nn.Parameter(torch.full((n_states, n_features), 0.1, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    # ---------------- Utility / Context Methods ----------------
    @torch.no_grad()
    def _spread_means(
        self,
        means: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        n_iter: int = 10,
        min_dist: float = 1e-3,
    ) -> torch.Tensor:
        """Vectorized jitter of means with optional context modulation to ensure minimal separation."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        K, F = means.shape

        # initial jitter
        candidate = means + scale * torch.randn_like(means)
        last_mod = candidate

        for i in range(n_iter):

            # --- FIX: recompute context delta for the *current* candidate ---
            if context is not None:
                delta = self._prepare_delta(candidate, context)
            else:
                delta = 0.0

            candidate_mod = candidate + delta

            # pairwise squared distances (vectorized)
            norms = candidate_mod.pow(2).sum(dim=1, keepdim=True)
            dist_sq = norms + norms.T - 2.0 * (candidate_mod @ candidate_mod.T)
            dist_sq.fill_diagonal_(float("inf"))

            # all states sufficiently separated?
            if torch.all(dist_sq.min(dim=1).values > min_dist):
                return candidate_mod

            # incremental jitter
            jitter_scale = scale * 0.1 * (1 - i / max(1, n_iter - 1))
            candidate = candidate + jitter_scale * torch.randn_like(means)
            last_mod = candidate_mod

        return last_mod

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
        """Estimate emission distribution parameters (continuous or discrete) with optional context/theta."""
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std() < EPS:
                X = X + 1e-3 * torch.randn_like(X)

        # ------------------------------------------------------------------
        # CONTINUOUS
        # ------------------------------------------------------------------
        if etype in {"gaussian", "laplace", "studentt"}:
            means = self._emission_means.clone()

            # weighted means
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)          # [T,K]
                w_sum = w.sum(dim=0) + EPS           # [K]
                means = (w.T @ X) / w_sum.unsqueeze(1)

            # theta adjustment (state-wise broadcasting)
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                means = means + theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # context adjustment
            if context is not None:
                means = self._apply_context(means, context)

            # ---------------- Gaussian ----------------
            if etype == "gaussian":
                if X is not None and posterior is not None:
                    diff = X[:, None, :] - means[None, :, :]      # [T,K,F]
                    weighted = diff * w[:, :, None]               # [T,K,F]
                    covs = torch.einsum("tkf,tkd->kfd", weighted, diff)
                    covs = covs / w_sum[:, None, None]
                else:
                    covs = self._emission_covs.clone()

                # ensure PD
                I = torch.eye(F, device=self.device)
                covs = covs + self.min_covar * I[None, :]
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info == 0:
                            break
                        covs[k] = covs[k] + jitter * I
                        jitter *= 2

                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covariance_matrix=covs)

            # ---------------- Laplace / StudentT ----------------
            else:
                if X is not None and posterior is not None:
                    diff = (X[:, None, :] - means[None, :, :]).abs() * w[:, :, None]
                    scales = diff.sum(dim=0) / w_sum[:, None]
                else:
                    scales = self._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()

                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))

                cls = Laplace if etype == "laplace" else StudentT
                return Independent(cls(loc=means, scale=scales), 1)

        # ------------------------------------------------------------------
        # DISCRETE
        # ------------------------------------------------------------------
        elif etype in {"categorical", "bernoulli", "poisson"}:

            # weighted sufficient stats
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)           # [T,K]
                w_sum = w.sum(dim=0) + EPS            # [K]

                if etype == "categorical":
                    # X ∈ {0..F-1}, count frequencies per state
                    logits = torch.zeros((K, F), device=self.device)
                    for k in range(K):
                        counts = torch.bincount(X.long(), weights=w[:, k], minlength=F)
                        logits[k] = torch.log(counts / counts.sum() + EPS)

                else:  # bernoulli / poisson use weighted mean
                    rate = (w.T @ X.float()) / w_sum[:, None]
                    logits = torch.log(rate.clamp_min(EPS))

            else:
                logits = torch.full((K, F), -math.log(F), dtype=DTYPE, device=self.device)

            # theta adjustment
            if theta is not None:
                theta_vec = theta.mean(dim=0) if theta.ndim > 1 else theta
                logits = logits + theta_scale * theta_vec.unsqueeze(0).expand(K, -1)

            # context modulation
            if context is not None:
                logits = self._apply_context(logits, context)

            # finalize
            if etype == "categorical":
                self._emission_params.copy_(torch.softmax(logits, dim=-1))
                return Categorical(logits=logits)

            elif etype == "bernoulli":
                self._emission_params.copy_(logits)
                return Independent(Bernoulli(logits=logits), 1)

            else:  # poisson
                self._emission_params.copy_(logits)
                return Independent(Poisson(logits=logits), 1)

        # ------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

    # ---------------- Forward / Distribution ----------------
    def _modulate(self, tensor: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply context modulation and optional adaptive scaling.
        Consistent with updated vectorized _apply_context and _adaptive_scale.
        """

        if context is None:
            return tensor

        # First: pure contextual projection (no scaling)
        modulated = self._apply_context(tensor, context)

        # Optional adaptive scaling based on context norm
        if self.adaptive_scale:
            # context: [C] or [..., C]
            if context.ndim == 1:
                # scalar
                norm = context.norm() + EPS
            else:
                # collapse batch/time dims → scalar
                norm = context.norm(dim=-1).mean() + EPS

            scale = self.scale / norm
            modulated = modulated * scale

        return modulated

    def forward(self, context=None, return_dist=False):
        etype = self.emission_type

        if etype == "gaussian":
            mu = self._modulate(self.mu, context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                var += self._modulate(var, context).abs()
            cov = torch.diag_embed(var)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(cov)
            dist = Independent(Normal(loc=mu, scale=var.sqrt()), 1)

        elif etype in {"laplace", "studentt"}:
            loc = self._modulate(self.loc, context)
            scale = torch.clamp(self.scale_param, min=self.min_covar)
            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale ** 2))
            dist_cls = Laplace if etype == "laplace" else StudentT
            dist = Independent(dist_cls(loc=loc, scale=scale) if etype == "laplace" else StudentT(df=self.dof, loc=loc, scale=scale), 1)

        else:
            base_param = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._modulate(base_param, context)
            self._emission_params.copy_(out)

            if etype == "categorical":
                dist = Categorical(logits=out)
            elif etype == "bernoulli":
                dist = Independent(Bernoulli(logits=out), 1)
            else:  # poisson
                # Ensure rate parameter is positive
                rate = torch.exp(out)
                dist = Independent(Poisson(rate), 1)

        if return_dist:
            return dist

        return (self._emission_means, self._emission_covs) if etype in {"gaussian", "laplace", "studentt"} else self._emission_params

    def log_prob(self, x, context=None):
        dist = self.forward(context=context, return_dist=True)
        
        # Ensure proper shape for multivariate distributions
        if isinstance(dist, Independent) and x.ndim == 2 and x.shape[1] == self.n_features:
            x = x.unsqueeze(1)  # [T,1,F] or [B,T,1,F]
        return dist.log_prob(x)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,))
        return samples.to(dtype=DTYPE, device=self.device)

    def parameters_tensor(self):
        if self.emission_type in {"gaussian","laplace","studentt"}:
            return self._emission_means,self._emission_covs
        return self._emission_params

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
        """
        EM-style incremental update of emission parameters.

        Args:
            X: Observations [T,D] or [B,T,D]
            posterior: Soft assignment of states [T,K] or [B,T,K]
            theta: Optional global adjustment tensor
            context: Optional context tensor [T,C] or [B,T,C]
            theta_scale: Scaling for theta adjustment
            update_rate: Interpolation factor (0 = no update, 1 = full update)
            init_spread: Spread factor for initialization jitter
            max_jitter: Maximum jitter iterations for covariance
            rank: Optional low-rank approximation for high-D features
        """
        new_dist = self._estimate_dist(
            X=X,
            posterior=posterior,
            theta=theta,
            context=context,
            theta_scale=theta_scale,
            init_spread=init_spread,
            max_jitter=max_jitter,
        )

        etype = self.emission_type

        # ---------------- Continuous distributions ----------------
        if etype == "gaussian":
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.loc)
            self._emission_covs.mul_(1 - update_rate).add_(update_rate * new_dist.covariance_matrix)
            self.mu.copy_(self._emission_means)
            self.log_var.copy_(
                torch.log(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS))
            )

        elif etype in {"laplace", "studentt"}:
            self._emission_means.mul_(1 - update_rate).add_(update_rate * new_dist.base_dist.loc)
            scale_sq = new_dist.base_dist.scale ** 2
            diag = (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * scale_sq
            self._emission_covs.copy_(torch.diag_embed(diag))
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(diag, min=EPS)))

        # ---------------- Discrete distributions ----------------
        else:
            if etype == "categorical":
                new_logits = new_dist.logits
            elif etype == "bernoulli":
                new_logits = new_dist.base_dist.logits
            else:  # poisson
                new_logits = torch.log(torch.clamp(new_dist.base_dist.rate, min=EPS))

            self._emission_params.mul_(1 - update_rate).add_(update_rate * new_logits)

            param_attr = getattr(self, "logits", getattr(self, "log_rate", None))
            if param_attr is not None:
                param_attr.copy_(self._emission_params)

        return new_dist

    @torch.no_grad()
    def initialize(
        self,
        X: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        mode: str = "data",
        iters: int = 15,
    ):
        """
        Initialize emission parameters and return a Distribution.

        Modes:
            - "default": leave parameters unchanged, return forward() dist
            - "data": compute global mean/cov/logits and duplicate over K states
            - "kmeans": KMeans clustering → per-state mean + covariance
            - "kmeans_ctx": KMeans++ weighted by context norms
        """
        etype = self.emission_type
        K, F = self.n_states, self.n_features

        # Default: just return forward()
        if X is None or mode == "default":
            return self.forward(context=context, return_dist=True)

        # Flatten batch if needed
        Xf = X.reshape(-1, F).to(dtype=DTYPE) if X.ndim == 3 else X.to(dtype=DTYPE)
        device = Xf.device
        N = Xf.shape[0]

        # Helper: KMeans (vectorized)
        def run_kmeans(Xflat, K, iters):
            idx = torch.randperm(Xflat.shape[0], device=device)[:K]
            centers = Xflat[idx].clone()
            for _ in range(iters):
                dist = torch.cdist(Xflat, centers)
                labels = dist.argmin(dim=1)
                for k in range(K):
                    pts = Xflat[labels == k]
                    if pts.shape[0] >= 2:
                        centers[k] = pts.mean(dim=0)
            return centers, labels

        # Helper: Context-weighted KMeans++
        def run_kmeans_ctx(Xflat, Cflat, K, iters):
            w = torch.norm(Cflat, dim=-1)
            w = torch.softmax(w, dim=0)
            idx = torch.multinomial(w, 1)
            centers = [Xflat[idx].clone()]
            for _ in range(K - 1):
                dist2 = torch.stack([torch.norm(Xflat - c, dim=1) for c in centers], dim=1).min(dim=1)[0]
                probs = (dist2 + EPS) * w
                idx = torch.multinomial(probs, 1)
                centers.append(Xflat[idx].clone())
            centers = torch.cat(centers, dim=0)
            for _ in range(iters):
                dist = torch.cdist(Xflat, centers)
                labels = dist.argmin(dim=1)
                for k in range(K):
                    pts = Xflat[labels == k]
                    if pts.shape[0] >= 2:
                        centers[k] = pts.mean(dim=0)
            return centers, labels

        # DATA INIT
        if mode == "data":
            mean = Xf.mean(0)
            xc = Xf - mean
            cov = (xc.T @ xc) / max(N - 1, 1)
            cov = cov + self.min_covar * torch.eye(F, device=device)

            means = mean.expand(K, F).clone()
            covs = cov.unsqueeze(0).expand(K, F, F).clone()

            if etype == "gaussian":
                self.mu.copy_(means)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1)))
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covs)

            elif etype in {"laplace", "studentt"}:
                sc = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                self.loc.copy_(means)
                self.scale_param.copy_(sc)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(sc ** 2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=sc), 1)

            else:
                logits = torch.log_softmax(Xf.mean(0).expand(K, F), dim=-1)
                param = getattr(self, "logits", getattr(self, "log_rate", None))
                param.copy_(logits)
                self._emission_params.copy_(logits)
                if etype == "categorical":
                    return Categorical(logits=logits)
                elif etype == "bernoulli":
                    return Independent(Bernoulli(logits=logits), 1)
                else:
                    return Independent(Poisson(logits), 1)

        # KMEANS INIT
        if mode in {"kmeans", "kmeans_ctx"}:
            if mode == "kmeans":
                centers, labels = run_kmeans(Xf, K, iters)
            else:
                if context is None:
                    raise ValueError("context required for mode='kmeans_ctx'")
                Cflat = context.reshape(-1, context.shape[-1]).to(dtype=DTYPE) if context.ndim == 3 else context.to(dtype=DTYPE)
                centers, labels = run_kmeans_ctx(Xf, Cflat, K, iters)

            means = torch.zeros(K, F, dtype=DTYPE, device=device)
            covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)

            for k in range(K):
                pts = Xf[labels == k]
                if pts.shape[0] < 2:
                    pts = Xf[:min(5, N)]
                mk = pts.mean(0)
                ck = torch.cov(pts.T) + self.min_covar * torch.eye(F, device=device)
                means[k] = mk
                covs[k] = ck

            if etype == "gaussian":
                self.mu.copy_(means)
                self.log_var.copy_(torch.log(torch.diagonal(covs, dim1=-2, dim2=-1)))
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(means, covs)

            elif etype in {"laplace", "studentt"}:
                sc = torch.sqrt(torch.diagonal(covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                self.loc.copy_(means)
                self.scale_param.copy_(sc)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(sc ** 2))
                dist_cls = Laplace if etype == "laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=sc), 1)

            else:
                logits = centers
                param = getattr(self, "logits", getattr(self, "log_rate", None))
                param.copy_(logits)
                self._emission_params.copy_(logits)
                if etype == "categorical":
                    return Categorical(logits=logits)
                elif etype == "bernoulli":
                    return Independent(Bernoulli(logits=logits), 1)
                else:
                    return Independent(Poisson(logits), 1)

        # Fallback: use estimated distribution
        return self._estimate_dist(X=X, context=context)


class Initial(Contextual):
    """
    Contextual initial-state distribution for HSMMs.

    • Supports neural/contextual modulation
    • Temperature scaling
    • Learnable logits + optional neural gating
    • Deterministic caching
    • Fully differentiable using custom Categorical
    """

    def __init__(
        self,
        n_states: int,
        context_dim: int | None = None,
        hidden_dim: int | None = None,
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
            activation="tanh",
            final_activation="tanh",
            cache_enabled=True,
            cache_limit=cache_limit,
            debug=debug,
        )

        self.n_states = n_states
        self.scale = scale
        self.temperature = max(temperature, 1e-6)

        # --------------------------------------------------
        # Logit parameters
        # --------------------------------------------------
        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # --------------------------------------------------
        # Optional neural gate
        # --------------------------------------------------
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h),
                nn.ReLU(),
                nn.Linear(h, n_states),
            )
        else:
            self._context_gate = None

    # ----------------------------------------------------------------------
    # Initialization modes
    # ----------------------------------------------------------------------
    def _init_logits(self, n_states: int, mode: str):
        if mode == "uniform":
            return torch.full((n_states,), -math.log(n_states), dtype=DTYPE, device=self.device)
        if mode == "biased":
            w = torch.linspace(0.8, 0.2, n_states, dtype=DTYPE, device=self.device)
            return torch.log(w / w.sum())
        if mode == "normal":
            return torch.randn(n_states, dtype=DTYPE, device=self.device) * 0.1
        raise ValueError(f"Unknown init_mode: {mode}")

    @torch.no_grad()
    def initialize(self, mode="uniform") -> "Categorical":
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ----------------------------------------------------------------------
    # Context modulation
    # ----------------------------------------------------------------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]):
        if context is None:
            return logits

        # reshape context to [B,T,C]
        if context.ndim == 1:
            context = context.view(1, 1, -1)
        elif context.ndim == 2:
            context = context.unsqueeze(1)
        elif context.ndim != 3:
            raise ValueError(f"Unsupported context shape: {context.shape}")

        B, T, _ = context.shape
        base_logits = logits.view(1, 1, -1).expand(B, T, -1)

        mod = super()._apply_context(base_logits, context)

        if self._context_gate is not None:
            gate = self._context_gate(context)
            mod = mod + gate

        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)))

        return mod

    # ----------------------------------------------------------------------
    # Temperature + normalization + caching
    # ----------------------------------------------------------------------
    def _modulate(self, context=None, temperature=None):
        ctx_key = self._context_hash(context)
        cached = self._cache_get(ctx_key)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, 1e-6)
        mod = self._apply_context(self.logits, context)
        mod = mod / temp
        mod = mod - mod.logsumexp(-1, keepdim=True)
        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

        self._cache_set(ctx_key, mod.detach())
        return mod

    # ----------------------------------------------------------------------
    # Forward: return probs or distribution
    # ----------------------------------------------------------------------
    def forward(self, context=None, log=False, return_dist=False, temperature=None):
        mod = self._modulate(context, temperature)
        if return_dist:
            return Categorical(logits=mod)
        return F.log_softmax(mod, -1) if log else F.softmax(mod, -1)

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------
    def sample(self, context=None, temperature=None):
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        logits = dist.logits
        if logits.ndim == 3:
            B, T, K = logits.shape
            flat = logits.view(B * T, K)
            return torch.multinomial(F.softmax(flat, -1), 1).view(B, T)
        if logits.ndim == 2:
            return torch.multinomial(dist.probs, 1).squeeze(-1)
        return dist.sample(hard=True)

    # ----------------------------------------------------------------------
    # log_prob
    # ----------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, context=None, temperature=None):
        dist = self.forward(context=context, temperature=temperature, return_dist=True)
        if dist.logits.ndim == 3:
            B, T, _ = dist.logits.shape
            return dist.log_prob(x.view(B * T)).view(B, T)
        return dist.log_prob(x)

    # ----------------------------------------------------------------------
    # log_matrix (for forward-backward)
    # ----------------------------------------------------------------------
    def log_matrix(self, context=None, temperature=None):
        logits = self._modulate(context, temperature)
        return F.log_softmax(logits, -1)

    # ----------------------------------------------------------------------
    # EM-style update
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None:
            lr = update_rate if update_rate is not None else 1.0
            logits = self.logits.unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or 1.0), dim=-1)
            loss = -(posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(lr * p.grad)
            self._invalidate_cache()


class Duration(Contextual):
    """
    Contextual categorical duration distribution per state for HSMMs.

    - Supports batch/time-varying contexts [B,T,C]
    - Neural gating, temperature annealing, smoothing, caching
    - Returns a Categorical whose last-dim = max_duration
    """

    def __init__(
        self,
        n_states: int,
        scale: float = 1.0,
        cache_limit: int = 32,
        max_duration: int = 30,
        gate_factor: float = 0.5,
        temperature: float = 1.0,
        min_temperature: float = 1e-6,
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
            cache_enabled=True,
            activation="tanh",
            debug=debug,
        )

        self.scale = scale
        self.n_states = n_states
        self.gate_factor = gate_factor
        self.max_duration = max_duration
        self.min_temperature = min_temperature
        self.smooth_factor = float(smooth_factor)
        self.temperature = max(temperature, min_temperature)

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        self.logits = nn.Parameter(init_logits.clone())  # learnable logits
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Neural gate
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h, device=self.device, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, device=self.device, dtype=DTYPE),
            )
        else:
            self._context_gate = None

        # Durations vector [1..max_duration]
        self.register_buffer(
            "_durations", torch.arange(1, max_duration + 1, dtype=DTYPE, device=self.device)
        )

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            return torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE, device=self.device)
        if mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE, device=self.device)
            w = w.unsqueeze(0).repeat(n_states, 1)
            w /= w.sum(dim=1, keepdim=True)
            return torch.log(w)
        if mode == "normal":
            x = torch.randn(n_states, max_duration, dtype=DTYPE, device=self.device) * 0.1
            x -= torch.arange(max_duration, dtype=DTYPE, device=self.device) * 0.05
            return x
        raise ValueError(f"Unknown init_mode '{mode}'")

    @torch.no_grad()
    def initialize(self, mode: str = "uniform") -> Categorical:
        logits = self._init_logits(self.n_states, self.max_duration, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None:
            context = context.to(self.device, DTYPE)
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)
            elif context.ndim == 2:
                context = context.unsqueeze(1)
            elif context.ndim != 3:
                raise ValueError(f"Unsupported context shape {context.shape}")
            batch_mode = True
        else:
            batch_mode = False

        base = logits
        if batch_mode:
            B, T, _ = context.shape
            base_exp = base.view(1, 1, self.n_states, self.max_duration).expand(B, T, -1, -1)
            mod = super()._apply_context(base_exp, context)
        else:
            mod = super()._apply_context(base, context)

        # Neural gate
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            gate = gate.unsqueeze(-1) if gate.ndim == 3 else gate
            mod = mod + gate * self.gate_factor

        # Smoothing
        if self.smooth_factor > 0:
            smooth_logits = torch.log(torch.ones_like(mod) * self.smooth_factor)
            mod = torch.logsumexp(torch.stack([mod, smooth_logits], dim=-1), dim=-1)

        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

        # update summary buffer
        with torch.no_grad():
            if batch_mode:
                self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)))
            else:
                self._mod_logits_buffer.copy_(mod)

        return mod

    # ---------------- Temperature & caching ----------------
    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, self.min_temperature)
        mod = self._apply_context(self.logits, context) / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)
        self._cache_set(key, mod.detach())
        return mod

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
        logits = dist.logits
        if logits.ndim == 4:
            # vectorized sampling without flattening
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs.reshape(-1, self.max_duration), 1).view(*logits.shape[:3])
        if logits.ndim == 2:
            return torch.multinomial(dist.probs, 1).squeeze(-1)
        return dist.sample()

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        logits = dist.logits
        return dist.log_prob(x.to(torch.long).view(-1)).view(*x.shape)

    # ---------------- Log matrix / expected duration ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        mod = self._mod_logits(context, temperature)
        if mod.ndim == 2:
            return F.log_softmax(mod, dim=-1)
        if mod.ndim == 3:
            B, K, D = mod.shape
            T = 1 if context is None else context.shape[1]
            return F.log_softmax(mod.unsqueeze(1).expand(B, T, K, D), dim=-1)
        if mod.ndim == 4:
            return F.log_softmax(mod, dim=-1)
        raise ValueError(f"Unexpected mod_logits shape {mod.shape}")

    def expected_duration(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, temperature=temperature)
        return torch.sum(probs * self._durations, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, temperature=temperature)
        return torch.argmax(probs, dim=-1) + 1

    # ---------------- EM-style update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            logits = self.logits.unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or 1.0), dim=-1)
            loss = -(posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()


class Transition(Contextual):
    """
    Contextual transition distribution per state for HSMMs.

    - Supports batch/time-varying contexts [B,T,C]
    - Neural gating, temperature annealing, caching
    - Enforces transition_type constraints: ergodic, semi, left-to-right
    - Returns a Categorical whose last-dim = target states
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        transition_type: Union[str, constraints.Transitions] = "ergodic",
        init_mode: str = "diag_bias",
        temperature: float = 1.0,
        gate_factor: float = 0.5,
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

        self.n_states = n_states
        self.transition_type = constraints._resolve_type(transition_type, constraints.Transitions)
        self.temperature = max(temperature, 1e-6)
        self.gate_factor = gate_factor
        self.scale = scale

        init_logits = self._init_logits(n_states, init_mode)
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Optional neural context gate
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, dtype=DTYPE),
            )
        else:
            self._context_gate = None

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
        return self._apply_transition_constraints(logits)

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        mod = super()._apply_context(base, context)

        # Neural gating
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            while gate.ndim < mod.ndim:
                gate = gate.unsqueeze(1)
            mod = mod + gate.unsqueeze(-1) * self.gate_factor

        # Apply structural constraints
        mod = self._apply_transition_constraints(mod)

        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)) if mod.ndim > 2 else mod)
        return mod

    # ---------------- Structural constraints ----------------
    def _apply_transition_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply structural constraints to transition logits according to transition_type.
        Supports arbitrary leading batch/time dimensions.
        """
        shape = logits.shape
        logits = logits.clone()
        n_states = self.n_states

        # Create base masks on [n_states, n_states]
        if self.transition_type == "semi":
            mask = torch.eye(n_states, dtype=torch.bool, device=logits.device)
            logits[..., mask] = -float("inf")
        elif self.transition_type == "left-to-right":
            tril_mask = torch.tril(torch.ones(n_states, n_states, dtype=torch.bool, device=logits.device), -1)
            logits[..., tril_mask] = -float("inf")
        # ergodic -> no constraint

        return logits

    # ---------------- Temperature-annealed logits ----------------
    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, 1e-6)
        mod = self._apply_context(self.logits, context) / temp
        mod = mod - mod.logsumexp(dim=-1, keepdim=True)
        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)
        self._cache_set(key, mod.detach())
        return mod

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
        logits = dist.logits
        if logits.ndim == 3:
            B, T, K = logits.shape
            flat = logits.reshape(B * T, K)
            samp = torch.multinomial(F.softmax(flat, dim=-1), 1).view(B, T)
            return samp
        return dist.sample()

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        logits = dist.logits
        if logits.ndim == 3:
            B, T, K = logits.shape
            x_flat = x.view(-1).to(torch.long)
            return dist.log_prob(x_flat).view(B, T)
        return dist.log_prob(x.to(torch.long))

    # ---------------- Log matrix / Expected transitions ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        mod_logits = self._mod_logits(context, temperature)
        if mod_logits.ndim == 2:
            return F.log_softmax(mod_logits, dim=-1)
        elif mod_logits.ndim == 3:
            B, T, K = mod_logits.shape
            return F.log_softmax(mod_logits.unsqueeze(-2).expand(B, T, self.n_states, K), dim=-1)
        elif mod_logits.ndim == 4:
            return F.log_softmax(mod_logits, dim=-1)
        raise ValueError(f"Unexpected mod_logits shape {mod_logits.shape}")

    def expected_transitions(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return self.forward(context=context, log=False, temperature=temperature)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        return torch.argmax(self.forward(context=context, log=False, temperature=temperature), dim=-1)

    # ---------------- EM / Learnable update ----------------
    @torch.no_grad()
    def update(
        self,
        new_logits: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        from_probs: bool = False,
        update_rate: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        if new_logits is not None:
            if from_probs:
                new_logits = torch.log(new_logits.clamp_min(EPS))
            if temperature is not None:
                new_logits = new_logits / max(temperature, 1e-6)
            new_logits = torch.clamp(new_logits, -MAX_LOGITS, MAX_LOGITS)
            new_logits = self._apply_transition_constraints(new_logits)
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            logits = self.logits.unsqueeze(0) if posterior.ndim == 2 else self.logits.unsqueeze(0).unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or self.temperature), dim=-1)
            loss = - (posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            for p in self.parameters():
                if p.grad is not None:
                    p.data.add_((update_rate or 1.0) * p.grad)
            self._invalidate_cache()

