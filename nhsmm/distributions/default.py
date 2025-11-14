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
        super().__init__(batch_shape=batch_shape,
                         event_shape=torch.Size([]),
                         validate_args=validate_args)

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
    Neural-ready context-modulated parameter adapter for HSMMs.

    Features:
    - Supports 3D time-varying context [B, T, D]
    - Optional temporal Conv1d and spatial Linear adapters
    - Gradient-safe caching
    - Fully batch/time-step aware broadcasting
    - Persistent BatchNorm1d for trainable delta normalization
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: int | None = None,
        hidden_dim: int | None = None,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        final_activation: str = "tanh",
        cache_enabled: bool = True,
        cache_grad_safe: bool = False,
        activation: str = "tanh",
        max_delta: float = 0.5,
        cache_limit: int = 32,
        device: torch.device | None = None,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.allow_projection = allow_projection
        self.max_delta = max_delta
        self.cache_enabled = cache_enabled
        self.cache_grad_safe = cache_grad_safe
        self.cache_limit = cache_limit
        self.debug = debug

        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)
        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)

        # Context encoder
        self.context_net: nn.Module | None = None
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Linear(hidden_dim, target_dim),
            ).to(self.device, DTYPE)
            self._init_weights(self.context_net)

        # Adapters
        self.temporal_adapter: nn.Conv1d | None = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
            if temporal_adapter else None
        )
        self.spatial_adapter: nn.Linear | None = (
            nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE)
            if spatial_adapter else None
        )
        for a in [self.temporal_adapter, self.spatial_adapter]:
            if a is not None:
                nn.init.xavier_uniform_(a.weight)

        self._proj: nn.Linear | None = None
        self._adapter_cache: dict[str, torch.Tensor] = {}
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.register_buffer("_param_version", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_last_param_sum", torch.tensor(0.0, dtype=DTYPE))

        # Persistent batch norm for delta
        self._batchnorm: nn.BatchNorm1d | None = None

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
    def _validate_context(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if context is None:
            return None
        if context.ndim not in (1, 2, 3):
            raise ValueError(f"Expected context ndim 1,2,3; got {context.shape}")
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
    def _context_hash(self, context: torch.Tensor | None) -> str:
        if context is None:
            return f"none-v{int(self._param_version.item())}"
        flat = context.detach().float().view(-1)
        hash_val = int(torch.sum(flat * torch.arange(1, flat.numel() + 1, device=self.device))) % 1_000_000_007
        return f"{hash_val}-v{int(self._param_version.item())}"

    @torch.no_grad()
    def _cache_get(self, key: str) -> torch.Tensor | None:
        if not self.cache_enabled:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    @torch.no_grad()
    def _cache_set(self, key: str, value: torch.Tensor):
        if not self.cache_enabled:
            return
        self._cache[key] = value if self.cache_grad_safe else value.detach().to(self.device, DTYPE)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def _invalidate_cache(self):
        self._cache.clear()
        self._adapter_cache.clear()
        self._param_version += 1
        if self.debug:
            logger.debug(f"[Contextual] Cache invalidated (v={int(self._param_version.item())})")

    # ---------------- Delta Preparation ----------------
    def _prepare_delta(
        self,
        delta: torch.Tensor,
        base: torch.Tensor | None = None,
        scale: float = 0.1,
        grad_scale: float | None = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        residual: bool = False,
        use_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)
        if l2_normalize:
            delta = F.normalize(delta, dim=-1, eps=EPS)
        if layer_norm:
            delta = F.layer_norm(delta, delta.shape[-1:])
        if batch_norm:
            flat = delta.flatten(0, -2)
            if self._batchnorm is None or self._batchnorm.num_features != flat.shape[-1]:
                self._batchnorm = nn.BatchNorm1d(flat.shape[-1], affine=True, eps=EPS).to(self.device, DTYPE)
            delta = self._batchnorm(flat).view(delta.shape)

        # Adapter application
        if not skip_adapters:
            if use_cache and cache_key and cache_key in self._adapter_cache:
                delta = self._adapter_cache[cache_key]
            else:
                if self.temporal_adapter is not None:
                    delta = delta.transpose(-2, -1)
                    delta = self.temporal_adapter(delta)
                    delta = delta.transpose(-2, -1)
                if self.spatial_adapter is not None:
                    delta = self.spatial_adapter(delta)
                if cache_key:
                    self._adapter_cache[cache_key] = delta

        delta = self.final_activation_fn(delta) * scale
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_scale is not None:
            delta = delta * grad_scale
        if residual and base is not None:
            delta = delta + base
        if self.max_delta is not None:
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        return delta

    # ---------------- Context Modulation ----------------
    def _apply_context(
        self,
        base: torch.Tensor,
        context: torch.Tensor | None,
        grad_scale: float | None = None,
        skip_adapters: bool = False,
        l2_normalize: bool = False,
        scale: float = 0.1,
    ) -> torch.Tensor:
        context = self._validate_context(context)
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if context is None:
            result = base
        else:
            delta = self.context_net(context) if self.context_net else (self._proj(context) if self._proj else 0)
            while delta.ndim < base.ndim:
                delta = delta.unsqueeze(0)
            delta = delta.expand_as(base)
            delta = self._prepare_delta(delta, scale=scale, grad_scale=grad_scale, skip_adapters=skip_adapters, l2_normalize=l2_normalize)
            result = base + delta

        self._cache_set(key, result)
        return result

    # ---------------- API ----------------
    def initialize(self, mode: str = "uniform", **kwargs):
        if self.debug:
            logger.debug(f"[Contextual.initialize] mode={mode} (noop)")
        return self


class Emission(Contextual):
    """Contextual emission distribution supporting Gaussian, Laplace, StudentT, Categorical, Bernoulli, Poisson."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        emission_type: str = "gaussian",
        k_means: bool = True,
        min_covar: float = 1e-6,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
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
    def _spread_means(self, means: torch.Tensor, context: Optional[torch.Tensor] = None,
                      scale: float = 1.0, n_iter: int = 10, min_dist: float = 1e-3) -> torch.Tensor:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        candidate = means + scale * torch.randn_like(means)
        for i in range(n_iter):
            candidate_mod = candidate + (self._prepare_delta(candidate, context) if context is not None else 0)
            dist_sq = torch.cdist(candidate_mod, candidate_mod, p=2)**2
            dist_sq.fill_diagonal_(float('inf'))
            if torch.all(dist_sq.min(dim=1).values > min_dist):
                return candidate_mod
            candidate += scale * 0.1 * (1 - i / n_iter) * torch.randn_like(means)
        return candidate_mod

    def _adapt(self, tensor: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and self.adaptive_scale:
            return tensor * self.scale / (context.norm(dim=-1, keepdim=True) + EPS)
        return tensor

    # ---------------- Distribution Estimation ----------------
    @torch.no_grad()
    def _estimate_dist(self, X: Optional[torch.Tensor] = None, posterior: Optional[torch.Tensor] = None,
                       emission_type: Optional[str] = None, theta: Optional[torch.Tensor] = None,
                       context: Optional[torch.Tensor] = None, theta_scale: float = 0.1,
                       init_spread: float = 1.0, max_jitter: int = 5):
        etype = emission_type or self.emission_type
        K, F = self.n_states, self.n_features

        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std() < EPS:
                X += 1e-3 * torch.randn_like(X)

        # Continuous distributions
        if etype in {"gaussian","laplace","studentt"}:
            means = self._emission_means.clone()
            if X is not None and posterior is not None:
                w = posterior.clamp_min(EPS)
                w_sum = w.sum(dim=0, keepdim=True)
                means = (w.transpose(0,1) @ X) / w_sum.transpose(0,1)
            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim==2 else theta
                means += theta_scale*theta_tensor.expand(K,-1)
            if context is not None:
                means = self._apply_context(means, context, self.scale)

            if etype=="gaussian":
                if X is not None and posterior is not None:
                    diff = X.unsqueeze(1)-means.unsqueeze(0)
                    weighted = diff * w.unsqueeze(-1)
                    covs = torch.einsum("tkf,tkd->kfd", weighted, diff)/w_sum.transpose(0,1).unsqueeze(-1)
                else:
                    covs = self._emission_covs.clone()
                covs += self.min_covar * torch.eye(F, device=self.device).unsqueeze(0)
                for k in range(K):
                    jitter = self.min_covar
                    for _ in range(max_jitter):
                        _, info = torch.linalg.cholesky_ex(covs[k])
                        if info==0: break
                        covs[k]+=jitter*torch.eye(F,device=self.device)
                        jitter*=2
                self._emission_means.copy_(means)
                self._emission_covs.copy_(covs)
                return MultivariateNormal(loc=means, covariance_matrix=covs)
            else:
                scales = ((X.unsqueeze(1)-means.unsqueeze(0)).abs()*w.unsqueeze(-1)).sum(dim=0)/w_sum.transpose(0,1) if X is not None else self._emission_covs.diagonal(dim1=-2,dim2=-1).sqrt()
                scales = scales.clamp_min(self.min_covar)
                self._emission_means.copy_(means)
                self._emission_covs.copy_(torch.diag_embed(scales**2))
                dist_cls = Laplace if etype=="laplace" else StudentT
                return Independent(dist_cls(loc=means, scale=scales),1)

        # Discrete distributions
        elif etype in {"categorical","bernoulli","poisson"}:
            if X is not None:
                if etype=="categorical":
                    counts = torch.stack([torch.bincount(X[:,f].long(),minlength=F) for f in range(F)],dim=1).T.float()
                    logits = torch.log((counts/counts.sum(-1,keepdim=True))+EPS)
                elif etype=="bernoulli":
                    logits = torch.log(X.float().mean(0,keepdim=True)+EPS).expand(K,-1)
                else: # poisson
                    logits = torch.log(X.float().mean(0,keepdim=True)+EPS).expand(K,-1)
            else:
                logits = torch.full((K,F), -math.log(F), dtype=DTYPE, device=self.device)

            if theta is not None:
                theta_tensor = theta.mean(dim=0,keepdim=True) if theta.ndim==2 else theta
                logits += theta_scale*theta_tensor.expand(K,-1)
            if context is not None:
                logits = self._apply_context(logits, context, self.scale)

            self._emission_params.copy_(torch.softmax(logits, dim=-1) if etype=="categorical" else logits)
            if etype=="categorical": return Categorical(logits=logits)
            elif etype=="bernoulli": return Independent(Bernoulli(logits=logits),1)
            else: return Independent(Poisson(logits),1)
        else:
            raise ValueError(f"Unsupported emission_type: {etype}")

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
            X: Observations [B,T,D]
            posterior: Soft assignment of states [B,T,K]
            theta: Optional global adjustment tensor
            context: Optional context tensor [B,T,C]
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

        if etype == "gaussian":
            self._emission_means.copy_((1 - update_rate) * self._emission_means + update_rate * new_dist.loc)
            self._emission_covs.copy_((1 - update_rate) * self._emission_covs + update_rate * new_dist.covariance_matrix)
            self.mu.copy_(self._emission_means)
            self.log_var.copy_(torch.log(torch.clamp(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS)))
        elif etype in {"laplace", "studentt"}:
            self._emission_means.copy_((1 - update_rate) * self._emission_means + update_rate * new_dist.base_dist.loc)
            self._emission_covs.copy_(torch.diag_embed(
                (1 - update_rate) * self._emission_covs.diagonal(dim1=-2, dim2=-1) + update_rate * new_dist.base_dist.scale ** 2
            ))
            self.loc.copy_(self._emission_means)
            self.scale_param.copy_(torch.sqrt(torch.clamp(
                torch.diagonal(self._emission_covs, dim1=-2, dim2=-1), min=EPS
            )))
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

    # ---------------- Forward / Distribution ----------------
    def forward(self, context=None, return_dist=False):
        etype = self.emission_type
        if etype=="gaussian":
            mu = self._adapt(self._apply_context(self.mu, context, self.scale), context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var: var += self._apply_context(var, context, self.scale).abs()
            cov = torch.diag_embed(var)
            self._emission_means.copy_(mu)
            self._emission_covs.copy_(cov)
            dist = Independent(Normal(mu,var.sqrt()),1)
        elif etype in {"laplace","studentt"}:
            loc = self._adapt(self._apply_context(self.loc, context, self.scale), context)
            scale = torch.clamp(self.scale_param, min=self.min_covar)
            self._emission_means.copy_(loc)
            self._emission_covs.copy_(torch.diag_embed(scale**2))
            dist = Independent(Laplace(loc,scale),1) if etype=="laplace" else Independent(StudentT(df=self.dof,loc=loc,scale=scale),1)
        else:
            base_param = getattr(self, "logits", getattr(self, "log_rate", None))
            out = self._adapt(self._apply_context(base_param, context, self.scale), context)
            self._emission_params.copy_(out)
            dist = Categorical(logits=out) if etype=="categorical" else Independent(Bernoulli(logits=out),1) if etype=="bernoulli" else Independent(Poisson(out),1)
        if return_dist: return dist
        return (self._emission_means,self._emission_covs) if etype in {"gaussian","laplace","studentt"} else self._emission_params

    def log_prob(self,x,context=None):
        dist = self.forward(context=context,return_dist=True)
        if x.ndim==2 and isinstance(dist,Independent): x=x.unsqueeze(1)
        return dist.log_prob(x)

    def sample(self,n_samples=1,context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)

    def parameters_tensor(self):
        if self.emission_type in {"gaussian","laplace","studentt"}:
            return self._emission_means,self._emission_covs
        return self._emission_params


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

        self.logits = nn.Parameter(init_logits.clone())  # learnable core
        self.register_buffer("_logits_buffer", init_logits.clone())  # raw baseline
        self.register_buffer("_mod_logits_buffer", init_logits.clone())  # context-mod baseline

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
            return torch.full(
                (n_states,), -math.log(n_states), dtype=DTYPE, device=self.device
            )
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

        # Normalize shapes
        if context.ndim == 1:  # [C]
            context = context.unsqueeze(0).unsqueeze(0)  # [1,1,C]
        elif context.ndim == 2:  # [B,C]
            context = context.unsqueeze(1)  # [B,1,C]
        elif context.ndim != 3:  # [B,T,C]
            raise ValueError(f"Unsupported context shape: {context.shape}")

        B, T, _ = context.shape
        base_logits = logits.view(1, 1, -1).expand(B, T, -1)  # [B,T,K]

        # Contextual modulation from superclass
        mod = super()._apply_context(base_logits, context, scale=self.scale)

        # Optional neural gate
        if self._context_gate is not None:
            gate = self._context_gate(context)  # [B,T,K]
            mod = mod + gate

        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

        # Cache "average" representation
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)))

        return mod

    # ----------------------------------------------------------------------
    # Temperature + normalization + caching
    # ----------------------------------------------------------------------
    def _mod_logits(self, context=None, temperature=None):
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
        mod = self._mod_logits(context, temperature)
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
            x_flat = x.view(B * T)
            return dist.log_prob(x_flat).view(B, T)

        return dist.log_prob(x)

    # ----------------------------------------------------------------------
    # log_matrix (for forward-backward)
    # ----------------------------------------------------------------------
    def log_matrix(self, context=None, temperature=None):
        logits = self._mod_logits(context, temperature)
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

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            logits = self.logits.unsqueeze(0) if posterior.ndim == 1 else self.logits.unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or 1.0), dim=-1)
            loss = - (posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(update_rate or 1.0 * p.grad)
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
        self.max_duration = max_duration
        self.gate_factor = gate_factor
        self.min_temperature = 1e-6
        self.smooth_factor = float(smooth_factor)
        self.temperature = max(temperature, self.min_temperature)

        init_logits = self._init_logits(n_states, max_duration, init_mode)
        # learnable logits shaped [n_states, max_duration]
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # optional neural gate -> produces [B,T,n_states]
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h, device=self.device, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, device=self.device, dtype=DTYPE),
            )
        else:
            self._context_gate = None

        # durations vector [1..max_duration]
        self.register_buffer("_durations", torch.arange(1, max_duration + 1, dtype=DTYPE, device=self.device))

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, max_duration: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            return torch.full((n_states, max_duration), -math.log(max_duration), dtype=DTYPE, device=self.device)
        if mode == "short_bias":
            w = torch.linspace(0.7, 0.3, max_duration, dtype=DTYPE, device=self.device).unsqueeze(0).repeat(n_states, 1)
            w = w / w.sum(dim=1, keepdim=True)
            return torch.log(w)
        if mode == "normal":
            x = torch.randn(n_states, max_duration, dtype=DTYPE, device=self.device) * 0.1
            x = x - torch.arange(max_duration, dtype=DTYPE, device=self.device) * 0.05
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
        # logits expected shape [n_states, max_duration] or expanded first
        if context is not None and context.device != self.device:
            context = context.to(self.device, DTYPE)

        # normalize context to [B,T,C] for consistent broadcasting
        if context is None:
            batch_mode = False
        else:
            if context.ndim == 1:
                context = context.unsqueeze(0).unsqueeze(0)  # [1,1,C]
            elif context.ndim == 2:
                context = context.unsqueeze(1)  # [B,1,C]
            elif context.ndim != 3:
                raise ValueError(f"Unsupported context shape {context.shape}")
            batch_mode = True

        base = logits
        # expand base to [B,T,n_states,max_duration] when batch_mode else [n_states,max_duration]
        if batch_mode:
            B, T, _ = context.shape
            base_exp = base.view(1, 1, self.n_states, self.max_duration).expand(B, T, -1, -1)
            mod = super()._apply_context(base_exp, context, scale=self.scale)  # expects shape broadcastable
            # result shape [B,T,n_states,max_duration]
        else:
            # pass through contextual adapters (no-op if none)
            mod = super()._apply_context(base, context, scale=self.scale)  # [n_states, max_duration]

        # Neural gating: gate produces [B,T,n_states] (logit additive)
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)  # [B,T,n_states]
            if not batch_mode:
                # shouldn't happen but keep safe
                gate = gate.unsqueeze(1)
            # add gate across duration axis
            mod = mod + gate.unsqueeze(-1) * self.gate_factor

        # smoothing (adds a small uniform mass via log-sum-exp)
        if self.smooth_factor and self.smooth_factor > 0:
            smooth_logits = torch.log(torch.ones_like(mod) * float(self.smooth_factor))
            mod = torch.logsumexp(torch.stack([mod, smooth_logits], dim=-1), dim=-1)

        mod = torch.clamp(mod, -MAX_LOGITS, MAX_LOGITS)

        # update summary buffers (mean across B,T if batch_mode)
        with torch.no_grad():
            if batch_mode:
                self._mod_logits_buffer.copy_(mod.mean(dim=(0, 1)))
            else:
                self._mod_logits_buffer.copy_(mod)

        return mod

    # ---------------- Temperature-annealed logits & caching ----------------
    def _mod_logits(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        key = self._context_hash(context)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        temp = max(temperature if temperature is not None else self.temperature, self.min_temperature)
        mod = self._apply_context(self.logits, context) / temp

        # normalize along duration axis (last dim)
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
            # Categorical expects last-dim = categories (durations)
            return Categorical(logits=mod_logits)
        return F.log_softmax(mod_logits, dim=-1) if log else F.softmax(mod_logits, dim=-1)

    # ---------------- Sampling / Log-prob ----------------
    def sample(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        logits = dist.logits  # shape: [n_states,max_duration] or [B,T,n_states,max_duration]

        if logits.ndim == 4:
            B, T, K, D = logits.shape
            flat = logits.view(B * T * K, D)
            samp = torch.multinomial(F.softmax(flat, dim=-1), 1).view(B, T, K)
            return samp  # integer durations in [1..D] (indexes)
        if logits.ndim == 2:
            return torch.multinomial(dist.probs, 1).squeeze(-1)
        # fallback to distribution sample
        return dist.sample()

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True, temperature=temperature)
        logits = dist.logits

        if logits.ndim == 4:
            B, T, K, D = logits.shape
            # accept x shaped (B, T, K) or (B, K) or flattened
            x_flat = x.view(-1).to(torch.long)
            lp = dist.log_prob(x_flat).view(B, T, K)
            return lp
        # logits ndim 2 -> [n_states, max_duration] or [B, n_states]? dist.log_prob handles leading dims
        return dist.log_prob(x.to(torch.long))

    # ---------------- Log matrix / expected duration ----------------
    def log_matrix(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Returns log probabilities shaped for forward-backward:
         - If no context: [n_states, max_duration]
         - If context [B,T,C]: returns [B, T, n_states, max_duration]
        """
        if context is None:
            return F.log_softmax(self._mod_logits(None, temperature), dim=-1)

        # normalize shapes to [B,T,C]
        if context.ndim == 1:
            context = context.unsqueeze(0).unsqueeze(0)
        elif context.ndim == 2:
            context = context.unsqueeze(0)
        elif context.ndim != 3:
            raise ValueError(f"Unsupported context shape {context.shape}")

        B, T, _ = context.shape
        mod = self._mod_logits(context, temperature)

        # mod can be [n_states,max_duration], [B,T,n_states,max_duration], or [B,n_states,max_duration]
        if mod.ndim == 2:
            return F.log_softmax(mod, dim=-1)
        if mod.ndim == 3:
            # [B, n_states, max_duration] -> expand over T
            return F.log_softmax(mod.unsqueeze(1).expand(B, T, -1, -1), dim=-1)
        if mod.ndim == 4:
            return F.log_softmax(mod, dim=-1)
        raise ValueError(f"Unexpected mod_logits shape {mod.shape}")

    def expected_duration(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, temperature=temperature)
        # probs shape: [n_states, D] or [B,T,n_states,D]
        return torch.sum(probs * self._durations, dim=-1)

    def mode(self, context: Optional[torch.Tensor] = None, temperature: Optional[float] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, temperature=temperature)
        return torch.argmax(probs, dim=-1) + 1  # +1 maps index -> duration

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
            self.logits.data.copy_(new_logits)
            self._logits_buffer.copy_(new_logits)
            self._mod_logits_buffer.copy_(new_logits)
            self._invalidate_cache()
            return

        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            logits = self.logits.unsqueeze(0) if posterior.ndim == 2 else self.logits.unsqueeze(0).unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or 1.0), dim=-1)
            loss = - (posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(update_rate or 1.0 * p.grad)
            self._invalidate_cache()


class Transition(Contextual):
    """
    Contextual transition distribution per state for HSMMs.
    
    - Supports batch/time-varying contexts [B,T,C]
    - Neural gating, temperature annealing, caching
    - Returns a Categorical whose last-dim = target states
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
        self.logits = nn.Parameter(init_logits.clone())
        self.register_buffer("_logits_buffer", init_logits.clone())
        self.register_buffer("_mod_logits_buffer", init_logits.clone())

        # Optional neural context gate
        if context_dim is not None:
            h = hidden_dim or 64
            self._context_gate = nn.Sequential(
                nn.Linear(context_dim, h, device=self.device, dtype=DTYPE),
                nn.ReLU(),
                nn.Linear(h, n_states, device=self.device, dtype=DTYPE),
            )
        else:
            self._context_gate = None

    # ---------------- Initialization ----------------
    def _init_logits(self, n_states: int, mode: str) -> torch.Tensor:
        if mode == "uniform":
            return torch.full((n_states, n_states), -math.log(n_states), dtype=DTYPE, device=self.device)
        if mode == "diag_bias":
            m = torch.full((n_states, n_states), 0.1, dtype=DTYPE, device=self.device)
            m.fill_diagonal_(0.7)
            m /= m.sum(dim=1, keepdim=True)
            return torch.log(m)
        if mode == "normal":
            return torch.randn(n_states, n_states, dtype=DTYPE, device=self.device) * 0.1
        raise ValueError(f"Unknown init_mode '{mode}'")

    @torch.no_grad()
    def initialize(self, mode: str = "diag_bias") -> Categorical:
        logits = self._init_logits(self.n_states, mode)
        self.logits.data.copy_(logits)
        self._logits_buffer.copy_(logits)
        self._mod_logits_buffer.copy_(logits)
        self._invalidate_cache()
        return Categorical(logits=logits)

    # ---------------- Contextual modulation ----------------
    def _apply_context(self, logits: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is not None and context.device != self.device:
            context = context.to(self.device, DTYPE)

        batch_mode = context is not None and context.ndim > 1
        mod_logits = super()._apply_context(logits if not batch_mode else logits.unsqueeze(0), context, scale=self.scale)

        # Neural gating
        if self._context_gate is not None and context is not None:
            gate = self._context_gate(context)
            if gate.ndim < mod_logits.ndim:
                gate = gate.unsqueeze(1)
            mod_logits = mod_logits + gate.unsqueeze(-1) * self.gate_factor

        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)
        with torch.no_grad():
            self._mod_logits_buffer.copy_(mod_logits.mean(dim=0) if batch_mode else mod_logits)
        return mod_logits

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
            flat = logits.view(B * T, K)
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
        else:
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
        """
        Neural + EM-ready update of transition logits.
        - `new_logits`: direct logits (EM-style)
        - `posterior`: expected transitions (batch x n_states x n_states)
        - `from_probs`: whether new_logits is already a probability
        - `update_rate`: scale for gradient-based update
        """
        # EM-style logits
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

        # Neural / gradient-based update
        if posterior is not None and any(p.requires_grad for p in self.parameters()):
            # posterior shape: [B,T,n_states,n_states] or [n_states,n_states]
            logits = self.logits.unsqueeze(0) if posterior.ndim == 2 else self.logits.unsqueeze(0).unsqueeze(0)
            log_probs = F.log_softmax(logits / (temperature or self.temperature), dim=-1)
            loss = - (posterior * log_probs).sum() / posterior.sum()
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.data.add_(update_rate or 1.0 * p.grad)
            self._invalidate_cache()

