# nhsmm/defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli, MultivariateNormal, Laplace, StudentT, Independent

import logging
from sklearn.cluster import KMeans
from typing import Optional, Union, Literal, Tuple, Dict, Any

EPS = 1e-12
MAX_LOGITS = 50.0
DTYPE = torch.float32

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("nhsmm")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)  # default level; change to DEBUG if needed
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class HSMMError(ValueError):
    """Custom error class for HSMM module."""
    pass


class Contextual(nn.Module):
    """
    Context-modulated parameter adapter with safe delta handling.

    Supports:
      - Single-sample or sequence-level contexts ([D] or [T, D])
      - Optional context projection if dimension mismatch
      - Temporal and spatial adapters for delta modulation
      - Activation, scaling, clamping, and gradient scaling
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: Optional[int],
        aggregate_context: bool = True,
        hidden_dim: Optional[int] = None,
        aggregate_method: str = "mean",
        final_activation: str = "tanh",
        activation: str = "tanh",
        device: Optional[torch.device] = None,
        max_delta: float = 0.5,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = target_dim
        self.context_dim = context_dim
        self.aggregate_context = aggregate_context
        self.aggregate_method = aggregate_method
        self.allow_projection = allow_projection
        self.max_delta = max_delta
        self.final_activation_name = final_activation
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

        # Adapters
        self.temporal_adapter = nn.Conv1d(target_dim, target_dim, 3, padding=1, bias=False).to(self.device, DTYPE) if temporal_adapter else None
        self.spatial_adapter = nn.Linear(target_dim, target_dim, bias=False).to(self.device, DTYPE) if spatial_adapter else None
        if self.temporal_adapter: nn.init.xavier_uniform_(self.temporal_adapter.weight)
        if self.spatial_adapter: nn.init.xavier_uniform_(self.spatial_adapter.weight)

        self._proj: Optional[nn.Linear] = None

    # ---------------- Helpers ----------------
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
                logger.debug(f"[Contextual] Added projection {in_dim} → {out_dim}")

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if context.ndim not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D context, got shape {context.shape}")
        context = context.to(self.device, DTYPE)
        in_dim = context.shape[-1]
        if self.context_dim is None:
            self.context_dim = in_dim
        if in_dim != self.context_dim:
            self._ensure_projection(in_dim)
            context = self._proj(context)
        return context

    def _prepare_delta(self, delta: torch.Tensor, scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)

        if self.temporal_adapter:
            delta_ = delta.view(1, -1, 1)  # single-sample
            delta_ = self.temporal_adapter(delta_)
            delta = delta_.view(-1)

        if self.spatial_adapter:
            delta = self.spatial_adapter(delta)

        activation = self._get_activation(self.final_activation_name)
        delta = activation(delta) * scale

        if self.max_delta is not None:
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)

        if grad_scale is not None:
            delta = delta * grad_scale

        return delta

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        if context is None:
            return base
        context = self._validate_context(context)
        delta = self.context_net(context) if self.context_net else self._proj(context)
        delta = self._prepare_delta(delta, scale=scale, grad_scale=grad_scale)
        result = base + torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        return torch.clamp(result, -MAX_LOGITS, MAX_LOGITS)

    # ---------------- Public ----------------
    def get_config(self) -> Dict[str, Any]:
        return dict(
            context_dim=self.context_dim,
            target_dim=self.target_dim,
            aggregate_context=self.aggregate_context,
            aggregate_method=self.aggregate_method,
            device=str(self.device),
            dtype=str(DTYPE),
            debug=self.debug,
            temporal_adapter=self.temporal_adapter is not None,
            spatial_adapter=self.spatial_adapter is not None,
            allow_projection=self.allow_projection,
            max_delta=self.max_delta,
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

        self.dof = dof
        self.seed = seed
        self.scale = scale
        self.k_means = k_means
        self.n_states = n_states
        self.min_covar = min_covar
        self.n_features = n_features
        self.modulate_var = modulate_var
        self.emission_type = emission_type
        self.adaptive_scale = adaptive_scale
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State gating for context modulation
        self.state_gate = (
            nn.Sequential(nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE), nn.Sigmoid())
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
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    # ---------------- Context Modulation ----------------
    def _modulate_per_state(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None or self.state_gate is None:
            return base
        gate = self.state_gate(context)
        if self.adaptive_scale:
            norm = context.norm() + EPS
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

        if X is not None:
            X = X.to(dtype=DTYPE, device=self.device)
            if X.std() < EPS:
                X += 1e-3 * torch.randn_like(X)

        # --- Gaussian ---
        if emission_type == "gaussian":
            if X is not None:
                if posterior is not None:
                    norm = posterior.sum(0, keepdim=True).T.clamp_min(EPS)
                    means = (posterior.T @ X) / norm
                    covs = torch.stack(
                        [
                            ((posterior[:, k:k+1] * (X - means[k]).unsqueeze(0)).T @ (X - means[k]).unsqueeze(0)) / norm[k]
                            + self.min_covar * torch.eye(F, device=self.device, dtype=DTYPE)
                            for k in range(K)
                        ]
                    )
                elif k_means:
                    km = KMeans(n_clusters=K, n_init=10, random_state=self.seed)
                    km.fit(X.cpu().numpy())
                    labels = torch.tensor(km.labels_, dtype=torch.long, device=self.device)
                    means = torch.tensor(km.cluster_centers_, dtype=DTYPE, device=self.device)
                    covs = torch.zeros(K, F, F, dtype=DTYPE, device=self.device)
                    for k in range(K):
                        cluster_points = X[labels == k]
                        if cluster_points.shape[0] <= 1:
                            covs[k] = torch.eye(F, device=self.device, dtype=DTYPE) * self.min_covar
                        else:
                            centered = cluster_points - cluster_points.mean(0, keepdim=True)
                            covs[k] = (centered.T @ centered) / (cluster_points.shape[0] - 1) + torch.eye(F, device=self.device, dtype=DTYPE) * self.min_covar
                else:
                    means = X.mean(0, keepdim=True).expand(K, -1)
                    covs = torch.stack([((X - X.mean(0))**2).mean(0).diag() + self.min_covar for _ in range(K)])
            else:
                means = self._emission_means.clone()
                covs = self._emission_covs.clone()

            if theta is not None:
                means += theta_scale * theta.unsqueeze(0).expand(K, -1)
            means = self._spread_means(means, scale=init_spread)
            if context is not None:
                means = self._modulate_per_state(means, context)
            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)
            return MultivariateNormal(means, covs)

        # --- Categorical/Bernoulli/Poisson ---
        elif emission_type in {"categorical", "bernoulli", "poisson"}:
            if X is not None:
                if emission_type == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    params = counts / counts.sum(-1, keepdim=True)
                else:
                    params = X.float().mean(0, keepdim=True).expand(K, -1)
            else:
                params = torch.full((K, F), 1 / F, dtype=DTYPE, device=self.device)
            if context is not None:
                params = self._modulate_per_state(params, context)
            self._emission_params.copy_(params)
            return params

        # --- Laplace / StudentT ---
        elif emission_type in {"laplace", "studentt"}:
            if X is not None:
                if posterior is not None:
                    norm = posterior.sum(0, keepdim=True).T.clamp_min(EPS)
                    means = (posterior.T @ X) / norm
                    scales = torch.stack([((posterior[:, k:k+1] * (X - means[k]).abs()).sum(0) / norm[k]) for k in range(K)])
                elif k_means:
                    km = KMeans(n_clusters=K, n_init=10, random_state=self.seed)
                    km.fit(X.cpu().numpy())
                    labels = torch.tensor(km.labels_, dtype=torch.long, device=self.device)
                    means = torch.tensor(km.cluster_centers_, dtype=DTYPE, device=self.device)
                    scales = torch.zeros(K, F, dtype=DTYPE, device=self.device)
                    for k in range(K):
                        cluster_points = X[labels == k]
                        scales[k] = cluster_points.std(0, unbiased=True).clamp_min(EPS) if cluster_points.shape[0] > 0 else torch.ones(F, device=self.device) * EPS
                else:
                    means = X.mean(0, keepdim=True).expand(K, -1)
                    scales = X.std(0, keepdim=True).expand(K, -1).clamp_min(EPS)
            else:
                means = self._emission_means.clone()
                scales = torch.ones_like(means)

            self._emission_means.copy_(means)
            self._emission_covs.copy_(torch.diag_embed(scales**2))
            return Independent(Laplace(means, scales), 1) if emission_type == "laplace" else Independent(StudentT(df=self.dof, loc=means, scale=scales), 1)

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
            return Independent(Normal(mu, var.sqrt()), 1) if return_dist else (mu, var)

        if self.emission_type in {"categorical", "bernoulli", "poisson"}:
            base = getattr(self, "logits", getattr(self, "log_rate", None))
            out = F.softplus(base) if self.emission_type == "poisson" else base
            out = self._modulate_per_state(self._apply_context(out, context, self.scale), context)
            if return_dist:
                if self.emission_type == "categorical":
                    return Categorical(logits=out)
                if self.emission_type == "bernoulli":
                    return Independent(Bernoulli(logits=out), 1)
                return Independent(Poisson(out), 1)
            return out

    # ---------------- Log-Prob and Sampling ----------------
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, return_dist=True).log_prob(x.to(self.device, DTYPE))

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, return_dist=True).sample((n_samples,)).to(self.device, DTYPE)


class Duration(Contextual):
    """
    Contextual categorical duration distribution per state with buffer caching.
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
        )

        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale

        # Initialize logits
        if init_mode == "uniform":
            logits_init = torch.log(torch.ones(n_states, max_duration) / max_duration)
        elif init_mode == "short_bias":
            logits_init = torch.log(
                torch.linspace(0.7, 0.3, max_duration).unsqueeze(0).repeat(n_states, 1)
            )
        elif init_mode == "normal":
            logits_init = torch.randn(n_states, max_duration) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'")

        self.logits = nn.Parameter(logits_init.to(self.device, dtype=DTYPE))

        # --- Buffers for caching ---
        self.register_buffer("_cached_logits", None)
        self.register_buffer("_cached_softmax", None)
        self.register_buffer("_cached_log_softmax", None)
        self.register_buffer("_cached_dist", None)
        self.register_buffer("_cached_context_hash", None)

    # --- Context hash for caching ---
    def _context_hash(self, context: Optional[torch.Tensor]):
        if context is None:
            return None
        return context.detach().cpu().sum().item()

    # --- Context-modulated logits with caching ---
    def _mod_logits(self, context: Optional[torch.Tensor]):
        ctx_hash = self._context_hash(context)

        if (self._cached_logits is not None) and (self._cached_context_hash == ctx_hash):
            return self._cached_logits

        mod_logits = self._apply_context(self.logits.flatten(), context, self.scale)
        mod_logits = mod_logits.view(self.n_states, self.max_duration) / self.temperature
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        # Cache and invalidate dependent buffers
        self._cached_logits = mod_logits
        self._cached_context_hash = ctx_hash
        self._cached_softmax = None
        self._cached_log_softmax = None
        self._cached_dist = None

        return mod_logits

    # --- Forward: return logits, softmax, or distribution ---
    def forward(self, context=None, log: bool = False, return_dist: bool = False):
        mod_logits = self._mod_logits(context)

        if return_dist:
            if self._cached_dist is None:
                self._cached_dist = Categorical(logits=mod_logits)
            return self._cached_dist

        if log:
            if self._cached_log_softmax is None:
                self._cached_log_softmax = F.log_softmax(mod_logits, dim=-1)
            return self._cached_log_softmax

        if self._cached_softmax is None:
            self._cached_softmax = F.softmax(mod_logits, dim=-1)
        return self._cached_softmax

    # --- Log-probability of samples ---
    def log_prob(self, x: torch.Tensor, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    # --- Full log-probability matrix ---
    def log_matrix(self, context=None):
        return F.log_softmax(self.forward(context=context, log=False), dim=-1)

    # --- Expected duration per state ---
    def expected_duration(self, context=None):
        probs = self.forward(context=context, log=False)
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        return (probs * durations).sum(dim=-1)

    # --- Most likely duration per state ---
    def mode(self, context=None):
        probs = self.forward(context=context, log=False)
        return torch.argmax(probs, dim=-1) + 1

    # --- Sampling ---
    def sample(self, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample().to(self.device, DTYPE)


class Transition(Contextual):
    """
    Contextual transition distribution per state with buffer caching.
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
        device: Optional[torch.device] = None,
    ):
        target_dim = n_states * n_states
        super().__init__(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation="tanh",
            debug=debug,
            device=device,
        )

        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)
        self.scale = scale

        # Initialize logits
        if init_mode == "uniform":
            logits_init = torch.log(torch.ones(n_states, n_states) / n_states)
        elif init_mode == "diag_bias":
            logits_init = torch.full((n_states, n_states), 0.1)
            diag_val = torch.log(torch.tensor(0.7, dtype=DTYPE))
            logits_init.fill_diagonal_(diag_val)
            logits_init = torch.log(logits_init)
        elif init_mode == "normal":
            logits_init = torch.randn(n_states, n_states) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'")

        self.logits = nn.Parameter(logits_init.to(self.device, dtype=DTYPE))

        # --- Buffers for caching ---
        self.register_buffer("_cached_logits", None)
        self.register_buffer("_cached_softmax", None)
        self.register_buffer("_cached_log_softmax", None)
        self.register_buffer("_cached_dist", None)
        self.register_buffer("_cached_context_hash", None)

    # --- Context hash for caching ---
    def _context_hash(self, context: Optional[torch.Tensor]):
        if context is None:
            return None
        return context.detach().cpu().sum().item()

    # --- Context-modulated logits with caching ---
    def _mod_logits(self, context: Optional[torch.Tensor]):
        ctx_hash = self._context_hash(context)
        if (self._cached_logits is not None) and (self._cached_context_hash == ctx_hash):
            return self._cached_logits

        mod_logits = self._apply_context(self.logits.flatten(), context, self.scale)
        mod_logits = mod_logits.view(self.n_states, self.n_states) / self.temperature
        mod_logits = torch.clamp(mod_logits, -MAX_LOGITS, MAX_LOGITS)

        # Cache and invalidate dependent buffers
        self._cached_logits = mod_logits
        self._cached_context_hash = ctx_hash
        self._cached_softmax = None
        self._cached_log_softmax = None
        self._cached_dist = None

        return mod_logits

    # --- Forward: return logits, softmax, log_softmax, or distribution ---
    def forward(self, context=None, log: bool = False, return_dist: bool = False):
        mod_logits = self._mod_logits(context)

        if return_dist:
            if self._cached_dist is None:
                self._cached_dist = Categorical(logits=mod_logits)
            return self._cached_dist

        if log:
            if self._cached_log_softmax is None:
                self._cached_log_softmax = F.log_softmax(mod_logits, dim=-1)
            return self._cached_log_softmax

        if self._cached_softmax is None:
            self._cached_softmax = F.softmax(mod_logits, dim=-1)
        return self._cached_softmax

    # --- Log-probability of samples ---
    def log_prob(self, x: torch.Tensor, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, dtype=torch.long))

    # --- Full log-probability matrix ---
    def log_matrix(self, context=None):
        return F.log_softmax(self.forward(context=context, log=False), dim=-1)

    # --- Expected transitions per state ---
    def expected_transitions(self, context=None):
        return self.forward(context=context, log=False)

    # --- Most likely next state per state ---
    def mode(self, context=None):
        probs = self.forward(context=context, log=False)
        return torch.argmax(probs, dim=-1)

    # --- Sampling ---
    def sample(self, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample().to(self.device, DTYPE)
