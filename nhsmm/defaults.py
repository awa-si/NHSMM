# nhsmm/defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli, Independent

import logging
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
    """Context-modulated parameter adapter with safe delta handling."""

    def __init__(
        self,
        context_dim: Optional[int],
        target_dim: int,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        hidden_dim: Optional[int] = None,
        activation: str = "tanh",
        final_activation: str = "tanh",
        device: Optional[torch.device] = None,
        debug: bool = False,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        max_delta: Optional[float] = 0.5,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.aggregate_context = aggregate_context
        self.aggregate_method = aggregate_method.lower()
        self.debug = debug
        self.final_activation_name = final_activation
        self.allow_projection = allow_projection
        self.max_delta = max_delta

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
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _ensure_projection(self, in_dim: int):
        if not self.allow_projection:
            raise ValueError(f"Context last dimension must be {self.context_dim}, got {in_dim}")
        if self._proj is None:
            out_dim = self.context_dim or self.target_dim
            self._proj = nn.Linear(in_dim, out_dim, device=self.device, dtype=DTYPE)
            nn.init.normal_(self._proj.weight, 0.0, 1e-3)
            nn.init.zeros_(self._proj.bias)
            if self.debug: logger.debug(f"[Contextual] Added projection {in_dim} → {out_dim}")

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None: return None
        context = context.to(self.device, DTYPE).unsqueeze(0) if context.ndim == 1 else context
        in_dim = context.shape[-1]
        if self.context_dim is None:
            self.context_dim = in_dim
        if in_dim != self.context_dim:
            self._ensure_projection(in_dim)
            context = self._proj(context)
        return context

    def _prepare_delta(self, delta: torch.Tensor, base_shape: Tuple[int, ...], scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)
        if self.temporal_adapter and delta.ndim == 2: delta = delta.unsqueeze(1)
        if self.temporal_adapter: delta = self.temporal_adapter(delta.transpose(1,2)).transpose(1,2)
        if self.spatial_adapter: delta = self.spatial_adapter(delta)
        if self.aggregate_context and delta.ndim > 1:
            delta = {
                "mean": delta.mean(dim=0, keepdim=True),
                "sum": delta.sum(dim=0, keepdim=True),
                "max": delta.max(dim=0, keepdim=True)[0],
            }.get(self.aggregate_method, delta)
        delta = self._get_activation(self.final_activation_name)(delta) * scale
        if self.max_delta is not None: delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        if grad_scale is not None: delta = delta * grad_scale
        while delta.ndim < len(base_shape): delta = delta.unsqueeze(0)
        return delta.expand(*base_shape)

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1, grad_scale: Optional[float] = None) -> torch.Tensor:
        if context is None: return base
        context = self._validate_context(context)
        delta = self.context_net(context) if self.context_net else self._proj(context)
        delta = self._prepare_delta(delta, base.shape, scale, grad_scale)
        result = base + torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        return result

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
            if m is not None: m.to(device, **kwargs)
        return self


class Emission(Contextual):
    """Contextual emission distribution: Gaussian, Categorical, Bernoulli."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        emission_type: str = "gaussian",
        min_covar: float = 1e-6,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        aggregate_context: bool = True,
        aggregate_method: str = "mean",
        scale: float = 0.1,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
        temporal_adapter: bool = False,
        spatial_adapter: bool = False,
        allow_projection: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            context_dim=context_dim,
            target_dim=n_states * n_features,
            aggregate_context=aggregate_context,
            aggregate_method=aggregate_method,
            hidden_dim=hidden_dim,
            device=None,
            debug=debug,
            temporal_adapter=temporal_adapter,
            spatial_adapter=spatial_adapter,
            allow_projection=allow_projection,
        )
        self.n_states = n_states
        self.n_features = n_features
        self.scale = scale
        self.min_covar = min_covar
        self.modulate_var = modulate_var
        self.adaptive_scale = adaptive_scale
        self.emission_type = emission_type.lower()

        # Optional per-state gating
        self.state_gate = nn.Sequential(
            nn.Linear(context_dim, n_states, device=self.device, dtype=DTYPE),
            nn.Sigmoid()
        ) if context_dim is not None else None

        self._init_parameters()

    def _init_parameters(self):
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features, device=self.device, dtype=DTYPE) * 0.1)
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -1.0, device=self.device, dtype=DTYPE))
        elif self.emission_type in ["categorical", "bernoulli"]:
            self.logits = nn.Parameter(torch.zeros(self.n_states, self.n_features, device=self.device, dtype=DTYPE))
        else:
            raise ValueError(f"Unsupported emission_type: {self.emission_type}")

    def _modulate_per_state(self, base: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None or self.state_gate is None:
            return base
        gate = self.state_gate(context)
        if self.adaptive_scale:
            ctx_norm = context.norm(dim=-1, keepdim=True)
            adaptive_factor = self.scale / (ctx_norm + 1e-6)
            gate = gate * adaptive_factor
        while gate.ndim < base.ndim:
            gate = gate.unsqueeze(-1)
        return base * gate

    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            mu = self._modulate_per_state(mu, context)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                delta_var = self._apply_context(var, context, self.scale).abs()
                delta_var = self._modulate_per_state(delta_var, context)
                var = torch.clamp(var + delta_var, min=self.min_covar)
            if return_dist:
                return Independent(Normal(mu, var.sqrt()), 1)
            return mu, var

        logits = self._apply_context(self.logits, context, self.scale)
        logits = self._modulate_per_state(logits, context)
        if return_dist:
            return Categorical(logits=logits) if self.emission_type == "categorical" else Independent(Bernoulli(logits=logits), 1)
        return logits

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, DTYPE))

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)


class Duration(Contextual):
    """Contextual categorical duration distribution per state."""

    def __init__(self, n_states: int, max_duration: int = 25,
                 context_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None,
                 aggregate_context: bool = True,
                 aggregate_method: str = "mean",
                 temperature: float = 1.0,
                 scale: float = 1.0,
                 debug: bool = False):
        super().__init__(
            context_dim=context_dim,
            target_dim=n_states * max_duration,
            aggregate_context=aggregate_context,
            aggregate_method=aggregate_method,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            debug=debug
        )
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.logits = nn.Parameter(torch.randn(n_states, max_duration, device=self.device, dtype=DTYPE) * 0.1)

    def forward(self, context=None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist: return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_prob(self, x, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, DTYPE))

    def expected_duration(self, context=None):
        probs = self.forward(context=context, log=False)
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        return (probs * durations).sum(dim=-1)

    def sample(self, n_samples: int = 1, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)


class Transition(Contextual):
    """Contextual transition distribution (categorical) per state."""

    def __init__(self, n_states: int,
                 context_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None,
                 aggregate_context: bool = True,
                 aggregate_method: str = "mean",
                 temperature: float = 1.0,
                 scale: float = 1.0,
                 debug: bool = False):
        super().__init__(
            context_dim=context_dim,
            target_dim=n_states * n_states,
            aggregate_context=aggregate_context,
            aggregate_method=aggregate_method,
            hidden_dim=hidden_dim,
            activation="tanh",
            final_activation="tanh",
            debug=debug
        )
        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.logits = nn.Parameter(torch.randn(n_states, n_states, device=self.device, dtype=DTYPE) * 0.1)

    def forward(self, context=None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist: return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_prob(self, x, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, DTYPE))

    def expected_transitions(self, context=None):
        return self.forward(context=context, log=False)

    def sample(self, n_samples: int = 1, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.sample((n_samples,)).to(self.device, DTYPE)
