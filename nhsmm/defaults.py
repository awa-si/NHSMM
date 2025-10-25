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


logger = logging.getLogger("nhsmm")
if not logger.hasHandlers():
    # logger.setLevel(logging.DEBUG)
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
    Unified context-modulated parameter adapter.

    Key features:
    - Dynamically adapts to mismatched context dimensions (learnable projection).
    - Supports both temporal (Conv1d) and spatial (Linear) adapters.
    - Works symmetrically for Emission, Transition, and Duration modules.
    - Aggregates context embeddings (mean/sum/max) if enabled.
    - Fully device/dtype-safe and drop-in compatible.
    """

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

        # Core context encoder
        self.context_net = None
        if context_dim is not None:
            hidden_dim = hidden_dim or max(context_dim, target_dim)
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=DTYPE),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=DTYPE),
            )
            self._init_weights(self.context_net)

        # Adapters
        self.temporal_adapter = (
            nn.Conv1d(target_dim, target_dim, kernel_size=3, padding=1, bias=False).to(self.device, DTYPE)
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

        # Dynamic projection if incoming context doesn't match expected size
        self._proj = None

    # ---------- Internal helpers ----------

    def _get_activation(self, name: str) -> nn.Module:
        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }
        if name.lower() not in acts:
            raise ValueError(f"Unsupported activation: {name}")
        return acts[name.lower()]

    def _init_weights(self, module: nn.Module) -> None:
        for m in module:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _ensure_projection(self, in_dim: int) -> None:
        """Ensure projection exists if context dim mismatches."""
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
        context = context.to(self.device, DTYPE)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        in_dim = context.shape[-1]

        # Infer context_dim if not set yet
        if self.context_dim is None:
            self.context_dim = in_dim
            if self.debug:
                print(f"[Contextual] Inferred context_dim = {self.context_dim}")

        # Auto-project mismatched contexts
        if in_dim != self.context_dim:
            self._ensure_projection(in_dim)
            context = self._proj(context)
        return context

    # ---------- Core operations ----------

    def _prepare_delta(
        self,
        delta: torch.Tensor,
        base_shape: Tuple[int, ...],
        scale: float = 0.1,
        grad_scale: Optional[float] = None
    ) -> torch.Tensor:
        delta = delta.to(self.device, DTYPE)

        # Apply adapters
        if self.temporal_adapter:
            if delta.ndim == 2:
                delta = delta.unsqueeze(1)
            delta = delta.transpose(1, 2)
            delta = self.temporal_adapter(delta)
            delta = delta.transpose(1, 2)
        if self.spatial_adapter:
            delta = self.spatial_adapter(delta)

        # Aggregate across batch
        if self.aggregate_context and delta.ndim > 1:
            if self.aggregate_method == "mean":
                delta = delta.mean(dim=0, keepdim=True)
            elif self.aggregate_method == "sum":
                delta = delta.sum(dim=0, keepdim=True)
            elif self.aggregate_method == "max":
                delta, _ = delta.max(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unsupported aggregate_method: {self.aggregate_method}")

        # Apply final activation and scaling
        delta = self._get_activation(self.final_activation_name)(delta) * scale
        if grad_scale is not None:
            delta = delta * grad_scale

        return delta.expand_as(torch.zeros(base_shape, device=self.device, dtype=DTYPE))

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        scale: float = 0.1,
        grad_scale: Optional[float] = None
    ) -> torch.Tensor:
        """Apply context modulation to a base tensor."""
        if context is None:
            return base

        context = self._validate_context(context)
        if self.context_net:
            delta = self.context_net(context)
        else:
            if self._proj is None:
                self._ensure_projection(context.shape[-1])
            delta = self._proj(context)

        delta = self._prepare_delta(delta, base.shape, scale=scale, grad_scale=grad_scale)
        result = base + torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))

        if self.debug:
            print(f"[Contextual] base={tuple(base.shape)} ctx={tuple(context.shape)} "
                  f"delta={tuple(delta.shape)} → result={tuple(result.shape)}")
        return result

    # ---------- Utilities ----------

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
        )

    def to(self, device, **kwargs) -> "Contextual":
        self.device = torch.device(device)
        super().to(device, **kwargs)
        if self.context_net:
            self.context_net.to(device, **kwargs)
        if self.temporal_adapter:
            self.temporal_adapter.to(device, **kwargs)
        if self.spatial_adapter:
            self.spatial_adapter.to(device, **kwargs)
        if self._proj:
            self._proj.to(device, **kwargs)
        return self


class Emission(Contextual):
    """Contextual emission distribution: Gaussian, Categorical, or Bernoulli."""

    def __init__(self, n_states, n_features, min_covar=1e-3, context_dim=None,
                 hidden_dim=None, emission_type="gaussian", aggregate_context=True,
                 aggregate_method="mean", device=None, scale=0.1, modulate_var=False,
                 debug=False):
        super().__init__(context_dim, n_states * n_features, aggregate_context, aggregate_method,
                         hidden_dim, "tanh", device, debug)
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.scale = scale
        self.modulate_var = modulate_var
        self.emission_type = emission_type.lower()
        self._init_parameters()

    def _init_parameters(self):
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features, device=self.device, dtype=DTYPE) * 0.1)
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -1.0, device=self.device, dtype=DTYPE))
        elif self.emission_type in ["categorical", "bernoulli"]:
            self.logits = nn.Parameter(torch.zeros(self.n_states, self.n_features, device=self.device, dtype=DTYPE))
        else:
            raise HSMMError(f"Unsupported emission_type: {self.emission_type}")

    def forward(self, context=None, return_dist=False):
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar)
            if self.modulate_var:
                var = self._apply_context(var, context, self.scale).abs() + self.min_covar
            if return_dist:
                return Independent(Normal(mu, var.sqrt()), 1)
            return mu, var

        logits = self._apply_context(self.logits, context, self.scale)
        if return_dist:
            return Categorical(logits=logits) if self.emission_type == "categorical" else Independent(Bernoulli(logits=logits), 1)
        return logits

    def log_prob(self, x, context=None):
        dist = self.forward(context=context, return_dist=True)
        return dist.log_prob(x.to(self.device, DTYPE))

    def sample(self, n_samples=1, context=None, state_indices=None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.rsample((n_samples,)) if self.emission_type == "gaussian" else dist.sample((n_samples,))
        if state_indices is not None:
            samples = samples[state_indices]
        return samples.to(self.device, DTYPE)


class Duration(Contextual):
    """Contextual duration distribution (categorical)."""

    def __init__(self, n_states, max_duration=20, context_dim=None,
                 aggregate_context=True, aggregate_method="mean", hidden_dim=None,
                 device=None, temperature=1.0, scale=0.1, debug=False):
        super().__init__(context_dim, n_states*max_duration, aggregate_context, aggregate_method,
                         hidden_dim, "tanh", device, debug)
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.logits = nn.Parameter(torch.randn(n_states, max_duration, device=self.device, dtype=DTYPE)*0.1)

    def forward(self, context=None, log=False, return_dist=False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_prob(self, x, context=None):
        return self.forward(context=context, return_dist=True).log_prob(x.to(self.device, DTYPE))

    def expected_duration(self, context=None):
        probs = self.forward(context=context, log=False, return_dist=False)
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        return (probs * durations).sum(dim=-1)

    def sample(self, n_samples=1, context=None, state_indices=None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).transpose(0,1)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        return samples


class Transition(Contextual):
    """Contextual transition distribution (categorical)."""

    def __init__(self, n_states, context_dim=None, hidden_dim=None,
                 aggregate_context=True, aggregate_method="mean", device=None,
                 temperature=1.0, scale=0.1, debug=False):
        super().__init__(context_dim, n_states*n_states, aggregate_context, aggregate_method,
                         hidden_dim, "tanh", device, debug)
        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.logits = nn.Parameter(torch.randn(n_states, n_states, device=self.device, dtype=DTYPE)*0.1)

    def forward(self, context=None, log=False, return_dist=False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_prob(self, x, context=None):
        return self.forward(context=context, return_dist=True).log_prob(x.to(self.device, DTYPE))

    def expected_transitions(self, context=None):
        return self.forward(context=context, log=False, return_dist=False)

    def sample(self, n_samples=1, context=None, state_indices=None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).transpose(0,1)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        return samples
