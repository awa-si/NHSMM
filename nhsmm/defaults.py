# nhsmm/defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli, Independent
from typing import Optional, Dict, Any

EPS = 1e-12
MAX_LOGITS = 50.0
DTYPE = torch.float64

class HSMMError(ValueError):
    pass

class Contextual(nn.Module):
    """Base class for context-modulated parameters."""

    def __init__(
        self,
        context_dim: Optional[int],
        target_dim: int,
        aggregate_context: bool = True,
        hidden_dim: Optional[int] = None,
        activation: str = "tanh",
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if context_dim is not None and context_dim <= 0:
            raise HSMMError(f"context_dim must be positive, got {context_dim}")
        if target_dim <= 0:
            raise HSMMError(f"target_dim must be positive, got {target_dim}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregate_context = aggregate_context
        self.context_dim = context_dim
        self.target_dim = target_dim

        if context_dim:
            hidden_dim = hidden_dim or max(context_dim, target_dim)
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=DTYPE),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=DTYPE),
            )
            self._init_weights()
        else:
            self.context_net = None

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
        }
        if name.lower() not in activations:
            raise HSMMError(f"Unsupported activation: {name}")
        return activations[name.lower()]

    def _init_weights(self):
        if self.context_net:
            for m in self.context_net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None or self.context_net is None:
            return None
        if not torch.is_tensor(context):
            raise HSMMError(f"Context must be a tensor, got {type(context)}")
        context = context.to(self.device, DTYPE)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[-1] != self.context_dim:
            raise HSMMError(f"Context last dimension must be {self.context_dim}, got {context.shape[-1]}")
        return context

    def _contextual_modulation(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        if self.context_net is None or context is None:
            return torch.zeros_like(base)
        context = self._validate_context(context)
        delta = self.context_net(context)
        # reshape delta to broadcast over base
        while delta.ndim < base.ndim:
            delta = delta.unsqueeze(1)
        if self.aggregate_context:
            delta = delta.mean(dim=0, keepdim=True)
        return scale * torch.tanh(delta + torch.zeros_like(base))

    def _apply_context(self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        modulated = base + self._contextual_modulation(base, context, scale)
        return torch.where(torch.isfinite(modulated), modulated, base)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            context_dim=self.context_dim,
            target_dim=self.target_dim,
            aggregate_context=self.aggregate_context,
            device=str(self.device),
            dtype=str(DTYPE),
        )

    def to(self, device, **kwargs):
        self.device = torch.device(device)
        super().to(device, **kwargs)
        if self.context_net:
            self.context_net.to(device, **kwargs)
        return self

class Emission(Contextual):
    """Contextual emission distribution."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-3,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        emission_type: str = "gaussian",
        aggregate_context: bool = True,
        device: Optional[torch.device] = None,
        scale: float = 0.1,
    ):
        target_dim = n_states * n_features
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device)
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.scale = scale
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

    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            var = torch.clamp(F.softplus(self.log_var), min=self.min_covar) + EPS
            if return_dist:
                return Independent(Normal(mu, var.sqrt()), 1)
            return mu, var

        logits = self._apply_context(self.logits, context, self.scale)
        if return_dist:
            if self.emission_type == "categorical":
                return Categorical(logits=logits)
            else:
                return Independent(Bernoulli(logits=logits), 1)
        return logits

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        x = x.to(self.device, DTYPE)
        if self.emission_type == "gaussian" and x.ndim < dist.event_shape[0] + 1:
            x = x.unsqueeze(1)
        return dist.log_prob(x)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,))  # (n_samples, n_states, F)
        samples = samples.transpose(0, 1)  # (n_states, n_samples, F)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        return samples

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            scale=self.scale,
            emission_type=self.emission_type,
        ))
        return cfg

class Duration(Contextual):
    """Contextual duration distribution for HSMM states."""

    def __init__(
        self,
        n_states: int,
        max_duration: int = 20,
        context_dim: Optional[int] = None,
        aggregate_context: bool = True,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        scale: float = 0.1
    ):
        target_dim = n_states * max_duration
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device)
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = scale
        self.logits = nn.Parameter(torch.randn(n_states, max_duration, device=self.device, dtype=DTYPE) * 0.1)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def expected_duration(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.forward(context=context, log=False, return_dist=False)
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)
        return (probs * durations).sum(dim=-1)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).T  # (n_states, n_samples)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        return samples

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            max_duration=self.max_duration,
            temperature=self.temperature,
            scale=self.scale,
        ))
        return cfg

class Transition(Contextual):
    """Contextual transition distribution with batched context support."""

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        aggregate_context: bool = True,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        scale: float = 0.1
    ):
        target_dim = n_states * n_states
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device)
        self.logits = nn.Parameter(torch.randn(n_states, n_states, device=self.device, dtype=DTYPE) * 0.1)
        self.temperature = max(temperature, 1e-6)
        self.n_states = n_states
        self.scale = scale

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def expected_transitions(self, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, log=False, return_dist=False)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).transpose(0,1)  # (n_states, n_samples)
        if state_indices is not None:
            samples = samples[state_indices, ...]
        return samples

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(n_states=self.n_states, temperature=self.temperature))
        return cfg
