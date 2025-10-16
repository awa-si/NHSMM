# defaults.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution, Normal, Independent, Bernoulli

from typing import Optional, Union, Tuple, Dict, Any

# ============================================================
# Constants and Errors
# ============================================================
EPS = 1e-8
MAX_LOGITS = 50.0
DTYPE = torch.float64

class HSMMError(Exception):
    pass

# ============================================================
# Contextual Base
# ============================================================
class Contextual(nn.Module):
    def __init__(
        self,
        context_dim: Optional[int],
        target_dim: int,
        aggregate_context: bool = True,
        hidden_dim: Optional[int] = None,
        activation: str = "tanh",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        if context_dim is not None and context_dim <= 0:
            raise HSMMError(f"context_dim must be positive, got {context_dim}")
        if target_dim <= 0:
            raise HSMMError(f"target_dim must be positive, got {target_dim}")

        self.context_dim = context_dim
        self.target_dim = target_dim
        self.aggregate_context = aggregate_context
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        if context_dim is not None:
            hidden_dim = hidden_dim or max(context_dim, target_dim)
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim, device=self.device, dtype=self.dtype),
                nn.LayerNorm(hidden_dim, device=self.device, dtype=self.dtype),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim, device=self.device, dtype=self.dtype),
            )
            self._initialize_weights()
        else:
            self.context_net = None

    def _get_activation(self, name: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
        }.get(name.lower(), nn.Tanh())

    def _initialize_weights(self):
        if self.context_net:
            for m in self.context_net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.constant_(m.bias, 0.0)

    def _validate_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None or self.context_net is None:
            return None
        if not torch.is_tensor(context):
            raise HSMMError(f"Context must be a tensor, got {type(context)}")
        context = context.to(self.device, self.dtype)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[-1] != self.context_dim:
            raise HSMMError(f"Context last dimension must be {self.context_dim}, got {context.shape[-1]}")
        return context

    def _contextual_modulation(
        self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1
    ) -> torch.Tensor:
        if self.context_net is None or context is None:
            return torch.zeros_like(base)
        context = self._validate_context(context)
        delta = self.context_net(context)
        while delta.ndim < base.ndim:
            delta = delta.unsqueeze(1)
        delta = delta.expand_as(base)
        if self.aggregate_context and delta.shape[0] > 1:
            delta = delta.mean(dim=0, keepdim=True)
        return scale * torch.tanh(delta).clamp(-3.0, 3.0)

    def _apply_context(
        self, base: torch.Tensor, context: Optional[torch.Tensor], scale: float = 0.1
    ) -> torch.Tensor:
        modulated = base + self._contextual_modulation(base, context, scale)
        return torch.where(torch.isfinite(modulated), modulated, base)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            context_dim=self.context_dim,
            target_dim=self.target_dim,
            aggregate_context=self.aggregate_context,
            device=str(self.device),
            dtype=str(self.dtype),
        )

    def to(self, device, **kwargs):
        self.device = torch.device(device)
        super().to(device, **kwargs)
        if self.context_net is not None:
            self.context_net = self.context_net.to(device, **kwargs)
        return self

# ============================================================
# Emission Module
# ============================================================
class Emission(Contextual):
    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-3,
        context_dim: Optional[int] = None,
        scale: float = 0.1,
        emission_type: str = "gaussian",
        aggregate_context: bool = True,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE,
    ):
        target_dim = n_states * n_features
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device, dtype)
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.scale = scale
        self.emission_type = emission_type.lower()
        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features, device=self.device, dtype=self.dtype) * 0.1)
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -2.0, device=self.device, dtype=self.dtype))
        elif self.emission_type in ["categorical", "bernoulli"]:
            self.logits = nn.Parameter(torch.zeros(self.n_states, self.n_features, device=self.device, dtype=self.dtype))
        else:
            raise HSMMError(f"Unsupported emission_type: {self.emission_type}")

    def forward(self, context: Optional[torch.Tensor] = None, return_dist: bool = False):
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            var = F.softplus(self.log_var) + self.min_covar + EPS
            return (mu, var) if not return_dist else Independent(Normal(mu, var.sqrt()), 1)
        logits = self._apply_context(self.logits, context, self.scale)
        if not return_dist:
            return logits
        if self.emission_type == "categorical":
            return Categorical(logits=logits)
        return Independent(Bernoulli(logits=logits), 1)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        dist = self.forward(context=context, return_dist=True)
        x = x.to(self.device, self.dtype)
        if self.emission_type == "gaussian" and x.ndim == 2:
            x = x.unsqueeze(1)
        return dist.log_prob(x)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,))  # (n_samples, B, F)
        samples = samples.transpose(0, 1)    # (B, n_samples, F)
        if state_indices is not None:
            samples = samples[state_indices]
        return samples

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            scale=self.scale,
            emission_type=self.emission_type,
        ))
        return cfg

# ============================================================
# Duration Module
# ============================================================
class Duration(Contextual):
    def __init__(self, n_states: int, max_duration: int = 20, context_dim: Optional[int] = None, temperature: float = 1.0,
                 aggregate_context: bool = True, hidden_dim: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        target_dim = n_states * max_duration
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device, dtype)
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(temperature, 1e-6)
        self.scale = 0.1
        self.logits = nn.Parameter(torch.randn(n_states, max_duration, device=self.device, dtype=self.dtype) * 0.1)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True, return_dist=False)

    def probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=False, return_dist=False)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).T
        if state_indices is not None:
            samples = samples[state_indices]
        return samples

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            max_duration=self.max_duration,
            temperature=self.temperature,
        ))
        return cfg

# ============================================================
# Transition Module
# ============================================================
class Transition(Contextual):
    def __init__(self, n_states: int, context_dim: Optional[int] = None, temperature: float = 1.0,
                 aggregate_context: bool = True, hidden_dim: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        target_dim = n_states * n_states
        super().__init__(context_dim, target_dim, aggregate_context, hidden_dim, "tanh", device, dtype)
        self.n_states = n_states
        self.temperature = max(temperature, 1e-6)
        self.scale = 0.1
        self.logits = nn.Parameter(torch.randn(n_states, n_states, device=self.device, dtype=self.dtype) * 0.1)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        logits = torch.clamp(logits, -MAX_LOGITS, MAX_LOGITS)
        if return_dist:
            return Categorical(logits=logits)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True, return_dist=False)

    def probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=False, return_dist=False)

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dist = self.forward(context=context, return_dist=True)
        samples = dist.sample((n_samples,)).T
        if state_indices is not None:
            samples = samples[state_indices]
        return samples

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(dict(
            n_states=self.n_states,
            temperature=self.temperature,
        ))
        return cfg
