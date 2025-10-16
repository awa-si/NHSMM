import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Distribution
from typing import Optional, Union, List, Tuple

DTYPE = torch.float64
EPS = 1e-8

# ============================================================
# Base: Contextual Mixin
# ============================================================
class Contextual(nn.Module):
    """
    Base class for context-aware HSMM submodules.
    Provides optional modulation of internal parameters through a learned network.
    """

    def __init__(
        self,
        context_dim: Optional[int],
        target_dim: int,
        aggregate_context: Union[str, bool] = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.aggregate_context = aggregate_context
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, target_dim, device=self.device, dtype=DTYPE),
                nn.LayerNorm(target_dim, device=self.device, dtype=DTYPE),
                nn.Tanh(),
                nn.Linear(target_dim, target_dim, device=self.device, dtype=DTYPE),
            )
        else:
            self.context_net = None

    def _contextual_modulation(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        scale: float = 0.1,
    ) -> torch.Tensor:
        """Compute context modulation delta."""
        if self.context_net is None or context is None:
            return torch.zeros_like(base)

        if context.ndim == 1:
            context = context.unsqueeze(0)

        delta = self.context_net(context).reshape(-1, *base.shape)
        if self.aggregate_context:
            delta = delta.mean(dim=0)

        return scale * torch.tanh(delta).clamp(-3.0, 3.0)

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor],
        scale: float = 0.1,
    ) -> torch.Tensor:
        """Apply context modulation to base tensor."""
        return base + self._contextual_modulation(base, context, scale)

    def to(self, device):
        self.device = torch.device(device)
        super().to(device)
        return self


# ============================================================
# Emission Module
# ============================================================
class Emission(Contextual):
    """
    Emission distribution module for HSMMs.
    Supports Gaussian and Categorical emissions, optionally modulated by context.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-3,
        context_dim: Optional[int] = None,
        scale: float = 0.1,
        emission_type: str = "gaussian",
        aggregate_context: Union[str, bool] = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__(context_dim, n_states * n_features, aggregate_context, device)
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.scale = scale
        self.emission_type = emission_type.lower()

        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -2.0, dtype=DTYPE, device=self.device))
        elif self.emission_type == "categorical":
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        return_dist: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[Distribution]]:
        """Compute emission parameters or distributions."""
        if self.emission_type == "gaussian":
            mu = self._apply_context(self.mu, context, self.scale)
            var = F.softplus(self.log_var) + self.min_covar
            if not return_dist:
                return mu, var
            cov = torch.diag_embed(var)
            return [MultivariateNormal(mu[k], cov[k]) for k in range(self.n_states)]

        logits = self._apply_context(self.logits, context, self.scale)
        if not return_dist:
            return logits

        if context is not None and context.ndim == 3:
            B, T, _ = context.shape
            logits_t = logits.unsqueeze(0).expand(B, -1, -1)
            return [[Categorical(logits=logits_t[b, k]) for k in range(self.n_states)] for b in range(B)]

        return [Categorical(logits=logits[k]) for k in range(self.n_states)]


# ============================================================
# Duration Module
# ============================================================
class Duration(Contextual):
    """
    Per-state duration probabilities with optional context modulation.
    """

    def __init__(
        self,
        n_states: int,
        max_duration: int = 20,
        context_dim: Optional[int] = None,
        temperature: float = 1.0,
        aggregate_context: Union[str, bool] = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__(context_dim, n_states * max_duration, aggregate_context, device)
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = max(float(temperature), 1e-6)
        self.logits = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE, device=self.device))
        self.scale = 0.1

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        if not return_dist:
            return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

        if context is not None and context.ndim == 3:
            B, T, _ = context.shape
            logits_t = logits.unsqueeze(0).expand(B, -1, -1)
            return [[Categorical(logits=logits_t[b, k]) for k in range(self.n_states)] for b in range(B)]

        return [Categorical(logits=logits[k]) for k in range(self.n_states)]

    def log_probs(self, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, log=True, return_dist=False)


# ============================================================
# Transition Module
# ============================================================
class Transition(Contextual):
    """
    Learnable per-state transitions with optional context modulation.
    """

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        temperature: float = 1.0,
        aggregate_context: Union[str, bool] = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__(context_dim, n_states * n_states, aggregate_context, device)
        self.n_states = n_states
        self.temperature = max(float(temperature), 1e-6)
        self.logits = nn.Parameter(torch.zeros(n_states, n_states, dtype=DTYPE, device=self.device))
        self.scale = 0.1

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        logits = self._apply_context(self.logits, context, self.scale) / self.temperature
        if not return_dist:
            return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

        if context is not None and context.ndim == 3:
            B, T, _ = context.shape
            logits_t = logits.unsqueeze(0).expand(B, -1, -1)
            return [[Categorical(logits=logits_t[b, k]) for k in range(self.n_states)] for b in range(B)]

        return [Categorical(logits=logits[k]) for k in range(self.n_states)]

    def log_probs(self, context: Optional[torch.Tensor] = None):
        return self.forward(context=context, log=True, return_dist=False)
