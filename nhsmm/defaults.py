import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Distribution
from typing import Optional, Union, List, Tuple

DTYPE = torch.float64


# -----------------------------
# Emission Module
# -----------------------------
class DefaultEmission(nn.Module):
    """
    Gaussian or Categorical emissions with optional context modulation.

    Args:
        n_states: Number of hidden states.
        n_features: Dimensionality of observations.
        min_covar: Minimum variance for Gaussian emissions.
        context_dim: Optional dimensionality of context vector.
        scale: Scaling factor for context shift.
        emission_type: 'gaussian' or 'categorical'.
        aggregate_context: If True, average context over batch.
        device: Torch device.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_covar: float = 1e-3,
        context_dim: Optional[int] = None,
        scale: float = 0.1,
        emission_type: str = "gaussian",
        aggregate_context: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.scale = scale
        self.aggregate_context = aggregate_context
        self.emission_type = emission_type.lower()
        self.device = device or torch.device("cpu")

        # Parameters
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
            self.log_var = nn.Parameter(torch.full((n_states, n_features), -2.0, dtype=DTYPE, device=self.device))
        elif self.emission_type == "categorical":
            self.logits = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE, device=self.device))
        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")

        # Optional context adapter
        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * n_features, device=self.device, dtype=DTYPE),
                nn.LayerNorm(n_states * n_features, device=self.device, dtype=DTYPE),
                nn.Tanh(),
                nn.Linear(n_states * n_features, n_states * n_features, device=self.device, dtype=DTYPE),
            )
        else:
            self.context_net = None

    def _contextual_shift(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.context_net is None or context is None:
            return None
        # Ensure batch dimension
        context = context.unsqueeze(0) if context.ndim == 1 else context
        delta = self.context_net(context).view(-1, self.n_states, self.n_features)
        if self.aggregate_context:
            delta = delta.mean(dim=0)
        return self.scale * torch.tanh(delta).clamp(-3.0, 3.0)

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        return_dist: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[Distribution]]:
        """
        Returns emission parameters or distributions.

        Returns:
            - Gaussian: mu, var or list of MultivariateNormal
            - Categorical: logits or list of Categorical
        """
        delta = self._contextual_shift(context)

        if self.emission_type == "gaussian":
            mu = self.mu + (delta if delta is not None else 0)
            var = F.softplus(self.log_var) + self.min_covar
            if not return_dist:
                return mu, var
            # Vectorized batch distribution
            cov = torch.diag_embed(var)
            return [MultivariateNormal(loc=mu[k], covariance_matrix=cov[k]) for k in range(self.n_states)]

        logits = self.logits + (delta if delta is not None else 0)
        if not return_dist:
            return logits
        return [Categorical(logits=logits[k]) for k in range(self.n_states)]


# -----------------------------
# Duration Module
# -----------------------------
class DefaultDuration(nn.Module):
    """Per-state duration probabilities with optional context modulation."""

    def __init__(
        self,
        n_states: int,
        max_duration: int = 20,
        context_dim: Optional[int] = None,
        temperature: float = 1.0,
        aggregate_context: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = temperature
        self.aggregate_context = aggregate_context
        self.device = device or torch.device("cpu")

        self.logits = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE, device=self.device))
        self.mod_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE, device=self.device))

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * max_duration, device=self.device, dtype=DTYPE),
                nn.LayerNorm(n_states * max_duration, device=self.device, dtype=DTYPE),
                nn.Tanh(),
                nn.Linear(n_states * max_duration, n_states * max_duration, device=self.device, dtype=DTYPE),
            )
        else:
            self.context_net = None

    def _contextual_logits(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.logits
        if self.context_net is not None and context is not None:
            context = context.unsqueeze(0) if context.ndim == 1 else context
            delta = self.context_net(context).view(-1, self.n_states, self.max_duration)
            if self.aggregate_context:
                delta = delta.mean(dim=0)
            logits = logits + self.mod_scale * torch.tanh(delta).clamp(-3.0, 3.0)
        return logits

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False) -> torch.Tensor:
        logits = self._contextual_logits(context) / self.temperature
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True)


# -----------------------------
# Transition Module
# -----------------------------
class DefaultTransition(nn.Module):
    """Learnable per-state transitions with optional context modulation."""

    def __init__(
        self,
        n_states: int,
        context_dim: Optional[int] = None,
        temperature: float = 1.0,
        aggregate_context: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_states = n_states
        self.temperature = temperature
        self.aggregate_context = aggregate_context
        self.device = device or torch.device("cpu")

        self.logits = nn.Parameter(torch.zeros(n_states, n_states, dtype=DTYPE, device=self.device))
        self.mod_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE, device=self.device))

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * n_states, device=self.device, dtype=DTYPE),
                nn.LayerNorm(n_states * n_states, device=self.device, dtype=DTYPE),
                nn.Tanh(),
                nn.Linear(n_states * n_states, n_states * n_states, device=self.device, dtype=DTYPE),
            )
        else:
            self.context_net = None

    def _contextual_logits(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.logits
        if self.context_net is not None and context is not None:
            context = context.unsqueeze(0) if context.ndim == 1 else context
            delta = self.context_net(context).view(-1, self.n_states, self.n_states)
            if self.aggregate_context:
                delta = delta.mean(dim=0)
            logits = logits + self.mod_scale * torch.tanh(delta).clamp(-3.0, 3.0)
        return logits

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False) -> torch.Tensor:
        logits = self._contextual_logits(context) / self.temperature
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def log_probs(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(context=context, log=True)
