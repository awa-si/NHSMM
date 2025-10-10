from typing import Optional, Union
import torch
from torch import nn
from torch.distributions import Poisson, LogNormal
from nhsmm.utilities import constraints, utils


class NeuralDuration(nn.Module):
    """
    Neural duration distribution for HSMM state durations.

    Supports:
        - Parametric Poisson or LogNormal durations
        - Posterior-weighted updates
        - Optional neural encoder for contextual parameters
    """

    EPS = 1e-8

    def __init__(
        self,
        rate: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        mode: str = "poisson",
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ):
        super().__init__()
        self.mode = mode
        self.encoder = encoder
        self.context_mode = context_mode

        if mode == "poisson":
            if rate is None:
                raise ValueError("Poisson duration requires 'rate'.")
            self.dist = Poisson(rate.clamp_min(self.EPS))
        elif mode == "lognormal":
            if mean is None or std is None:
                raise ValueError("LogNormal duration requires 'mean' and 'std'.")
            self.dist = LogNormal(mean, std.clamp_min(self.EPS))
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    @property
    def dof(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def sample_duration_pdf(
        cls,
        n_states: int,
        mode: str = "poisson",
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        mean_scale: float = 10.0,
        std_scale: float = 1.0,
    ) -> "NeuralDuration":
        """Initialize random duration parameters."""
        if mode == "poisson":
            rate = mean_scale * torch.rand(n_states).clamp_min(cls.EPS)
            return cls(rate=rate, mode=mode, encoder=encoder, context_mode=context_mode)
        elif mode == "lognormal":
            mean = torch.randn(n_states) * 0.1
            std = std_scale * torch.ones(n_states)
            return cls(mean=mean, std=std, mode=mode, encoder=encoder, context_mode=context_mode)
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    def estimate_duration_pdf(
        self,
        durations: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
        inplace: bool = False,
    ) -> "NeuralDuration":
        """Estimate duration parameters from posterior weights."""
        if self.encoder is not None and theta is not None:
            params = self._contextual_params(durations, theta)
        else:
            params = self._compute_mle(durations, posterior)

        if inplace:
            self._assign_params(params)
            return self
        return NeuralDuration(**params, mode=self.mode, encoder=self.encoder, context_mode=self.context_mode)

    def _compute_mle(self, durations: torch.Tensor, posterior: torch.Tensor):
        """Compute simple posterior-weighted MLE parameters."""
        weights = posterior.sum(dim=1, keepdim=True).clamp_min(self.EPS)
        weighted_mean = (posterior @ durations.unsqueeze(-1)) / weights

        if self.mode == "poisson":
            return {"rate": weighted_mean.squeeze(-1)}
        elif self.mode == "lognormal":
            log_dur = durations.clamp_min(self.EPS).log().unsqueeze(-1)
            weighted_log_mean = (posterior @ log_dur) / weights
            weighted_log_var = (posterior @ ((log_dur - weighted_log_mean) ** 2)) / weights
            return {"mean": weighted_log_mean.squeeze(-1), "std": weighted_log_var.sqrt().squeeze(-1)}
        else:
            raise ValueError

    def _contextual_params(self, durations: torch.Tensor, theta: utils.ContextualVariables):
        """Use encoder to predict duration parameters."""
        encoded = (
            self.encoder(**theta.X) if isinstance(theta.X, dict)
            else self.encoder(*theta.X) if isinstance(theta.X, tuple)
            else self.encoder(theta.X)
        )
        if self.mode == "poisson":
            rate = torch.nn.functional.softplus(encoded).squeeze(-1)
            return {"rate": rate}
        elif self.mode == "lognormal":
            mean, log_std = torch.chunk(encoded, 2, dim=-1)
            std = log_std.exp().clamp_min(self.EPS)
            return {"mean": mean.squeeze(-1), "std": std.squeeze(-1)}
        else:
            raise ValueError

    def _assign_params(self, params: dict):
        if self.mode == "poisson":
            self.dist = Poisson(params["rate"].clamp_min(self.EPS))
        else:
            self.dist = LogNormal(params["mean"], params["std"].clamp_min(self.EPS))

    def log_prob(self, durations: torch.Tensor) -> torch.Tensor:
        """Compute log p(duration | state)."""
        return self.dist.log_prob(durations.unsqueeze(0))

    def to(self, device: Union[str, torch.device]) -> "NeuralDuration":
        for p in self.parameters():
            p.data = p.data.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        return self
