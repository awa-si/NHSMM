import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Poisson, LogNormal, Distribution

from typing import Optional, Union

from nhsmm.defaults import DTYPE, Duration, EPS
from nhsmm.utilities import utils


class NeuralDuration(Duration):
    """
    Neural duration distribution for HSMM state durations.

    Features:
        - Poisson or LogNormal parametric durations
        - Posterior-weighted estimation
        - Optional neural encoder for context modulation
        - Forward probabilities with optional log
        - Batched context support
        - Duration sampling
    """

    def __init__(
        self,
        n_states: int,
        mode: str = "poisson",
        rate: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        max_duration: int = 20,
        device: Optional[torch.device] = None,
    ):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device)
        self.mode = mode
        self.encoder = encoder
        self.context_mode = context_mode
        self.scale = 0.1  # context modulation scale

        # initialize distribution
        if mode == "poisson":
            if rate is None:
                raise ValueError("Poisson duration requires 'rate'.")
            self._assign_params({"rate": rate})
        elif mode == "lognormal":
            if mean is None or std is None:
                raise ValueError("LogNormal duration requires 'mean' and 'std'.")
            self._assign_params({"mean": mean, "std": std})
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    @property
    def dof(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        context: Optional[Union[torch.Tensor, utils.ContextualVariables]] = None,
        log: bool = False,
        return_dist: bool = False,
    ) -> Union[torch.Tensor, List[Distribution]]:
        durations = torch.arange(1, self.max_duration + 1, dtype=DTYPE, device=self.device).unsqueeze(0)

        # base logits
        if self.mode == "poisson":
            rate = self.dist.rate.unsqueeze(-1)
            logits = durations * torch.log(rate) - rate - torch.lgamma(durations + 1)
        else:  # lognormal
            mean = self.dist.mean.unsqueeze(-1)
            std = self.dist.stddev.unsqueeze(-1).clamp_min(EPS)
            logits = -0.5 * ((durations - mean) / std) ** 2 - torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=DTYPE))

        # apply neural encoder context
        if self.encoder is not None and context is not None:
            params = self._contextual_params(durations, context)
            if self.mode == "poisson":
                rate = params["rate"].unsqueeze(-1)
                logits = durations * torch.log(rate) - rate - torch.lgamma(durations + 1)
            else:
                mean = params["mean"].unsqueeze(-1)
                std = params["std"].unsqueeze(-1).clamp_min(EPS)
                logits = -0.5 * ((durations - mean) / std) ** 2 - torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=DTYPE))

        if return_dist:
            if self.mode == "poisson":
                return [Poisson(rate=self.dist.rate[k]) for k in range(self.n_states)]
            return [LogNormal(mean=self.dist.mean[k], std=self.dist.stddev[k]) for k in range(self.n_states)]

        probs = F.softmax(logits, dim=-1)
        return torch.log(probs) if log else probs

    def sample(
        self,
        n_samples: int = 1,
        context: Optional[Union[torch.Tensor, utils.ContextualVariables]] = None,
    ) -> torch.Tensor:
        dists = self.forward(context=context, return_dist=True)
        return torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)

    def estimate_duration_pdf(
        self,
        durations: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
        inplace: bool = False,
    ) -> "NeuralDuration":
        if self.encoder is not None and theta is not None:
            params = self._contextual_params(durations, theta)
        else:
            params = self._compute_mle(durations, posterior)

        if inplace:
            self._assign_params(params)
            return self

        return NeuralDuration(
            n_states=self.n_states,
            mode=self.mode,
            encoder=self.encoder,
            context_mode=self.context_mode,
            max_duration=self.max_duration,
            device=self.device,
            **params
        )

    def _compute_mle(self, durations: torch.Tensor, posterior: torch.Tensor):
        weights = posterior.sum(dim=1, keepdim=True).clamp_min(EPS)
        weighted_mean = (posterior @ durations.unsqueeze(-1)) / weights

        if self.mode == "poisson":
            return {"rate": weighted_mean.squeeze(-1)}
        log_dur = durations.clamp_min(EPS).log().unsqueeze(-1)
        weighted_log_mean = (posterior @ log_dur) / weights
        weighted_log_var = (posterior @ ((log_dur - weighted_log_mean) ** 2)) / weights
        return {"mean": weighted_log_mean.squeeze(-1), "std": weighted_log_var.sqrt().squeeze(-1)}

    def _contextual_params(self, durations: torch.Tensor, context: Union[torch.Tensor, utils.ContextualVariables]):
        if isinstance(context, utils.ContextualVariables):
            X = context.X
            if isinstance(X, dict):
                encoded = self.encoder(**X)
            elif isinstance(X, tuple):
                encoded = self.encoder(*X)
            else:
                encoded = self.encoder(X)
        else:
            encoded = self.encoder(context)

        if self.mode == "poisson":
            return {"rate": F.softplus(encoded).squeeze(-1)}
        mean, log_std = torch.chunk(encoded, 2, dim=-1)
        return {"mean": mean.squeeze(-1), "std": log_std.exp().clamp_min(EPS).squeeze(-1)}

    def _assign_params(self, params: dict):
        if self.mode == "poisson":
            self.dist = Poisson(params["rate"].clamp_min(EPS))
        else:
            self.dist = LogNormal(params["mean"], params["std"].clamp_min(EPS))

    def log_prob(self, durations: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(durations.unsqueeze(0))

    def to(self, device: Union[str, torch.device]) -> "NeuralDuration":
        super().to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        return self
