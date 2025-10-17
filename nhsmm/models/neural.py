# models/neural.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Independent, Categorical, Poisson, LogNormal, MultivariateNormal

from typing import Optional, Union, Tuple
from sklearn.cluster import KMeans

from nhsmm.defaults import DTYPE, EPS, Duration, Emission, Transition
from nhsmm.models.hsmm import HSMM
from nhsmm.utilities import utils


class NeuralEmission(Emission):
    """Neural/contextual emission distribution for HSMM states."""

    def __init__(
        self,
        emission_type: str,
        params: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.context_mode = context_mode.lower()
        if self.context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

        emission_type = emission_type.lower()
        self.emission_type = emission_type
        self.encoder = encoder

        if emission_type == "gaussian":
            mu, cov = params
            n_states, n_features = mu.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type="gaussian")
            self.mu = nn.Parameter(mu.to(self.device, self.dtype))
            # Ensure diagonal covariance if 2D input
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            self.cov = nn.Parameter(cov.to(self.device, self.dtype))
        else:
            logits = params
            n_states, n_features = logits.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type=emission_type)
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))

    def contextual_params(self, theta: Optional[utils.ContextualVariables] = None):
        """Return context-modulated emission parameters."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        X = getattr(theta, "X", theta)
        if isinstance(X, tuple):
            out = self.encoder(*X)
        elif isinstance(X, dict):
            out = self.encoder(**X)
        else:
            out = self.encoder(X)

        if self.emission_type == "gaussian":
            mu, cov = out
            # Ensure proper device/dtype
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            return mu.to(self.device, self.dtype), cov.to(self.device, self.dtype)
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, self.dtype)
        X = X.to(self.device, self.dtype)

        # flatten batch if necessary
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
        if posterior.ndim > 2:
            posterior = posterior.reshape(-1, posterior.shape[-1])

        if self.emission_type == "gaussian":
            counts = posterior.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (posterior.T @ X) / counts
            cov_new = torch.stack([
                ((X - mu_new[s]).T * posterior[:, s]) @ (X - mu_new[s]) / counts[s, 0]
                + torch.eye(self.n_features, device=self.device, dtype=self.dtype) * EPS
                for s in range(self.n_states)
            ])
            if inplace:
                self.mu.data.copy_(mu_new)
                self.cov.data.copy_(cov_new)
                return self
            return NeuralEmission("gaussian", (mu_new, cov_new), encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device, dtype=self.dtype)
        else:
            X_onehot = X if X.ndim == 2 else F.one_hot(X.long(), num_classes=self.n_features).float()
            weighted_counts = posterior.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device, dtype=self.dtype)

    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None, n_categories: int = None,
                   alpha: float = 1.0, encoder: Optional[nn.Module] = None, device: Optional[torch.device] = None,
                   dtype: torch.dtype = DTYPE):
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=dtype) * 0.1
            cov = torch.stack([torch.ones(n_features, device=device, dtype=dtype) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder, device=device, dtype=dtype)
        else:
            logits = torch.distributions.Dirichlet(
                torch.ones(n_categories, device=device, dtype=dtype) * alpha
            ).sample([n_states])
            return cls(emission_type.lower(), logits.clamp_min(EPS).log(), encoder=encoder,
                       device=device, dtype=dtype)

    def __repr__(self):
        if self.emission_type == "gaussian":
            return f"NeuralEmission(Gaussian, n_states={self.n_states}, n_features={self.n_features})"
        return f"NeuralEmission({self.emission_type.capitalize()}, n_states={self.n_states}, n_categories={self.n_features})"


class NeuralDuration(Duration):
    """Neural/contextual duration distribution for HSMM states."""

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
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1

        if self.mode == "poisson":
            self.rate = nn.Parameter(
                rate if rate is not None else torch.ones(n_states, device=self.device, dtype=self.dtype)
            )
        else:
            self.mean = nn.Parameter(
                mean if mean is not None else torch.ones(n_states, device=self.device, dtype=self.dtype)
            )
            self.std = nn.Parameter(
                std if std is not None else torch.ones(n_states, device=self.device, dtype=self.dtype)
            )

    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        """Return context-modulated parameters."""
        X = getattr(context, "X", context)
        encoded = None
        if self.encoder and X is not None:
            encoded = self.encoder(X)
            if isinstance(encoded, tuple):
                encoded = encoded[0]

        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded, 2, dim=-1)
            else:
                mean, log_std = self.mean, self.std.log()
            return mean.squeeze(-1), log_std.exp().clamp_min(EPS).squeeze(-1)

    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=self.dtype)
        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 2 \
                     - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)
        probs = F.softmax(logits, dim=-1)

        if return_dist:
            if self.mode == "poisson":
                return [Poisson(rate[k]) for k in range(self.n_states)]
            else:
                return [LogNormal(mean[k], std[k]) for k in range(self.n_states)]

        return torch.log(probs) if log else probs

    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        """Sample durations for each state with optional context."""
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    def to(self, device: torch.device):
        """Move duration parameters and encoder to device."""
        super().to(device)
        self.device = device
        if hasattr(self, "rate"):
            self.rate = nn.Parameter(self.rate.to(device))
        if hasattr(self, "mean"):
            self.mean = nn.Parameter(self.mean.to(device))
        if hasattr(self, "std"):
            self.std = nn.Parameter(self.std.to(device))
        if self.encoder:
            self.encoder.to(device)
        return self


class NeuralTransition(Transition):
    """Neural/contextual transition distribution for HSMM states."""

    def __init__(
        self,
        n_states: int,
        logits: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(n_states=n_states, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))
        else:
            # initialize uniform transitions
            self.logits = nn.Parameter(torch.zeros(n_states, n_states, device=self.device, dtype=self.dtype))

    def contextual_logits(self, theta: Optional[torch.Tensor] = None):
        """Return transition logits modulated by context."""
        if self.encoder is None or theta is None:
            return self.logits

        X = getattr(theta, "X", theta)
        out = self.encoder(X)
        if isinstance(out, tuple):
            out = out[0]  # handle encoders returning (output, hidden)

        # reshape for batched output if needed
        if out.ndim > 2:
            out = out.view(-1, self.n_states, self.n_states)

        return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior: torch.Tensor, inplace: bool = True):
        """Update logits from posterior probabilities."""
        logits = torch.log((posterior + EPS).to(self.device, self.dtype))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(
            n_states=self.n_states,
            logits=logits,
            encoder=self.encoder,
            context_mode=self.context_mode,
            device=self.device,
            dtype=self.dtype
        )

    @classmethod
    def initialize(
        cls,
        n_states: int,
        alpha: float = 1.0,
        batch: int = 1,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE
    ):
        """Initialize transition logits with Dirichlet samples."""
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        probs = torch.distributions.Dirichlet(
            torch.ones(n_states, device=device, dtype=dtype) * alpha
        ).sample([batch])

        if batch == 1:
            probs = probs.squeeze(0)

        return cls(
            n_states=n_states,
            logits=torch.log(probs.clamp_min(EPS)),
            encoder=encoder,
            device=device,
            dtype=dtype
        )

    def to(self, device: torch.device):
        """Move parameters and encoder to the given device."""
        super().to(device)
        self.device = device
        if hasattr(self, "logits"):
            self.logits = nn.Parameter(self.logits.to(device))
        if self.encoder:
            self.encoder.to(device)
        return self


class NeuralHSMM(HSMM, nn.Module):
    """Trainable NeuralHSMM with EM, Viterbi, context, and gradient support."""

    def __init__(
        self,
        n_states: int,
        max_duration: int,
        n_features: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        emission_type: str = "gaussian",
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        context_dim: Optional[int] = None,
        min_covar: float = 1e-3,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.device = device or torch.device("cpu")
        self.dt = DTYPE
        self.n_features = n_features
        self.min_covar = min_covar
        self.encoder = encoder
        self._params = {'emission_type': emission_type.lower()}

        # Neural modules
        self.transition_module = NeuralTransition.initialize(n_states, alpha=alpha, encoder=encoder)
        self.duration_module = NeuralDuration(n_states, mode="poisson", encoder=encoder, max_duration=max_duration)
        self.emission_module = NeuralEmission.initialize(
            emission_type=emission_type.lower(),
            n_states=n_states,
            n_features=n_features,
            encoder=encoder,
        )

        # Context
        self.context_dim = context_dim
        self._context: Optional[torch.Tensor] = None
        self.context_embedding: Optional[nn.Embedding] = None
        self.ctx_transition: Optional[nn.Linear] = None
        self.ctx_duration: Optional[nn.Linear] = None
        self.ctx_emission: Optional[nn.Linear] = None

        if context_dim is not None and kwargs.get("n_context_states") is not None:
            n_ctx = kwargs['n_context_states']
            self.context_embedding = nn.Embedding(n_ctx, context_dim)
            nn.init.normal_(self.context_embedding.weight, mean=0.0, std=1e-3)

        if context_dim is not None:
            self.ctx_transition = nn.Linear(context_dim, n_states * n_states)
            self.ctx_duration = nn.Linear(context_dim, n_states * max_duration)
            self.ctx_emission = nn.Linear(context_dim, n_states * n_features)
            for m in [self.ctx_transition, self.ctx_duration, self.ctx_emission]:
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
                m.to(dtype=self.dt, device=self.device)

        # Base HSMM initialization
        super().__init__(n_states=n_states, n_features=n_features, max_duration=max_duration, alpha=alpha, seed=seed)
        self._params['emission_pdf'] = self.sample_emission_pdf()

    # ----------------------
    # Properties
    # ----------------------
    @property
    def emission_type(self) -> str:
        return self._params.get('emission_type', 'gaussian')

    @property
    def pdf(self) -> Distribution:
        return self._params.get('emission_pdf', None)

    @property
    def dof(self) -> int:
        nS, nD, nF = self.n_states, self.max_duration, self.n_features
        dof = (nS - 1) + nS * (nS - 1) + nS * (nD - 1)
        pdf = self.pdf
        if pdf is not None:
            if isinstance(pdf, Categorical):
                dof += nS * (pdf.logits.shape[1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                dof += nS * (2 * nF)
        return dof

    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: float = 0.1,
        temperature: float = 1.0
    ) -> Distribution:
        """
        Initialize emission distributions for multiple sequences, optionally using data X
        and neural context theta.

        Args:
            X: Optional observation tensor (B, T, F) or (T, F) for data-driven initialization.
            theta: Optional encoded context tensor (B, H) or (H,) to modulate emissions.
            scale: Scaling factor for context modulation.
            temperature: Optional softmax temperature for categorical distributions.

        Returns:
            torch.distributions.Distribution
        """
        nS, nF = self.n_states, self.n_features
        dt, dev = self.dt, self.device

        # Flatten X if needed
        if X is not None:
            X_flat = X.reshape(-1, nF).to(dtype=dt, device=dev)

        # Compute context modulation
        delta_mean = None
        if theta is not None and self.ctx_emission:
            theta_proc = theta.unsqueeze(0) if theta.ndim == 1 else theta
            delta = self.ctx_emission(theta_proc)
            if delta.ndim > 2:
                delta = delta.view(-1, nS, nF)
            delta_mean = delta.mean(dim=0)  # (nS, nF)

        if self.emission_type == "categorical":
            probs = (X_flat.mean(dim=0, keepdim=True).repeat(nS, 1) if X is not None
                     else torch.full((nS, nF), 1.0 / nF, dtype=dt, device=dev))
            if delta_mean is not None:
                probs = probs + scale * torch.tanh(delta_mean)
            return Categorical(probs=F.softmax(probs / temperature, dim=-1).clamp_min(1e-8))

        elif self.emission_type == "gaussian":
            if X is not None:
                mean = X_flat.mean(dim=0, keepdim=True).repeat(nS, 1)
                var = X_flat.var(dim=0, unbiased=False, keepdim=True).repeat(nS, 1).clamp_min(self.min_covar)
            else:
                mean = torch.zeros(nS, nF, dtype=dt, device=dev)
                var = torch.full((nS, nF), self.min_covar, dtype=dt, device=dev)
            if delta_mean is not None:
                mean = mean + scale * torch.tanh(delta_mean)
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        scale: float = 0.1
    ) -> Distribution:
        """
        Estimate context-aware emission distribution given observations and posterior responsibilities.

        Args:
            X: Observation tensor of shape (T, F)
            posterior: Posterior probabilities of shape (T, K)
            theta: Optional context tensor for neural modulation (T, H) or (B, H)
            scale: Scaling factor for context effect

        Returns:
            torch.distributions.Distribution (Categorical or MultivariateNormal)
        """
        K, F = self.n_states, self.n_features
        dt, dev = self.dt, self.device

        X = X.to(dtype=dt, device=dev)
        posterior = posterior.to(dtype=dt, device=dev)

        # Apply context modulation if provided
        delta = None
        if theta is not None:
            theta_proc = theta.to(dtype=dt, device=dev)
            if self.ctx_emission:
                delta = self.ctx_emission(theta_proc)
                if delta.ndim > 2:
                    delta = delta.mean(dim=0)
            else:
                delta = theta_proc[:, :K*F] if theta_proc.ndim == 2 else theta_proc[0, :K*F]
            delta = scale * torch.tanh(delta)

        if self.emission_type == "categorical":
            probs = posterior.T @ X  # (K, F)
            probs /= probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            if delta is not None:
                delta = delta.view(K, F)
                probs = (probs + delta).clamp_min(1e-8)
            return Categorical(probs=probs)

        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0, keepdim=True).clamp_min(1e-12)  # (1, K)
            weights = posterior / Nk
            mean = weights.T @ X  # (K, F)

            diff = X.unsqueeze(1) - mean.unsqueeze(0)  # (T, K, F)
            w_diff = diff * posterior.unsqueeze(-1)
            cov = torch.einsum("tkf,tkh->kfh", w_diff, diff) / Nk.squeeze(0).unsqueeze(-1).unsqueeze(-1)
            cov += self.min_covar * torch.eye(F, dtype=dt, device=dev).unsqueeze(0)

            if delta is not None:
                mean = mean + delta.view(K, F)

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    # ----------------------
    # Context management
    # ----------------------
    def set_context(self, context: Optional[torch.Tensor]):
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(dtype=self.dt, device=self.device)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def clear_context(self):
        self._context = None

    def _combine_context(self, theta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Safely combine external context with encoder outputs.

        Args:
            theta: Tensor of shape (..., H1) or None.

        Returns:
            Combined tensor of shape (..., H1 + H2) if both exist, else the available tensor.
        """
        if self._context is None:
            return theta
        if theta is None:
            return self._context

        # Ensure batch dimensions match
        if theta.shape[0] != self._context.shape[0]:
            if theta.shape[0] == 1:
                theta = theta.expand(self._context.shape[0], -1)
            elif self._context.shape[0] == 1:
                ctx = self._context.expand(theta.shape[0], -1)
                return torch.cat([theta, ctx], dim=-1)
            else:
                raise ValueError(f"Batch size mismatch: theta {theta.shape[0]} vs _context {self._context.shape[0]}")
        return torch.cat([theta, self._context], dim=-1)

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor], scale: float = 0.1) -> Distribution:
        """
        Compute context-adjusted emission distribution.

        Args:
            X: Observations tensor (T, F) or (B, T, F)
            theta: Encoded context tensor (B, H)
            scale: Scaling factor for context contribution

        Returns:
            Contextually adjusted torch.distributions.Distribution
        """
        pdf = self.pdf
        if pdf is None:
            return pdf

        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return pdf

        dt, dev = self.dt, self.device

        if isinstance(pdf, Categorical):
            delta = self.ctx_emission(theta_combined) if self.ctx_emission else theta_combined[:, :self.n_states * self.n_features]
            delta = delta.view(-1, self.n_states, self.n_features)
            delta_logits = scale * torch.tanh(delta).mean(dim=0)
            return Categorical(logits=(pdf.logits + delta_logits).to(dtype=dt, device=dev))

        elif isinstance(pdf, MultivariateNormal):
            K, F = pdf.mean.shape
            total_dim = K * F
            mean_shift = self.ctx_emission(theta_combined) if self.ctx_emission else theta_combined[:, :total_dim]
            mean_shift = scale * mean_shift.view(-1, total_dim).mean(dim=0).view(K, F)
            return MultivariateNormal(
                loc=(pdf.mean + mean_shift).to(dtype=dt, device=dev),
                covariance_matrix=pdf.covariance_matrix.to(dtype=dt, device=dev)
            )

        else:
            raise TypeError(f"Unsupported PDF type {type(pdf)}")

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor], scale: float = 0.1, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute duration probabilities adjusted by context embeddings.

        Args:
            theta: Encoded context tensor (B, H)
            scale: Scaling factor for context contribution
            temperature: Optional temperature for softmax

        Returns:
            Tensor of shape (n_states, max_duration) with softmax probabilities
        """
        logits = self.duration_module.logits.to(dtype=self.dt, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(logits / temperature, dim=-1)

        delta = self.ctx_duration(theta_combined) if self.ctx_duration else theta_combined[:, :self.n_states * self.max_duration]
        delta = delta.view(-1, self.n_states, self.max_duration)
        delta_mean = delta.mean(dim=0)
        return F.softmax(logits + scale * torch.tanh(delta_mean), dim=-1)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor], scale: float = 0.1, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute transition matrix adjusted by context embeddings.

        Args:
            theta: Encoded context tensor (B, H)
            scale: Scaling factor for context contribution
            temperature: Optional temperature for softmax

        Returns:
            Tensor of shape (n_states, n_states) with softmax probabilities
        """
        logits = self.transition_module.logits.to(dtype=self.dt, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(logits / temperature, dim=-1)

        delta = self.ctx_transition(theta_combined) if self.ctx_transition else theta_combined[:, :self.n_states * self.n_states]
        delta = delta.view(-1, self.n_states, self.n_states)
        delta_mean = delta.mean(dim=0)
        return F.softmax(logits + scale * torch.tanh(delta_mean), dim=-1)

    # ----------------------
    # Encoder
    # ----------------------
    def encode_observations(self, X: torch.Tensor, detach: bool = True) -> Optional[torch.Tensor]:
        """
        Encode input observations using the model's encoder (if available).

        Args:
            X: Input tensor of shape (T, F) or (B, T, F).
            detach: If True, detach encoded tensor from computation graph.

        Returns:
            Encoded tensor of shape (B, H) or None if no encoder is defined.
        """
        if self.encoder is None:
            return None

        # Ensure batch dimension
        inp = X if X.ndim == 3 else X.unsqueeze(0)
        inp = inp.to(dtype=self.dt, device=self.device)

        # Forward through encoder
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]  # handle encoders that return (output, hidden)

        # Detach if requested
        if detach:
            out = out.detach()

        # Ensure correct dtype/device
        out = out.to(dtype=self.dt, device=self.device)

        # Reduce temporal dimension if necessary
        if out.ndim == 3:
            # Average over time dimension
            return out.mean(dim=1)
        elif out.ndim == 2:
            return out
        else:
            return out.unsqueeze(0)

    def forward(
        self,
        X: torch.Tensor,
        return_pdf: bool = False,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through NeuralHSMM encoder with optional context.

        Args:
            X: Observation tensor of shape (T, F) or (B, T, F).
            return_pdf: If True, return the contextualized emission PDF.
            context: Optional external context tensor (B, H) or (H,).
            context_ids: Optional indices for context embeddings.

        Returns:
            Encoded observations tensor (theta) or emission PDF if return_pdf=True.
        """
        # Backup previous context
        prev_ctx = self._context

        # Compute context embeddings if indices provided
        ctx_emb = None
        if context_ids is not None and self.context_embedding is not None:
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:  # average over sequence/time dim if present
                ctx_emb = ctx_emb.mean(dim=1)

        # Combine provided context and embedding
        if context is not None:
            combined_ctx = context if ctx_emb is None else torch.cat([context, ctx_emb], dim=-1)
        else:
            combined_ctx = ctx_emb

        # Set combined context if available
        if combined_ctx is not None:
            self.set_context(combined_ctx)

        # Encode observations
        theta = self.encode_observations(X)

        if return_pdf:
            pdf = self._contextual_emission_pdf(X, theta)
            self._context = prev_ctx  # restore previous context
            return pdf

        # Restore previous context
        self._context = prev_ctx
        return theta

    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:
        """
        Minimal NeuralHSMM-aware predict using parent HSMM.

        Computes context-aware PDFs and passes sequences to HSMM predict.
        """
        # Encode and combine context
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # Compute context-modulated parameters
        pdf = self._contextual_emission_pdf(X, theta)
        transition = self._contextual_transition_matrix(theta)
        duration = self._contextual_duration_pdf(theta)

        # Temporarily patch HSMM attributes
        prev_pdf = self._params['emission_pdf']
        prev_transition = getattr(self, "_transition_logits", None)
        prev_duration = getattr(self, "_duration_logits", None)

        self._params['emission_pdf'] = pdf
        if transition is not None:
            self._transition_logits = transition
        if duration is not None:
            self._duration_logits = duration

        # Call parent HSMM predict (doesn’t take extra kwargs)
        preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

        # Restore original parameters
        self._params['emission_pdf'] = prev_pdf
        if prev_transition is not None:
            self._transition_logits = prev_transition
        if prev_duration is not None:
            self._duration_logits = prev_duration

        return preds

    def initialize_emissions(self, X, method: str = "moment"):
        """
        Initialize emission parameters from data X.

        Args:
            X: Observations (T, F) for Gaussian or (T,) / (T, K) for categorical.
            method: "moment" or "kmeans" for Gaussian initialization.
        """
        K = self.n_states
        dt, dev = self.dt, self.device

        # Ensure tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=dt, device=dev)
        X = X.to(dtype=dt, device=dev)

        if self.emission_type == "gaussian":
            if X.ndim != 2:
                raise ValueError(f"Gaussian emissions require X with shape (T, F), got {X.shape}")
            T, F = X.shape

            if method == "moment":
                mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)
            elif method == "kmeans":
                X_np = X.cpu().numpy()
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=K, n_init=10, random_state=0)
                labels = torch.tensor(km.fit_predict(X_np), device=dev)
                mu_init = torch.zeros(K, F, dtype=dt, device=dev)
                var_init = torch.zeros(K, F, dtype=dt, device=dev)
                for k in range(K):
                    mask = labels == k
                    Nk = mask.sum()
                    if Nk > 0:
                        mu_init[k] = X[mask].mean(dim=0)
                        var_init[k] = X[mask].var(dim=0, unbiased=False)
                    else:
                        mu_init[k] = X.mean(dim=0)
                        var_init[k] = X.var(dim=0, unbiased=False)
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            var_init = torch.clamp(var_init, min=self.min_covar)

            # Set parameters
            if hasattr(self.emission_module, "mu"):
                self.emission_module.mu.data.copy_(mu_init)
            else:
                self.emission_module.mu = nn.Parameter(mu_init)

            if hasattr(self.emission_module, "cov"):
                self.emission_module.cov.data.copy_(torch.diag_embed(var_init))
            else:
                self.emission_module.cov = nn.Parameter(torch.diag_embed(var_init))

            self._params['emission_pdf'] = MultivariateNormal(
                loc=mu_init,
                covariance_matrix=torch.diag_embed(var_init)
            )

        elif self.emission_type in ["categorical", "bernoulli"]:
            # Flatten to 1D for single-column categorical input
            if X.ndim == 1 or X.shape[1] == 1:
                labels = X.squeeze(-1).long()
                if labels.max() >= K:
                    raise ValueError(f"Label value {labels.max().item()} exceeds n_states={K}")
                counts = torch.bincount(labels, minlength=K).to(dtype=dt, device=dev)
            else:
                counts = X.sum(dim=0).to(dtype=dt, device=dev)

            probs = counts / counts.sum()
            logits = probs.clamp_min(1e-8).log().unsqueeze(0).repeat(K, 1)

            # Ensure dummy parameters exist to avoid attribute errors
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.cov = nn.Parameter(torch.zeros_like(logits))
            self._params['emission_pdf'] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

    def decode(
        self,
        X: torch.Tensor,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Decode an observation sequence X into the most likely state sequence using NeuralHSMM.

        Args:
            X: Observation tensor (T, F) or (B, T, F) for batch.
            algorithm: Decoding strategy, either "viterbi" or "map".
            duration_weight: Scaling factor for duration contribution.
            context: Optional context tensor (B, H) or (H,).
            context_ids: Optional context embedding indices.

        Returns:
            np.ndarray of predicted state indices (T,) or (B, T) for batch.
        """
        # Ensure tensor on correct device
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=self.dt, device=self.device)
        else:
            X = X.to(dtype=self.dt, device=self.device)

        # Forward pass: encode observations and combine context
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # Compute context-modulated parameters
        _ = self._contextual_transition_matrix(theta, scale=duration_weight)
        _ = self._contextual_duration_pdf(theta, scale=duration_weight)
        _ = self._contextual_emission_pdf(X, theta)

        # Decode using NeuralHSMM predict
        preds = self.predict(X, algorithm=algorithm, context=theta)

        # Normalize output to numpy array
        if isinstance(preds, list):
            preds = preds[0] if len(preds) > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        elif not torch.is_tensor(preds):
            preds = torch.as_tensor(preds, dtype=torch.long, device=self.device)

        return preds.detach().cpu().numpy()
