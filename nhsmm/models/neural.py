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

        if emission_type == "gaussian":
            mu, cov = params
            n_states, n_features = mu.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type="gaussian")
            self.mu = nn.Parameter(mu.to(self.device, self.dtype))
            self.cov = nn.Parameter(cov.to(self.device, self.dtype))
        else:
            logits = params
            n_states, n_features = logits.shape
            super().__init__(n_states=n_states, n_features=n_features, emission_type=emission_type)
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))

        self.encoder = encoder

    def contextual_params(self, theta: Optional[utils.ContextualVariables] = None):
        """Return context-modulated emission parameters."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        X = getattr(theta, "X", theta)
        out = self.encoder(*X) if isinstance(X, tuple) else self.encoder(**X) if isinstance(X, dict) else self.encoder(X)

        if self.emission_type == "gaussian":
            mu, cov = out
            return mu.to(self.device, self.dtype), cov.to(self.device, self.dtype)
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, self.dtype)
        X = X.to(self.device, self.dtype)

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
            cov = torch.stack([torch.eye(n_features, device=device, dtype=dtype) for _ in range(n_states)])
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

    def __init__(self, n_states: int, mode: str = "poisson",
                 rate: Optional[torch.Tensor] = None, mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None, encoder: Optional[nn.Module] = None,
                 context_mode: str = "none", max_duration: int = 20,
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1
        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=self.dtype))

    def _contextual_params(self, context=None):
        X = getattr(context, "X", context)
        encoded = self.encoder(X) if self.encoder else None
        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        mean, log_std = torch.chunk(encoded, 2, dim=-1) if encoded is not None else (self.mean, self.std.log())
        return mean.squeeze(-1), log_std.exp().clamp_min(EPS).squeeze(-1)

    def forward(self, context=None, log=False, return_dist=False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=self.dtype)
        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
        else:
            mean, std = self._contextual_params(context)
            logits = -0.5 * ((durations.unsqueeze(0) - mean.unsqueeze(-1)) / std.unsqueeze(-1))**2 - torch.log(std.unsqueeze(-1)) - 0.5 * torch.log(2 * torch.pi)
        probs = F.softmax(logits, dim=-1)
        if return_dist:
            return [Poisson(rate[k]) if self.mode=="poisson" else LogNormal(mean[k], std[k]) for k in range(self.n_states)]
        return torch.log(probs) if log else probs

    def sample(self, n_samples=1, context=None, state_indices=None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    def to(self, device):
        super().to(device)
        if self.encoder:
            self.encoder.to(device)
        return self


class NeuralTransition(Transition):
    """Neural/contextual transition distribution for HSMM states."""

    def __init__(self, n_states: int, logits: Optional[torch.Tensor] = None,
                 encoder: Optional[nn.Module] = None, context_mode: str = "none",
                 device: Optional[torch.device] = None, dtype: torch.dtype = DTYPE):
        super().__init__(n_states=n_states, device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, self.dtype))

    def contextual_logits(self, theta=None):
        if self.encoder is None or theta is None:
            return self.logits
        X = getattr(theta, "X", theta)
        out = self.encoder(X)
        return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    def update(self, posterior, inplace=True):
        logits = torch.log((posterior + EPS).to(self.device, self.dtype))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(n_states=self.n_states, logits=logits, encoder=self.encoder,
                                context_mode=self.context_mode, device=self.device, dtype=self.dtype)

    @classmethod
    def initialize(cls, n_states, alpha=1.0, batch=1, encoder=None, device=None, dtype=DTYPE):
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        probs = torch.distributions.Dirichlet(torch.ones(n_states, device=device, dtype=dtype)*alpha).sample([batch])
        if batch == 1:
            probs = probs.squeeze(0)
        return cls(n_states=n_states, logits=torch.log(probs.clamp_min(EPS)), encoder=encoder, device=device, dtype=dtype)


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
        if self._context is None:
            return theta
        if theta is None:
            return self._context
        return torch.cat([theta, self._context], dim=-1)

    # ----------------------
    # Emission PDF
    # ----------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        nS, nF = self.n_states, self.n_features
        if self.emission_type == "categorical":
            logits = torch.full((nS, nF), 1.0 / nF, dtype=self.dt, device=self.device).log()
            return Categorical(logits=logits)
        elif self.emission_type == "gaussian":
            mean = torch.zeros(nS, nF, dtype=self.dt, device=self.device)
            var = torch.full((nS, nF), self.min_covar, dtype=self.dt, device=self.device)
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[torch.Tensor] = None) -> Distribution:
        K, F = self.n_states, self.n_features
        dt, dev = self.dt, self.device
        if self.emission_type == "categorical":
            probs = (posterior.T @ X).to(dtype=dt, device=dev)
            probs /= probs.sum(dim=1, keepdim=True) + 1e-12
            return Categorical(probs=probs.clamp_min(1e-8))
        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0, keepdim=True) + 1e-12
            weights = posterior / Nk
            mean = weights.T @ X
            diff = X.unsqueeze(1) - mean.unsqueeze(0)
            w = posterior.unsqueeze(-1)
            cov = torch.einsum("tkf,tkh->kfh", w * diff, diff) / Nk.squeeze(0).unsqueeze(-1).unsqueeze(-1)
            cov += self.min_covar * torch.eye(F, dtype=dt, device=dev).unsqueeze(0)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    # ----------------------
    # Contextual hooks
    # ----------------------
    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor], scale: float = 0.1) -> Distribution:
        pdf = self.pdf
        if pdf is None:
            return pdf
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return pdf

        if isinstance(pdf, Categorical):
            # compute delta logits in batch
            if self.ctx_emission:
                delta = self.ctx_emission(theta_combined).view(-1, self.n_states, self.n_features)
            else:
                delta = theta_combined[:, :self.n_states * self.n_features].view(-1, self.n_states, self.n_features)
            delta_logits = scale * torch.tanh(delta).mean(dim=0)
            return Categorical(logits=(pdf.logits + delta_logits).to(dtype=self.dt, device=self.device))

        elif isinstance(pdf, MultivariateNormal):
            K, F = pdf.mean.shape
            total_dim = K * F
            if self.ctx_emission:
                mean_shift = self.ctx_emission(theta_combined).view(-1, total_dim)
            else:
                mean_shift = theta_combined[:, :total_dim]
            mean_shift = scale * mean_shift.mean(dim=0).view(K, F)
            return MultivariateNormal(
                loc=(pdf.mean + mean_shift).to(dtype=self.dt, device=self.device),
                covariance_matrix=pdf.covariance_matrix.to(dtype=self.dt, device=self.device)
            )

        else:
            raise ValueError(f"Unsupported PDF type {type(pdf)}")

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        logits = self.duration_module.logits.to(dtype=self.dt, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(logits / self.duration_module.temperature, dim=-1)

        if self.ctx_duration:
            delta = self.ctx_duration(theta_combined).view(-1, self.n_states, self.max_duration)
        else:
            delta = theta_combined[:, :self.n_states * self.max_duration].view(-1, self.n_states, self.max_duration)

        delta_mean = delta.mean(dim=0)
        return F.softmax(logits + scale * torch.tanh(delta_mean), dim=-1)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor], scale: float = 0.1) -> torch.Tensor:
        logits = self.transition_module.logits.to(dtype=self.dt, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(logits / self.transition_module.temperature, dim=-1)

        if self.ctx_transition:
            delta = self.ctx_transition(theta_combined).view(-1, self.n_states, self.n_states)
        else:
            delta = theta_combined[:, :self.n_states * self.n_states].view(-1, self.n_states, self.n_states)

        delta_mean = delta.mean(dim=0)
        return F.softmax(logits + scale * torch.tanh(delta_mean), dim=-1)

    # ----------------------
    # Encoder
    # ----------------------
    def encode_observations(self, X: torch.Tensor, detach=True) -> Optional[torch.Tensor]:
        if self.encoder is None:
            return None
        inp = X if X.ndim == 3 else X.unsqueeze(0)
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]
        if detach:
            out = out.detach()
        out = out.to(dtype=self.dt, device=self.device)
        if out.ndim == 3:
            return out.mean(dim=1)
        return out if out.ndim == 2 else out.unsqueeze(0)

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(self, X: torch.Tensor, return_pdf: bool = False, context: Optional[torch.Tensor] = None, context_ids: Optional[torch.Tensor] = None):
        prev_ctx = self._context
        ctx_emb = self.context_embedding(context_ids) if context_ids is not None and self.context_embedding else None
        if ctx_emb is not None and ctx_emb.ndim == 3:
            ctx_emb = ctx_emb.mean(dim=1)
        combined_ctx = ctx_emb if context is None else (context if ctx_emb is None else torch.cat([context, ctx_emb], dim=-1))
        if combined_ctx is not None:
            self.set_context(combined_ctx)
        theta = self.encode_observations(X)
        if return_pdf:
            pdf = self._contextual_emission_pdf(X, theta)
            self._context = prev_ctx
            return pdf
        self._context = prev_ctx
        return theta

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().predict(X, *args, **kwargs)

    # ----------------------
    # Emission initializer
    # ----------------------
    def initialize_emissions(self, X, method: str = "moment"):
        K = self.n_states
        X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=self.dt, device=self.device)
        X = X.to(dtype=self.dt, device=self.device)

        if self.emission_type == "gaussian":
            T, F = X.shape
            if method == "moment":
                mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)
            elif method == "kmeans":
                X_np = X.detach().cpu().numpy().reshape(-1, F) if X.ndim == 2 else X.detach().cpu().numpy().reshape(-1, 1)
                km = KMeans(n_clusters=K, n_init=10, random_state=0)
                labels = km.fit_predict(X_np)
                labels_torch = torch.tensor(labels, dtype=torch.long, device=self.device)
                mu_init = torch.zeros(K, F, dtype=self.dt, device=self.device)
                var_init = torch.zeros(K, F, dtype=self.dt, device=self.device)
                for k in range(K):
                    mask = labels_torch == k
                    Nk = mask.sum()
                    mu_init[k] = X[mask].mean(dim=0) if Nk > 0 else X.mean(dim=0)
                    var_init[k] = X[mask].var(dim=0, unbiased=False) if Nk > 0 else X.var(dim=0, unbiased=False)
            else:
                raise ValueError(f"Unknown initialization method '{method}'")
            var_init = torch.clamp(var_init, min=self.min_covar)
            self.emission_module.mu.data.copy_(mu_init)
            self.emission_module.log_var.data.copy_(torch.log(torch.expm1(var_init)))
            self._params['emission_pdf'] = MultivariateNormal(loc=mu_init, covariance_matrix=torch.diag_embed(var_init))

        elif self.emission_type == "categorical":
            if X.ndim == 1 or X.shape[1] == 1:
                labels = X.squeeze(-1)
                if labels.max() >= K:
                    raise ValueError(f"Label value {labels.max().item()} exceeds n_states={K}")
                counts = torch.bincount(labels, minlength=K).to(dtype=self.dt, device=self.device)
            else:
                counts = X.sum(dim=0).to(dtype=self.dt, device=self.device)
            probs = counts / counts.sum()
            logits = probs.clamp_min(1e-8).log().unsqueeze(0).repeat(K, 1)
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.log_var = nn.Parameter(torch.zeros_like(logits))
            self._params['emission_pdf'] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

