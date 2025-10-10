# models/neural_trainable.py
from __future__ import annotations
from typing import Optional
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, MultivariateNormal

from nhsmm.models.base import BaseHSMM, DTYPE
from nhsmm.utilities import utils


# -----------------------------
# Default modules
# -----------------------------
class DefaultEmission(nn.Module):
    """Gaussian emission with optional context modulation."""
    def __init__(self, n_states, n_features, min_covar=1e-3, context_dim=None):
        super().__init__()
        self.n_states, self.n_features, self.min_covar = n_states, n_features, min_covar
        self.mu = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        self.log_var = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, n_states * n_features),
            nn.Tanh(),
            nn.Linear(n_states * n_features, n_states * n_features)
        ) if context_dim is not None else None

    def forward(self, context: Optional[torch.Tensor] = None):
        mu, var = self.mu, F.softplus(self.log_var) + self.min_covar
        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.n_features)
            mu = mu + 0.1 * torch.tanh(delta)
        return mu, var


class DefaultDuration(nn.Module):
    """Per-state learnable duration probabilities."""
    def __init__(self, n_states, max_duration=20, context_dim=None, temperature=1.0):
        super().__init__()
        self.n_states, self.max_duration, self.temperature = n_states, max_duration, temperature
        self.logits = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE))
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, n_states * max_duration),
            nn.Tanh(),
            nn.Linear(n_states * max_duration, n_states * max_duration)
        ) if context_dim is not None else None

    def forward(self, context: Optional[torch.Tensor] = None):
        logits = self.logits
        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.max_duration)
            logits = logits + 0.1 * torch.tanh(delta)
        return F.softmax(logits / self.temperature, dim=-1)


class DefaultTransition(nn.Module):
    """Per-state learnable transition probabilities."""
    def __init__(self, n_states, context_dim=None, temperature=1.0):
        super().__init__()
        self.n_states, self.temperature = n_states, temperature
        self.logits = nn.Parameter(torch.zeros(n_states, n_states, dtype=DTYPE))
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, n_states * n_states),
            nn.Tanh(),
            nn.Linear(n_states * n_states, n_states * n_states)
        ) if context_dim is not None else None

    def forward(self, context: Optional[torch.Tensor] = None):
        logits = self.logits
        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.n_states)
            logits = logits + 0.1 * torch.tanh(delta)
        return F.softmax(logits / self.temperature, dim=-1)


class NeuralHSMM(BaseHSMM, nn.Module):
    """Trainable NeuralHSMM with EM, Viterbi, context, and gradient support."""

    def __init__(self, n_states, max_duration, n_features, alpha=1.0, seed=None,
                 encoder: Optional[nn.Module] = None, emission_type="gaussian",
                 min_covar=1e-3, device: Optional[torch.device] = None,
                 context_dim: Optional[int] = None, **kwargs):

        nn.Module.__init__(self)
        self._params = {'emission_type': emission_type.lower()}
        self.device = device or torch.device("cpu")
        self.n_features = n_features
        self.min_covar = min_covar
        self.encoder = encoder

        # Context
        self.context_dim = context_dim
        self._context: Optional[torch.Tensor] = None
        self.ctx_A, self.ctx_D, self.ctx_E = None, None, None
        self.context_embedding = None

        if context_dim is not None and kwargs.get('n_context_states', None) is not None:
            n_ctx = kwargs['n_context_states']
            self.context_embedding = nn.Embedding(n_ctx, context_dim)
            nn.init.normal_(self.context_embedding.weight, mean=0.0, std=1e-3)

        if context_dim is not None:
            self.ctx_A = nn.Linear(context_dim, n_states * n_states, bias=True)
            self.ctx_D = nn.Linear(context_dim, n_states * max_duration, bias=True)
            self.ctx_E = nn.Linear(context_dim, n_states * n_features, bias=True)
            for m in [self.ctx_A, self.ctx_D, self.ctx_E]:
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
                m.to(dtype=DTYPE, device=self.device)

        # BaseHSMM initialization
        super().__init__(n_states=n_states, max_duration=max_duration, alpha=alpha, seed=seed)
        self._params['emission_pdf'] = self.sample_emission_pdf()

        # Default modules
        self.emission_module = DefaultEmission(n_states, n_features, min_covar, context_dim)
        self.duration_module = DefaultDuration(n_states, max_duration, context_dim)
        self.transition_module = DefaultTransition(n_states, context_dim)

    # ----------------------
    # Core properties
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
        ctx = context.detach().to(dtype=DTYPE, device=self.device)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def clear_context(self):
        self._context = None

    def _combine_context(self, theta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self._context is not None and theta is not None:
            return torch.cat([theta, self._context], dim=-1)
        return theta if theta is not None else self._context

    # ----------------------
    # Emission PDF functions
    # ----------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        nS, nF, dev, dt = self.n_states, self.n_features, self.device, DTYPE
        if self.emission_type == "categorical":
            logits = torch.full((nS, nF), 1.0 / nF, dtype=dt, device=dev).log()
            return Categorical(logits=logits)
        elif self.emission_type == "gaussian":
            mean = torch.zeros(nS, nF, dtype=dt, device=dev)
            var = torch.full((nS, nF), self.min_covar, dtype=dt, device=dev)
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(self, X, posterior, theta=None) -> Distribution:
        K, F, dt, dev = self.n_states, self.n_features, DTYPE, self.device
        if self.emission_type == "categorical":
            probs = (posterior.T @ X)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
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
    def _contextual_emission_pdf(self, X, theta, scale: float = 0.1) -> Distribution:
        """
        Returns the context-modulated emission PDF.
        Vectorized, GPU-friendly, supports both Gaussian and Categorical emissions.
        """
        pdf = self.pdf
        if pdf is None:
            return pdf

        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return pdf

        if isinstance(pdf, Categorical):
            # Categorical: delta shape [n_states, n_features]
            if self.ctx_E is not None:
                delta = self.ctx_E(theta_combined.unsqueeze(0)).squeeze(0).view(self.n_states, self.n_features)
            else:
                delta = theta_combined[:self.n_states * self.n_features].view(self.n_states, self.n_features)
            delta = scale * torch.tanh(delta)
            # Broadcast sum over classes
            delta_sum = delta.sum(dim=1, keepdim=True)
            return Categorical(logits=pdf.logits + delta_sum.expand_as(pdf.logits))

        elif isinstance(pdf, MultivariateNormal):
            K, F = pdf.mean.shape
            total_dim = K * F
            if self.ctx_E is not None:
                mean_shift = self.ctx_E(theta_combined.unsqueeze(0)).squeeze(0)[:total_dim]
            else:
                mean_shift = theta_combined[:total_dim]
            mean_shift = scale * mean_shift.view(K, F)
            return MultivariateNormal(loc=pdf.mean + mean_shift, covariance_matrix=pdf.covariance_matrix)

        else:
            raise ValueError(f"Unsupported PDF type {type(pdf)}")

    def _contextual_transition_matrix(self, theta, scale: float = 0.1) -> torch.Tensor:
        """
        Returns the context-modulated transition matrix.
        Fully vectorized; applies smooth tanh modulation.
        """
        logits = self.transition_module.logits.to(dtype=DTYPE, device=self.device)
        theta_combined = self._combine_context(theta)

        if theta_combined is None:
            return F.softmax(logits / self.transition_module.temperature, dim=-1)

        if self.ctx_A is not None:
            delta = self.ctx_A(theta_combined.unsqueeze(0)).squeeze(0).view(self.n_states, self.n_states)
        else:
            delta = theta_combined[:self.n_states * self.n_states].view(self.n_states, self.n_states)

        delta = scale * torch.tanh(delta)
        return F.softmax(logits + delta, dim=-1)

    def _contextual_duration_pdf(self, theta, scale: float = 0.1) -> torch.Tensor:
        """
        Returns the context-modulated duration probabilities.
        Fully vectorized; applies smooth tanh modulation.
        """
        logits = self.duration_module.logits.to(dtype=DTYPE, device=self.device)
        theta_combined = self._combine_context(theta)

        if theta_combined is None:
            return F.softmax(logits / self.duration_module.temperature, dim=-1)

        if self.ctx_D is not None:
            delta = self.ctx_D(theta_combined.unsqueeze(0)).squeeze(0).view(self.n_states, self.max_duration)
        else:
            delta = theta_combined[:self.n_states * self.max_duration].view(self.n_states, self.max_duration)

        delta = scale * torch.tanh(delta)
        return F.softmax(logits + delta, dim=-1)

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
        out = out.to(dtype=DTYPE, device=self.device)
        if out.ndim == 3:
            return out.mean(dim=1)
        elif out.ndim == 2:
            return out
        elif out.ndim == 1:
            return out.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape}")

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(self, X: torch.Tensor, return_pdf: bool = False, context: Optional[torch.Tensor] = None, context_ids: Optional[torch.Tensor] = None):
        prev_ctx = self._context
        ctx_emb = None
        if context_ids is not None and self.context_embedding is not None:
            ctx_emb = self.context_embedding(context_ids)
            if ctx_emb.ndim == 3:
                ctx_emb = ctx_emb.mean(dim=1)
        combined_ctx = None
        if context is not None and ctx_emb is not None:
            combined_ctx = torch.cat([context, ctx_emb], dim=-1)
        elif context is not None:
            combined_ctx = context
        elif ctx_emb is not None:
            combined_ctx = ctx_emb
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
    # Emission initializer (extended for categorical)
    # ----------------------
    def initialize_emissions(self, X, method: str = "moment"):
        K = self.n_states

        # ensure X is a torch.Tensor on correct device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=DTYPE, device=self.device)
        else:
            X = X.to(dtype=DTYPE, device=self.device)

        if self.emission_type == "gaussian":
            T, F = X.shape

            if method == "moment":
                mu_init = X.mean(dim=0, keepdim=True).repeat(K, 1)
                var_init = X.var(dim=0, unbiased=False, keepdim=True).repeat(K, 1)

            elif method == "kmeans":
                # Use sklearn KMeans
                X_np = X.detach().cpu().numpy()
                if X_np.ndim == 1:
                    X_np = X_np.reshape(-1, 1)
                km = KMeans(n_clusters=K, n_init=10, random_state=0)
                labels = km.fit_predict(X_np)
                labels_torch = torch.tensor(labels, dtype=torch.long, device=self.device)

                # Vectorized mean and variance per cluster
                mu_init = torch.zeros(K, F, dtype=DTYPE, device=self.device)
                var_init = torch.zeros(K, F, dtype=DTYPE, device=self.device)
                for k in range(K):
                    mask = labels_torch == k
                    Nk = mask.sum()
                    if Nk == 0:
                        # fallback to global mean/var if cluster empty
                        mu_init[k] = X.mean(dim=0)
                        var_init[k] = X.var(dim=0, unbiased=False)
                    else:
                        mu_init[k] = X[mask].mean(dim=0)
                        var_init[k] = X[mask].var(dim=0, unbiased=False)
            else:
                raise ValueError(f"Unknown initialization method '{method}'")

            var_init = torch.clamp(var_init, min=self.min_covar)
            self.emission_module.mu.data.copy_(mu_init)
            self.emission_module.log_var.data.copy_(torch.log(torch.expm1(var_init)))
            cov = torch.diag_embed(var_init)
            self._params['emission_pdf'] = MultivariateNormal(loc=mu_init, covariance_matrix=cov)

        elif self.emission_type == "categorical":
            # X assumed integer labels or one-hot
            if X.ndim == 1 or X.shape[1] == 1:
                labels = X.squeeze(-1)
                if labels.max() >= K:
                    raise ValueError(f"Label value {labels.max().item()} exceeds n_states={K}")
                counts = torch.bincount(labels, minlength=K).to(dtype=DTYPE, device=self.device)
            else:
                counts = X.sum(dim=0).to(dtype=DTYPE, device=self.device)

            probs = counts / counts.sum()
            probs = probs.clamp_min(1e-8)
            logits = probs.log().unsqueeze(0).repeat(K, 1)

            # Dummy mu/log_var for categorical
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.log_var = nn.Parameter(torch.zeros_like(logits))
            self._params['emission_pdf'] = Categorical(logits=logits)

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")
