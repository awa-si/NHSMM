# nhsmm/hsmm/NeuralHSMM.py
from __future__ import annotations
from typing import Optional

from torch.distributions import Distribution, Categorical, MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
import torch

from nhsmm.models.base import BaseHSMM, DTYPE
from nhsmm.utilities import utils

# -----------------------------
# Default modules (pluggable)
# -----------------------------
class DefaultEmission(nn.Module):
    """
    Gaussian emission model with optional generic context modulation.
    Supports per-state parameters and context vector (continuous or embedded).

    Args:
        n_states (int): Number of HSMM hidden states.
        n_features (int): Feature dimensionality.
        min_covar (float): Min variance for numerical stability.
        context_dim (int, optional): Dimensionality of combined context vector.
    """
    def __init__(self, n_states, n_features, min_covar=1e-3, context_dim=None):
        super().__init__()
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.context_dim = context_dim

        self.mu = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        self.log_var = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * n_features),
                nn.Tanh(),
                nn.Linear(n_states * n_features, n_states * n_features)
            )
        else:
            self.context_net = None

    def forward(self, X: torch.Tensor = None, context: torch.Tensor = None):
        mu, var = self.mu, F.softplus(self.log_var) + self.min_covar

        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.n_features)
            mu = mu + 0.1 * torch.tanh(delta)

        return mu, var


class DefaultDuration(nn.Module):
    """
    Learnable duration module with optional generic context.
    Outputs per-state duration probabilities.

    Args:
        n_states (int): Number of hidden states.
        max_duration (int): Maximum duration length.
        context_dim (int, optional): Dimensionality of context vector.
        temperature (float): Softmax scaling factor.
    """
    def __init__(self, n_states, max_duration=20, context_dim=None, temperature=1.0):
        super().__init__()
        self.n_states = n_states
        self.max_duration = max_duration
        self.temperature = temperature

        self.logits = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE))

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * max_duration),
                nn.Tanh(),
                nn.Linear(n_states * max_duration, n_states * max_duration)
            )
        else:
            self.context_net = None

    def forward(self, context: torch.Tensor = None):
        logits = self.logits
        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.max_duration)
            logits = logits + 0.1 * torch.tanh(delta)
        return F.softmax(logits / self.temperature, dim=-1)


class DefaultTransition(nn.Module):
    """
    Learnable transition matrix with optional generic context.
    Outputs per-state transition probabilities.

    Args:
        n_states (int): Number of hidden states.
        context_dim (int, optional): Dimensionality of context vector.
        temperature (float): Softmax scaling factor.
    """
    def __init__(self, n_states, context_dim=None, temperature=1.0):
        super().__init__()
        self.n_states = n_states
        self.temperature = temperature

        self.logits = nn.Parameter(torch.zeros(n_states, n_states, dtype=DTYPE))

        if context_dim is not None:
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, n_states * n_states),
                nn.Tanh(),
                nn.Linear(n_states * n_states, n_states * n_states)
            )
        else:
            self.context_net = None

    def forward(self, context: torch.Tensor = None):
        logits = self.logits
        if self.context_net is not None and context is not None:
            delta = self.context_net(context).reshape(self.n_states, self.n_states)
            logits = logits + 0.1 * torch.tanh(delta)
        return F.softmax(logits / self.temperature, dim=-1)


# -----------------------------
# NeuralHSMM
# -----------------------------
class NeuralHSMM(BaseHSMM, nn.Module):
    """
    Hidden semi-Markov model with neural contextual encoder (CNN/LSTM/Transformer).

    Extends BaseHSMM:
    - Neural encoder extracts sequence-level context (`theta`) from X
    - Adapts emission, transition, and duration distributions based on context
    - EM, Viterbi, scoring remain fully compatible

    New optional functionality:
    - `context_dim`: if provided, learned linear adapters are created to modulate
      transitions (A), durations (D) and emissions (E) using an external context vector.
    - `set_context(context)` / `clear_context()` to attach/detach a context vector.
    """

    def __init__(self, n_states: int, max_duration: int, n_features: int, alpha: float = 1.0, seed: Optional[int] = None, encoder: Optional[nn.Module] = None, emission_type: str = "gaussian", min_covar: float = 1e-3, device: Optional[torch.device] = None, context_dim: Optional[int] = None, **kwargs):

        nn.Module.__init__(self)
        self._params = {'emission_type': emission_type.lower()}
        self.device = device or torch.device("cpu")
        self.n_features = n_features
        self.min_covar = min_covar
        self.encoder = encoder

        # Context handling
        self.context_dim = context_dim
        self._context: Optional[torch.Tensor] = None  # stored context embedding (optional)

        # context adapters (learned shifts). Created only if context_dim is provided.
        self.ctx_A: Optional[nn.Linear] = None
        self.ctx_D: Optional[nn.Linear] = None
        self.ctx_E: Optional[nn.Linear] = None

        # Optional embedding for any categorical context (flexible)
        self.context_embedding: Optional[nn.Embedding] = None
        if context_dim is not None and kwargs.get('n_context_states', None) is not None:
            n_ctx = kwargs['n_context_states']
            self.context_embedding = nn.Embedding(n_ctx, context_dim)
            nn.init.normal_(self.context_embedding.weight, mean=0.0, std=1e-3)

        if context_dim is not None:
            # create linear layers that map context -> required sizes.
            # We initialize them so they are near-zero (neutral) and cast to DTYPE.
            self.ctx_A = nn.Linear(context_dim, n_states * n_states, bias=True)
            self.ctx_D = nn.Linear(context_dim, n_states * max_duration, bias=True)
            # emission projection maps to per-state per-feature shifts (K * F)
            self.ctx_E = nn.Linear(context_dim, n_states * n_features, bias=True)

            # initialize small weights and biases centered near zero
            nn.init.normal_(self.ctx_A.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.ctx_A.bias, mean=0.0, std=1e-3)
            nn.init.normal_(self.ctx_D.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.ctx_D.bias, mean=0.0, std=1e-3)
            nn.init.normal_(self.ctx_E.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.ctx_E.bias, mean=0.0, std=1e-3)

            # cast params to DTYPE & device (keep module registered)
            self.ctx_A.to(dtype=DTYPE, device=self.device)
            self.ctx_D.to(dtype=DTYPE, device=self.device)
            self.ctx_E.to(dtype=DTYPE, device=self.device)

        # Initialize BaseHSMM (required for EM, Viterbi, scoring)
        super().__init__(n_states=n_states, max_duration=max_duration, alpha=alpha, seed=seed)
        self._params['emission_pdf'] = self.sample_emission_pdf()

        # Default pluggable modules
        self.emission_module = DefaultEmission(n_states, n_features, min_covar, context_dim=context_dim)
        self.duration_module = DefaultDuration(n_states, max_duration, context_dim=context_dim)
        self.transition_module = DefaultTransition(n_states, context_dim=context_dim)

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
        """
        Total degrees of freedom (number of learnable parameters)
        including initial, transition, duration, and emission parameters.
        Assumes diagonal covariance for Gaussian emissions.
        """
        nS, nD, nF = self.n_states, self.max_duration, self.n_features

        # Core HSMM parameters
        dof = (nS - 1)              # initial state probabilities (pi)
        dof += nS * (nS - 1)        # transition matrix (A)
        dof += nS * (nD - 1)        # duration distributions (D)

        # Emission parameters
        pdf = self.pdf
        if pdf is not None:
            if isinstance(pdf, Categorical):
                dof += nS * (pdf.logits.shape[1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                # diagonal Gaussian: mean + variance per feature
                dof += nS * (2 * nF)

        return dof

    # ----------------------
    # Context management
    # ----------------------
    def set_context(self, context: Optional[torch.Tensor]):
        """
        Attach an external context tensor that will be used by contextual hooks.

        `context` expected shape: (context_dim,) or (batch, context_dim).
        If context_dim was provided at construction, the linear adapters are used.
        Otherwise a direct additive/truncate/pad approach is used.
        """
        if context is None:
            self._context = None
            return

        ctx = context.detach().to(dtype=DTYPE, device=self.device)
        # if vector, keep as (1, D) for convenience
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def clear_context(self):
        """Clear any stored context to avoid leakage across calls."""
        self._context = None

    # ----------------------
    # Emission PDFs
    # ----------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        """
        Initialize emission distribution for each hidden state.
        Supports 'categorical' and 'gaussian' types.

        If X is provided, can optionally be used to initialize parameters
        from empirical data (currently unused).
        """
        nS, nF, dev, dt = self.n_states, self.n_features, self.device, DTYPE

        if self.emission_type == "categorical":
            # Uniform categorical logits per state
            logits = torch.full((nS, nF), fill_value=1.0 / nF, dtype=dt, device=dev).log()
            return Categorical(logits=logits)

        elif self.emission_type == "gaussian":
            # Mean = 0, diagonal covariance = min_covar * I
            mean = torch.zeros(nS, nF, dtype=dt, device=dev)
            var = torch.full((nS, nF), self.min_covar, dtype=dt, device=dev)
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> Distribution:
        """
        Estimate emission parameters (M-step) from data and posterior responsibilities.

        Parameters
        ----------
        X : (T, F) tensor
            Observations.
        posterior : (T, K) tensor
            State posterior probabilities.
        theta : Optional contextual tensor (unused for now)
        """
        K, F, dev, dt = self.n_states, self.n_features, self.device, DTYPE

        if self.emission_type == "categorical":
            # Assume X is one-hot encoded or counts
            if X.ndim != 2 or X.shape[1] != F:
                raise ValueError("For categorical emissions, X must be one-hot encoded with shape (T, n_features)")
            probs = posterior.T @ X
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
            return Categorical(probs=probs.clamp_min(1e-8))

        elif self.emission_type == "gaussian":
            # Normalized weights per state
            Nk = posterior.sum(dim=0, keepdim=True) + 1e-12
            weights = posterior / Nk
            mean = weights.T @ X  # (K, F)

            # Center and compute covariances
            diff = X.unsqueeze(1) - mean.unsqueeze(0)  # (T, K, F)
            w = posterior.unsqueeze(-1)
            cov = torch.einsum("tkf,tkh->kfh", w * diff, diff) / Nk.squeeze(0).unsqueeze(-1).unsqueeze(-1)

            # Regularize covariance for stability
            reg = self.min_covar * torch.eye(F, dtype=dt, device=dev).unsqueeze(0)
            cov = cov + reg

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    # ----------------------
    # Contextual hooks
    # ----------------------
    def _contextual_emission_pdf(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor],
        scale: float = 0.1
    ) -> Distribution:
        """Modulate emission distribution using contextual features."""
        pdf = self.pdf
        if pdf is None:
            return pdf

        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return pdf

        if isinstance(pdf, Categorical):
            n_classes = pdf.logits.shape[-1]
            if self.ctx_E is not None:
                ctx_out = self.ctx_E(theta_combined.unsqueeze(0)).squeeze(0)
                delta = ctx_out.view(self.n_states, self.n_features).sum(dim=1)[:n_classes]
            else:
                flat = theta_combined
                if flat.numel() < n_classes:
                    flat = torch.cat([flat, torch.zeros(n_classes - flat.numel(), dtype=DTYPE, device=self.device)])
                delta = flat[:n_classes]
            delta = scale * torch.tanh(delta.to(dtype=pdf.logits.dtype, device=pdf.logits.device))
            return Categorical(logits=pdf.logits + delta.unsqueeze(0).expand_as(pdf.logits))

        elif isinstance(pdf, MultivariateNormal):
            K, F = pdf.mean.shape
            total_dim = K * F
            if self.ctx_E is not None:
                ctx_out = self.ctx_E(theta_combined.unsqueeze(0)).squeeze(0)
                mean_shift = ctx_out[:total_dim]
            else:
                flat = theta_combined
                if flat.numel() < total_dim:
                    flat = torch.cat([flat, torch.zeros(total_dim - flat.numel(), dtype=DTYPE, device=self.device)])
                mean_shift = flat[:total_dim]
            mean_shift = scale * mean_shift.to(dtype=pdf.mean.dtype, device=pdf.mean.device).reshape(K, F)
            return MultivariateNormal(loc=pdf.mean + mean_shift, covariance_matrix=pdf.covariance_matrix)

        else:
            raise ValueError(f"Unsupported pdf type {type(pdf)}")

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """Contextually modulate transition probabilities."""
        base_logits = self.transition_module.logits.to(dtype=DTYPE, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(base_logits, dim=-1)

        n = self.n_states
        if self.ctx_A is not None:
            proj = self.ctx_A(theta_combined.unsqueeze(0)).squeeze(0)
            delta = 0.1 * torch.tanh(proj).reshape(n, n)
        else:
            flat = theta_combined
            if flat.numel() < n * n:
                flat = torch.cat([flat, torch.zeros(n * n - flat.numel(), dtype=DTYPE, device=self.device)])
            delta = 0.1 * torch.tanh(flat[: n * n]).reshape(n, n)
        return F.softmax(base_logits + delta.to(dtype=base_logits.dtype, device=base_logits.device), dim=-1)

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """Contextually modulate per-state duration distributions."""
        base_logits = self.duration_module.logits.to(dtype=DTYPE, device=self.device)
        theta_combined = self._combine_context(theta)
        if theta_combined is None:
            return F.softmax(base_logits, dim=-1)

        n_states, n_durations = base_logits.shape
        if self.ctx_D is not None:
            proj = self.ctx_D(theta_combined.unsqueeze(0)).squeeze(0)
            delta = 0.1 * torch.tanh(proj).reshape(n_states, n_durations)
        else:
            flat = theta_combined
            if flat.numel() < n_states * n_durations:
                flat = torch.cat([flat, torch.zeros(n_states * n_durations - flat.numel(), dtype=DTYPE, device=self.device)])
            delta = 0.1 * torch.tanh(flat[: n_states * n_durations]).reshape(n_states, n_durations)
        return F.softmax(base_logits + delta.to(dtype=base_logits.dtype, device=base_logits.device), dim=-1)

    # ----------------------
    # Encoder forward
    # ----------------------
    def encode_observations(self, X: torch.Tensor, detach: bool = True) -> Optional[torch.Tensor]:
        if self.encoder is None:
            return None

        inp = X if X.ndim == 3 else X.unsqueeze(0)
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]

        if detach:
            out = out.detach()

        out = out.to(dtype=DTYPE, device=self.device)

        if out.ndim == 3:  # (batch, seq_len, features)
            vec = out.mean(dim=1)
        elif out.ndim == 2:  # (batch, features)
            vec = out
        elif out.ndim == 1:  # (features,)
            vec = out.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape} (ndim={out.ndim})")

        return vec

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(
        self,
        X: torch.Tensor,
        return_pdf: bool = False,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional context:
        - context: continuous context vector
        - context_ids: discrete IDs to embed via self.context_embedding
        """
        prev_ctx = self._context

        # Process discrete context IDs
        ctx_emb = None
        if context_ids is not None and hasattr(self, "context_embedding") and self.context_embedding is not None:
            ctx_emb = self.context_embedding(context_ids)
            if ctx_emb.ndim == 3:  # (batch, seq_len, embed_dim)
                ctx_emb = ctx_emb.mean(dim=1)

        # Merge continuous and embedded context
        combined_ctx = None
        if context is not None and ctx_emb is not None:
            combined_ctx = torch.cat([context, ctx_emb], dim=-1)
        elif context is not None:
            combined_ctx = context
        elif ctx_emb is not None:
            combined_ctx = ctx_emb

        if combined_ctx is not None:
            self.set_context(combined_ctx)

        # Encode observations
        theta = self.encode_observations(X)

        if return_pdf:
            pdf = self._contextual_emission_pdf(X, theta)
            self._context = prev_ctx  # restore previous context
            return pdf

        self._context = prev_ctx  # restore previous context
        return theta

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().predict(X, *args, **kwargs)
