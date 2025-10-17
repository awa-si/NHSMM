# nhsmm/models/neural.py
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
        device = device or torch.device("cpu")
        dtype = dtype or DTYPE
        self.device = device
        self.dtype = dtype
        self.context_mode = context_mode.lower()
        if self.context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

        emission_type = emission_type.lower()
        if emission_type == "gaussian":
            mu, cov = params
            n_states, n_features = mu.shape
        else:
            logits = params
            n_states, n_features = logits.shape

        super().__init__(n_states=n_states, n_features=n_features, emission_type=emission_type)
        self.encoder = encoder

        if emission_type == "gaussian":
            self.mu = nn.Parameter(mu.to(device, dtype))
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            self.cov = nn.Parameter(cov.to(device, dtype))
        else:
            self.logits = nn.Parameter(params.to(device, dtype))

    # ---------------- Contextual Params ---------------- #
    def contextual_params(self, theta: Optional[torch.Tensor] = None):
        """Return context-modulated parameters."""
        if self.encoder is None or theta is None:
            return (self.mu, self.cov) if self.emission_type == "gaussian" else self.logits

        out = self.encoder(theta)
        if self.emission_type == "gaussian":
            mu, cov = out
            if cov.ndim == 2:
                cov = torch.diag_embed(cov)
            return mu.to(self.device, self.dtype), cov.to(self.device, self.dtype)
        else:
            return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    # ---------------- Forward / Log-Prob ---------------- #
    def forward(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None, log: bool = False):
        X = X.to(self.device, self.dtype)
        batch_mode = X.ndim == 3
        B, T, F = (X.shape if batch_mode else (1, *X.shape))
        K = self.n_states

        if self.emission_type == "gaussian":
            mu, cov = self.contextual_params(theta)
            dist = MultivariateNormal(mu, covariance_matrix=cov)
            logp = dist.log_prob(X.unsqueeze(-2))  # broadcast over states
            return logp if log else logp.exp()
        else:
            logits = self.contextual_params(theta)
            probs = F.softmax(logits, dim=-1)
            if batch_mode:
                X_idx = X.long().view(B * T)
                logp = torch.log(probs[:, X_idx % self.n_features] + EPS).T.view(B, T, K)
            else:
                X_idx = X.long()
                logp = torch.log(probs[:, X_idx] + EPS).T
            return logp if log else logp.exp()

    def log_prob(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None):
        return self.forward(X, theta=theta, log=True)

    # ---------------- EM Update ---------------- #
    def update(self, posterior: torch.Tensor, X: torch.Tensor, inplace: bool = True):
        posterior = posterior.to(self.device, self.dtype)
        X = X.to(self.device, self.dtype)

        X_flat = X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X
        posterior_flat = posterior.reshape(-1, posterior.shape[-1]) if posterior.ndim > 2 else posterior

        if self.emission_type == "gaussian":
            counts = posterior_flat.sum(dim=0).clamp_min(EPS)[:, None]
            mu_new = (posterior_flat.T @ X_flat) / counts
            cov_new = torch.stack([
                ((X_flat - mu_new[s]).T * posterior_flat[:, s]) @ (X_flat - mu_new[s]) / counts[s, 0]
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
            X_onehot = F.one_hot(X_flat.long(), num_classes=self.n_features).float()
            weighted_counts = posterior_flat.T @ X_onehot
            logits_new = (weighted_counts / weighted_counts.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_min(EPS).log()
            if inplace:
                self.logits.data.copy_(logits_new)
                return self
            return NeuralEmission(self.emission_type, logits_new, encoder=self.encoder,
                                  context_mode=self.context_mode, device=self.device, dtype=self.dtype)

    # ---------------- Initialization ---------------- #
    @classmethod
    def initialize(cls, emission_type: str, n_states: int, n_features: int = None,
                   n_categories: int = None, alpha: float = 1.0,
                   encoder: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        device = device or torch.device("cpu")
        if emission_type.lower() == "gaussian":
            mu = torch.randn(n_states, n_features, device=device, dtype=DTYPE) * 0.1
            cov = torch.stack([torch.ones(n_features, device=device, dtype=DTYPE) for _ in range(n_states)])
            return cls("gaussian", (mu, cov), encoder=encoder, device=device)
        else:
            logits = torch.distributions.Dirichlet(
                torch.ones(n_categories, device=device, dtype=DTYPE) * alpha
            ).sample([n_states]).clamp_min(EPS).log()
            return cls(emission_type.lower(), logits, encoder=encoder, device=device)

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
    ):
        super().__init__(n_states=n_states, max_duration=max_duration, device=device, dtype=DTYPE)
        self.device = device or torch.device("cpu")
        self.mode = mode.lower()
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1

        if self.mode == "poisson":
            self.rate = nn.Parameter(rate if rate is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
        else:
            self.mean = nn.Parameter(mean if mean is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))
            self.std = nn.Parameter(std if std is not None else torch.ones(n_states, device=self.device, dtype=DTYPE))

    # ---------------- Contextual Parameters ---------------- #
    def _contextual_params(self, context: Optional[torch.Tensor] = None):
        X = getattr(context, "X", context)
        encoded = self.encoder(X) if self.encoder and X is not None else None

        if self.mode == "poisson":
            rate = F.softplus(encoded).squeeze(-1) if encoded is not None else self.rate
            return rate.clamp_min(EPS)
        else:
            if encoded is not None:
                mean, log_std = torch.chunk(encoded, 2, dim=-1)
            else:
                mean, log_std = self.mean, self.std.log()
            return mean.squeeze(-1), log_std.exp().clamp_min(EPS).squeeze(-1)

    # ---------------- Forward / Probabilities ---------------- #
    def forward(self, context: Optional[torch.Tensor] = None, log: bool = False, return_dist: bool = False):
        durations = torch.arange(1, self.max_duration + 1, device=self.device, dtype=DTYPE)

        if self.mode == "poisson":
            rate = self._contextual_params(context)
            logits = durations.unsqueeze(0) * torch.log(rate.unsqueeze(-1)) \
                     - rate.unsqueeze(-1) - torch.lgamma(durations.unsqueeze(0) + 1)
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

    # ---------------- Sampling ---------------- #
    def sample(self, n_samples: int = 1, context: Optional[torch.Tensor] = None, state_indices: Optional[torch.Tensor] = None):
        dists = self.forward(context=context, return_dist=True)
        samples = torch.stack([dist.sample((n_samples,)) for dist in dists], dim=0)
        return samples if state_indices is None else samples[state_indices]

    # ---------------- Device Management ---------------- #
    def to(self, device: torch.device):
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
    ):
        super().__init__(n_states=n_states, device=device, dtype=DTYPE)
        self.device = device or torch.device("cpu")
        self.encoder = encoder
        self.context_mode = context_mode.lower()
        self.scale = 0.1

        if logits is not None:
            self.logits = nn.Parameter(logits.to(self.device, DTYPE))
        else:
            self.logits = nn.Parameter(torch.zeros(n_states, n_states, device=self.device, dtype=DTYPE))

    # ---------------- Contextual Logits ---------------- #
    def contextual_logits(self, theta: Optional[torch.Tensor] = None):
        if self.encoder is None or theta is None:
            return self.logits

        X = getattr(theta, "X", theta)
        out = self.encoder(X)
        if isinstance(out, tuple):
            out = out[0]  # handle encoder returning (output, hidden)

        if out.ndim > 2:
            out = out.view(-1, self.n_states, self.n_states)

        return torch.log_softmax(out, dim=-1).clamp_min(EPS)

    # ---------------- EM Update ---------------- #
    def update(self, posterior: torch.Tensor, inplace: bool = True):
        logits = torch.log((posterior + EPS).to(self.device, DTYPE))
        if inplace:
            self.logits.data.copy_(logits)
            return self
        return NeuralTransition(
            n_states=self.n_states,
            logits=logits,
            encoder=self.encoder,
            context_mode=self.context_mode,
            device=self.device,
        )

    # ---------------- Initialization ---------------- #
    @classmethod
    def initialize(
        cls,
        n_states: int,
        alpha: float = 1.0,
        batch: int = 1,
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        probs = torch.distributions.Dirichlet(
            torch.ones(n_states, device=device, dtype=DTYPE) * alpha
        ).sample([batch])

        if batch == 1:
            probs = probs.squeeze(0)

        return cls(
            n_states=n_states,
            logits=torch.log(probs.clamp_min(EPS)),
            encoder=encoder,
            device=device,
        )

    # ---------------- Device Management ---------------- #
    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        if hasattr(self, "logits"):
            self.logits = nn.Parameter(self.logits.to(device))
        if self.encoder:
            self.encoder.to(device)
        return self


class NeuralHSMM(HSMM, nn.Module):
    """Trainable NeuralHSMM with EM, Viterbi, context, and gradient support.

    This class preserves the same public methods you already used:
    - encode_observations, forward, predict, decode
    - context setters, contextual PDF factories
    - sample_emission_pdf, _estimate_emission_pdf, initialize_emissions
    """

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
        self.min_covar = float(min_covar)
        self.encoder = encoder
        self._params: Dict[str, Any] = {"emission_type": emission_type.lower()}

        # Neural modules (preserve your previous initializers)
        # Assumes NeuralTransition.initialize exists as in your other code.
        self.transition_module = Transition(n_states=n_states, context_dim=context_dim)  # base Transition
        self.duration_module = Duration(n_states=n_states, max_duration=max_duration, context_dim=context_dim)
        self.emission_module = Emission(n_states=n_states, n_features=n_features, context_dim=context_dim, emission_type=emission_type.lower())

        # optional encoder-aware wrappers (if you provided Neural* classes, replace above)
        # keep the encoder ref for higher-level encoding use
        if encoder is not None:
            self.encoder = encoder.to(device=self.device, dtype=self.dt)

        # Context components (small dense layers that modulate base params)
        self.context_dim = context_dim
        self._context: Optional[torch.Tensor] = None
        self.context_embedding: Optional[nn.Embedding] = None
        self.ctx_transition: Optional[nn.Linear] = None
        self.ctx_duration: Optional[nn.Linear] = None
        self.ctx_emission: Optional[nn.Linear] = None

        if context_dim is not None:
            # allow optional context embedding if number of discrete context states given
            n_ctx = kwargs.get("n_context_states", None)
            if n_ctx is not None:
                self.context_embedding = nn.Embedding(n_ctx, context_dim)
                nn.init.normal_(self.context_embedding.weight, mean=0.0, std=1e-3)

            # small linear modulators (kept lightweight)
            self.ctx_transition = nn.Linear(context_dim, n_states * n_states, device=self.device, dtype=self.dt)
            self.ctx_duration = nn.Linear(context_dim, n_states * max_duration, device=self.device, dtype=self.dt)
            self.ctx_emission = nn.Linear(context_dim, n_states * n_features, device=self.device, dtype=self.dt)
            for m in (self.ctx_transition, self.ctx_duration, self.ctx_emission):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)

        # Base HSMM initialization (calls parent HSMM constructor)
        super().__init__(n_states=n_states, n_features=n_features, max_duration=max_duration, alpha=alpha, seed=seed)

        # initialize emission PDF (a sensible default)
        self._params["emission_pdf"] = self.sample_emission_pdf()

    # -----------------------------
    # Basic properties
    # -----------------------------
    @property
    def emission_type(self) -> str:
        return self._params.get("emission_type", "gaussian")

    @property
    def pdf(self):
        return self._params.get("emission_pdf", None)

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

    # -----------------------------
    # Sampling / initialization helpers
    # -----------------------------
    def sample_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: float = 0.1,
        temperature: float = 1.0
    ):
        """Create an emission Distribution conditioned on optional X and theta.

        Returns Categorical or MultivariateNormal (batched over states).
        """
        nS, nF = self.n_states, self.n_features
        dt, dev = self.dt, self.device

        X_flat = None
        if X is not None:
            X_flat = X.reshape(-1, nF).to(dtype=dt, device=dev)

        delta_mean = None
        if theta is not None and self.ctx_emission is not None:
            t = theta.to(dtype=dt, device=dev)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            delta = self.ctx_emission(t)
            # ensure shape [B, nS*nF] -> average over batch
            if delta.ndim > 2:
                delta = delta.view(delta.shape[0], -1)
            delta_mean = delta.mean(dim=0).view(nS, nF)

        if self.emission_type == "categorical":
            if X_flat is not None:
                # build per-state initial probs from data (replicate)
                probs = X_flat.mean(dim=0, keepdim=True).repeat(nS, 1)
            else:
                probs = torch.full((nS, nF), 1.0 / nF, dtype=dt, device=dev)

            if delta_mean is not None:
                probs = probs + scale * torch.tanh(delta_mean)

            logits = (probs / temperature)
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            return Categorical(logits=logits.to(device=dev, dtype=dt))

        elif self.emission_type == "gaussian":
            if X_flat is not None:
                mean = X_flat.mean(dim=0, keepdim=True).repeat(nS, 1)
                var = X_flat.var(dim=0, unbiased=False, keepdim=True).repeat(nS, 1).clamp_min(self.min_covar)
            else:
                mean = torch.zeros(nS, nF, dtype=dt, device=dev)
                var = torch.full((nS, nF), self.min_covar, dtype=dt, device=dev)

            if delta_mean is not None:
                mean = mean + scale * torch.tanh(delta_mean)

            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean.to(dev, dt), covariance_matrix=cov.to(dev, dt))

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        scale: float = 0.1
    ):
        """Estimate an emission distribution from weighted data (posterior responsibilities)."""
        K, F = self.n_states, self.n_features
        dt, dev = self.dt, self.device

        X = X.to(dtype=dt, device=dev)
        posterior = posterior.to(dtype=dt, device=dev)

        # compute context delta if present
        delta = torch.zeros((K, F), dtype=dt, device=dev)
        if theta is not None:
            t = theta.to(dtype=dt, device=dev)
            if self.ctx_emission is not None:
                d = self.ctx_emission(t)
                if d.ndim > 2:
                    d = d.mean(dim=0)
                d = d.view(-1, K, F) if d.ndim == 2 and d.shape[1] == K * F else d.view(K, F)
                delta = scale * torch.tanh(d.mean(dim=0) if d.ndim == 3 else d)
            else:
                # fallback slice
                if t.ndim == 2:
                    delta = t[:, : K * F].mean(dim=0).view(K, F)
                else:
                    delta = t[: K * F].view(K, F)
            delta = delta.to(dtype=dt, device=dev)

        if self.emission_type == "categorical":
            probs = posterior.T @ X  # (K,F)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
            probs = (probs + delta).clamp_min(EPS)
            probs = probs / probs.sum(dim=1, keepdim=True)
            return Categorical(probs=probs.to(device=dev, dtype=dt))

        elif self.emission_type == "gaussian":
            Nk = posterior.sum(dim=0).clamp_min(EPS)  # (K,)
            Nk_mat = Nk.unsqueeze(1)  # (K,1)
            mean = (posterior.T @ X) / Nk_mat  # (K, F)

            diff = X.unsqueeze(1) - mean.unsqueeze(0)  # (T, K, F)
            weighted = diff * posterior.unsqueeze(-1)  # (T, K, F)
            cov = torch.einsum("tkf,tkh->kfh", weighted, diff) / Nk_mat.unsqueeze(-1)  # (K, F, F)
            # regularize diagonal
            cov = cov + torch.eye(F, device=dev, dtype=dt).unsqueeze(0) * self.min_covar

            mean = mean + delta
            return MultivariateNormal(loc=mean.to(dev, dt), covariance_matrix=cov.to(dev, dt))

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def initialize_emissions(self, X, method: str = "moment"):
        """Initialize emission parameters from raw data X (keeps original behavior)."""
        K = self.n_states
        dt, dev = self.dt, self.device

        if not torch.is_tensor(X): X = torch.tensor(X, dtype=dt, device=dev)
        X = X.to(dtype=dt, device=dev)

        if self.emission_type == "gaussian":
            if X.ndim != 2:
                raise ValueError(f"Gaussian emissions require X shape (T,F), got {X.shape}")
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

            var_init = var_init.clamp_min(self.min_covar)
            cov = torch.diag_embed(var_init)

            # set module & pdf
            if hasattr(self.emission_module, "mu"):
                self.emission_module.mu.data.copy_(mu_init)
            else:
                self.emission_module.mu = nn.Parameter(mu_init)

            if hasattr(self.emission_module, "cov"):
                self.emission_module.cov.data.copy_(cov)
            else:
                self.emission_module.cov = nn.Parameter(cov)

            self._params["emission_pdf"] = MultivariateNormal(loc=mu_init.to(dev, dt), covariance_matrix=cov.to(dev, dt))

        elif self.emission_type in ("categorical", "bernoulli"):
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                labels = X.squeeze(-1).long()
                counts = torch.bincount(labels, minlength=self.n_features).to(dtype=dt, device=dev)
            else:
                counts = X.sum(dim=0).to(dtype=dt, device=dev)

            probs = counts / counts.sum()
            logits = (probs.clamp_min(EPS).log()).unsqueeze(0).repeat(K, 1)

            # keep placeholders consistent
            self.emission_module.mu = nn.Parameter(torch.zeros_like(logits))
            self.emission_module.cov = nn.Parameter(torch.zeros_like(logits))
            self._params["emission_pdf"] = Categorical(logits=logits.to(dev, dt))

        else:
            raise RuntimeError(f"Unsupported emission_type '{self.emission_type}'")

    # -----------------------------
    # Context handling
    # -----------------------------
    def set_context(self, context: Optional[torch.Tensor]):
        """Store a (B,H) or (H,) context tensor on model device/dtype."""
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(device=self.device, dtype=self.dt)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def clear_context(self):
        self._context = None

    def _combine_context(self, theta: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Combine encoder-produced theta with externally set context self._context safely."""
        if self._context is None:
            if theta is None:
                return None
            return theta.to(device=self.device, dtype=self.dt)
        if theta is None:
            return self._context.to(device=self.device, dtype=self.dt)

        theta = theta.to(device=self.device, dtype=self.dt)
        ctx = self._context.to(device=self.device, dtype=self.dt)

        # align batch dims
        if theta.shape[0] != ctx.shape[0]:
            if theta.shape[0] == 1:
                theta = theta.expand(ctx.shape[0], -1)
            elif ctx.shape[0] == 1:
                ctx = ctx.expand(theta.shape[0], -1)
            else:
                raise ValueError(f"Batch mismatch: theta {theta.shape} vs stored context {ctx.shape}")

        return torch.cat([theta, ctx], dim=-1)

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor], scale: float = 0.1):
        """Return a contextually-shifted emission Distribution (same API as sample_emission_pdf)."""
        pdf = self.pdf
        if pdf is None:
            return None

        combined = self._combine_context(theta)
        if combined is None:
            return pdf

        dev, dt = self.device, self.dt

        if isinstance(pdf, Categorical):
            # compute additive logits shift and average over batch
            if self.ctx_emission is not None:
                delta = self.ctx_emission(combined)
            else:
                delta = combined[:, : self.n_states * self.n_features]
            if delta.ndim == 2 and delta.shape[1] == self.n_states * self.n_features:
                delta = delta.view(delta.shape[0], self.n_states, self.n_features)
                delta_mean = delta.mean(dim=0)
            else:
                delta_mean = delta.view(self.n_states, self.n_features) if delta.ndim == 1 else delta.mean(dim=0)
            logits = getattr(pdf, "logits", None)
            if logits is None:
                raise RuntimeError("Categorical pdf missing logits.")
            logits = (logits + (scale * torch.tanh(delta_mean)).to(device=dev, dtype=dt)).to(device=dev, dtype=dt)
            return Categorical(logits=logits)

        elif isinstance(pdf, MultivariateNormal):
            mean = pdf.mean.to(device=dev, dtype=dt)
            cov = pdf.covariance_matrix.to(device=dev, dtype=dt)
            # compute mean shift
            if self.ctx_emission is not None:
                delta = self.ctx_emission(combined)
            else:
                delta = combined[:, : (self.n_states * self.n_features)]
            if delta.ndim == 2:
                delta = delta.view(delta.shape[0], self.n_states, self.n_features)
                delta_mean = delta.mean(dim=0)
            else:
                delta_mean = delta.view(self.n_states, self.n_features)
            new_mean = (mean + (scale * torch.tanh(delta_mean)).to(device=dev, dtype=dt)).to(device=dev, dtype=dt)
            return MultivariateNormal(loc=new_mean, covariance_matrix=cov)

        else:
            raise TypeError(f"Unsupported PDF type: {type(pdf)}")

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor], scale: float = 0.1, temperature: float = 1.0):
        """Return (n_states, max_duration) duration probs, optionally modulated by context."""
        logits = getattr(self.duration_module, "logits", None)
        if logits is None:
            # fallback: ask duration_module for probs
            out = self.duration_module.forward(context=None, log=False)
            return out

        combined = self._combine_context(theta)
        if combined is None or self.ctx_duration is None:
            return F.softmax(logits.to(dtype=self.dt, device=self.device) / temperature, dim=-1)

        delta = self.ctx_duration(combined)
        if delta.ndim == 2 and delta.shape[1] == self.n_states * self.max_duration:
            delta = delta.view(-1, self.n_states, self.max_duration).mean(dim=0)
        else:
            delta = delta.view(self.n_states, self.max_duration)
        return F.softmax((logits.to(dtype=self.dt, device=self.device) + scale * torch.tanh(delta)) / temperature, dim=-1)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor], scale: float = 0.1, temperature: float = 1.0):
        """Return (n_states, n_states) transition matrix modulated by context."""
        logits = getattr(self.transition_module, "logits", None)
        if logits is None:
            out = self.transition_module.forward(context=None, log=False)
            return out

        combined = self._combine_context(theta)
        if combined is None or self.ctx_transition is None:
            return F.softmax(logits.to(dtype=self.dt, device=self.device) / temperature, dim=-1)

        delta = self.ctx_transition(combined)
        if delta.ndim == 2 and delta.shape[1] == self.n_states * self.n_states:
            delta = delta.view(-1, self.n_states, self.n_states).mean(dim=0)
        else:
            delta = delta.view(self.n_states, self.n_states)
        return F.softmax((logits.to(dtype=self.dt, device=self.device) + scale * torch.tanh(delta)) / temperature, dim=-1)

    # -----------------------------
    # Encoder / forward helpers
    # -----------------------------
    def encode_observations(self, X: torch.Tensor, detach: bool = True) -> Optional[torch.Tensor]:
        """Encode observations with attached encoder. Returns (B, H) or None."""
        if self.encoder is None:
            return None

        inp = X if X.ndim == 3 else X.unsqueeze(0)
        inp = inp.to(dtype=self.dt, device=self.device)

        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]
        if detach:
            out = out.detach()
        out = out.to(dtype=self.dt, device=self.device)

        if out.ndim == 3:
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
        """Top-level forward: set/merge context, encode, optionally return contextual emission PDF."""
        prev_ctx = self._context

        ctx_emb = None
        if context_ids is not None and self.context_embedding is not None:
            ctx_emb = self.context_embedding(context_ids.to(self.device))
            if ctx_emb.ndim == 3:
                ctx_emb = ctx_emb.mean(dim=1)

        if context is not None:
            ctx = context if ctx_emb is None else torch.cat([context.to(device=self.device, dtype=self.dt), ctx_emb.to(device=self.device, dtype=self.dt)], dim=-1)
        else:
            ctx = ctx_emb

        if ctx is not None:
            self.set_context(ctx)

        theta = self.encode_observations(X)

        if return_pdf:
            pdf = self._contextual_emission_pdf(X, theta)
            self._context = prev_ctx
            return pdf

        self._context = prev_ctx
        return theta

    # -----------------------------
    # Prediction / decoding
    # -----------------------------
    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> List[torch.Tensor]:
        """Compute context-aware emission/transition/duration then call HSMM.predict (keeps interface)."""
        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)
        pdf = self._contextual_emission_pdf(X, theta)
        transition = self._contextual_transition_matrix(theta)
        duration = self._contextual_duration_pdf(theta)

        # Temporarily set HSMM internals for prediction
        prev_pdf = self._params.get("emission_pdf", None)
        prev_transition_logits = getattr(self, "_transition_logits", None)
        prev_duration_logits = getattr(self, "_duration_logits", None)

        self._params["emission_pdf"] = pdf
        if transition is not None:
            # copy into internal logits if expected shape matches
            if isinstance(transition, torch.Tensor) and transition.shape == (self.n_states, self.n_states):
                self._transition_logits = transition
        if duration is not None:
            if isinstance(duration, torch.Tensor) and duration.shape == (self.n_states, self.max_duration):
                self._duration_logits = duration

        preds = super().predict(X, lengths=lengths, algorithm=algorithm, batch_size=batch_size)

        # restore
        self._params["emission_pdf"] = prev_pdf
        if prev_transition_logits is not None:
            self._transition_logits = prev_transition_logits
        if prev_duration_logits is not None:
            self._duration_logits = prev_duration_logits

        return preds

    def decode(
        self,
        X: torch.Tensor,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        duration_weight: float = 0.0,
        context: Optional[torch.Tensor] = None,
        context_ids: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Convenience wrapper that encodes context, updates contextual PDFs and returns numpy predictions."""
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=self.dt, device=self.device)
        else:
            X = X.to(dtype=self.dt, device=self.device)

        theta = self.forward(X, return_pdf=False, context=context, context_ids=context_ids)

        # compute contextually-modulated params (side-effects kept minimal)
        _ = self._contextual_transition_matrix(theta, scale=duration_weight)
        _ = self._contextual_duration_pdf(theta, scale=duration_weight)
        _ = self._contextual_emission_pdf(X, theta)

        preds = self.predict(X, algorithm=algorithm, context=theta, context_ids=context_ids)
        # unify to numpy
        if isinstance(preds, list):
            if len(preds) == 1:
                out = preds[0]
            else:
                out = torch.stack([p.to(self.device) if torch.is_tensor(p) else torch.tensor(p, device=self.device) for p in preds])
        elif torch.is_tensor(preds):
            out = preds
        else:
            out = torch.as_tensor(preds, dtype=torch.long, device=self.device)

        return out.detach().cpu().numpy()
